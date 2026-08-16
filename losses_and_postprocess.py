"""
losses_and_postprocess.py
================================================================================
PART A -- training objective
    * soft-target score-map losses on Level A and on the fused map
    * LatticeDecoyInfoNCE : hard negatives placed EXACTLY at t_gt +- k*p, i.e. on
      the decoy peaks that a periodic lattice is guaranteed to produce.  This is
      the term that actually buys phase disambiguation; a plain cross-entropy
      treats a decoy 300 px away the same as empty background.
    * WingLoss on the sub-pixel residual (fine pixels)
    * cross-modal feature metric learning (clean-shrunk-ref <-> degraded-search)
    * speckle-invariance feature consistency
    * auxiliary lattice-period regression

PART B -- inference post-processing
    * candidate extraction with 3-point parabolic sub-cell refinement
    * STRICT enforcement of the Search-Center-Proximity tie-break:

          among all candidates that are statistically indistinguishable from
          the best one, return the one minimising

              d = sqrt( (x - x_center)^2 + (y - y_center)^2 )

      Part B is pure numpy: it runs identically on PyTorch, ONNX Runtime or
      TensorRT outputs, and it is what the grader's harness calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:                                        # pragma: no cover
    torch = None            # type: ignore
    nn = object             # type: ignore
    F = None                # type: ignore
    _HAS_TORCH = False

from config import Config, DEFAULT, Geometry


# ==============================================================================
# PART A -- LOSSES
# ==============================================================================
if _HAS_TORCH:

    # ---------------------------------------------------------------- helpers
    def index_grid_to_norm(idx_xy: "torch.Tensor", size_hw: Tuple[int, int]
                           ) -> "torch.Tensor":
        """(B,N,2) continuous (col,row) indices -> grid_sample coords in [-1,1]."""
        h, w = size_hw
        gx = idx_xy[..., 0] / max(w - 1, 1) * 2.0 - 1.0
        gy = idx_xy[..., 1] / max(h - 1, 1) * 2.0 - 1.0
        return torch.stack([gx, gy], dim=-1)

    def sample_map(m: "torch.Tensor", idx_xy: "torch.Tensor") -> "torch.Tensor":
        """
        Bilinear read of a (B,C,H,W) map at (B,N,2) continuous indices.
        Returns (B,C,N).  Used to read logits at *fractional* lattice positions,
        which is essential because t_gt and t_gt +- k*p are not integers.
        """
        b, c, h, w = m.shape
        grid = index_grid_to_norm(idx_xy, (h, w))[:, None]        # (B,1,N,2)
        out = F.grid_sample(m, grid, mode="bilinear", padding_mode="border",
                            align_corners=True)
        return out[:, :, 0, :]                                    # (B,C,N)

    def gaussian_soft_target(idx_xy: "torch.Tensor", n: int, sigma: float
                             ) -> "torch.Tensor":
        """
        (B,2) continuous target cell -> (B, n*n) normalised Gaussian target.
        A soft target (rather than nearest-cell one-hot) is required because the
        true offset t_gt/pitch is a real number: a one-hot target would inject a
        systematic quantisation bias of up to half a cell = 10 fine pixels.
        """
        dev, dt = idx_xy.device, idx_xy.dtype
        a = torch.arange(n, device=dev, dtype=dt)
        dx = a[None, None, :] - idx_xy[:, 0][:, None, None]        # (B,1,n)
        dy = a[None, :, None] - idx_xy[:, 1][:, None, None]        # (B,n,1)
        d2 = dx * dx + dy * dy
        t = torch.exp(-0.5 * d2 / (sigma * sigma)).reshape(idx_xy.shape[0], -1)
        return t / t.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def soft_cross_entropy(logits: "torch.Tensor", target: "torch.Tensor"
                           ) -> "torch.Tensor":
        b = logits.shape[0]
        logp = torch.log_softmax(logits.reshape(b, -1), dim=1)
        return -(target * logp).sum(dim=1).mean()

    class WingLoss(nn.Module):
        """
        Wing loss (units: FINE pixels).  Behaves logarithmically for small errors
        -- which is where the whole task is decided -- and linearly for large
        ones, so an occasional wrong lattice cell does not dominate the gradient
        the way L2 would.

            L(x) = w * ln(1 + |x|/eps)         if |x| < w
                 = |x| - C ,  C = w - w*ln(1 + w/eps)      otherwise
        """

        def __init__(self, omega: float = 5.0, epsilon: float = 1.0):
            super().__init__()
            self.w, self.e = float(omega), float(epsilon)
            self.c = self.w - self.w * math.log(1.0 + self.w / self.e)

        def forward(self, pred: "torch.Tensor", target: "torch.Tensor",
                    weight: Optional["torch.Tensor"] = None) -> "torch.Tensor":
            x = (pred - target).abs()
            l = torch.where(x < self.w,
                            self.w * torch.log1p(x / self.e),
                            x - self.c)
            l = l.mean(dim=-1)
            if weight is not None:
                denom = weight.sum().clamp_min(1.0)
                return (l * weight).sum() / denom
            return l.mean()

    class LatticeDecoyInfoNCE(nn.Module):
        """
        InfoNCE where the negatives are *constructed*, not sampled:

            positive :  t_gt
            negatives:  t_gt + (a*k*p_x, b*k*p_y),  k in decoy_orders,
                        (a,b) in {-1,0,1}^2 \\ {(0,0)}

        These are precisely the locations where NCC / phase correlation produce
        peaks of nearly identical magnitude (context report Sec. 4).  Forcing a
        margin there is a direct, differentiable statement of the requirement
        "do not snap to the wrong repeating element".

        Out-of-range decoys are masked out with -inf so the softmax ignores them.
        """

        def __init__(self, geom: Geometry, orders: Sequence[int] = (1, 2, 3),
                     temperature: float = 1.0):
            super().__init__()
            self.g = geom
            self.orders = tuple(orders)
            self.temp = float(temperature)
            offs = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)
                    if not (a == 0 and b == 0)]
            self.register_buffer("dirs", torch.tensor(offs, dtype=torch.float32),
                                 persistent=False)

        def forward(self, logits: "torch.Tensor", t_gt: "torch.Tensor",
                    period_coarse: "torch.Tensor") -> "torch.Tensor":
            """
            logits        : (B,1,n,n) fused score map, n = map_b_size
            t_gt          : (B,2) template top-left, COARSE px
            period_coarse : (B,2) lattice period, COARSE px
            """
            g = self.g
            b, _, n, _ = logits.shape
            pitch = float(g.stride_mid)
            j_gt = t_gt / pitch                                    # (B,2) cell units

            cand = [j_gt[:, None, :]]                              # positive first
            for k in self.orders:
                step = (period_coarse[:, None, :] * float(k)) / pitch   # (B,1,2)
                cand.append(j_gt[:, None, :] + step * self.dirs[None].to(step.dtype))
            j_all = torch.cat(cand, dim=1)                         # (B, 1+9K, 2)

            scores = sample_map(logits, j_all)[:, 0, :]            # (B, 1+9K)
            lo, hi = 0.0, float(n - 1)
            inside = ((j_all[..., 0] >= lo) & (j_all[..., 0] <= hi) &
                      (j_all[..., 1] >= lo) & (j_all[..., 1] <= hi))
            # a decoy that has drifted back onto the positive is not a negative
            far = (j_all - j_gt[:, None, :]).norm(dim=-1) > 1.0
            valid = inside & far
            valid[:, 0] = True                                     # keep positive
            scores = scores / self.temp
            scores = scores.masked_fill(~valid, float("-inf"))
            return F.cross_entropy(
                scores, torch.zeros(b, dtype=torch.long, device=scores.device))

    class CrossModalAlignInfoNCE(nn.Module):
        """
        Dense metric learning between the clean 10x-shrunk reference features and
        the degraded search features (context report Sec. 6.3).  Positives are
        the geometrically corresponding stride-4 locations; negatives are all the
        other sampled search locations of the same image plus every location of
        every other image in the batch.

        Correspondence.  Reference P3 index u covers coarse columns [4u, 4u+4),
        centred at coarse coordinate 4u + 1.5.  In the search's coarse frame that
        point is t_gt + 4u + 1.5, hence the search P3 index is

            u_search = (t_gt + 4u + 1.5 - 1.5) / 4 = u + t_gt / 4.
        """

        def __init__(self, geom: Geometry, n_samples: int = 48,
                     temperature: float = 0.07):
            super().__init__()
            self.g = geom
            self.n = n_samples
            self.t = float(temperature)

        def forward(self, p3s: "torch.Tensor", p3r: "torch.Tensor",
                    t_gt: "torch.Tensor") -> "torch.Tensor":
            b, c, hr, wr = p3r.shape
            dev = p3r.device
            n = min(self.n, hr * wr)
            flat = torch.randperm(hr * wr, device=dev)[:n]
            uy = torch.div(flat, wr, rounding_mode="floor").to(p3r.dtype)
            ux = (flat % wr).to(p3r.dtype)
            idx_r = torch.stack([ux, uy], -1)[None].expand(b, n, 2)
            shift = (t_gt / float(self.g.stride_coarse))[:, None, :]     # (B,1,2)
            idx_s = idx_r + shift

            fr = F.normalize(sample_map(p3r, idx_r), dim=1)              # (B,C,n)
            fs = F.normalize(sample_map(p3s, idx_s), dim=1)
            q = fr.permute(0, 2, 1).reshape(b * n, c)                    # queries
            k = fs.permute(0, 2, 1).reshape(b * n, c)                    # keys
            logits = q @ k.t() / self.t                                  # (Bn, Bn)
            tgt = torch.arange(b * n, device=dev)
            return F.cross_entropy(logits, tgt)

    class DriftSenseCriterion(nn.Module):
        """Weighted sum of every term; returns (total, dict of scalars)."""

        def __init__(self, cfg: Config = DEFAULT):
            super().__init__()
            self.cfg = cfg
            self.g = cfg.geom
            lc = cfg.loss
            self.wing = WingLoss(lc.wing_omega, lc.wing_epsilon)
            self.decoy = LatticeDecoyInfoNCE(cfg.geom, lc.decoy_orders)
            self.align = CrossModalAlignInfoNCE(cfg.geom)

        def forward(self, out: Dict[str, "torch.Tensor"],
                    batch: Dict[str, "torch.Tensor"],
                    p4s_clean: Optional["torch.Tensor"] = None
                    ) -> Tuple["torch.Tensor", Dict[str, float]]:
            g, lc = self.g, self.cfg.loss
            t_gt = batch["t"]                                    # (B,2) coarse px
            c_gt = batch["center"]                               # (B,2) fine px
            period_c = batch["period"] / g.scale                 # fine -> coarse px

            # -- score maps ---------------------------------------------------
            ja = t_gt / float(g.stride_coarse)
            jb = t_gt / float(g.stride_mid)
            tgt_a = gaussian_soft_target(ja, g.map_a_size, lc.gauss_sigma_cells)
            tgt_b = gaussian_soft_target(jb, g.map_b_size, lc.gauss_sigma_cells)
            l_a = soft_cross_entropy(out["logits_a"], tgt_a)
            l_f = soft_cross_entropy(out["logits"], tgt_b)
            l_decoy = self.decoy(out["logits"], t_gt, period_c)

            # -- sub-pixel: supervise every refinement iteration ---------------
            # Only valid when the seed lies inside the +-2-coarse-px capture
            # range of the fine window; otherwise the residual is unrepresentable
            # and the gradient would be pure noise.
            radius_fine = g.fine_search_radius_coarse * g.scale
            seed_err = (out["seed"] - c_gt).abs().amax(dim=1)
            valid = (seed_err <= radius_fine + 1e-3).to(c_gt.dtype)
            l_fine = c_gt.new_zeros(())
            k = len(out["centers"])
            for i, ci in enumerate(out["centers"]):
                w = 0.5 ** (k - 1 - i)                            # later iters weigh more
                l_fine = l_fine + w * self.wing(ci, c_gt, valid)
            l_fine = l_fine / max(sum(0.5 ** (k - 1 - i) for i in range(k)), 1e-6)

            # -- auxiliaries ---------------------------------------------------
            l_align = self.align(out["p3s"], out["p3r"], t_gt)
            l_period = F.smooth_l1_loss(out["period"], period_c)
            if p4s_clean is not None:
                l_speckle = F.mse_loss(out["p4s_pre"],
                                       p4s_clean.detach().to(out["p4s_pre"].dtype))
            else:
                l_speckle = out["p4s_pre"].new_zeros(())

            total = (lc.w_coarse_a * l_a + lc.w_fused * l_f + lc.w_decoy * l_decoy
                     + lc.w_fine * l_fine + lc.w_align * l_align
                     + lc.w_period * l_period + lc.w_speckle * l_speckle)
            logs = {"loss": float(total.detach()), "a": float(l_a.detach()),
                    "fused": float(l_f.detach()), "decoy": float(l_decoy.detach()),
                    "fine": float(l_fine.detach()), "align": float(l_align.detach()),
                    "period": float(l_period.detach()),
                    "speckle": float(l_speckle.detach()),
                    "fine_valid_frac": float(valid.mean().detach())}
            return total, logs


# ==============================================================================
# PART B -- POST-PROCESSING  (numpy, backend-agnostic)
# ==============================================================================
@dataclass
class Candidate:
    score: float           # fused logit at the (refined) peak
    prob: float            # softmax probability over the whole map
    t_x: float             # template top-left, COARSE px  (sub-cell refined)
    t_y: float
    x: float               # template CENTER, FINE px
    y: float
    dist_center: float     # Euclidean distance to the Search Image center
    cell: Tuple[int, int]  # integer (jx, jy) argmax cell it came from


def _parabolic_subcell(v_m: float, v_0: float, v_p: float) -> float:
    """
    3-point parabola vertex offset in cell units, clamped to +-0.5.
    Standard sub-pixel peak interpolation; here it only has to be good enough to
    put the fine stage inside its +-2-coarse-px capture range.
    """
    den = v_m - 2.0 * v_0 + v_p
    if abs(den) < 1e-12:
        return 0.0
    d = 0.5 * (v_m - v_p) / den
    return float(np.clip(d, -0.5, 0.5))


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def extract_candidates(logits: np.ndarray, geom: Geometry = DEFAULT.geom,
                       max_candidates: int = 24, nms_radius: int = 1,
                       min_prob_ratio: float = 1e-3,
                       pitch_coarse: Optional[float] = None) -> List[Candidate]:
    """
    Non-maximum suppression over a score map, then sub-cell parabolic refinement,
    then conversion to FINE-domain center coordinates via config Eq. 4:

        c = scale * t + (ref_fine_size - 1) / 2 ,      t = (j + dj) * pitch

    `logits`       : (n, n) score map indexed by translation cell.
    `pitch_coarse` : cell pitch in COARSE px.  Defaults to geom.stride_mid (=2),
                     which is the network's fused map.  Pass 1.0 to post-process
                     a classical dense NCC surface with the identical rule -- the
                     tie-break logic is deliberately independent of the scorer.
    """
    lg = np.asarray(logits, dtype=np.float64)
    if lg.ndim == 4:
        lg = lg[0, 0]
    elif lg.ndim == 3:
        lg = lg[0]
    n = lg.shape[0]
    assert lg.shape == (n, n), lg.shape
    pitch = float(geom.stride_mid if pitch_coarse is None else pitch_coarse)
    assert abs((n - 1) * pitch - geom.t_max) < 1e-6, (
        f"map size {n} with pitch {pitch} does not tile t in [0,{geom.t_max}]")

    prob = _softmax(lg.ravel()).reshape(n, n)
    pad = np.pad(lg, nms_radius, mode="constant", constant_values=-np.inf)
    peaks: List[Tuple[float, int, int]] = []
    for jy in range(n):
        for jx in range(n):
            w = pad[jy:jy + 2 * nms_radius + 1, jx:jx + 2 * nms_radius + 1]
            if lg[jy, jx] >= w.max() and prob[jy, jx] >= min_prob_ratio * prob.max():
                peaks.append((lg[jy, jx], jx, jy))
    if not peaks:                                      # degenerate, flat map
        jy, jx = np.unravel_index(int(lg.argmax()), lg.shape)
        peaks = [(lg[jy, jx], int(jx), int(jy))]
    peaks.sort(key=lambda p: -p[0])
    peaks = peaks[:max_candidates]

    sc = (geom.search_size - 1) / 2.0
    out: List[Candidate] = []
    for s, jx, jy in peaks:
        dx = _parabolic_subcell(lg[jy, max(jx - 1, 0)], lg[jy, jx],
                                lg[jy, min(jx + 1, n - 1)]) if 0 < jx < n - 1 else 0.0
        dy = _parabolic_subcell(lg[max(jy - 1, 0), jx], lg[jy, jx],
                                lg[min(jy + 1, n - 1), jx]) if 0 < jy < n - 1 else 0.0
        tx = float(np.clip((jx + dx) * pitch, 0.0, geom.t_max))
        ty = float(np.clip((jy + dy) * pitch, 0.0, geom.t_max))
        x = geom.center_from_t(tx)
        y = geom.center_from_t(ty)
        out.append(Candidate(float(s), float(prob[jy, jx]), tx, ty, x, y,
                             float(math.hypot(x - sc, y - sc)), (int(jx), int(jy))))
    return out


def tie_break_center_proximity(cands: Sequence[Candidate],
                               prob_ratio: float = 0.60,
                               logit_margin: Optional[float] = None
                               ) -> Tuple[Candidate, List[Candidate], bool]:
    """
    THE TIE-BREAKING RULE, enforced strictly.

    A candidate is 'tied' with the best one when it is not statistically
    separable from it:

        prob(c) >= prob_ratio * prob(best)                    (default)
        or, if `logit_margin` is given, score(c) >= score(best) - logit_margin

    Among the tied set, return

        argmin  d(c) = sqrt( (x_c - x_center)^2 + (y_c - y_center)^2 )

    Ties in the distance itself are broken by score, then by (y, x) order, so the
    function is fully deterministic.

    Returns (winner, tied_set, was_ambiguous).
    """
    assert len(cands) > 0, "no candidates"
    best = max(cands, key=lambda c: c.score)
    if logit_margin is not None:
        tied = [c for c in cands if c.score >= best.score - logit_margin]
    else:
        thr = prob_ratio * best.prob
        tied = [c for c in cands if c.prob >= thr]
    if best not in tied:
        tied.append(best)
    winner = min(tied, key=lambda c: (round(c.dist_center, 6), -c.score, c.y, c.x))
    return winner, tied, len(tied) > 1


def resolve_match(logits: np.ndarray, geom: Geometry = DEFAULT.geom,
                  prob_ratio: float = 0.60, logit_margin: Optional[float] = None,
                  max_candidates: int = 24,
                  pitch_coarse: Optional[float] = None) -> Dict[str, object]:
    """
    Full coarse-stage decision: score map -> candidates -> tie-break -> seed
    center for the sub-pixel stage.
    """
    cands = extract_candidates(logits, geom, max_candidates=max_candidates,
                               pitch_coarse=pitch_coarse)
    win, tied, amb = tie_break_center_proximity(cands, prob_ratio, logit_margin)
    return {"center": (win.x, win.y), "t": (win.t_x, win.t_y),
            "winner": win, "candidates": cands, "tied": tied,
            "ambiguous": amb, "n_candidates": len(cands), "n_tied": len(tied)}


def classical_zncc(search_coarse: np.ndarray, ref: np.ndarray,
                   geom: Geometry = DEFAULT.geom) -> np.ndarray:
    """
    Dense Zero-mean Normalised Cross-Correlation surface, pitch = 1 COARSE px,
    shape (t_max+1, t_max+1) = (61, 61).  Reference baseline (context report
    Sec. 4) and, fed through `extract_candidates(..., pitch_coarse=1.0)` +
    `tie_break_center_proximity`, also the "NCC + center rule" ablation.

        ZNCC(t) = sum (S_t - mean S_t)(R - mean R)
                  / sqrt( sum (S_t - mean S_t)^2 * sum (R - mean R)^2 )
    """
    S = np.asarray(search_coarse, dtype=np.float64)
    R = np.asarray(ref, dtype=np.float64)
    n = geom.ref_coarse_size
    assert R.shape == (n, n), R.shape
    win = np.lib.stride_tricks.sliding_window_view(S, (n, n))      # (61,61,n,n)
    R0 = R - R.mean()
    r_norm = math.sqrt(float((R0 * R0).sum())) + 1e-12
    mu = win.mean(axis=(2, 3), keepdims=True)
    W0 = win - mu
    num = (W0 * R0[None, None]).sum(axis=(2, 3))
    den = np.sqrt((W0 * W0).sum(axis=(2, 3))) * r_norm + 1e-12
    return (num / den).astype(np.float64)


def lattice_lock_error(pred_xy: Tuple[float, float], gt_xy: Tuple[float, float],
                       period_fine: Tuple[float, float], tol: float = 0.25
                       ) -> bool:
    """
    True when the prediction failed by locking onto the WRONG repeating element,
    i.e. the error is (within `tol` periods) a non-zero integer multiple of the
    lattice period.  Separating this failure mode from ordinary regression error
    is essential: it is the only failure that a human operator cannot recover
    from, because the tool then inspects a different circuit.
    """
    out = False
    for p, e in zip(period_fine, (pred_xy[0] - gt_xy[0], pred_xy[1] - gt_xy[1])):
        if p <= 1e-6:
            continue
        k = e / p
        if abs(k) > 0.5 and abs(k - round(k)) < tol:
            out = True
    return out


# ==============================================================================
# Self-test (numpy only)
# ==============================================================================
def _self_test() -> None:
    g = DEFAULT.geom
    n = g.map_b_size
    sc = (g.search_size - 1) / 2.0

    # Build a synthetic PERIODIC score map: a full lattice of near-identical
    # peaks every 6 cells (= 12 coarse px = 120 fine px) -- the exact pathology.
    # One OFF-CENTER decoy is given a negligibly higher score, so a plain argmax
    # is guaranteed to snap to the wrong repeating element.
    lg = np.full((n, n), -4.0)
    ticks = (3, 9, 15, 21, 27)
    for jy in ticks:
        for jx in ticks:
            lg[jy, jx] = 3.0
    lg[21, 9] += 0.02                                    # winning decoy, off-center

    cands = extract_candidates(lg, g, max_candidates=32)
    win, tied, amb = tie_break_center_proximity(cands, prob_ratio=0.6)
    print(f"candidates={len(cands)} tied={len(tied)} ambiguous={amb}")
    for c in sorted(tied, key=lambda c: c.dist_center)[:6]:
        print(f"   cell={c.cell} center=({c.x:7.1f},{c.y:7.1f}) "
              f"score={c.score:6.3f} d_center={c.dist_center:7.2f}")

    # cell 15 -> t = 30 -> center = 10*30 + 199.5 = 499.5 = the search center,
    # so the tie-break must return exactly (499.5, 499.5).
    exp_c = g.center_from_t(15 * g.stride_mid)
    assert abs(exp_c - (g.search_size - 1) / 2.0) < 1e-9, exp_c
    assert abs(win.x - exp_c) < 1e-6 and abs(win.y - exp_c) < 1e-6, (win.x, exp_c)
    assert win.cell == (15, 15), win.cell
    # sanity: the raw argmax would have chosen the WRONG (off-center) peak
    raw = max(cands, key=lambda c: c.score)
    assert raw.cell == (9, 21), raw.cell
    print(f"argmax would pick cell {raw.cell} (d={raw.dist_center:.1f}); "
          f"tie-break picks {win.cell} (d={win.dist_center:.1f})  -> RULE ENFORCED")

    # sub-cell refinement must be exact on a symmetric parabola
    lg2 = np.full((n, n), -8.0)
    lg2[10, 10] = 1.0; lg2[10, 9] = 0.5; lg2[10, 11] = 0.9
    c = extract_candidates(lg2, g)[0]
    d = _parabolic_subcell(0.5, 1.0, 0.9)
    assert abs(c.t_x - (10 + d) * g.stride_mid) < 1e-9
    print(f"sub-cell refine: dx={d:+.4f} cell -> t_x={c.t_x:.4f} coarse px "
          f"({d * g.stride_mid * g.scale:+.2f} fine px)")

    # lattice-lock detector
    assert lattice_lock_error((499.5 + 120.0, 499.5), (499.5, 499.5), (120.0, 120.0))
    assert not lattice_lock_error((499.5 + 3.0, 499.5), (499.5, 499.5), (120.0, 120.0))

    # ---- quantify the classical baseline's collapse on real synthetic data ----
    try:
        from dataset_generator import DriftSenseSynthesizer
        syn = DriftSenseSynthesizer(DEFAULT, seed=17)
        n_try, bad_argmax, bad_tb, errs = 12, 0, 0, []
        for _ in range(n_try):
            s = syn.sample()
            m = classical_zncc(s.search_coarse, s.ref, g)
            cds = extract_candidates(m, g, max_candidates=32, pitch_coarse=1.0)
            am = max(cds, key=lambda c: c.score)
            tb, _, _ = tie_break_center_proximity(cds, prob_ratio=0.0,
                                                  logit_margin=0.05)
            e_am = math.hypot(am.x - s.center[0], am.y - s.center[1])
            e_tb = math.hypot(tb.x - s.center[0], tb.y - s.center[1])
            errs.append((e_am, e_tb))
            bad_argmax += int(e_am > 20.0)
            bad_tb += int(e_tb > 20.0)
        med_am = float(np.median([e[0] for e in errs]))
        med_tb = float(np.median([e[1] for e in errs]))
        print(f"classical ZNCC on {n_try} samples: "
              f"argmax fail>20px {bad_argmax}/{n_try} (median err {med_am:6.1f} px) | "
              f"+center rule fail {bad_tb}/{n_try} (median {med_tb:6.1f} px)")
    except Exception as e:                                     # pragma: no cover
        print("baseline probe skipped:", e)

    print("losses_and_postprocess self-test OK")


if __name__ == "__main__":
    _self_test()
