"""
drift_sense_model.py
================================================================================
Coarse-to-Fine Navigation Recovery Network (CFNRN) for DRIFT-SENSE.

                      Reference R (40x40, already 10x-shrunk)
                                    |
   Search S (1000x1000) --avg_pool(10)--> S_c (100x100)
                                    |
              +---------------------+---------------------+
              |        SHARED-WEIGHT BAFPN ENCODER        |   (same scale domain,
              |   C2/s2  C3/s4  C4/s8  C5/s16 + GALM +    |    so weight sharing
              |   SPAM-aligned top-down fusion            |    is *correct*)
              +---------------------+---------------------+
                                    |
        LoFTR-lite interleaved self/cross attention on the s8 level
        (absolute 2-D positional encoding = the only thing that can
         break the translational symmetry of a periodic lattice)
                                    |
        +---------------------------+---------------------------+
        |  Level A: correlation @ s4  -> 16x16 map, pitch 40 fine px  (global)
        |  Level B: correlation @ s2  -> 31x31 map, pitch 20 fine px  (sharp)
        |  Fusion : A regridded onto B (exact, align_corners) + center prior
        +---------------------------+---------------------------+
                                    |
                     candidate extraction + CENTER TIE-BREAK
                          (losses_and_postprocess.py)
                                    |
        Level C: phase-aware resampling of the FULL-RES search at the
                 candidate center -> 44x44 coarse patch -> stride-1
                 correlation tensor (5x5) -> soft-argmax + residual MLP,
                 iterated 2x (Lucas-Kanade style)  -> sub-pixel (x, y)

All index<->coordinate conversions follow config.py Eq. 1-4 exactly.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, DEFAULT, Geometry


# ==============================================================================
# 0.  Input conditioning
# ==============================================================================
class InputNorm(nn.Module):
    """
    log1p + per-image standardisation.

    Rationale.  Speckle is multiplicative:  I' = I * n.  Hence
        log(I') = log(I) + log(n),
    i.e. the log transform converts an intensity-dependent, unbounded,
    heavy-tailed corruption into a bounded *additive* one, which every
    convolution / attention layer downstream can average away.  Per-image
    standardisation then removes the residual detector gain, DC bias and the
    low-order charging gradient.  Nothing is clipped, so the "intensities beyond
    the ground-truth range" requirement is preserved end to end.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.log1p(x.clamp_min(0.0))
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        s = x.std(dim=(1, 2, 3), keepdim=True)
        return (x - m) / (s + self.eps)


class DespeckleStem(nn.Module):
    """
    Tiny residual denoiser applied in the normalised log domain.  Trained
    *implicitly* by the speckle-consistency loss (features of the noisy view are
    pulled towards features of the clean view), so it never needs paired
    clean/noisy supervision at inference.
    """

    def __init__(self, ch: int = 16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, 3, padding=1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


# ==============================================================================
# 1.  Backbone + BAFPN (GALM lateral, SPAM alignment)
# ==============================================================================
def _conv_bn(cin: int, cout: int, k: int = 3, s: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride=s, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.c1 = _conv_bn(cin, cout, 3, stride)
        self.c2 = nn.Sequential(nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                                nn.BatchNorm2d(cout))
        self.short = (nn.Identity() if (cin == cout and stride == 1) else
                      nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                                    nn.BatchNorm2d(cout)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.c2(self.c1(x)) + self.short(x))


class GroupAggregationLateral(nn.Module):
    """
    GALM (context report Sec. 6.1).  A plain 1x1 channel squeeze in an FPN
    lateral destroys texture detail that nanometre metrology needs.  GALM keeps
    semantically-similar channels together (grouped 1x1), re-weights each group
    with a squeeze-excitation gate, and only then mixes groups.
    """

    def __init__(self, cin: int, cout: int, groups: int = 4):
        super().__init__()
        assert cin % groups == 0 and cout % groups == 0
        self.groups = groups
        self.grouped = nn.Conv2d(cin, cout, 1, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(cout)
        self.gate = nn.Sequential(nn.Linear(cout, max(8, cout // 8)),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(max(8, cout // 8), groups),
                                  nn.Sigmoid())
        self.mix = nn.Conv2d(cout, cout, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(self.grouped(x))
        b, c, h, w = y.shape
        g = self.gate(y.mean(dim=(2, 3)))                      # (B, groups)
        y = y.view(b, self.groups, c // self.groups, h, w) * g[:, :, None, None, None]
        return self.mix(y.view(b, c, h, w))


class SpatialFeatureAlignment(nn.Module):
    """
    SPAM (context report Sec. 6.1).  Naive top-down addition fuses a deep,
    spatially-smeared feature onto a shallow, spatially-crisp one; any residual
    misregistration shows up as *aliasing*, which is precisely the failure mode
    we are fighting.  SPAM predicts a dense 2-D flow from the concatenation and
    warps the coarse map onto the fine map before addition.
    """

    def __init__(self, dim: int, max_offset: float = 2.0):
        super().__init__()
        self.max_offset = max_offset
        self.pred = nn.Sequential(_conv_bn(2 * dim, dim // 2, 3),
                                  nn.Conv2d(dim // 2, 2, 3, padding=1))
        nn.init.zeros_(self.pred[-1].weight)
        nn.init.zeros_(self.pred[-1].bias)

    def forward(self, coarse_up: torch.Tensor, fine: torch.Tensor) -> torch.Tensor:
        b, _, h, w = fine.shape
        flow = torch.tanh(self.pred(torch.cat([coarse_up, fine], 1))) * self.max_offset
        yy, xx = torch.meshgrid(
            torch.arange(h, device=fine.device, dtype=fine.dtype),
            torch.arange(w, device=fine.device, dtype=fine.dtype), indexing="ij")
        gx = (xx[None] + flow[:, 0]) / max(w - 1, 1) * 2 - 1
        gy = (yy[None] + flow[:, 1]) / max(h - 1, 1) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1)                    # (B,H,W,2)
        return F.grid_sample(coarse_up, grid, mode="bilinear",
                             padding_mode="border", align_corners=True)


class BAFPNEncoder(nn.Module):
    """
    Shared-weight encoder producing P2 (stride 2), P3 (stride 4) and P4 (stride 8)
    in the COARSE domain.  Reference and coarse-search are processed by the same
    weights because, after the 10x re-gridding, they live in the same scale space.
    """

    def __init__(self, mc, dim: Optional[int] = None):
        super().__init__()
        w = mc.width
        dim = dim or mc.fpn_dim
        self.norm = InputNorm()
        self.despeckle = DespeckleStem()
        self.stem = _conv_bn(1, w, 3, 1)
        self.c2 = ResBlock(w, w, stride=2)            # s2
        self.c3 = ResBlock(w, 2 * w, stride=2)        # s4
        self.c4 = ResBlock(2 * w, 4 * w, stride=2)    # s8
        self.c5 = ResBlock(4 * w, 4 * w, stride=2)    # s16

        Lat = GroupAggregationLateral if mc.use_galm else \
            (lambda ci, co: nn.Conv2d(ci, co, 1, bias=False))
        self.l2, self.l3 = Lat(w, dim), Lat(2 * w, dim)
        self.l4, self.l5 = Lat(4 * w, dim), Lat(4 * w, dim)
        self.use_spam = mc.use_spam
        if self.use_spam:
            self.a4, self.a3, self.a2 = (SpatialFeatureAlignment(dim) for _ in range(3))
        self.s4, self.s3, self.s2 = (_conv_bn(dim, dim, 3) for _ in range(3))

    # -- split so the attention module can sit between P4 and P3 --------------
    def trunk(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.despeckle(self.norm(x))
        f = self.stem(x)
        c2 = self.c2(f); c3 = self.c3(c2); c4 = self.c4(c3); c5 = self.c5(c4)
        p5 = self.l5(c5)
        lat4 = self.l4(c4)
        up = F.interpolate(p5, size=lat4.shape[-2:], mode="bilinear", align_corners=False)
        if self.use_spam:
            up = self.a4(up, lat4)
        p4 = self.s4(lat4 + up)
        return {"c2": c2, "c3": c3, "p4": p4}

    def top_down(self, cache: Dict[str, torch.Tensor], p4: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        lat3 = self.l3(cache["c3"])
        up = F.interpolate(p4, size=lat3.shape[-2:], mode="bilinear", align_corners=False)
        if self.use_spam:
            up = self.a3(up, lat3)
        p3 = self.s3(lat3 + up)

        lat2 = self.l2(cache["c2"])
        up = F.interpolate(p3, size=lat2.shape[-2:], mode="bilinear", align_corners=False)
        if self.use_spam:
            up = self.a2(up, lat2)
        p2 = self.s2(lat2 + up)
        return p2, p3


# ==============================================================================
# 2.  LoFTR-lite: interleaved self / cross attention with absolute 2-D PE
# ==============================================================================
def sine_pe_2d(h: int, w: int, dim: int, device, dtype) -> torch.Tensor:
    """Absolute 2-D sinusoidal positional encoding, shape (1, dim, h, w)."""
    assert dim % 4 == 0, "attn_dim must be divisible by 4"
    d = dim // 4
    yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                            torch.arange(w, device=device, dtype=dtype), indexing="ij")
    freq = torch.exp(torch.arange(d, device=device, dtype=dtype)
                     * (-math.log(10000.0) / max(d - 1, 1)))
    ang_x = xx[..., None] * freq
    ang_y = yy[..., None] * freq
    pe = torch.cat([ang_x.sin(), ang_x.cos(), ang_y.sin(), ang_y.cos()], dim=-1)
    return pe.permute(2, 0, 1)[None]


class AttnBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim),
                                nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(self.n1(q), self.n2(kv), self.n2(kv), need_weights=False)
        q = q + a
        return q + self.ff(q)


class LoFTRLite(nn.Module):
    """
    Self-attention gives every ambiguous lattice cell a *global* receptive field,
    so it can read off its absolute phase from the array boundary / guard bands /
    unique markers.  Cross-attention then aligns the two modalities (clean
    shrunk reference vs. degraded search) before any correlation is computed.
    """

    def __init__(self, mc):
        super().__init__()
        d = mc.attn_dim
        self.proj_in = nn.Conv2d(mc.fpn_dim, d, 1)
        self.proj_out = nn.Conv2d(d, mc.fpn_dim, 1)
        self.blocks = nn.ModuleList([
            nn.ModuleList([AttnBlock(d, mc.attn_heads) for _ in range(4)])
            for _ in range(mc.attn_layers)])
        self.dim = d

    @staticmethod
    def _flat(x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2)                    # (B, HW, C)

    def forward(self, ps: torch.Tensor, pr: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        zs, zr = self.proj_in(ps), self.proj_in(pr)
        bs, _, hs, ws = zs.shape
        _, _, hr, wr = zr.shape
        zs = zs + sine_pe_2d(hs, ws, self.dim, zs.device, zs.dtype)
        zr = zr + sine_pe_2d(hr, wr, self.dim, zr.device, zr.dtype)
        ts, tr = self._flat(zs), self._flat(zr)
        for self_s, self_r, cross_s, cross_r in self.blocks:
            ts = self_s(ts, ts)          # global context inside the search image
            tr = self_r(tr, tr)          # global context inside the template
            ts_new = cross_s(ts, tr)     # search  <- template
            tr = cross_r(tr, ts)         # template <- search
            ts = ts_new
        zs = ts.transpose(1, 2).reshape(bs, self.dim, hs, ws)
        zr = tr.transpose(1, 2).reshape(bs, self.dim, hr, wr)
        return ps + self.proj_out(zs), pr + self.proj_out(zr)


# ==============================================================================
# 3.  Correlation primitives
# ==============================================================================
def correlate_valid(s: torch.Tensor, r: torch.Tensor, normalize: bool = True,
                    impl: str = "conv") -> torch.Tensor:
    """
    Batched 'valid' cross-correlation of a per-sample template against a
    per-sample search feature map.

        s : (B, C, H, W)      r : (B, C, h, w)   ->   (B, 1, H-h+1, W-w+1)

    With `normalize=True` every spatial feature vector is L2-normalised first,
    so the output is the *mean cosine similarity* over the template support --
    a learned, illumination-invariant analogue of ZNCC, bounded in [-1, 1].

    impl='conv'   : grouped conv2d (weight = activation).  Fast, low memory.
    impl='unfold' : im2col + bmm.  Use when an inference backend refuses a
                    Conv node whose W is not an initializer (some ORT builds).
    """
    b, c, h, w = s.shape
    kh, kw = r.shape[-2:]
    if normalize:
        s = F.normalize(s, dim=1, eps=1e-6)
        r = F.normalize(r, dim=1, eps=1e-6)
    denom = float(kh * kw)
    if impl == "conv":
        out = F.conv2d(s.reshape(1, b * c, h, w), r, groups=b)      # (1,B,Ho,Wo)
        return out.reshape(b, 1, out.shape[-2], out.shape[-1]) / denom
    cols = F.unfold(s, kernel_size=(kh, kw))                        # (B, C*kh*kw, L)
    ker = r.reshape(b, 1, c * kh * kw)
    out = torch.bmm(ker, cols)                                      # (B, 1, L)
    ho, wo = h - kh + 1, w - kw + 1
    return out.reshape(b, 1, ho, wo) / denom


# ==============================================================================
# 4.  Auxiliary periodicity head
# ==============================================================================
class PeriodicityHead(nn.Module):
    """
    Regresses the lattice basis periods (p_x, p_y) in COARSE pixels from the
    *self*-similarity map of the search features.  Two uses:
      * auxiliary supervision -> forces the trunk to explicitly encode the
        symmetry group instead of ignoring it;
      * at inference the predicted period tells the post-processor exactly where
        the decoy peaks must lie (t_gt +- k*p), which makes candidate
        enumeration and the tie-break provably complete.
    """

    def __init__(self, dim: int, ker: int, corr_dim: int = 32):
        super().__init__()
        self.ker = ker
        self.proj = nn.Conv2d(dim, corr_dim, 1, bias=False)
        self.head = nn.Sequential(
            _conv_bn(1, 16, 3, 2), _conv_bn(16, 32, 3, 2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 2))

    def forward(self, p3_search: torch.Tensor) -> torch.Tensor:
        f = self.proj(p3_search)
        _, _, h, w = f.shape
        k = min(self.ker, h, w)
        y0, x0 = (h - k) // 2, (w - k) // 2
        ker = f[:, :, y0:y0 + k, x0:x0 + k]
        ac = correlate_valid(f, ker)                     # self-similarity map
        return F.softplus(self.head(ac)) + 1e-3          # coarse px, strictly > 0


# ==============================================================================
# 5.  Score-map fusion (Level A + Level B + center prior)
# ==============================================================================
class FusionHead(nn.Module):
    def __init__(self, mc, geom: Geometry):
        super().__init__()
        self.geom = geom
        cin = 3 if mc.use_center_prior else 2
        self.use_center_prior = mc.use_center_prior
        self.body = nn.Sequential(_conv_bn(cin, 32, 3), _conv_bn(32, 32, 3),
                                  nn.Conv2d(32, 1, 1))
        self.wa = nn.Parameter(torch.tensor(1.0))
        self.wb = nn.Parameter(torch.tensor(1.0))
        self.temp = nn.Parameter(torch.tensor(float(mc.softmax_temp_init)))
        self.register_buffer("center_prior", self._make_center_prior(geom),
                             persistent=False)

    @staticmethod
    def _make_center_prior(g: Geometry) -> torch.Tensor:
        """Normalised Euclidean distance from each candidate center to the
        geometric center of the Search Image (the tie-break metric itself)."""
        n = g.map_b_size
        j = torch.arange(n, dtype=torch.float32)
        cx = g.scale * (g.stride_mid * j) + (g.ref_fine_size - 1) / 2.0
        sc = (g.search_size - 1) / 2.0
        dx = (cx - sc)[None, :].expand(n, n)
        dy = (cx - sc)[:, None].expand(n, n)
        d = torch.sqrt(dx * dx + dy * dy) / (g.search_size / 2.0)
        return d[None, None]

    def forward(self, logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
        # exact re-gridding of A (16, pitch 4) onto B (31, pitch 2): endpoints
        # coincide (t=0 and t=60), hence align_corners=True is mathematically right.
        a_up = F.interpolate(logits_a, size=logits_b.shape[-2:],
                             mode="bilinear", align_corners=True)
        feats = [a_up, logits_b]
        if self.use_center_prior:
            feats.append(self.center_prior.to(a_up.dtype).expand_as(a_up))
        res = self.body(torch.cat(feats, 1))
        return self.temp * (self.wa * a_up + self.wb * logits_b) + res


# ==============================================================================
# 6.  Level C: phase-aware resampling + sub-pixel head
# ==============================================================================
class PhaseAwareResampler(nn.Module):
    """
    The reference lost its high frequencies for good, so the sub-pixel signal
    cannot come from super-resolving it.  It comes from the *sampling phase*:
    the reference is one particular 10x area-average of the layout, and the
    full-resolution search image lets us synthesise the area-average at ANY
    continuous phase.  Matching the reference against the correctly-phased
    downsample is what yields sub-coarse-pixel accuracy.

    Implementation (exact, differentiable, ONNX-friendly):
      1. box[i, j] = mean(fine[i:i+scale, j:j+scale])   -- separable avg_pool,
         stride 1, so every integer *and* (via bilinear interpolation) every
         fractional phase is available.  box index i represents fine coordinate
         i + (scale-1)/2.
      2. sample box at fine coords  c + scale * (k - (n-1)/2)   (config Eq. 2),
         i.e. at box coords  c + scale*(k - (n-1)/2) - (scale-1)/2.

    Because the reference was built with the *same* area-mean convention, step 2
    reproduces the reference's own sampling operator exactly when c = c_gt.
    """

    def __init__(self, geom: Geometry):
        super().__init__()
        self.g = geom

    def box_filter(self, search_fine: torch.Tensor) -> torch.Tensor:
        s = self.g.scale
        x = F.avg_pool2d(search_fine, (s, 1), stride=1)
        return F.avg_pool2d(x, (1, s), stride=1)          # (B,1,H-s+1,W-s+1)

    def forward(self, box: torch.Tensor, center_xy: torch.Tensor, out_size: int
                ) -> torch.Tensor:
        """center_xy: (B,2) in FINE pixels -> (B,1,out_size,out_size) coarse patch."""
        s, dev, dt = self.g.scale, box.device, box.dtype
        b, _, hb, wb = box.shape
        k = torch.arange(out_size, device=dev, dtype=dt) - (out_size - 1) / 2.0
        off = s * k                                                # fine-domain offsets
        fx = center_xy[:, 0:1] + off[None, :]                      # (B, n)
        fy = center_xy[:, 1:2] + off[None, :]
        bx = fx - (s - 1) / 2.0
        by = fy - (s - 1) / 2.0
        gx = bx / (wb - 1) * 2 - 1
        gy = by / (hb - 1) * 2 - 1
        grid = torch.stack([gx[:, None, :].expand(b, out_size, out_size),
                            gy[:, :, None].expand(b, out_size, out_size)], dim=-1)
        return F.grid_sample(box, grid, mode="bilinear",
                             padding_mode="border", align_corners=True)


class SubPixelHead(nn.Module):
    """
    Stride-1 high-resolution correlation tensor + learnable neighbourhood
    consensus (GLU-Net / ECO-TR style), evaluated on the phase-resampled patch.

    Offset semantics.  With patch size n_p and template size n_r, the valid
    correlation index m in [0, n_p-n_r] aligns the template to patch[m:m+n_r];
    the template centre then sits at patch index m + (n_r-1)/2, i.e. at fine
    coordinate  c + scale * (m - (n_p-n_r)/2).  Hence

        delta_coarse(m) = m - (n_p - n_r) / 2      in COARSE pixels
        delta_fine      = scale * delta_coarse

    Sub-pixel comes from (i) a soft-argmax (expectation under a learned-
    temperature softmax) over that 5x5 tensor and (ii) an MLP residual on the
    flattened tensor + pooled descriptors, which corrects the well-known
    soft-argmax bias caused by asymmetric side-lobes -- exactly the asymmetry a
    periodic lattice produces.
    """

    def __init__(self, mc, geom: Geometry):
        super().__init__()
        self.g = geom
        d = mc.fine_dim
        self.norm = InputNorm()
        self.despeckle = DespeckleStem(ch=12)
        self.enc = nn.Sequential(_conv_bn(1, d, 3), _conv_bn(d, d, 3),
                                 nn.Conv2d(d, d, 3, padding=1))
        n = geom.fine_corr_size
        self.temp = nn.Parameter(torch.tensor(20.0))
        self.mlp = nn.Sequential(
            nn.Linear(n * n + 2 * d, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, 2))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("grid1d",
                             torch.arange(n, dtype=torch.float32)
                             - (geom.fine_patch_coarse - geom.ref_coarse_size) / 2.0,
                             persistent=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(self.despeckle(self.norm(x)))

    def forward(self, patch: torch.Tensor, ref_feat: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pf = self.encode(patch)
        corr = correlate_valid(pf, ref_feat)                       # (B,1,n,n)
        b, _, n, _ = corr.shape
        flat = corr.reshape(b, n * n)
        p = torch.softmax(flat * self.temp, dim=1).reshape(b, n, n)
        gx = self.grid1d.to(corr.dtype)
        dx = (p.sum(dim=1) * gx[None]).sum(1)                      # coarse px
        dy = (p.sum(dim=2) * gx[None]).sum(1)
        soft = torch.stack([dx, dy], 1)
        ctx = torch.cat([flat, pf.mean(dim=(2, 3)), ref_feat.mean(dim=(2, 3))], 1)
        delta = soft + self.mlp(ctx)                               # coarse px
        conf = p.reshape(b, n * n).max(dim=1).values
        return delta, corr, conf


# ==============================================================================
# 7.  Full network
# ==============================================================================
class DriftSenseNet(nn.Module):
    """
    Coarse-to-Fine Navigation Recovery Network.

    forward() returns everything the loss and the post-processor need:
        logits_a  (B,1,16,16)  Level-A global score map      (pitch 40 fine px)
        logits_b  (B,1,31,31)  Level-B sharp score map       (pitch 20 fine px)
        logits    (B,1,31,31)  fused score map -> candidates + tie-break
        period    (B,2)        lattice period, COARSE px
        center    (B,2)        final sub-pixel (x, y) in FINE px
        centers   list[(B,2)]  per-iteration centers (for deep supervision)
        seed      (B,2)        the candidate centre the fine stage started from
    """

    def __init__(self, cfg: Config = DEFAULT, corr_impl: str = "conv"):
        super().__init__()
        self.cfg, self.g, self.mc = cfg, cfg.geom, cfg.model
        self.corr_impl = corr_impl
        self.encoder = BAFPNEncoder(cfg.model)
        self.attn = LoFTRLite(cfg.model)
        cd = 32
        self.proj_a = nn.Conv2d(cfg.model.fpn_dim, cd, 1, bias=False)
        self.proj_b = nn.Conv2d(cfg.model.fpn_dim, cd, 1, bias=False)
        self.fusion = FusionHead(cfg.model, cfg.geom)
        self.period_head = PeriodicityHead(
            cfg.model.fpn_dim, ker=self.g.ref_coarse_size // self.g.stride_coarse)
        self.resampler = PhaseAwareResampler(cfg.geom)
        self.subpixel = SubPixelHead(cfg.model, cfg.geom)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def make_coarse(search_fine: torch.Tensor, scale: int) -> torch.Tensor:
        """Exact block-mean 10x downsample -- identical to the data generator."""
        return F.avg_pool2d(search_fine, scale, stride=scale)

    def _pyramid(self, search_coarse: torch.Tensor, ref: torch.Tensor
                 ) -> Dict[str, torch.Tensor]:
        cs = self.encoder.trunk(search_coarse)
        cr = self.encoder.trunk(ref)
        p4s, p4r = self.attn(cs["p4"], cr["p4"])
        p2s, p3s = self.encoder.top_down(cs, p4s)
        p2r, p3r = self.encoder.top_down(cr, p4r)
        return {"p2s": p2s, "p3s": p3s, "p2r": p2r, "p3r": p3r,
                "p4s": p4s, "p4r": p4r,
                # pre-attention trunk features: target/anchor of the
                # speckle-invariance consistency loss (cheap to recompute on a
                # clean view without running the attention stack twice)
                "p4s_pre": cs["p4"], "p4r_pre": cr["p4"]}

    def score_maps(self, search_coarse: torch.Tensor, ref: torch.Tensor
                   ) -> Dict[str, torch.Tensor]:
        f = self._pyramid(search_coarse, ref)
        la = correlate_valid(self.proj_a(f["p3s"]), self.proj_a(f["p3r"]),
                             impl=self.corr_impl)
        lb = correlate_valid(self.proj_b(f["p2s"]), self.proj_b(f["p2r"]),
                             impl=self.corr_impl)
        assert la.shape[-1] == self.g.map_a_size, (la.shape, self.g.map_a_size)
        assert lb.shape[-1] == self.g.map_b_size, (lb.shape, self.g.map_b_size)
        fused = self.fusion(la, lb)
        period = self.period_head(f["p3s"])
        return {"logits_a": la, "logits_b": lb, "logits": fused,
                "period": period, **f}

    def center_from_index(self, idx: torch.Tensor) -> torch.Tensor:
        """Flat index into the (31,31) fused map -> candidate center, FINE px."""
        n = self.g.map_b_size
        jy = torch.div(idx, n, rounding_mode="floor").to(torch.float32)
        jx = (idx % n).to(torch.float32)
        s, m = self.g.scale, self.g.stride_mid
        off = (self.g.ref_fine_size - 1) / 2.0
        return torch.stack([s * m * jx + off, s * m * jy + off], dim=-1)

    def argmax_center(self, logits: torch.Tensor) -> torch.Tensor:
        b = logits.shape[0]
        idx = logits.reshape(b, -1).argmax(dim=1)
        return self.center_from_index(idx)

    # ------------------------------------------------------------ fine stage
    def refine(self, search_fine: torch.Tensor, ref: torch.Tensor,
               seed_center: torch.Tensor, iters: Optional[int] = None
               ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        iters = iters or self.g.fine_iters
        box = self.resampler.box_filter(search_fine)
        ref_feat = self.subpixel.encode(ref)
        c = seed_center
        centers: List[torch.Tensor] = []
        conf = torch.zeros(c.shape[0], device=c.device, dtype=c.dtype)
        for _ in range(iters):
            patch = self.resampler(box, c, self.g.fine_patch_coarse)
            delta, _, conf = self.subpixel(patch, ref_feat)
            c = c + self.g.scale * delta                    # coarse px -> fine px
            centers.append(c)
        return c, centers, conf

    # ---------------------------------------------------------------- forward
    def forward(self, search: torch.Tensor, ref: torch.Tensor,
                search_coarse: Optional[torch.Tensor] = None,
                seed_center: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        if search_coarse is None:
            search_coarse = self.make_coarse(search, self.g.scale)
        out = self.score_maps(search_coarse, ref)
        seed = self.argmax_center(out["logits"]) if seed_center is None else seed_center
        center, centers, conf = self.refine(search, ref, seed)
        out.update({"center": center, "centers": centers, "seed": seed,
                    "fine_conf": conf})
        return out


# ==============================================================================
# 8.  Deployment graphs (two-stage so the tie-break stays exact)
# ==============================================================================
class CoarseScorerExport(nn.Module):
    """
    Graph 1 -- inputs: search (1,1,1000,1000), ref (1,1,40,40)
               outputs: logits (1,1,31,31), period (1,2)

    The tie-break rule must run on the *candidate list*, so it deliberately
    lives outside the graph (pure host-side arithmetic on a 31x31 map, ~microseconds).
    """

    def __init__(self, net: DriftSenseNet):
        super().__init__()
        self.net = net

    def forward(self, search: torch.Tensor, ref: torch.Tensor):
        sc = DriftSenseNet.make_coarse(search, self.net.g.scale)
        o = self.net.score_maps(sc, ref)
        return o["logits"], o["period"]


class FineRefinerExport(nn.Module):
    """
    Graph 2 -- inputs: search (1,1,1000,1000), ref (1,1,40,40), center (1,2)
               outputs: center_refined (1,2), conf (1,)

    Called once, after the host has applied the center-proximity tie-break.
    """

    def __init__(self, net: DriftSenseNet):
        super().__init__()
        self.net = net

    def forward(self, search: torch.Tensor, ref: torch.Tensor,
                center: torch.Tensor):
        c, _, conf = self.net.refine(search, ref, center)
        return c, conf


# ==============================================================================
# 9.  Smoke test
# ==============================================================================
def _smoke() -> None:
    torch.manual_seed(0)
    cfg = DEFAULT
    g = cfg.geom
    net = DriftSenseNet(cfg).eval()
    n_par = sum(p.numel() for p in net.parameters())
    print(f"params: {n_par/1e6:.2f} M")

    b = 2
    search = torch.rand(b, 1, g.search_size, g.search_size)
    ref = torch.rand(b, 1, g.ref_coarse_size, g.ref_coarse_size)
    with torch.no_grad():
        out = net(search, ref)
    print("logits_a", tuple(out["logits_a"].shape),
          "logits_b", tuple(out["logits_b"].shape),
          "fused", tuple(out["logits"].shape),
          "center", tuple(out["center"].shape),
          "period", tuple(out["period"].shape))
    assert out["logits_a"].shape[-2:] == (g.map_a_size, g.map_a_size)
    assert out["logits_b"].shape[-2:] == (g.map_b_size, g.map_b_size)

    # --- geometric identity test (the single most important correctness check):
    # --- resampling the FULL-RES search at the ground-truth centre must
    # --- reproduce, to float precision, the exact 10x area-average that the
    # --- data generator used to build the reference.  If this fails, every
    # --- sub-pixel number downstream is meaningless.
    from dataset_generator import DriftSenseSynthesizer, area_downsample
    import numpy as np
    syn = DriftSenseSynthesizer(cfg, seed=11)
    s = syn.sample("sram_contact")
    fine = torch.from_numpy(np.ascontiguousarray(s.search[None, None])).float()
    box = net.resampler.box_filter(fine)
    c = torch.tensor([[s.center[0], s.center[1]]], dtype=torch.float32)
    patch = net.resampler(box, c, g.ref_coarse_size)                     # 40x40
    x0, y0 = int(round(s.t[0] * g.scale)), int(round(s.t[1] * g.scale))
    tgt = torch.from_numpy(area_downsample(
        s.search[y0:y0 + g.ref_fine_size, x0:x0 + g.ref_fine_size], g.scale)
    )[None, None].float()
    err = (patch - tgt).abs().max().item()
    print(f"phase-resampler identity max|err| = {err:.3e}  (must be < 1e-4)")
    assert err < 1e-4, "phase-aware resampler is not consistent with the generator"

    # --- sub-pixel sensitivity: a 1-fine-pixel shift of the sampling centre
    # --- must produce a measurable change in the resampled patch, otherwise
    # --- sub-pixel regression has no signal to learn from.
    p1 = net.resampler(box, c + torch.tensor([[1.0, 0.0]]), g.ref_coarse_size)
    print(f"d(patch)/d(1 fine px shift) = {(p1 - patch).abs().mean().item():.3e}")

    # --- index <-> coordinate round trip
    idx = torch.tensor([0, g.map_b_size * g.map_b_size - 1])
    cc = net.center_from_index(idx)
    assert abs(cc[0, 0].item() - g.center_from_t(0)) < 1e-4
    assert abs(cc[1, 0].item() - g.center_from_t(g.t_max)) < 1e-4
    print("center_from_index endpoints:", cc.tolist())

    # --- unfold correlation must equal conv correlation
    a = torch.randn(3, 8, 25, 25); bb = torch.randn(3, 8, 10, 10)
    d = (correlate_valid(a, bb, impl="conv") - correlate_valid(a, bb, impl="unfold")
         ).abs().max().item()
    print(f"corr impl agreement max|diff| = {d:.3e}")
    assert d < 1e-4
    print("drift_sense_model smoke OK")


if __name__ == "__main__":
    _smoke()
