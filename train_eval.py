"""
train_eval.py
================================================================================
End-to-end training, evaluation and deployment pipeline for DRIFT-SENSE.

    python train_eval.py train                 # train (synthetic, infinite data)
    python train_eval.py eval  --ckpt runs/drift_sense/best.pt
    python train_eval.py eval  --ckpt ... --ood      # held-out layout families
    python train_eval.py export --ckpt ... --onnx out/            # ONNX + ORT bench
    python train_eval.py bench --ckpt ...                         # torch latency
    python train_eval.py baseline                                 # classical ZNCC

Metrics reported
    MAE / RMSE / median / p95 of the Euclidean center error (FINE pixels)
    success @ {0.5, 1, 2, 5, 10} fine px
    sub-pixel accuracy rate            = P(error < 1 fine px)   [= 0.1 coarse px]
    lattice-lock failure rate          = P(error is a non-zero multiple of the
                                          lattice period)  <-- the fatal mode
    tie-break activation / rescue rate
    SSIM (and an optional LPIPS-style perceptual distance) between the reference
        and the phase-resampled patch at the predicted center -- a *verification*
        signal available at inference time, with no ground truth needed.
    latency: torch eager, torch CUDA-graph-free, ONNX Runtime, TensorRT hint
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config, DEFAULT, Geometry
from dataset_generator import DriftSenseDataset, DriftSenseSynthesizer
from drift_sense_model import (CoarseScorerExport, DriftSenseNet, FineRefinerExport)
from losses_and_postprocess import (DriftSenseCriterion, classical_zncc,
                                    extract_candidates, lattice_lock_error,
                                    resolve_match, tie_break_center_proximity)


# ==============================================================================
# 0.  Utilities
# ==============================================================================
def set_seed(s: int) -> None:
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Meter:
    def __init__(self) -> None:
        self.d: Dict[str, float] = {}
        self.n = 0

    def add(self, logs: Dict[str, float]) -> None:
        for k, v in logs.items():
            self.d[k] = self.d.get(k, 0.0) + float(v)
        self.n += 1

    def mean(self) -> Dict[str, float]:
        return {k: v / max(self.n, 1) for k, v in self.d.items()}


# ==============================================================================
# 1.  Verification metrics (SSIM / LPIPS-style)
# ==============================================================================
def _gauss_win(ks: int, sigma: float, device, dtype) -> torch.Tensor:
    x = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def ssim(a: torch.Tensor, b: torch.Tensor, ks: int = 11, sigma: float = 1.5
         ) -> torch.Tensor:
    """
    Standard SSIM on (B,1,H,W).  Inputs are standardised first, so the metric
    reports *structural* agreement only -- the correct behaviour here, because
    the reference and the search branch have different gain, bias and speckle
    realisations by construction.
    """
    def _std(x: torch.Tensor) -> torch.Tensor:
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        s = x.std(dim=(1, 2, 3), keepdim=True)
        return (x - m) / (s + 1e-6)

    a, b = _std(a), _std(b)
    w = _gauss_win(ks, sigma, a.device, a.dtype)
    dr = float(torch.maximum(a.max(), b.max()) - torch.minimum(a.min(), b.min()))
    c1, c2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
    mu_a = F.conv2d(a, w, padding=ks // 2)
    mu_b = F.conv2d(b, w, padding=ks // 2)
    saa = F.conv2d(a * a, w, padding=ks // 2) - mu_a * mu_a
    sbb = F.conv2d(b * b, w, padding=ks // 2) - mu_b * mu_b
    sab = F.conv2d(a * b, w, padding=ks // 2) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / \
        ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return s.flatten(1).mean(1)


class PerceptualDistance(nn.Module):
    """
    LPIPS-style perceptual distance: mean squared distance between unit-
    normalised VGG-16 feature maps.  Requires downloadable torchvision weights;
    if unavailable the module reports NaN instead of silently degrading to a
    meaningless number.  (Honest labelling: the official LPIPS linear
    calibration weights are NOT used here -- this is the uncalibrated variant.)
    """

    def __init__(self) -> None:
        super().__init__()
        self.ok = False
        try:
            from torchvision.models import VGG16_Weights, vgg16
            v = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
            for p in v.parameters():
                p.requires_grad_(False)
            self.v = v
            self.ok = True
        except Exception as e:                                  # pragma: no cover
            print(f"[warn] LPIPS-style metric disabled ({type(e).__name__}: {e})")

    @torch.no_grad()
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not self.ok:
            return torch.full((a.shape[0],), float("nan"), device=a.device)

        def prep(x: torch.Tensor) -> torch.Tensor:
            m = x.mean(dim=(1, 2, 3), keepdim=True)
            s = x.std(dim=(1, 2, 3), keepdim=True)
            x = ((x - m) / (s + 1e-6)).clamp(-3, 3) / 6.0 + 0.5
            return x.repeat(1, 3, 1, 1)

        fa = F.normalize(self.v(prep(a)), dim=1)
        fb = F.normalize(self.v(prep(b)), dim=1)
        return ((fa - fb) ** 2).sum(1).flatten(1).mean(1)


# ==============================================================================
# 2.  Inference: coarse scoring -> tie-break -> sub-pixel refinement
# ==============================================================================
@torch.no_grad()
def predict(net: DriftSenseNet, search: torch.Tensor, ref: torch.Tensor,
            search_coarse: Optional[torch.Tensor] = None,
            prob_ratio: float = 0.60, return_diag: bool = False
            ) -> Tuple[torch.Tensor, List[Dict[str, object]]]:
    """
    The deployed decision path, batched.

      1. one coarse forward -> fused 31x31 score map
      2. per sample, on the host: NMS -> candidate list -> CENTER-PROXIMITY
         TIE-BREAK  (this must happen BEFORE refinement, otherwise the sub-pixel
         head would already have committed to the argmax peak)
      3. one fine forward from the tie-broken seed -> sub-pixel (x, y)
    """
    g = net.g
    if search_coarse is None:
        search_coarse = DriftSenseNet.make_coarse(search, g.scale)
    out = net.score_maps(search_coarse, ref)
    lg = out["logits"].detach().float().cpu().numpy()
    diags: List[Dict[str, object]] = []
    seeds = []
    for i in range(lg.shape[0]):
        r = resolve_match(lg[i, 0], g, prob_ratio=prob_ratio)
        seeds.append(r["center"])
        diags.append(r)
    seed = torch.tensor(seeds, dtype=search.dtype, device=search.device)
    center, _, conf = net.refine(search, ref, seed)
    if return_diag:
        for i, d in enumerate(diags):
            d["seed"] = seeds[i]
            d["fine_conf"] = float(conf[i])
            d["period_coarse"] = out["period"][i].tolist()
    return center, diags


# ==============================================================================
# 3.  Training
# ==============================================================================
def train_forward(net: DriftSenseNet, crit: DriftSenseCriterion,
                  batch: Dict[str, torch.Tensor], cfg: Config
                  ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    One training step's forward pass.

    Scheduled sampling on the fine-stage seed.  With probability
    `teacher_forcing_p` the seed is the ground-truth centre plus uniform jitter
    inside the +-2-coarse-px capture range; otherwise it is the network's own
    argmax.  Pure teacher forcing never teaches the head to cope with real seeds;
    pure self-seeding gives the head no learnable signal until the coarse stage
    already works.  The mixture makes both stages trainable from scratch
    simultaneously.
    """
    g, lc = net.g, cfg.loss
    out = net.score_maps(batch["search_coarse"], batch["ref"])
    b = batch["ref"].shape[0]
    dev, dt = batch["center"].device, batch["center"].dtype

    argmax_seed = net.argmax_center(out["logits"])
    rad = g.fine_search_radius_coarse * g.scale                # +-20 fine px
    jitter = (torch.rand(b, 2, device=dev, dtype=dt) * 2 - 1) * rad
    gt_seed = batch["center"] + jitter
    use_gt = (torch.rand(b, 1, device=dev) < lc.teacher_forcing_p).to(dt)
    seed = use_gt * gt_seed + (1 - use_gt) * argmax_seed

    # the fine stage is deliberately kept in fp32: it resolves 1/200 of a coarse
    # pixel, which is below fp16 resolution for coordinates of order 1e3.
    with torch.autocast(device_type=dev.type, enabled=False):
        center, centers, conf = net.refine(batch["search"].float(),
                                           batch["ref"].float(), seed.float())
    out.update({"center": center, "centers": centers, "seed": seed,
                "fine_conf": conf})

    with torch.no_grad():
        p4_clean = net.encoder.trunk(batch["search_coarse_clean"])["p4"]
    return crit(out, batch, p4s_clean=p4_clean)


def lr_at(it: int, cfg: Config) -> float:
    tc = cfg.train
    total = tc.epochs * tc.iters_per_epoch
    if it < tc.warmup_iters:
        return tc.lr * (it + 1) / tc.warmup_iters
    p = (it - tc.warmup_iters) / max(total - tc.warmup_iters, 1)
    return 0.5 * tc.lr * (1 + math.cos(math.pi * min(p, 1.0)))


def train(cfg: Config = DEFAULT, resume: Optional[str] = None) -> str:
    set_seed(cfg.train.seed)
    dev = pick_device()
    os.makedirs(cfg.train.out_dir, exist_ok=True)
    print(f"device={dev}")

    net = DriftSenseNet(cfg).to(dev)
    crit = DriftSenseCriterion(cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    scaler = torch.amp.GradScaler(enabled=cfg.train.amp and dev.type == "cuda")
    start_ep, best = 0, float("inf")
    if resume and os.path.isfile(resume):
        ck = torch.load(resume, map_location=dev)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_ep, best = ck["epoch"] + 1, ck.get("best", float("inf"))
        print(f"resumed from {resume} @ epoch {start_ep}")

    tr = DriftSenseDataset(cfg, layouts=cfg.train.train_layouts,
                           length=cfg.train.iters_per_epoch * cfg.train.batch_size,
                           seed=cfg.train.seed)
    va = DriftSenseDataset(cfg, layouts=cfg.train.train_layouts,
                           length=cfg.train.val_iters * cfg.train.batch_size,
                           seed=cfg.train.seed + 99991)
    kw = dict(batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers,
              pin_memory=(dev.type == "cuda"), drop_last=True,
              persistent_workers=cfg.train.num_workers > 0)
    dl_tr, dl_va = DataLoader(tr, shuffle=False, **kw), DataLoader(va, shuffle=False, **kw)

    it = start_ep * cfg.train.iters_per_epoch
    ckpt_path = os.path.join(cfg.train.out_dir, "best.pt")
    for ep in range(start_ep, cfg.train.epochs):
        net.train()
        m, t0 = Meter(), time.time()
        for batch in dl_tr:
            batch = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            for gp in opt.param_groups:
                gp["lr"] = lr_at(it, cfg)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=scaler.is_enabled()):
                loss, logs = train_forward(net, crit, batch, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.train.grad_clip)
            scaler.step(opt); scaler.update()
            m.add(logs); it += 1
        tr_logs = m.mean()
        ev = evaluate(net, dl_va, cfg, dev, tag=f"val ep{ep}")
        score = ev["mae_px"]
        flag = ""
        if score < best:
            best = score
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                        "epoch": ep, "best": best, "cfg": asdict(cfg)}, ckpt_path)
            flag = "  *saved*"
        print(f"ep{ep:03d} {time.time()-t0:6.1f}s lr={lr_at(it,cfg):.2e} "
              + " ".join(f"{k}={v:.3f}" for k, v in tr_logs.items())
              + f" | val MAE={score:.3f}px sub-px={ev['subpixel_rate']:.3f}"
              + f" lock-fail={ev['lattice_lock_rate']:.3f}{flag}")
    print(f"best MAE {best:.3f} px -> {ckpt_path}")
    return ckpt_path


# ==============================================================================
# 4.  Evaluation
# ==============================================================================
@torch.no_grad()
def evaluate(net: DriftSenseNet, loader: DataLoader, cfg: Config,
             dev: torch.device, tag: str = "eval",
             perceptual: Optional[PerceptualDistance] = None,
             verbose: bool = False) -> Dict[str, float]:
    net.eval()
    g = net.g
    errs: List[float] = []
    locks: List[float] = []
    amb: List[float] = []
    rescue: List[float] = []
    ssims: List[float] = []
    lpips: List[float] = []
    for batch in loader:
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        center, diags = predict(net, batch["search"], batch["ref"],
                                batch["search_coarse"], return_diag=True)
        gt = batch["center"]
        e = (center - gt).norm(dim=1)
        errs += e.tolist()

        # verification: does the patch resampled at the prediction match the ref?
        box = net.resampler.box_filter(batch["search"])
        patch = net.resampler(box, center, g.ref_coarse_size)
        ssims += ssim(patch, batch["ref"]).tolist()
        if perceptual is not None:
            lpips += perceptual(patch, batch["ref"]).tolist()

        for i, d in enumerate(diags):
            per = tuple(batch["period"][i].tolist())
            gt_i = (float(gt[i, 0]), float(gt[i, 1]))
            locks.append(float(lattice_lock_error(
                (float(center[i, 0]), float(center[i, 1])), gt_i, per)))
            amb.append(float(bool(d["ambiguous"])))
            # would the plain argmax peak have been a lattice-lock failure that
            # the center rule then rescued?
            raw = max(d["candidates"], key=lambda c: c.score)
            raw_bad = lattice_lock_error((raw.x, raw.y), gt_i, per)
            rescue.append(float(raw_bad and not locks[-1]))

    a = np.asarray(errs, dtype=np.float64)
    res = {
        "n": float(a.size),
        "mae_px": float(a.mean()),
        "rmse_px": float(np.sqrt((a ** 2).mean())),
        "median_px": float(np.median(a)),
        "p95_px": float(np.percentile(a, 95)),
        "subpixel_rate": float((a < 1.0).mean()),           # < 1 fine px
        "acc_0.5px": float((a < 0.5).mean()),
        "acc_1px": float((a < 1.0).mean()),
        "acc_2px": float((a < 2.0).mean()),
        "acc_5px": float((a < 5.0).mean()),
        "acc_10px": float((a < 10.0).mean()),
        "lattice_lock_rate": float(np.mean(locks)),
        "ambiguous_rate": float(np.mean(amb)),
        "tiebreak_rescue_rate": float(np.mean(rescue)),
        "ssim": float(np.mean(ssims)),
        "coarse_px_err": float(a.mean() / g.scale),          # error in ref pixels
    }
    if lpips:
        res["lpips_style"] = float(np.mean(lpips))
    if verbose:
        print(f"[{tag}] " + json.dumps(res, indent=2))
    return res


def evaluate_ckpt(ckpt: str, cfg: Config = DEFAULT, ood: bool = False,
                  n_batches: int = 64, with_lpips: bool = False) -> Dict[str, float]:
    dev = pick_device()
    net = DriftSenseNet(cfg).to(dev)
    net.load_state_dict(torch.load(ckpt, map_location=dev)["model"])
    layouts = cfg.train.ood_layouts if ood else cfg.train.train_layouts
    ds = DriftSenseDataset(cfg, layouts=layouts,
                           length=n_batches * cfg.train.batch_size, seed=777)
    dl = DataLoader(ds, batch_size=cfg.train.batch_size,
                    num_workers=cfg.train.num_workers, drop_last=True)
    per = PerceptualDistance().to(dev) if with_lpips else None
    tag = "OOD" if ood else "in-distribution"
    return evaluate(net, dl, cfg, dev, tag=tag, perceptual=per, verbose=True)


# ==============================================================================
# 5.  Classical baseline (for the comparison table in the write-up)
# ==============================================================================
def run_baseline(cfg: Config = DEFAULT, n: int = 64, seed: int = 4242
                 ) -> Dict[str, Dict[str, float]]:
    """Dense ZNCC on the coarse pair: raw argmax vs. argmax + the center rule."""
    syn = DriftSenseSynthesizer(cfg, seed=seed)
    g = cfg.geom
    out: Dict[str, List[Tuple[float, float]]] = {"argmax": [], "argmax+center": []}
    for _ in range(n):
        s = syn.sample()
        m = classical_zncc(s.search_coarse, s.ref, g)
        cands = extract_candidates(m, g, max_candidates=48, pitch_coarse=1.0)
        am = max(cands, key=lambda c: c.score)
        tb, _, _ = tie_break_center_proximity(cands, prob_ratio=0.0,
                                              logit_margin=0.02)
        for name, c in (("argmax", am), ("argmax+center", tb)):
            e = math.hypot(c.x - s.center[0], c.y - s.center[1])
            lock = lattice_lock_error((c.x, c.y), s.center, s.period)
            out[name].append((e, float(lock)))
    res = {}
    for k, v in out.items():
        a = np.asarray([x[0] for x in v]); l = np.asarray([x[1] for x in v])
        res[k] = {"mae_px": float(a.mean()), "median_px": float(np.median(a)),
                  "acc_10px": float((a < 10).mean()),
                  "acc_2px": float((a < 2).mean()),
                  "lattice_lock_rate": float(l.mean())}
    print(json.dumps(res, indent=2))
    return res


# ==============================================================================
# 6.  Latency: torch / ONNX Runtime / TensorRT
# ==============================================================================
@torch.no_grad()
def bench_torch(net: DriftSenseNet, dev: torch.device, iters: int = 200,
                warmup: int = 30) -> Dict[str, float]:
    g = net.g
    net.eval().to(dev)
    s = torch.rand(1, 1, g.search_size, g.search_size, device=dev)
    r = torch.rand(1, 1, g.ref_coarse_size, g.ref_coarse_size, device=dev)
    sc = DriftSenseNet.make_coarse(s, g.scale)
    seed = torch.tensor([[499.5, 499.5]], device=dev)

    def timeit(fn) -> float:
        for _ in range(warmup):
            fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    t_coarse = timeit(lambda: net.score_maps(sc, r))
    t_fine = timeit(lambda: net.refine(s, r, seed))
    t_pool = timeit(lambda: DriftSenseNet.make_coarse(s, g.scale))
    out = {"coarse_ms": t_coarse, "fine_ms": t_fine, "pool_ms": t_pool,
           "total_ms": t_coarse + t_fine + t_pool}
    print("torch latency: " + " ".join(f"{k}={v:.2f}" for k, v in out.items()))
    return out


def export_onnx(net: DriftSenseNet, out_dir: str, opset: int = 17,
                verify: bool = True) -> Dict[str, str]:
    """
    Two static-shape graphs, by design:

      coarse_scorer.onnx : search, ref            -> logits(1,1,31,31), period(1,2)
      fine_refiner.onnx  : search, ref, center    -> center(1,2), conf(1,)

    The center-proximity tie-break sits between them on the host.  Keeping it out
    of the graph is not a limitation, it is a requirement: the rule needs the
    *candidate set*, and burning an argmax into the graph would destroy exactly
    the information the rule consumes.  Its cost is a 31x31 NMS -- microseconds.

    If a backend rejects a Conv whose weight is an activation (the batched
    correlation), re-export with net.corr_impl = 'unfold'.
    """
    os.makedirs(out_dir, exist_ok=True)
    g = net.g
    net = net.eval().cpu()
    s = torch.rand(1, 1, g.search_size, g.search_size)
    r = torch.rand(1, 1, g.ref_coarse_size, g.ref_coarse_size)
    c = torch.tensor([[499.5, 499.5]])
    paths = {"coarse": os.path.join(out_dir, "coarse_scorer.onnx"),
             "fine": os.path.join(out_dir, "fine_refiner.onnx")}

    torch.onnx.export(CoarseScorerExport(net), (s, r), paths["coarse"],
                      input_names=["search", "ref"],
                      output_names=["logits", "period"],
                      opset_version=opset, do_constant_folding=True)
    torch.onnx.export(FineRefinerExport(net), (s, r, c), paths["fine"],
                      input_names=["search", "ref", "center"],
                      output_names=["center_refined", "conf"],
                      opset_version=opset, do_constant_folding=True)
    print(f"exported:\n  {paths['coarse']}\n  {paths['fine']}")

    if verify:
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            prov = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if "CUDAExecutionProvider" in ort.get_available_providers()
                    else ["CPUExecutionProvider"])
            sc = ort.InferenceSession(paths["coarse"], so, providers=prov)
            sf = ort.InferenceSession(paths["fine"], so, providers=prov)
            fs = {"search": s.numpy(), "ref": r.numpy()}
            lo, per = sc.run(None, fs)
            with torch.no_grad():
                lo_t, per_t = CoarseScorerExport(net)(s, r)
            d = float(np.abs(lo - lo_t.numpy()).max())
            print(f"ORT parity (coarse logits) max|diff| = {d:.3e}")

            # end-to-end with the tie-break on the host
            res = resolve_match(lo[0, 0], g)
            cen = np.asarray([res["center"]], dtype=np.float32)
            cr, cf = sf.run(None, {"search": s.numpy(), "ref": r.numpy(),
                                   "center": cen})
            print(f"ORT end-to-end center = {cr[0].tolist()} (conf {float(cf[0]):.3f})")

            def bench(sess, feeds, n=100) -> float:
                for _ in range(20):
                    sess.run(None, feeds)
                t0 = time.perf_counter()
                for _ in range(n):
                    sess.run(None, feeds)
                return (time.perf_counter() - t0) / n * 1e3

            tc = bench(sc, fs)
            tf = bench(sf, {"search": s.numpy(), "ref": r.numpy(), "center": cen})
            t_tb0 = time.perf_counter()
            for _ in range(1000):
                resolve_match(lo[0, 0], g)
            t_tb = (time.perf_counter() - t_tb0) / 1000 * 1e3
            print(f"ORT latency: coarse={tc:.2f}ms  tie-break={t_tb:.3f}ms  "
                  f"fine={tf:.2f}ms  total={tc+t_tb+tf:.2f}ms  providers={prov}")
        except Exception as e:
            print(f"[warn] ORT verification skipped ({type(e).__name__}: {e})")

    print("\nTensorRT (FP16) build, run on the fab controller:")
    for k in ("coarse", "fine"):
        print(f"  trtexec --onnx={paths[k]} --saveEngine={paths[k][:-5]}.fp16.plan "
              f"--fp16 --builderOptimizationLevel=5 --useCudaGraph")
    print("  # INT8 is NOT recommended for the fine head: its output is a "
          "sub-pixel coordinate,\n  # and INT8 activation quantisation of the "
          "5x5 correlation tensor costs ~0.3 fine px.")
    return paths


# ==============================================================================
# 7.  CLI
# ==============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="DRIFT-SENSE train / eval / export")
    ap.add_argument("mode", choices=["train", "eval", "export", "bench",
                                     "baseline", "selftest"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--onnx", default="out_onnx")
    ap.add_argument("--ood", action="store_true")
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--batches", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    cfg = DEFAULT
    if a.epochs is not None:
        cfg.train.epochs = a.epochs
    if a.workers is not None:
        cfg.train.num_workers = a.workers

    if a.mode == "train":
        train(cfg, resume=a.resume)
    elif a.mode == "eval":
        assert a.ckpt, "--ckpt required"
        evaluate_ckpt(a.ckpt, cfg, ood=a.ood, n_batches=a.batches,
                      with_lpips=a.lpips)
    elif a.mode == "export":
        net = DriftSenseNet(cfg)
        if a.ckpt:
            net.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
        export_onnx(net, a.onnx)
    elif a.mode == "bench":
        net = DriftSenseNet(cfg)
        if a.ckpt:
            net.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
        bench_torch(net, pick_device())
    elif a.mode == "baseline":
        run_baseline(cfg, n=a.batches)
    elif a.mode == "selftest":
        selftest(cfg)


def selftest(cfg: Config = DEFAULT) -> None:
    """
    Overfit-one-batch test: the fastest way to prove the whole graph -- geometry,
    losses, gradients -- is wired correctly.  A correct implementation drives the
    center error on a single fixed batch below 1 fine pixel within a few hundred
    steps.
    """
    dev = pick_device()
    print(f"device={dev}")
    set_seed(0)
    net = DriftSenseNet(cfg).to(dev)
    crit = DriftSenseCriterion(cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=6e-4)

    syn = DriftSenseSynthesizer(cfg, seed=5)
    samples = [syn.sample("dram_capacitor") for _ in range(2)]
    batch = {k: torch.stack([torch.from_numpy(np.ascontiguousarray(
                 s.as_dict()[k])).float() for s in samples]).to(dev)
             for k in ("search", "search_coarse", "search_coarse_clean", "ref",
                       "center", "t", "period")}

    print(f"gt centers: {batch['center'].tolist()}")
    for it in range(301):
        net.train()
        opt.zero_grad(set_to_none=True)
        loss, logs = train_forward(net, crit, batch, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if it % 50 == 0:
            net.eval()
            with torch.no_grad():
                c, d = predict(net, batch["search"], batch["ref"],
                               batch["search_coarse"], return_diag=True)
            e = (c - batch["center"]).norm(dim=1)
            print(f"it{it:04d} loss={logs['loss']:7.3f} fine={logs['fine']:6.3f} "
                  f"decoy={logs['decoy']:6.3f} | err={e.tolist()} "
                  f"n_tied={[x['n_tied'] for x in d]}")
    print("selftest done (expect err -> sub-pixel; if not, geometry is wrong)")


if __name__ == "__main__":
    main()
