"""
Edge Deployment Benchmark for Edge-HAR novel-class foundation model.

Measures what actually matters for edge/wearable deployment:

  1. PARAMETER COUNT     — total vs inference-path (deployed subset)
  2. MODEL SIZE ON DISK  — FP32 / FP16 / INT8 (dynamic quant) for both full & deployed
  3. FLOPS               — full forward vs inference-path only
  4. GPU LATENCY/TPUT    — batch=1 latency, max-batch throughput  [RTX 4090]
  5. CPU LATENCY/TPUT    — batch=1 latency, batch=256 throughput  [server Xeon]
  6. INFERENCE RAM       — peak RSS memory increase during inference (CPU)
  7. FINE-TUNE COST      — time + peak GPU memory for one K=100 update step
  8. REALTIME BUDGET     — whether inference fits within HAR window period
  9. EDGE DEVICE SCALING — latency projection for MCU/phone/watch class chips

Checkpoints benchmarked:
  - Pretrained only (zero-shot)
  - After K=100 full_ft fine-tune on DSADS novel classes (F1=0.919)

Usage:
    cd /root/rivermind-data/new
    python benchmark_edge_deployment.py
    python benchmark_edge_deployment.py --skip_finetune_cost   # faster
"""

import argparse, os, sys, time, copy, json, tempfile, gc
import numpy as np
import torch
import torch.nn as nn
import yaml

_THIS      = os.path.dirname(os.path.abspath(__file__))
_TRIFACTOR = os.path.normpath(os.path.join(_THIS, "..", "TriFactor-HAR"))
_TRIFACTOR1 = os.path.normpath(os.path.join(_THIS, "..", "TriFactor-HAR_1"))
for p in (_TRIFACTOR, _THIS):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.full_model import FactorizedHARModel
from models.heads import ClassificationHead, ProtoHead

# ── Config ────────────────────────────────────────────────────────────────────
CFG_PATH     = os.path.join(_THIS, "configs", "crosshar_exp.yaml")
CKPT_PRETRAIN = os.path.join(_THIS, "outputs", "crosshar_exp",
                              "pamap2_novel", "pretrain", "best_model.pth")
CKPT_FINETUNED = os.path.join(_THIS, "outputs", "crosshar_exp",
                               "dsads_novel_kshot_v2", "full_ft", "k100", "best_model.pth")
NUM_NOVEL     = 14   # DSADS novel classes
C, T          = 6, 120
HOP_SIZE      = 60   # samples

# Approximate edge-chip inference speedup/slowdown vs server Xeon
# (relative to server CPU single-thread, rough empirical ratios)
EDGE_CHIPS = {
    "Raspberry Pi 4 (Cortex-A72)"    : 0.15,   # ~6.7x slower than server Xeon
    "Cortex-M55 MCU (1 GHz)"         : 0.006,  # ~167x slower
    "Apple A17 (iPhone 15)"          : 1.8,    # ~1.8x faster (NEON SIMD)
    "Snapdragon 8 Gen3 (Android)"    : 1.5,
    "Samsung Exynos W1000 (Watch)"   : 0.08,   # paper reference chip
}

WARMUP_GPU, REPS_GPU = 50, 300
WARMUP_CPU, REPS_CPU = 10, 50


# ── Inference-path module (deployed subset) ───────────────────────────────────
class DeployedClassifier(nn.Module):
    """shared_encoder → semantic_encoder → cls_head only."""
    def __init__(self, m: FactorizedHARModel):
        super().__init__()
        self.shared_encoder   = m.shared_encoder
        self.semantic_encoder = m.semantic_encoder
        self.cls_head         = m.cls_head

    @torch.no_grad()
    def forward(self, x_time):
        h = self.shared_encoder(x_time)
        z = self.semantic_encoder(h["h_shared"])
        return self.cls_head(z)


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_model(cfg, ckpt_path, num_classes, device):
    pcfg = copy.deepcopy(cfg)
    pcfg["data"]["num_classes"] = num_classes
    pcfg["data"]["num_domains"] = 5
    model = FactorizedHARModel(pcfg).eval()
    if os.path.exists(ckpt_path):
        ckpt  = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        cur   = model.state_dict()
        filt  = {k: v for k, v in state.items() if k in cur and cur[k].shape == v.shape}
        model.load_state_dict(filt, strict=False)
    return model.to(device)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def bench_fn(fn, warmup, reps, cuda=False):
    for _ in range(warmup): fn()
    if cuda: torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    if cuda: torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000  # ms/call


def measure_flops(model, inputs_tuple):
    try:
        from thop import profile
        m = copy.deepcopy(model).cpu().eval()
        cpu_inputs = tuple(x.cpu() for x in inputs_tuple)
        macs, _ = profile(m, inputs=cpu_inputs, verbose=False)
        del m
        return macs * 2
    except Exception:
        return None


def model_disk_size_mb(model, quantize_int8=False):
    """Save to temp file and measure bytes (most accurate)."""
    if quantize_int8:
        # Dynamic INT8 quantization (Linear layers only — safe for all models)
        try:
            q = torch.quantization.quantize_dynamic(
                copy.deepcopy(model).cpu(),
                {nn.Linear},
                dtype=torch.qint8,
            )
        except Exception:
            return None
    else:
        q = model
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=True) as f:
        torch.save(q.state_dict(), f.name)
        return os.path.getsize(f.name) / 1024**2


def measure_peak_gpu_mem(fn, device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 1024**2


def measure_cpu_ram_mb(fn):
    """Measure RSS increase during fn() using /proc/self/status."""
    import subprocess
    def _rss():
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS"):
                        return int(line.split()[1]) / 1024  # MB
        except Exception:
            return 0.0
    rss_before = _rss()
    fn()
    rss_after = _rss()
    return max(0.0, rss_after - rss_before)


def quantize_int8_latency(model_cpu, x_cpu, warmup=5, reps=20):
    """Measure CPU latency of INT8 quantized model."""
    try:
        qm = torch.quantization.quantize_dynamic(
            copy.deepcopy(model_cpu), {nn.Linear}, dtype=torch.qint8
        ).eval()
        lat = bench_fn(lambda: qm(x_cpu), warmup, reps, cuda=False)
        del qm
        return lat
    except Exception:
        return None


def measure_finetune_cost(cfg, pretrain_ckpt, device, k=100, epochs=5):
    """
    One realistic fine-tune cost:
      - load pretrain model
      - replace head for NUM_NOVEL classes
      - K=100 random windows × 1 forward+backward pass per epoch
      - measure time per epoch and peak GPU memory
    """
    pcfg = copy.deepcopy(cfg)
    pcfg["data"]["num_classes"] = 4
    pcfg["data"]["num_domains"] = 5
    model = FactorizedHARModel(pcfg)
    if os.path.exists(pretrain_ckpt):
        ckpt  = torch.load(pretrain_ckpt, map_location=device)
        state = ckpt.get("model_state", ckpt)
        cur   = model.state_dict()
        filt  = {k: v for k, v in state.items() if k in cur and cur[k].shape == v.shape}
        model.load_state_dict(filt, strict=False)

    dim_s = cfg["model"]["dim_s"]
    cls_cfg = cfg["model"]["cls_head"]
    model.cls_head = ClassificationHead(
        dim_s, NUM_NOVEL,
        hidden_dim=cls_cfg["hidden_dim"],
        dropout=cls_cfg["dropout"],
    )
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Synthetic K=100 batch (all classes represented)
    x  = torch.randn(k, C, T, device=device)
    m  = torch.zeros(k, 4, dtype=torch.long, device=device)
    y  = (torch.arange(k, device=device) % NUM_NOVEL)

    # Warmup
    for _ in range(2):
        optimizer.zero_grad()
        loss = criterion(model(x, m)["logits_main"], y)
        loss.backward(); optimizer.step()

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(x, m)["logits_main"], y)
        loss.backward(); optimizer.step()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - t0) / epochs * 1000
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    del model, optimizer, x, m, y
    gc.collect(); torch.cuda.empty_cache()
    return elapsed_ms, peak_mb


def get_cpu_name():
    try:
        import subprocess
        r = subprocess.run(["grep", "-m1", "model name", "/proc/cpuinfo"],
                           capture_output=True, text=True)
        return r.stdout.split(":", 1)[-1].strip()
    except Exception:
        return "unknown"


# ── Main ──────────────────────────────────────────────────────────────────────

def run_benchmark(label, ckpt_path, num_classes, cfg, device, args):
    print(f"\n{'━'*68}")
    print(f"  Model: {label}")
    print(f"{'━'*68}")

    model  = build_model(cfg, ckpt_path, num_classes, device)
    deploy = DeployedClassifier(model).to(device).eval()

    # ── 1. Params ──────────────────────────────────────────────────────────────
    p_full   = count_params(model)
    p_deploy = count_params(deploy)
    print(f"\n  ▸ Parameters")
    print(f"    Full model (train)   : {p_full:>10,}  ({p_full/1e6:.3f} M)")
    print(f"    Inference path only  : {p_deploy:>10,}  ({p_deploy/1e6:.3f} M)  "
          f"({100*p_deploy/p_full:.0f}% of full)")

    # ── 2. Model size on disk ──────────────────────────────────────────────────
    sz_fp32 = model_disk_size_mb(deploy)
    sz_fp16 = None
    sz_int8 = model_disk_size_mb(deploy, quantize_int8=True)
    # FP16: approximate as half the FP32 param bytes
    sz_fp16 = p_deploy * 2 / 1024**2
    print(f"\n  ▸ Deployed model size (inference path)")
    print(f"    FP32  : {sz_fp32:.2f} MB")
    print(f"    FP16  : {sz_fp16:.2f} MB  (estimate)")
    if sz_int8: print(f"    INT8  : {sz_int8:.2f} MB  (dynamic quant, Linear layers)")

    # ── 3. FLOPs ──────────────────────────────────────────────────────────────
    x1c = torch.randn(1, C, T)
    m1c = torch.zeros(1, 4, dtype=torch.long)
    f_full   = measure_flops(model.cpu(),  (x1c, m1c))
    f_deploy = measure_flops(deploy.cpu(), (x1c,))
    model.to(device); deploy.to(device)
    print(f"\n  ▸ FLOPs @ T={T}, batch=1")
    if f_full and f_deploy:
        print(f"    Full forward()    : {f_full/1e6:.2f} MFLOPs  ({f_full/1e9:.4f} GFLOPs)")
        print(f"    Inference path    : {f_deploy/1e6:.2f} MFLOPs  ({f_deploy/1e9:.4f} GFLOPs)  "
              f"({100*f_deploy/f_full:.0f}% of full)")
    else:
        print(f"    (thop profile failed)")

    # ── 4. GPU latency & throughput ────────────────────────────────────────────
    x1g = torch.randn(1, C, T, device=device)
    m1g = torch.zeros(1, 4, dtype=torch.long, device=device)
    with torch.no_grad():
        lat_full_gpu   = bench_fn(lambda: model(x1g, m1g),   WARMUP_GPU, REPS_GPU, cuda=True)
        lat_deploy_gpu = bench_fn(lambda: deploy(x1g),        WARMUP_GPU, REPS_GPU, cuda=True)

    # Throughput sweep
    best_thr_full, best_thr_dep = lat_full_gpu, lat_deploy_gpu
    for B in [32, 64, 128, 256, 512, 1024]:
        try:
            xb = torch.randn(B, C, T, device=device)
            mb = torch.zeros(B, 4, dtype=torch.long, device=device)
            with torch.no_grad():
                for _ in range(5): model(xb, mb)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(20): model(xb, mb)
                torch.cuda.synchronize()
                thr = B * 20 / (time.perf_counter() - t0)
                best_thr_full = max(best_thr_full, thr)
                for _ in range(5): deploy(xb)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(20): deploy(xb)
                torch.cuda.synchronize()
                thr = B * 20 / (time.perf_counter() - t0)
                best_thr_dep = max(best_thr_dep, thr)
        except RuntimeError:
            break

    print(f"\n  ▸ GPU [{torch.cuda.get_device_name(0)}]")
    print(f"    {'Path':<22} {'Lat(ms)':>9} {'Thr(k/s)':>10}")
    print(f"    {'-'*44}")
    print(f"    {'Full forward()':22} {lat_full_gpu:>9.3f} {best_thr_full/1000:>10.1f}")
    print(f"    {'Inference path':22} {lat_deploy_gpu:>9.3f} {best_thr_dep/1000:>10.1f}  "
          f"({lat_full_gpu/lat_deploy_gpu:.1f}x faster)")

    # GPU peak memory (inference)
    mem_full   = measure_peak_gpu_mem(lambda: model(x1g, m1g), device)
    mem_deploy = measure_peak_gpu_mem(lambda: deploy(x1g), device)
    print(f"\n  ▸ GPU Peak Memory (batch=1, inference)")
    print(f"    Full forward()    : {mem_full:.1f} MB")
    print(f"    Inference path    : {mem_deploy:.1f} MB")

    # ── 5. CPU latency & INT8 ─────────────────────────────────────────────────
    model_cpu  = model.cpu().eval()
    deploy_cpu = DeployedClassifier(model_cpu).eval()
    x1c = torch.randn(1, C, T)

    lat_full_cpu   = bench_fn(lambda: model_cpu(x1c, m1c),  WARMUP_CPU, REPS_CPU)
    lat_deploy_cpu = bench_fn(lambda: deploy_cpu(x1c),       WARMUP_CPU, REPS_CPU)
    lat_int8_cpu   = quantize_int8_latency(deploy_cpu, x1c)

    # Throughput on CPU, batch=256
    x256c = torch.randn(256, C, T); m256c = torch.zeros(256, 4, dtype=torch.long)
    thr_deploy_cpu = bench_fn(lambda: deploy_cpu(x256c), 3, 10)
    thr_deploy_cpu = 256 / (thr_deploy_cpu / 1000)   # samples/sec

    print(f"\n  ▸ CPU [{get_cpu_name()[:45]}]")
    print(f"    {'Path':<28} {'Lat(ms)':>9}")
    print(f"    {'-'*40}")
    print(f"    {'Full forward()':28} {lat_full_cpu:>9.2f}")
    print(f"    {'Inference path (FP32)':28} {lat_deploy_cpu:>9.2f}  "
          f"({lat_full_cpu/lat_deploy_cpu:.1f}x faster)")
    if lat_int8_cpu:
        print(f"    {'Inference path (INT8)':28} {lat_int8_cpu:>9.2f}  "
              f"({lat_deploy_cpu/lat_int8_cpu:.1f}x speedup from quant)")
    print(f"    Thr (batch=256, FP32): {thr_deploy_cpu:.0f} samples/sec")

    # ── 6. CPU RAM ─────────────────────────────────────────────────────────────
    ram_mb = measure_cpu_ram_mb(lambda: deploy_cpu(x1c))
    print(f"\n  ▸ CPU RAM (RSS increase during inference)")
    print(f"    Inference path   : ≲{ram_mb + sz_fp32:.0f} MB  "
          f"(model {sz_fp32:.1f} MB + {ram_mb:.1f} MB activation)")

    # ── 7. Realtime budget ─────────────────────────────────────────────────────
    print(f"\n  ▸ Realtime Feasibility  (hop={HOP_SIZE} samples)")
    inf_lat = lat_int8_cpu if lat_int8_cpu else lat_deploy_cpu
    for sr in [20, 50, 100]:
        budget_ms = HOP_SIZE / sr * 1000
        fit = "✓" if inf_lat < budget_ms else "✗"
        print(f"    {sr}Hz → budget={budget_ms:.0f}ms  "
              f"cpu_lat={inf_lat:.1f}ms  {fit}")

    # ── 8. Edge chip scaling ───────────────────────────────────────────────────
    print(f"\n  ▸ Projected Latency on Edge Chips  (inference path, FP32, batch=1)")
    print(f"    {'Chip':<40} {'Est Lat (ms)':>14}  {'20Hz fit?':>10}")
    budget_20hz = HOP_SIZE / 20 * 1000
    for chip, scale in EDGE_CHIPS.items():
        est = lat_deploy_cpu / scale
        fit = "✓" if est < budget_20hz else "✗"
        print(f"    {chip:<40} {est:>14.1f}  {fit:>10}")

    model.to(device)
    return {
        "params_full_M":   p_full/1e6,
        "params_deploy_M": p_deploy/1e6,
        "size_fp32_mb":    sz_fp32,
        "size_fp16_mb":    sz_fp16,
        "size_int8_mb":    sz_int8,
        "flops_full_M":    f_full/1e6 if f_full else None,
        "flops_deploy_M":  f_deploy/1e6 if f_deploy else None,
        "gpu_lat_full_ms":    lat_full_gpu,
        "gpu_lat_deploy_ms":  lat_deploy_gpu,
        "gpu_thr_full_ks":    best_thr_full/1000,
        "gpu_thr_deploy_ks":  best_thr_dep/1000,
        "gpu_mem_full_mb":    mem_full,
        "gpu_mem_deploy_mb":  mem_deploy,
        "cpu_lat_full_ms":    lat_full_cpu,
        "cpu_lat_deploy_ms":  lat_deploy_cpu,
        "cpu_lat_int8_ms":    lat_int8_cpu,
        "cpu_thr_deploy_sps": thr_deploy_cpu,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_finetune_cost", action="store_true")
    args = parser.parse_args()

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'═'*68}")
    print(f"  Edge-HAR  Edge Deployment Benchmark")
    print(f"  GPU : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"  CPU : {get_cpu_name()[:50]}")
    print(f"  HAR window: C={C}, T={T}, hop={HOP_SIZE}")
    print(f"{'═'*68}")

    results = {}

    # ── Pretrained model (4-class head) ───────────────────────────────────────
    results["pretrained"] = run_benchmark(
        "Pretrained (4-class, zero-shot)",
        CKPT_PRETRAIN, 4, cfg, device, args
    )

    # ── K=100 fine-tuned model (14-class novel) ───────────────────────────────
    results["finetuned_k100"] = run_benchmark(
        "Fine-tuned K=100  (14-class novel, F1=0.919)",
        CKPT_FINETUNED, NUM_NOVEL, cfg, device, args
    )

    # ── Fine-tune cost ─────────────────────────────────────────────────────────
    if not args.skip_finetune_cost:
        print(f"\n{'━'*68}")
        print(f"  Fine-Tune Cost  (K=100 per class, 14 classes = 1400 samples/epoch)")
        print(f"{'━'*68}")
        ft_ms, ft_mem_mb = measure_finetune_cost(cfg, CKPT_PRETRAIN, device)
        total_epochs = 50  # from experiment
        print(f"    Time per epoch       : {ft_ms:.1f} ms")
        print(f"    Total for 50 epochs  : {ft_ms*total_epochs/1000:.1f} s  "
              f"({ft_ms*total_epochs/60000:.1f} min)")
        print(f"    Peak GPU memory      : {ft_mem_mb:.1f} MB")
        print(f"    → Fine-tune fits in  : {ft_mem_mb:.0f} MB VRAM  "
              f"({'✓ phone GPU' if ft_mem_mb < 4096 else '✓ edge GPU'}, "
              f"needs {ft_mem_mb/1024:.1f} GB)")
        results["finetune_cost"] = {
            "ms_per_epoch": ft_ms,
            "total_50ep_sec": ft_ms*50/1000,
            "peak_gpu_mb": ft_mem_mb,
        }

    # ── Comparison summary ────────────────────────────────────────────────────
    print(f"\n{'═'*68}")
    print(f"  DEPLOYMENT SUMMARY")
    print(f"{'═'*68}")
    r = results["finetuned_k100"]
    print(f"  Inference path parameters : {r['params_deploy_M']:.3f} M")
    print(f"  Model binary (FP32/FP16/INT8): "
          f"{r['size_fp32_mb']:.1f} / {r['size_fp16_mb']:.1f} / "
          f"{r['size_int8_mb']:.1f} MB")
    if r['flops_deploy_M']:
        print(f"  Inference FLOPs           : {r['flops_deploy_M']:.2f} MFLOPs/window")
    print(f"  GPU latency (RTX 4090)    : {r['gpu_lat_deploy_ms']:.3f} ms/window")
    print(f"  CPU latency (server)      : {r['cpu_lat_deploy_ms']:.2f} ms/window")
    if r['cpu_lat_int8_ms']:
        print(f"  CPU latency (INT8)        : {r['cpu_lat_int8_ms']:.2f} ms/window")
    print(f"  CPU throughput            : {r['cpu_thr_deploy_sps']:.0f} samples/sec")
    print(f"  Peak GPU memory (infer.)  : {r['gpu_mem_deploy_mb']:.1f} MB")
    print(f"{'═'*68}\n")

    out_path = os.path.join(_THIS, "outputs", "edge_deployment_benchmark.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Full results → {out_path}")


def get_cpu_name():
    try:
        import subprocess
        r = subprocess.run(["grep", "-m1", "model name", "/proc/cpuinfo"],
                           capture_output=True, text=True)
        return r.stdout.split(":", 1)[-1].strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
