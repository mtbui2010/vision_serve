"""--bench: latency at three levels, each warm-up runs then N timed runs, p50 / p95 in ms.

  framework  the original model's forward (torch in-process) on the preprocessed tensor
  onnx       the exported ONNX alone (onnxruntime in-process, best provider this onnxruntime build has)
  server     one whole POST /api/predict over HTTP: upload, decode, Go preprocess, ONNX Runtime,
             postprocess, JSON — what a user of the server experiences

They measure DIFFERENT things on possibly DIFFERENT devices: the first two exclude image decoding
and pre/postprocessing; the server number includes them plus HTTP. Compare like with like. On a
shared GPU every number includes whatever else is running there (the GPU state before the run is
recorded); treat small differences as noise.
"""
from __future__ import annotations

import os
import subprocess
import time
from typing import Callable, List, Optional

from .common import log
from .constants import TEXT_ARCHS
from .metrics import percentile_ms
from .report import INFO, SKIP, TierResult, fmt

ORT_PREFERENCE = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "DmlExecutionProvider",
                  "OpenVINOExecutionProvider", "CPUExecutionProvider"]  # TensorRT left out: engine build time


def time_fn(fn: Callable[[], object], warmup: int = 3, iters: int = 50, clock=time.perf_counter) -> List[float]:
    for _ in range(max(0, warmup)):
        fn()
    out = []
    for _ in range(max(1, iters)):
        t0 = clock()
        fn()
        out.append((clock() - t0) * 1000.0)
    return out


def ort_providers(available: List[str]) -> List[str]:
    chosen = [p for p in ORT_PREFERENCE if p in available]
    return chosen or ["CPUExecutionProvider"]


def gpu_snapshot() -> Optional[str]:
    """`nvidia-smi` one-liner for the visible GPU(s), or None. Recorded, never interpreted."""
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if q.returncode != 0:
        return None
    lines = q.stdout.strip().splitlines()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis and all(v.strip().isdigit() for v in vis.split(",")):
        keep = {v.strip() for v in vis.split(",")}
        lines = [ln for ln in lines if ln.split(",")[0].strip() in keep]
    return "; ".join(lines) or None


def _device_name(dev: str) -> str:
    """'cuda' -> 'cuda:0 (NVIDIA RTX A6000, CUDA_VISIBLE_DEVICES=3)' so a number can be traced to a GPU."""
    if not str(dev).startswith("cuda"):
        return str(dev)
    try:
        import torch
        i = torch.device(dev).index or 0
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        return f"cuda:{i} ({torch.cuda.get_device_name(i)}" + (f", CUDA_VISIBLE_DEVICES={vis})" if vis else ")")
    except Exception:  # noqa: BLE001
        return str(dev)


def tier_speed(plan, client, bundle, image, prompt=None, warmup: int = 3, iters: int = 50) -> TierResult:
    name = bundle.name
    levels, notes = {}, []
    if bundle.architecture in TEXT_ARCHS:
        return TierResult("speed", "speed", SKIP, "text towers are not benchmarked", model=name)
    pre = client.preprocess(name, image.src, prompt=prompt)
    feeds = dict(pre.inputs)
    gpu_before = gpu_snapshot()

    # framework
    fw = plan.framework
    fn = fw.bench_fn(feeds) if fw is not None else None
    if fn is not None:
        log(f"bench: {name}: framework forward on {fw.device} ...")
        levels["framework"] = {**percentile_ms(time_fn(fn, warmup, iters)), "device": _device_name(fw.device),
                               "what": f"{fw.framework} forward of the original model (no pre/postprocess)"}
    else:
        levels["framework"] = {"skipped": "no in-process framework model for this format"}

    # onnx
    import onnxruntime as ort
    providers = ort_providers(ort.get_available_providers())
    sess = ort.InferenceSession(plan.onnx_path, providers=providers)
    names = {i.name for i in sess.get_inputs()}
    of = {k: v for k, v in feeds.items() if k in names}
    used = sess.get_providers()[0]
    log(f"bench: {name}: onnxruntime {ort.__version__} with {used} ...")
    levels["onnx"] = {**percentile_ms(time_fn(lambda: sess.run(None, of), warmup, iters)), "provider": used,
                      "what": f"onnxruntime {ort.__version__} session.run (no pre/postprocess)"}
    del sess

    # server
    src = image.src
    payload = src.read_bytes() if hasattr(src, "read_bytes") else src
    last = {}

    def call():
        last["r"] = client.predict(name, payload, prompt=prompt, max_grasps_per_object=None)
    log(f"bench: {name}: server /api/predict ...")
    ms = time_fn(call, warmup, iters)
    r = last.get("r")
    levels["server"] = {**percentile_ms(ms), "device": (r.device if r is not None else "") or "unreported",
                        "server_duration_ms_last": (r.duration_ms if r is not None else None),
                        "what": "POST /api/predict over HTTP (decode + preprocess + inference + postprocess + JSON)",
                        "image": image.label}

    parts = []
    for k in ("framework", "onnx", "server"):
        lv = levels[k]
        if "p50_ms" in lv:
            where = lv.get("device") or lv.get("provider")
            parts.append(f"{k} {fmt(lv['p50_ms'])}/{fmt(lv['p95_ms'])} ms ({str(where).split(' (')[0]})")
    notes.append("p50/p95 over " + f"{iters} runs after {warmup} warm-up; levels measure different things "
                 "(framework/onnx = model forward only; server = whole HTTP request) and may run on different "
                 "devices")
    if gpu_before:
        notes.append(f"GPU state before the run (index, util, mem used, total): {gpu_before} — a shared GPU "
                     "inflates and jitters every number")
    return TierResult("speed", "latency p50/p95", INFO, "; ".join(parts), model=name,
                      metrics={"levels": levels, "warmup": warmup, "iters": iters, "gpu_before": gpu_before},
                      notes=notes)
