"""
VisionServe comprehensive benchmark — covers all 16 models via Go HTTP baseline.

Each model is benchmarked in an isolated subprocess with its own server port.
A ThreadPoolExecutor controls parallelism (1 worker for GPU, 4 for CPU).

Usage:
    python3 benchmarks/bench_all_models.py [--device gpu|cpu|both]
                                           [--models rf-detr mobile-sam ...]
                                           [--workers N]
                                           [--runs N]
                                           [--out /path/to/out.json]

Results are saved to benchmarks/results/bench_all_models_{device}_{timestamp}.json
"""

import argparse
import concurrent.futures
import glob
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

# ─── paths ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
BIN = str(ROOT / "bin" / "visionserve")
MODELS_DIR = str(ROOT / "models")
IMG = "/tmp/cats.jpg"
OUT_DIR = ROOT / "benchmarks" / "results"

# Machine-specific ORT library paths (same convention as bench.py)
_GPU_ORT = (
    "/home/trung/trung_workdir/vision_forge/frontend/node_modules/"
    "onnxruntime-node/bin/napi-v6/linux/x64/libonnxruntime.so.1"
)
_CPU_ORT = (
    "/home/trung/miniconda3/envs/label/lib/python3.12/site-packages/"
    "onnxruntime/capi/libonnxruntime.so.1.26.0"
)
_GPU_ORT_DIR = (
    "/home/trung/trung_workdir/vision_forge/frontend/node_modules/"
    "onnxruntime-node/bin/napi-v6/linux/x64"
)

# ─── benchmark constants ──────────────────────────────────────────────────────

N_WARM = 3
N_RUNS = 20
BASE_PORT = 11440      # first model uses this port; each model gets BASE_PORT + index
GPU_WORKERS = 1        # sequential: VRAM isolation
CPU_WORKERS = 4        # parallel: CPU models safe to run concurrently

# ─── model list ──────────────────────────────────────────────────────────────
# (name, task, prompt_box, prompt_text)
MODELS = [
    ("rf-detr",           "detection",      None,            None),
    ("rf-detr-nano",      "detection",      None,            None),
    ("rt-detr",           "detection",      None,            None),
    ("scrfd",             "detection",      None,            None),
    ("mobile-sam",        "segmentation",   "100,100,200,200", None),
    ("efficient-sam",     "segmentation",   "100,100,200,200", None),
    ("sam2",              "segmentation",   "100,100,200,200", None),
    ("nano-sam",          "segmentation",   "100,100,200,200", None),
    ("grounding-dino",    "open_vocab",     None,            "cat. remote."),
    ("grounded-sam",      "open_vocab",     None,            "cat. remote."),
    ("depth-anything-v2", "depth",          None,            None),
    ("midas",             "depth",          None,            None),
    ("efficientnet-b0",   "classification", None,            None),
    ("mobilenet-v3",      "classification", None,            None),
    ("clip",              "embed",          None,            None),
    ("paddle-ocr",        "ocr",            None,            None),
]

# ─── thread-safe print ────────────────────────────────────────────────────────

_print_lock = threading.Lock()


def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs, flush=True)


# ─── helpers ─────────────────────────────────────────────────────────────────

def ensure_image():
    if not os.path.exists(IMG) or os.path.getsize(IMG) < 10_000:
        subprocess.run(
            ["curl", "-sL",
             "http://images.cocodataset.org/val2017/000000039769.jpg",
             "-o", IMG],
            check=True,
        )


def p(times):
    s = sorted(times)
    n = len(s)
    return dict(
        p50=round(statistics.median(s), 1),
        p95=round(s[int(n * 0.95)], 1),
        p99=round(s[min(int(n * 0.99), n - 1)], 1),
        mean=round(statistics.mean(s), 1),
        rps=round(1000 / statistics.mean(s), 2),
        n=n,
    )


def rss(pid=None):
    try:
        import psutil
        return psutil.Process(pid).memory_info().rss // (1024 * 1024)
    except Exception:
        return None


def vram(pid=None):
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.strip().splitlines():
            p2, m = line.split(",")
            if pid is None or int(p2.strip()) == pid:
                return int(m.strip().split()[0])
    except Exception:
        return None
    return None


def gpu_ort():
    return _GPU_ORT


def cpu_ort():
    return _CPU_ORT


def gpu_ld():
    """LD_LIBRARY_PATH for CUDA EP (cuDNN from label env wheels + node ORT)."""
    nv = ":".join(
        glob.glob(
            "/home/trung/miniconda3/envs/label/lib/python3.12/site-packages/nvidia/*/lib"
        )
    )
    return f"{_GPU_ORT_DIR}:{nv}"


def hw():
    try:
        g = subprocess.check_output(["nvidia-smi", "-L"], text=True).strip().splitlines()[0]
    except Exception:
        g = "N/A"
    try:
        cores = int(subprocess.check_output(["nproc"]).strip())
    except Exception:
        cores = None
    try:
        ram = int(
            subprocess.check_output(
                "free -m|awk '/Mem/{print $2}'", shell=True
            ).strip()
        ) // 1024
    except Exception:
        ram = None
    return dict(gpu=g, cpu_cores=cores, ram_gb=ram)


def health_wait(url, timeout=90):
    for _ in range(timeout * 2):
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def predict_http(url, model, img_path, prompt=None, box=None, timeout=300):
    """POST multipart/form-data to /api/predict.

    - ``prompt`` is sent as the ``prompt`` field (text tasks).
    - ``box`` is sent as the ``box`` field (segmentation tasks).
    """
    boundary = "BENCHBND"
    body = b""
    fields = [("model", model)]
    if prompt is not None:
        fields.append(("prompt", prompt))
    if box is not None:
        fields.append(("box", box))
    for name, val in fields:
        body += (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"{name}\"\r\n\r\n"
            f"{val}\r\n"
        ).encode()
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    body += (
        f"--{boundary}\r\n"
        "Content-Disposition: form-data; name=\"image\"; filename=\"i.jpg\"\r\n"
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode() + img_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def model_size_mb(model_name):
    """Sum of all .onnx file sizes in the model directory, in MB (1 decimal)."""
    model_dir = Path(MODELS_DIR) / model_name
    total = sum(f.stat().st_size for f in model_dir.glob("*.onnx"))
    return round(total / (1024 * 1024), 1)


def has_onnx(model_name):
    model_dir = Path(MODELS_DIR) / model_name
    return model_dir.is_dir() and any(model_dir.glob("*.onnx"))


def build_env(device):
    env = os.environ.copy()
    if device == "gpu":
        env["ORT_DYLIB_PATH"] = gpu_ort()
        env["LD_LIBRARY_PATH"] = gpu_ld()
    else:
        env["ORT_DYLIB_PATH"] = cpu_ort()
        env.pop("LD_LIBRARY_PATH", None)
    return env


# ─── per-model benchmark ─────────────────────────────────────────────────────

def bench_model(name, task, box, text, device, port, n_runs=N_RUNS):
    """Benchmark a single model on the given device.

    Returns a result dict with latency stats, or {"skipped": reason} /
    {"error": reason} on failure.
    """
    label = f"[{device}/{name}]"

    # 1. Check model directory + ONNX files
    if not has_onnx(name):
        tprint(f"{label} SKIPPED — no ONNX files")
        return {"model": name, "device": device, "skipped": "no ONNX files"}

    # 2. Compute model size
    size_mb = model_size_mb(name)

    # 3. Build env
    env = build_env(device)

    # 4. Launch server
    tprint(f"{label} starting (port {port})…")
    srv = subprocess.Popen(
        [BIN, "serve", "--models", MODELS_DIR, "--addr", f":{port}"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    health_url = f"http://localhost:{port}/api/health"
    predict_url = f"http://localhost:{port}/api/predict"

    def cleanup(err_msg):
        try:
            srv.terminate()
            srv.wait(5)
        except Exception:
            pass
        return {"model": name, "device": device, "model_size_mb": size_mb, "error": err_msg}

    # 5. Wait for health
    if not health_wait(health_url, timeout=90):
        return cleanup("server health timeout (90s)")

    # Determine prompt/box fields for this model
    req_kwargs = {}
    if text is not None:
        req_kwargs["prompt"] = text
    elif box is not None:
        req_kwargs["box"] = box

    # 6. Cold-start timing (includes model load on first predict)
    t0 = time.perf_counter()
    try:
        predict_http(predict_url, name, IMG, timeout=300, **req_kwargs)
    except Exception as e:
        return cleanup(f"cold start failed: {e}")
    cold_ms = round((time.perf_counter() - t0) * 1000, 1)

    # 7. Warm-up
    for _ in range(N_WARM):
        try:
            predict_http(predict_url, name, IMG, timeout=120, **req_kwargs)
        except Exception:
            pass

    # 8. Measure N_RUNS requests
    times = []
    srv_times = []
    errors = 0
    for _ in range(n_runs):
        t = time.perf_counter()
        try:
            r = predict_http(predict_url, name, IMG, timeout=120, **req_kwargs)
            times.append((time.perf_counter() - t) * 1000)
            if "duration_ms" in r:
                srv_times.append(r["duration_ms"])
        except Exception:
            errors += 1

    if not times:
        return cleanup(f"all {n_runs} runs failed ({errors} errors)")

    # 9. Memory snapshot
    rss_mb = rss(srv.pid)
    vram_mb = vram(srv.pid) if device == "gpu" else None

    # 10. Terminate server
    try:
        srv.terminate()
        srv.wait(5)
    except Exception:
        pass
    time.sleep(1)

    # 11. Build result
    stats = p(times)
    result = {
        "model": name,
        "device": device,
        "model_size_mb": size_mb,
        "cold_ms": cold_ms,
        **stats,
        "srv_p50": round(statistics.median(sorted(srv_times)), 1) if srv_times else None,
        "rss_mb": rss_mb,
        "vram_mb": vram_mb,
    }

    tprint(
        f"{label} p50={result['p50']}ms  p95={result['p95']}ms  "
        f"rps={result['rps']}  vram={vram_mb}MB  rss={rss_mb}MB"
    )
    return result


# ─── parallel runner ─────────────────────────────────────────────────────────

def run_device(device, model_subset, workers, n_runs):
    """Run all models for one device using a ThreadPoolExecutor.

    Each model gets a unique port (BASE_PORT + global model index) to avoid
    port conflicts when multiple workers run in parallel.
    """
    # Build a list of (port, name, task, box, text) for the requested subset
    jobs = []
    for idx, (name, task, box, text) in enumerate(MODELS):
        if model_subset and name not in model_subset:
            continue
        port = BASE_PORT + idx
        jobs.append((port, name, task, box, text))

    results = [None] * len(jobs)

    def _run(i, port, name, task, box, text):
        try:
            return bench_model(name, task, box, text, device, port, n_runs=n_runs)
        except Exception as e:
            tprint(f"[{device}/{name}] unexpected error: {e}")
            return {"model": name, "device": device, "error": str(e)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_run, i, port, name, task, box, text): i
            for i, (port, name, task, box, text) in enumerate(jobs)
        }
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                port, name, *_ = jobs[i]
                results[i] = {"model": name, "device": device, "error": str(e)}

    return results


# ─── summary table ────────────────────────────────────────────────────────────

def print_summary(device, results, hardware):
    gpu_str = hardware.get("gpu", "N/A")
    cores = hardware.get("cpu_cores", "?")
    ram = hardware.get("ram_gb", "?")

    tprint(f"\n=== VisionServe benchmark — {device.upper()} ===")
    tprint(f"Hardware: {gpu_str}, {cores} cores, {ram} GB RAM\n")

    header = (
        f"{'Model':<22} {'Task':<16} {'Size MB':>8}  "
        f"{'p50 ms':>7} {'p95 ms':>7} {'RPS':>6}  "
        f"{'VRAM MB':>8} {'RSS MB':>7}"
    )
    sep = "─" * len(header)
    tprint(header)
    tprint(sep)

    for r in results:
        if r is None:
            continue
        name = r.get("model", "?")
        dev  = r.get("device", "?")
        if "skipped" in r:
            tprint(f"[SKIPPED] {name:<14} — {r['skipped']}")
            continue
        if "error" in r:
            tprint(f"[ERROR]   {name:<14} — {r['error']}")
            continue

        # look up task from MODELS list
        task = next((t for n, t, *_ in MODELS if n == name), "?")
        size = r.get("model_size_mb", "?")
        p50  = r.get("p50", "?")
        p95  = r.get("p95", "?")
        rps  = r.get("rps", "?")
        vm   = r.get("vram_mb") if r.get("vram_mb") is not None else "—"
        rsm  = r.get("rss_mb", "?")

        tprint(
            f"{name:<22} {task:<16} {size:>8}  "
            f"{p50:>7} {p95:>7} {rps:>6}  "
            f"{str(vm):>8} {str(rsm):>7}"
        )
    tprint(sep)


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VisionServe comprehensive benchmark across all models."
    )
    parser.add_argument(
        "--device", choices=["gpu", "cpu", "both"], default="both",
        help="Which device(s) to benchmark (default: both)",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="NAME",
        help="Restrict to these model names (default: all)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Override worker count (default: 1 for GPU, 4 for CPU)",
    )
    parser.add_argument(
        "--runs", type=int, default=N_RUNS,
        help=f"Number of measurement runs per model (default: {N_RUNS})",
    )
    parser.add_argument(
        "--out", default=None, metavar="PATH",
        help="Output JSON path (default: benchmarks/results/bench_all_models_<device>_<ts>.json)",
    )
    args = parser.parse_args()

    ensure_image()
    hardware = hw()
    print(
        f"Hardware: {hardware['gpu']}, {hardware['cpu_cores']} cores, "
        f"{hardware['ram_gb']} GB RAM\n"
    )

    model_subset = set(args.models) if args.models else None

    devices = ["gpu", "cpu"] if args.device == "both" else [args.device]
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_output = {}

    for device in devices:
        workers = args.workers if args.workers is not None else (GPU_WORKERS if device == "gpu" else CPU_WORKERS)
        tprint(f"\n{'='*60}")
        tprint(f"Device: {device.upper()}   workers={workers}   runs={args.runs}")
        tprint(f"{'='*60}")

        results = run_device(device, model_subset, workers, n_runs=args.runs)

        print_summary(device, results, hardware)

        out_data = {
            "hardware": hardware,
            "device": device,
            "timestamp": timestamp,
            "n_runs": args.runs,
            "results": results,
        }
        all_output[device] = out_data

        if args.out:
            out_path = Path(args.out)
        else:
            out_path = OUT_DIR / f"bench_all_models_{device}_{timestamp}.json"

        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2)
        tprint(f"\nSaved → {out_path}")
