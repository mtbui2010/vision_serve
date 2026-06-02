#!/usr/bin/env python3
"""
benchmarks/bench_fastapi_ort.py — Benchmark FastAPI + ONNX Runtime HTTP server.

Measures cold start, warm latency p50/p95/p99/mean, throughput, RSS, VRAM
for each (model, device) combination, then computes HTTP overhead vs
python_ort_direct baseline.

Output: benchmarks/results/fastapi_ort.json
"""

import glob
import io
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import threading
import urllib.request
import urllib.error
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results"

PY          = "/home/trung/miniconda3/envs/label/bin/python3"
SERVER_PORT = 11436
SERVER_URL  = f"http://localhost:{SERVER_PORT}"
TEST_IMAGE  = "/tmp/cats.jpg"
CATS_URL    = "http://images.cocodataset.org/val2017/000000039769.jpg"
RESULTS_FILE = RESULTS_DIR / "fastapi_ort.json"

NV_BASE = "/home/trung/miniconda3/envs/label/lib/python3.12/site-packages/nvidia"
ORT_GPU_DIR = (
    "/home/trung/trung_workdir/vision_forge/frontend"
    "/node_modules/onnxruntime-node/bin/napi-v6/linux/x64"
)

WARMUP_REQS  = 5
MEASURE_REQS = 30
SERVER_START_TIMEOUT = 120  # seconds

# (model_name, prompt)
MODELS = [
    ("rf-detr",        None),
    ("grounding-dino", "cat. remote."),
    ("grounded-sam",   "cat. remote."),
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def log(msg: str):
    print(f"[bench-fastapi] {msg}", flush=True)


def ensure_image():
    if not os.path.exists(TEST_IMAGE) or os.path.getsize(TEST_IMAGE) < 10_000:
        log(f"Downloading test image → {TEST_IMAGE}")
        subprocess.run(["curl", "-sL", CATS_URL, "-o", TEST_IMAGE], check=True)
    log(f"Test image OK: {TEST_IMAGE} ({os.path.getsize(TEST_IMAGE)//1024} KB)")


def hardware_info() -> dict:
    gpu = None
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            gpu = r.stdout.strip().splitlines()[0]
    except Exception:
        pass
    try:
        cores = int(subprocess.check_output(["nproc"], timeout=5).strip())
    except Exception:
        cores = None
    try:
        raw = subprocess.check_output("free -m | awk '/Mem/{print $2}'",
                                      shell=True, timeout=5).strip()
        ram_gb = int(raw) // 1024
    except Exception:
        ram_gb = None
    return {"gpu": gpu, "cpu_cores": cores, "ram_gb": ram_gb}


def get_rss_mb(pid: int) -> float | None:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except Exception:
        pass
    return None


def get_vram_mb(pid: int) -> int | None:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[0] == str(pid):
                    import re
                    m = re.match(r"(\d+)", parts[1])
                    if m:
                        return int(m.group(1))
    except Exception:
        pass
    return None


def compute_percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (p / 100.0) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return round(s[lo] + (idx - lo) * (s[hi] - s[lo]), 2)


def build_env(device: str) -> dict:
    env = os.environ.copy()
    if device == "gpu":
        # Add NVIDIA cuda wheel libs so ORT CUDA EP can find libcudnn.so.9 etc.
        libs = ":".join(glob.glob(f"{NV_BASE}/*/lib"))
        env["LD_LIBRARY_PATH"] = libs + ":" + env.get("LD_LIBRARY_PATH", "")
        env["USE_CUDA"] = "1"
        log(f"GPU LD_LIBRARY_PATH prefix: {libs[:80]}...")
    else:
        env.pop("USE_CUDA", None)
    return env


# ── HTTP client helpers ────────────────────────────────────────────────────────

def _multipart_body(model: str, img_path: str, prompt: str | None):
    """Build raw multipart/form-data bytes. Returns (body_bytes, content_type_header)."""
    boundary = "BENCH_BOUNDARY_7F3A"
    parts = b""
    for field, val in [("model", model)] + ([("prompt", prompt)] if prompt else []):
        parts += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field}"\r\n\r\n'
            f"{val}\r\n"
        ).encode()
    with open(img_path, "rb") as f:
        img_data = f.read()
    parts += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="img.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode()
    parts += img_data + f"\r\n--{boundary}--\r\n".encode()
    ctype = f"multipart/form-data; boundary={boundary}"
    return parts, ctype


def send_predict(model: str, img_path: str, prompt: str | None, timeout: int = 300):
    """
    Send POST /predict. Returns (wall_ms, duration_ms_from_json, error_str).
    """
    body, ctype = _multipart_body(model, img_path, prompt)
    req = urllib.request.Request(
        f"{SERVER_URL}/predict",
        data=body,
        headers={"Content-Type": ctype},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        data = json.loads(raw)
        duration_ms = data.get("duration_ms")
        return wall_ms, duration_ms, None
    except Exception as e:
        return None, None, str(e)


def wait_for_health(timeout: int = SERVER_START_TIMEOUT) -> bool:
    """Poll GET /health until 200 or timeout. Returns True if healthy."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"{SERVER_URL}/health", timeout=2
            ) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


# ── Python ORT direct benchmark (for overhead calculation) ────────────────────

PYTHON_ORT_DIRECT_SCRIPT = r"""
import time, json, statistics, sys, os
import numpy as np
import onnxruntime as ort
from PIL import Image

model_name = sys.argv[1]
device     = sys.argv[2]
n_runs     = int(sys.argv[3]) if len(sys.argv) > 3 else 30

ROOT = sys.argv[4] if len(sys.argv) > 4 else "."
providers = (["CUDAExecutionProvider","CPUExecutionProvider"]
             if device == "gpu" else ["CPUExecutionProvider"])

MEAN = np.array([0.485,0.456,0.406],dtype=np.float32)
STD  = np.array([0.229,0.224,0.225],dtype=np.float32)

IMG = "/tmp/cats.jpg"
img = Image.open(IMG).convert("RGB")

def preprocess_rfdetr(img):
    x = np.array(img.resize((560,560),Image.BILINEAR),dtype=np.float32)/255.0
    return ((x-MEAN)/STD).transpose(2,0,1)[None].astype(np.float32)

def preprocess_gdino(img, prompt="cat. remote."):
    x = np.array(img.resize((800,800),Image.BILINEAR),dtype=np.float32)/255.0
    x = ((x-MEAN)/STD).transpose(2,0,1)[None].astype(np.float32)
    pm = np.ones((1,800,800),dtype=np.int64)
    ids = np.array([[101,4937,1012,6556,1012,102]],dtype=np.int64)
    msk = np.ones_like(ids); typ = np.zeros_like(ids)
    return x, pm, ids, msk, typ

if model_name == "rf-detr":
    mp = os.path.join(ROOT,"models/rf-detr/rf-detr-base-real.onnx")
    t0 = time.perf_counter()
    sess = ort.InferenceSession(mp, providers=providers)
    inp = preprocess_rfdetr(img)
    _ = sess.run(None, {"input": inp})
    cold_ms = (time.perf_counter()-t0)*1000
    for _ in range(5): sess.run(None,{"input":inp})
    times=[]
    for _ in range(n_runs):
        t=time.perf_counter(); sess.run(None,{"input":inp}); times.append((time.perf_counter()-t)*1000)

elif model_name == "grounding-dino":
    mp = os.path.join(ROOT,"models/grounding-dino/model.onnx")
    t0 = time.perf_counter()
    sess = ort.InferenceSession(mp, providers=providers)
    pv,pm,ids,msk,typ = preprocess_gdino(img)
    def run_gdino():
        return sess.run(None,{"pixel_values":pv,"pixel_mask":pm,"input_ids":ids,"attention_mask":msk,"token_type_ids":typ})
    _ = run_gdino()
    cold_ms = (time.perf_counter()-t0)*1000
    for _ in range(5): run_gdino()
    times=[]
    for _ in range(n_runs):
        t=time.perf_counter(); run_gdino(); times.append((time.perf_counter()-t)*1000)

elif model_name == "grounded-sam":
    enc_path = os.path.join(ROOT,"models/mobile-sam/mobile_sam_encoder.onnx")
    dec_path = os.path.join(ROOT,"models/mobile-sam/mobile_sam_decoder_single.onnx")
    gdino_path = os.path.join(ROOT,"models/grounding-dino/model.onnx")
    t0 = time.perf_counter()
    gdino_sess = ort.InferenceSession(gdino_path, providers=providers)
    enc_sess   = ort.InferenceSession(enc_path,   providers=providers)
    dec_sess   = ort.InferenceSession(dec_path,   providers=providers)
    # preprocess once
    pv,pm,ids,msk,typ = preprocess_gdino(img)
    orig_w, orig_h = img.size
    scale = 1024.0/max(orig_h,orig_w)
    new_w=int(orig_w*scale+0.5); new_h=int(orig_h*scale+0.5)
    enc_in = np.asarray(img.resize((new_w,new_h),Image.BILINEAR),dtype=np.float32)
    orig_size = np.array([orig_h,orig_w],dtype=np.float32)
    mask_input = np.zeros((1,1,256,256),dtype=np.float32)
    has_mask = np.zeros(1,dtype=np.float32)
    box_pt = np.array([[[10,60],[320,437]]],dtype=np.float32)*scale
    pt_labels = np.array([[2,3]],dtype=np.float32)
    def run_pipeline():
        gdino_sess.run(None,{"pixel_values":pv,"pixel_mask":pm,"input_ids":ids,"attention_mask":msk,"token_type_ids":typ})
        emb = enc_sess.run(None,{"input_image":enc_in})[0]
        dec_sess.run(None,{"image_embeddings":emb,"point_coords":box_pt,"point_labels":pt_labels,"mask_input":mask_input,"has_mask_input":has_mask,"orig_im_size":orig_size})
    run_pipeline()
    cold_ms=(time.perf_counter()-t0)*1000
    for _ in range(5): run_pipeline()
    times=[]
    for _ in range(n_runs):
        t=time.perf_counter(); run_pipeline(); times.append((time.perf_counter()-t)*1000)
else:
    print(json.dumps({"error": f"unknown model {model_name}"}))
    sys.exit(0)

times.sort(); n=len(times)
rss=None
try:
    import psutil; rss=psutil.Process().memory_info().rss//(1024*1024)
except: pass
res={
    "cold_start_ms": round(cold_ms,1),
    "p50_ms":  round(statistics.median(times),1),
    "p95_ms":  round(times[int(n*0.95)],1),
    "p99_ms":  round(times[min(int(n*0.99),n-1)],1),
    "mean_ms": round(statistics.mean(times),1),
    "throughput_rps": round(1000/statistics.mean(times),2),
    "rss_mb":  rss,
    "n_runs":  n,
}
print(json.dumps(res))
"""


def bench_python_ort_direct(device: str, quick: bool = False) -> dict:
    """Run Python ORT direct (no HTTP) as baseline for overhead calculation."""
    n = 10 if quick else MEASURE_REQS
    script_path = "/tmp/_bench_ort_direct.py"
    with open(script_path, "w") as f:
        f.write(PYTHON_ORT_DIRECT_SCRIPT)

    env = build_env(device)
    results = {}
    for model, _ in MODELS:
        log(f"  python_ort_direct {model} {device}...")
        try:
            r = subprocess.run(
                [PY, script_path, model, device, str(n), str(ROOT)],
                capture_output=True, text=True, timeout=600,
                cwd=str(ROOT), env=env,
            )
            stdout = r.stdout.strip()
            if stdout:
                last_line = [l for l in stdout.splitlines() if l.strip().startswith("{")]
                if last_line:
                    results[model] = json.loads(last_line[-1])
                else:
                    results[model] = {"error": f"no JSON in output: {stdout[:200]}"}
            else:
                results[model] = {"error": f"no output (stderr: {r.stderr[:200]})"}
        except subprocess.TimeoutExpired:
            results[model] = {"error": "timeout"}
        except Exception as e:
            results[model] = {"error": str(e)}
    return results


# ── FastAPI server benchmark ───────────────────────────────────────────────────

def bench_fastapi(model: str, prompt: str | None, device: str) -> dict:
    """
    Benchmark one (model, device) combination.
    Starts the FastAPI server, measures, stops it.
    Returns dict with all metrics.
    """
    log(f"\n{'='*60}")
    log(f"FastAPI bench  model={model}  device={device}")
    log(f"{'='*60}")

    env = build_env(device)

    # Path to the fastapi_server module (benchmarks package in ROOT)
    # Run as: python3 -m uvicorn benchmarks.fastapi_server:app --port 11436
    cmd = [
        PY, "-m", "uvicorn",
        "benchmarks.fastapi_server:app",
        "--port", str(SERVER_PORT),
        "--workers", "1",
        "--log-level", "warning",
        "--no-access-log",
    ]
    log(f"Starting: {' '.join(cmd)}")

    result = {
        "cold_start_ms": None,
        "latency_p50_ms": None,
        "latency_p95_ms": None,
        "latency_p99_ms": None,
        "latency_mean_ms": None,
        "throughput_rps": None,
        "server_inference_p50_ms": None,
        "rss_mb": None,
        "vram_mb": None,
        "n_runs": MEASURE_REQS,
        "_error": None,
    }

    t_launch = time.perf_counter()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as e:
        result["_error"] = f"Failed to start server: {e}"
        log(f"ERROR: {result['_error']}")
        return result

    pid = proc.pid
    log(f"Server PID={pid}")

    # Wait for health endpoint
    health_ok = wait_for_health(timeout=60)
    if not health_ok:
        proc.kill()
        proc.wait(5)
        result["_error"] = "Server did not become healthy within 60s"
        log(f"ERROR: {result['_error']}")
        return result

    t_http_up = time.perf_counter()
    log(f"HTTP /health up in {(t_http_up - t_launch)*1000:.0f} ms")

    # Cold start: first predict request (triggers model load)
    log("Sending cold-start request (triggers lazy model load)...")
    t_cold_start = time.perf_counter()
    cold_wall, cold_dur, cold_err = send_predict(model, TEST_IMAGE, prompt, timeout=300)
    t_cold_end = time.perf_counter()

    if cold_err:
        proc.kill()
        proc.wait(5)
        result["_error"] = f"Cold-start request failed: {cold_err}"
        log(f"ERROR: {result['_error']}")
        return result

    cold_start_ms = (t_cold_end - t_launch) * 1000.0
    result["cold_start_ms"] = round(cold_start_ms, 1)
    log(f"Cold start: {cold_start_ms:.0f} ms  (wall_ms={cold_wall:.1f}, server={cold_dur} ms)")

    # Warmup
    log(f"Warmup ({WARMUP_REQS} requests)...")
    for i in range(WARMUP_REQS):
        w, d, err = send_predict(model, TEST_IMAGE, prompt)
        if err:
            log(f"  warmup {i+1} FAILED: {err}")
        else:
            log(f"  warmup {i+1}: wall={w:.1f} ms  server={d} ms")

    # Memory snapshot (post-warmup, model loaded)
    rss  = get_rss_mb(pid)
    vram = get_vram_mb(pid)
    result["rss_mb"]  = rss
    result["vram_mb"] = vram
    log(f"Memory: RSS={rss} MB  VRAM={vram} MB")

    # Timed measurement
    log(f"Timed measurement ({MEASURE_REQS} requests)...")
    wall_times:   list[float] = []
    server_times: list[float] = []
    t_batch = time.perf_counter()

    for i in range(MEASURE_REQS):
        w, d, err = send_predict(model, TEST_IMAGE, prompt)
        if err:
            log(f"  req {i+1:02d} FAILED: {err}")
            continue
        wall_times.append(w)
        if d is not None:
            server_times.append(float(d))
        log(f"  req {i+1:02d}/{MEASURE_REQS}: wall={w:.1f} ms  server={d} ms")

    elapsed = time.perf_counter() - t_batch

    if not wall_times:
        proc.kill()
        proc.wait(5)
        result["_error"] = "All measurement requests failed"
        return result

    result["latency_p50_ms"]  = compute_percentile(wall_times, 50)
    result["latency_p95_ms"]  = compute_percentile(wall_times, 95)
    result["latency_p99_ms"]  = compute_percentile(wall_times, 99)
    result["latency_mean_ms"] = round(statistics.mean(wall_times), 2)
    result["throughput_rps"]  = round(len(wall_times) / elapsed, 3)
    if server_times:
        result["server_inference_p50_ms"] = compute_percentile(server_times, 50)

    log(f"p50={result['latency_p50_ms']} p95={result['latency_p95_ms']} "
        f"p99={result['latency_p99_ms']} mean={result['latency_mean_ms']} ms")
    log(f"Throughput: {result['throughput_rps']} req/s")
    log(f"Server-side inference p50: {result['server_inference_p50_ms']} ms")

    # Shutdown
    log(f"Stopping server PID={pid}...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(3)
    log("Server stopped.")
    time.sleep(1)  # brief pause to release port

    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log("=== FastAPI + ONNX Runtime Benchmark ===")
    log(f"Server port : {SERVER_PORT}")
    log(f"Results     : {RESULTS_FILE}")
    log(f"Warmup/Runs : {WARMUP_REQS}/{MEASURE_REQS}")

    ensure_image()
    hw = hardware_info()
    log(f"Hardware: {hw}")

    # ── Step 1: Collect python_ort_direct baseline ────────────────────────────
    log("\n[1/3] Python ORT direct baseline (no HTTP)")
    ort_direct: dict[str, dict[str, dict]] = {}
    for device in ("cpu", "gpu"):
        log(f"  device={device}")
        ort_direct[device] = bench_python_ort_direct(device)

    # ── Step 2: Benchmark FastAPI server ─────────────────────────────────────
    log("\n[2/3] FastAPI + ORT HTTP server benchmark")
    fastapi_results: dict[str, dict[str, dict]] = {}
    notes: list[str] = []

    for model, prompt in MODELS:
        fastapi_results[model] = {}
        for device in ("cpu", "gpu"):
            metrics = bench_fastapi(model, prompt, device)
            err = metrics.pop("_error", None)
            if err:
                notes.append(f"{model}/{device}: {err}")
                log(f"NOTE: {model}/{device}: {err}")
            fastapi_results[model][device] = metrics

    # ── Step 3: Compute HTTP overhead vs python_ort_direct ────────────────────
    log("\n[3/3] Computing HTTP overhead (FastAPI p50 − ORT direct p50)")
    http_overhead: dict[str, dict] = {}
    for model, _ in MODELS:
        http_overhead[model] = {}
        for device in ("cpu", "gpu"):
            fa_metrics  = fastapi_results.get(model, {}).get(device, {})
            ort_metrics = ort_direct.get(device, {}).get(model, {})
            fa_p50   = fa_metrics.get("latency_p50_ms")
            ort_p50  = ort_metrics.get("p50_ms")
            # server_inference_p50 is the FastAPI server-side only (pre+infer+post, no HTTP)
            fa_srv50 = fa_metrics.get("server_inference_p50_ms")
            overhead = None
            if fa_p50 is not None and ort_p50 is not None:
                overhead = round(fa_p50 - ort_p50, 2)
            # framework_cost = fa_server_p50 - ort_p50
            # (pure FastAPI/uvicorn parsing overhead, same processing path)
            fw_cost = None
            if fa_srv50 is not None and ort_p50 is not None:
                fw_cost = round(fa_srv50 - ort_p50, 2)
            http_overhead[model][device] = {
                "fastapi_p50_ms":         fa_p50,
                "python_ort_p50_ms":      ort_p50,
                "http_overhead_ms":       overhead,
                "fastapi_server_only_p50_ms": fa_srv50,
                "framework_cost_ms":      fw_cost,
                "note": (
                    "http_overhead_ms = total wall (incl. HTTP) - raw ORT; "
                    "framework_cost_ms = server-side only - raw ORT"
                ),
            }
            log(f"  {model}/{device}: http_overhead={overhead} ms  "
                f"framework_cost={fw_cost} ms")

    # ── Build output JSON ─────────────────────────────────────────────────────
    output = {
        "baseline": "python_fastapi_ort_http",
        "hardware": hw,
        "models": fastapi_results,
        "python_ort_direct": ort_direct,
        "http_overhead": http_overhead,
        "notes": "; ".join(notes) if notes else "All benchmarks completed successfully.",
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nResults written → {RESULTS_FILE}")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'Model':<18} {'Dev':<5} {'p50(wall)':<12} {'p50(srv)':<12} "
          f"{'ORT p50':<12} {'HTTP ovhd':<12} {'tput rps':<10}")
    print("-" * 90)
    for model, _ in MODELS:
        for device in ("cpu", "gpu"):
            fa  = fastapi_results.get(model, {}).get(device, {})
            ov  = http_overhead.get(model, {}).get(device, {})
            p50_wall = fa.get("latency_p50_ms",          "N/A")
            p50_srv  = fa.get("server_inference_p50_ms", "N/A")
            ort_p50  = ov.get("python_ort_p50_ms",       "N/A")
            ovhd     = ov.get("http_overhead_ms",         "N/A")
            tput     = fa.get("throughput_rps",           "N/A")
            print(f"{model:<18} {device:<5} {str(p50_wall):<12} {str(p50_srv):<12} "
                  f"{str(ort_p50):<12} {str(ovhd):<12} {str(tput):<10}")
    print("=" * 90)

    # ── FastAPI vs VisionServe overhead comparison ───────────────────────────
    vs_path = RESULTS_DIR / "visionserve.json"
    if vs_path.exists():
        try:
            with open(vs_path) as f:
                vs_data = json.load(f)
            print("\n── FastAPI vs VisionServe (Go) HTTP comparison ──────────────────────────────")
            print(f"{'Model':<18} {'Dev':<5} {'FastAPI p50':<14} {'Go p50':<14} {'Delta ms':<12}")
            print("-" * 65)
            for model, _ in MODELS:
                for device in ("cpu", "gpu"):
                    fa_p50 = fastapi_results.get(model, {}).get(device, {}).get("latency_p50_ms")
                    vs_p50 = (vs_data.get("models", {}).get(model, {})
                              .get(device, {}).get("latency_p50_ms"))
                    if fa_p50 is not None and vs_p50 is not None:
                        delta = round(fa_p50 - vs_p50, 2)
                    else:
                        delta = "N/A"
                    print(f"{model:<18} {device:<5} {str(fa_p50):<14} {str(vs_p50):<14} {str(delta):<12}")
            print("-" * 65)
        except Exception as e:
            print(f"\n(VisionServe results not available for comparison: {e})")

    # Print full JSON
    print("\n--- fastapi_ort.json ---")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
