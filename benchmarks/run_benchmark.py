#!/usr/bin/env python3
"""
VisionServe Performance Benchmark
Measures cold start, warm latency (p50/p95/p99/mean), throughput,
RSS memory, and VRAM for each (model, device) combination.
"""

import subprocess
import json
import os
import time
import signal
import re
import statistics
import sys
import shutil
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
WORKDIR         = "/home/trung/trung_workdir/vision_serve"
BINARY          = f"{WORKDIR}/bin/visionserve"
MODELS_DIR      = f"{WORKDIR}/models"
SERVER_ADDR     = ":11435"
SERVER_URL      = "http://localhost:11435"
TEST_IMAGE      = "/tmp/cats.jpg"
RESULTS_FILE    = f"{WORKDIR}/benchmarks/results/visionserve.json"

WARMUP_REQS     = 5
MEASURE_REQS    = 30
STARTUP_TIMEOUT = 120   # seconds to wait for server health

# (model_name, prompt_or_None)
MODELS = [
    ("rf-detr",        None),
    ("grounding-dino", "cat. remote."),
    ("grounded-sam",   "cat. remote."),
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def log(msg: str):
    print(f"[bench] {msg}", flush=True)


def run(cmd: list[str], capture=True, env=None, timeout=None):
    """Run a command; return (stdout, returncode)."""
    try:
        r = subprocess.run(
            cmd, capture_output=capture, text=True, env=env, timeout=timeout
        )
        return r.stdout.strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "", -1
    except FileNotFoundError as e:
        return str(e), -2


def ensure_test_image():
    if not os.path.exists(TEST_IMAGE):
        log(f"Downloading test image to {TEST_IMAGE} ...")
        out, rc = run([
            "curl", "-sL",
            "http://images.cocodataset.org/val2017/000000039769.jpg",
            "-o", TEST_IMAGE,
        ], timeout=60)
        if rc != 0:
            raise RuntimeError(f"Failed to download test image: {out}")
    log(f"Test image OK: {TEST_IMAGE}")


def collect_hardware():
    gpu, cpu_cores, ram_gb = None, None, None
    out, rc = run(["nvidia-smi", "-L"])
    if rc == 0 and out:
        gpu = out.split("\n")[0].strip()

    out, rc = run(["nproc"])
    if rc == 0:
        try:
            cpu_cores = int(out.strip())
        except ValueError:
            pass

    out, rc = run(["free", "-m"])
    if rc == 0:
        for line in out.splitlines():
            if line.startswith("Mem:"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        ram_gb = round(int(parts[1]) / 1024)
                    except ValueError:
                        pass
                break

    return {"gpu": gpu, "cpu_cores": cpu_cores, "ram_gb": ram_gb}


def get_ort_cpu_path():
    result, rc = run([
        "bash", "-c",
        "find $HOME -name 'libonnxruntime.so*' 2>/dev/null | grep -v node_modules | grep 'onnxruntime/capi.*\\.so\\.[0-9]' | head -1"
    ])
    return result.strip() if rc == 0 else ""


def build_server_env(device: str):
    """Build the environment dict for starting the server process."""
    env = os.environ.copy()
    env["PATH"] = f"/home/trung/go-sdk/go/bin:{env.get('PATH', '')}"
    env["GOFLAGS"] = "-mod=mod"

    if device == "gpu":
        # Source gpu-env.sh to get ORT_DYLIB_PATH and LD_LIBRARY_PATH
        out, rc = run([
            "bash", "-c",
            f"source {WORKDIR}/scripts/gpu-env.sh && echo ORT_DYLIB_PATH=$ORT_DYLIB_PATH && echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
        ])
        if rc == 0:
            for line in out.splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip()
        log(f"GPU env: ORT_DYLIB_PATH={env.get('ORT_DYLIB_PATH','NOT SET')}")
    else:
        # CPU: find the CPU ORT lib
        cpu_ort = get_ort_cpu_path()
        if cpu_ort:
            env["ORT_DYLIB_PATH"] = cpu_ort
        log(f"CPU env: ORT_DYLIB_PATH={env.get('ORT_DYLIB_PATH','NOT SET')}")

    return env


def wait_for_health(url: str, timeout: int = STARTUP_TIMEOUT):
    """Poll /api/health until 200 OK. Returns elapsed ms, or -1 on timeout."""
    health_url = f"{url}/api/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            out, rc = run(
                ["curl", "-sf", health_url],
                timeout=5,
            )
            if rc == 0:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def get_rss_mb(pid: int):
    """Read VmRSS from /proc/<pid>/status in MB."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return round(kb / 1024, 1)
    except Exception:
        pass
    return None


def get_vram_mb(pid: int):
    """Get VRAM usage for the server pid from nvidia-smi."""
    out, rc = run([
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader",
    ])
    if rc != 0:
        return None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(pid):
            # parts[1] is like "1234 MiB"
            m = re.match(r"(\d+)", parts[1])
            if m:
                return int(m.group(1))
    return None


def curl_predict(model: str, image: str, prompt: str | None, url: str):
    """
    Send one predict request. Returns (wall_ms, duration_ms_from_json, error_str).
    Uses curl -s -w '%{time_total}' to get wall time.
    """
    resp_file = "/tmp/_bench_resp.json"
    cmd = [
        "curl", "-sf",
        "-w", "%{time_total}",
        "-o", resp_file,
        "-F", f"model={model}",
        "-F", f"image=@{image}",
    ]
    if prompt:
        cmd += ["-F", f"prompt={prompt}"]
    cmd.append(f"{url}/api/predict")

    start = time.monotonic()
    out, rc = run(cmd, timeout=300)
    wall_elapsed = (time.monotonic() - start) * 1000  # ms

    if rc != 0:
        return None, None, f"curl rc={rc}"

    # Parse wall time from curl output (the last token is time_total in seconds)
    wall_ms = None
    try:
        # curl -w writes time to stdout after -o captures body
        wall_ms = float(out.strip()) * 1000
    except (ValueError, AttributeError):
        wall_ms = wall_elapsed  # fallback to Python-measured

    # Parse duration_ms from JSON response
    duration_ms = None
    try:
        with open(resp_file) as f:
            data = json.load(f)
        duration_ms = data.get("duration_ms")
    except Exception:
        pass

    return wall_ms, duration_ms, None


def compute_percentile(values: list[float], p: float) -> float:
    """Compute p-th percentile (0-100)."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (p / 100) * (len(sorted_v) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_v) - 1)
    frac = idx - lo
    return round(sorted_v[lo] + frac * (sorted_v[hi] - sorted_v[lo]), 2)


def run_model_device(model: str, prompt: str | None, device: str):
    """
    Run a full benchmark for one (model, device) pair.
    Returns a dict with all metrics.
    """
    log(f"\n{'='*60}")
    log(f"Benchmarking  model={model}  device={device}")
    log(f"{'='*60}")

    env = build_server_env(device)
    result = {
        "cold_start_ms": None,
        "latency_p50_ms": None,
        "latency_p95_ms": None,
        "latency_p99_ms": None,
        "latency_mean_ms": None,
        "server_inference_p50_ms": None,
        "throughput_rps": None,
        "rss_mb": None,
        "vram_mb": None,
        "n_runs": MEASURE_REQS,
        "_error": None,
    }

    # ── Start server ────────────────────────────────────────────────────────
    cmd = [
        BINARY, "serve",
        "--models", MODELS_DIR,
        "--addr", SERVER_ADDR,
    ]
    log(f"Starting: {' '.join(cmd)}")
    t_launch = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd,
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

    # ── Wait for health (cold start) ────────────────────────────────────────
    # The first predict request triggers model load; health just confirms HTTP is up.
    health_ok = wait_for_health(SERVER_URL, timeout=30)
    if not health_ok:
        proc.kill()
        result["_error"] = "Server did not become healthy within 30s"
        log(f"ERROR: {result['_error']}")
        return result
    t_http_up = time.monotonic()
    log(f"HTTP up in {(t_http_up - t_launch)*1000:.0f} ms (process → health endpoint)")

    # Cold start = time from process launch until first successful PREDICT response
    # (includes HTTP startup + model load + first inference)
    log("Sending cold-start request (triggers model load) ...")
    t_cold_start = time.monotonic()
    cold_wall, cold_dur, cold_err = curl_predict(model, TEST_IMAGE, prompt, SERVER_URL)
    t_cold_end = time.monotonic()

    if cold_err:
        proc.kill()
        result["_error"] = f"Cold-start predict failed: {cold_err}"
        log(f"ERROR: {result['_error']}")
        return result

    cold_start_ms = (t_cold_end - t_launch) * 1000   # launch → first response
    result["cold_start_ms"] = round(cold_start_ms, 1)
    log(f"Cold start: {cold_start_ms:.0f} ms (launch → first predict response)")
    log(f"  (cold predict wall: {cold_wall:.1f} ms, server inference: {cold_dur} ms)")

    # ── Warmup ──────────────────────────────────────────────────────────────
    log(f"Running {WARMUP_REQS} warmup requests ...")
    for i in range(WARMUP_REQS):
        w, d, err = curl_predict(model, TEST_IMAGE, prompt, SERVER_URL)
        if err:
            log(f"  warmup {i+1}/{WARMUP_REQS} FAILED: {err}")
        else:
            log(f"  warmup {i+1}/{WARMUP_REQS}: wall={w:.1f} ms  server={d} ms")

    # ── Memory measurement (post-warmup) ────────────────────────────────────
    rss = get_rss_mb(pid)
    vram = get_vram_mb(pid)
    result["rss_mb"] = rss
    result["vram_mb"] = vram
    log(f"Memory: RSS={rss} MB  VRAM={vram} MB")

    # ── Timed measurement (30 requests) ─────────────────────────────────────
    log(f"Running {MEASURE_REQS} timed requests ...")
    wall_times = []
    server_times = []
    t_batch_start = time.monotonic()

    for i in range(MEASURE_REQS):
        w, d, err = curl_predict(model, TEST_IMAGE, prompt, SERVER_URL)
        if err:
            log(f"  req {i+1}/{MEASURE_REQS} FAILED: {err}")
            continue
        wall_times.append(w)
        if d is not None:
            server_times.append(float(d))
        log(f"  req {i+1:02d}/{MEASURE_REQS}: wall={w:.1f} ms  server={d} ms")

    t_batch_end = time.monotonic()
    batch_elapsed = t_batch_end - t_batch_start

    if not wall_times:
        proc.kill()
        result["_error"] = "All measurement requests failed"
        log(f"ERROR: {result['_error']}")
        return result

    result["latency_p50_ms"]   = compute_percentile(wall_times, 50)
    result["latency_p95_ms"]   = compute_percentile(wall_times, 95)
    result["latency_p99_ms"]   = compute_percentile(wall_times, 99)
    result["latency_mean_ms"]  = round(statistics.mean(wall_times), 2)
    result["throughput_rps"]   = round(len(wall_times) / batch_elapsed, 3)
    if server_times:
        result["server_inference_p50_ms"] = compute_percentile(server_times, 50)

    log(f"Latency p50={result['latency_p50_ms']} p95={result['latency_p95_ms']} "
        f"p99={result['latency_p99_ms']} mean={result['latency_mean_ms']} ms")
    log(f"Throughput: {result['throughput_rps']} req/s  ({len(wall_times)}/{MEASURE_REQS} succeeded)")
    log(f"Server-side inference p50: {result['server_inference_p50_ms']} ms")

    # ── Shutdown ─────────────────────────────────────────────────────────────
    log(f"Stopping server (PID {pid}) ...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    log("Server stopped.")
    time.sleep(2)

    return result


def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    log("=== VisionServe Performance Benchmark ===")
    log(f"Binary: {BINARY}")
    log(f"Models dir: {MODELS_DIR}")
    log(f"Results: {RESULTS_FILE}")

    # ── Pre-checks ────────────────────────────────────────────────────────────
    if not os.path.exists(BINARY):
        log(f"ERROR: Binary not found: {BINARY}")
        sys.exit(1)

    ensure_test_image()

    # ── Hardware info ─────────────────────────────────────────────────────────
    hardware = collect_hardware()
    log(f"Hardware: {hardware}")

    # ── Run benchmarks ────────────────────────────────────────────────────────
    results_models = {}
    notes_list = []

    for (model, prompt) in MODELS:
        results_models[model] = {}
        for device in ("cpu", "gpu"):
            metrics = run_model_device(model, prompt, device)

            # Strip internal _error, promote to notes
            err = metrics.pop("_error", None)
            if err:
                notes_list.append(f"{model}/{device}: {err}")
                # Replace failed metrics with None
                for k in metrics:
                    if metrics[k] is not None and k != "n_runs":
                        metrics[k] = None

            results_models[model][device] = metrics

    # ── Build output JSON ─────────────────────────────────────────────────────
    if not notes_list:
        notes = "All benchmarks completed successfully."
    else:
        notes = "; ".join(notes_list)

    output = {
        "baseline": "visionserve_go_http",
        "hardware": hardware,
        "models": results_models,
        "notes": notes,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nResults written to {RESULTS_FILE}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print(f"{'Model':<18} {'Device':<8} {'p50 ms':>10} {'Throughput':>12} {'VRAM MB':>10}")
    print("-"*80)
    for model in results_models:
        for device in ("cpu", "gpu"):
            m = results_models[model][device]
            p50   = m.get("latency_p50_ms")
            tput  = m.get("throughput_rps")
            vram  = m.get("vram_mb")
            p50_s   = f"{p50}"   if p50   is not None else "N/A"
            tput_s  = f"{tput}"  if tput  is not None else "N/A"
            vram_s  = f"{vram}"  if vram  is not None else "N/A"
            print(f"{model:<18} {device:<8} {p50_s:>10} {tput_s:>12} {vram_s:>10}")
    print("="*80)

    # Print full JSON to stdout as well
    print("\n--- visionserve.json ---")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
