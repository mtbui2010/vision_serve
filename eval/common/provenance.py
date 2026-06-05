"""Provenance sidecar writer.

Every run emits a ``<output>.meta.json`` capturing enough context to make each measured
number traceable and reproducible: git commit, host, EP/device the server selected, GPU
clock, request count, loadgen tool+version, dataset path, and a UTC timestamp.

Nothing here fabricates a measurement — it only records the *environment* of a real run.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from typing import Any, Optional


def _git_commit(repo: Optional[str] = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo or os.getcwd(),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"


def _nvidia_smi_query() -> dict[str, Any]:
    """Best-effort GPU name + clock readback via nvidia-smi. Empty dict if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,clocks.gr,clocks.max.gr,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        )
        rows = [r.strip() for r in out.decode().splitlines() if r.strip()]
        return {"gpus": rows}
    except Exception:  # noqa: BLE001
        return {}


def build_meta(
    *,
    experiment: str,
    target: str,
    model: str,
    device_reported: str = "",
    n_requests: Optional[int] = None,
    loadgen_tool: str = "",
    loadgen_version: str = "",
    dataset: str = "",
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble the provenance dict. ``device_reported`` should be read back from the
    server ``Result.device`` field so the EP that actually ran is recorded, not assumed."""
    meta: dict[str, Any] = {
        "experiment": experiment,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "target": target,
        "model": model,
        "device_reported": device_reported,
        "n_requests": n_requests,
        "loadgen_tool": loadgen_tool,
        "loadgen_version": loadgen_version,
        "dataset": dataset,
        "gpu": _nvidia_smi_query(),
        # NOTE: no measured latency/accuracy values live here — only run context.
    }
    if extra:
        meta["extra"] = extra
    return meta


def write_meta(out_path: str, meta: dict[str, Any]) -> str:
    """Write ``<out_path>.meta.json`` next to a results file and return its path."""
    meta_path = out_path + ".meta.json" if not out_path.endswith(".meta.json") else out_path
    os.makedirs(os.path.dirname(os.path.abspath(meta_path)) or ".", exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return meta_path
