"""Thin wrappers around the external open-loop load generators (wrk2, vegeta).

Each runner shells out to the tool, captures stdout, and returns a parsed
:class:`RunOutcome` with CO-correct percentiles (ms) + request/error counts. We never compute
RPS as ``1000/mean_latency`` (the coordinated-omission flaw the revision plan calls out); RPS is
the *measured* completed-request rate reported by the tool.

If the chosen tool is not installed the runner raises a clear error rather than degrading to a
closed-loop generator (ab/wrk), because closed-loop hides p99+.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RunOutcome:
    tool: str
    percentiles_ms: dict[str, float] = field(default_factory=dict)  # "50","95","99","99.9"
    requests: int = 0
    duration_s: float = 0.0
    rps_measured: float = 0.0   # completed requests / wall time (NOT 1000/latency)
    errors: int = 0
    raw_stdout: str = ""

    def as_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "tool": self.tool,
            "requests": self.requests,
            "duration_s": round(self.duration_s, 4),
            "rps_measured": round(self.rps_measured, 3),
            "errors": self.errors,
        }
        row.update({f"p{p}_ms": v for p, v in self.percentiles_ms.items()})
        return row


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise RuntimeError(
            f"'{binary}' not found on PATH. Install it (see eval/README.md). "
            "Do NOT substitute ab/wrk — closed-loop generators hide tail latency."
        )
    return path


# ---- wrk2 -------------------------------------------------------------------------------------

_WRK2_P = re.compile(r"^p([\d.]+)\s+(\d+)\s*$")


def parse_wrk2_stdout(text: str) -> RunOutcome:
    """Parse the custom ``done()`` block emitted by post.lua (percentiles in microseconds)."""
    out = RunOutcome(tool="wrk2", raw_stdout=text)
    requests = 0
    duration_us = 0.0
    errs = 0
    for line in text.splitlines():
        m = _WRK2_P.match(line.strip())
        if m:
            pct, us = m.group(1), float(m.group(2))
            # keep the percentiles the harness reports
            if pct in ("50", "95", "99", "99.9"):
                out.percentiles_ms[pct] = us / 1000.0
            continue
        if line.startswith("requests "):
            requests = int(line.split()[1])
        elif line.startswith("duration_us "):
            duration_us = float(line.split()[1])
        elif line.startswith("errors_"):
            errs += int(line.split()[1])
    out.requests = requests
    out.duration_s = duration_us / 1_000_000.0
    out.errors = errs
    if out.duration_s > 0:
        out.rps_measured = requests / out.duration_s
    return out


def run_wrk2(
    url: str,
    payload_path: str,
    rate: int,
    connections: int,
    duration_s: int,
    threads: int = 4,
    timeout_s: int = 2,
) -> RunOutcome:
    """Run wrk2 at a constant arrival ``rate`` (open-loop). ``connections`` = concurrency C."""
    wrk2 = _require("wrk2")
    lua = os.path.join(os.path.dirname(__file__), "post.lua")
    env = dict(os.environ, PAYLOAD=payload_path)
    cmd = [
        wrk2, "-t", str(threads), "-c", str(connections),
        "-R", str(rate), "-d", f"{duration_s}s",
        "--timeout", f"{timeout_s}s", "-s", lua, url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    text = proc.stdout + "\n" + proc.stderr
    return parse_wrk2_stdout(text)


# ---- vegeta -----------------------------------------------------------------------------------


def run_vegeta(
    url: str,
    payload_path: str,
    rate: int,
    duration_s: int,
    connections: int = 0,
    content_type: str = "application/json",
) -> RunOutcome:
    """Run vegeta at a constant ``rate`` (open-loop) and parse its JSON report.

    vegeta is constant-arrival-rate by construction; ``-rate`` sets requests/sec. ``connections``
    caps max in-flight (vegeta ``-max-workers``) for the concurrency axis.
    """
    vegeta = _require("vegeta")
    # vegeta reads a "targets" file describing the HTTP request + a separate body file.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        tf.write(f"POST {url}\n")
        tf.write(f"Content-Type: {content_type}\n")
        tf.write(f"@{payload_path}\n")
        targets_path = tf.name

    attack = [
        vegeta, "attack", "-targets", targets_path,
        "-rate", str(rate), "-duration", f"{duration_s}s",
    ]
    if connections > 0:
        attack += ["-max-workers", str(connections)]
    try:
        att = subprocess.run(attack, capture_output=True, check=False)
        rep = subprocess.run(
            [vegeta, "report", "-type", "json"],
            input=att.stdout, capture_output=True, check=False,
        )
        return _parse_vegeta_json(rep.stdout.decode("utf-8", "replace"))
    finally:
        try:
            os.unlink(targets_path)
        except OSError:
            pass


def _parse_vegeta_json(text: str) -> RunOutcome:
    out = RunOutcome(tool="vegeta", raw_stdout=text)
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        return out
    lat = d.get("latencies", {})  # nanoseconds
    # vegeta reports p50/p95/p99 by default; map to ms. p99.9 needs -buckets/custom report.
    mapping = {"50": "50th", "95": "95th", "99": "99th"}
    for key, vkey in mapping.items():
        if vkey in lat:
            out.percentiles_ms[key] = lat[vkey] / 1_000_000.0
    out.requests = int(d.get("requests", 0))
    out.duration_s = d.get("duration", 0) / 1_000_000_000.0
    out.rps_measured = float(d.get("rate", 0.0))  # measured rate vegeta achieved
    success = float(d.get("success", 1.0))
    out.errors = int(round(out.requests * (1.0 - success)))
    return out


def get_tool_version(tool: str) -> str:
    """Best-effort version string for provenance."""
    path = shutil.which(tool)
    if not path:
        return "not-installed"
    flag = "--version" if tool == "vegeta" else "--version"
    try:
        r = subprocess.run([path, flag], capture_output=True, text=True, check=False)
        return (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr) else ""
    except Exception:  # noqa: BLE001
        return "unknown"


_AVAILABLE: Optional[str] = None


def autodetect_tool(preferred: Optional[str] = None) -> str:
    """Return the first available load generator, honoring ``preferred``."""
    candidates = [preferred] if preferred else []
    candidates += ["wrk2", "vegeta"]
    for c in candidates:
        if c and shutil.which(c):
            return c
    raise RuntimeError("no open-loop load generator found (need wrk2 or vegeta); see eval/README.md")
