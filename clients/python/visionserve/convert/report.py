"""The verification report: one row per check, a compact table, convert-report.json, and a short
summary appended to the installed manifest's header comments.

Statuses, worst first: FAIL > ERROR > WARN > PASS. SKIP (a check that could not run, always with
the reason) and INFO (speed numbers) never change the overall status. ERROR means the CHECK broke
(a bug, a network error), not the model; it is printed loudly but does not uninstall anything.

Pure Python + stdlib: imported by unit tests with no framework installed.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
from pathlib import Path
from typing import Dict, List, Optional

PASS, WARN, FAIL, SKIP, ERROR, INFO = "PASS", "WARN", "FAIL", "SKIP", "ERROR", "INFO"
_RANK = {PASS: 0, SKIP: 0, INFO: 0, WARN: 1, ERROR: 2, FAIL: 3}

# Defaults for every graded metric. Override with `--threshold key=value` (repeatable) or the
# Python API's `thresholds={...}`. Gray levels are 0-255 pixel units: the tensor difference is
# multiplied back by the manifest's std*255, so 1.0 means "one pixel level".
DEFAULT_THRESHOLDS: Dict[str, float] = {
    # B1: mean |server - reference| over the image tensor, in gray levels.
    "b1_mean_warn": 2.0,
    "b1_mean_fail": 8.0,
    # B2 detection: matched fraction (higher is better), mean |Δconf|, mean box error / image diagonal.
    "b2_match_warn": 0.95,
    "b2_match_fail": 0.80,
    "b2_conf_warn": 0.03,
    "b2_conf_fail": 0.10,
    "b2_box_warn": 0.01,
    "b2_box_fail": 0.03,
    # B2 classification: top-1 agreement (higher is better), max |Δprob|.
    "b2_top1_warn": 1.0,
    "b2_top1_fail": 0.75,
    "b2_prob_warn": 0.05,
    "b2_prob_fail": 0.20,
    # B2 embeddings: min cosine; depth: min Pearson r (higher is better).
    "b2_cos_warn": 0.995,
    "b2_cos_fail": 0.95,
    "b2_depth_warn": 0.99,
    "b2_depth_fail": 0.95,
    # Detection matching: IoU for "same object", and the band above conf_threshold in which a
    # detection present on only one side is a threshold-boundary flip, not a disagreement.
    "b2_iou": 0.5,
    # P (--precision): distance between the reduced model and the FP32 ONNX on the calibration images
    # (precision.output_error: 1 - matched IoU of the detections as a set; relative L2 for other heads).
    # PROVISIONAL: set from RF-DETR-base on 5 images (fp16 0.018, int8 all layers 0.36).
    "p_err_warn": 0.05,
    "p_err_fail": 0.25,
    "b2_boundary": 0.05,
}


def parse_thresholds(items) -> Dict[str, float]:
    """['b1_mean_fail=10', ...] -> defaults updated. Unknown keys are refused (a typo must not
    silently leave the default in force)."""
    out = dict(DEFAULT_THRESHOLDS)
    for it in items or []:
        if isinstance(it, dict):
            pairs = it.items()
        else:
            if "=" not in str(it):
                raise ValueError(f"--threshold expects key=value, got {it!r}")
            k, v = str(it).split("=", 1)
            pairs = [(k.strip(), v.strip())]
        for k, v in pairs:
            if k not in DEFAULT_THRESHOLDS:
                raise ValueError(f"unknown threshold {k!r}; known: {', '.join(sorted(DEFAULT_THRESHOLDS))}")
            out[k] = float(v)
    return out


def grade_high(value, warn, fail) -> str:
    """Higher is better (agreement, cosine, r)."""
    if value is None:
        return SKIP
    if value < fail:
        return FAIL
    if value < warn:
        return WARN
    return PASS


def grade_low(value, warn, fail) -> str:
    """Lower is better (differences, errors)."""
    if value is None:
        return SKIP
    if value > fail:
        return FAIL
    if value > warn:
        return WARN
    return PASS


def worst(*statuses) -> str:
    s = [x for x in statuses if x]
    return max(s, key=lambda x: _RANK.get(x, 0)) if s else SKIP


@dataclasses.dataclass
class TierResult:
    tier: str                  # "A", "B1", "B2", "C", "speed"
    title: str                 # what was compared, e.g. "outputs vs rfdetr RFDETR.predict"
    status: str
    summary: str               # one line for the table
    model: str = ""
    metrics: dict = dataclasses.field(default_factory=dict)
    notes: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Report:
    models: List[str]
    task: str = ""
    architecture: str = ""
    tiers: List[TierResult] = dataclasses.field(default_factory=list)
    server: dict = dataclasses.field(default_factory=dict)
    thresholds: dict = dataclasses.field(default_factory=dict)
    created: str = dataclasses.field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat())
    uninstalled: bool = False
    restored: List[str] = dataclasses.field(default_factory=list)  # previous versions put back after a FAIL
    report_path: Optional[str] = None

    # ---- status ----------------------------------------------------------------------------
    @property
    def status(self) -> str:
        s = worst(*(t.status for t in self.tiers))
        return PASS if s in (SKIP, INFO) else s

    @property
    def ok(self) -> bool:
        """False only when a check FAILED (an ERROR or WARN is reported but does not fail)."""
        return self.status != FAIL

    def tier(self, name: str, model: Optional[str] = None) -> Optional[TierResult]:
        for t in self.tiers:
            if t.tier == name and (model is None or t.model == model):
                return t
        return None

    def add(self, t: TierResult) -> TierResult:
        self.tiers.append(t)
        return t

    # ---- rendering -------------------------------------------------------------------------
    def table(self) -> str:
        head = f"VisionServe convert report: {', '.join(self.models)}"
        if self.task:
            head += f" ({self.task}, {self.architecture})"
        lines = [head, f"overall: {self.status}" + ("  [model UNINSTALLED]" if self.uninstalled else "")
                 + (f"  [previous version restored: {', '.join(self.restored)}]" if self.restored else "")]
        multi = len({t.model for t in self.tiers if t.model}) > 1
        rows = []
        for t in self.tiers:
            name = t.title + (f" [{t.model}]" if multi and t.model else "")
            rows.append((t.tier, name, t.status, t.summary))
        w0 = max([4] + [len(r[0]) for r in rows])
        w1 = min(46, max([5] + [len(r[1]) for r in rows]))
        lines.append(f"  {'tier'.ljust(w0)}  {'check'.ljust(w1)}  status  result")
        for tier, name, st, summ in rows:
            if len(name) > w1:
                name = name[: w1 - 1] + "~"
            lines.append(f"  {tier.ljust(w0)}  {name.ljust(w1)}  {st.ljust(6)}  {summ}")
        notes = [(t.tier, n) for t in self.tiers for n in t.notes]
        if notes:
            lines.append("notes:")
            lines += [f"  [{tier}] {n}" for tier, n in notes]
        if self.report_path:
            lines.append(f"full report: {self.report_path}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.table()

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["status"] = self.status
        return d

    def to_json(self) -> str:
        return json.dumps(_jsonable(self.to_dict()), indent=2, sort_keys=False)

    # ---- persistence -----------------------------------------------------------------------
    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path = str(path)
        path.write_text(self.to_json() + "\n")
        return path

    def manifest_comment_lines(self, model: Optional[str] = None) -> List[str]:
        """Short '# ...' lines for the manifest header (Go's YAML parser ignores comments)."""
        out = [f"# verify {self.created[:10]}: overall {self.status} (details: convert-report.json)"]
        for t in self.tiers:
            if model and t.model and t.model != model:
                continue
            s = f"{t.tier} {t.title}: {t.status}" + (f", {t.summary}" if t.summary else "")
            out.append("#   " + _one_line(s, 150))
        return out


def annotate_manifest(manifest: Path, lines: List[str]) -> None:
    """Insert `lines` (each starting with '#') after the manifest's leading comment block,
    replacing a previous verify block. Comment-only: the YAML body is untouched."""
    text = Path(manifest).read_text()
    body = text.splitlines()
    i = 0
    head = []
    while i < len(body) and body[i].startswith("#"):
        head.append(body[i])
        i += 1
    # drop a previous verify block (re-runs replace it instead of stacking)
    kept, skipping = [], False
    for ln in head:
        if ln.startswith("# verify "):
            skipping = True
            continue
        if skipping and ln.startswith("#   "):
            continue
        skipping = False
        kept.append(ln)
    for ln in lines:
        if not ln.startswith("#") or "\n" in ln:
            raise ValueError(f"manifest annotation must be single '#' lines, got {ln!r}")
    new = kept + list(lines) + body[i:]
    Path(manifest).write_text("\n".join(new) + ("\n" if text.endswith("\n") else ""))


def _one_line(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 3] + "..."


def _jsonable(x):
    """numpy scalars/arrays and Paths -> plain JSON types."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if np is not None:
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return str(x)
    return x


def fmt(v, nd=3) -> str:
    """Compact number formatting for summaries."""
    if v is None:
        return "n/a"
    if isinstance(v, int):
        return str(v)
    v = float(v)
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
        return f"{v:.2e}"
    return f"{v:.{nd}g}"
