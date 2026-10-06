"""`visionserve sensitivity` and `visionserve optimize`: reduced precision for an INSTALLED model,
aimed at an edge target (Jetson Orin / Thor) or a plain CPU / CUDA host.

    python -m visionserve.convert sensitivity <model> --models DIR --images DIR [...]
    python -m visionserve.convert optimize    <model> --models DIR --target T --images DIR [...]

(The Go binary's `visionserve sensitivity|optimize` runs exactly this, in the converter image or with
a local Python.) Both reuse the converter's precision code (precision.py: FP16, INT8 QDQ, mixed
per-layer formats, one-layer-at-a-time sensitivity) on the model's ONNX file; nothing here edits a
graph by itself.

What is measured where. Output error (vs FP32) and accuracy (--labels, mAP of the SERVED model
through a temporary VisionServe server) are measured on this machine and carry over to the target as
long as the target runs the same ONNX file on an execution provider that computes the same numbers.
Latency does NOT carry over: it is measured on this host and labelled so. A Jetson's latency is
measured on the Jetson (`visionserve bench <model>-<format>`). See website/docs/guides/edge.md.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as _dt
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .common import MODELS_DIR, ConvertError, log
from .edgereport import FAIL, PASS, WARN, Chart, Report, Row, Table, bar_chart, fmt_num

TOOLS = ("sensitivity", "optimize")

# ------------------------------------------------------------------------------------------
# Presets: ONE table, mirrored in website/docs/guides/edge.md. A preset says which variants are
# worth building for a target, which execution provider runs them there, and which of them is
# expected to be fastest on that EP (used only when latency cannot be measured on a comparable
# EP here). "verified" lists what was actually run on the hardware.
# ------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Preset:
    target: str
    hardware: str
    software: str
    ep: str                        # EP chain VisionServe uses there by default
    candidates: tuple              # variants built and measured (fp32 is always the baseline row)
    mixed_formats: tuple           # the per-layer ladder of the "mixed" candidate
    host_ep: str                   # "gpu" | "cpu": latency here is comparable when measured on this EP class
    verified: str = ""
    notes: tuple = ()


PRESETS: Dict[str, Preset] = {
    "jetson-orin": Preset(
        target="jetson-orin",
        hardware="Jetson AGX Orin / Orin NX / Orin Nano: Ampere GPU (sm_87), FP16 and INT8 tensor cores, no FP8",
        software="JetPack 6.x (6.1/6.2: CUDA 12.6, cuDNN 9.3, TensorRT 10.3; 6.0: TensorRT 8.6); "
                 "ONNX Runtime with the CUDA EP (e.g. the Jetson AI Lab onnxruntime-gpu build for jp6/cu126)",
        ep="cuda → cpu (TensorRT opt-in: --tensorrt)",
        candidates=("fp16", "int8", "mixed"), mixed_formats=("int8", "fp16"), host_ep="gpu",
        verified="latency/accuracy workflow run on an RTX A6000 (Ampere, same EP); not run on an Orin by this tool",
        notes=("FP16 halves the file. Whether it is FASTER on the CUDA EP depends on the graph: the COCO RF-DETR nano "
               "export in FP16 ran 3.5x SLOWER than FP32 on ONNX Runtime 1.26's CUDA EP (RTX A6000, 16.1 vs 4.6 ms "
               "per session run). ONNX Runtime's CUDA EP does not run QDQ INT8 faster (it dequantizes): INT8 and "
               "mixed pay off in size, and in speed only under TensorRT. Measure, then bench on the device.",)),
    "jetson-thor": Preset(
        target="jetson-thor",
        hardware="Jetson AGX Thor: Blackwell GPU (sm_110), FP16 / INT8 / FP8 / FP4 tensor cores",
        software="JetPack 7.x (7.0/7.1: CUDA 13.0, TensorRT 10.13; 7.2: CUDA 13.2, TensorRT 10.16), SBSA-aligned; "
                 "ONNX Runtime with the CUDA EP for CUDA 13 (e.g. the Jetson AI Lab onnxruntime-gpu 1.24 wheel for "
                 "sbsa/cu130, which carries sm_110 kernels)",
        ep="cuda → cpu (TensorRT opt-in: --tensorrt)",
        candidates=("fp16", "int8", "mixed"), mixed_formats=("int8", "fp16"), host_ep="gpu",
        verified="UNTESTED ON HARDWARE: the same candidates as Orin; nothing here was run on a Thor",
        notes=("Untested on a Thor. Check on the device: the CUDA EP loads (bench reports gpu:0, not cpu), FP16 "
               "and INT8 outputs match FP32 within the budget there, TensorRT 10.13+ builds the INT8 QDQ engine, "
               "and memory via tegrastats (nvidia-smi reports no memory on Thor).",
               "FP8 / FP4 are what Thor adds, but ONNX Runtime 1.26 cannot build or run them (sensitivity can "
               "SIMULATE them: --formats int8,fp8,fp4). Deploying FP8/FP4 means native TensorRT with NVIDIA's "
               "Model Optimizer, outside VisionServe.")),
    "cuda": Preset(
        target="cuda",
        hardware="a desktop / server NVIDIA GPU",
        software="ONNX Runtime with the CUDA EP",
        ep="cuda → cpu (TensorRT opt-in: --tensorrt)",
        candidates=("fp16", "mixed"), mixed_formats=("int8", "fp16"), host_ep="gpu",
        verified="run on an RTX A6000",
        notes=("FP16 halves the file. Whether it is FASTER on the CUDA EP depends on the graph: the COCO RF-DETR nano "
               "export in FP16 ran 3.5x SLOWER than FP32 on ONNX Runtime 1.26's CUDA EP (RTX A6000, 16.1 vs 4.6 ms "
               "per session run). ONNX Runtime's CUDA EP does not run QDQ INT8 faster (it dequantizes): INT8 and "
               "mixed pay off in size, and in speed only under TensorRT. Measure, then bench on the device.",)),
    "cpu": Preset(
        target="cpu",
        hardware="x86-64 or arm64 CPU",
        software="ONNX Runtime CPU EP",
        ep="cpu",
        candidates=("int8", "mixed", "int4"), mixed_formats=("int8",), host_ep="cpu",
        verified="run on rf-detr-nano on a shared 48-core x86-64 host",
        notes=("FP16 is left out: the CPU EP has few FP16 kernels and inserts casts, so it is usually slower than "
               "FP32. INT4 is weight-only (MatMulNBits): smaller, but it does not touch convolutions.",)),
}


def preset_table_rows() -> List[List[str]]:
    """The preset table as rows (docs and --help print the same thing)."""
    return [[p.target, ", ".join(("fp32 (baseline)",) + p.candidates), "/".join(p.mixed_formats), p.ep, p.verified]
            for p in PRESETS.values()]


# ------------------------------------------------------------------------------------------
# the installed model
# ------------------------------------------------------------------------------------------

@dataclasses.dataclass
class InstalledModel:
    name: str
    dir: Path
    manifest: dict
    role_key: str                  # "model_file" or "files.model"
    onnx: Path
    spec: object
    input_name: str
    input_shape: list
    labels: Optional[List[str]]
    task: str
    architecture: str


def find_model(models_dir, name: str) -> InstalledModel:
    import yaml
    from .common import onnx_io
    from .spec import spec_from_manifest
    root = Path(models_dir)
    cand = root / name
    found = None
    if (cand / "manifest.yaml").is_file():
        found = cand
    else:
        for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")) if root.is_dir() else []:
            mf = d / "manifest.yaml"
            if mf.is_file():
                try:
                    if (yaml.safe_load(mf.read_text()) or {}).get("name") == name:
                        found = d
                        break
                except Exception:  # noqa: BLE001 — a broken neighbour must not stop the search
                    continue
    if found is None:
        raise ConvertError(f"model {name!r} not found in {root} (visionserve list --models {root})")
    doc = yaml.safe_load((found / "manifest.yaml").read_text()) or {}
    files = doc.get("files") or {}
    if files:
        if set(files) != {"model"}:
            raise ConvertError(f"{name} has {len(files)} ONNX sessions ({', '.join(sorted(files))}); sensitivity and "
                               "optimize support single-session image models (one ONNX file) for now")
        role_key, rel = "files.model", files["model"]
    elif doc.get("model_file"):
        role_key, rel = "model_file", doc["model_file"]
    else:
        raise ConvertError(f"{name}: the manifest names no ONNX file (model_file / files.model)")
    onnx_path = found / rel
    if not onnx_path.is_file():
        raise ConvertError(f"{name}: {onnx_path} is missing")
    ins, _ = onnx_io(onnx_path)
    if len(ins) != 1:
        raise ConvertError(f"{name}: the graph has {len(ins)} inputs ({[n for n, _, _ in ins]}); single-input "
                           "image models only")
    in_name, in_shape = ins[0][0], ins[0][1]
    if any(isinstance(d, str) for d in in_shape):
        raise ConvertError(f"{name}: input {in_name} has dynamic dims {in_shape}; static-shape image models only")
    labels = None
    if doc.get("labels") and (found / str(doc["labels"])).is_file():
        labels = [ln.strip() for ln in (found / str(doc["labels"])).read_text().splitlines() if ln.strip()]
    return InstalledModel(name=doc.get("name") or name, dir=found, manifest=doc, role_key=role_key, onnx=onnx_path,
                          spec=spec_from_manifest(doc), input_name=in_name, input_shape=list(in_shape),
                          labels=labels, task=str(doc.get("task") or ""), architecture=str(doc.get("architecture") or ""))


def calibration_feeds(m: InstalledModel, images, n: int) -> List[dict]:
    from .precision import load_calibration
    return load_calibration(m.spec, m.input_name, m.input_shape, images, n)


def _providers(gpu: bool, n_images: int):
    """ORT providers for the precision measurements, or None (= precision.py's CPU default)."""
    if not gpu:
        return None
    from . import precision
    rp = getattr(precision, "resolve_providers", None)
    if rp is not None:
        try:
            return rp("cuda", n_images)
        except Exception as e:  # noqa: BLE001
            log(f"edge: --gpu: {e}; measuring on CPU")
            return None
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _accepts(fn, kw: str) -> bool:
    try:
        return kw in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def measure(m: InstalledModel, feeds, formats: Sequence[str], method: str, gpu: bool):
    """Per-layer sensitivity (precision.measure_sensitivity). Returns (scores, note on the EP)."""
    from .precision import measure_sensitivity

    def progress(i, n, name):
        if i == 0 or (i + 1) * 10 // n != i * 10 // n:
            log(f"  sensitivity {i + 1}/{n} ({name.split(':')[0]})")
    kw = {"formats": list(formats)}
    provs = _providers(gpu, len(feeds))
    ep_note = "CPU (ONNX Runtime CPU EP)"
    if provs is not None:
        if _accepts(measure_sensitivity, "providers"):
            kw["providers"] = provs
            ep_note = f"providers {provs}"
        else:
            ep_note = "CPU: --gpu needs a converter with GPU sensitivity (providers=); this one measures on CPU"
    log(f"edge: measuring {', '.join(formats)} sensitivity of every MatMul/Gemm/Conv on {len(feeds)} image(s), {ep_note} ...")
    scores = measure_sensitivity(str(m.onnx), feeds, formats[0], method, progress, **kw)
    return scores, ep_note


def scores_from_json(payload: dict):
    """LayerScores back from a sensitivity.json (visionserve sensitivity --save)."""
    from .precision import LayerScore
    fields = {f.name for f in dataclasses.fields(LayerScore)}
    out = []
    for d in payload.get("layers") or []:
        d = {k: v for k, v in d.items() if k in fields}
        for k in ("output_err", "sqnr_db", "act_absmax", "act_outlier_ratio"):
            if d.get(k) is None:
                d[k] = math.inf if k == "output_err" else 0.0
        d["errs"] = {f: (math.inf if v is None else float(v)) for f, v in (d.get("errs") or {}).items()}
        out.append(LayerScore(**d))
    return out


# ------------------------------------------------------------------------------------------
# sensitivity
# ------------------------------------------------------------------------------------------

def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", help="installed model name")
    p.add_argument("--models", default=str(MODELS_DIR), help="registry directory")
    p.add_argument("--images", required=True, metavar="DIR", help="photos like the deployment's (8-32)")
    p.add_argument("--calib-method", choices=("minmax", "entropy", "percentile"), default="percentile",
                   help="INT8 activation range (default percentile)")
    p.add_argument("--gpu", action="store_true", help="measure with the CUDA EP where supported")
    p.add_argument("--json", action="store_true", help="print one JSON object only")
    p.add_argument("--report", metavar="FILE.html", help="write a self-contained HTML report")


def sensitivity_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="visionserve sensitivity", allow_abbrev=False,
                                description="Per-layer sensitivity of an installed model to reduced precision.")
    add_common(p)
    p.add_argument("--formats", default="int8", help="formats per layer: int8, fp16, int4, fp8, fp4 (default int8; "
                                                     "fp8/fp4 simulated)")
    p.add_argument("--sens-images", type=int, default=8, metavar="N", help="images per layer (default 8)")
    p.add_argument("--threshold", type=float, default=0.05, metavar="E",
                   help="a layer is sensitive when its error alone exceeds E (default 0.05)")
    p.add_argument("--top", type=int, default=15, metavar="N", help="rows in the table (default 15)")
    p.add_argument("--save", metavar="FILE", help="write the per-layer scores (sensitivity.json)")
    return p


_FMT_NAMES = {"int8": "INT8", "fp16": "FP16", "int4": "INT4", "fp8": "FP8", "fp4": "FP4"}


def _short(name: str, n: int = 56) -> str:
    return name if len(name) <= n else "…" + name[-(n - 1):]


def run_sensitivity(args) -> Report:
    from .precision import parse_formats, sensitivity_payload
    if not (args.threshold > 0 and math.isfinite(args.threshold)):
        raise ConvertError("--threshold must be a finite number > 0")
    if args.sens_images < 1 or args.top < 1:
        raise ConvertError("--sens-images and --top must be at least 1")
    formats = parse_formats(args.formats)
    if not formats or formats == ["fp32"]:
        raise ConvertError("--formats needs at least one of int8, fp16, int4, fp8, fp4")
    formats = [f for f in formats if f != "fp32"]
    # parse_formats orders most-aggressive first; the FIRST format the user wrote ranks the layers
    asked = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    formats = sorted(formats, key=lambda f: asked.index(f) if f in asked else 99)
    m = find_model(args.models, args.model)
    if not Path(args.images).is_dir():
        raise ConvertError(f"--images {args.images}: expected a directory of photos")
    t0 = time.time()
    feeds = calibration_feeds(m, args.images, args.sens_images)
    scores, ep_note = measure(m, feeds, formats, args.calib_method, args.gpu)
    took = time.time() - t0
    first = formats[0]
    payload = sensitivity_payload(scores, first, args.calib_method, len(feeds), formats=formats)
    payload.update({"model": m.name, "onnx": m.onnx.name, "threshold": args.threshold, "measured_on": ep_note})
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(payload, indent=1) + "\n")
        log(f"edge: per-layer scores written to {args.save}")
    return sensitivity_report(m, scores, formats, args, len(feeds), took, ep_note, payload)


def sensitivity_report(m, scores, formats, args, n_images, took, ep_note, payload) -> Report:
    first = formats[0]
    ranked = sorted(scores, key=lambda s: s.rank)
    thr = args.threshold

    def err(s, f):
        return s.errs.get(f)
    sens = {f: [s for s in ranked if err(s, f) is not None and not (err(s, f) <= thr)] for f in formats}
    measured = {f: [s for s in ranked if err(s, f) is not None] for f in formats}
    n_sens = len(sens[first])
    worst = ranked[0] if ranked else None
    fname = _FMT_NAMES.get(first, first)
    if worst is None:
        raise ConvertError("no layer was measured")
    if n_sens == 0:
        verdict = PASS
        reason = (f"no layer is sensitive to {fname} on its own (every one stays under {fmt_num(thr)}); the worst is "
                  f"{_short(worst.name, 48)} ({fmt_num(err(worst, first))})")
    else:
        verdict = WARN
        reason = (f"{n_sens} of {len(measured[first])} layers are sensitive to {fname}; the worst is "
                  f"{_short(worst.name, 48)} ({worst.op}, error {fmt_num(err(worst, first))}) — keep it in FP32")
    summary = [
        Row("model", "model", m.name, note=f"{m.task}, {m.architecture}, {m.onnx.name}"),
        Row("layers", "layers measured", len(ranked), note="MatMul / Gemm / Conv, each reduced alone"),
        Row("formats", "formats", formats, text=", ".join(_FMT_NAMES.get(f, f) for f in formats),
            note="fp8/fp4 are simulated (nothing is built)" if set(formats) & {"fp8", "fp4"} else ""),
    ]
    for f in formats:
        es = [err(s, f) for s in measured[f] if err(s, f) is not None and math.isfinite(err(s, f))]
        skipped = len(ranked) - len(measured[f])
        summary.append(Row(f"sensitive_{f}", f"sensitive to {_FMT_NAMES.get(f, f)}", len(sens[f]),
                           text=f"{len(sens[f])} of {len(measured[f])}",
                           note=f"error alone > {fmt_num(thr)}; mean {fmt_num(sum(es) / len(es)) if es else 'n/a'}"
                           + (f"; {skipped} layer(s) cannot take {_FMT_NAMES.get(f, f)} and stay float" if skipped else "")))
    summary += [
        Row("worst_layer", "worst layer", worst.name, text=_short(worst.name, 64), note=f"{worst.op}, error {fmt_num(err(worst, first))}"),
        Row("images", "images", n_images, note=f"from {args.images}"),
        Row("seconds", "time", round(took, 1), text=f"{took:.0f} s", note=ep_note),
    ]
    header = ["rank", "layer", "op"] + [f"err {_FMT_NAMES.get(f, f)}" for f in formats] + ["w-SQNR dB", "outlier"]
    rows = []
    for s in ranked[:args.top]:
        cells = [str(s.rank), _short(s.name), s.op]
        for f in formats:
            e = err(s, f)
            cells.append("-" if e is None else (fmt_num(e) + (" !" if e > thr else "")))
        cells += ["-" if s.weight_sqnr_db is None else f"{s.weight_sqnr_db:.1f}", f"{s.act_outlier_ratio:.1f}"]
        rows.append(cells)
    top = Table(f"Most sensitive layers (top {min(args.top, len(ranked))} of {len(ranked)}, by {fname} error)", header, rows,
                notes=[f"err = distance between the model's outputs with ONLY this layer reduced and the FP32 outputs "
                       f"(0 = the same; for a detector, 1 - matched IoU of the detections as a set). '!' marks > {fmt_num(thr)}.",
                       "w-SQNR = how well the weight alone survives (higher is safer). outlier = largest activation / "
                       "its 99.9th percentile: a large ratio means a few huge values eat the INT8 range."])
    chart_rows = ranked[:min(args.top, 20)]
    charts = [Chart(f"{fname} error per layer, most sensitive first (dashed: --threshold {fmt_num(thr)})",
                    bar_chart([_short(s.name, 44) for s in chart_rows], [err(s, first) for s in chart_rows],
                              mark=thr, mark_label="threshold", log=True), alt="sensitivity bar chart")]
    keep = [s.name for s in sens[first]]
    nxt = []
    if keep:
        nxt.append(f"Keep the {len(keep)} sensitive layer(s) in FP32 and reduce the rest: `visionserve optimize {m.name} "
                   f"--target jetson-orin --images {args.images}` searches that mix for you"
                   + (f" (add --sensitivity {args.save} to reuse these scores)." if args.save else "."))
    else:
        nxt.append(f"Every layer tolerates {fname} alone, but errors add up: `visionserve optimize {m.name} --target "
                   f"jetson-orin --images {args.images}` measures the whole reduced model.")
    nxt.append("A layer's score is a proxy (one layer at a time). Only accuracy on labelled data (optimize --labels) "
               "says whether a reduced model is good enough.")
    if not args.save:
        nxt.append("Save the scores with --save FILE to reuse them in optimize (--sensitivity FILE).")
    notes = []
    if set(formats) & {"fp8", "fp4"}:
        notes.append("FP8 / FP4 are simulated round trips: they show what Thor-class hardware would cost in accuracy, "
                     "not something ONNX Runtime can build or run.")
    details = {"model": m.name, "onnx": str(m.onnx), "formats": formats, "threshold": thr, "images": n_images,
               "calib_method": args.calib_method, "measured_on": ep_note, "seconds": took,
               "sensitive": {f: [s.name for s in sens[f]] for f in formats}, "layers": payload["layers"]}
    return Report(f"visionserve sensitivity {m.name}", verdict, reason, summary, [top], charts, notes, nxt, details)


# ------------------------------------------------------------------------------------------
# optimize
# ------------------------------------------------------------------------------------------

def optimize_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="visionserve optimize", allow_abbrev=False,
                                description="Build and measure reduced-precision variants of an installed model for a "
                                            "target, and recommend one.")
    add_common(p)
    p.add_argument("--target", required=True, choices=sorted(PRESETS), help="deployment target (preset)")
    p.add_argument("--labels", metavar="FILE", help="COCO annotations json: mAP of every variant vs FP32")
    p.add_argument("--label-images", metavar="DIR", help="images of --labels (default: next to the json, or --images)")
    p.add_argument("--max-labels", type=int, default=100, metavar="N", help="cap on labelled images (default 100)")
    p.add_argument("--max-drop", type=float, default=1.0, metavar="P", help="mAP points a variant may lose (default 1.0)")
    p.add_argument("--max-output-err", type=float, default=0.05, metavar="E",
                   help="output error vs FP32 a variant may have (default 0.05); also the mixed search's target")
    p.add_argument("--install", action="store_true", help="install the recommended variant as <model>-<format>")
    p.add_argument("--install-format", metavar="FMT", help="with --install: install this variant instead of the "
                   "recommended one (e.g. fp16 when size matters more than speed); it must be inside the budget")
    p.add_argument("--force", action="store_true", help="with --install: replace an installed <model>-<format>")
    p.add_argument("--tensorrt", action="store_true", help="also measure under the TensorRT EP (flagged)")
    p.add_argument("--sensitivity", metavar="FILE", help="reuse per-layer scores (visionserve sensitivity --save)")
    p.add_argument("--calib-n", type=int, default=32, metavar="N", help="calibration images used (default 32)")
    p.add_argument("--sens-images", type=int, default=8, metavar="N", help="images per layer for sensitivity (default 8)")
    p.add_argument("--requests", type=int, default=20, metavar="N", help="timed requests per variant (default 20)")
    p.add_argument("--keep-dir", metavar="DIR", help="keep the built variants in DIR (default: a temporary dir)")
    p.add_argument("--server-timeout", type=float, default=120.0, help=argparse.SUPPRESS)
    return p


@dataclasses.dataclass
class Variant:
    fmt: str                      # fp32 | fp16 | int8 | mixed | int4
    path: Optional[Path] = None
    size: int = 0
    err_mean: Optional[float] = None
    err_max: Optional[float] = None
    note: str = ""
    error: str = ""               # build failure
    assignment: Optional[dict] = None
    ep: str = "default"           # "default" | "tensorrt"
    lat_p50: Optional[float] = None
    lat_p95: Optional[float] = None
    device: str = ""
    map: Optional[float] = None
    map50: Optional[float] = None
    eval_error: str = ""
    sens_measured: tuple = ()     # mixed: formats measured here because the --sensitivity file lacked them
    warn: str = ""                # a WARN finding for the report (e.g. the mixed candidate was skipped)

    @property
    def label(self) -> str:
        return self.fmt + (" (TensorRT)" if self.ep == "tensorrt" else "")


def complete_scores(m: InstalledModel, scores, ladder: Sequence[str], feeds, method: str, gpu: bool, v: Variant,
                    source: str):
    """Scores covering every format of the mixed `ladder`: the given ones (a --sensitivity file),
    plus the formats they lack measured here and merged in per layer. Returns (scores, the formats
    measured). When measuring is impossible, sets v.warn (a WARN naming the fix) and raises, so the
    mixed candidate is skipped and the other candidates still run."""
    from .precision import parse_formats
    have = {f for s in scores for f in s.errs}
    missing = [f for f in parse_formats(list(ladder)) if f not in have]
    if not missing:
        return scores, ()
    fix = f"`visionserve sensitivity {m.name} --formats {','.join(ladder)} --save FILE`"
    log(f"edge: {source} has no {', '.join(missing)} scores (only {', '.join(sorted(have)) or 'none'}); measuring "
        f"{', '.join(missing)} here for the mixed candidate and reusing the rest")
    try:
        extra, _ = measure(m, feeds, missing, method, gpu)
    except Exception as e:  # noqa: BLE001 — the mixed row says why; the other candidates go on
        v.warn = (f"mixed skipped: {source} has no {', '.join(missing)} scores and measuring them here failed "
                  f"({type(e).__name__}: {e}). Re-run {fix} and pass that file to --sensitivity")
        raise ConvertError(v.warn) from e
    by_name = {s.name: s for s in extra}
    if by_name and not any(s.name in by_name for s in scores):
        v.warn = (f"mixed skipped: the layers in {source} are not this model's ({m.onnx.name}); re-run {fix} for "
                  f"{m.name} and pass that file to --sensitivity")
        raise ConvertError(v.warn)
    merged = []
    for s in scores:
        add = by_name.pop(s.name, None)
        merged.append(dataclasses.replace(s, errs={**s.errs, **(add.errs if add is not None else {})}))
    # Layers the file did not score (e.g. a MatMul INT8 skips) but the new format applies to.
    for i, s in enumerate(sorted(by_name.values(), key=lambda s: s.rank), len(merged) + 1):
        merged.append(dataclasses.replace(s, rank=i))
    return merged, tuple(missing)


def _size(p) -> int:
    return Path(p).stat().st_size if p and Path(p).is_file() else 0


def build_variants(m: InstalledModel, preset: Preset, feeds, args, workdir: Path, scores=None):
    """Build every candidate of the preset; returns (variants, sensitivity scores or None)."""
    from . import precision as P
    src = str(m.onnx)
    out: List[Variant] = [Variant("fp32", path=m.onnx, size=_size(m.onnx), err_mean=0.0, err_max=0.0,
                                  note="the installed model")]
    cmp = (lambda dst: P.compare_models(src, str(dst), feeds))
    method = args.calib_method
    for fmt in preset.candidates:
        v = Variant(fmt)
        t0 = time.time()
        dst = workdir / f"{m.onnx.stem}-{fmt}.onnx"
        try:
            if fmt == "fp16":
                P.convert_fp16(src, dst)
                v.note = "FP16 weights and activations, float32 inputs/outputs"
            elif fmt == "int8":
                P.quantize_int8(src, dst, feeds, method, workdir=workdir)
                v.note = f"INT8 QDQ on every MatMul/Gemm/Conv, {method} calibration on {len(feeds)} image(s)"
            elif fmt == "int4":
                import onnx
                model0 = onnx.load(src)
                P.ensure_node_names(model0)       # the names build_mixed gives the same graph
                ok4 = set(P.int4_eligible(model0))
                names = [n.name for n in model0.graph.node if n.op_type in P.QUANT_OPS]
                del model0
                asg = {n: ("int4" if n in ok4 else "fp32") for n in names}
                P.build_mixed(src, dst, asg, feeds, method, "rtn", 32, fp16_rest=False, workdir=workdir)
                v.note = f"INT4 weight-only on {len(ok4)} MatMul(s) (RTN, block 32); the rest FP32"
            elif fmt == "mixed":
                ladder = list(preset.mixed_formats)
                sfeeds = feeds[:max(1, args.sens_images)]
                if scores is None:
                    scores, _ = measure(m, sfeeds, P.parse_formats(ladder), method, args.gpu)
                else:
                    scores, v.sens_measured = complete_scores(m, scores, ladder, sfeeds, method, args.gpu, v,
                                                              getattr(args, "sensitivity", None) or "the given scores")
                missing = [f for f in ladder if not any(f in s.errs for s in scores)]
                if missing:
                    raise ConvertError(f"the sensitivity scores do not cover {', '.join(missing)}; re-run "
                                       f"`visionserve sensitivity --formats {','.join(ladder)} --save FILE`")

                def build(asg):
                    P.build_mixed(src, dst, asg, feeds, method, "rtn", 32, fp16_rest="fp16" in ladder, workdir=workdir)
                    return dst, cmp(dst)
                asg, _, _, tau, steps = P.search_assignment(scores, ladder, build, args.max_output_err)
                build(asg)          # the search may end on another step's build: rebuild the chosen one
                v.assignment = asg
                cnt = {f: sum(1 for x in asg.values() if x == f) for f in ("int4", "int8", "fp16", "fp32")}
                v.note = ("per layer " + " ".join(f"{f}={c}" for f, c in cnt.items() if c)
                          + f" (sensitivity-driven, target error {fmt_num(args.max_output_err)}, {len(steps)} builds)")
                if v.sens_measured:
                    v.note += (f"; {', '.join(v.sens_measured)} sensitivity measured here, the rest reused from "
                               f"{args.sensitivity}")
            else:
                raise ConvertError(f"unknown candidate {fmt}")
            r = cmp(dst)
            v.path, v.size = dst, _size(dst)
            v.err_mean, v.err_max = r.get("err_mean"), r.get("err_max")
            log(f"edge: {fmt}: {v.size / 1e6:.1f} MB, output error {fmt_num(v.err_mean)} ({time.time() - t0:.0f} s)")
        except Exception as e:  # noqa: BLE001 — one failed candidate is a row, not an abort
            v.error = f"{type(e).__name__}: {e}"[:300]
            log(f"edge: {fmt}: build failed: {v.error}")
        out.append(v)
    return out, scores


# --- installing a variant -------------------------------------------------------------------

_SKIP_COPY = {"sensitivity.json", "optimize-report.json", "convert-report.json"}


def stage_variant(m: InstalledModel, v: Variant, new_name: str, dest_root: Path, provenance: List[str]) -> Path:
    """A model folder <dest_root>/<new_name>: the source folder's files except ONNX graphs, the variant's
    ONNX, and the manifest with the new name, the new file and its sha256. Preprocessing, labels and
    every other setting are the source's, unchanged."""
    import yaml
    from .common import sha256, validate_name
    validate_name(new_name)
    d = dest_root / new_name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in m.dir.iterdir():
        if f.is_file() and f.suffix.lower() != ".onnx" and not f.name.endswith(".onnx.data") \
                and f.name not in _SKIP_COPY and f.name != "manifest.yaml":
            shutil.copy2(f, d / f.name)
    fname = f"{m.onnx.stem}-{v.fmt}.onnx"
    shutil.copyfile(v.path, d / fname)
    doc = json.loads(json.dumps(m.manifest))           # deep copy, plain types
    doc["name"] = new_name
    digest = sha256(d / fname)
    if m.role_key == "files.model":
        doc["files"]["model"] = fname
        if isinstance(doc.get("sha256"), dict):
            doc["sha256"]["model"] = digest
        elif doc.get("sha256"):
            doc["sha256"] = {"model": digest}
    else:
        doc["model_file"] = fname
        if doc.get("sha256") is not None:
            doc["sha256"] = digest
    header = ["# Generated by `visionserve optimize` from " + m.name + " on " + _dt.date.today().isoformat() + "."]
    header += ["# " + ln for ln in provenance]
    header += ["# Same preprocessing, labels and postprocessing as " + m.name + "; only the ONNX file differs.", ""]
    (d / "manifest.yaml").write_text("\n".join(header) + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    return d


def install_variant(staged: Path, models_dir: Path, force: bool) -> None:
    """`visionserve pull <folder>`: the registry's own validation (license, architecture, files)."""
    from .common import VISIONSERVE_BIN
    cmd = [VISIONSERVE_BIN, "pull", str(staged), "--models", str(models_dir)] + (["--force"] if force else [])
    log("$ " + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise ConvertError(f"visionserve refused {staged.name} (see above); nothing was installed")


# --- measuring through a temporary server -----------------------------------------------------

def _eval_set(args):
    if not args.labels:
        return None
    from .evaluate import load_eval
    if not Path(args.labels).is_file():
        raise ConvertError(f"--labels {args.labels}: no such file (a COCO annotations json)")
    images = args.label_images
    ev = None
    try:
        ev = load_eval(args.labels, images, args.max_labels)
    except ValueError:
        if images is None:
            ev = load_eval(args.labels, args.images, args.max_labels)
        else:
            raise
    if ev.kind != "detection":
        raise ConvertError("--labels must be a COCO detection json")
    return ev


def _served_map(client, name, ev, labels):
    from .evaluate import coco_map, map_labels
    mapping, unmapped, _ = map_labels(labels, ev.categories)
    if not mapping:
        raise ConvertError("no model class maps to a category of --labels by name")
    res = []
    for path, img_id in ev.items:
        r = client.predict(name, path, max_grasps_per_object=None)
        for d in r.detections:
            if d.cls in mapping:
                res.append({"image_id": img_id, "category_id": mapping[d.cls], "bbox": [float(x) for x in d.bbox],
                            "score": float(d.conf)})
    cats = sorted(set(mapping.values()))
    return coco_map(ev.coco, res, [i for _, i in ev.items], cats), len(cats)


def _latency(client, name, images: List[Path], n: int):
    from .metrics import percentile_ms
    for p in images[:2]:                                  # warm-up: allocations, kernel selection
        client.predict(name, p, max_grasps_per_object=None)
    ms, dev = [], ""
    for i in range(max(1, n)):
        r = client.predict(name, images[i % len(images)], max_grasps_per_object=None)
        ms.append(float(r.duration_ms))
        dev = r.device or dev
    return percentile_ms(ms), dev


def measure_served(m: InstalledModel, variants: List[Variant], args, workdir: Path, ev, ep_env: Optional[str],
                   label: str) -> Optional[str]:
    """Install every built variant into a scratch registry, start a temporary server on it and measure
    server-side latency (and mAP with --labels). Returns an error string, or None."""
    from .constants import IMAGE_EXT
    from .serverctl import start_temporary
    reg = workdir / f"registry-{label}"
    reg.mkdir(parents=True, exist_ok=True)
    names = {}
    for v in variants:
        if v.path is None or v.error:
            continue
        nm = f"{m.name}-optimize-{v.fmt}"
        stage_variant(m, v, nm, reg, [f"scratch copy for measurement ({v.fmt})"])
        names[v.fmt] = nm
    env = dict(os.environ)
    env.pop("VISIONSERVE_EP", None)
    env.pop("VISIONSERVE_TENSORRT", None)
    if ep_env:
        env["VISIONSERVE_EP"] = ep_env
    h = start_temporary(reg, workdir=workdir, log=log, env=env, timeout=args.server_timeout)
    if h is None:
        return "could not start a temporary VisionServe server (see the log above)"
    imgs = sorted(f for f in Path(args.images).iterdir() if f.suffix.lower() in IMAGE_EXT)[:16]
    try:
        client = h.client
        for v in variants:
            nm = names.get(v.fmt)
            if nm is None:
                continue
            try:
                log(f"edge: {v.label}: latency on this host ({args.requests} requests) ...")
                pct, v.device = _latency(client, nm, imgs, args.requests)
                v.lat_p50, v.lat_p95 = pct.get("p50_ms"), pct.get("p95_ms")
                if ev is not None:
                    log(f"edge: {v.label}: mAP on {len(ev.items)} labelled image(s) ...")
                    (v.map, v.map50), _ = _served_map(client, nm, ev, m.labels)
            except Exception as e:  # noqa: BLE001 — one variant failing to serve is a row
                v.eval_error = f"{type(e).__name__}: {e}"[:300]
                log(f"edge: {v.label}: {v.eval_error}")
    finally:
        h.close()
    return None


# --- the decision ---------------------------------------------------------------------------------

def _device_class(dev: str) -> str:
    if not dev:
        return "unknown"
    return "cpu" if dev == "cpu" else "gpu"


def decide(variants: List[Variant], preset: Preset, args, have_labels: bool):
    """-> (recommended Variant or None, rule text, {label: status text}, TensorRT alternative or None).

    A variant is inside the budget when its output error <= --max-output-err and (with --labels) its
    mAP drop vs FP32 <= --max-drop. TensorRT rows need --labels (accuracy under TensorRT itself).
    The recommendation comes from the rows of the target's DEFAULT EP chain (TensorRT is opt-in there);
    TensorRT rows (--tensorrt) give a separate, flagged alternative (the 4th return value).
    Among the rows inside the budget: the fastest MEASURED one when this host measured on the target's EP class (gpu for
    the Jetson / cuda presets, cpu for cpu) — two within 10 % count as a tie and the smaller file
    wins — else the preset's expected-speed order, then size. FP32 is the answer when nothing else
    fits."""
    base = next(v for v in variants if v.fmt == "fp32" and v.ep == "default")
    status, ok = {}, []
    for v in variants:
        why = []
        if v.error:
            status[v.label] = "build failed"
            continue
        if v.eval_error and v.lat_p50 is None:
            status[v.label] = "could not be served"
            continue
        if v.fmt != "fp32":
            if v.err_mean is None or not math.isfinite(v.err_mean) or v.err_mean > args.max_output_err:
                why.append(f"error {fmt_num(v.err_mean)} > {fmt_num(args.max_output_err)}")
            if have_labels:
                if v.map is None or base.map is None:
                    why.append("mAP not measured")
                elif base.map - v.map > args.max_drop:
                    why.append(f"mAP −{base.map - v.map:.2f} > {fmt_num(args.max_drop)}")
        if v.ep == "tensorrt" and not have_labels:
            why.append("TensorRT: accuracy unmeasured (needs --labels)")
        if why:
            status[v.label] = "over budget: " + "; ".join(why)
        else:
            status[v.label] = "within budget"
            ok.append(v)
    pick, rule = _choose([v for v in ok if v.ep == "default"], preset)
    if pick is not None:
        status[pick.label] = "recommended ✓"
    trt_pick = None
    trt_ok = [v for v in ok if v.ep == "tensorrt"]
    if trt_ok:
        trt_pick, _ = _choose(trt_ok, preset)
        if trt_pick is not None:
            status[trt_pick.label] = "best under TensorRT (opt-in alternative)"
    return pick, rule, status, trt_pick


def _choose(ok: List[Variant], preset: Preset):
    """The rule of decide() over one EP's rows that are inside the budget -> (Variant or None, rule).

    Latency measured here on the target's EP class: the fastest row, FP32 INCLUDED (a reduced variant that
    is slower than FP32 is not recommended for speed); rows within 10 % of the fastest are a tie and the
    smallest file wins. Otherwise speed is unknown: the smallest reduced variant inside the budget."""
    if not ok:
        return None, "nothing is inside the budget"
    measured_class = {_device_class(v.device) for v in ok if v.lat_p50 is not None}
    comparable = measured_class == {preset.host_ep} and all(v.lat_p50 is not None for v in ok)
    if comparable:
        best = min(v.lat_p50 for v in ok)
        near = [v for v in ok if v.lat_p50 <= best * 1.10]
        return min(near, key=lambda v: (v.size, v.lat_p50)), (
            f"fastest measured on this host's {preset.host_ep.upper()} (the target's EP class), FP32 included; "
            "within 10 % of the fastest the smaller file wins")
    reduced = [v for v in ok if v.fmt != "fp32"] or ok
    return min(reduced, key=lambda v: v.size), (
        "the smallest variant inside the budget: this host's latency is not comparable with the target's "
        f"(measured on {', '.join(sorted(measured_class)) or 'nothing'}, the target runs on {preset.host_ep}), so "
        "speed is unknown until you bench on the device")


def run_optimize(args) -> Report:
    from . import precision as P
    preset = PRESETS[args.target]
    for k in ("max_drop", "max_output_err"):
        v = getattr(args, k)
        if not (math.isfinite(v) and v > 0):
            raise ConvertError(f"--{k.replace('_', '-')} must be a finite number > 0")
    if args.requests < 1 or args.calib_n < 1 or args.sens_images < 1 or args.max_labels < 1:
        raise ConvertError("--requests, --calib-n, --sens-images and --max-labels must be at least 1")
    if args.install_format and (not args.install or args.install_format not in preset.candidates):
        raise ConvertError(f"--install-format needs --install and one of the {args.target} candidates: "
                           f"{', '.join(preset.candidates)}")
    if args.install_format and (Path(args.models) / f"{args.model}-{args.install_format}").exists() and not args.force:
        raise ConvertError(f"{args.model}-{args.install_format} is already installed in {args.models}; pass --force")
    if args.tensorrt and not args.target.startswith("jetson") and args.target != "cuda":
        raise ConvertError("--tensorrt applies to the jetson-* and cuda targets")
    if not Path(args.images).is_dir():
        raise ConvertError(f"--images {args.images}: expected a directory of photos")
    m = find_model(args.models, args.model)
    ev = _eval_set(args)
    if ev is not None and m.task not in ("detection",):
        raise ConvertError(f"--labels measures detection mAP; {m.name} is a {m.task} model")
    if ev is not None and not m.labels:
        raise ConvertError(f"{m.name} has no labels file: --labels cannot map its classes")
    scores = None
    if args.sensitivity:
        scores = scores_from_json(json.loads(Path(args.sensitivity).read_text()))
        log(f"edge: reusing {len(scores)} layer scores from {args.sensitivity}")
    t0 = time.time()
    tmp = None
    if args.keep_dir:
        work = Path(args.keep_dir)
        work.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.TemporaryDirectory(prefix="vsoptimize-")
        work = Path(tmp.name)
    try:
        feeds = calibration_feeds(m, args.images, args.calib_n)
        variants, scores = build_variants(m, preset, feeds, args, work, scores)
        # latency (+ accuracy) through a temporary server on a scratch registry
        ep_default = None if args.gpu else "cpu"
        err = measure_served(m, variants, args, work, ev, ep_default, "default")
        server_error = err
        if args.tensorrt:
            trt = [dataclasses.replace(v, ep="tensorrt", lat_p50=None, lat_p95=None, device="", map=None, map50=None,
                                       eval_error="") for v in variants if not v.error]
            e2 = measure_served(m, trt, args, work, ev, "tensorrt,cuda", "tensorrt")
            server_error = server_error or e2
            variants += trt
        pick, rule, status, trt_pick = decide(variants, preset, args, ev is not None)
        installed, install_note = None, ""
        chosen = pick
        if args.install and args.install_format:
            chosen = next((v for v in variants if v.fmt == args.install_format and v.ep == "default"), None)
            if chosen is None or status.get(chosen.label) not in ("within budget", "recommended ✓"):
                install_note = (f"--install-format {args.install_format}: not installed, it is not inside the budget "
                                f"({status.get(chosen.label) if chosen else 'not built'})")
                chosen = None
        if args.install and chosen is not None and chosen.fmt != "fp32":
            pick_ = pick
            pick = chosen
            new = f"{m.name}-{pick.fmt}"
            prov = [f"precision: {pick.fmt} ({pick.note})", f"target preset: {preset.target}",
                    f"output error vs FP32: mean {fmt_num(pick.err_mean)}, max {fmt_num(pick.err_max)} on "
                    f"{len(feeds)} image(s) of {Path(args.images).name}"]
            base = next(v for v in variants if v.fmt == "fp32" and v.ep == "default")
            if pick.map is not None and base.map is not None:
                prov.append(f"mAP {pick.map:.2f} vs FP32 {base.map:.2f} on {len(ev.items)} labelled image(s)")
            staged = stage_variant(m, pick, new, work / "install", prov)
            (staged / "optimize-report.json").write_text(json.dumps(
                {"source": m.name, "precision": pick.fmt, "target": preset.target, "note": pick.note,
                 "output_err_mean": pick.err_mean, "output_err_max": pick.err_max, "map": pick.map,
                 "map_fp32": base.map, "assignment": pick.assignment}, indent=1) + "\n")
            dest = Path(args.models) / new
            if dest.exists() and not args.force:
                raise ConvertError(f"{new} is already installed in {args.models}; pass --force to replace it")
            install_variant(staged, Path(args.models), args.force)
            installed = new
            pick = pick_
        rep = optimize_report(m, preset, variants, pick, rule, status, args, ev, len(feeds), time.time() - t0,
                              installed, server_error, trt_pick)
        for v in variants:
            if v.warn and v.ep == "default":
                rep.notes.append((WARN, v.warn))
            elif v.sens_measured and v.ep == "default":
                rep.notes.append(f"{args.sensitivity} has no {', '.join(v.sens_measured)} scores: the mixed candidate "
                                 f"measured them here ({len(feeds[:max(1, args.sens_images)])} image(s)) and reused "
                                 "the file's other scores. Save both with `visionserve sensitivity --formats "
                                 f"{','.join(preset.mixed_formats)} --save FILE` to skip that next time.")
        if install_note:
            rep.notes.append((WARN, install_note))
            if rep.verdict == PASS:
                rep.verdict, rep.reason = WARN, install_note
        return rep
    finally:
        if tmp is not None:
            tmp.cleanup()


def optimize_report(m, preset, variants, pick, rule, status, args, ev, n_feeds, took, installed, server_error,
                    trt_pick=None) -> Report:
    import platform
    base = next(v for v in variants if v.fmt == "fp32" and v.ep == "default")
    host = f"{platform.node() or 'this host'} ({platform.machine()})"
    dev_seen = sorted({v.device for v in variants if v.device})
    lat_where = f"{host}, {', '.join(dev_seen) or 'not measured'}"
    rows = []
    hl = None
    for i, v in enumerate(variants):
        if pick is v:
            hl = i
        size = f"{v.size / 1e6:.1f} MB" + (f" ({base.size / v.size:.1f}× smaller)" if v.size and v.fmt != "fp32" else "")
        err = "-" if v.error else f"{fmt_num(v.err_mean)} / {fmt_num(v.err_max)}"
        if ev is None:
            acc = "not measured"
        elif v.map is None:
            acc = "n/a"
        else:
            acc = f"{v.map:.2f}" + ("" if v.fmt == "fp32" and v.ep == "default" or base.map is None
                                   else f" ({v.map - base.map:+.2f})")
        lat = "-" if v.lat_p50 is None else f"{fmt_num(v.lat_p50)} / {fmt_num(v.lat_p95)} ms" + (f" ({v.device})" if v.device else "")
        st = status.get(v.label, "")
        if v.error:
            st = "build failed: " + v.error[:120]
        elif v.eval_error:
            st = (st + "; " if st else "") + "serving: " + v.eval_error[:120]
        rows.append([("✓ " if pick is v else "") + v.label, size, err, acc, lat, st])
    have_labels = ev is not None
    table = Table(f"Variants for {preset.target}", ["variant", "file size", "output error mean / max", "mAP (Δ vs FP32)",
                                                     "latency p50 / p95 (THIS host)", "status"], rows, highlight=hl,
                  notes=[f"Output error: distance to the FP32 outputs on {n_feeds} image(s) of --images (0 = the same "
                         "detections; 1 = nothing in common). Budget: " + fmt_num(args.max_output_err) + ".",
                         ("mAP: COCO mAP@[.5:.95] of the SERVED model at its conf_threshold on " + str(len(ev.items))
                          + f" labelled image(s); budget: drop <= {fmt_num(args.max_drop)} points." if have_labels else
                          "mAP: not measured (pass --labels FILE.json)."),
                         f"Latency: server-side duration_ms on {lat_where}. It is NOT the {preset.target}'s latency: "
                         f"run `visionserve bench {m.name}-<format>` on the device.",
                         "Recommendation rule: " + rule + "."])
    variants_ok = [v for v in variants if not v.error]
    charts = [Chart("File size (MB)", bar_chart([v.label for v in variants_ok], [v.size / 1e6 for v in variants_ok],
                                                unit="MB", highlight=[i for i, v in enumerate(variants_ok) if v is pick])),
              Chart("Output error vs FP32 (dashed: --max-output-err)",
                    bar_chart([v.label for v in variants_ok if v.fmt != "fp32"],
                              [v.err_mean for v in variants_ok if v.fmt != "fp32"], mark=args.max_output_err,
                              highlight=[i for i, v in enumerate([x for x in variants_ok if x.fmt != "fp32"]) if v is pick]))]
    if have_labels:
        charts.append(Chart("mAP of the served model", bar_chart([v.label for v in variants_ok],
                                                                [v.map for v in variants_ok],
                                                                highlight=[i for i, v in enumerate(variants_ok) if v is pick])))
    built = [v for v in variants if v.fmt != "fp32" and not v.error]
    # verdict
    if base.lat_p50 is None and server_error:
        verdict, reason = FAIL, f"the variants could not be measured: {server_error}"
    elif not built:
        verdict, reason = FAIL, "no variant could be built: " + "; ".join(f"{v.fmt}: {v.error[:80]}" for v in variants if v.error)
    elif pick is None or pick.fmt == "fp32":
        verdict = WARN
        smaller = [v for v in variants if v.ep == "default" and v.fmt != "fp32"
                   and status.get(v.label) == "within budget"]
        if smaller and pick is not None and pick.lat_p50:
            alt = min(smaller, key=lambda v: v.size)
            reason = (f"keep FP32 for speed: no reduced variant is as fast on this host's {preset.host_ep.upper()} "
                      f"({m.name}-{alt.fmt} is {base.size / max(alt.size, 1):.1f}× smaller but "
                      f"{(alt.lat_p50 or 0) / pick.lat_p50:.1f}× slower here); use it only if size matters, and bench "
                      "both on the device")
        else:
            reason = (f"no variant stays inside the budget (output error <= {fmt_num(args.max_output_err)}"
                      + (f", mAP drop <= {fmt_num(args.max_drop)}" if have_labels else "") + "): keep FP32")
    else:
        parts = [f"{base.size / pick.size:.1f}× smaller", f"output error {fmt_num(pick.err_mean)}"]
        if pick.map is not None and base.map is not None:
            parts.append(f"mAP {pick.map - base.map:+.2f}")
        if pick.lat_p50 is not None and base.lat_p50 is not None:
            parts.append(f"{fmt_num(pick.lat_p50)} vs {fmt_num(base.lat_p50)} ms FP32 on this host ({pick.device})")
        verdict = PASS if have_labels else WARN
        reason = f"recommended {m.name}-{pick.fmt}: " + ", ".join(parts)
        if not have_labels:
            reason += "; accuracy NOT measured (pass --labels)"
        elif "fastest measured" in rule and any(v.ep == "default" and v.lat_p50 and v.lat_p95
                                                 and v.lat_p95 > 1.5 * v.lat_p50 for v in variants):
            verdict = WARN
            reason += "; latency here was noisy, so the speed ranking may be noise (see Notes)"
    summary = [
        Row("model", "model", m.name, note=f"{m.task}, {m.architecture}, {m.onnx.name} ({base.size / 1e6:.1f} MB)"),
        Row("target", "target", preset.target, note=preset.hardware),
        Row("recommended", "recommended", None if pick is None else f"{m.name}-{pick.fmt}",
            text="keep FP32" if pick is None or pick.fmt == "fp32" else f"{m.name}-{pick.fmt}"
            + (" (TensorRT)" if pick.ep == "tensorrt" else ""), note=None if pick is None else pick.note or ""),
        Row("budget", "budget", {"max_output_err": args.max_output_err, "max_drop": args.max_drop if have_labels else None},
            text=f"output error <= {fmt_num(args.max_output_err)}" + (f", mAP drop <= {fmt_num(args.max_drop)}" if have_labels else ""),
            note="" if have_labels else "accuracy not measured: pass --labels FILE.json"),
        Row("latency_host", "latency measured on", lat_where,
            note=(f"this host, not the {preset.target}" if preset.target.startswith("jetson")
                  else "this host: another machine of the same kind can differ")),
        Row("tensorrt_alternative", "under TensorRT (opt-in)",
            None if trt_pick is None else f"{m.name}-{trt_pick.fmt}",
            text="-" if trt_pick is None else f"{m.name}-{trt_pick.fmt}: {fmt_num(trt_pick.lat_p50)} ms here, mAP "
            + ("n/a" if trt_pick.map is None or base.map is None else f"{trt_pick.map - base.map:+.2f}"),
            note="" if trt_pick is None else "serve with --tensorrt; TensorRT measured 6.8 mAP lower on GroundingDINO "
            "(BUGS_TO_FIX.md #3): this row's accuracy was measured under TensorRT"),
        Row("installed", "installed", installed, text=installed or "no", note="" if installed else
            ("pass --install" if pick is not None and pick.fmt != "fp32" else "")),
        Row("seconds", "time", round(took, 1), text=f"{took / 60:.1f} min"),
    ]
    notes = [f"Preset {preset.target}: {preset.software}. Default EP there: {preset.ep}. Verified: {preset.verified}."]
    notes += list(preset.notes)
    if any(v.ep == "tensorrt" for v in variants):
        notes.append("TensorRT rows (--tensorrt): measured under the TensorRT EP on this host, which runs INT8 QDQ on "
                     "the INT8 tensor cores. TensorRT is opt-in on the target and measured 6.8 mAP lower on "
                     "GroundingDINO (BUGS_TO_FIX.md #3), so these rows give a flagged alternative, never the "
                     "recommendation, and only with --labels (accuracy measured under TensorRT itself).")
    noisy = [v.label for v in variants if v.lat_p50 and v.lat_p95 and v.lat_p95 > 1.5 * v.lat_p50]
    if noisy:
        notes.append(f"Latency on this host was noisy (p95 > 1.5 × p50 for {', '.join(noisy)}): a shared or busy "
                     "GPU/CPU. The speed ranking, and so the recommendation, may be noise; re-run on an idle machine "
                     "with more --requests, and in any case measure on the device.")
    if not args.gpu and preset.host_ep == "gpu":
        notes.append("Latency was measured on this host's CPU (no --gpu), which says nothing about a GPU target: the "
                     "recommendation is the smallest variant inside the budget. Pass --gpu on a machine with an NVIDIA "
                     "GPU to measure on the CUDA EP.")
    nxt = []
    deploy = installed or (f"{m.name}-{pick.fmt}" if pick is not None and pick.fmt != "fp32" else None)
    if installed:
        nxt.append(f"Copy the model to the device: `scp -r {Path(args.models) / installed} <jetson>:~/.visionserve/models/` "
                   "(or the registry its server uses).")
    elif deploy:
        nxt.append(f"Install it: re-run with --install (it becomes {deploy}), then copy that folder to the device.")
    else:
        smaller = [v for v in variants if v.ep == "default" and v.fmt != "fp32" and status.get(v.label) == "within budget"]
        if smaller:
            alt = min(smaller, key=lambda v: v.size)
            deploy = f"{m.name}-{alt.fmt}"
            nxt.append(f"If size matters more than speed: re-run with --install --install-format {alt.fmt} (it becomes "
                       f"{deploy}).")
    if deploy:
        nxt.append(f"On the device: `visionserve bench {deploy} --images DIR` and `visionserve bench {m.name} --images DIR` "
                   "— only that comparison says how much faster it is there.")
    if not have_labels:
        nxt.append("Measure accuracy before deploying: re-run with --labels instances.json (COCO format, 100+ images like "
                   "the deployment's).")
    if args.target == "jetson-thor":
        nxt.append("Thor is untested: on the device, check that bench reports gpu:0 (not cpu) and that outputs match FP32 "
                   "(edge guide, 'Thor checklist').")
    details = {"model": m.name, "target": preset.target, "preset": dataclasses.asdict(preset), "images": str(args.images),
               "calibration_images": n_feeds, "labels": args.labels, "label_images": len(ev.items) if ev else 0,
               "max_output_err": args.max_output_err, "max_drop": args.max_drop, "rule": rule,
               "recommended": None if pick is None else {"format": pick.fmt, "ep": pick.ep},
               "installed": installed, "latency_host": lat_where, "seconds": took,
               "variants": [{"format": v.fmt, "ep": v.ep, "size_bytes": v.size, "err_mean": v.err_mean,
                             "err_max": v.err_max, "map": v.map, "map50": v.map50, "latency_p50_ms": v.lat_p50,
                             "latency_p95_ms": v.lat_p95, "device": v.device, "status": status.get(v.label, ""),
                             "note": v.note, "error": v.error or v.eval_error or None} for v in variants]}
    return Report(f"visionserve optimize {m.name} --target {preset.target}", verdict, reason, summary, [table], charts,
                  notes, nxt, details)


# ------------------------------------------------------------------------------------------
# entry point
# ------------------------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    tool = argv[0] if argv else ""
    if tool not in TOOLS:
        log(f"error: expected one of {', '.join(TOOLS)}")
        return 2
    parser = sensitivity_parser() if tool == "sensitivity" else optimize_parser()
    try:
        args = parser.parse_args(argv[1:])
    except SystemExit as e:                 # argparse: --help exits 0, a usage error 2
        return int(e.code or 0)
    try:
        with contextlib.redirect_stdout(sys.stderr):   # onnxruntime's quantizer print()s progress
            rep = run_sensitivity(args) if tool == "sensitivity" else run_optimize(args)
    except ConvertError as e:
        log(f"error: {e}")
        return 2
    except Exception as e:  # noqa: BLE001 — a measurement that broke is a FAIL with the reason
        import traceback
        log(traceback.format_exc(limit=6))
        rep = Report(f"visionserve {tool} {getattr(args, 'model', '')}", FAIL, f"{type(e).__name__}: {e}"[:300],
                     next_steps=["Re-run with the same flags; if it fails again, report the traceback above."])
    if getattr(args, "report", None):
        rep.write_html(args.report)
        log(f"report: {args.report}")
    sys.stdout.write(rep.json_text() + "\n" if args.json else rep.text())
    sys.stdout.flush()
    return rep.exit_code
