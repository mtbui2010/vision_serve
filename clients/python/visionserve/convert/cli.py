"""`visionserve-convert <format> <source> --name NAME [...]` — the converter image's entrypoint.

Every format module exposes:
    HELP: str
    add_arguments(parser)                      family-specific flags
    convert(args, workdir: Path) -> [Bundle]   export + parity; NO installing

The CLI owns what is common: the license gate, the staging dir, the contract check, the
hand-off to `visionserve pull <folder>`, and the verification that follows:

    export -> tier A (ONNX vs framework parity, in the family) -> install
           -> tier B1 (server preprocessing vs the reference preprocessing, real images)
           -> tier B2 (server outputs vs the original framework pipeline)
           -> tier C  (--eval: mAP / top-1 on your labelled data, reference vs served)
           -> speed   (--bench: framework / ONNX / server p50, p95)
           -> report: table on stderr, <registry>/<name>/convert-report.json, manifest header comments.
    A FAILED check uninstalls the model (unless --keep-on-fail) and exits 1.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import tempfile
from pathlib import Path

from .common import MODELS_DIR, ConvertError, install, log

# format name -> module (imported lazily: each pulls in a heavy framework)
FORMATS = {
    "rfdetr": "visionserve.convert.families.rfdetr",
    "hf": "visionserve.convert.families.hf",
    "torchscript": "visionserve.convert.families.torch_generic",
    "pytorch": "visionserve.convert.families.torch_generic",
    "tensorflow": "visionserve.convert.families.tf",
    "keras": "visionserve.convert.families.tf",
    "tflite": "visionserve.convert.families.tf",
}

# Generic tasks for formats that carry no architecture of their own (TorchScript, PyTorch+script,
# TensorFlow, Keras, TFLite). Each maps to a Go architecture whose I/O contract is checked.
TASK_PRESETS = {
    #  task            architecture    manifest task     postprocess block
    "classification": ("efficientnet", "classification", {"type": "classification", "max_detections": 5}),
    "detection":      ("rt-detr",      "detection",      {"type": "rt-detr", "box_format": "cxcywh",
                                                          "conf_threshold": 0.5, "max_detections": 300}),
    "depth":          ("midas",        "depth",          {"type": "depth"}),
    "embed":          ("clip",         "embed",          {"type": "embed"}),
}


def add_generic_arguments(p: argparse.ArgumentParser) -> None:
    """Flags for formats whose checkpoint does not say what the model IS."""
    p.add_argument("--task", required=True, choices=sorted(TASK_PRESETS),
                   help="what the model does; selects the VisionServe architecture and its I/O contract: "
                        "classification = one [1,C] logits output (softmax); detection = DETR-style, NMS-free, "
                        "logits [1,Q,C] (sigmoid) + boxes [1,Q,4] cxcywh normalised; depth = one [1,H,W] "
                        "output; embed = one [1,D] embedding")
    p.add_argument("--input", required=True, metavar="WxH", help="input resolution, e.g. 224x224")
    p.add_argument("--mean", default="0.485,0.456,0.406", help="per-channel mean after /255 (default ImageNet)")
    p.add_argument("--std", default="0.229,0.224,0.225", help="per-channel std after /255 (default ImageNet); "
                   "a graph that expects raw 0-255 pixels: --mean 0,0,0 --std 0.00392157,0.00392157,0.00392157")
    p.add_argument("--letterbox", action="store_true", help="aspect-preserving resize + pad (default: squash) — "
                   "use it ONLY if the model was trained that way")


def parse_wxh(s: str):
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise ConvertError(f"--input must look like 224x224, got {s!r}")


def parse_floats(s: str):
    xs = [float(v) for v in s.split(",")]
    if len(xs) != 3:
        raise ConvertError(f"expected 3 comma-separated values, got {s!r}")
    return xs


def read_labels(path):
    if not path:
        return None
    lines = [l.strip() for l in Path(path).read_text().splitlines()]
    return [l for l in lines if l]


def generic_bundle(args, onnx_path, labels=None, notes=None):
    """Bundle for a format that carries no architecture: everything comes from --task/--input/..."""
    from .common import Bundle, resolve_license
    arch, task, post = TASK_PRESETS[args.task]
    w, h = parse_wxh(args.input)
    return Bundle(name=args.name, task=task, architecture=arch,
                  license=resolve_license(args.license, None, args.source),
                  width=w, height=h, onnx={"model": str(onnx_path)}, letterbox=args.letterbox,
                  mean=parse_floats(args.mean), std=parse_floats(args.std), postprocess=dict(post),
                  labels=labels if labels is not None else read_labels(args.labels),
                  notes=notes or [f"{args.format} {Path(args.source).name}"])


def build_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False: the Docker wrapper (`visionserve convert`, Go) mounts the paths of the flags
    # it knows by their FULL name; an abbreviation (--imag DIR) would reach here unmounted.
    ap = argparse.ArgumentParser(prog="visionserve convert", allow_abbrev=False,
                                 description="Convert a checkpoint to ONNX + manifest and install it into the "
                                             "VisionServe model registry.")
    sub = ap.add_subparsers(dest="format", required=True, metavar="FORMAT")
    for fmt, modname in FORMATS.items():
        p = sub.add_parser(fmt, help=_help(modname, fmt), allow_abbrev=False)
        p.add_argument("source", help="checkpoint file or directory (inside the container: under /in)")
        p.add_argument("--name", required=True, help="registry name of the installed model (letters, digits, "
                                                       "'.', '_', '-'; it becomes a directory under --models)")
        p.add_argument("--license", help="license of the ORIGINAL model (Apache-2.0, MIT, BSD-3-Clause, "
                                         "BSD-2-Clause). Required unless the checkpoint declares one.")
        p.add_argument("--labels", help="class names, one per line, in the model's output order")
        p.add_argument("--force", action="store_true", help="replace an installed model of the same name")
        p.add_argument("--dry-run", action="store_true", help="convert + check, print the manifest, install nothing")
        p.add_argument("--tolerance", type=float, default=1e-3,
                       help="max |onnx - original| / max(1, |original|) accepted by the parity check")
        p.add_argument("--opset", type=int, default=17, help="ONNX opset (17 is what VisionServe's ORT builds run)")
        p.add_argument("--models", default=str(MODELS_DIR), help="registry directory (default $VISIONSERVE_MODELS)")
        add_verify_arguments(p)
        try:
            _family(modname).add_arguments(p, fmt)
        except ImportError as e:
            # The image keeps TensorFlow and PyTorch in separate venvs (their numpy/protobuf pins
            # conflict); the entrypoint routes each format to the right one. A family whose
            # framework is absent in THIS venv just isn't usable here.
            p.set_defaults(_unavailable=f"{fmt}: {e}")
    return ap


def add_verify_arguments(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("verification (tiers B/C, speed) — run against a VisionServe server after install")
    g.add_argument("--images", metavar="DIR", help="real images for tier B (B1 preprocessing, B2 outputs); default: "
                   "up to --verify images of --eval, else B1 uses a synthetic image and B2 is skipped")
    g.add_argument("--verify", type=int, default=8, metavar="N", help="images used for tier B (default 8)")
    g.add_argument("--eval", metavar="PATH", help="tier C: a COCO json (detection; also a dir with "
                   "_annotations.coco.json) or an ImageFolder dir (classification): mAP / top-1 of the original "
                   "framework vs the served model")
    g.add_argument("--eval-images", metavar="DIR", help="image dir for a COCO --eval json (default: found next to it)")
    g.add_argument("--eval-max", type=int, default=200, metavar="N", help="cap on --eval images (default 200)")
    g.add_argument("--max-map-drop", type=float, default=1.0, metavar="POINTS",
                   help="tier C FAILs (and uninstalls) when served accuracy is more than this many points "
                        "(mAP, or top-1 %%) below the reference, and WARNs above half of it (default 1.0: "
                        "WARN > 0.5, FAIL > 1.0). Resize-filter differences between Go and the framework "
                        "already cost ~0.5 and the reference itself moves ~0.1 between CPU and GPU, while "
                        "real preprocessing bugs cost several points (letterbox -7.35, token padding -4.3)")
    g.add_argument("--bench", action="store_true", help="latency of the framework model, the ONNX model and a "
                   "whole server request (p50/p95)")
    g.add_argument("--bench-iters", type=int, default=50, metavar="N", help="timed runs per level (default 50)")
    g.add_argument("--bench-warmup", type=int, default=3, metavar="N", help="warm-up runs per level (default 3)")
    g.add_argument("--reference-script", metavar="FILE.py",
                   help="your training-time preprocessing: defines preprocess(pil_image) -> CHW float32 array, "
                        "optionally predict(pil_image) -> detections [{cls, conf, bbox:[x,y,w,h]}] / class "
                        "probabilities. The strongest B1 check; required for a faithful check of TorchScript / "
                        "PyTorch / TensorFlow models, which carry no official preprocessing")
    g.add_argument("--prompt", help="text prompt for open-vocabulary models (default: the --eval category names, "
                   "else 'object.')")
    g.add_argument("--server", metavar="URL", help="VisionServe server to verify against (default "
                   "$VISIONSERVE_HOST or http://localhost:11435). It must serve THIS --models registry; otherwise "
                   "a temporary local server is started on it")
    g.add_argument("--no-server", action="store_true", help="skip every server-side check (tiers B/C, server speed)")
    g.add_argument("--device", default="auto", help="device for the reference framework model and its speed "
                   "level: auto (cuda if visible, else cpu), cpu, cuda, cuda:N")
    g.add_argument("--keep-on-fail", action="store_true", help="keep the installed model when a check FAILS")
    g.add_argument("--threshold", action="append", metavar="KEY=VALUE",
                   help="override a pass/warn/fail threshold, e.g. b1_mean_fail=10 (repeatable; keys in "
                        "convert/report.py DEFAULT_THRESHOLDS)")


def _family(modname):
    return importlib.import_module(modname)


def _help(modname, fmt):
    try:
        return _family(modname).help_for(fmt)
    except Exception:  # a family whose framework is missing still lists itself
        return fmt


class _InstallTxn:
    """Makes `install + verify` all-or-nothing for the registry.

    Before the Go binary installs, every registry directory this run is about to REPLACE (--force)
    is moved aside to <models>/.convert-backup/<name> (a rename: no copy of GB-sized weights). On a
    FAILED check, a refused install of a later bundle (siglip/clip image + text), an exception or
    Ctrl-C, rollback() removes what this run installed and moves the previous versions back; a
    model whose directory existed and was NOT ours to replace is never touched. commit() drops the
    backups."""

    def __init__(self, models_dir: Path, names, force: bool):
        import time
        self.models_dir = Path(models_dir)
        self.names = list(names)
        self.backups = {}
        self.preexisting = {n for n in self.names if (self.models_dir / n).exists()}
        if not force:
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        root = self.models_dir / ".convert-backup"
        for n in sorted(self.preexisting):
            root.mkdir(parents=True, exist_ok=True)
            dst = root / f"{n}-{stamp}"
            k = 1
            while dst.exists():
                dst = root / f"{n}-{stamp}.{k}"
                k += 1
            (self.models_dir / n).rename(dst)
            self.backups[n] = dst
            log(f"--force: previous {n} moved to {dst} (restored if this conversion fails)")

    def rollback(self):
        """Remove this run's installs, restore backups. Returns the names restored."""
        import shutil
        restored = []
        for n in self.names:
            d = self.models_dir / n
            if n in self.backups:
                if d.exists():
                    shutil.rmtree(d)
                self.backups[n].rename(d)
                restored.append(n)
            elif n not in self.preexisting and d.exists():
                shutil.rmtree(d)
        self.backups = {}
        self._cleanup_root()
        return restored

    def commit(self):
        import shutil
        for b in self.backups.values():
            shutil.rmtree(b, ignore_errors=True)
        self.backups = {}
        self._cleanup_root()

    def _cleanup_root(self):
        root = self.models_dir / ".convert-backup"
        try:
            root.rmdir()  # only when empty
        except OSError:
            pass


def _validate_inputs(args):
    """Everything about the user's input that can be checked BEFORE converting: a typo must not
    cost an export + install (and a bad --name must never reach rmtree)."""
    from .common import validate_name
    from .report import parse_thresholds
    validate_name(args.name)
    try:
        parse_thresholds(getattr(args, "threshold", None))
        if getattr(args, "reference_script", None):
            from .reference import load_reference_script
            load_reference_script(args.reference_script)
    except Exception as e:  # noqa: BLE001 — user input / user code
        raise ConvertError(f"{type(e).__name__}: {e}")
    for flag in ("images", "eval"):
        v = getattr(args, flag, None)
        if v and not Path(v).exists():
            raise ConvertError(f"--{flag} {v}: no such file or directory")
    if getattr(args, "images", None) and not Path(args.images).is_dir():
        raise ConvertError(f"--images {args.images}: expected a DIRECTORY of images")
    v = getattr(args, "verify", None)
    if v is not None and int(v) < 1:
        raise ConvertError(f"--verify {v}: must be at least 1 (use --no-server to skip the server checks)")


def run(args):
    """Convert, install and verify. Returns the Report; raises ConvertError when conversion itself
    is refused (license, parity, contract, `visionserve pull`). A FAILED verification does not
    raise: the report says so (report.ok is False) and the model has been uninstalled — and a
    version it replaced (--force) restored — unless --keep-on-fail."""
    import shutil
    from . import common, serverctl
    from .report import FAIL, Report, TierResult, annotate_manifest
    from .verify import run_verification, tier_a_results

    _validate_inputs(args)
    fam = _family(FORMATS[args.format])
    models_dir = Path(args.models)
    with tempfile.TemporaryDirectory(prefix="vsconvert-") as tmp:
        work = Path(tmp)
        common.PARITY_RECORDS.clear()
        bundles = fam.convert(args, work)
        names = [common.validate_name(b.name) for b in bundles]
        report = Report(models=names, task=bundles[0].task, architecture=bundles[0].architecture)
        if args.dry_run:
            install(bundles, work / "out", models_dir, args.force, args.dry_run)
            keep = models_dir / ".convert-dry-run"
            if keep.exists():
                shutil.rmtree(keep)
            shutil.copytree(work / "out", keep)
            log(f"dry run: bundles copied to {keep} for inspection")
            for t in tier_a_results(list(common.PARITY_RECORDS), names):
                report.add(t)
            run_verification(bundles, args, report, models_dir, work)
            log(report.table())
            return report
        # A running server that ALREADY has one of these names keeps serving the copy it loaded
        # (Go re-scans only for unknown names): verification must then use a fresh server.
        stale = serverctl.registered_on_server(names, url=getattr(args, "server", None),
                                               no_server=getattr(args, "no_server", False), log=log)
        txn = _InstallTxn(models_dir, names, args.force)
        try:
            install(bundles, work / "out", models_dir, args.force, args.dry_run)
            for t in tier_a_results(list(common.PARITY_RECORDS), names):
                report.add(t)
            try:
                run_verification(bundles, args, report, models_dir, work, fresh_server=bool(stale))
            except Exception as e:  # noqa: BLE001 — the model is installed but was never verified
                import traceback
                log(f"verify: aborted by {type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}")
                report.add(TierResult("verify", "verification", FAIL,
                                      f"verification aborted: {type(e).__name__}: {e}"[:300],
                                      model=names[0]))
        except BaseException:  # refused install of a later bundle, Ctrl-C, ...: registry as before
            restored = txn.rollback()
            log("conversion interrupted: this run's installs removed"
                + (f"; previous {', '.join(restored)} restored" if restored else ""))
            raise

    if not report.ok and not args.keep_on_fail:
        report.restored = txn.rollback()
        report.uninstalled = True
        report.save(models_dir / ".convert-failed" / names[0] / "convert-report.json")
        log(report.table())
        log(f"FAILED: {', '.join(names)} uninstalled from {models_dir}"
            + (f"; the previous {', '.join(report.restored)} restored" if report.restored else "")
            + " (re-run with --keep-on-fail to keep it)")
        return report
    if report.ok:
        txn.commit()
    elif txn.backups:  # --keep-on-fail: the failed model stays; so does the version it replaced
        log("--keep-on-fail: the replaced version(s) are kept at "
            + ", ".join(str(b) for b in txn.backups.values()))
    for n in names:
        d = models_dir / n
        if d.is_dir():
            report.save(d / "convert-report.json")
            annotate_manifest(d / "manifest.yaml", report.manifest_comment_lines(n))
    log(report.table())
    return report


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "_unavailable", None):
        log(f"error: this Python environment cannot run {args._unavailable}")
        return 2
    try:
        report = run(args)
    except ConvertError as e:
        log(f"error: {e}")
        return 2
    except ValueError as e:  # bad --threshold / --eval / --reference-script
        log(f"error: {e}")
        return 2
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
