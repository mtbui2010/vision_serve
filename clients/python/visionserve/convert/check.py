"""`visionserve check <model>`: does the served model behave like my training pipeline?

The converter's verification tiers (verify.py, evaluate.py), run on a model that is ALREADY
INSTALLED, against a RUNNING server, in one command:

  B1  preprocessing: /api/preprocess vs a reference preprocessing of the same photos (--images).
      The reference, strongest first:
        --reference SCRIPT.py   your training transform: preprocess(pil) [and predict(pil)];
        --checkpoint PATH       the original framework's own pipeline (RF-DETR .pth, HuggingFace dir);
        the architecture's own recipe, when it is known without weights (RF-DETR: the rfdetr package
                                squashes to the input size with ImageNet mean/std);
        the manifest's declared preprocessing (this checks the server's code, NOT that the manifest
                                matches training; the summary says so).
  B2  outputs: /api/predict vs the original model. Only with a reference MODEL (--checkpoint, or
      --reference); otherwise skipped, with a note saying how to enable it.
  C   accuracy on --labels (COCO json for detection; a folder-per-class directory or a CSV
      `image,label` for classification): served vs reference when a reference model exists,
      the served number alone otherwise.

Output (shared with the Go reports): the first line is `PASS|WARN|FAIL: <one sentence>`, then a
plain-language summary (what was checked, numbers in words, a likely cause and fix for every FAIL
and WARN, from diagnose.py's fingerprints), next steps, and the converter's detailed table.
--json prints one object {"verdict", "reason", "summary", "details"} and nothing else on stdout;
--report FILE.html writes a self-contained page (htmlreport.py). Progress goes to stderr.
Exit code: 0 PASS or WARN, 1 FAIL, 2 usage or setup error (no server, model not installed, ...).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .common import IMAGENET_MEAN, IMAGENET_STD, ConvertError, log
from .constants import IMAGE_EXT, TEXT_ARCHS
from .reference import ManifestReference, OnnxForward, Reference, UserReference, load_reference_script
from .report import ERROR, FAIL, INFO, PASS, SKIP, WARN, Report, TierResult, parse_thresholds
from .spec import (ARCHS, PAD_MODES, SpecError, apply_spec, describe, resolve_arch, spec_from_legacy, spec_from_manifest,
                   unit_normalisation)

DEFAULT_URL = "http://localhost:11435"
EXIT_OK, EXIT_FAIL, EXIT_SETUP = 0, 1, 2

# Architectures whose outputs reference.decode_outputs decodes like the Go side: a --reference
# script without predict() still gets B2 there (its tensor through the installed ONNX).
_DECODABLE = ("rt-detr", "rf-detr", "efficientnet", "mobilenet-v3", "midas", "depth-anything-v2", "clip",
              "siglip-image")

TIER_NAMES = {"load": "Server loads the model", "B1": "Preprocessing (B1)", "B2": "Outputs vs original (B2)",
              "C": "Accuracy on your labels (C)"}


class SetupError(Exception):
    """A usage or setup problem (exit 2). The message says what to do."""


# --------------------------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False: the Go wrapper mounts the path flags it knows by their FULL name.
    ap = argparse.ArgumentParser(
        prog="visionserve check", allow_abbrev=False,
        description="Check that an installed model, served by a running VisionServe server, behaves like "
                    "your training pipeline: preprocessing (B1), outputs vs the original model (B2) and "
                    "accuracy on your labels (C).")
    ap.add_argument("model", help="installed model name (see `visionserve list`)")
    ap.add_argument("--images", required=True, metavar="DIR", help="a folder of real photos (B1/B2 use up to "
                    "--max-images of them, spread over the folder; C uses it as the image folder of --labels)")
    ap.add_argument("--labels", metavar="FILE", help="tier C: a COCO json (detection), or a folder-per-class "
                    "directory / CSV `image,label` (classification)")
    ref = ap.add_mutually_exclusive_group()
    ref.add_argument("--reference", metavar="SCRIPT.py", help="your training preprocessing: defines "
                     "preprocess(pil_image) -> CHW float32 array, optionally predict(pil_image) -> "
                     "[{cls, conf, bbox:[x,y,w,h]}] / class probabilities")
    ref.add_argument("--checkpoint", metavar="PATH", help="the original model the ONNX was exported from: an "
                     "RF-DETR .pth or a HuggingFace model directory; enables B2 and a reference for C")
    ap.add_argument("--server", metavar="URL", help=f"running VisionServe server (default $VISIONSERVE_HOST or "
                    f"{DEFAULT_URL})")
    ap.add_argument("--models", metavar="DIR", help="registry the server serves (default $VISIONSERVE_MODELS, "
                    "else ~/.visionserve/models): the manifest is read from there")
    ap.add_argument("--report", metavar="FILE.html", help="also write a self-contained HTML report")
    ap.add_argument("--json", action="store_true", help="print one JSON object instead of text")
    ap.add_argument("--max-images", type=int, default=8, metavar="N", help="photos used for B1/B2 (default 8)")
    ap.add_argument("--labels-max", type=int, default=200, metavar="N", help="cap on --labels images (default "
                    "200; fewer than ~200 makes a 0.5-threshold mAP noisy)")
    ap.add_argument("--prompt", help="text prompt for open-vocabulary models (default: the --labels category "
                    "names, else 'object.')")
    ap.add_argument("--device", default="auto", help="device for --checkpoint: auto, cpu, cuda, cuda:N")
    ap.add_argument("--max-map-drop", type=float, default=1.0, metavar="POINTS", help="C fails when the served "
                    "model scores more than this below the reference (WARN above half of it; default 1.0)")
    ap.add_argument("--threshold", action="append", metavar="KEY=VALUE", help="override a pass/warn/fail "
                    "threshold, e.g. b1_mean_fail=10 (keys in convert/report.py DEFAULT_THRESHOLDS)")
    return ap


def _default_models_dir() -> Path:
    env = os.environ.get("VISIONSERVE_MODELS")
    return Path(env) if env else Path.home() / ".visionserve" / "models"


# --------------------------------------------------------------------------------------------
# the installed model
# --------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Installed:
    bundle: object          # common.Bundle describing the installed model as the server reads it
    doc: dict               # the parsed manifest
    spec: object            # spec.Spec the server's model applies (spec.resolve_arch); None when
    #                         the architecture's export fixes it (the input block is reference only)
    manifest_path: Path
    onnx_path: Optional[Path]

    @property
    def uses_block(self) -> bool:
        return isinstance(self.doc.get("preprocess"), dict)

    @property
    def fixed_by_export(self) -> bool:
        return self.spec is None


def find_manifest(models_dir: Path, name: str) -> Path:
    """<models>/<name>/manifest.yaml, else the directory whose manifest says `name: <name>` (the
    Go registry keys on the manifest's name, not the directory's)."""
    import yaml
    direct = models_dir / name / "manifest.yaml"
    if direct.is_file():
        return direct
    if models_dir.is_dir():
        for d in sorted(models_dir.iterdir()):
            mf = d / "manifest.yaml"
            if not mf.is_file():
                continue
            try:
                if (yaml.safe_load(mf.read_text()) or {}).get("name") == name:
                    return mf
            except Exception:  # noqa: BLE001 — a broken neighbour is not our model
                continue
    raise SetupError(f"model {name!r} is not installed in {models_dir} (no manifest names it). See "
                     "`visionserve list`, or pass --models DIR of the registry the server serves.")


def _read_labels(d: Path, v) -> Optional[list]:
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str) and (d / v).is_file():
        lines = [ln.strip() for ln in (d / v).read_text().splitlines()]
        while lines and not lines[-1]:
            lines.pop()
        return lines
    return None


def load_installed(models_dir: Path, name: str) -> Installed:
    import yaml
    from .common import Bundle
    mf = find_manifest(models_dir, name)
    d = mf.parent
    try:
        doc = yaml.safe_load(mf.read_text()) or {}
    except yaml.YAMLError as e:
        raise SetupError(f"{mf}: not valid YAML: {e}")
    arch = str(doc.get("architecture") or "")
    try:
        declared = spec_from_manifest(doc)
        # What the server's model really applies: the architecture reads legacy fields its own
        # way (SCRFD's `letterbox: true` is a top-left pad in 0..255 units), exactly as Go.
        spec = resolve_arch(declared, arch)
    except SpecError as e:
        raise SetupError(f"{mf}: {e}")
    eff = spec or declared
    mean, std = unit_normalisation(eff)  # the bundle's mean/std are in [0,1] units
    files = doc.get("files") if isinstance(doc.get("files"), dict) else {}
    rel = doc.get("model_file") or (files.get("model") or next(iter(files.values()), None) if files else None)
    onnx_path = (d / rel) if rel else None
    b = Bundle(name=str(doc.get("name") or name), task=str(doc.get("task") or ""),
               architecture=arch, license=str(doc.get("license") or ""),
               width=int(eff.width or 0), height=int(eff.height or 0),
               onnx={"model": str(onnx_path)} if onnx_path else {}, layout=eff.layout or "NCHW",
               letterbox=eff.resize in PAD_MODES, crop="center" if eff.resize == "center_crop" else None,
               keep_aspect=eff.resize == "keep_aspect", multiple_of=int(eff.multiple_of or 0),
               mean=mean, std=std, postprocess=dict(doc.get("postprocess") or {}),
               labels=_read_labels(d, doc.get("labels")))
    return Installed(b, doc, spec, mf, onnx_path if onnx_path and onnx_path.is_file() else None)


def image_input(onnx_path) -> Optional[str]:
    """The ONNX input the photo goes to: a float32 4-D input, else the first float32 input."""
    if onnx_path is None:
        return None
    from .common import onnx_io
    ins, _ = onnx_io(onnx_path)
    for name, shp, et in ins:
        if len(shp) == 4 and et == 1:
            return name
    for name, shp, et in ins:
        if et == 1:
            return name
    return None


# --------------------------------------------------------------------------------------------
# references
# --------------------------------------------------------------------------------------------

class SpecReference(ManifestReference):
    """The manifest's declared preprocessing exactly as the server resolves it (the `preprocess:`
    block included), re-implemented in numpy/PIL."""

    def __init__(self, bundle, input_name, spec, forward=None, forward_desc=""):
        super().__init__(bundle, input_name, forward, forward_desc)
        self.spec = spec
        self.short_pre = "the manifest's declared preprocessing"

    def preprocess(self, pil, prompt=None):
        x, _ = apply_spec(pil, self.spec)
        return {self.input_name: x}


class RFDETRRecipeReference(Reference):
    """How the rfdetr package prepares a photo, known without the weights: RFDETR.predict (and its
    training resize) squash to the square input size, then /255 and ImageNet mean/std. The same
    transform as the converter's RFDETRReference.preprocess."""
    kind = "official"
    framework = "torch"
    short = short_pre = "the rfdetr package's preprocessing"
    mean, std = IMAGENET_MEAN, IMAGENET_STD

    def __init__(self, width: int, height: int, input_name: str):
        self.width, self.height, self.input_name = int(width), int(height), input_name
        self.description = (f"how the rfdetr package prepares a photo (RFDETR.predict, training): squash to "
                            f"{self.width}x{self.height}, /255, ImageNet mean/std (torchvision F.resize) — known "
                            "without a checkpoint")

    def preprocess(self, pil, prompt=None):
        try:
            import torchvision.transforms.functional as F
        except ImportError:  # the same squash in PIL: within ~0.3 gray levels of torchvision
            x, _ = apply_spec(pil, spec_from_legacy(self.width, self.height, "", False, None, False, 0,
                                                    IMAGENET_MEAN, IMAGENET_STD))
            return {self.input_name: x}
        t = F.to_tensor(pil.convert("RGB"))
        t = F.normalize(F.resize(t, [self.height, self.width]), IMAGENET_MEAN, IMAGENET_STD)
        return {self.input_name: t[None].numpy().astype("float32")}


# architecture -> the recipe its framework always uses (only where no checkpoint is needed to know it)
RECIPES = {"rf-detr": RFDETRRecipeReference}


def checkpoint_reference(path, inst: Installed, input_name: str, workdir: Path) -> Reference:
    """The original framework pipeline from the checkpoint the installed ONNX was exported from."""
    p = Path(path)
    b = inst.bundle
    conf = float((b.postprocess or {}).get("conf_threshold", 0.5))
    try:
        if b.architecture == "rf-detr":
            return _rfdetr_reference(p, b, input_name, conf, workdir)
        if (p / "config.json").is_file():
            return _hf_reference(p, b, conf)
    except ConvertError as e:
        raise SetupError(f"--checkpoint {p}: {e}")
    raise SetupError(f"--checkpoint for a {b.architecture} model is not supported (RF-DETR .pth and HuggingFace "
                     "classification / detection / depth directories are). Pass --reference SCRIPT.py with "
                     "preprocess() and predict() instead.")


def _rfdetr_reference(p: Path, b, input_name: str, conf: float, workdir: Path) -> Reference:
    from .common import license_scan
    from .families import rfdetr as fam
    src = fam._find_checkpoint(p)
    license_scan(source=src)
    ck = fam._load_checkpoint(src)
    sd, targs, model_name = fam._split_checkpoint(ck)
    del ck
    variant, how = fam.detect_variant(sd, targs, model_name, src.name)
    n_logits = int(sd["class_embed.bias"].shape[0])
    log(f"check: --checkpoint {src.name}: RF-DETR {variant} (from {how}), {n_logits} logits, built at "
        f"{b.width}x{b.height}")
    model, _ = fam._build(variant, sd, targs, n_logits, int(b.width) or None, workdir)
    labels = b.labels if b.labels and len(b.labels) == n_logits else fam.labels_for(targs.get("class_names"),
                                                                                    n_logits)
    if b.labels and len(b.labels) != n_logits:
        log(f"check: warning: the manifest has {len(b.labels)} labels but the checkpoint {n_logits} logits — "
            "is this the checkpoint the model was exported from?")
    ref = fam.RFDETRReference(model, labels, conf, input_name or "input")
    ref.mean, ref.std = list(model.means), list(model.stds)
    return ref


def _hf_reference(p: Path, b, conf: float) -> Reference:
    from .families import hf
    cfg = json.loads((p / "config.json").read_text())
    archs = cfg.get("architectures") or []
    if any("ForImageClassification" in a for a in archs):
        from transformers import AutoModelForImageClassification as Auto
        task = "classification"
    elif any(a.endswith("ForObjectDetection") for a in archs) and b.task == "detection":
        from transformers import AutoModelForObjectDetection as Auto
        task = "detection"
    elif any(a.endswith("ForDepthEstimation") for a in archs):
        from transformers import AutoModelForDepthEstimation as Auto
        task = "depth"
    else:
        raise SetupError(f"--checkpoint {p}: architectures {archs} are not supported by check (classification, "
                         "object detection and depth are); pass --reference SCRIPT.py instead")
    return hf.HFImageReference(p, hf._load(Auto, p), task, b.labels, conf=conf)


@dataclasses.dataclass
class CheckPlan:
    b1: Optional[Reference]
    b2: Optional[Reference]
    b1_source: str           # user | checkpoint | recipe | manifest | fixed (by the export) | none
    b2_note: str = ""


def build_plan(args, inst: Installed, input_name: Optional[str], device: str, workdir: Path) -> CheckPlan:
    b = inst.bundle
    if args.reference:
        s = load_reference_script(args.reference)
        name = Path(args.reference).name
        decodable = b.architecture in _DECODABLE and inst.onnx_path is not None
        fwd = OnnxForward(inst.onnx_path) if decodable else None
        if s["predict"] is not None:
            desc = f"your {name} (preprocess + predict)"
        elif decodable:
            desc = f"your {name} preprocess + the installed ONNX (CPU), decoded like the server"
        else:
            desc = f"your {name} preprocess"
        ref = UserReference(b, input_name, s["preprocess"], s["predict"], fwd, desc)
        b2 = ref if (s["predict"] is not None or decodable) else None
        note = "" if b2 else (f"your {name} has no predict(), and check cannot decode {b.architecture} outputs "
                              "itself: define predict(pil) to compare outputs")
        return CheckPlan(ref, b2, "user", note)
    if args.checkpoint:
        ref = checkpoint_reference(args.checkpoint, inst, input_name, workdir)
        ref.to(device)
        return CheckPlan(ref, ref, "checkpoint")
    note = ("no reference model was given. Pass --checkpoint PATH (the checkpoint the ONNX was exported from) "
            "or --reference SCRIPT.py to compare the served outputs with the original model")
    recipe = RECIPES.get(b.architecture)
    if recipe is not None and input_name:
        return CheckPlan(recipe(b.width, b.height, input_name), None, "recipe", note)
    if inst.fixed_by_export:
        return CheckPlan(None, None, "fixed", note)
    if input_name:
        return CheckPlan(SpecReference(b, input_name, inst.spec), None, "manifest", note)
    return CheckPlan(None, None, "none", note)


class _MemoReference:
    """Delegates to a Reference, remembering predict() for the B photos (kept alive by the caller,
    so their id() is stable) — the figures reuse B2's predictions instead of recomputing them."""

    def __init__(self, ref, keep_ids):
        self._ref, self._keep, self._memo = ref, set(keep_ids), {}

    def __getattr__(self, k):
        return getattr(self._ref, k)

    def predict(self, pil, prompt=None):
        key = (id(pil), prompt)
        if id(pil) not in self._keep:
            return self._ref.predict(pil, prompt)
        if key not in self._memo:
            self._memo[key] = self._ref.predict(pil, prompt)
        return self._memo[key]


class _MemoClient:
    """The SDK client, remembering predict()/preprocess() per (model, photo path, prompt)."""

    def __init__(self, client):
        self._c, self._memo = client, {}

    def __getattr__(self, k):
        return getattr(self._c, k)

    def _get(self, kind, fn, model, image, prompt, **kw):
        if not isinstance(image, (str, Path)):
            return fn(model, image, prompt=prompt, **kw)
        key = (kind, model, str(image), prompt)
        if key not in self._memo:
            self._memo[key] = fn(model, image, prompt=prompt, **kw)
        return self._memo[key]

    def predict(self, model, image, prompt=None, **kw):
        return self._get("predict", self._c.predict, model, image, prompt, **kw)

    def preprocess(self, model, image=None, prompt=None, **kw):
        return self._get("preprocess", self._c.preprocess, model, image, prompt, **kw)


# --------------------------------------------------------------------------------------------
# server
# --------------------------------------------------------------------------------------------

def _registered(url: str, name: str) -> bool:
    """POST /api/load: False only for 404 (the server's registry has no such model). The server
    re-scans its registry for a name it does not know, so a model installed after it started is
    found here although /api/models did not list it yet; a load that fails for another reason is
    reported by the check itself."""
    from ..client import Client, VisionServeError
    try:
        Client(url, timeout=600).load(name)
    except VisionServeError as e:
        return getattr(e, "status", None) != 404
    return True


def connect(url: str, name: str, models_dir: Path, health=None, listed=None, registered=None):
    """Fail (SetupError) unless a server answers at `url` and serves `name`."""
    from . import serverctl
    health = health or serverctl._health
    listed = listed or serverctl._listed
    registered = registered or _registered
    if not health(url):
        raise SetupError(f"no VisionServe server answers at {url}. Start one in another terminal, on the "
                         f"registry that holds {name!r}:\n  visionserve serve --models {models_dir}\nthen re-run "
                         "this command (or pass --server URL of a running server).")
    try:
        have = listed(url)
    except Exception as e:  # noqa: BLE001
        raise SetupError(f"the server at {url} answers /api/health but GET /api/models failed: {e}")
    if name not in have and not registered(url, name):
        shown = ", ".join(sorted(have)[:8]) + (" ..." if len(have) > 8 else "")
        raise SetupError(f"the server at {url} does not serve {name!r} (it lists: {shown or 'nothing'}). It "
                         f"serves another registry: restart it with --models {models_dir}, or pass --server "
                         "URL of the server that serves this registry.")


# --------------------------------------------------------------------------------------------
# labels (tier C)
# --------------------------------------------------------------------------------------------

def load_labels(path, images_dir, max_n: int, workdir: Path):
    """--labels -> evaluate.EvalSet. A COCO json is restricted to the photos present in --images
    (so a full instances_val2017.json works with a folder holding a subset); a directory is an
    ImageFolder; a .csv holds `image,label` rows (image relative to --images)."""
    from .evaluate import EvalSet, load_eval
    p, root = Path(path), Path(images_dir)
    if p.suffix.lower() == ".csv":
        items = []
        with open(p, newline="") as f:
            for row in csv.reader(f):
                if len(row) < 2 or not row[0].strip():
                    continue
                img, lab = row[0].strip(), row[1].strip()
                if not items and img.lower() in ("image", "file", "filename", "path"):
                    continue  # header
                f_img = Path(img) if Path(img).is_absolute() else root / img
                if f_img.is_file() and f_img.suffix.lower() in IMAGE_EXT:
                    items.append((f_img, lab))
                if max_n and len(items) >= max_n:
                    break
        if not items:
            raise SetupError(f"--labels {p}: no `image,label` row names a photo in {root}")
        labels = sorted({lab for _, lab in items})
        return EvalSet("classification", items, f"{p} ({len(items)} images, {len(labels)} classes)", labels=labels)
    if p.is_file() and p.suffix.lower() == ".json":
        d = json.loads(p.read_text())
        imgs = [im for im in d.get("images") or []
                if (root / im["file_name"]).is_file() or (root / Path(im["file_name"]).name).is_file()]
        if not imgs:
            raise SetupError(f"--labels {p}: none of its {len(d.get('images') or [])} images is in {root}")
        d["images"] = imgs
        keep = {im["id"] for im in imgs}
        d["annotations"] = [a for a in d.get("annotations") or [] if a.get("image_id") in keep]
        sub = workdir / p.name
        sub.write_text(json.dumps(d))
        ev = load_eval(sub, root, max_n)
        ev.source = str(ev.source).replace(str(sub), str(p))  # name the user's file, not the temp copy
        return ev
    if p.is_dir():
        return load_eval(p, None, max_n)
    raise SetupError(f"--labels {p}: expected a COCO .json, a .csv `image,label`, or a folder-per-class directory")


def served_accuracy(client, b, ev, prompt=None) -> TierResult:
    """Tier C without a reference model: the served mAP / top-1 alone (status INFO; WARN when it
    is near zero, which almost always means the label names or the box mapping are wrong)."""
    from .evaluate import _key, coco_map, map_labels, topk_hits
    name = b.name
    title = f"served accuracy on {Path(str(ev.source).split(' (')[0]).name}"
    if ev.kind == "detection" and b.task in ("detection", "open_vocab"):
        labels = b.labels
        if b.task == "open_vocab":
            labels = [s.strip() for s in (prompt or "").split(".") if s.strip()]
        mapping, unmapped, uncovered = map_labels(labels, ev.categories)
        if not mapping:
            return TierResult("C", title, FAIL, f"no model class maps to a dataset category by name (model: "
                              f"{(labels or [])[:5]}..., dataset: {list(ev.categories.values())[:5]}...)", model=name)
        res = []
        for n, (path, img_id) in enumerate(ev.items):
            for d in client.predict(name, path, prompt=prompt, max_grasps_per_object=None).detections:
                if d.cls in mapping:
                    res.append({"image_id": img_id, "category_id": mapping[d.cls],
                                "bbox": [float(v) for v in d.bbox], "score": float(d.conf)})
            if (n + 1) % 25 == 0:
                log(f"  C: {n + 1}/{len(ev.items)} images")
        img_ids = [i for _, i in ev.items]
        cat_ids = sorted(set(mapping.values()))
        m, m50 = coco_map(ev.coco, res, img_ids, cat_ids)
        conf = (b.postprocess or {}).get("conf_threshold")
        st = WARN if m50 < 1.0 else INFO
        notes = [f"served only (no reference model to compare with), at conf_threshold {conf}: below a paper mAP "
                 "computed at ~0.001"]
        if uncovered:
            notes.append(f"dataset categories the model has no class for (excluded): {uncovered[:12]}")
        return TierResult("C", title, st, f"mAP {m:.2f} served, mAP50 {m50:.2f}; {len(img_ids)} images, "
                          f"{len(cat_ids)} classes", model=name, notes=notes, metrics={
                              "kind": "detection", "served_only": True, "images": len(img_ids),
                              "classes": len(cat_ids), "conf_threshold": conf, "map_served": m,
                              "map50_served": m50, "n_dets_served": len(res), "unmapped_labels": unmapped,
                              "uncovered_categories": uncovered})
    if ev.kind == "classification" and b.task == "classification":
        known = {_key(lab) for lab in (b.labels or [])}
        items = [(p, g) for p, g in ev.items if _key(g) in known]
        miss = sorted({g for _, g in ev.items if _key(g) not in known})
        if not items:
            return TierResult("C", title, FAIL, f"no label maps to a model class (labels {miss[:5]}...)", model=name)
        h1 = h5 = 0
        for p, g in items:
            ranked = [c.cls for c in sorted(client.predict(name, p, max_grasps_per_object=None).classifications,
                                            key=lambda c: -c.conf)]
            h1 += topk_hits(ranked, g, 1)
            h5 += topk_hits(ranked, g, 5)
        t1, t5 = 100.0 * h1 / len(items), 100.0 * h5 / len(items)
        st = WARN if t1 < 1.0 else INFO
        notes = ["served only (no reference model to compare with)"]
        if miss:
            notes.append(f"labels with no matching model class (skipped): {miss[:12]}")
        return TierResult("C", title, st, f"top-1 {t1:.2f} served, top-5 {t5:.2f}; {len(items)} images", model=name,
                          notes=notes, metrics={"kind": "classification", "served_only": True, "images": len(items),
                                                "top1_served": t1, "top5_served": t5, "skipped_labels": miss})
    return TierResult("C", "accuracy", SKIP, f"{ev.kind} labels do not apply to a {b.task} model", model=name)


# --------------------------------------------------------------------------------------------
# running the tiers
# --------------------------------------------------------------------------------------------

@dataclasses.dataclass
class CheckRun:
    args: object
    inst: Installed
    url: str
    report: Report
    plan: Optional[CheckPlan] = None
    images: list = dataclasses.field(default_factory=list)
    images_total: int = 0
    evalset: object = None
    figures: list = dataclasses.field(default_factory=list)  # [{"kind", "title", "caption", "panels": [(PIL, label)]}]
    implied_norm: Optional[Tuple[list, list]] = None
    b1_shapes: Optional[Tuple[list, list]] = None


def prepare(args):
    """Everything that can be refused before any work: flags, files, registry, server."""
    from .verify import list_images
    imgs = Path(args.images)
    if not imgs.is_dir():
        raise SetupError(f"--images {imgs}: not a directory of photos")
    photos = list_images(imgs)
    if not photos:
        raise SetupError(f"--images {imgs}: no image files ({', '.join(sorted(IMAGE_EXT))})")
    if args.max_images < 1:
        raise SetupError("--max-images must be at least 1")
    for flag in ("labels", "reference", "checkpoint"):
        v = getattr(args, flag)
        if v and not Path(v).exists():
            raise SetupError(f"--{flag} {v}: no such file or directory")
    if args.report and not Path(args.report).resolve().parent.is_dir():
        raise SetupError(f"--report {args.report}: its directory does not exist")
    try:
        parse_thresholds(args.threshold)
        if args.reference:
            load_reference_script(args.reference)
    except Exception as e:  # noqa: BLE001 — user input / user code
        raise SetupError(f"{type(e).__name__}: {e}")
    models_dir = Path(args.models) if args.models else _default_models_dir()
    inst = load_installed(models_dir, args.model)
    url = (args.server or os.environ.get("VISIONSERVE_HOST") or DEFAULT_URL).rstrip("/")
    connect(url, inst.bundle.name, models_dir)
    return inst, url, photos


def run_check(args, inst: Installed, url: str, photos: list, workdir: Path) -> CheckRun:
    from ..client import Client
    from .evaluate import tier_c
    from .reference import resolve_device
    from .verify import _guard, default_prompt, load_images, spread, tier_b1, tier_b2

    b = inst.bundle
    name = b.name
    th = parse_thresholds(args.threshold)
    report = Report(models=[name], task=b.task, architecture=b.architecture, thresholds=th,
                    server={"url": url, "temporary": False})
    run = CheckRun(args, inst, url, report, images_total=len(photos))
    if args.labels:
        run.evalset = load_labels(args.labels, args.images, args.labels_max, workdir)
        log(f"check: labels {run.evalset.source}")
    run.images = load_images(spread(photos, args.max_images))
    prompt = default_prompt(b, run.evalset, args.prompt)
    client = _MemoClient(Client(url, timeout=600))

    log(f"check: {name} on {url}: loading")
    try:
        client.load(name)
    except Exception as e:  # noqa: BLE001 — the server refused: that IS the finding
        report.add(TierResult("load", "the server loads the model", FAIL, str(e)[:400], model=name))
        for t in ("B1", "B2", "C"):
            report.add(TierResult(t, TIER_NAMES[t], SKIP, "the model does not load", model=name))
        return run

    input_name = image_input(inst.onnx_path)
    if input_name is None and b.architecture not in TEXT_ARCHS:
        # The ONNX file is not readable here (e.g. a symlink out of the mounted registry): the
        # server's own /api/preprocess names the image input just as well.
        if inst.onnx_path is None:
            log("check: the model's ONNX file is not readable here; taking the image input's name from the server")
        try:
            input_name = _image_key(client.preprocess(name, run.images[0].src, prompt=prompt).inputs)
        except Exception as e:  # noqa: BLE001 — B1 then reports the server's refusal itself
            log(f"check: /api/preprocess failed: {e}")
    device = resolve_device(args.device)
    try:
        run.plan = build_plan(args, inst, input_name, device, workdir)
    except SetupError:
        raise
    except Exception as e:  # noqa: BLE001 — building a reference from user files
        raise SetupError(f"cannot build the reference: {type(e).__name__}: {e}")
    plan = run.plan
    keep = [id(im.pil) for im in run.images]
    b2 = _MemoReference(plan.b2, keep) if plan.b2 is not None else None
    from .verify import Plan
    vplan = Plan(b, input_name, plan.b1, b2, None, str(inst.onnx_path or ""))

    if b.architecture in TEXT_ARCHS and plan.b1_source not in ("user", "checkpoint"):
        report.add(TierResult("B1", "token ids", SKIP, "a text model: pass --reference with your tokenizer to "
                              "compare token ids", model=name))
    elif plan.b1 is None and plan.b1_source == "fixed":
        report.add(TierResult("B1", "preprocessing", SKIP, f"{b.architecture}'s preprocessing is fixed by its "
                              "export (the manifest's input block is reference only): pass --reference with your "
                              "preprocessing to compare", model=name))
    elif plan.b1 is None:
        report.add(TierResult("B1", "preprocessing", SKIP, "the model's ONNX file has no float image input to "
                              "compare", model=name))
    else:
        log(f"check: B1 on {len(run.images)} photo(s) vs {plan.b1.description}")
        report.add(_guard("B1", "preprocessing", name, lambda: tier_b1(vplan, client, name, run.images, th, prompt)))
    if b2 is None:
        report.add(TierResult("B2", "outputs vs the original model", SKIP, plan.b2_note, model=name))
    else:
        log(f"check: B2 vs {b2.description}")
        report.add(_guard("B2", "outputs", name, lambda: tier_b2(vplan, client, name, run.images, th, prompt)))
    if run.evalset is None:
        report.add(TierResult("C", "accuracy", SKIP, "no --labels: accuracy not measured", model=name))
    elif b2 is None:
        log(f"check: C (served only) on {len(run.evalset.items)} labelled photo(s)")
        report.add(_guard("C", "accuracy", name, lambda: served_accuracy(client, b, run.evalset, prompt)))
    else:
        log(f"check: C on {len(run.evalset.items)} labelled photo(s), reference vs served")
        report.add(_guard("C", "accuracy", name,
                          lambda: tier_c(vplan, client, b, run.evalset, args.max_map_drop, prompt)))
    try:
        _figures(run, vplan, client, prompt)
    except Exception as e:  # noqa: BLE001 — a figure must never hide the verdict
        log(f"check: figures skipped: {type(e).__name__}: {e}")
    return run


def _image_key(d: dict) -> Optional[str]:
    keys = [k for k, v in d.items() if np.asarray(v).ndim in (3, 4) and np.asarray(v).dtype != np.int64]
    return keys[0] if keys else None


def _chw(t) -> np.ndarray:
    a = np.asarray(t, np.float32)
    if a.ndim == 4:
        a = a[0]
    if a.ndim == 3 and a.shape[0] not in (1, 3) and a.shape[-1] in (1, 3):
        a = a.transpose(2, 0, 1)
    return a


def _ref_norm(ref, b) -> Tuple[list, list]:
    mean = getattr(ref, "mean", None) or b.mean or [0.0, 0.0, 0.0]
    std = getattr(ref, "std", None) or b.std or [1.0, 1.0, 1.0]
    return list(mean), list(std)


def _figures(run: CheckRun, vplan, client, prompt) -> None:
    from . import htmlreport as hr
    from .diagnose import _implied_norm, affine_fit
    from .metrics import as_dets, match_detections
    b = run.inst.bundle
    t1 = run.report.tier("B1")
    if t1 is not None and t1.status not in (SKIP, ERROR) and vplan.b1 is not None and t1.metrics.get("per_image"):
        per = t1.metrics["per_image"]
        worst = max(per, key=lambda p: p.get("mean_levels", float("inf")) if p.get("same_shape") else float("inf"))
        im = next(i for i in run.images if i.label == worst["image"])
        res = client.preprocess(b.name, im.src, prompt=prompt)
        rin = vplan.b1.preprocess(im.pil, prompt)
        sk = vplan.input_name if vplan.input_name in res.inputs else _image_key(res.inputs)
        rk = vplan.input_name if vplan.input_name in rin else _image_key(rin)
        s, r = _chw(res.inputs[sk]), _chw(rin[rk])
        mean, std = _ref_norm(vplan.b1, b)
        panels = [(hr.tensor_image(r, mean, std), "reference: " + _ref_phrase(run.plan)),
                  (hr.tensor_image(s, mean, std), "server: /api/preprocess")]
        if s.shape == r.shape:
            sstd = np.resize(np.asarray(b.std if b.std else [1.0], np.float32), s.shape[0])
            lv = (np.abs(s - r) * sstd[:, None, None] * 255.0).mean(0)
            top = max(float(run.report.thresholds.get("b1_mean_fail", 8.0)), float(np.percentile(lv, 99.5)))
            panels.append((hr.heatmap_image(lv, top), f"|Δ| in gray levels: mean {lv.mean():.1f}, max {lv.max():.0f} "
                                                      f"(black 0, yellow {top:.0f} or more)"))
            if "normalisation" in (t1.metrics.get("diagnosis") or {}) and s.shape[0] == 3:
                fits = [affine_fit(s[k].astype(np.float64), r[k].astype(np.float64)) for k in range(3)]
                a = np.array([f[0] for f in fits])
                bb = np.array([f[1] for f in fits])
                m_r, s_r = _implied_norm(a, bb, np.asarray(b.mean or [0, 0, 0], np.float64),
                                         np.asarray(b.std or [1, 1, 1], np.float64))
                run.implied_norm = ([float(v) for v in m_r], [float(v) for v in s_r])
        else:
            run.b1_shapes = (list(s.shape), list(r.shape))
        run.figures.append({"kind": "b1", "title": f"What the model sees: {im.label} (the photo with the largest "
                            "difference)",
                            "caption": "Both tensors drawn with the reference's normalisation, i.e. as the model "
                                       "reads them. The heatmap is bright where the two differ.",
                            "panels": panels})
    t2 = run.report.tier("B2")
    if t2 is not None and t2.status not in (SKIP, ERROR) and vplan.b2 is not None and \
            b.task in ("detection", "open_vocab"):
        scored = []
        for im in run.images:
            ref = vplan.b2.predict(im.pil, prompt)
            srv = client.predict(b.name, im.src, prompt=prompt, max_grasps_per_object=None)
            rd, sd = as_dets(ref.detections or []), as_dets(srv.detections)
            pairs, ur, us = match_detections(rd, sd, run.report.thresholds.get("b2_iou", 0.5))
            scored.append((len(ur) + len(us), len(rd) + len(sd), im, rd, sd))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        top = scored[:2]
        if top:
            run.figures.append({
                "kind": "b2", "title": "Boxes: the original model vs the served model (the photos where they "
                                       "disagree most)",
                "caption": "Thick blue: the original model (reference). Thin orange: the served model.",
                "panels": [(hr.draw_boxes(im.pil, rd, sd),
                            f"{im.label}: reference {len(rd)} / served {len(sd)} boxes, "
                            + (f"{miss} found by only one side" if miss else "every box matched"))
                           for miss, _, im, rd, sd in top]})


# --------------------------------------------------------------------------------------------
# plain-language summary
# --------------------------------------------------------------------------------------------

def _ref_phrase(plan: Optional[CheckPlan]) -> str:
    """Who the B1 reference is, in a few words."""
    src = plan.b1_source if plan else "none"
    if src == "user":
        return "your " + Path(str(plan.b1.description).split(" ")[1]).name if plan.b1 else "your script"
    if src == "recipe":
        return getattr(plan.b1, "short_pre", "") or "the architecture's own preprocessing"
    return {"checkpoint": "the original framework's pipeline",
            "manifest": "the manifest's declared preprocessing"}.get(src, "the reference")


def _fmt_list(v) -> str:
    return "[" + ", ".join(f"{float(x):.3g}" for x in v) + "]"


def _snap_norm(mean, std):
    """Recognise the usual normalisations in implied values; else round to 3 decimals."""
    m, s = np.asarray(mean), np.asarray(std)
    for name, mm, ss in (("ImageNet", IMAGENET_MEAN, IMAGENET_STD), ("0.5", [0.5] * 3, [0.5] * 3),
                         ("none", [0.0] * 3, [1.0] * 3)):
        if np.allclose(m, mm, atol=0.03) and np.allclose(s, ss, atol=0.03):
            return list(mm), list(ss), name
    return [round(float(x), 3) for x in m], [round(float(x), 3) for x in s], ""


# diagnose.py code -> priority (lower first): the one cause shown first when several fire. A
# channel swap also shifts the per-channel means, so its normalisation note is a side effect.
_PRIORITY = ["shape", "letterbox_server", "letterbox_reference", "crop", "flip", "channel_swap", "scale_255",
             "scale_inv255", "normalisation", "spatial", "resample"]


def _same_norm(inst: Installed, mean, std, tol: float = 1e-3) -> bool:
    """The model already normalises with this mean/std ([0,1] units, as the bundle holds them)."""
    b = inst.bundle
    m, s = b.mean or [0.0] * 3, b.std or [1.0] * 3
    return (len(m) == len(mean) == 3 and len(s) == len(std) == 3 and
            np.allclose(np.asarray(m, np.float64), np.asarray(mean, np.float64), atol=tol) and
            np.allclose(np.asarray(s, np.float64), np.asarray(std, np.float64), rtol=10 * tol, atol=0.0))


_GEOMETRY = {"squash": "stretches the photo", "letterbox": "letterboxes it (centred)",
             "top_left_pad": "pastes it at the top-left of a padded canvas", "center_crop": "keeps only its centre",
             "keep_aspect": "keeps its aspect ratio without padding", "long_side": "scales its long side, no padding",
             "long_side_pad": "scales its long side and pads bottom/right", "none": "feeds it at its own size"}


def _server_geometry(inst: Installed) -> str:
    """What the server does to the photo's geometry, in words (stretches it, unless resolved otherwise)."""
    mode = (inst.spec.resize if inst.spec is not None else "") or "squash"
    return _GEOMETRY.get(mode, "stretches the photo")


def _already_set(inst: Installed, setting: str, wanted: Optional[str]) -> str:
    """The fix when the manifest already says what a fix would set: never tell the user to set a
    value that is set. Says how the architecture serves it, and whether it can serve `wanted`."""
    arch = inst.bundle.architecture or "this architecture"
    served = (f"serves {describe(inst.spec)}" if inst.spec is not None else
              "uses the preprocessing its export fixes")
    msg = f"{setting} is already set in {inst.manifest_path}, and {arch} {served} from it"
    a = ARCHS.get(inst.bundle.architecture)
    if wanted and a is not None and wanted not in a.modes:
        return (msg + f"; {arch} cannot serve {wanted.replace('_', ' ')} (it serves "
                f"{', '.join(m.replace('_', ' ') for m in a.modes)}), so check that the reference really is "
                "how this model is fed")
    return msg + "; compare the two pictures in --report to find what else differs"


def causes_for(codes: Dict[str, int], run: CheckRun) -> List[dict]:
    """[{code, cause, fix}] for B1's diagnosis codes, most decisive first. Fixes name the manifest
    field in the style the manifest already uses (a `preprocess:` block or the legacy input.*)."""
    inst = run.inst
    mf = str(inst.manifest_path)
    block = inst.uses_block
    W, H = inst.bundle.width, inst.bundle.height
    # "training" when the reference knows how the model was trained; else just "the reference"
    who = "training" if run.plan and run.plan.b1_source in ("recipe", "checkpoint", "user") else "the reference"
    try:
        declared = spec_from_manifest(inst.doc)
    except SpecError:
        declared = None
    out = []
    for code in sorted(codes, key=lambda c: _PRIORITY.index(c) if c in _PRIORITY else 99):
        if code == "letterbox_server":
            cause = (f"the server letterboxes (shrinks the photo and adds bars) while {who} stretches the "
                     f"whole photo to {W}x{H}")
            setting = "`preprocess.resize: squash`" if block else "`input.letterbox: false`"
            fix = f"in {mf} set {setting} (unless the model really was trained letterboxed)"
            if declared is not None and declared.resize == "squash":
                fix = _already_set(inst, setting, "squash")
        elif code == "letterbox_reference":
            cause = f"{who} letterboxes (keeps the aspect ratio and pads) while the server {_server_geometry(inst)}"
            setting = "`preprocess.resize: letterbox`" if block else "`input.letterbox: true`"
            fix = f"in {mf} set {setting}"
            if declared is not None and declared.resize == "letterbox":
                fix = _already_set(inst, setting, "letterbox")
        elif code == "crop":
            cause = f"{who} uses only the centre of the photo (a centre crop) while the server keeps all of it"
            setting = "`preprocess.resize: center_crop`" if block else "`input.crop: center`"
            fix = f"in {mf} set {setting}"
            if declared is not None and declared.resize == "center_crop":
                fix = _already_set(inst, setting, "center_crop")
        elif code == "flip":
            cause = "the reference is mirrored left-right compared with the server"
            fix = "check your reference script: a random horizontal flip is probably still on"
        elif code == "channel_swap":
            cause = f"{who} reads colour channels in BGR order (OpenCV's cv2.imread) while the server feeds RGB"
            fix = ("the server always feeds RGB: convert to RGB in training (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) "
                   "and retrain, or fix the reference script if only it reads BGR")
        elif code == "scale_255":
            cause = f"{who} uses 0-255 pixel values while the server divides by 255 (0-1)"
            fix = (f"if the model expects 0-255 pixels, set mean [0, 0, 0] and std [0.00392157, 0.00392157, "
                   f"0.00392157] in {mf}; otherwise your reference skips the /255 (ToTensor)")
            if _same_norm(inst, [0.0] * 3, [1.0 / 255] * 3):
                fix = _already_set(inst, "0-255 pixels (mean 0, std 1/255)", None)
        elif code == "scale_inv255":
            cause = "the reference values are 255 times smaller than the server's (divided by 255 twice?)"
            fix = f"check the std in {mf} (1/255 values mean raw pixels) and the reference's /255"
        elif code == "normalisation":
            if run.implied_norm is not None:
                m, s, known = _snap_norm(*run.implied_norm)
                what = f"{known} mean/std" if known and known != "none" else (
                    "no mean/std (just /255)" if known == "none" else "a different mean/std")
                cause = (f"the colours are normalised differently: {who} looks like {what} "
                         f"(mean {_fmt_list(m)}, std {_fmt_list(s)}), the manifest declares mean "
                         f"{_fmt_list(inst.bundle.mean or [0, 0, 0])}, std {_fmt_list(inst.bundle.std or [1, 1, 1])}")
                field = "`preprocess.mean` / `preprocess.std`" if block else "`input.normalize.mean` / `std`"
                fix = f"in {mf} set {field} to {_fmt_list(m)} / {_fmt_list(s)}"
                if _same_norm(inst, m, s):
                    fix = _already_set(inst, f"{field} {_fmt_list(m)} / {_fmt_list(s)}", None)
            else:
                cause = "the colours are normalised with a different mean/std than the manifest declares"
                fix = f"set the mean/std in {mf} to the values your training transform uses"
        elif code == "shape":
            srv, ref = run.b1_shapes or ([], [])
            cause = (f"the server builds a {srv or '?'} input but the reference builds {ref or '?'}"
                     if srv else "the server and the reference build inputs of different sizes")
            fix = (f"set the input size in {mf} (`preprocess.size` or `input.width/height`) to the training size; if "
                   "the training keeps the aspect ratio, use `preprocess.resize: keep_aspect`")
        elif code == "spatial":
            cause = "the picture is placed differently (another resize geometry: crop, aspect ratio or filter)"
            fix = "compare the two pictures in --report, then set `preprocess.resize` to how training resized"
        elif code == "resample":
            cause = "small resize-filter or JPEG-decoder differences (Go vs PIL/torchvision interpolate differently)"
            fix = "usually nothing to fix; if outputs (B2) or accuracy (C) suffer, try `preprocess.resample: bicubic`"
            eff = inst.spec
            if eff is not None and (eff.resample or ("bicubic" if eff.resize in ("center_crop", "keep_aspect")
                                                     else "bilinear")) == "bicubic":
                fix = "usually nothing to fix: the server already resizes bicubic, as PIL's BICUBIC"
        else:
            continue
        out.append({"code": code, "cause": cause, "fix": fix, "images": int(codes[code])})
    if "channel_swap" in codes:  # its normalisation note is the swap's side effect
        out = [c for c in out if c["code"] != "normalisation"]
    return out


def _b1_row(t: TierResult, run: CheckRun) -> dict:
    m = t.metrics
    ref = _ref_phrase(run.plan)
    n = m.get("images", len(run.images))
    row = {"tier": "B1", "name": TIER_NAMES["B1"], "status": t.status, "cause": None, "fix": None, "causes": []}
    if t.status == SKIP:
        row["finding"] = f"Not checked: {t.summary.rstrip('.')}."
        return row
    if t.status == ERROR:
        row["finding"] = f"The check could not run: {t.summary}."
        return row
    ml = m.get("mean_levels")
    if ml is None:
        row["finding"] = f"The server's input has a different size than {ref}'s: {t.summary}."
    elif run.plan and run.plan.b1_source == "manifest":
        how = {PASS: "exactly as", WARN: "almost as", FAIL: "differently from how"}[t.status]
        row["finding"] = (f"The server prepares photos {how} the manifest declares: average difference {ml:.1f} gray "
                          f"levels (out of 255) over {n} photos. This checks the server's code, not that the "
                          "manifest matches your training (pass --reference or --checkpoint for that).")
    elif t.status == PASS:
        row["finding"] = (f"The model sees the same picture as in training: average difference {ml:.1f} gray levels "
                          f"(out of 255) over {n} photos — fine. Reference: {ref}.")
    elif t.status == WARN:
        row["finding"] = (f"The model sees almost the same picture as in training: average difference {ml:.1f} gray "
                          f"levels (out of 255) over {n} photos — small, but above the "
                          f"{run.report.thresholds['b1_mean_warn']:g} a resize filter alone explains. Reference: {ref}.")
    else:
        row["finding"] = (f"The model sees a different picture than in training: average difference {ml:.1f} gray "
                          f"levels (out of 255) over {n} photos, where more than "
                          f"{run.report.thresholds['b1_mean_fail']:g} costs accuracy. Reference: {ref}.")
    if t.status in (WARN, FAIL):
        row["causes"] = causes_for(m.get("diagnosis") or {}, run)
        if row["causes"]:
            row["cause"], row["fix"] = row["causes"][0]["cause"], row["causes"][0]["fix"]
        elif m.get("diagnosis") is not None:
            row["cause"] = "no known fingerprint matched"
            row["fix"] = "compare the two pictures in --report with your training transform"
    return row


def _b2_row(t: TierResult, run: CheckRun) -> dict:
    m = t.metrics
    row = {"tier": "B2", "name": TIER_NAMES["B2"], "status": t.status, "cause": None, "fix": None}
    if t.status == SKIP:
        row["finding"] = f"Not compared: {t.summary.rstrip('.')}."
        return row
    if t.status == ERROR:
        row["finding"] = f"The comparison could not run: {t.summary}."
        return row
    word = {PASS: "the same", WARN: "mostly the same", FAIL: "different"}.get(t.status, "")
    if "matched_frac" in m and m.get("matched_frac") is not None:
        eff = max(m["matched"] + m["unmatched_ref"], m["matched"] + m["unmatched_srv"])
        row["finding"] = (f"The served model finds {word} objects as the original model: {m['matched']} of {eff} "
                          f"boxes match, confidence differs by {m['mean_dconf'] or 0:.3f} on average and boxes by "
                          f"{m['mean_box_px'] or 0:.1f} px.")
    elif "top1_agree" in m:
        row["finding"] = (f"The served model gives {word} top class as the original model on "
                          f"{100 * (m['top1_agree'] or 0):.0f}% of {m.get('n', '?')} photos; probabilities differ by "
                          f"up to {m.get('max_dprob') or 0:.3f}.")
    else:
        row["finding"] = f"The served outputs are {word} as the original model's: {t.summary}."
    if t.status in (WARN, FAIL):
        b1 = run.report.tier("B1")
        if b1 is not None and b1.status in (WARN, FAIL):
            row["cause"] = "the preprocessing difference found in B1"
            row["fix"] = "fix B1 first, then re-run"
        elif m.get("matched_frac") is None and "matched_frac" in m:
            row["cause"] = "nothing was detected on either side"
            row["fix"] = "use photos that contain the model's classes"
        else:
            row["cause"] = ("outputs differ although the photo is prepared the same way: the label file order, the "
                            "box format (postprocess.box_format), or an ONNX file that is not this checkpoint's export")
            row["fix"] = "compare labels and postprocess in the manifest with training; re-export if the sha256 differs"
    return row


def _c_row(t: TierResult, run: CheckRun) -> dict:
    m = t.metrics
    row = {"tier": "C", "name": TIER_NAMES["C"], "status": t.status, "cause": None, "fix": None}
    if t.status == SKIP:
        row["finding"] = ("Not measured: pass --labels (a COCO json for detection; a folder per class or a CSV "
                          "`image,label` for classification)." if not run.args.labels else f"Not measured: {t.summary}.")
        return row
    if t.status == ERROR:
        row["finding"] = f"The measurement could not run: {t.summary}."
        return row
    if t.status == FAIL and not m:
        row["finding"] = f"Could not score the labels: {t.summary}."
        row["cause"] = "the model's class names do not match the label names"
        row["fix"] = "make the label names match the model's labels file (case, spaces and '_' are ignored)"
        return row
    if m.get("kind") == "detection":
        conf = m.get("conf_threshold")
        if m.get("served_only"):
            row["finding"] = (f"On your {m['images']} labelled photos the served model scores mAP {m['map_served']:.1f} "
                              f"(mAP50 {m['map50_served']:.1f}) at its confidence threshold {conf}. There is no "
                              "reference model to compare with, so keep this number for your records "
                              "(--checkpoint adds the original model's score).")
        else:
            d = m["delta_map"]
            how = "the same within noise" if abs(d) <= run.args.max_map_drop / 2 else (
                "lower" if d < 0 else "higher")
            row["finding"] = (f"On your {m['images']} labelled photos the served model scores mAP {m['map_served']:.1f}, "
                              f"the original model {m['map_ref']:.1f} ({d:+.1f} points: {how}).")
    else:
        if m.get("served_only"):
            row["finding"] = (f"On your {m['images']} labelled photos the served model's top-1 accuracy is "
                              f"{m['top1_served']:.1f}% (top-5 {m['top5_served']:.1f}%). No reference model to "
                              "compare with.")
        else:
            d = m["delta_top1"]
            row["finding"] = (f"On your {m['images']} labelled photos: top-1 {m['top1_served']:.1f}% served vs "
                              f"{m['top1_ref']:.1f}% for the original model ({d:+.1f} points).")
    if t.status == WARN and m.get("served_only"):
        row["cause"] = "a near-zero score usually means the label names do not match the model's classes"
        row["fix"] = "check that the model's labels file uses the dataset's category names"
    elif t.status in (WARN, FAIL):
        b1 = run.report.tier("B1")
        row["cause"] = ("the preprocessing difference found in B1" if b1 is not None and b1.status in (WARN, FAIL)
                        else "the served model loses accuracy although B1 found no preprocessing difference")
        row["fix"] = "fix B1 first, then re-run" if "B1" in row["cause"] else "compare the outputs in B2 with --report"
    return row


def _load_row(t: TierResult, run: CheckRun) -> dict:
    return {"tier": "load", "name": TIER_NAMES["load"], "status": t.status,
            "finding": f"The server cannot load the model: {t.summary}",
            "cause": "the manifest and the model file disagree (size, layout or files)",
            "fix": f"correct {run.inst.manifest_path} as the error says, then restart the server"}


def summarise(run: CheckRun) -> dict:
    """The JSON object: {"verdict", "reason", "summary", "details"}."""
    rep = run.report
    name = run.inst.bundle.name
    rows = []
    for t in rep.tiers:
        fn = {"load": _load_row, "B1": _b1_row, "B2": _b2_row, "C": _c_row}.get(t.tier)
        if fn is not None:
            rows.append(fn(t, run))
    graded = [r for r in rows if r["status"] in (PASS, WARN, FAIL, INFO)]
    fails = [r for r in rows if r["status"] == FAIL]
    warns = [r for r in rows if r["status"] in (WARN, ERROR)]
    n = len(run.images)
    if fails:
        verdict = FAIL
        r = fails[0]
        what = {"load": "cannot be loaded by the server",
                "B1": "does not see photos the way it was trained" if run.plan and run.plan.b1_source != "manifest"
                else "is not prepared the way its manifest declares",
                "B2": "gives different answers than the original model",
                "C": "loses accuracy compared with the original model"}[r["tier"]]
        reason = f"{name} {what}" + (f": {r['cause']}." if r.get("cause") else ".")
    elif warns or not graded:
        verdict = WARN
        if not graded:
            reason = f"nothing about {name} could be compared (every check was skipped)."
        else:
            r = warns[0]
            what = {"B1": "its preprocessing differs slightly from the reference",
                    "B2": "some outputs differ from the original model",
                    "C": "its accuracy needs a look"}.get(r["tier"], "a check needs a look")
            if r["status"] == ERROR:
                what = f"the {r['name']} check could not run"
            reason = f"{name} mostly behaves as expected, but {what}" + (f": {r['cause']}." if r.get("cause") else ".")
    else:
        verdict = PASS
        b1 = rep.tier("B1")
        src = run.plan.b1_source if run.plan else "none"
        if src == "manifest":
            reason = (f"{name} prepares photos exactly as its manifest declares on {n} photos; whether that matches "
                      "your training needs --reference or --checkpoint.")
        else:
            reason = f"{name} behaves like its training pipeline on {n} photos"
            if b1 is not None and b1.metrics.get("mean_levels") is not None:
                reason += f" (preprocessing within {b1.metrics['mean_levels']:.1f} gray levels)"
            if rep.tier("B2") is not None and rep.tier("B2").status == SKIP:
                reason += "; outputs were not compared (no reference model)"
            reason += "."
    summary = {
        "model": name, "task": run.inst.bundle.task, "architecture": run.inst.bundle.architecture,
        "server": run.url, "manifest": str(run.inst.manifest_path),
        "images": n, "images_dir": str(run.args.images), "images_in_dir": run.images_total,
        "labels": str(run.args.labels) if run.args.labels else None,
        "reference": {"preprocessing": run.plan.b1.description if run.plan and run.plan.b1 is not None else None,
                      "outputs": run.plan.b2.description if run.plan and run.plan.b2 is not None else None,
                      "kind": run.plan.b1_source if run.plan else None},
        "checks": rows,
        "next_steps": next_steps(verdict, rows, run),
    }
    details = rep.to_dict()
    details["table"] = rep.table()
    return {"verdict": verdict, "reason": reason, "summary": summary, "details": details}


def next_steps(verdict: str, rows: List[dict], run: CheckRun) -> List[str]:
    out = []
    model = run.inst.bundle.name
    for r in rows:
        if r["status"] in (FAIL, WARN) and r.get("fix") and r["fix"] not in ("fix B1 first, then re-run",):
            out.append(r["fix"][0].upper() + r["fix"][1:] + ".")
    if any(r["status"] == FAIL for r in rows) and any(".yaml" in s and "is already set in" not in s for s in out):
        out.append("Restart `visionserve serve` (it reads a manifest once), then re-run this check.")
    st = {r["tier"]: r["status"] for r in rows}
    if st.get("B2") == SKIP and not (run.args.reference or run.args.checkpoint):
        out.append(f"To compare outputs with the original model, re-run with --checkpoint PATH (the checkpoint "
                   f"{model} was exported from) or --reference SCRIPT.py.")
    if st.get("C") == SKIP and not run.args.labels:
        out.append("To measure accuracy, add --labels (a COCO json for detection; a folder per class or a CSV for "
                   "classification), ideally 200 photos or more.")
    if not out and verdict == PASS:
        out.append("Nothing to fix.")
    return out


# --------------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------------

def host_paths(obj, mapping: Optional[Dict[str, str]] = None):
    """Every string in `obj` with in-container path prefixes replaced by the host paths they are
    mounted from. The Go wrapper passes the mounts as $VISIONSERVE_CHECK_PATHS (JSON
    {container prefix: host path}), so a fix names the user's own manifest, not /root/.models/..."""
    if mapping is None:
        try:
            mapping = json.loads(os.environ.get("VISIONSERVE_CHECK_PATHS") or "{}")
        except ValueError:
            mapping = {}
    if not mapping:
        return obj
    import re
    pat = re.compile("(?:" + "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r")(?!\w)")

    def fix(x):
        if isinstance(x, str):
            return pat.sub(lambda m: mapping[m.group(0)], x)
        if isinstance(x, dict):
            return {k: fix(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(fix(v) for v in x)
        return x
    return fix(obj)


def _wrap(text: str, width: int) -> List[str]:
    # paths and URLs stay whole: a wrapped path cannot be copied back
    return textwrap.wrap(text, width=max(40, width), break_long_words=False, break_on_hyphens=False) or [""]


def render_text(out: dict, width: int = 100) -> str:
    s = out["summary"]
    lines = [f"{out['verdict']}: {out['reason']}", ""]
    lines.append(f"Summary: {s['model']} ({s['task']}, {s['architecture']}) on {s['server']}, {s['images']} photo(s) "
                 f"from {s['images_dir']}")
    rows = s["checks"]
    w0 = max(len("check"), *(len(r["name"]) for r in rows))
    lines.append(f"  {'check'.ljust(w0)}  status  what we found")
    pad = " " * (2 + w0 + 2 + 8)
    for r in rows:
        body = _wrap(r["finding"], width - len(pad))
        lines.append(f"  {r['name'].ljust(w0)}  {r['status'].ljust(6)}  {body[0]}")
        lines += [pad + b for b in body[1:]]
        for label, key in (("Likely cause", "cause"), ("Fix", "fix")):
            if r.get(key):
                for i, b in enumerate(_wrap(f"{label}: {r[key]}.", width - len(pad))):
                    lines.append(pad + ("" if i == 0 else "  ") + b)
    lines += ["", "Next steps"]
    lines += [f"  {i}. {t}" for i, t in enumerate(s["next_steps"], 1)]
    lines += ["", "Details (the converter's tier report)"]
    lines += ["  " + ln for ln in out["details"]["table"].splitlines()]
    return "\n".join(lines)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    def setup_error(msg: str) -> int:
        if args.json:
            print(json.dumps({"verdict": "ERROR", "reason": msg, "summary": {}, "details": {}}))
        log(f"error: {msg}")
        return EXIT_SETUP

    try:
        inst, url, photos = prepare(args)
        with tempfile.TemporaryDirectory(prefix="vscheck-") as tmp:
            run = run_check(args, inst, url, photos, Path(tmp))
    except SetupError as e:
        return setup_error(str(e))
    out = summarise(run)
    if args.report:
        out["summary"]["report"] = str(args.report)
    out = host_paths(out)
    if args.report:
        from .htmlreport import render_html
        Path(args.report).write_text(render_html(out, run.figures), encoding="utf-8")
        log(f"check: HTML report written to {out['summary']['report']}")
    if args.json:
        from .report import _jsonable
        print(json.dumps(_jsonable(out)))
    else:
        print(render_text(out))
    sys.stdout.flush()
    return EXIT_FAIL if out["verdict"] == FAIL else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
