"""`visionserve convert hf <dir | org/name>` — HuggingFace transformers checkpoints.

A HuggingFace checkpoint says what it IS (config.json: model_type / architectures), so unlike the
generic formats there is no --task: the family dispatches to a handler that maps the checkpoint onto
an EXISTING VisionServe Go architecture and reproduces that package's exact I/O contract:

    model_type / architectures         Go architecture(s)          Go package
    grounding-dino                     grounding-dino              internal/models/groundingdino
    siglip                             siglip-image + siglip-text  internal/models/siglip
    clip                               clip + clip-text            internal/models/clip
    rt_detr, rt_detr_v2                rt-detr                     internal/models/rtdetr
    *ForImageClassification            efficientnet (generic)      internal/models/classification
    *ForDepthEstimation                midas (generic)            internal/models/depth

Anything else (DETR's softmax + no-object column, OWL-ViT/OWLv2's per-image text queries, ...) is
REFUSED: no VisionServe decoder reads those outputs, and a graph nobody can decode correctly is
worse than no graph.

Every graph is checked against the framework (parity) on the exact tensor the Go preprocess feeds
(common.to_nchw with the manifest's own mean/std). The license comes from the model card and
cannot be overridden by --license (common.resolve_license).

Heavy imports (torch, transformers, huggingface_hub, onnx) happen inside functions only: the CLI
imports every family to build its parser, and the TensorFlow venv has none of them.
"""
from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

from ..common import (IMAGENET_MEAN, IMAGENET_STD, Bundle, ConvertError, license_scan, log, parity,
                      resolve_license, sample_image, to_nchw)
from ..constants import HUB_ID_RE, parse_wxh
from ..reference import Reference

HELP = ("HuggingFace transformers checkpoint: a local model directory (config.json + weights) or a "
        "hub id such as google/siglip-base-patch16-224. Supported: grounding-dino, siglip, clip, "
        "rt_detr, *ForImageClassification, *ForDepthEstimation.")

# PIL resample codes used by preprocessor_config.json
_RESAMPLE = {0: "nearest", 1: "lanczos", 2: "bilinear", 3: "bicubic", 4: "box", 5: "hamming"}

# model_types that look like they might fit but whose OUTPUTS no Go architecture decodes.
_REFUSED = {
    "detr": "DETR scores classes with a SOFTMAX over C+1 columns (the last is 'no object'); the "
            "rt-detr decoder applies a per-class SIGMOID with no no-object column, so every score "
            "would be wrong",
    "conditional_detr": "Conditional-DETR's head is not the rt-detr contract (verify-before-decode)",
    "deformable_detr": "Deformable-DETR needs a custom CUDA op to trace and its decode (top-k over "
                       "queries x classes) is not the rt-detr per-query argmax",
    "yolos": "YOLOS scores with a softmax + no-object column like DETR",
    "owlvit": "OWL-ViT emits per-image-query logits against text embeddings computed in the same "
              "graph; the owlvit Go package decodes a different, pre-exported graph",
    "owlv2": "OWLv2 emits per-image-query logits against text embeddings computed in the same "
             "graph; no VisionServe architecture decodes this export",
    "siglip2": "SigLIP-2 uses the Gemma tokenizer and NaFlex patching; internal/models/siglip "
               "implements the SigLIP-1 Unigram tokenizer and a fixed 224x224 input only",
    "mask2former": "segmentation masks per query are not part of any converter-reachable Go decoder",
    "maskformer": "segmentation masks per query are not part of any converter-reachable Go decoder",
}


# --------------------------------------------------------------------------------------------
# CLI surface (stdlib + numpy only)
# --------------------------------------------------------------------------------------------

def help_for(fmt: str) -> str:
    return HELP


def add_arguments(p, fmt: str) -> None:
    p.add_argument("--revision", help="hub revision (branch, tag or commit) to convert; default main")
    p.add_argument("--input", metavar="WxH",
                   help="override the input resolution read from preprocessor_config.json (required when "
                        "the processor declares none, e.g. GLPN). Ignored for grounding-dino (800x800) "
                        "and siglip (224x224), whose Go packages fix the size.")
    p.add_argument("--conf-threshold", type=float, default=None,
                   help="postprocess conf_threshold written to the manifest (rt-detr default 0.5, "
                        "grounding-dino 0.3)")


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------

def convert(args, workdir: Path) -> list:
    workdir = Path(workdir)
    src, card_license, origin = _fetch(args)
    license_scan(source=src)  # pickles, metadata, a LICENSE file next to the weights
    lic = resolve_license(args.license, card_license, f"the model card of {origin}")
    cfg = _read_json(src / "config.json")
    if cfg is None:
        raise ConvertError(f"{src} has no config.json — not a HuggingFace transformers checkpoint")
    mt = (cfg.get("model_type") or "").strip()
    archs = cfg.get("architectures") or []
    handler = _dispatch(mt, archs)
    log(f"hf: {origin}  model_type={mt!r} architectures={archs} -> {handler.__name__[1:]}  license={lic}")
    ctx = _Ctx(args=args, src=src, cfg=cfg, license=lic, origin=origin, work=workdir)
    return handler(ctx)


def _dispatch(mt: str, archs):
    if mt == "grounding-dino":
        return _grounding_dino
    if mt == "siglip":
        return _siglip
    if mt == "clip":
        return _clip
    if mt in ("rt_detr", "rt_detr_v2"):
        return _rt_detr
    if mt in _REFUSED:
        raise ConvertError(f"model_type {mt!r} ({', '.join(archs) or 'no architectures'}) is not supported: "
                           f"{_REFUSED[mt]}. No VisionServe architecture decodes it.")
    if any("ForImageClassification" in a for a in archs):
        return _classification
    if any(a.endswith("ForDepthEstimation") for a in archs):
        return _depth
    raise ConvertError(f"model_type {mt!r} (architectures {archs or 'none declared'}) is not supported: no "
                       "VisionServe architecture decodes its outputs. Supported: grounding-dino, siglip, clip, "
                       "rt_detr, *ForImageClassification, *ForDepthEstimation.")


class _Ctx:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def notes(self, *extra):
        return [f"hf {self.origin}", *extra]


# --------------------------------------------------------------------------------------------
# Source + license
# --------------------------------------------------------------------------------------------

_WEIGHT_SUFFIXES = (".safetensors",)
_SIDE_SUFFIXES = (".json", ".txt", ".model")


def _fetch(args):
    """Return (local_dir, license_from_card, human_origin). For a hub id the license is gated
    BEFORE the weights are downloaded."""
    p = Path(args.source)
    if p.is_dir():
        return p, card_license(p / "README.md"), str(p)
    if p.exists():
        raise ConvertError(f"{p} is a file; the hf format takes a model DIRECTORY (config.json + weights) "
                           "or a hub id like google/vit-base-patch16-224")
    if not HUB_ID_RE.match(args.source):
        raise ConvertError(f"{args.source!r} is neither an existing directory nor a hub id (org/name)")

    from huggingface_hub import HfApi, snapshot_download
    try:
        info = HfApi().model_info(args.source, revision=getattr(args, "revision", None))
    except Exception as e:  # network, 404, gated
        raise ConvertError(f"cannot read {args.source} from the HuggingFace hub: {e}")
    hub_lic = None
    if info.card_data is not None:
        hub_lic = _one_license(info.card_data.get("license"), args.source)
    # Gate now: a copyleft card must not cost a multi-GB download before it is refused.
    resolve_license(args.license, hub_lic, f"the model card of {args.source}")

    top = [s.rfilename for s in (info.siblings or []) if "/" not in s.rfilename]
    wanted = [f for f in top if f.endswith(_SIDE_SUFFIXES) or f == "README.md"]
    weights = [f for f in top if f.endswith(_WEIGHT_SUFFIXES)]
    if not weights:  # no safetensors: fall back to PyTorch pickles (scanned for AGPL markers)
        weights = [f for f in top if re.fullmatch(r"pytorch_model.*\.bin(\.index\.json)?", f)]
    if not weights:
        raise ConvertError(f"{args.source} has no PyTorch weights (model*.safetensors / pytorch_model*.bin) "
                           "at the repository root")
    log(f"hf: downloading {args.source}@{info.sha[:12]}: {', '.join(sorted(weights))}")
    local = Path(snapshot_download(args.source, revision=info.sha, allow_patterns=wanted + weights))
    readme_lic = card_license(local / "README.md")
    if hub_lic and readme_lic and hub_lic.lower() != readme_lic.lower():
        raise ConvertError(f"{args.source}: hub card data says {hub_lic!r} but README.md says {readme_lic!r}")
    return local, hub_lic or readme_lic, f"{args.source}@{info.sha[:12]}"


def _one_license(v, where) -> Optional[str]:
    if v is None or v == "" or v == []:
        return None
    if isinstance(v, (list, tuple)):
        if len(v) != 1:
            raise ConvertError(f"{where} declares several licenses {list(v)}; convert it only after checking "
                               "which one covers the weights")
        v = v[0]
    return str(v).strip()


def card_license(readme: Path) -> Optional[str]:
    """`license:` from a model card's YAML front matter (scalar or one-item list), else None."""
    try:
        # utf-8-sig: a BOM before the front matter must not hide `license: agpl-3.0` (the user's
        # --license would then be taken instead); \r\n cards are normalised the same way.
        text = Path(readme).read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return None
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    m = re.match(r"^---\s*\n(.*?)\n---\s*(\n|$)", text, re.S)
    if not m:
        return None
    lines = m.group(1).splitlines()
    for i, line in enumerate(lines):
        mm = re.match(r"^license\s*:\s*(.*?)\s*$", line)
        if not mm:
            continue
        val = mm.group(1).strip().strip("'\"")
        if val.startswith("["):
            items = [x.strip().strip("'\"") for x in val.strip("[]").split(",") if x.strip()]
            return _one_license(items, readme)
        if val:
            return val
        items = []
        for nxt in lines[i + 1:]:
            li = re.match(r"^\s*-\s*(.+?)\s*$", nxt)
            if not li:
                break
            items.append(li.group(1).strip("'\""))
        return _one_license(items, readme)
    return None


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None


# --------------------------------------------------------------------------------------------
# Preprocessing (preprocessor_config.json -> what the Go side feeds)
# --------------------------------------------------------------------------------------------

def _geometry(ctx, go_resample="bilinear", keep_aspect=False):
    """(width, height, mean, std, notes) for an image tower, as the Go preprocess must feed it:
    squash to WxH, /255, (x-mean)/std. Divergences from the HF processor that the Go side cannot
    reproduce (centre crop, shortest-edge resize, resample filter) are recorded in the manifest
    header, not hidden. keep_aspect=True: the caller serves the processor's keep-aspect rule
    (manifest input.keep_aspect), so that is no divergence."""
    pp = _read_json(ctx.src / "preprocessor_config.json") or {}
    notes = []
    w = h = None
    if pp.get("do_center_crop") and pp.get("crop_size"):
        w, h = _size_wh(pp["crop_size"])
        notes.append(f"HF resizes then centre-crops to {w}x{h}; VisionServe squashes the whole frame to {w}x{h}")
    elif pp.get("do_resize", True) and pp.get("size") is not None:
        w, h = _size_wh(pp["size"])
        if isinstance(pp["size"], dict) and "shortest_edge" in pp["size"]:
            notes.append(f"HF resizes the shortest edge to {w}; VisionServe squashes to {w}x{h}")
        if pp.get("keep_aspect_ratio") and not keep_aspect:
            notes.append(f"HF keeps the aspect ratio (multiple of {pp.get('ensure_multiple_of', 1)}); "
                         f"VisionServe squashes to {w}x{h}")
    if getattr(ctx.args, "input", None):
        w, h = parse_wxh(ctx.args.input)
    if not w or not h:
        raise ConvertError("preprocessor_config.json declares no fixed input size (e.g. GLPN's size_divisor); "
                           "pass --input WxH")
    mult = pp.get("ensure_multiple_of") or pp.get("size_divisor")
    if mult and (w % mult or h % mult):
        raise ConvertError(f"input {w}x{h} is not a multiple of {mult}, which this model requires")

    if pp.get("do_normalize", True) and pp.get("image_mean") is not None:
        mean, std = list(map(float, pp["image_mean"])), list(map(float, pp["image_std"]))
    elif pp.get("do_normalize", True) and not pp:
        mean, std = list(IMAGENET_MEAN), list(IMAGENET_STD)
        notes.append("no preprocessor_config.json: assumed ImageNet mean/std")
    else:
        mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    # Go always divides by 255. A processor that rescales by rf (or not at all, rf=1) is folded in:
    # (x*255*rf - m)/s == (x - m/k)/(s/k) with k = 255*rf.
    rf = float(pp.get("rescale_factor", 1 / 255)) if pp.get("do_rescale", True) else 1.0
    k = 255.0 * rf
    if abs(k - 1.0) > 1e-6:
        mean, std = [m / k for m in mean], [s / k for s in std]
        notes.append(f"processor rescale factor {rf:g} folded into mean/std")
    if len(mean) == 1:
        mean, std = mean * 3, std * 3

    rs = _RESAMPLE.get(pp.get("resample"), None)
    if rs and rs != go_resample:
        notes.append(f"HF resamples {rs}; the Go preprocess resizes {go_resample} (small input drift, "
                     "parity is measured on the Go-fed tensor)")
    return w, h, mean, std, notes


def _size_wh(size):
    if isinstance(size, int):
        return size, size
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return int(size[1]), int(size[0])
    if isinstance(size, dict):
        if "height" in size and "width" in size:
            return int(size["width"]), int(size["height"])
        if "shortest_edge" in size:
            return int(size["shortest_edge"]), int(size["shortest_edge"])
    return None, None


def _labels(ctx, n=None):
    if getattr(ctx.args, "labels", None):
        lines = [l.strip() for l in Path(ctx.args.labels).read_text().splitlines()]
        return [l for l in lines if l]
    id2 = ctx.cfg.get("id2label") or {}
    if not id2:
        return None
    id2 = {int(k): v for k, v in id2.items()}
    n = n or len(id2)
    missing = [i for i in range(n) if i not in id2]
    if missing:
        raise ConvertError(f"config.id2label has gaps (no names for ids {missing[:5]}...); pass --labels")
    return [str(id2[i]).replace("\n", " ") for i in range(n)]


# --------------------------------------------------------------------------------------------
# Export helpers
# --------------------------------------------------------------------------------------------

def _export(module, example, path: Path, input_names, output_names, dynamic_axes, opset, legacy_only=False):
    """torch.onnx.export, legacy tracer first (static shapes where not declared dynamic — what the
    contract check reads), dynamo as a fallback. The result is re-saved as ONE file when < 2 GB so
    the manifest's sha256 pins a single artifact."""
    import onnx
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    before = set(path.parent.iterdir())

    def created():  # only files the exporter itself wrote next to `path`
        return [f for f in path.parent.iterdir() if f not in before and f != path and f.is_file()]

    kw = dict(input_names=list(input_names), output_names=list(output_names),
              dynamic_axes=dynamic_axes or None, do_constant_folding=True)
    with torch.no_grad():
        try:
            torch.onnx.export(module, tuple(example), str(path), opset_version=opset, dynamo=False, **kw)
        except Exception as e:
            if legacy_only:
                raise ConvertError(f"ONNX export failed: {type(e).__name__}: {e}")
            log(f"  legacy exporter failed ({type(e).__name__}: {str(e)[:200]}); retrying with dynamo")
            for f in created() + ([path] if path.exists() else []):
                f.unlink()
            try:
                torch.onnx.export(module, tuple(example), str(path), opset_version=max(opset, 18),
                                  dynamo=True, **kw)
            except Exception as e2:
                raise ConvertError(f"ONNX export failed with both exporters: {type(e2).__name__}: {e2}")
    m = onnx.load(str(path))
    sidecars = created()
    if m.ByteSize() < 2_000_000_000:
        onnx.save(m, str(path))
        keep = set()
    else:
        keep = {path.parent / (path.name + ".data")}
        onnx.save(m, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                  location=path.name + ".data")
    for f in sidecars:
        if f not in keep and f.exists():
            f.unlink()
    return path


def _features(out):
    """get_image_features / get_text_features return a tensor in transformers 4 and a
    BaseModelOutputWithPooling in 5; the features are its pooler_output."""
    import torch
    if isinstance(out, torch.Tensor):
        return out
    return out.pooler_output


def _np(t):
    return t.detach().cpu().numpy()


def _tol(ctx):
    return float(getattr(ctx.args, "tolerance", 1e-3))


def _opset(ctx):
    return int(getattr(ctx.args, "opset", 17))


def _load(auto_cls, src):
    import torch
    try:
        m = auto_cls.from_pretrained(str(src), torch_dtype=torch.float32)
    except Exception as e:
        raise ConvertError(f"transformers could not load {src}: {type(e).__name__}: {e}")
    return m.eval()


def _reference(fn, what):
    """Run the framework once BEFORE exporting, so a model that rejects the input (fixed position
    embeddings vs --input, a bad config) fails with one clear line instead of two exporter traces."""
    import torch
    try:
        with torch.no_grad():
            return fn()
    except Exception as e:
        raise ConvertError(f"{what}: the model rejects the input VisionServe would feed it: "
                           f"{type(e).__name__}: {e}")


def _images(w, h, mean, std, n=1):
    return np.concatenate([to_nchw(sample_image(w, h, seed=i), mean, std) for i in range(n)], 0)


# --------------------------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------------------------

def _classification(ctx):
    import torch
    from transformers import AutoModelForImageClassification
    model = _load(AutoModelForImageClassification, ctx.src)
    w, h, mean, std, notes = _geometry(ctx)

    class Net(torch.nn.Module):
        def __init__(s, m):
            super().__init__()
            s.m = m

        def forward(s, pixel_values):
            return s.m(pixel_values=pixel_values).logits

    net = Net(model).eval()
    x = _images(w, h, mean, std)
    ref = _np(_reference(lambda: net(torch.from_numpy(x)), f"classification at {w}x{h}"))
    path = _export(net, (torch.from_numpy(x),), ctx.work / "model.onnx", ["pixel_values"], ["logits"],
                   None, _opset(ctx))
    err = parity(path, {"pixel_values": x}, [ref], _tol(ctx), " (classification)")
    labels = _labels(ctx, ref.shape[-1])
    return [Bundle(name=ctx.args.name, task="classification", architecture="efficientnet", license=ctx.license,
                   width=w, height=h, onnx={"model": str(path)}, mean=mean, std=std,
                   postprocess={"type": "classification", "max_detections": 5}, labels=labels,
                   notes=ctx.notes(f"{type(model).__name__}: logits [1,{ref.shape[-1]}] -> softmax (Go)",
                                   f"parity max|d|/scale {err:.2e}", *notes),
                   reference=HFImageReference(ctx.src, model, "classification", labels, divergence=_divergence(notes)))]


def _depth(ctx):
    import torch
    from transformers import AutoModelForDepthEstimation
    model = _load(AutoModelForDepthEstimation, ctx.src)
    keep, multiple = _dpt_keep_aspect(ctx)
    w, h, mean, std, notes = _geometry(ctx, go_resample="bicubic" if keep else "bilinear", keep_aspect=keep)

    class Net(torch.nn.Module):
        def __init__(s, m):
            super().__init__()
            s.m = m

        def forward(s, pixel_values):
            d = s.m(pixel_values=pixel_values).predicted_depth
            return d if d.dim() == 3 else d.reshape(d.shape[0], d.shape[-2], d.shape[-1])

    net = Net(model).eval()
    x = _images(w, h, mean, std)
    ref = _np(_reference(lambda: net(torch.from_numpy(x)), f"depth at {w}x{h}"))
    if ref.ndim != 3:
        raise ConvertError(f"depth: predicted_depth has shape {ref.shape}; the Go depth decoder needs [1,H,W]")
    # Keep-aspect serving feeds a different HxW per image: the graph needs dynamic H/W, and parity
    # is checked at sizes OTHER than the traced one too (a tracer that baked the 518 grid into
    # e.g. the position-embedding interpolation would pass at 518x518 and be wrong elsewhere).
    extra = []
    if keep:
        from ..reference import dpt_keep_aspect_size
        try:
            for ow, oh in ((848, 480), (480, 640)):  # a landscape and a portrait photo
                sw, sh = dpt_keep_aspect_size(ow, oh, w, h, multiple)
                xs = _images(sw, sh, mean, std)
                with torch.no_grad():
                    extra.append((f"{sw}x{sh}", xs, _np(net(torch.from_numpy(xs)))))
        except Exception as e:  # e.g. DPT's own ViT reassembles a SQUARE patch grid
            log(f"  depth: the model rejects non-square inputs ({type(e).__name__}: {str(e)[:120]}); "
                "serving the squash instead of the processor's keep-aspect resize")
            keep, multiple, extra = False, 0, []
            w, h, mean, std, notes = _geometry(ctx)
            notes.append(f"the model itself rejects non-square inputs ({type(e).__name__}), so keep-aspect "
                         "cannot be served")
    dyn = ({"pixel_values": {2: "height", 3: "width"}, "predicted_depth": {1: "height", 2: "width"}}
           if keep else None)
    with _traceable_int(model if keep else None):
        path = _export(net, (torch.from_numpy(x),), ctx.work / "model.onnx", ["pixel_values"], ["predicted_depth"],
                       dyn, _opset(ctx))
    err = parity(path, {"pixel_values": x}, [ref], _tol(ctx), f" (depth {w}x{h})")
    for size, xs, rs in extra:
        err = max(err, parity(path, {"pixel_values": xs}, [rs], _tol(ctx), f" (depth {size})"))
    kind = ("METRIC depth (larger = farther)" if ctx.cfg.get("model_type") in ("glpn", "zoedepth")
            or ctx.cfg.get("depth_estimation_type") == "metric" else "relative inverse depth (larger = nearer)")
    geo = ([f"keep-aspect resize as {_pp_type(ctx)} does: scale closest to 1 of {w}/W, {h}/H, sides rounded "
            f"to a multiple of {multiple or 1}, bicubic (dynamic H/W graph)"] if keep else [])
    return [Bundle(name=ctx.args.name, task="depth", architecture="midas", license=ctx.license,
                   width=w, height=h, onnx={"model": str(path)}, mean=mean, std=std,
                   keep_aspect=keep, multiple_of=multiple if keep else 0,
                   postprocess={"type": "depth"},
                   notes=ctx.notes(f"{type(model).__name__}: predicted_depth [1,H,W], "
                                   f"{kind}; Go min-max normalises it", *geo,
                                   f"parity max|d|/scale {err:.2e}", *notes),
                   reference=HFImageReference(ctx.src, model, "depth", divergence=_divergence(notes)))]


@contextlib.contextmanager
def _traceable_int(model):
    """Keep `int(<traced size>)` symbolic in the model's own module while exporting with dynamic H/W.
    Depth Anything's head upsamples to (int(patch_h * 14), int(patch_w * 14)); under the legacy
    tracer those int() calls freeze the TRACED size into the graph, so a dynamic-axes export returns
    a 518x518 map for every input (caught by the multi-size parity below). Tensors pass through
    unchanged, everything else still goes to the builtin."""
    import builtins
    import sys
    mod = sys.modules.get(type(model).__module__) if model is not None else None
    if mod is None or "int" in vars(mod):
        yield
        return
    import torch
    mod.int = lambda v: v if isinstance(v, torch.Tensor) else builtins.int(v)
    try:
        yield
    finally:
        del mod.int


def _pp_type(ctx) -> str:
    return str((_read_json(ctx.src / "preprocessor_config.json") or {}).get("image_processor_type") or "")


def _dpt_keep_aspect(ctx):
    """(keep_aspect, multiple_of) when the checkpoint's processor resizes with DPT's keep-aspect rule
    (Depth Anything V1/V2, DPT): the Go depth preprocess reproduces exactly that rule. Other
    processors (ZoeDepth pads, GLPN uses size_divisor) keep the squash and its divergence note."""
    pp = _read_json(ctx.src / "preprocessor_config.json") or {}
    if (_pp_type(ctx).startswith("DPTImageProcessor") and pp.get("do_resize", True) and pp.get("keep_aspect_ratio")
            and not pp.get("do_pad") and isinstance(pp.get("size"), dict) and "height" in pp["size"]):
        return True, int(pp.get("ensure_multiple_of") or 1)
    return False, 0


def _rt_detr(ctx):
    import torch
    from transformers import AutoModelForObjectDetection
    model = _load(AutoModelForObjectDetection, ctx.src)
    w, h, mean, std, notes = _geometry(ctx)
    pp = _read_json(ctx.src / "preprocessor_config.json") or {}
    if pp.get("do_pad"):
        notes.append("HF processor pads; VisionServe squashes (letterbox: false)")

    class Net(torch.nn.Module):
        def __init__(s, m):
            super().__init__()
            s.m = m

        def forward(s, pixel_values):
            o = s.m(pixel_values=pixel_values)
            return o.logits, o.pred_boxes

    net = Net(model).eval()
    x = _images(w, h, mean, std)
    ref = [_np(t) for t in _reference(lambda: net(torch.from_numpy(x)), f"rt-detr at {w}x{h}")]
    q, c = ref[0].shape[1], ref[0].shape[2]
    if c == 4:
        raise ConvertError("rt-detr: this head has exactly 4 classes, so logits [1,Q,4] and pred_boxes [1,Q,4] "
                           "have the same shape and internal/models/rtdetr (which finds the boxes by last dim "
                           "== 4) cannot tell them apart")
    if ref[1].shape != (1, q, 4):
        raise ConvertError(f"rt-detr: pred_boxes has shape {ref[1].shape}, expected [1,{q},4]")
    if not (0.0 <= float(ref[1].min()) and float(ref[1].max()) <= 1.0):
        raise ConvertError("rt-detr: pred_boxes are not normalised to [0,1]; the Go decoder expects cxcywh/[0,1]")
    path = _export(net, (torch.from_numpy(x),), ctx.work / "model.onnx", ["pixel_values"],
                   ["logits", "pred_boxes"], None, _opset(ctx))
    err = parity(path, {"pixel_values": x}, ref, _tol(ctx), " (rt-detr)")
    conf = ctx.args.conf_threshold if getattr(ctx.args, "conf_threshold", None) is not None else 0.5
    labels = _labels(ctx, c)
    return [Bundle(name=ctx.args.name, task="detection", architecture="rt-detr", license=ctx.license,
                   width=w, height=h, onnx={"model": str(path)}, mean=mean, std=std, letterbox=False,
                   postprocess={"type": "rt-detr", "box_format": "cxcywh", "conf_threshold": conf,
                                "max_detections": min(300, q)},
                   labels=labels,
                   notes=ctx.notes(f"{type(model).__name__}: logits [1,{q},{c}] sigmoid (NMS-free) + "
                                   "pred_boxes cxcywh normalised", f"parity max|d|/scale {err:.2e}", *notes),
                   reference=HFImageReference(ctx.src, model, "detection", labels, conf=conf,
                                              divergence=_divergence(notes)))]


# ---- SigLIP ---------------------------------------------------------------------------------

SIGLIP_SIZE = 224   # internal/models/siglip ImageSize (fixed)
SIGLIP_TEXT_LEN = 64  # internal/models/siglip tokenizer maxLen (fixed)


def _siglip(ctx):
    import torch
    from transformers import AutoModel, AutoTokenizer
    vc, tc = ctx.cfg.get("vision_config") or {}, ctx.cfg.get("text_config") or {}
    if vc.get("image_size", SIGLIP_SIZE) != SIGLIP_SIZE:
        raise ConvertError(f"siglip: image_size {vc.get('image_size')} — internal/models/siglip feeds a fixed "
                           f"{SIGLIP_SIZE}x{SIGLIP_SIZE} crop; only *-224 checkpoints are servable")
    if tc.get("max_position_embeddings", SIGLIP_TEXT_LEN) != SIGLIP_TEXT_LEN:
        raise ConvertError(f"siglip: text context {tc.get('max_position_embeddings')} != {SIGLIP_TEXT_LEN}, "
                           "the length the Go SigLIP tokenizer pads to")
    tokjson = ctx.src / "tokenizer.json"
    tj = _read_json(tokjson)
    if tj is None or (tj.get("model") or {}).get("type") != "Unigram":
        raise ConvertError("siglip: the Go text tower needs a SentencePiece-Unigram tokenizer.json next to the "
                           "model; this checkpoint has none (or a different tokenizer model)")
    if not any(p[0] == "</s>" for p in tj["model"].get("vocab", [])):
        raise ConvertError("siglip: tokenizer.json has no '</s>' piece (the Go tokenizer's EOS/pad)")
    pp = _read_json(ctx.src / "preprocessor_config.json") or {}
    if [round(float(v), 4) for v in pp.get("image_mean", [0.5] * 3)] != [0.5] * 3 or \
       [round(float(v), 4) for v in pp.get("image_std", [0.5] * 3)] != [0.5] * 3:
        raise ConvertError("siglip: internal/models/siglip normalises with mean=std=0.5; this checkpoint's "
                           f"processor declares {pp.get('image_mean')}/{pp.get('image_std')}")
    mean = std = [0.5, 0.5, 0.5]
    model = _load(AutoModel, ctx.src)

    class Image(torch.nn.Module):
        def __init__(s, m):
            super().__init__()
            s.v = m.vision_model

        def forward(s, pixel_values):
            return s.v(pixel_values=pixel_values).pooler_output

    class Text(torch.nn.Module):
        def __init__(s, m):
            super().__init__()
            s.t = m.text_model

        def forward(s, input_ids):
            return s.t(input_ids=input_ids).pooler_output

    tol, ops = _tol(ctx), _opset(ctx)
    (ctx.work / "image").mkdir(parents=True, exist_ok=True)
    (ctx.work / "text").mkdir(parents=True, exist_ok=True)

    x = _images(SIGLIP_SIZE, SIGLIP_SIZE, mean, std, n=2)  # batch 2: the crop namer batches N crops
    ip = _export(Image(model).eval(), (torch.from_numpy(x[:1]),), ctx.work / "image" / "model.onnx",
                 ["pixel_values"], ["image_embeds"], {"pixel_values": {0: "N"}, "image_embeds": {0: "N"}}, ops)
    with torch.no_grad():
        iref = _np(_features(model.get_image_features(pixel_values=torch.from_numpy(x))))
    ierr = parity(ip, {"pixel_values": x}, [iref], tol, " (siglip-image vs get_image_features)")

    tok = AutoTokenizer.from_pretrained(str(ctx.src))
    phrases = ["a photo of a cat", "remote control", "water bottle"]
    ids = tok(phrases, padding="max_length", max_length=SIGLIP_TEXT_LEN, truncation=True,
              return_tensors="np")["input_ids"].astype(np.int64)
    tp = _export(Text(model).eval(), (torch.from_numpy(ids[:1]),), ctx.work / "text" / "model.onnx",
                 ["input_ids"], ["text_embeds"], {"input_ids": {0: "N"}, "text_embeds": {0: "N"}}, ops)
    with torch.no_grad():
        tref = _np(_features(model.get_text_features(input_ids=torch.from_numpy(ids))))
    terr = parity(tp, {"input_ids": ids}, [tref], tol, " (siglip-text vs get_text_features)")
    d = iref.shape[1]

    img = Bundle(name=f"{ctx.args.name}-image", task="embed", architecture="siglip-image", license=ctx.license,
                 width=SIGLIP_SIZE, height=SIGLIP_SIZE, onnx={"model": str(ip)}, mean=mean, std=std,
                 single_file=False,
                 notes=ctx.notes(f"SigLIP image tower: pixel_values [N,3,224,224] -> image_embeds [N,{d}] "
                                 "(not normalised; Go L2-normalises). Squash, BICUBIC, mean=std=0.5.",
                                 f"parity vs get_image_features max|d|/scale {ierr:.2e}",
                                 f"text tower: {ctx.args.name}-text"),
                 reference=HFImageReference(ctx.src, model, "embed"))
    txt = Bundle(name=f"{ctx.args.name}-text", task="embed", architecture="siglip-text", license=ctx.license,
                 width=SIGLIP_TEXT_LEN, height=1, onnx={"model": str(tp)}, single_file=False,
                 extra_files={"tokenizer.json": str(tokjson)},
                 notes=ctx.notes(f"SigLIP text tower: input_ids [N,64] int64 -> text_embeds [N,{tref.shape[1]}]; "
                                 "tokenizer.json is read by the pure-Go Unigram tokenizer.",
                                 "input width/height describe the TOKEN input (64 x 1); nothing reads them.",
                                 f"parity vs get_text_features max|d|/scale {terr:.2e}",
                                 f"image tower: {ctx.args.name}-image"),
                 reference=HFTokenReference(tok, "siglip"))
    return [img, txt]


# ---- CLIP -----------------------------------------------------------------------------------

CLIP_TEXT_LEN = 77          # internal/models/clip ContextLength (fixed)
CLIP_BOS, CLIP_EOS = 49406, 49407  # internal/models/clip TokenBOS / TokenEOS (fixed)


def _clip(ctx):
    import torch
    from transformers import AutoTokenizer, CLIPModel
    tc = ctx.cfg.get("text_config") or {}
    if tc.get("max_position_embeddings", CLIP_TEXT_LEN) != CLIP_TEXT_LEN:
        raise ConvertError(f"clip: text context {tc.get('max_position_embeddings')} != {CLIP_TEXT_LEN}, "
                           "the length the Go CLIP tokenizer pads to")
    vocab = _read_json(ctx.src / "vocab.json")
    if vocab is None or not (ctx.src / "merges.txt").exists():
        raise ConvertError("clip: the Go text tower reads vocab.json + merges.txt from the model directory; "
                           "this checkpoint lacks them")
    if vocab.get("<|startoftext|>") != CLIP_BOS or vocab.get("<|endoftext|>") != CLIP_EOS:
        raise ConvertError(f"clip: the Go tokenizer hard-codes BOS/EOS {CLIP_BOS}/{CLIP_EOS}; this vocab maps "
                           f"them to {vocab.get('<|startoftext|>')}/{vocab.get('<|endoftext|>')}")
    w, h, mean, std, notes = _geometry(ctx, go_resample="bicubic")
    crop = _center_crop_mode(ctx, w, h)
    if crop:
        # internal/models/clip implements this exact recipe (input.crop: center, bicubic), so these
        # are no longer divergences: drop the notes that would make B1 report them as such.
        notes = [n for n in notes if not n.startswith("HF resizes")]
    model = _load(CLIPModel, ctx.src)

    class Image(torch.nn.Module):
        def __init__(s, m):
            super().__init__()
            s.v, s.p = m.vision_model, m.visual_projection

        def forward(s, pixel_values):
            return s.p(s.v(pixel_values=pixel_values).pooler_output)

    class Text(torch.nn.Module):
        # Pooling at the first <|endoftext|> is done inside HF; the Go tokenizer pads WITH
        # <|endoftext|> and the tower is causal, so no attention_mask input is needed
        # (scripts/export_onnx_clip_text.py).
        def __init__(s, m):
            super().__init__()
            s.t, s.p = m.text_model, m.text_projection

        def forward(s, input_ids):
            return s.p(s.t(input_ids=input_ids).pooler_output)

    tol, ops = _tol(ctx), _opset(ctx)
    (ctx.work / "image").mkdir(parents=True, exist_ok=True)
    (ctx.work / "text").mkdir(parents=True, exist_ok=True)
    x = _images(w, h, mean, std)
    ip = _export(Image(model).eval(), (torch.from_numpy(x),), ctx.work / "image" / "model.onnx",
                 ["pixel_values"], ["image_embeds"], None, ops)
    with torch.no_grad():
        iref = _np(_features(model.get_image_features(pixel_values=torch.from_numpy(x))))
    ierr = parity(ip, {"pixel_values": x}, [iref], tol, " (clip vs get_image_features)")

    tok = AutoTokenizer.from_pretrained(str(ctx.src))
    phrases = ["a photo of a cat", "remote control", "water bottle"]
    raw = tok(phrases, truncation=True, max_length=CLIP_TEXT_LEN)["input_ids"]
    ids = np.full((len(phrases), CLIP_TEXT_LEN), CLIP_EOS, np.int64)  # pad exactly as Go does
    for i, r in enumerate(raw):
        ids[i, :len(r)] = r
    tp = _export(Text(model).eval(), (torch.from_numpy(ids[:1]),), ctx.work / "text" / "model.onnx",
                 ["input_ids"], ["text_embeds"], {"input_ids": {0: "N"}, "text_embeds": {0: "N"}}, ops)
    with torch.no_grad():
        tref = _np(_features(model.get_text_features(input_ids=torch.from_numpy(ids))))
    terr = parity(tp, {"input_ids": ids}, [tref], tol, " (clip-text vs get_text_features)")

    img = Bundle(name=ctx.args.name, task="embed", architecture="clip", license=ctx.license, width=w, height=h,
                 onnx={"model": str(ip)}, mean=mean, std=std, postprocess={"type": "embed"}, crop=crop,
                 notes=ctx.notes(f"CLIP image tower: [1,3,{h},{w}] -> image_embeds [1,{iref.shape[1]}] (projected)",
                                 f"parity vs get_image_features max|d|/scale {ierr:.2e}",
                                 f"text tower: {ctx.args.name}-text", *notes),
                 reference=HFImageReference(ctx.src, model, "embed", divergence=_divergence(notes)))
    txt = Bundle(name=f"{ctx.args.name}-text", task="embed", architecture="clip-text", license=ctx.license,
                 width=CLIP_TEXT_LEN, height=1, onnx={"model": str(tp)}, postprocess={"type": "embed"},
                 single_file=False,
                 extra_files={"vocab.json": str(ctx.src / "vocab.json"), "merges.txt": str(ctx.src / "merges.txt")},
                 notes=ctx.notes(f"CLIP text tower: input_ids [N,77] int64 -> text_embeds [N,{tref.shape[1]}] "
                                 "(projected); vocab.json + merges.txt read by the pure-Go BPE tokenizer.",
                                 "input width/height describe the TOKEN input (77 x 1); nothing reads them.",
                                 f"parity vs get_text_features max|d|/scale {terr:.2e}",
                                 f"image tower: {ctx.args.name}"),
                 reference=HFTokenReference(tok, "clip"))
    return [img, txt]


# ---- GroundingDINO --------------------------------------------------------------------------

GDINO_SIZE = 800  # internal/models/groundingdino inputSize (fixed, squash)
_GDINO_SPECIAL = (101, 102, 1012, 1029)  # [CLS] [SEP] . ?


def gdino_exportable_masks(input_ids):
    """Drop-in for transformers' generate_masks_with_special_tokens_and_transfer_map that EXPORTS
    with a dynamic phrase count. Copied from models/grounding-dino/export_gdino_fixedmask.py (the
    repo's corrected export; README.md there has the defect analysis) — the community export baked
    a Python loop's trip count so only the FIRST phrase got its attention block.

    isin -> Equal/Or chain; cummax/cummin -> LxL compare + ReduceMax/ReduceMin; torch.eye -> (i==j).
    """
    import torch
    n = input_ids.shape[1]
    sm = (input_ids == 101) | (input_ids == 102) | (input_ids == 1012) | (input_ids == 1029)
    idx = torch.arange(n, device=input_ids.device).unsqueeze(0)
    j = idx.unsqueeze(1)
    i = idx.unsqueeze(2)
    smj = sm.unsqueeze(1)
    prev = torch.where(smj & (j <= i), j, torch.full_like(j, -1)).max(dim=2)[0]
    nxt = torch.where(smj & (j >= i), j, torch.full_like(j, n)).min(dim=2)[0]
    valid = (nxt != 0) & (nxt != n - 1) & (nxt != n)
    am = (nxt.unsqueeze(2) == nxt.unsqueeze(1)) & valid.unsqueeze(1)
    am = (i == j) | am
    pos = torch.clamp(idx.expand_as(nxt) - prev - 1, min=0)
    return am, torch.where(valid, pos, torch.zeros_like(pos))


def gdino_demote_double(path: Path) -> int:
    """The legacy exporter promotes the deformable-attention sampling grid to float64 and ORT has no
    GridSample kernel for a double grid; PyTorch runs it in float32, so demote every double (same
    step as export_gdino_fixedmask.py::demote_double). Returns the number of edits."""
    import onnx
    from onnx import TensorProto, numpy_helper
    D, F = TensorProto.DOUBLE, TensorProto.FLOAT

    def demote(t):
        if t.data_type != D:
            return False
        t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name))
        return True

    m = onnx.load(str(path))
    g, n = m.graph, 0
    for node in g.node:
        for a in node.attribute:
            if a.name in ("to", "dtype") and a.i == D:
                a.i = F
                n += 1
            elif a.type == onnx.AttributeProto.TENSOR and demote(a.t):
                n += 1
            elif a.type == onnx.AttributeProto.TENSORS:
                n += sum(demote(t) for t in a.tensors)
    n += sum(demote(t) for t in g.initializer)
    for coll in (g.value_info, g.input, g.output):
        for v in coll:
            if v.type.tensor_type.elem_type == D:
                v.type.tensor_type.elem_type = F
                n += 1
    onnx.save(m, str(path))
    return n


GDINO_PARITY_FLOOR = 0.1  # queries scoring below this are never decoded (defaults: box 0.3, text 0.25)


def gdino_parity(onnx_path, cases, tol, floor=GDINO_PARITY_FLOOR) -> float:
    """Decode-level parity for GroundingDINO, where common.parity's element-wise comparison is the
    wrong test (measured, not assumed):

      * logits past the prompt length are -inf in both graphs, so an element-wise diff is NaN;
      * the decoder's 900 queries are a top-k over ~13k encoder tokens. On a synthetic image the
        low-score tail is full of near-ties, so ORT and PyTorch legitimately pick different tail
        anchors: ~150-250 of 900 query slots differ (box |d| up to 0.6), all scoring < 0.035.

    So: (1) the -inf/NaN pattern must match exactly; (2) every query that could ever be decoded
    (max sigmoid over the prompt tokens >= floor, on either side) must have a counterpart on the
    other side whose sigmoid scores AND cxcywh box agree within `tol`; (3) at least one such query
    must exist, so the check is not vacuous. Tail swaps below `floor` are counted and reported.
    """
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    worst, n_dec, n_swap = 0.0, 0, 0
    for prompt, feeds, (rl, rb) in cases:
        gl, gb = sess.run(None, feeds)[:2]
        if gl.shape != rl.shape or gb.shape != rb.shape:
            raise ConvertError(f"parity (grounding-dino): ONNX shapes {gl.shape}/{gb.shape} != "
                               f"reference {rl.shape}/{rb.shape}")
        if np.isnan(gl).any() or np.isnan(gb).any():
            raise ConvertError("parity (grounding-dino): the ONNX graph emits NaN")
        if not np.array_equal(np.isinf(gl), np.isinf(rl)):
            raise ConvertError("parity (grounding-dino): the -inf text-padding pattern of the logits differs")
        L = feeds["input_ids"].shape[1]
        pr = 1.0 / (1.0 + np.exp(-rl[0, :, :L].astype(np.float64)))
        pg = 1.0 / (1.0 + np.exp(-gl[0, :, :L].astype(np.float64)))
        br, bg = rb[0].astype(np.float64), gb[0].astype(np.float64)
        sr, sg = pr.max(-1), pg.max(-1)
        # distance of every query to its nearest counterpart on the other side (scores + box)
        d = np.maximum(np.abs(br[:, None] - bg[None]).max(-1), np.abs(pr[:, None] - pg[None]).max(-1))
        dr, dg = d.min(1), d.min(0)
        dec_r, dec_g = sr >= floor, sg >= floor
        w = float(max(dr[dec_r].max(initial=0.0), dg[dec_g].max(initial=0.0)))
        swaps = int((dr > tol).sum())
        log(f"  parity (grounding-dino, {prompt!r}): {int(dec_r.sum())} queries scoring >= {floor:g} matched, "
            f"max|d| (sigmoid, box) = {w:.2e}; {swaps} tail queries (all < {floor:g}) differ by top-k tie-break")
        worst, n_dec, n_swap = max(worst, w), n_dec + int(dec_r.sum()) + int(dec_g.sum()), n_swap + swaps
    if n_dec == 0:
        raise ConvertError(f"parity (grounding-dino): no query scored >= {floor:g} on the parity inputs, "
                           "so nothing decodable was compared")
    if worst > tol:
        raise ConvertError(f"parity (grounding-dino): decodable detections differ from transformers by {worst:.2e} "
                           f"(> tolerance {tol:g}). Not installed.")
    from ..common import record_parity
    record_parity(" (grounding-dino, decodable queries)", worst, tol)
    return worst


def _grounding_dino(ctx):
    import onnx
    import torch
    import transformers.models.grounding_dino.modeling_grounding_dino as gd
    from transformers import AutoTokenizer, GroundingDinoForObjectDetection

    special = tuple(getattr(gd, "SPECIAL_TOKENS", _GDINO_SPECIAL))
    if tuple(sorted(special)) != tuple(sorted(_GDINO_SPECIAL)):
        raise ConvertError(f"grounding-dino: transformers' SPECIAL_TOKENS {special} differ from the ids the "
                           f"exportable mask hard-codes {_GDINO_SPECIAL}; re-verify before exporting")
    vocab = ctx.src / "vocab.txt"
    if not vocab.exists():
        raise ConvertError("grounding-dino: the Go BERT tokenizer reads vocab.txt from the model directory; "
                           "this checkpoint has none")
    pp = _read_json(ctx.src / "preprocessor_config.json") or {}
    for key, want in (("image_mean", IMAGENET_MEAN), ("image_std", IMAGENET_STD)):
        got = pp.get(key, want)
        if not np.allclose(got, want, atol=1e-3):
            raise ConvertError(f"grounding-dino: processor {key} {got} differs from the ImageNet constants "
                               "internal/models/groundingdino hard-codes")
    if ctx.cfg.get("max_text_len", 256) != 256:
        log(f"  note: max_text_len {ctx.cfg.get('max_text_len')} (logits last dim)")

    model = _load(GroundingDinoForObjectDetection, ctx.src)
    model.config.disable_custom_kernels = True  # fused MSDA kernel is not traceable
    if getattr(model.config, "backbone_config", None) is not None:
        model.config.backbone_config.disable_custom_kernels = True

    class Net(torch.nn.Module):
        def __init__(s, m):
            super().__init__()
            s.m = m

        def forward(s, pixel_values, pixel_mask, input_ids, attention_mask, token_type_ids):
            o = s.m(pixel_values=pixel_values, pixel_mask=pixel_mask, input_ids=input_ids,
                    attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
            return o.logits, o.pred_boxes

    net = Net(model).eval()
    tok = AutoTokenizer.from_pretrained(str(ctx.src))

    def feeds(prompt, seed=0):
        t = tok(prompt, return_tensors="np")
        ids = t["input_ids"].astype(np.int64)
        return {"pixel_values": to_nchw(sample_image(GDINO_SIZE, GDINO_SIZE, seed), IMAGENET_MEAN, IMAGENET_STD),
                "pixel_mask": np.ones((1, GDINO_SIZE, GDINO_SIZE), np.int64),
                "input_ids": ids, "attention_mask": np.ones_like(ids),
                "token_type_ids": np.zeros_like(ids)}

    order = ["pixel_values", "pixel_mask", "input_ids", "attention_mask", "token_type_ids"]
    # Trace with a TWO-phrase prompt on purpose: a baked single-phrase loop could not pass unnoticed.
    ex = feeds("cat. remote.")
    path = ctx.work / "model.onnx"
    orig = gd.generate_masks_with_special_tokens_and_transfer_map
    gd.generate_masks_with_special_tokens_and_transfer_map = gdino_exportable_masks
    try:
        _export(net, tuple(torch.from_numpy(ex[k]) for k in order), path, order, ["logits", "pred_boxes"],
                {k: {1: "sequence_length"} for k in ("input_ids", "attention_mask", "token_type_ids")},
                _opset(ctx), legacy_only=True)
    finally:
        gd.generate_masks_with_special_tokens_and_transfer_map = orig
    n = gdino_demote_double(path)
    log(f"  demoted {n} float64 attributes/tensors to float32")
    ops = {nd.op_type for nd in onnx.load(str(path), load_external_data=False).graph.node}
    if "NonZero" in ops:  # what groundingdino.SupportsJointTextPass keys on
        raise ConvertError("grounding-dino: the exported graph still contains NonZero — the fixed-mask patch did "
                           "not take effect; the Go side would fall back to one pass per phrase")

    # Parity against the UNPATCHED transformers model (its own vectorised mask), on prompts with a
    # different phrase count and length than the trace — this validates the patched mask too.
    cases = []
    for prompt in ["cat. remote.", "a cup. remote control. water bottle.", "chair. tv. vase."]:
        f = feeds(prompt)
        with torch.no_grad():
            ref = [_np(t) for t in net(*(torch.from_numpy(f[k]) for k in order))]
        cases.append((prompt, f, ref))
    worst = gdino_parity(path, cases, _tol(ctx))

    conf = ctx.args.conf_threshold if getattr(ctx.args, "conf_threshold", None) is not None else 0.3
    return [Bundle(name=ctx.args.name, task="open_vocab", architecture="grounding-dino", license=ctx.license,
                   width=GDINO_SIZE, height=GDINO_SIZE, onnx={"model": str(path)}, mean=IMAGENET_MEAN,
                   std=IMAGENET_STD, single_file=False, prefer=("cuda", "cpu"),
                   postprocess={"conf_threshold": conf, "text_threshold": 0.25},
                   extra_files={"vocab.txt": str(vocab)},
                   notes=ctx.notes("GroundingDINO, FIXED dynamic text-attention mask (models/grounding-dino/"
                                   "export_gdino_fixedmask.py) -> one joint pass per prompt in Go.",
                                   "inputs pixel_values [1,3,800,800] (squash, ImageNet), pixel_mask, input_ids/"
                                   "attention_mask/token_type_ids [1,L]; outputs logits [1,Q,256], pred_boxes [1,Q,4]",
                                   f"parity vs transformers, 3 prompts, every query scoring >= {GDINO_PARITY_FLOOR:g}: "
                                   f"max|d| (sigmoid, cxcywh) {worst:.2e}",
                                   "CUDA, not TensorRT: see models/grounding-dino/manifest.yaml"),
                   reference=HFGroundingDinoReference(ctx.src, model, conf, 0.25))]


# --------------------------------------------------------------------------------------------
# Reference pipelines for tiers B/C/speed (convert/reference.py): the checkpoint's OWN processor
# and post-processing, i.e. what `transformers` users run.
# --------------------------------------------------------------------------------------------

def _center_crop_mode(ctx, w, h) -> Optional[str]:
    """"center" when the HF processor resizes the SHORTEST EDGE to the crop size and then centre-crops
    to a w×h square — the recipe the Go side implements as `input.crop: center` (CLIP). None for
    anything else (a crop_pct-style resize to 256 then crop 224 is a different recipe)."""
    pp = _read_json(ctx.src / "preprocessor_config.json") or {}
    if not (pp.get("do_center_crop") and pp.get("crop_size")):
        return None
    size = pp.get("size")
    short = size.get("shortest_edge") if isinstance(size, dict) else size
    return "center" if (w == h and short == w) else None


def _divergence(notes) -> Optional[str]:
    """The geometry/filter notes `_geometry` wrote about where the HF processor and the Go squash
    differ — deliberate and documented, so B1 reports them as WARN, not as a bug."""
    d = [n for n in notes if n.startswith("HF ")]
    return "; ".join(d) if d else None


class _HFBase(Reference):
    kind = "official"
    framework = "torch"

    def to(self, device):
        self.model.to(device)
        self.device = str(device)
        return self


class HFImageReference(_HFBase):
    """AutoImageProcessor(images=...) -> the model -> the processor's official post-processing:
    detection: post_process_object_detection(threshold=conf); classification: softmax(logits);
    depth: predicted_depth; embed: get_image_features."""

    def __init__(self, src, model, task, labels=None, conf=None, input_name="pixel_values", divergence=None):
        self.src, self.model, self.task, self.labels = src, model, task, labels
        self.conf, self.input_name, self.known_divergence = conf, input_name, divergence
        self.device = "cpu"
        self._proc = None
        what = {"detection": "post_process_object_detection", "classification": "softmax",
                "depth": "predicted_depth", "embed": "get_image_features"}[task]
        self.description = f"transformers AutoImageProcessor + {type(model).__name__} + {what}"
        self.short = f"HF processor + {what}"
        self.short_pre = "HF AutoImageProcessor"

    @property
    def processor(self):
        if self._proc is None:
            from transformers import AutoImageProcessor
            self._proc = AutoImageProcessor.from_pretrained(str(self.src))
            self.description += f" ({type(self._proc).__name__})"
        return self._proc

    def preprocess(self, pil, prompt=None):
        pv = self.processor(images=pil.convert("RGB"), return_tensors="np")["pixel_values"]
        return {self.input_name: np.asarray(pv, np.float32)}

    def _label(self, k):
        if self.labels and 0 <= k < len(self.labels):
            return self.labels[k]
        return str((self.model.config.id2label or {}).get(k, f"class_{k}"))

    def predict(self, pil, prompt=None):
        import torch
        from ..reference import Prediction
        pil = pil.convert("RGB")
        pv = self.processor(images=pil, return_tensors="pt")["pixel_values"].to(self.device)
        with torch.no_grad():
            if self.task == "embed":
                return Prediction(embedding=_np(_features(self.model.get_image_features(pixel_values=pv)))[0])
            out = self.model(pixel_values=pv)
        if self.task == "classification":
            p = torch.softmax(out.logits[0].float(), -1).cpu().numpy()
            return Prediction(probs={self._label(i): float(v) for i, v in enumerate(p)})
        if self.task == "depth":
            d = _np(out.predicted_depth)
            return Prediction(depth=d.reshape(d.shape[-2], d.shape[-1]))
        w, h = pil.size
        res = self.processor.post_process_object_detection(out, threshold=float(self.conf),
                                                           target_sizes=[(h, w)])[0]
        dets = []
        for sc, lab, bx in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()):
            x0, y0, x1, y1 = bx
            dets.append({"cls": self._label(int(lab)), "conf": float(sc), "bbox": [x0, y0, x1 - x0, y1 - y0]})
        return Prediction(detections=dets)

    def bench_fn(self, feeds):
        import torch
        from ..reference import torch_bench_fn
        m, task = self.model, self.task

        class Fwd(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.m = m

            def forward(s, x):
                return s.m.get_image_features(pixel_values=x) if task == "embed" else s.m(pixel_values=x)
        return torch_bench_fn(Fwd().eval(), {self.input_name: feeds[self.input_name]}, self.device)


class HFGroundingDinoReference(_HFBase):
    """AutoProcessor(images, text) -> GroundingDinoForObjectDetection ->
    post_process_grounded_object_detection(threshold=box, text_threshold=text), the thresholds the
    manifest serves with."""
    known_divergence = ("HF resizes the shortest edge to 800 (longest <= 1333) keeping the aspect ratio; "
                        "internal/models/groundingdino squashes to 800x800")

    def __init__(self, src, model, box_threshold, text_threshold):
        self.src, self.model = src, model
        self.box, self.text = float(box_threshold), float(text_threshold)
        self.input_name, self.device, self._proc = "pixel_values", "cpu", None
        self.short = "HF GroundingDinoProcessor + post_process_grounded"
        self.short_pre = "HF GroundingDinoProcessor"
        self.description = (f"transformers GroundingDinoProcessor + post_process_grounded_object_detection "
                            f"(box {self.box:g}, text {self.text:g})")

    @property
    def processor(self):
        if self._proc is None:
            from transformers import AutoProcessor
            self._proc = AutoProcessor.from_pretrained(str(self.src))
        return self._proc

    def preprocess(self, pil, prompt=None):
        t = self.processor(images=pil.convert("RGB"), text=prompt or "object.", return_tensors="np")
        out = {}
        for k, v in t.items():
            v = np.asarray(v)
            out[k] = v.astype(np.float32) if v.dtype.kind == "f" else v.astype(np.int64)
        return out

    def predict(self, pil, prompt=None):
        import torch
        from ..reference import Prediction
        pil = pil.convert("RGB")
        inputs = self.processor(images=pil, text=prompt or "object.", return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        w, h = pil.size
        res = self.processor.post_process_grounded_object_detection(
            out, inputs["input_ids"], threshold=self.box, text_threshold=self.text, target_sizes=[(h, w)])[0]
        labs = res.get("text_labels", res.get("labels"))
        dets = []
        for sc, lab, bx in zip(res["scores"].tolist(), list(labs), res["boxes"].tolist()):
            x0, y0, x1, y1 = bx
            dets.append({"cls": str(lab), "conf": float(sc), "bbox": [x0, y0, x1 - x0, y1 - y0]})
        return Prediction(detections=dets)

    def bench_fn(self, feeds):
        import torch
        from ..reference import torch_bench_fn
        order = ["pixel_values", "pixel_mask", "input_ids", "attention_mask", "token_type_ids"]
        m = self.model

        class Fwd(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.m = m

            def forward(s, *xs):
                return s.m(**dict(zip(order, xs)))
        return torch_bench_fn(Fwd().eval(), {k: feeds[k] for k in order if k in feeds}, self.device,
                              [k for k in order if k in feeds])


class HFTokenReference(_HFBase):
    """Token ids exactly as the Go text towers pad them: SigLIP max_length 64 (pad </s>),
    CLIP 77 padded WITH <|endoftext|>."""

    def __init__(self, tok, mode):
        self.tok, self.mode, self.model, self.device = tok, mode, None, "cpu"
        self.input_name = "input_ids"
        self.description = self.short = f"transformers {type(tok).__name__} ({mode} padding)"

    def to(self, device):
        return self

    def preprocess(self, pil, prompt=None):
        text = prompt or "a photo of a cat"
        if self.mode == "siglip":
            ids = self.tok([text], padding="max_length", max_length=SIGLIP_TEXT_LEN, truncation=True,
                           return_tensors="np")["input_ids"].astype(np.int64)
        else:
            raw = self.tok([text], truncation=True, max_length=CLIP_TEXT_LEN)["input_ids"][0]
            ids = np.full((1, CLIP_TEXT_LEN), CLIP_EOS, np.int64)
            ids[0, :len(raw)] = raw
        return {"input_ids": ids}
