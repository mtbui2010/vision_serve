"""RF-DETR training checkpoint (.pth) -> ONNX for the Go `rf-detr` architecture.

Source: the `checkpoint_best_total.pth` (or any `checkpoint*.pth`) that the `rfdetr` package writes
while training — a dict with `model` (state_dict), `args` (training args, incl. `class_names`),
and, since rfdetr 1.7, `model_name` (e.g. "RFDETRSmall").

What this module does, in order:
  1. refuse Ultralytics pickles; load the checkpoint with `weights_only=True` (no code execution);
  2. pick the variant (nano/small/medium/base/large): --variant > `model_name` > args/filename >
     state_dict shapes (patch size, decoder depth, positional-embedding grid);
  3. rebuild that variant with the installed `rfdetr`, load the weights STRICTLY (every tensor must
     land — rfdetr's own loader is lenient and would silently keep random weights);
  4. export with rfdetr's own ONNX exporter (legacy TorchScript path, `dynamo=False`, what rfdetr
     ships and tests), static batch 1, square input;
  5. parity: the PyTorch model in export mode vs ONNX Runtime on `to_nchw(sample_image)` (2 images).

Output contract (Go `internal/models/rfdetr`): input [1,3,R,R] ImageNet-normalised and SQUASHED
(letterbox false — rfdetr trains with square_resize_div_64 and `predict()` squashes too; see
BUGS_TO_FIX.md #1); outputs `dets` [1,Q,4] cxcywh normalised + `labels` [1,Q,C] (sigmoid).

Labels. A fine-tune's head has C = len(class_names) + 1 logits. The extra slot is the LAST one:
rfdetr maps class_id i -> class_names[i] for fine-tunes (0-based; `RFDETR.predict`), and this was
verified on a real 22-class tabletop checkpoint by matching torch predictions against ground-truth
boxes (137/141 IoU>0.5 matches correct with 0-based names, 1/141 with a 1-based shift; index 22
never won). So labels = class_names + ["N/A"]. The one exception is rfdetr's own COCO checkpoints
(class_names == COCO's 80 names, C = 91): those logits are indexed by the sparse COCO category id,
so labels[0] = "N/A" and the gaps are "N/A" — same as models/rf-detr/coco91.txt.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from ..common import (IMAGENET_MEAN, IMAGENET_STD, Bundle, ConvertError, canonical_license, log,
                      license_scan, onnx_io, resolve_license, sample_image, to_nchw)
from ..reference import Reference

# variant name -> rfdetr.variants class. XLarge / 2XLarge are deliberately absent: their weights
# are published under Roboflow's Platform Model License (rfdetr_plus), not Apache-2.0.
VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
    "large-legacy": "RFDETRLargeDeprecated",
}

# Structural signature of each variant, as it shows in the state_dict:
#   (patch size, backbone embed dim, positional-embedding grid side, decoder layers)
_SIGNATURES = {
    "nano": (16, 384, 24, 2),
    "small": (16, 384, 32, 3),
    "medium": (16, 384, 36, 4),
    "large": (16, 384, 44, 4),
    "base": (14, 384, 37, 3),
    "large-legacy": (14, 768, 37, 3),
}

_PE_KEY = "backbone.0.encoder.encoder.embeddings.position_embeddings"
_PATCH_KEY = "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight"

POSTPROCESS = {"type": "detr", "box_format": "cxcywh", "conf_threshold": 0.5, "max_detections": 300}


def help_for(fmt: str) -> str:
    return "RF-DETR training checkpoint (.pth from the `rfdetr` package) -> rf-detr detector"


def add_arguments(p: argparse.ArgumentParser, fmt: str) -> None:
    p.add_argument("--variant", choices=sorted(VARIANTS),
                   help="RF-DETR size; default: detected from the checkpoint (model_name, args, or the "
                        "state_dict's patch size / decoder depth / positional-embedding grid)")
    p.add_argument("--resolution", type=int,
                   help="square input side in pixels; default: the checkpoint's args.resolution, else the "
                        "variant's training default (nano 384, small 512, medium 576, base 560, large 704). "
                        "Must be divisible by patch_size * num_windows")
    p.add_argument("--query-feats", action="store_true",
                   help="also emit query_feats [1,Q,D] (the decoder features the class head reads) as a "
                        "third output, after dets and labels — for open-vocabulary heads built on top")


# --------------------------------------------------------------------------------------------
# checkpoint reading
# --------------------------------------------------------------------------------------------

def _find_checkpoint(src: Path) -> Path:
    if src.is_file():
        return src
    if src.is_dir():
        for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth",
                     "checkpoint.pth"):
            if (src / name).is_file():
                return src / name
        pths = sorted(src.glob("*.pth"))
        if len(pths) == 1:
            return pths[0]
        raise ConvertError(f"{src}: expected one RF-DETR checkpoint (*.pth), found {[p.name for p in pths]}")
    raise ConvertError(f"{src}: no such file or directory")


def _load_checkpoint(path: Path):
    """torch.load with weights_only=True (no pickle code execution). Legacy rfdetr checkpoints keep
    their args in an argparse.Namespace, which is allow-listed for that purpose only."""
    import torch
    try:
        with torch.serialization.safe_globals([argparse.Namespace]):
            return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001 — torch raises UnpicklingError/RuntimeError variants
        raise ConvertError(f"{path.name}: cannot be loaded as plain weights ({str(e).splitlines()[0]}). "
                           "RF-DETR checkpoints contain only tensors, dicts and the training args; refusing "
                           "to unpickle arbitrary objects.")


def _split_checkpoint(ck):
    """-> (state_dict, args dict, model_name or None)."""
    if not isinstance(ck, dict):
        raise ConvertError("checkpoint is not a dict; expected an rfdetr training checkpoint")
    if "model" in ck and isinstance(ck["model"], dict):
        sd = ck["model"]
    elif "class_embed.bias" in ck:
        sd = ck  # a bare state_dict
    else:
        raise ConvertError(f"checkpoint has no 'model' state_dict (keys: {sorted(ck)[:10]}); "
                           "is this an rfdetr training checkpoint?")
    if "class_embed.bias" not in sd:
        raise ConvertError("state_dict has no class_embed.bias — not an RF-DETR detector checkpoint")
    args = ck.get("args") if sd is not ck else None
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    if not isinstance(args, dict):
        args = {}
    name = ck.get("model_name") if sd is not ck else None
    return sd, args, (name if isinstance(name, str) and name.strip() else None)


def _signature(sd):
    """(patch, embed_dim, pe_grid, dec_layers) read from tensor shapes, or None if unreadable."""
    try:
        pe = sd[_PE_KEY]
        patch = int(sd[_PATCH_KEY].shape[-1])
    except KeyError:
        return None
    n = int(pe.shape[1]) - 1  # minus the class token
    grid = int(round(n ** 0.5))
    if grid * grid != n:
        return None
    layers = {int(k.split(".")[3]) for k in sd if k.startswith("transformer.decoder.layers.")}
    return patch, int(pe.shape[2]), grid, (max(layers) + 1 if layers else 0)


def _variant_from_signature(sig):
    if sig is None:
        return None
    for v, s in _SIGNATURES.items():
        if s == sig:
            return v
    # A custom-resolution fine-tune changes only the PE grid: match on the rest.
    near = [v for v, s in _SIGNATURES.items() if (s[0], s[1], s[3]) == (sig[0], sig[1], sig[3])]
    return near[0] if len(near) == 1 else None


def _variant_from_name(text: str):
    t = text.lower().replace("_", "-")
    if "xlarge" in t or "2xl" in t:
        raise ConvertError(f"{text!r} is an RF-DETR XLarge/2XLarge model: those weights are published under "
                           "Roboflow's Platform Model License, not Apache-2.0, and cannot be served "
                           "(CLAUDE.md rule 1)")
    if "seg" in t:
        raise ConvertError(f"{text!r} is an RF-DETR segmentation model; the rf-detr architecture serves "
                           "boxes only (no mask head) — not supported by this converter")
    t = t.replace("rfdetr", "").replace("rf-detr", "")
    if "deprecated" in t and "large" in t:
        return "large-legacy"
    for v in ("medium", "small", "nano", "large", "base"):  # 'large' before 'base' is irrelevant here
        if v in t:
            return v
    return None


def detect_variant(sd, args: dict, model_name, filename: str, forced=None):
    """Return (variant, how) — `how` is a human-readable reason for the log."""
    sig = _signature(sd)
    from_shapes = _variant_from_signature(sig)
    if forced:
        return forced, "--variant"
    if model_name:
        v = _variant_from_name(model_name)
        if v:
            return v, f"checkpoint model_name={model_name!r}"
    for key in ("model_name", "pretrain_weights", "encoder_name"):
        val = args.get(key)
        if isinstance(val, str) and val.strip().lower() not in ("", "none", "null"):
            v = _variant_from_name(val)
            if v:
                return v, f"checkpoint args.{key}={val!r}"
    if from_shapes:
        return from_shapes, f"state_dict shapes (patch, dim, pe grid, decoder layers) = {sig}"
    v = _variant_from_name(filename)
    if v:
        return v, f"file name {filename!r}"
    raise ConvertError(f"cannot tell which RF-DETR variant this is (state_dict signature {sig}); "
                       f"pass --variant ({', '.join(sorted(VARIANTS))})")


# --------------------------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------------------------

def labels_for(names, n_logits: int):
    """Map class names onto the C logit slots (see module docstring for how this was verified)."""
    if not names:
        return None
    names = [str(n) for n in names]
    if len(names) == n_logits:
        return names
    if len(names) > n_logits:
        raise ConvertError(f"{len(names)} class names but the head has only {n_logits} logits")
    try:
        from rfdetr.assets.coco_classes import COCO_CLASS_NAMES, COCO_CLASSES
    except ImportError:  # pragma: no cover
        COCO_CLASS_NAMES, COCO_CLASSES = None, None
    if COCO_CLASS_NAMES is not None and names == list(COCO_CLASS_NAMES) and n_logits > max(COCO_CLASSES):
        out = ["N/A"] * n_logits  # COCO checkpoints index logits by sparse category id (1..90)
        for cid, name in COCO_CLASSES.items():
            out[cid] = name
        return out
    if len(names) == n_logits - 1:
        return names + ["N/A"]  # rfdetr fine-tune: num_classes = len(class_names), head = +1, extra LAST
    raise ConvertError(f"{len(names)} class names for a head with {n_logits} logits: an RF-DETR fine-tune has "
                       f"exactly one extra (N/A) slot. Pass --labels with {n_logits} or {n_logits - 1} lines "
                       "in the training order")


# --------------------------------------------------------------------------------------------
# build + export
# --------------------------------------------------------------------------------------------

def _build(variant, sd, args, n_logits, resolution, workdir: Path):
    """Rebuild `variant` with the installed rfdetr and load `sd` strictly. Returns (RFDETR, nn.Module)."""
    import torch
    import rfdetr.variants as rv

    rows = int(sd["query_feat.weight"].shape[0]) if "query_feat.weight" in sd else None
    group = int(args.get("group_detr") or 13)
    queries = int(args.get("num_queries") or (rows // group if rows else 300))
    if rows is not None and rows != queries * group:
        raise ConvertError(f"query_feat has {rows} rows, not num_queries*group_detr = {queries}*{group}")
    kw = dict(num_classes=n_logits - 1, num_queries=queries, group_detr=group, device="cpu")
    sig = _signature(sd)
    if sig:
        kw["positional_encoding_size"] = sig[2]  # keep the checkpoint's PE grid (no resize on load)
    if resolution:
        kw["resolution"] = resolution
    # rfdetr's constructor loads `pretrain_weights` with weights_only=False. Hand it a re-saved copy of
    # what we already loaded safely, never the user's pickle. (A non-None path also stops rfdetr from
    # fetching DINOv2 weights from the hub.)
    safe = workdir / "rfdetr-weights.pth"
    torch.save({"model": sd, "args": {"num_queries": queries, "group_detr": group}}, safe)
    try:
        model = getattr(rv, VARIANTS[variant])(pretrain_weights=str(safe), **kw)
    except Exception as e:  # noqa: BLE001
        raise ConvertError(f"rfdetr could not build {variant} from this checkpoint: {e}")
    net = model.model.model
    try:
        net.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        guess = _variant_from_signature(sig)
        hint = f" The tensor shapes look like --variant {guess}." if guess and guess != variant else ""
        raise ConvertError(f"weights do not fit RF-DETR {variant}: {str(e).splitlines()[0]} ...{hint}")
    net.eval()
    return model, net


class _Exportable:
    """Builds the nn.Module that is exported: the model in rfdetr export mode, optionally with the
    class head's input returned as a third output (query_feats)."""

    @staticmethod
    def make(net, query_feats: bool):
        import torch

        net = copy.deepcopy(net).cpu().eval()
        net.export()  # rfdetr: forward -> forward_export, returns (boxes, logits)
        if not query_feats:
            return net, ["dets", "labels"]

        class WithQueryFeats(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
                self._qf = []
                inner.class_embed.register_forward_hook(lambda m, i, o: self._qf.append(i[0]))

            def forward(self, x):
                self._qf.clear()
                out = self.inner(x)
                if not self._qf:
                    raise ConvertError("--query-feats: the class head was not called on decoder features "
                                       "(decoder-less checkpoint?)")
                return out[0], out[1], self._qf[-1]

        # eval() on the WRAPPER too: torch.onnx.export restores the exported module's own train flag
        # afterwards, recursively — a wrapper left in train mode would flip rfdetr back to training
        # (all group_detr*Q queries) for the parity run.
        return WithQueryFeats(net).eval(), ["dets", "labels", "query_feats"]


BOUNDARY_SCORE = 0.05     # a query below this sigmoid score in BOTH runs cannot reach any sane threshold


def detr_parity(onnx_path, feeds, reference, tol, what=""):
    """common.parity, made aware that DETR queries are a SET chosen by a top-K.

    RF-DETR is two-stage: the decoder's queries are the top-K encoder proposals. On a noise image a
    few proposals sit on a near-tie at the K-th place, and a 1e-6 float difference (ORT vs PyTorch)
    swaps which one gets in — measured on the 22-class tabletop fine-tune: seed 1 differs in exactly
    2 of 300 queries (ranks 280/281, sigmoid score 0.011/0.012), every other query to 1e-5; on a real
    tabletop image all 300 match to 2e-5. So: every query must match to `tol`, EXCEPT at most 2% of
    them whose best score is < BOUNDARY_SCORE in both runs. Anything else is a real export bug."""
    import numpy as np
    import onnxruntime as ort

    got = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(None, feeds)
    if len(got) < len(reference):
        raise ConvertError(f"parity{what}: ONNX graph returns {len(got)} outputs, reference has {len(reference)}")
    q = reference[0].shape[1]
    row_err = np.zeros(q)
    for i, (g, r) in enumerate(zip(got, reference)):
        g, r = np.asarray(g, np.float64), np.asarray(r, np.float64)
        if g.shape != r.shape:
            raise ConvertError(f"parity{what}: output {i} shape {g.shape} != reference {r.shape}")
        scale = max(1.0, float(np.abs(r).max()))
        row_err = np.maximum(row_err, np.abs(g - r).reshape(q, -1).max(-1) / scale)
    sig = lambda z: 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))  # noqa: E731
    li = 1 if reference[0].shape[-1] == 4 else 0  # logits = the first non-box output
    score_ref = sig(reference[li][0]).max(-1)
    score_got = sig(got[li][0]).max(-1)
    bad = np.nonzero(row_err > tol)[0]
    boundary = [int(k) for k in bad if score_ref[k] < BOUNDARY_SCORE and score_got[k] < BOUNDARY_SCORE]
    real = [int(k) for k in bad if k not in boundary]
    kept = np.setdiff1d(np.arange(q), boundary)
    worst = float(row_err[kept].max()) if kept.size else 0.0
    log(f"  parity{what}: {q} queries, max|Δ|/scale = {worst:.2e}"
        + (f" ({len(boundary)} low-score top-K boundary queries differ, ignored: rows {boundary}, "
           f"best score {float(max(score_ref[boundary].max(), score_got[boundary].max())):.3f})" if boundary else ""))
    if real or len(boundary) > max(1, q // 50):
        raise ConvertError(f"parity{what}: ONNX differs from the original on {len(bad)} of {q} queries "
                           f"(max|Δ|/scale {float(row_err.max()):.2e} > tolerance {tol:g}"
                           + (f"; rows {real[:10]} score up to {float(score_ref[real].max()):.2f}" if real else "")
                           + "). Not installed.")
    from ..common import record_parity
    record_parity(what, worst, tol, ignored_boundary_queries=len(boundary))
    return worst


def convert(args, workdir: Path):
    import torch

    lic = resolve_license(args.license or "Apache-2.0", None, args.source)  # gate before any work
    src = _find_checkpoint(Path(args.source))
    license_scan(source=src)
    ck = _load_checkpoint(src)
    sd, targs, model_name = _split_checkpoint(ck)
    trained_with = ck.get("rfdetr_version") if isinstance(ck, dict) else None
    del ck
    if targs.get("segmentation_head") or any(k.startswith("segmentation_head.") for k in sd):
        raise ConvertError("this is an RF-DETR segmentation checkpoint; the rf-detr architecture serves boxes "
                           "only — not supported by this converter")

    variant, how = detect_variant(sd, targs, model_name, src.name, getattr(args, "variant", None))
    log(f"rfdetr: variant {variant} (from {how}); trained with rfdetr {trained_with or 'unknown'}")
    n_logits = int(sd["class_embed.bias"].shape[0])

    res = getattr(args, "resolution", None) or (targs.get("resolution") if isinstance(targs.get("resolution"), int)
                                                else None)
    model, net = _build(variant, sd, targs, n_logits, res, workdir)
    cfg = model.model_config
    R = int(res or model.model.resolution)
    block = cfg.patch_size * cfg.num_windows
    if R % block:
        raise ConvertError(f"resolution {R} is not divisible by patch_size*num_windows = {block}")
    if targs.get("square_resize_div_64") is False:
        log("rfdetr: warning: trained with square_resize_div_64=False (aspect-preserving training resize); "
            "serving squashes to a square as rfdetr's own predict() does")

    # License: the variant's own declared license must be permissive (it is Apache-2.0 for every
    # variant we build); the served license is --license, defaulting to Apache-2.0.
    canonical_license(getattr(cfg, "license", "Apache-2.0"))

    from ..cli import read_labels
    names = read_labels(args.labels) if args.labels else targs.get("class_names")
    labels = labels_for(names, n_logits)
    if labels is None:
        log(f"rfdetr: warning: no class names in the checkpoint and no --labels; detections will be class_0.."
            f"class_{n_logits - 1}")
    else:
        log(f"rfdetr: {n_logits} logits -> labels {labels[:3]} ... {labels[-2:]}")

    # ---- export (rfdetr's own exporter: legacy TorchScript path, dynamo=False) ----
    from rfdetr.export._onnx.exporter import export_onnx
    exp, out_names = _Exportable.make(net, getattr(args, "query_feats", False))
    x0 = torch.from_numpy(to_nchw(sample_image(R, R, seed=0), IMAGENET_MEAN, IMAGENET_STD))
    onnx_dir = workdir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    log(f"rfdetr: exporting {variant} at {R}x{R}, opset {args.opset} ...")
    try:
        out = export_onnx(output_dir=str(onnx_dir), model=exp, input_names=["input"], input_tensors=x0,
                          output_names=out_names, dynamic_axes=None, verbose=False,
                          opset_version=args.opset, variant_name="model")
    except ConvertError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ConvertError(f"ONNX export failed: {e}")
    onnx_path = Path(out)

    exp.eval()
    ins, outs = onnx_io(onnx_path)
    log(f"rfdetr: ONNX inputs {[(n, s) for n, s, _ in ins]} outputs {[(n, s) for n, s, _ in outs]}")

    # ---- parity: PyTorch (export mode) vs ORT, two different images ----
    for seed in (0, 1):
        x = to_nchw(sample_image(R, R, seed=seed), IMAGENET_MEAN, IMAGENET_STD)
        with torch.no_grad():
            ref = [t.detach().cpu().numpy() for t in exp(torch.from_numpy(x))]
        detr_parity(onnx_path, {ins[0][0]: x}, ref, args.tolerance, f" rfdetr[seed {seed}]")

    notes = [f"rfdetr {src.name} ({VARIANTS[variant]}, variant from {how})",
             f"input {R}x{R} squashed (rfdetr trains with square_resize_div_64 — BUGS_TO_FIX.md #1); "
             f"{n_logits} logits" + (", the last one is the fine-tune's extra N/A slot"
                                     if names and len(names) == n_logits - 1 and labels[-1] == "N/A" else ""),
             f"parity vs PyTorch checked at tolerance {args.tolerance:g}"]
    if getattr(args, "query_feats", False):
        notes.append("outputs: dets, labels, query_feats (query_feats stays LAST: postprocess takes the first "
                     "non-box output as logits)")
    return [Bundle(name=args.name, task="detection", architecture="rf-detr", license=lic, width=R, height=R,
                   onnx={"model": str(onnx_path)}, letterbox=False, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                   postprocess=dict(POSTPROCESS), labels=labels, notes=notes,
                   reference=RFDETRReference(model, labels, POSTPROCESS["conf_threshold"], ins[0][0]))]


# --------------------------------------------------------------------------------------------
# reference pipeline for tiers B/C/speed (convert/reference.py)
# --------------------------------------------------------------------------------------------

class RFDETRReference(Reference):
    """rfdetr's OWN inference path: `RFDETR.predict` = torchvision F.to_tensor -> F.resize(img, [R, R])
    (a squash, antialiased bilinear) -> F.normalize(ImageNet) -> the PyTorch model -> rfdetr's
    PostProcess (top-300 over queries x classes, sigmoid) -> score > threshold. Class ids index the
    same `labels` list the manifest gets (fine-tune: 0-based class_names; COCO: sparse ids)."""
    kind = "official"
    known_divergence = None
    framework = "torch"
    short = "rfdetr RFDETR.predict"
    short_pre = "rfdetr predict() preprocessing"
    description = ("rfdetr RFDETR.predict (F.resize squash, ImageNet mean/std, rfdetr PostProcess top-300) "
                   "— the framework's own inference path")

    def __init__(self, model, labels, conf, input_name="input"):
        self.model, self.labels, self.conf, self.input_name = model, labels, float(conf), input_name
        self.device = "cpu"

    def to(self, device):
        import torch
        self.model.model.device = torch.device(device)
        self.model.model.model.to(device)
        self.device = str(device)
        return self

    def preprocess(self, pil, prompt=None):
        import torchvision.transforms.functional as F
        R = int(self.model.model.resolution)
        t = F.to_tensor(pil.convert("RGB"))
        t = F.normalize(F.resize(t, [R, R]), self.model.means, self.model.stds)
        return {self.input_name: t[None].numpy().astype("float32")}

    def predict(self, pil, prompt=None):
        from ..reference import Prediction
        det = self.model.predict(pil.convert("RGB"), threshold=self.conf, include_source_image=False)
        out = []
        for xyxy, c, k in zip(det.xyxy, det.confidence, det.class_id):
            k = int(k)
            lab = self.labels[k] if self.labels and 0 <= k < len(self.labels) else f"class_{k}"
            x0, y0, x1, y1 = (float(v) for v in xyxy)
            out.append({"cls": lab, "conf": float(c), "bbox": [x0, y0, x1 - x0, y1 - y0]})
        return Prediction(detections=out)

    def bench_fn(self, feeds):
        from ..reference import torch_bench_fn
        net = self.model.model.model.eval()
        return torch_bench_fn(net, {self.input_name: feeds[self.input_name]}, self.device)
