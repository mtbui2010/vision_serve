"""The "original framework" side of tiers B and C.

A Reference answers three questions for one installed model:

  preprocess(pil, prompt) -> {onnx_input_name: ndarray}   what the model was MEANT to be fed (B1)
  predict(pil, prompt)    -> Prediction                    what the original pipeline outputs (B2, C)
  bench_fn(feeds)         -> zero-arg callable             one framework forward, for --bench

Where each answer comes from, strongest first:

  user      the user's own training transform (`export(..., preprocess=fn)`) or a
            `--reference-script` defining preprocess(pil) [and predict(pil)];
  official  the framework's own pipeline, built by the family: rfdetr's RFDETR.predict, the HF
            AutoImageProcessor + post_process_*;
  manifest  no official pipeline exists (TorchScript / plain PyTorch / TensorFlow): the
            MANIFEST's declared preprocessing re-implemented here in numpy/PIL, and the exported
            ONNX (or the in-process framework module) decoded exactly like the Go side. This checks
            the Go implementation of the declared spec — NOT that the spec matches training.

Families attach their official Reference to `Bundle.reference`; this module holds the generic
pieces. numpy + PIL at import time; torch/onnxruntime only inside functions.
"""
from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .metrics import sigmoid, softmax
from .spec import (Spec, apply_spec, describe, dpt_keep_aspect_size, spec_from_legacy,  # noqa: F401
                   spec_from_manifest, spec_meta)


@dataclasses.dataclass
class Prediction:
    detections: Optional[List[dict]] = None   # [{"cls", "conf", "bbox": [x, y, w, h]}] ORIGINAL pixels
    probs: Optional[Dict[str, float]] = None  # classification: label -> probability (all classes)
    embedding: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None        # 2-D map


class Reference:
    description = "reference"
    short = ""                        # a few words for the report table; defaults to description
    short_pre = ""                    # the same, naming only the preprocessing (B1)
    kind = "official"                 # official | user | manifest
    known_divergence: Optional[str] = None  # a documented, deliberate difference of the Go preprocess
    framework = "torch"
    device = "cpu"
    input_name: Optional[str] = None

    def preprocess(self, pil, prompt=None) -> Optional[Dict[str, np.ndarray]]:
        return None

    def predict(self, pil, prompt=None) -> Optional[Prediction]:
        return None

    def forward(self, feeds: Dict[str, np.ndarray]) -> Optional[List[np.ndarray]]:
        """Raw model outputs (ONNX output order) for already-preprocessed feeds."""
        return None

    def bench_fn(self, feeds: Dict[str, np.ndarray]) -> Optional[Callable[[], None]]:
        return None

    def to(self, device: str) -> "Reference":
        return self


def resolve_device(want: Optional[str]) -> str:
    """'auto' -> cuda when torch sees a GPU, else cpu."""
    want = (want or "auto").lower()
    if want != "auto":
        return want
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def torch_bench_fn(module, feeds: Dict[str, np.ndarray], device: str, order: Optional[List[str]] = None):
    """A zero-arg callable running `module(*inputs)` under no_grad, synchronising CUDA so the
    wall time is the real forward time."""
    import torch
    xs = [torch.from_numpy(np.array(feeds[k], copy=True)).to(device) for k in (order or list(feeds))]
    cuda = str(device).startswith("cuda")

    def run():
        with torch.no_grad():
            module(*xs)
        if cuda:
            torch.cuda.synchronize()
    return run


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


# --------------------------------------------------------------------------------------------
# The manifest's declared preprocessing, re-implemented: the Spec semantics of
# internal/vision/preprocess (see spec.py; both sides run the same sync corpora)
# --------------------------------------------------------------------------------------------

def bundle_spec(bundle) -> Spec:
    """The Spec a Bundle (or any object with its fields) declares through its legacy manifest
    fields — what the server resolves for the manifest the Bundle renders."""
    return spec_from_legacy(bundle.width, bundle.height, getattr(bundle, "layout", "") or "",
                            getattr(bundle, "letterbox", False), getattr(bundle, "crop", None),
                            getattr(bundle, "keep_aspect", False), getattr(bundle, "multiple_of", 0),
                            getattr(bundle, "mean", None), getattr(bundle, "std", None))


def manifest_preprocess(pil, width: int = 0, height: int = 0, mean=None, std=None, letterbox: bool = False,
                        crop: Optional[str] = None, keep_aspect: bool = False, multiple_of: int = 0,
                        spec: Optional[Spec] = None):
    """-> (x float32, meta): the declared preprocessing — `spec` (e.g. spec_from_manifest of an
    installed manifest), or the legacy fields given — applied with the server's Spec semantics
    (spec.apply_spec). For the legacy fields: squash = PIL bilinear resize to WxH (imaging.Linear is
    the same antialiased triangle filter); letterbox = scale min(W/w, H/h), round half up, black pad,
    centred; crop: center = short side to target (bicubic), centred crop; keep_aspect =
    dpt_keep_aspect_size, bicubic, no crop/pad. Then x/255, (x-mean)/std, NCHW [1,3,H,W]."""
    if spec is None:
        spec = spec_from_legacy(width, height, "", letterbox, crop, keep_aspect, multiple_of, mean, std)
    return apply_spec(pil, spec)


def manifest_meta(bundle, ow: int, oh: int) -> dict:
    """The Meta the Go box decoders (rf-detr / rt-detr) map boxes back with. Like those decoders,
    a legacy input.crop is ignored (only architectures that map no boxes honour it)."""
    spec = bundle_spec(bundle)
    if spec.resize == "center_crop":
        spec = dataclasses.replace(spec, resize="squash")
    return spec_meta(spec, ow, oh)


# --------------------------------------------------------------------------------------------
# Decode raw outputs exactly like the Go architectures do
# --------------------------------------------------------------------------------------------

def _label(labels, k: int) -> str:
    return labels[k] if labels and 0 <= k < len(labels) else f"class_{k}"


def decode_outputs(outs: List[np.ndarray], bundle, meta: dict) -> Prediction:
    """Mirror of the Go decoders: rt-detr / rf-detr (per-query sigmoid argmax >= conf_threshold,
    cxcywh -> input px -> (x - pad)/scale, clamp, sort, max_detections; NO NMS), classification
    (softmax), depth ([1,H,W]), embed ([N,D] row 0)."""
    arch = bundle.architecture
    post = bundle.postprocess or {}
    if arch in ("rt-detr", "rf-detr"):
        box = next(o for o in outs if o.ndim == 3 and o.shape[-1] == 4)
        logit = next(o for o in outs if o.ndim == 3 and o is not box)
        p = sigmoid(logit[0])
        k = p.argmax(-1)
        s = p.max(-1)
        thr = float(post.get("conf_threshold", 0.5))
        W, H = bundle.width, bundle.height
        ow, oh = meta["orig_width"], meta["orig_height"]
        dets = []
        for q in np.nonzero(s >= thr)[0]:
            b = box[0, q].astype(np.float64)
            if post.get("box_format", "cxcywh") == "xyxy":
                x, y, w, h = b[0] * W, b[1] * H, (b[2] - b[0]) * W, (b[3] - b[1]) * H
            else:
                w, h = b[2] * W, b[3] * H
                x, y = b[0] * W - w / 2, b[1] * H - h / 2
            x, y = (x - meta["pad_x"]) / meta["scale_x"], (y - meta["pad_y"]) / meta["scale_y"]
            w, h = w / meta["scale_x"], h / meta["scale_y"]
            if x < 0:
                w, x = w + x, 0.0
            if y < 0:
                h, y = h + y, 0.0
            w, h = min(w, ow - x), min(h, oh - y)
            dets.append({"cls": _label(bundle.labels, int(k[q])), "conf": float(s[q]),
                         "bbox": [float(x), float(y), float(max(w, 0)), float(max(h, 0))]})
        dets.sort(key=lambda d: -d["conf"])
        md = int(post.get("max_detections", 0) or 0)
        return Prediction(detections=dets[:md] if md > 0 else dets)
    if arch in ("efficientnet", "mobilenet-v3"):
        pr = softmax(outs[0].reshape(-1))
        return Prediction(probs={_label(bundle.labels, i): float(v) for i, v in enumerate(pr)})
    if arch in ("midas", "depth-anything-v2"):
        d = np.asarray(outs[0])
        return Prediction(depth=d.reshape(d.shape[-2], d.shape[-1]))
    if arch in ("clip", "siglip-image"):
        return Prediction(embedding=np.asarray(outs[0])[0])
    raise ValueError(f"no generic decoder for architecture {arch!r}")


def normalise_prediction(out, bundle) -> Prediction:
    """A user predict()'s return value -> Prediction, by task."""
    if isinstance(out, Prediction):
        return out
    task = bundle.task
    if task in ("detection", "open_vocab"):
        from .metrics import as_dets
        return Prediction(detections=as_dets(out))
    if task == "classification":
        if isinstance(out, dict):
            return Prediction(probs={str(k): float(v) for k, v in out.items()})
        v = to_numpy(out).reshape(-1)
        return Prediction(probs={_label(bundle.labels, i): float(p) for i, p in enumerate(v)})
    if task == "depth":
        d = to_numpy(out)
        return Prediction(depth=d.reshape(d.shape[-2], d.shape[-1]))
    if task == "embed":
        return Prediction(embedding=to_numpy(out).reshape(-1))
    raise ValueError(f"cannot interpret a predict() result for task {task!r}")


# --------------------------------------------------------------------------------------------
# Generic references
# --------------------------------------------------------------------------------------------

class OnnxForward:
    """Raw outputs of the exported graph (CPU EP). Used when no framework module is in process."""

    def __init__(self, onnx_path):
        self.path = str(onnx_path)
        self._sess = None

    def __call__(self, feeds):
        import onnxruntime as ort
        if self._sess is None:
            self._sess = ort.InferenceSession(self.path, providers=["CPUExecutionProvider"])
        names = {i.name for i in self._sess.get_inputs()}
        return self._sess.run(None, {k: v for k, v in feeds.items() if k in names})


def _geometry_desc(bundle) -> str:
    return describe(bundle_spec(bundle))


class ManifestReference(Reference):
    kind = "manifest"
    framework = "onnx"

    def __init__(self, bundle, input_name: str, forward: Optional[Callable] = None, forward_desc: str = ""):
        self.bundle, self.input_name = bundle, input_name
        self._forward = forward
        self.short = "the manifest's declared preprocessing (numpy/PIL)"
        self.description = ("the manifest's declared preprocessing (numpy/PIL: "
                            f"{_geometry_desc(bundle)} to {bundle.width}x"
                            f"{bundle.height}, /255, mean/std) + {forward_desc or 'the exported ONNX'}, decoded "
                            "like the Go side — checks the Go implementation of the declared spec, NOT training "
                            "fidelity")

    def preprocess(self, pil, prompt=None):
        x, _ = apply_spec(pil, bundle_spec(self.bundle))
        return {self.input_name: x}

    def predict(self, pil, prompt=None):
        if self._forward is None:
            return None
        feeds = self.preprocess(pil)
        return decode_outputs(self._forward(feeds), self.bundle, manifest_meta(self.bundle, *pil.size))


class UserReference(Reference):
    """The user's own preprocess callable (Python API `preprocess=`) and/or `--reference-script`."""
    kind = "user"

    def __init__(self, bundle, input_name: str, preprocess_fn: Optional[Callable] = None,
                 predict_fn: Optional[Callable] = None, forward: Optional[Callable] = None,
                 description: str = "your preprocess"):
        self.bundle, self.input_name = bundle, input_name
        self._pre, self._pred, self._forward = preprocess_fn, predict_fn, forward
        self.description = description
        self.short_pre = description.split(" + ")[0]
        self.short = description.split(", decoded")[0]

    def preprocess(self, pil, prompt=None):
        if self._pre is None:
            return None
        x = to_numpy(self._pre(pil)).astype(np.float32)
        if x.ndim == 3:
            x = x[None]
        if x.ndim != 4:
            raise ValueError(f"your preprocess returned shape {x.shape}; expected CHW [3,H,W] (or [1,3,H,W])")
        return {self.input_name: np.ascontiguousarray(x)}

    def predict(self, pil, prompt=None):
        if self._pred is not None:
            return normalise_prediction(self._pred(pil), self.bundle)
        if self._forward is None or self._pre is None:
            return None
        # No user predict(): run the original model on the USER's tensor and decode it like Go.
        # Box mapping assumes the manifest's geometry (squash / letterbox) — a preprocess with a
        # different geometry needs its own predict() for detection.
        feeds = self.preprocess(pil)
        return decode_outputs(self._forward(feeds), self.bundle, manifest_meta(self.bundle, *pil.size))


def load_reference_script(path) -> dict:
    """Import a --reference-script; returns {"preprocess": fn, "predict": fn or None}."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"vs_reference_{path.stem}", str(path))
    if spec is None or spec.loader is None:
        raise ValueError(f"--reference-script {path}: not a loadable Python file")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pre = getattr(mod, "preprocess", None)
    if not callable(pre):
        raise ValueError(f"--reference-script {path.name} must define preprocess(pil_image) -> CHW float32 array")
    pred = getattr(mod, "predict", None)
    return {"preprocess": pre, "predict": pred if callable(pred) else None}


class TorchModuleReference(Reference):
    """An in-process torch module with no official pre/postprocessing (torch_generic): it supplies
    raw forward outputs (for B2 on the reference tensor) and the framework speed level."""
    kind = "manifest"
    description = "the original PyTorch module"

    def __init__(self, module, input_name: str):
        self.module, self.input_name = module, input_name

    def to(self, device):
        self.module.to(device)
        self.device = str(device)
        return self

    def forward(self, feeds):
        import torch
        x = torch.from_numpy(np.array(feeds[self.input_name], copy=True)).to(self.device)
        with torch.no_grad():
            out = self.module(x)
        out = out if isinstance(out, (list, tuple)) else [out]
        return [to_numpy(o) for o in out]

    def bench_fn(self, feeds):
        return torch_bench_fn(self.module, {self.input_name: feeds[self.input_name]}, self.device)
