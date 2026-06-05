"""Shared image preprocessing for the Python baselines.

Mirrors VisionServe's manifest-driven preprocess (``input:`` block in ``models/*/manifest.yaml``):
resize (optionally letterboxed) to WxH, NCHW layout, per-channel mean/std normalization.

This is intentionally a faithful, small reimplementation so the FastAPI+ORT baseline feeds
the ONNX graph the *same* tensor VisionServe would — the W1 control is "same ONNX, same input,
different serving layer". Pure numpy + Pillow (no torch) to keep the ORT baseline light.

NOTE: letterbox padding here matches the common "pad to square, center" convention. If a given
model's Go preprocess differs in a detail (pad color, align corners), verify against
``internal/models/<name>`` before trusting absolute-coordinate detection outputs. For W1 latency
this does not matter (we measure serving overhead); for accuracy parity use ``accuracy/parity.py``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

# ImageNet defaults (match most VisionServe manifests).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class PreprocessConfig:
    width: int = 224
    height: int = 224
    layout: str = "NCHW"
    letterbox: bool = False
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    scale: float = 255.0  # divide pixel [0,255] by this before (x-mean)/std

    @classmethod
    def from_manifest(cls, manifest_input: dict) -> "PreprocessConfig":
        """Build from a parsed manifest ``input:`` mapping."""
        norm = manifest_input.get("normalize", {}) or {}
        return cls(
            width=int(manifest_input.get("width", 224)),
            height=int(manifest_input.get("height", 224)),
            layout=manifest_input.get("layout", "NCHW"),
            letterbox=bool(manifest_input.get("letterbox", False)),
            mean=tuple(norm.get("mean", IMAGENET_MEAN)),  # type: ignore[arg-type]
            std=tuple(norm.get("std", IMAGENET_STD)),  # type: ignore[arg-type]
        )


@dataclass
class PreprocessMeta:
    """Bookkeeping to map model-space coords back to original image coords (letterbox)."""

    orig_w: int
    orig_h: int
    scale: float = 1.0
    pad_x: int = 0
    pad_y: int = 0
    extra: dict = field(default_factory=dict)


def _resize_letterbox(img: Image.Image, w: int, h: int) -> tuple[Image.Image, PreprocessMeta]:
    ow, oh = img.size
    scale = min(w / ow, h / oh)
    nw, nh = int(round(ow * scale)), int(round(oh * scale))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (w, h), (114, 114, 114))  # YOLO-style gray pad
    pad_x, pad_y = (w - nw) // 2, (h - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, PreprocessMeta(orig_w=ow, orig_h=oh, scale=scale, pad_x=pad_x, pad_y=pad_y)


def preprocess(image_bytes: bytes, cfg: PreprocessConfig) -> tuple[np.ndarray, PreprocessMeta]:
    """Decode + resize + normalize -> (NCHW float32 batch=1 tensor, meta)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if cfg.letterbox:
        img, meta = _resize_letterbox(img, cfg.width, cfg.height)
    else:
        ow, oh = img.size
        img = img.resize((cfg.width, cfg.height), Image.BILINEAR)
        meta = PreprocessMeta(orig_w=ow, orig_h=oh,
                              scale=cfg.width / ow, extra={"scale_y": cfg.height / oh})

    arr = np.asarray(img, dtype=np.float32) / cfg.scale  # HWC, [0,1]
    mean = np.array(cfg.mean, dtype=np.float32)
    std = np.array(cfg.std, dtype=np.float32)
    arr = (arr - mean) / std
    if cfg.layout.upper() == "NCHW":
        arr = np.transpose(arr, (2, 0, 1))  # CHW
    arr = np.expand_dims(arr, 0)  # add batch
    return np.ascontiguousarray(arr, dtype=np.float32), meta
