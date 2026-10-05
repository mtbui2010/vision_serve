"""The manifest's preprocessing as data — the Python twin of internal/vision/preprocess.

A `Spec` says how an image becomes a model input (resize mode, size, multiple_of, no_upscale,
resample, mean/std, rescale, layout, pad). The Go server has ONE implementation of it
(vision/preprocess.Spec.Apply); this module re-implements the same semantics in numpy/PIL so the
converter's tier B1 compares the server against the declared spec directly:

  spec_from_manifest(doc)  a manifest dict -> Spec, with the Go registry's rules: the
                           `preprocess:` block, the legacy input.* fields as aliases, a field
                           declared on both sides must agree (registry.Manifest.PreprocessSpec);
  spec_from_legacy(...)    the legacy fields alone (Spec.legacy = True);
  resolve_arch(spec, arch) what that architecture's Go model applies for it (Arch.Resolve and
                           the package's own rules: SCRFD's legacy letterbox + 0..255 units, …);
  apply_spec(pil, spec)    -> (tensor, meta), the geometry exactly as Go computes it.

Both sides run the shared corpora internal/registry/testdata/preprocess_sync.json (resolution),
internal/vision/preprocess/testdata/arch_resolve_sync.json (per architecture) and
internal/vision/preprocess/testdata/geometry_sync.json (tensor shape + meta per mode); see
tests/test_go_python_sync.py. Pixels differ from Go only by the resampler (PIL vs imaging, the
same filters), which is what B1 measures.

numpy + PIL only (PIL imported inside functions).
"""
from __future__ import annotations

import dataclasses
import math
from typing import List, Optional

import numpy as np

MODES = ("squash", "letterbox", "center_crop", "keep_aspect", "long_side", "long_side_pad",
         "top_left_pad", "none")
PAD_MODES = ("letterbox", "top_left_pad", "long_side_pad")  # what legacy `letterbox: true` means
LAYOUTS = ("NCHW", "NHWC", "HWC")


@dataclasses.dataclass
class Spec:
    """vision/preprocess.Spec. resize "" = the architecture's default (the generic architectures
    the converter writes all default to squash). mean/std are in [0,1] units, or in 0..255 units
    when rescale is False. pad: letterbox / top_left_pad — a pixel gray level normalised like the
    image; long_side_pad — the value written into the NORMALISED tensor (SAM)."""
    resize: str = ""
    width: int = 0
    height: int = 0
    multiple_of: int = 0
    no_upscale: bool = False
    crop_pct: float = 0.0  # center_crop: kept fraction of the resized short side (timm); 0 = 1
    resample: str = ""
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None
    rescale: bool = True
    layout: str = ""
    pad: float = 0.0
    legacy: bool = False  # mapped from the legacy input.* fields, not declared in a block

    def to_block(self) -> dict:
        """The `preprocess:` block declaring this spec (only the fields that differ from the
        defaults)."""
        d = {"resize": self.resize or "squash"}
        if self.width == self.height:
            d["size"] = int(self.width)
        else:
            d["width"], d["height"] = int(self.width), int(self.height)
        if self.multiple_of:
            d["multiple_of"] = int(self.multiple_of)
        if self.no_upscale:
            d["no_upscale"] = True
        if self.crop_pct:
            d["crop_pct"] = float(self.crop_pct)
        if self.resample:
            d["resample"] = self.resample
        if self.mean is not None and self.std is not None:
            d["mean"], d["std"] = [float(v) for v in self.mean], [float(v) for v in self.std]
        if not self.rescale:
            d["rescale"] = False
        if self.layout:
            d["layout"] = self.layout
        if self.pad:
            d["pad"] = float(self.pad)
        return d


class SpecError(ValueError):
    """An invalid or self-contradicting preprocessing declaration."""


# --------------------------------------------------------------------------------------------
# Resolution (registry.Manifest.PreprocessSpec, preprocess.FromLegacy, Spec.Validate)
# --------------------------------------------------------------------------------------------

def spec_from_legacy(width: int, height: int, layout: str = "", letterbox: bool = False,
                     crop: Optional[str] = None, keep_aspect: bool = False, multiple_of: int = 0,
                     mean=None, std=None) -> Spec:
    """preprocess.FromLegacy: keep_aspect (+ multiple_of) > crop: center > letterbox > squash."""
    s = Spec(width=int(width or 0), height=int(height or 0), layout=str(layout or "").strip().upper(),
             mean=None if mean is None else [float(v) for v in mean],
             std=None if std is None else [float(v) for v in std], legacy=True)
    if keep_aspect:
        s.resize, s.multiple_of = "keep_aspect", int(multiple_of or 0)
    elif crop == "center":
        s.resize = "center_crop"
    elif letterbox:
        s.resize = "letterbox"
    else:
        s.resize = "squash"
    return s


def _f32(xs) -> list:
    return [float(np.float32(v)) for v in xs]


def _fmt_floats(xs) -> str:
    return "[" + " ".join(f"{float(np.float32(v)):g}" for v in xs) + "]"


# What yaml.v3 (the Go registry's parser) decodes into a bool field from a string: the YAML 1.1
# spellings, quoted or not. "true"/"false" QUOTED, numbers and anything else are an error there.
_YAML_TRUE = frozenset("y Y yes Yes YES on On ON".split())
_YAML_FALSE = frozenset("n N no No NO off Off OFF".split())


def _yaml_bool(v, field: str) -> Optional[bool]:
    """A manifest boolean read the way the Go registry reads it (None = absent / null). PyYAML
    already turns unquoted true/false/yes/no/on/off into bool; y/n and quoted spellings arrive as
    str. Python truthiness would read "n" or "no" as True."""
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v in _YAML_TRUE:
            return True
        if v in _YAML_FALSE:
            return False
    raise SpecError(f"{field}: {v!r} is not a boolean")


def spec_from_manifest(doc: dict) -> Spec:
    """A parsed manifest (dict) -> Spec, exactly as the Go registry resolves it. Raises SpecError
    for an invalid block or a block that contradicts its legacy alias (naming both fields)."""
    inp = doc.get("input") or {}
    norm = inp.get("normalize") or {}
    l_mean, l_std = norm.get("mean") or None, norm.get("std") or None
    pre = doc.get("preprocess")
    # A key set to null is absent, as in Go (a nil pointer after yaml.Unmarshal).
    has_letterbox = _yaml_bool(inp.get("letterbox"), "input.letterbox") is not None
    has_keep = _yaml_bool(inp.get("keep_aspect"), "input.keep_aspect") is not None
    has_mult = inp.get("multiple_of") is not None
    letterbox = bool(_yaml_bool(inp.get("letterbox"), "input.letterbox"))
    crop = inp.get("crop") or ""
    keep = bool(_yaml_bool(inp.get("keep_aspect"), "input.keep_aspect"))
    l_w, l_h = int(inp.get("width") or 0), int(inp.get("height") or 0)
    l_mult, l_layout = int(inp.get("multiple_of") or 0), str(inp.get("layout") or "")
    if pre is None:
        return spec_from_legacy(l_w, l_h, l_layout, letterbox, crop, keep, l_mult, l_mean, l_std)

    def conflict(legacy: str, block: str):
        return SpecError(f"{legacy} conflicts with {block}: the preprocess: block and its legacy input.* alias "
                         "must agree (or drop one of them)")

    def tf(v) -> str:
        return "true" if v else "false"

    s = Spec()
    if keep:
        s.resize = "keep_aspect"
    elif crop == "center":
        s.resize = "center_crop"
    elif letterbox:
        s.resize = "letterbox"
    raw_r = pre.get("resize")
    if raw_r is not None and not isinstance(raw_r, str):
        # PyYAML reads an unquoted `no` / `off` / `yes` / `on` as a bool; Go sees the string.
        raise SpecError(f'preprocess.resize "{raw_r}" is invalid ({", ".join(MODES)})')
    r = (raw_r or "").strip().lower()
    if r:
        if r not in MODES:
            raise SpecError(f'preprocess.resize "{pre.get("resize")}" is invalid ({", ".join(MODES)})')
        block = "preprocess.resize: " + r
        if has_letterbox and letterbox != (r in PAD_MODES):
            raise conflict(f"input.letterbox: {tf(letterbox)}", block)
        if crop and (crop == "center") != (r == "center_crop"):
            raise conflict(f"input.crop: {crop}", block)
        if has_keep and keep != (r == "keep_aspect"):
            raise conflict(f"input.keep_aspect: {tf(keep)}", block)
        s.resize = r

    size, w, h = int(pre.get("size") or 0), int(pre.get("width") or 0), int(pre.get("height") or 0)
    if size < 0 or w < 0 or h < 0:
        raise SpecError("preprocess.size/width/height must be > 0")
    if size > 0:
        if (w and w != size) or (h and h != size):
            raise SpecError(f"preprocess.size: {size} disagrees with preprocess.width/height {w}x{h}")
        w, h = size, size
    if w > 0 and l_w > 0 and l_w != w:
        raise conflict(f"input.width: {l_w}", f"preprocess width {w}")
    if h > 0 and l_h > 0 and l_h != h:
        raise conflict(f"input.height: {l_h}", f"preprocess height {h}")
    s.width, s.height = w or l_w, h or l_h

    mult = int(pre.get("multiple_of") or 0)
    if mult:
        if has_mult and l_mult != mult:
            raise conflict(f"input.multiple_of: {l_mult}", f"preprocess.multiple_of: {mult}")
        s.multiple_of = mult
    elif s.resize in ("keep_aspect", "long_side_pad"):
        s.multiple_of = l_mult

    b_mean, b_std = pre.get("mean") or None, pre.get("std") or None
    if b_mean and l_mean and _f32(l_mean) != _f32(b_mean):
        raise conflict(f"input.normalize.mean: {_fmt_floats(l_mean)}", f"preprocess.mean: {_fmt_floats(b_mean)}")
    if b_std and l_std and _f32(l_std) != _f32(b_std):
        raise conflict(f"input.normalize.std: {_fmt_floats(l_std)}", f"preprocess.std: {_fmt_floats(b_std)}")
    s.mean = [float(v) for v in (b_mean or l_mean)] if (b_mean or l_mean) else None
    s.std = [float(v) for v in (b_std or l_std)] if (b_std or l_std) else None

    legacy_layout = l_layout.strip().upper()
    b_layout = str(pre.get("layout") or "").strip().upper()
    if b_layout:
        if legacy_layout and legacy_layout != b_layout:
            raise conflict(f"input.layout: {l_layout}", f"preprocess.layout: {pre.get('layout')}")
        s.layout = b_layout
    else:
        s.layout = legacy_layout

    s.no_upscale = bool(_yaml_bool(pre.get("no_upscale"), "preprocess.no_upscale"))
    if pre.get("crop_pct") is not None:
        s.crop_pct = float(np.float32(pre["crop_pct"]))
    s.resample = str(pre.get("resample") or "").strip().lower()
    rescale = _yaml_bool(pre.get("rescale"), "preprocess.rescale")
    if rescale is not None:
        s.rescale = rescale
    if pre.get("pad") is not None:
        s.pad = float(np.float32(pre["pad"]))
    validate(s)
    return s


def validate(s: Spec) -> None:
    """preprocess.Spec.Validate: strict for a declared spec, lenient for a legacy one."""
    if s.resize and s.resize not in MODES:
        raise SpecError(f'preprocess: resize "{s.resize}" is invalid ({", ".join(MODES)})')
    if s.resize != "none" and (s.width <= 0 or s.height <= 0):
        raise SpecError(f"preprocess: resize {s.resize} needs width/height > 0 (got {s.width}x{s.height})")
    if s.multiple_of < 0:
        raise SpecError("preprocess: multiple_of must be >= 0")
    if s.layout not in ("",) + LAYOUTS:
        raise SpecError(f'preprocess: layout "{s.layout}" is invalid (NCHW, NHWC or HWC)')
    if s.resample not in ("", "bilinear", "bicubic"):
        raise SpecError(f'preprocess: resample "{s.resample}" is invalid (bilinear or bicubic)')
    if not all(math.isfinite(v) for v in list(s.mean or []) + list(s.std or []) + [s.pad, s.crop_pct]):
        raise SpecError(f"preprocess: mean, std, pad and crop_pct must be finite numbers (got mean {s.mean}, "
                        f"std {s.std}, pad {s.pad}, crop_pct {s.crop_pct})")
    if s.legacy:
        return
    if s.multiple_of > 0 and s.resize not in ("keep_aspect", "long_side_pad"):
        raise SpecError(f"preprocess: multiple_of applies to keep_aspect and long_side_pad, not {s.resize}")
    if s.crop_pct != 0 and s.resize != "center_crop":
        raise SpecError(f"preprocess: crop_pct applies to center_crop, not {s.resize}")
    if s.crop_pct < 0 or s.crop_pct > 1:
        raise SpecError(f"preprocess: crop_pct must be in (0, 1] (the kept fraction of the resized short side), "
                        f"got {s.crop_pct:g}")
    if s.no_upscale and s.resize not in ("long_side", "long_side_pad"):
        raise SpecError(f"preprocess: no_upscale applies to long_side and long_side_pad, not {s.resize}")
    if s.resize == "none" and s.resample:
        raise SpecError("preprocess: resize none does not resample")
    if s.pad != 0 and s.resize not in PAD_MODES:
        raise SpecError(f"preprocess: pad applies to letterbox, top_left_pad and long_side_pad, not {s.resize}")
    if s.resize in ("letterbox", "top_left_pad") and not 0 <= s.pad <= 255:
        raise SpecError(f"preprocess: pad {s.pad:g} is a pixel value for {s.resize} and must be in [0,255]")
    if (not s.mean) != (not s.std):
        raise SpecError(f"preprocess: mean and std go together (got {len(s.mean or [])} mean, "
                        f"{len(s.std or [])} std values)")
    if s.mean and (len(s.mean) != 3 or len(s.std) != 3):
        raise SpecError(f"preprocess: mean and std need 3 values each (RGB), got {len(s.mean)} and {len(s.std)}")
    if s.std and any(v == 0 or v != v for v in s.std):
        raise SpecError(f"preprocess: std values must be non-zero numbers, got {s.std}")


# --------------------------------------------------------------------------------------------
# Per-architecture resolution (preprocess.Arch.Resolve + each model package's spec())
# --------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Arch:
    """vision/preprocess.Arch: the modes an architecture serves (modes[0] is its default) and the
    name its errors carry."""
    name: str
    modes: tuple


# Registered architecture (manifest `architecture:`) -> its Arch, as the Go model packages declare
# them (internal/models/<pkg>/preprocess.go). tests/test_go_python_sync.py checks the modes
# against the Go source and every case of internal/vision/preprocess/testdata/arch_resolve_sync.json.
ARCHS = {
    "rf-detr": Arch("rfdetr", ("squash", "letterbox")),
    "rt-detr": Arch("rtdetr", ("squash", "letterbox")),
    "efficientnet": Arch("classification", ("squash", "center_crop")),
    "mobilenet-v3": Arch("classification", ("squash", "center_crop")),
    "clip": Arch("clip", ("squash", "center_crop")),
    "midas": Arch("depth", ("squash", "keep_aspect")),
    "depth-anything-v2": Arch("depth", ("squash", "keep_aspect")),
    "scrfd": Arch("scrfd", ("top_left_pad",)),
}

# Architectures whose export fixes the preprocessing (preprocess.FixedByExport): a declared block
# is refused, the legacy input.* fields are reference only.
FIXED_BY_EXPORT = ("mobile-sam", "efficient-sam", "sam2", "nano-sam", "paddle-ocr")

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]  # internal/models/clip defaultMean / defaultStd
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def resolve_arch(s: Spec, architecture: str) -> Optional[Spec]:
    """The spec `architecture` really applies for the manifest's spec `s` (spec_from_manifest), as
    the Go model does: its Arch.Resolve, after the package's own adjustments.

      - a legacy spec keeps the architecture's historical reading: a mode it never honoured falls
        back to its default (SCRFD's `letterbox: true` is its top-left pad; rf-detr ignores
        `crop: center`), and input.layout is never read (NCHW);
      - a declared block must use one of its modes ("" = the default), else SpecError;
      - scrfd: a legacy normalize is in 0..255 units (rescale off); clip: CLIP's mean/std when
        none is declared, 224 when no size is.

    Returns None for an architecture whose export fixes the preprocessing (its input.* fields
    are reference only; a declared block raises SpecError), and `s` unchanged for one that does
    not resolve a manifest's preprocessing through an Arch (a pipeline, or a model the converter
    generated with the generic default)."""
    if architecture in FIXED_BY_EXPORT:
        if not s.legacy:
            raise SpecError(f"preprocess: {architecture}'s preprocessing is fixed by its export and does not "
                            "read a preprocess: block — remove the block (see docs/manifest-spec.md)")
        return None
    a = ARCHS.get(architecture)
    if a is None:
        return s
    s = dataclasses.replace(s, mean=None if s.mean is None else list(s.mean),
                            std=None if s.std is None else list(s.std))
    if architecture == "scrfd" and s.legacy and (s.mean or s.std):
        s.rescale = False
    if architecture == "clip":
        s.width = s.width if s.width > 0 else 224
        s.height = s.height if s.height > 0 else 224
        s.mean = s.mean or list(CLIP_MEAN)
        s.std = s.std or list(CLIP_STD)
    if s.legacy:
        if s.resize not in a.modes:
            s.resize = a.modes[0]
        s.layout = "NCHW"
    else:
        if not s.resize:
            s.resize = a.modes[0]
        if s.resize not in a.modes:
            raise SpecError(f'preprocess: {a.name} does not support resize "{s.resize}" (supported: '
                            f'{", ".join(a.modes)})')
    try:
        validate(s)
    except SpecError as e:
        raise SpecError(f"{a.name}: {e}") from None
    return s


def unit_normalisation(s: Spec):
    """(mean, std) in [0,1] units giving the same tensor as the spec: v = (p/255 - mean) / std.
    A spec with rescale off (mean/std in 0..255 units, or raw pixels) is converted; a rescaled one
    is returned as declared (None when it declares none)."""
    if s.rescale:
        return s.mean, s.std
    if not s.mean and not s.std:
        return [0.0, 0.0, 0.0], [1.0 / 255] * 3  # raw 0..255: v = p
    m, sd = _channels(s.mean, s.std, 255.0)
    return [float(v) for v in m], [float(v) for v in sd]


# --------------------------------------------------------------------------------------------
# Geometry (internal/vision/preprocess/geometry.go, float64 like Go)
# --------------------------------------------------------------------------------------------

def _round_half_away(x: float) -> int:
    """Go's math.Round for x >= 0 (Python's round() is half-to-even)."""
    f = math.floor(x)
    return int(f) + (1 if x - f >= 0.5 else 0)


def letterbox_size(w: int, h: int, W: int, H: int):
    scale = W / w
    if H / h < scale:
        scale = H / h
    nw, nh = max(1, int(w * scale + 0.5)), max(1, int(h * scale + 0.5))
    return nw, nh, scale, (W - nw) // 2, (H - nh) // 2


def cover_size(w: int, h: int, W: int, H: int):
    """CLIPImageProcessor: short side -> target, long side truncated; offset floored."""
    if w <= h:
        rw, rh = W, int(W * h / w)
    else:
        rw, rh = int(H * w / h), H
    rw, rh = max(rw, W), max(rh, H)
    return rw, rh, (rw - W) // 2, (rh - H) // 2


def center_crop_size(w: int, h: int, W: int, H: int, crop_pct: float = 0.0):
    """vision/preprocess.CenterCropSize: cover floor(W/crop_pct) x floor(H/crop_pct) (timm's
    math.floor(size / crop_pct)), keep the centred W x H; crop_pct <= 0 or >= 1 is cover_size."""
    sw, sh = W, H
    p = float(np.float32(crop_pct))
    if 0 < p < 1:
        sw, sh = int(math.floor(W / p + 1e-4)), int(math.floor(H / p + 1e-4))
    rw, rh, _, _ = cover_size(w, h, sw, sh)
    return rw, rh, (rw - W) // 2, (rh - H) // 2


def dpt_keep_aspect_size(w: int, h: int, tw: int, th: int, multiple: int = 1):
    """(new_w, new_h) for keep_aspect — vision/preprocess.DPTKeepAspectSize, which is HF's DPT
    get_resize_output_image_size(keep_aspect_ratio=True): both axes take whichever of tw/w, th/h is
    closer to 1 (ties: the height's), each side = round(scale*side / m) * m with Python's
    half-to-even round. Like the Go side (and unlike HF, which returns 0) a side is never < m."""
    m = int(multiple) if multiple and multiple > 0 else 1
    sw, sh = tw / w, th / h
    if abs(1 - sw) < abs(1 - sh):
        sh = sw
    else:
        sw = sh
    return max(m, round(sw * w / m) * m), max(m, round(sh * h / m) * m)


def top_left_size(w: int, h: int, W: int, H: int):
    """(new_w, new_h): InsightFace scrfd.py detect()'s fit by the aspect ratios, sides truncated
    (at least 1) — vision/preprocess.TopLeftSize. Like the Go side it does NOT return upstream's
    det_scale = new_h / h: the sides are truncated separately, so the content's x scale is
    new_w / w (a 10000x10 panorama -> 640x1: x scaled by 0.064, det_scale says 0.1). The Meta
    records each axis's own scale."""
    im_ratio, model_ratio = h / w, H / W
    if im_ratio > model_ratio:
        nh = H
        nw = int(nh / im_ratio)
    else:
        nw = W
        nh = int(nw * im_ratio)
    return max(1, nw), max(1, nh)


def long_side_size(w: int, h: int, W: int, H: int, no_upscale: bool = False):
    scale = min(W / w, H / h)
    if no_upscale and scale > 1.0:
        scale = 1.0
    return max(1, _round_half_away(w * scale)), max(1, _round_half_away(h * scale)), scale


# --------------------------------------------------------------------------------------------
# Apply
# --------------------------------------------------------------------------------------------

def _resample(spec: Spec):
    from PIL import Image
    r = spec.resample or ("bicubic" if spec.resize in ("center_crop", "keep_aspect") else "bilinear")
    return Image.BICUBIC if r == "bicubic" else Image.BILINEAR


def _channels(mean, std, scale: float = 1.0):
    """vision/preprocess.normalizer: 3 per-channel values; a missing mean entry is 0, a missing or
    zero std entry is 1 (after dividing by `scale`), extra entries are ignored — what the Go side
    (and the hand-written loops it replaced) does for odd legacy normalize lists."""
    m = np.zeros(3, np.float32)
    sd = np.ones(3, np.float32)
    for c, v in enumerate(list(mean or [])[:3]):
        m[c] = np.float32(v) / np.float32(scale)
    for c, v in enumerate(list(std or [])[:3]):
        v = np.float32(v) / np.float32(scale)
        if v != 0:
            sd[c] = v
    return m, sd


def _normalise(x: np.ndarray, spec: Spec) -> np.ndarray:
    """x: float32 [H,W,3] pixel values (0..255) -> the tensor values (vision/preprocess normalizer:
    v = (p/255 - mean[c]) / std[c]; with rescale off, mean/std are in 0..255 units, and no mean/std
    at all keeps v = p)."""
    if not spec.rescale and not spec.mean and not spec.std:
        return x
    mean, std = _channels(spec.mean, spec.std, 1.0 if spec.rescale else 255.0)
    return (x / np.float32(255) - mean) / std


def _layout(x: np.ndarray, spec: Spec) -> np.ndarray:
    # A legacy spec is always fed NCHW: no architecture ever read input.layout.
    lay = "NCHW" if spec.legacy else (spec.layout or "NCHW")
    if lay == "NHWC":
        return np.ascontiguousarray(x[None]).astype(np.float32)
    if lay == "HWC":
        return np.ascontiguousarray(x).astype(np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]).astype(np.float32)


def _meta(ow, oh, sx, sy, px=0, py=0) -> dict:
    return {"orig_width": ow, "orig_height": oh, "scale_x": sx, "scale_y": sy, "pad_x": px, "pad_y": py}


def apply_spec(pil, spec: Spec):
    """-> (tensor float32, meta) — what vision/preprocess.Spec.Apply feeds for `spec`, with PIL's
    resampler (imaging.Linear / CatmullRom are PIL's BILINEAR / BICUBIC filters)."""
    img = pil.convert("RGB")
    ow, oh = img.size
    W, H, mode = spec.width, spec.height, spec.resize or "squash"
    rs = _resample(spec)
    if mode == "keep_aspect":
        nw, nh = dpt_keep_aspect_size(ow, oh, W, H, spec.multiple_of)
        x = np.asarray(img.resize((nw, nh), rs), np.float32)
        meta = _meta(ow, oh, nw / ow, nh / oh)
    elif mode == "center_crop":
        rw, rh, ox, oy = center_crop_size(ow, oh, W, H, spec.crop_pct)
        x = np.asarray(img.resize((rw, rh), rs).crop((ox, oy, ox + W, oy + H)), np.float32)
        meta = _meta(ow, oh, rw / ow, rh / oh, -ox, -oy)
    elif mode in ("letterbox", "top_left_pad"):
        if mode == "letterbox":
            nw, nh, s, px, py = letterbox_size(ow, oh, W, H)
            sx = sy = s
        else:
            (nw, nh), px, py = top_left_size(ow, oh, W, H), 0, 0
            sx, sy = nw / ow, nh / oh
        # The canvas is a pixel image: the pad is normalised like the image.
        x = np.full((H, W, 3), np.float32(spec.pad), np.float32)
        content = np.asarray(img.resize((nw, nh), rs), np.float32)
        x[py:py + nh, px:px + nw] = content[:H - py, :W - px]
        meta = _meta(ow, oh, sx, sy, px, py)
    elif mode in ("long_side", "long_side_pad"):
        nw, nh, s = long_side_size(ow, oh, W, H, spec.no_upscale)
        x = _normalise(np.asarray(img.resize((nw, nh), rs), np.float32), spec)
        if mode == "long_side_pad":
            out_w, out_h = W, H
            if spec.multiple_of > 0:
                m = spec.multiple_of
                out_w, out_h = -(-nw // m) * m, -(-nh // m) * m
            # SAM: normalise, then pad the NORMALISED tensor (bottom/right) with `pad`.
            padded = np.full((out_h, out_w, 3), np.float32(spec.pad), np.float32)
            padded[:min(nh, out_h), :min(nw, out_w)] = x[:out_h, :out_w]
            x = padded
        return _layout(x, spec), _meta(ow, oh, s, s)
    elif mode == "none":
        x = np.asarray(img, np.float32)
        meta = _meta(ow, oh, 1.0, 1.0)
    else:  # squash
        x = np.asarray(img.resize((W, H), rs), np.float32)
        meta = _meta(ow, oh, W / ow, H / oh)
    return _layout(_normalise(x, spec), spec), meta


def spec_meta(spec: Spec, ow: int, oh: int) -> dict:
    """The Meta apply_spec returns for an ow x oh image, without touching pixels."""
    W, H, mode = spec.width, spec.height, spec.resize or "squash"
    if mode == "keep_aspect":
        nw, nh = dpt_keep_aspect_size(ow, oh, W, H, spec.multiple_of)
        return _meta(ow, oh, nw / ow, nh / oh)
    if mode == "center_crop":
        rw, rh, ox, oy = center_crop_size(ow, oh, W, H, spec.crop_pct)
        return _meta(ow, oh, rw / ow, rh / oh, -ox, -oy)
    if mode == "letterbox":
        _, _, s, px, py = letterbox_size(ow, oh, W, H)
        return _meta(ow, oh, s, s, px, py)
    if mode == "top_left_pad":
        nw, nh = top_left_size(ow, oh, W, H)
        return _meta(ow, oh, nw / ow, nh / oh)
    if mode in ("long_side", "long_side_pad"):
        s = long_side_size(ow, oh, W, H, spec.no_upscale)[2]
        return _meta(ow, oh, s, s)
    if mode == "none":
        return _meta(ow, oh, 1.0, 1.0)
    return _meta(ow, oh, W / ow, H / oh)


def describe(spec: Spec) -> str:
    """A few words for reports: the geometry and filter."""
    mode = spec.resize or "squash"
    if mode == "keep_aspect":
        return f"keep-aspect {_filter_name(spec)} (multiple of {spec.multiple_of or 1}) around"
    if mode == "center_crop":
        if spec.crop_pct and spec.crop_pct < 1:
            return f"centre crop {_filter_name(spec)} (crop_pct {spec.crop_pct:g})"
        return f"centre crop {_filter_name(spec)}"
    return f"{mode.replace('_', ' ')} {_filter_name(spec)}"


def _filter_name(spec: Spec) -> str:
    return spec.resample or ("bicubic" if spec.resize in ("center_crop", "keep_aspect") else "bilinear")
