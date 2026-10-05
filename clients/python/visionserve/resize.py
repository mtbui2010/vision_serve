"""Client-side resizing: shrink a large image to what the model can use before uploading it.

A model resizes every image to its own fixed input (RF-DETR 560x560, GroundingDINO 800x800, CLIP
224x224, ...) on the server. A 12 MP photo therefore spends most of its upload and decode time on
pixels the model throws away. The server publishes, per model, how far an image can be shrunk
without changing what the model sees (``GET /api/models``: ``max_useful_side`` bounds the
image's LONGER side, ``max_useful_short_side`` its SHORTER side; null = never shrink), and the
:class:`~visionserve.Client` shrinks to it and sends JPEG by default. Results are mapped back,
so every coordinate is still in ORIGINAL image pixels.

The rules, in order (see :func:`prepare_upload`):

* no hint for the model (masks, OCR, depth-aligned grasping, templates, ...) or ``resize="off"``:
  the image is sent exactly as before this feature: a path / bytes verbatim, a PIL image or
  ndarray as lossless PNG;
* the image (or the ``roi`` region, when one is given) is larger than the hint: it is decoded
  (JPEG at a reduced DCT scale, Pillow's draft mode),
  EXIF-rotated the way the server would rotate it (JPEG only), alpha dropped (the server's
  tensor ignores alpha too), shrunk with a Lanczos filter, and sent as JPEG (``jpeg_quality``,
  4:4:4 chroma) — or PNG with ``jpeg=False``;
* anything that is not shrunk goes out exactly as before (no re-encode);
* the loopback rule: when the server is on this machine (127.0.0.0/8, ::1, localhost) and
  ``resize="auto"``, a photo is shrunk only when that halves its sides or more (scale <= 0.5):
  on localhost the upload is free, and a milder shrink costs the client more to decode than the
  server saves. ``Result.client_resize.reason`` says when this rule kept a photo whole.

Needs Pillow; without it every image is sent as before (no warning — the result is correct, only
the upload is larger).
"""

from __future__ import annotations

import io
import math
import numbers
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

RESIZE_AUTO = "auto"
RESIZE_OFF = "off"

ResizeOption = Union[str, int]

# EXIF orientations that swap width and height (rotated by 90 or 270 degrees).
_SWAPPING_ORIENTATIONS = (5, 6, 7, 8)


def check_resize(value: Any) -> ResizeOption:
    """Validate a ``resize`` option: ``"auto"``, ``"off"``, or a positive int (the longest side
    to send, in pixels, whatever the model's hint says)."""
    if isinstance(value, str):
        v = value.strip().lower()
        if v in (RESIZE_AUTO, RESIZE_OFF):
            return v
        if v.isdigit():
            value = int(v)
        else:
            raise ValueError("resize must be 'auto', 'off' or a positive int, got %r" % (value,))
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) <= 0:
        raise ValueError("resize must be 'auto', 'off' or a positive int, got %r" % (value,))
    return int(value)


def check_quality(value: Any) -> int:
    """Validate ``jpeg_quality``: an int in 1..100."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not 1 <= int(value) <= 100:
        raise ValueError("jpeg_quality must be an int in 1..100, got %r" % (value,))
    return int(value)


@dataclass(frozen=True)
class ClientResize:
    """What the client did to the image before uploading it (:attr:`Result.client_resize`).

    Attributes:
        original_width, original_height: the image as the server would have seen it (after its
            EXIF rotation): the frame every returned coordinate is in.
        sent_width, sent_height: the image actually uploaded.
        jpeg_quality: the JPEG quality it was encoded at, or ``None`` when it was sent as PNG
            (``jpeg=False``) or not re-encoded.
        reason: why: ``"hint"`` (shrunk to the server's hint for the model), ``"resize=N"``
            (shrunk to an explicit longest side), or ``"loopback: scale 0.53 > 0.5, sent as is"``
            (larger than the hint, but the server is on this machine and shrinking would remove
            too little to pay for the client's decode; the photo went out exactly as given,
            ``resized`` is False).
    """

    original_width: int
    original_height: int
    sent_width: int
    sent_height: int
    jpeg_quality: Optional[int]
    reason: str = "hint"

    @property
    def resized(self) -> bool:
        """True when the uploaded image is smaller than the original (else it was only re-encoded)."""
        return (self.sent_width, self.sent_height) != (self.original_width, self.original_height)

    @property
    def scale_x(self) -> float:
        """``sent_width / original_width``: a sent-image x is ``original_x * scale_x``."""
        return self.sent_width / float(self.original_width)

    @property
    def scale_y(self) -> float:
        """``sent_height / original_height``."""
        return self.sent_height / float(self.original_height)

    # Multiply first, then divide: 300 * 1680 / 3000 is exactly 168, 300 * (1680 / 3000) is not.
    def to_sent_x(self, x: float) -> float:
        return x * self.sent_width / float(self.original_width)

    def to_sent_y(self, y: float) -> float:
        return y * self.sent_height / float(self.original_height)

    def to_original_x(self, x: float) -> float:
        return x * self.original_width / float(self.sent_width)

    def to_original_y(self, y: float) -> float:
        return y * self.original_height / float(self.sent_height)


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def target_size(
    width: int,
    height: int,
    *,
    max_side: Optional[int] = None,
    max_short_side: Optional[int] = None,
    region: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    """The size to send a ``width`` x ``height`` image at (unchanged when it is small enough).

    ``max_side`` bounds the LONGER side, ``max_short_side`` the SHORTER side (give one). With
    ``region`` = ``(w, h)`` of a region of interest, the bound applies to the region — the part the
    model will see — and the whole image is scaled by the same factor. One scale for both axes
    (the aspect ratio is kept); each side is rounded half up, never below 1.
    """
    rw, rh = region if region is not None else (width, height)
    if max_side:
        s = max_side / float(max(rw, rh))
    elif max_short_side:
        s = max_short_side / float(min(rw, rh))
    else:
        return width, height
    if s >= 1.0:
        return width, height
    return max(1, _round_half_up(width * s)), max(1, _round_half_up(height * s))


def roi_region(roi: Any, width: int, height: int) -> Optional[Tuple[int, int]]:
    """``(w, h)`` in pixels of the region the server will crop for ``roi`` on a ``width`` x
    ``height`` image (``internal/roi.Clamp``: fractions when both w and h are <= 1, rounded,
    clamped to the image), or ``None`` when it is unset or degenerate (the server then uses the
    whole image)."""
    if roi is None:
        return None
    x, y, w, h = (float(v) for v in roi)
    if w <= 0 or h <= 0:
        return None
    if w <= 1.0 and h <= 1.0:
        x, y, w, h = x * width, y * height, w * width, h * height
    x0, y0 = max(0, _round_go(x)), max(0, _round_go(y))
    x1, y1 = min(width, _round_go(x + w)), min(height, _round_go(y + h))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return x1 - x0, y1 - y0


def _round_go(v: float) -> int:
    """Go's math.Round: half away from zero."""
    return int(math.floor(abs(v) + 0.5)) * (1 if v >= 0 else -1)


# ---------------------------------------------------------------------- #
# Encoding
# ---------------------------------------------------------------------- #
def _pil():
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        return None
    return Image


def _to_rgb(img):
    """RGB pixels the way the server reads them: alpha dropped (its tensor ignores alpha),
    16-bit gray as its high byte (Go decodes Gray16 and keeps ``>> 8``; Pillow's own
    ``convert`` would clip instead)."""
    if img.mode == "RGB":
        return img
    if img.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I"):
        import numpy as np

        Image = _pil()
        a = np.asarray(img).astype(np.int64)
        a = np.clip(a >> 8, 0, 255).astype(np.uint8)
        return Image.fromarray(a, "L").convert("RGB")
    return img.convert("RGB")


def _encode(img, jpeg: bool, quality: int) -> Tuple[bytes, str]:
    buf = io.BytesIO()
    if jpeg:
        # 4:4:4 chroma (no subsampling). A shrunk image is only ~2x the model's input, so 4:2:0
        # would halve its colour to exactly the model's resolution: measured on CLIP, 4:2:0
        # moved the embedding to a mean cosine of 0.976 with the full-resolution one, 4:4:4 to
        # 0.995, for ~25% more bytes.
        img.save(buf, format="JPEG", quality=quality, subsampling=0)
        return buf.getvalue(), "image.jpg"
    img.save(buf, format="PNG", compress_level=1)
    return buf.getvalue(), "image.png"


class _Input:
    """An image input, inspected lazily: ``raw`` encoded bytes (path / bytes input) or ``pil``."""

    def __init__(self, raw: Optional[bytes], name: str, pil: Any):
        self.raw, self.name, self.pil = raw, name, pil
        self.is_jpeg = False
        self.orientation: Optional[int] = None


def _open_input(image: Any, ndarray_to_pil) -> Optional[_Input]:
    """Read ``image`` into an :class:`_Input`, or ``None`` for an input type the caller encodes
    the old way (unknown types raise there)."""
    if isinstance(image, (str, os.PathLike)):
        p = Path(image)
        return _Input(p.read_bytes(), p.name, None)
    if isinstance(image, (bytes, bytearray)):
        return _Input(bytes(image), "image", None)
    Image = _pil()
    if Image is not None and isinstance(image, Image.Image):
        return _Input(None, "image", image)
    try:
        import numpy as np
    except ImportError:
        return None
    if isinstance(image, np.ndarray):
        return _Input(None, "image", ndarray_to_pil(image))
    return None


def prepare_upload(
    image: Any,
    *,
    max_side: Optional[int],
    max_short_side: Optional[int],
    jpeg: bool,
    quality: int,
    roi: Any = None,
    encode_plain,
    ndarray_to_pil,
    reason: str = "hint",
    max_scale: Optional[float] = None,
) -> Tuple[bytes, str, Optional[ClientResize]]:
    """The bytes to upload for ``image`` under a hint, with what was done to it.

    ``max_side`` / ``max_short_side`` is the hint (both ``None``: send as before via
    ``encode_plain``). Only a photo that is SHRUNK is re-encoded; anything else goes out exactly
    as before this feature. ``max_scale`` (the loopback rule): shrink only when the scale
    (sent / original side) is at most this, else send as before and say why in the returned
    record. ``reason`` is recorded on a shrink ("hint" or "resize=N").

    Returns ``(bytes, filename, ClientResize or None)``; ``None`` means the input went out
    exactly as before (no hint, small enough, or not decodable here).
    """
    if not max_side and not max_short_side:
        data, name = encode_plain(image)
        return data, name, None
    Image = _pil()
    if Image is None:
        data, name = encode_plain(image)
        return data, name, None
    src = _open_input(image, ndarray_to_pil)
    if src is None:
        data, name = encode_plain(image)
        return data, name, None

    img = src.pil
    if src.raw is not None:
        try:
            img = Image.open(io.BytesIO(src.raw))
        except Exception:  # not an image Pillow reads: the server decides  # noqa: BLE001
            return src.raw, _plain_name(src.name), None
        src.is_jpeg = img.format == "JPEG"
        if src.is_jpeg:
            try:
                src.orientation = img.getexif().get(0x0112)
            except Exception:  # noqa: BLE001 — a broken EXIF block: the server ignores it too
                src.orientation = None
    w, h = img.size
    if src.orientation in _SWAPPING_ORIENTATIONS:
        w, h = h, w

    tw, th = target_size(w, h, max_side=max_side, max_short_side=max_short_side,
                         region=roi_region(roi, w, h) if roi is not None else None)
    resized = (tw, th) != (w, h)
    skipped = None
    if resized and max_scale is not None:
        scale = max(tw / float(w), th / float(h))
        if scale > max_scale:
            resized = False
            skipped = ClientResize(w, h, w, h, None, "loopback: scale %.2f > %g, sent as is" % (scale, max_scale))
    if not resized:
        # Not shrunk: the input goes out exactly as before this feature (a path / bytes verbatim,
        # a PIL image / array as lossless PNG). Re-encoding a photo the model reads at its own
        # size gains little and changes results (RF-DETR on arrays sent as JPEG: -0.17 mAP).
        if src.raw is not None:
            return src.raw, _plain_name(src.name), skipped
        data, name = encode_plain(image)
        return data, name, skipped

    if src.raw is not None and src.is_jpeg:
        # Let libjpeg decode at 1/2, 1/4 or 1/8 scale (never below the target): a 12 MP photo
        # decodes about 3x faster. The size is in the STORED orientation.
        img.draft("RGB", (th, tw) if src.orientation in _SWAPPING_ORIENTATIONS else (tw, th))
    if src.raw is not None and src.is_jpeg and src.orientation not in (None, 1):
        from PIL import ImageOps

        img = ImageOps.exif_transpose(img)
    img = _to_rgb(img)
    if resized:
        # Lanczos, not the server's own bilinear: two antialiased bilinear passes blur twice.
        # Measured on 500 COCO val images upscaled 4x through RF-DETR: bilinear -0.27 mAP,
        # Lanczos -0.02 (CPU cost: ~40 ms more per 12 MP photo, after the JPEG draft decode).
        img = img.resize((tw, th), Image.LANCZOS)
    data, name = _encode(img, jpeg, quality)
    return data, name, ClientResize(w, h, tw, th, quality if jpeg else None, reason)


def _plain_name(name: str) -> str:
    return name if name and name != "image" else "image.png"


# ---------------------------------------------------------------------- #
# Prompt and result mapping
# ---------------------------------------------------------------------- #
def scale_boxes(boxes: List[Sequence[float]], cr: ClientResize) -> List[List[float]]:
    """``[x, y, w, h]`` boxes from ORIGINAL to sent-image pixels (a malformed entry is left for
    the serializer to refuse)."""
    fx, fy = cr.to_sent_x, cr.to_sent_y
    return [[fx(b[0]), fy(b[1]), fx(b[2]), fy(b[3])] if _numbers(b, (4,)) else b for b in boxes]


def scale_points(points: List[Sequence[float]], cr: ClientResize) -> List[List[float]]:
    """``[x, y(, label)]`` points from ORIGINAL to sent-image pixels (the label is kept)."""
    return [[cr.to_sent_x(p[0]), cr.to_sent_y(p[1])] + list(p[2:]) if _numbers(p, (2, 3)) else p
            for p in points]


def _numbers(seq: Any, lengths: Tuple[int, ...]) -> bool:
    return (isinstance(seq, (list, tuple)) and len(seq) in lengths
            and all(isinstance(v, numbers.Real) and not isinstance(v, bool) for v in seq))


def scale_roi(roi: Sequence[float], cr: ClientResize) -> List[float]:
    """A pixel ROI to sent-image pixels; a fractional one (w and h <= 1) is unchanged."""
    if float(roi[2]) <= 1.0 and float(roi[3]) <= 1.0:
        return [float(v) for v in roi]
    return scale_boxes([roi], cr)[0]


def map_result(result: Any, cr: ClientResize) -> Any:
    """Map a Result computed on the sent image back to ORIGINAL pixels (in place; returned).

    Detection and mask boxes, grasp centres, jaw widths and angles (exactly, for the two axes'
    slightly different rounding) are mapped. A mask's RLE stays as received — at the sent size —
    and the mask records that size, so :meth:`Mask.to_ndarray` with the ORIGINAL size returns an
    original-size mask (nearest neighbour). The depth map stays at the model's resolution, as
    always. ``result.client_resize`` is set to ``cr``.
    """
    sx, sy = cr.scale_x, cr.scale_y
    fx, fy = cr.to_original_x, cr.to_original_y
    if cr.resized:
        for d in result.detections:
            d.bbox = [fx(d.bbox[0]), fy(d.bbox[1]), fx(d.bbox[2]), fy(d.bbox[3])]
        for m in result.masks:
            m.bbox = [fx(m.bbox[0]), fy(m.bbox[1]), fx(m.bbox[2]), fy(m.bbox[3])]
            if m.rle:
                m._rle_size = (cr.sent_width, cr.sent_height)
        for g in result.grasps:
            dx, dy = math.cos(g.theta) / sx, math.sin(g.theta) / sy
            g.x, g.y = fx(g.x), fy(g.y)
            g.width = g.width * math.hypot(dx, dy)
            g.theta = math.atan2(dy, dx)
    result.client_resize = cr
    return result
