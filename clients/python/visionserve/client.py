"""HTTP client for the VisionServe server.

Transport uses only the Python standard library (``urllib``), so the client has no
hard third-party dependencies. ``numpy`` and ``pillow`` are optional and only needed
for ndarray / PIL image inputs and for :meth:`Mask.to_ndarray`.
"""

from __future__ import annotations

import io
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib import error as urllib_error
from urllib import request as urllib_request

from .types import Detection, Mask, ModelInfo, Result, _is_loaded

# Type alias for accepted image inputs (documented in predict()).
ImageInput = Union[str, "os.PathLike[str]", bytes, "Any"]  # Any covers PIL / ndarray

BoxInput = Union[Sequence[float], Sequence[Sequence[float]], None]
PointInput = Union[Sequence[float], Sequence[Sequence[float]], None]


class VisionServeError(Exception):
    """Raised when the server returns a non-2xx response or transport fails."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class Client:
    """Client for the VisionServe HTTP API.

    Args:
        host:    base URL of the server, e.g. ``http://localhost:11435``.
        timeout: per-request timeout in seconds.
    """

    def __init__(self, host: str = "http://localhost:11435", timeout: float = 120):
        self.host = host.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def health(self) -> Dict[str, str]:
        """GET /api/health -> ``{"status": "ok"}``."""
        return self._get_json("/api/health")

    def list_models(self) -> List[ModelInfo]:
        """GET /api/models -> list of :class:`ModelInfo`."""
        data = self._get_json("/api/models")
        return [ModelInfo.from_json(x) for x in (data or [])]

    def load(self, model: str) -> Dict[str, str]:
        """POST /api/load -> ``{"model", "state"}``."""
        return self._post_json("/api/load", {"model": model})

    def unload(self, model: str) -> Dict[str, str]:
        """POST /api/unload -> ``{"model", "state"}``."""
        return self._post_json("/api/unload", {"model": model})

    def ps(self) -> List[ModelInfo]:
        """Return only the currently loaded models (filtered from /api/models)."""
        return [m for m in self.list_models() if _is_loaded(m)]

    def predict(
        self,
        model: str,
        image: ImageInput,
        *,
        prompt: Optional[str] = None,
        box: BoxInput = None,
        point: PointInput = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        bg_max_area: Optional[float] = None,
        fg_min_area: Optional[float] = None,
        grid_size: Optional[int] = None,
        method: Optional[str] = None,
        roi: BoxInput = None,
        dilate: Optional[int] = None,
        depth: "Any" = None,
        min_size: Optional[float] = None,
        max_size: Optional[float] = None,
        gripper_min: Optional[float] = None,
        gripper_max: Optional[float] = None,
        max_grasps_per_object: Optional[int] = 3,
    ) -> Result:
        """POST /api/predict (multipart) -> :class:`Result`.

        Args:
            model: model name (must be loaded, or the server may auto-load it).
            image: one of —
                * ``str`` / ``os.PathLike``: path to an image file on disk.
                * ``bytes``: already-encoded image (PNG/JPEG bytes), sent verbatim.
                * ``PIL.Image.Image``: encoded to PNG client-side.
                * ``numpy.ndarray``: HWC ``uint8`` (or float in ``[0, 1]`` -> scaled to
                  uint8); grayscale ``(H, W)`` is promoted to RGB. Encoded to PNG.
            prompt: free-text open-vocab prompt, e.g. ``"cat. remote."``.
                    For ``grounding-dino``, ``grounded-sam``, and ``grasp-gd`` models,
                    defaults to ``"object"`` when not provided.
            box:    ``[x, y, w, h]`` or a list of such boxes (SAM box prompt).
            point:  ``[x, y]`` / ``[x, y, label]`` or a list of such points
                    (label 1=foreground, 0=background; defaults to 1).
            box_threshold: GroundingDINO query-score threshold (``grounding-dino`` /
                    ``grounded-sam`` / ``grasp-gd``). ``None`` = server manifest/default.
            text_threshold: GroundingDINO token→label threshold; lower values keep more
                    prompt words in each label (e.g. ``"canned coffee"`` instead of just
                    ``"coffee"``). ``None`` = server manifest/default (0.25).
            bg_max_area, fg_min_area: ``foreground`` model only — a MobileSAM automask
                    whose area is ``>= bg_max_area`` percent of the image is treated as
                    BACKGROUND (a support surface); one ``< fg_min_area`` percent is dropped
                    as noise. ``None`` = model default (50 / 0). Use these to tune the
                    foreground union — do NOT use ``min_size`` / ``max_size`` for it (those
                    are an output bbox-area filter that drops the full-image union mask).
            grid_size: ``foreground`` model only — MobileSAM automask grid ``N`` (``N×N``
                    point prompts → ``N²`` decoder calls). Default ``8`` (fast, ~1 s); raise
                    to ``16`` to catch more small objects (~4× slower). ``None`` = default.
            roi:    optional region of interest ``[x, y, w, h]``. The server crops to it, runs
                    the model on the crop ONLY, and maps results back to original coordinates
                    — generic to every model. Accepts PIXELS or NORMALIZED ``0..1`` fractions
                    (auto-detected when ``w`` and ``h`` are ≤ 1); use fractions to stay
                    independent of the image resolution. ``None`` = full image.
            dilate: morph every output mask by ``|dilate|`` pixels (square kernel) — ``>0``
                    enlarges, ``<0`` shrinks, ``None``/``0`` = off.
            min_size, max_size: object bbox-area filter as a percent of the image
                    area (e.g. ``0.1`` = 0.1%); ``None`` = no limit.
            gripper_min, gripper_max: grasp models only — parallel-jaw opening bounds
                    in ORIGINAL-image pixels; ``None`` = use the manifest default.
            max_grasps_per_object: client-side post-filter — keep at most this many
                    highest-quality grasps per detected object (``None`` = keep all).

        Returns:
            :class:`Result`.
        """
        # Default prompt for open-vocab models so the server does not reject promptless requests.
        effective_prompt = prompt
        if (effective_prompt is None or not str(effective_prompt).strip()) and _is_open_vocab_model(model):
            effective_prompt = "object"
        # Normalize phrase separators to '.' (GroundingDINO's separator) and ensure a trailing
        # '.'. Guard on a non-empty prompt: plain models (e.g. rf-detr) pass prompt=None.
        if effective_prompt is not None and str(effective_prompt).strip():
            effective_prompt = str(effective_prompt).replace(",", ".").replace("|", ".")
            if "." not in effective_prompt:
                effective_prompt = effective_prompt + "."

        image_bytes, filename = _encode_image(image)

        fields: Dict[str, str] = {"model": model}
        if effective_prompt is not None and str(effective_prompt).strip():
            fields["prompt"] = str(effective_prompt)
        box_str = _serialize_boxes(box)
        if box_str:
            fields["box"] = box_str
        roi_str = _serialize_boxes(roi)  # single [x,y,w,h] → "x,y,w,h"
        if roi_str:
            fields["roi"] = roi_str
        point_str = _serialize_points(point)
        if point_str:
            fields["point"] = point_str
        for key, val in (
            ("box_threshold", box_threshold),
            ("text_threshold", text_threshold),
            ("bg_max_area", bg_max_area),
            ("fg_min_area", fg_min_area),
            ("grid_size", grid_size),
            ("method", method),
            ("dilate", dilate),
            ("min_size", min_size),
            ("max_size", max_size),
            ("gripper_min", gripper_min),
            ("gripper_max", gripper_max),
        ):
            if val is not None:
                fields[key] = str(val)

        extra_files = None
        if depth is not None:
            depth_bytes, dh, dw, dtype = _encode_depth(depth)
            fields["depth_dtype"] = dtype
            fields["depth_height"] = str(dh)
            fields["depth_width"] = str(dw)
            extra_files = [("depth", depth_bytes, "depth.bin")]

        body, content_type = _build_multipart(fields, image_bytes, filename, extra_files)
        data = self._post_raw("/api/predict", body, content_type)
        result = Result.from_json(data)
        if max_grasps_per_object is not None:
            result = result.filter_grasps(max_grasps_per_object)
        return result

    # ------------------------------------------------------------------ #
    # Transport
    # ------------------------------------------------------------------ #
    def _get_json(self, path: str) -> Any:
        return self._request("GET", path)

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Any:
        body = json.dumps(payload).encode("utf-8")
        return self._request("POST", path, body=body, content_type="application/json")

    def _post_raw(self, path: str, body: bytes, content_type: str) -> Any:
        return self._request("POST", path, body=body, content_type=content_type)

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[bytes] = None,
        content_type: Optional[str] = None,
    ) -> Any:
        url = self.host + path
        headers = {"Accept": "application/json"}
        if content_type:
            headers["Content-Type"] = content_type
        req = urllib_request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib_request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib_error.HTTPError as e:
            raw = e.read()
            message = _extract_error(raw) or e.reason or "HTTP error"
            raise VisionServeError(
                "%s %s -> %s: %s" % (method, path, e.code, message), status=e.code
            )
        except urllib_error.URLError as e:
            raise VisionServeError(
                "failed to reach VisionServe at %s: %s" % (url, e.reason)
            )
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            raise VisionServeError("invalid JSON response from %s: %s" % (url, e))


# ---------------------------------------------------------------------- #
# Image encoding
# ---------------------------------------------------------------------- #
def _encode_image(image: ImageInput) -> "tuple[bytes, str]":
    """Encode any accepted image input into (bytes, filename).

    File paths and raw bytes are passed through unchanged; PIL/ndarray are PNG-encoded.
    """
    # Path-like / str path
    if isinstance(image, (str, os.PathLike)):
        p = Path(image)
        return p.read_bytes(), p.name

    # Raw already-encoded bytes
    if isinstance(image, (bytes, bytearray)):
        return bytes(image), "image.png"

    # PIL image
    pil_image = _maybe_pil(image)
    if pil_image is not None:
        return _pil_to_png(pil_image), "image.png"

    # numpy ndarray
    ndarray = _maybe_ndarray(image)
    if ndarray is not None:
        return _ndarray_to_png(ndarray), "image.png"

    raise TypeError(
        "unsupported image type %r; expected path/str, bytes, PIL.Image, or "
        "numpy.ndarray" % type(image)
    )


def _maybe_pil(image: Any):
    try:
        from PIL import Image
    except ImportError:
        return None
    if isinstance(image, Image.Image):
        return image
    return None


def _maybe_ndarray(image: Any):
    try:
        import numpy as np
    except ImportError:
        return None
    if isinstance(image, np.ndarray):
        return image
    return None


def _pil_to_png(img) -> bytes:
    buf = io.BytesIO()
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    img.save(buf, format="PNG")
    return buf.getvalue()


def _ndarray_to_png(arr) -> bytes:
    import numpy as np

    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Encoding a numpy.ndarray image requires pillow. Install with: "
            "pip install 'visionserve[images]' (or pass an encoded bytes/path instead)."
        ) from e

    a = arr
    if a.dtype.kind == "f":
        # float assumed in [0, 1] -> scale to uint8
        a = np.clip(a, 0.0, 1.0)
        a = (a * 255.0 + 0.5).astype(np.uint8)
    elif a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)

    if a.ndim == 2:
        # grayscale -> RGB
        a = np.stack([a, a, a], axis=-1)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif a.ndim == 3 and a.shape[2] in (3, 4):
        pass
    else:
        raise ValueError(
            "unsupported ndarray shape %r; expected (H,W), (H,W,1), (H,W,3) or (H,W,4)"
            % (a.shape,)
        )

    img = Image.fromarray(a)
    buf = io.BytesIO()
    # Encode RGB/grayscale frames as JPEG (q92): a numpy frame is almost always photographic, and
    # JPEG decodes ~4-5x faster server-side than PNG (zlib inflate). Keep PNG only for RGBA, where
    # alpha must survive losslessly.
    if a.ndim == 3 and a.shape[2] == 4:
        img.save(buf, format="PNG")
    else:
        img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


# ---------------------------------------------------------------------- #
# Prompt / box / point serialization (server string formats)
# ---------------------------------------------------------------------- #
def _is_scalar_seq(seq: Any) -> bool:
    """True if seq looks like a flat sequence of numbers, e.g. [x, y, w, h]."""
    return (
        isinstance(seq, (list, tuple))
        and len(seq) > 0
        and all(isinstance(v, (int, float)) for v in seq)
    )


def _normalize_list(values: Any) -> List[Sequence[float]]:
    """Normalize a single tuple or a list of tuples into a list of tuples."""
    if values is None:
        return []
    if _is_scalar_seq(values):
        return [values]  # a single box/point
    return list(values)  # already a list of boxes/points


def _serialize_boxes(box: BoxInput) -> str:
    """Serialize boxes to the server format: ``"x,y,w,h"`` joined by ``";"``."""
    boxes = _normalize_list(box)
    parts = []
    for b in boxes:
        if len(b) != 4:
            raise ValueError("box must have 4 values [x,y,w,h], got %r" % (b,))
        parts.append(",".join(_fmt_num(v) for v in b))
    return ";".join(parts)


def _serialize_points(point: PointInput) -> str:
    """Serialize points to the server format: ``"x,y[,label]"`` joined by ``";"``."""
    points = _normalize_list(point)
    parts = []
    for p in points:
        if len(p) not in (2, 3):
            raise ValueError("point must have 2 or 3 values [x,y[,label]], got %r" % (p,))
        parts.append(",".join(_fmt_num(v) for v in p))
    return ";".join(parts)


def _fmt_num(v: float) -> str:
    """Format a number without a trailing ``.0`` for integers (server parses floats)."""
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return repr(v)


# ---------------------------------------------------------------------- #
# Multipart encoding (stdlib, no `requests` dependency)
# ---------------------------------------------------------------------- #
def _encode_depth(depth: "Any") -> "tuple[bytes, int, int, str]":
    """Encode a 2-D depth ndarray into (raw little-endian bytes, H, W, dtype). Integer arrays
    are sent as ``uint16`` (server normalizes /65535), float arrays as ``float32`` (as-is)."""
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "depth= requires numpy. Install with: pip install 'visionserve[images]'"
        ) from e
    arr = np.asarray(depth)
    if arr.ndim != 2:
        raise ValueError("depth must be a 2-D (H, W) array, got shape %r" % (arr.shape,))
    if np.issubdtype(arr.dtype, np.integer):
        arr = np.ascontiguousarray(arr.astype("<u2"))
        dtype = "uint16"
    else:
        arr = np.ascontiguousarray(arr.astype("<f4"))
        dtype = "float32"
    h, w = int(arr.shape[0]), int(arr.shape[1])
    return arr.tobytes(), h, w, dtype


def _build_multipart(
    fields: Dict[str, str], image_bytes: bytes, filename: str, extra_files=None
) -> "tuple[bytes, str]":
    """Build a ``multipart/form-data`` body with text fields + the image file (+ optional
    extra binary files, each a ``(field_name, bytes, filename)`` tuple)."""
    boundary = "----visionserve-" + uuid.uuid4().hex
    crlf = b"\r\n"
    out = io.BytesIO()

    for name, value in fields.items():
        out.write(b"--" + boundary.encode() + crlf)
        out.write(
            ('Content-Disposition: form-data; name="%s"' % name).encode("utf-8") + crlf
        )
        out.write(crlf)
        out.write(str(value).encode("utf-8") + crlf)

    def _write_file(field: str, data: bytes, fname: str) -> None:
        out.write(b"--" + boundary.encode() + crlf)
        out.write(
            ('Content-Disposition: form-data; name="%s"; filename="%s"' % (field, fname)).encode("utf-8")
            + crlf
        )
        out.write(b"Content-Type: application/octet-stream" + crlf)
        out.write(crlf)
        out.write(data + crlf)

    _write_file("image", image_bytes, filename)
    for field, data, fname in (extra_files or []):
        _write_file(field, data, fname)

    out.write(b"--" + boundary.encode() + b"--" + crlf)

    content_type = "multipart/form-data; boundary=%s" % boundary
    return out.getvalue(), content_type


_OPEN_VOCAB_MODELS = ("grounding-dino", "grounded-sam", "grasp-gd")


def _is_open_vocab_model(model: str) -> bool:
    """True when the model requires a text prompt and should default to 'object'."""
    return any(k in model for k in _OPEN_VOCAB_MODELS)


def _extract_error(raw: bytes) -> Optional[str]:
    """Pull the ``error`` field from a server ErrorResponse JSON body, if present."""
    try:
        d = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return raw.decode("utf-8", errors="replace") if raw else None
    if isinstance(d, dict) and "error" in d:
        return str(d["error"])
    return None
