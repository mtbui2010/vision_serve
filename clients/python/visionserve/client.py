"""HTTP client for the VisionServe server.

Transport uses only the Python standard library (``urllib``), so the client has no
hard third-party dependencies. ``numpy`` and ``pillow`` are optional and only needed
for ndarray / PIL image inputs and for :meth:`Mask.to_ndarray`.
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import numbers
import os
import re
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib import error as urllib_error
from urllib import request as urllib_request

from . import resize as _resize
from .resize import ClientResize, ResizeOption
from .types import Detection, Mask, ModelInfo, Result, _is_loaded

# Type alias for accepted image inputs (documented in predict()).
ImageInput = Union[str, "os.PathLike[str]", bytes, "Any"]  # Any covers PIL / ndarray

BoxInput = Union[Sequence[float], Sequence[Sequence[float]], None]
PointInput = Union[Sequence[float], Sequence[Sequence[float]], None]


class VisionServeError(Exception):
    """Raised when the server returns a non-2xx response or transport fails.

    ``status`` is the HTTP status code, or ``None`` when no answer came back (server unreachable,
    timeout, dropped connection). ``retry_after`` is the server's ``Retry-After`` in seconds (sent
    with a 503 when the model's queue is full), else ``None``.
    """

    def __init__(self, message: str, status: Optional[int] = None, retry_after: Optional[float] = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class Client:
    """Client for the VisionServe HTTP API.

    Args:
        host:    base URL of the server, e.g. ``http://localhost:11435``.
        timeout: per-request timeout in seconds.
        base64_arrays: ask the server for depth maps and embeddings as base64 float32
            (``encoding=base64``) instead of JSON number arrays: exact float32 values, about
            half the bytes, and several times cheaper to encode and parse for a large depth map.
            Opt-in (default ``False``): with it, ``depth_map`` / ``embeddings`` become read-only,
            list-like :class:`~visionserve.FloatArray` objects backed by numpy (zero-copy
            ``numpy.asarray``), which are not lists — ``json.dumps``, ``+`` and ``append`` need
            ``.tolist()`` first. Requires numpy (without it the base64 is decoded into plain
            lists). A server that predates the option ignores it and sends numbers.
        resize: client-side resizing of :meth:`predict` uploads (ON by default). ``"auto"``: shrink
            an image larger than the model can use to the server's hint for that model
            (``GET /api/models``, fetched once and cached): a 12 MP photo becomes ~1.7 MP for
            RF-DETR, while masks, OCR, grasping and template models (no hint) always get the full
            image. When ``host`` is this machine (``localhost``, 127.0.0.0/8, ``::1``) a photo is
            shrunk only when that at least halves its sides (the loopback rule; see
            :attr:`Result.client_resize`). ``"off"``: send every image exactly as given. An int ``N``: shrink to a longer
            side of ``N`` pixels for every model, whatever its hint. Results are always mapped
            back to ORIGINAL image pixels; :attr:`Result.client_resize` says what was done.
        jpeg: send a shrunk image as JPEG (default ``True``) instead of lossless PNG. An image that
            is not shrunk is always sent exactly as given (a path / bytes verbatim, PIL / numpy as
            lossless PNG).
        jpeg_quality: JPEG quality 1..100 (default 90).

    Client-side resizing needs Pillow (``pip install 'visionserve[images]'``); without it every
    image is sent as given. See ``docs/clients/python.md``, "Client-side resizing".
    """

    def __init__(
        self,
        host: str = "http://localhost:11435",
        timeout: float = 120,
        *,
        base64_arrays: bool = False,
        resize: ResizeOption = "auto",
        jpeg: bool = True,
        jpeg_quality: int = 90,
    ):
        self.host = host.rstrip("/")
        self.timeout = timeout
        # Opt-in: the default keeps Result.depth_map / embeddings plain lists, as they always were.
        self.base64_arrays = bool(base64_arrays)
        # ON by default — a user decision (2026-10-05) that overrides the "output-changing
        # behaviour is opt-in" rule for this feature; the measured cost is in the docs.
        self.resize = _resize.check_resize(resize)
        self.jpeg = bool(jpeg)
        self.jpeg_quality = _resize.check_quality(jpeg_quality)
        self._hints: Dict[str, "tuple[Optional[int], Optional[int]]"] = {}
        self._hints_fetched = float("-inf")  # monotonic time of the last /api/models fetch
        self._hints_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def health(self) -> Dict[str, str]:
        """GET /api/health -> ``{"status": "ok"}``."""
        return self._get_json("/api/health")

    def list_models(self) -> List[ModelInfo]:
        """GET /api/models -> list of :class:`ModelInfo`."""
        data = self._get_json("/api/models")
        infos = [ModelInfo.from_json(x) for x in (data or []) if isinstance(x, dict)]
        with self._hints_lock:
            self._hints = {m.name: (m.max_useful_side, m.max_useful_short_side) for m in infos}
            self._hints_fetched = time.monotonic()
        return infos

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
        claim_threshold: Optional[float] = None,
        crop_temp: Optional[float] = None,
        template_name: Optional[str] = None,
        resize: Optional[ResizeOption] = None,
        jpeg: Optional[bool] = None,
        jpeg_quality: Optional[int] = None,
    ) -> Result:
        """POST /api/predict (multipart) -> :class:`Result`.

        Args:
            model: model name (must be loaded, or the server may auto-load it).
            image: one of —
                * ``str`` / ``os.PathLike``: path to an image file on disk.
                * ``bytes``: already-encoded image (PNG/JPEG bytes).
                * ``PIL.Image.Image``: encoded client-side.
                * ``numpy.ndarray``: HWC ``uint8`` (or float in ``[0, 1]`` -> scaled to
                  uint8); grayscale ``(H, W)`` is promoted to RGB. Encoded client-side.
                  How it is sent depends on the client's ``resize`` / ``jpeg`` (see
                  :class:`Client`): by default, an image larger than the model can use is shrunk
                  and sent as JPEG; anything not shrunk goes out as before: paths and bytes
                  verbatim, PIL / ndarray images as lossless PNG.
            prompt: free-text open-vocab prompt, e.g. ``"cat. remote."``.
                    For ``grounding-dino``, ``grounded-sam``, and ``grasp-gd`` models,
                    defaults to ``"object"`` when not provided.
            box:    ``[x, y, w, h]`` or a list of such boxes (SAM box prompt).
            point:  ``[x, y]`` / ``[x, y, label]`` or a list of such points
                    (label 1=foreground, 0=background; defaults to 1).
            box_threshold: minimum box score for GroundingDINO (``grounding-dino`` / ``grounded-sam``
                    / ``grasp-gd``) and OWLv2 templates. ``None`` = server manifest/default.
            text_threshold: GroundingDINO second score floor: a box is kept only when its best
                    phrase scores above both ``box_threshold`` and ``text_threshold``. It does
                    not change labels (a label is always the whole prompt phrase). ``None`` =
                    server manifest/default (0.25).
            bg_max_area, fg_min_area: ``background`` model, ``method="sam"`` / ``"automask"``
                    only — a MobileSAM mask whose area is ``>= bg_max_area`` percent of the
                    image counts as BACKGROUND (a support surface); one ``< fg_min_area``
                    percent is dropped as noise. ``None`` = model default (50 / 0). Do NOT use
                    ``min_size`` / ``max_size`` for this (those are an output bbox-area filter
                    that can drop the surface mask).
            grid_size: MobileSAM automask grid ``N`` (``N×N`` point prompts → ``N²`` decoder
                    calls): ``mobile-sam`` with no prompt and ``grasp`` with no box (default 16),
                    ``background`` with ``method="automask"`` (default 8). Larger finds more
                    small objects but is slower; the server caps it at 64. ``None`` = default.
            roi:    optional region of interest ``[x, y, w, h]``. The server crops to it, runs
                    the model on the crop ONLY, and maps results back to original coordinates
                    — generic to every model. Accepts PIXELS or NORMALIZED ``0..1`` fractions
                    (auto-detected when ``w`` and ``h`` are ≤ 1); use fractions to stay
                    independent of the image resolution. ``None`` = full image.
            dilate: morph every output mask by ``|dilate|`` pixels (square kernel) — ``>0``
                    enlarges, ``<0`` shrinks, ``None``/``0`` = off.
            min_size, max_size: object bbox-area filter as a percent of the image
                    area (e.g. ``0.1`` = 0.1%), applied to detections and masks after the
                    model (grasp models also apply it to objects before planning grasps);
                    ``None`` = no limit.
            gripper_min, gripper_max: grasp models only — parallel-jaw opening bounds
                    in ORIGINAL-image pixels; ``None`` = use the manifest default.
            max_grasps_per_object: client-side post-filter — keep at most this many
                    highest-quality grasps per detected object (``None`` = keep all).
            claim_threshold: ``rfdetr-textalign`` (``method="dual"``) — probability in (0, 1) the
                    closed head must reach on a requested word before it names a detection;
                    ``>= 1`` = never claims. ``None`` = model default.
            crop_temp: softmax temperature of the crop namer (textalign / hybrid); lower is more
                    decisive. ``None`` = model default.
            template_name: ``instance_detection`` models — a template set registered via
                    ``POST /api/templates``.
            resize, jpeg, jpeg_quality: override the client's client-side resizing for this call
                    (see :class:`Client`); ``None`` = the client's setting. Boxes, points and a
                    pixel ``roi`` are scaled with the image, and every returned coordinate is
                    mapped back to ORIGINAL pixels. A request with ``depth``, ``dilate`` or
                    ``gripper_min`` / ``gripper_max`` (pixel quantities tied to the full image) or
                    ``template_name`` is always sent at full resolution.

        Boxes, points and numeric options may be Python numbers or numpy arrays / scalars.

        Returns:
            :class:`Result`.
        """
        effective_prompt = normalize_prompt(model, prompt)
        full_res = depth is not None or bool(dilate) or gripper_min is not None or gripper_max is not None \
            or template_name is not None
        image_bytes, filename, cr = self._prepare(
            model, image, resize, jpeg, jpeg_quality, roi=None if full_res else roi, full_res=full_res)
        if cr is not None and cr.resized:
            box = _resize.scale_boxes(_normalize_list(box), cr) if box is not None else None
            point = _resize.scale_points(_normalize_list(point), cr) if point is not None else None
            if roi is not None:
                roi = _resize.scale_roi(_single_box(roi), cr)

        fields: Dict[str, str] = {"model": model}
        if effective_prompt is not None:
            fields["prompt"] = effective_prompt
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
            ("min_size", min_size),
            ("max_size", max_size),
            ("gripper_min", gripper_min),
            ("gripper_max", gripper_max),
            ("claim_threshold", claim_threshold),
            ("crop_temp", crop_temp),
        ):
            if val is not None:
                fields[key] = _fmt_num(_plain(val))
        # The server parses these with Atoi: "2.0" would silently become 0 (= default / off).
        for key, val in (("grid_size", grid_size), ("dilate", dilate)):
            if val is not None:
                fields[key] = str(_as_int(key, val))
        for key, val in (("method", method), ("template_name", template_name)):
            if val is not None:
                fields[key] = str(val)
        if self.base64_arrays:
            fields["encoding"] = "base64"  # Result.from_json decodes it

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
        if cr is not None:
            _resize.map_result(result, cr)
        if max_grasps_per_object is not None:
            result = result.filter_grasps(max_grasps_per_object)
        return result

    def preprocess(
        self,
        model: str,
        image: Optional[ImageInput] = None,
        *,
        prompt: Optional[str] = None,
        box: BoxInput = None,
        point: PointInput = None,
        resize: ResizeOption = "off",
        jpeg: Optional[bool] = None,
        jpeg_quality: Optional[int] = None,
    ) -> "PreprocessResult":
        """POST /api/preprocess -> exactly what ``model`` would feed its first ONNX session.

        Runs NO inference. Use it to check that the server prepares inputs the way your model
        was trained: compare ``res.inputs[name]`` with the tensor your own pipeline builds for
        the same image (resize / normalisation / letterbox) or text (token ids, padding). A
        mismatch there serves a working model silently worse.

        Unlike :meth:`predict`, the image is sent as given by default (``resize="off"``): the
        point is comparing the server's tensor with yours for the SAME pixels. Pass
        ``resize="auto"`` (and ``jpeg`` / ``jpeg_quality``) to see what :meth:`predict` feeds
        the model with the client's resizing; ``res.client_resize`` then says what was done, and
        ``res.meta`` maps the tensor to the SENT image.

        Needs numpy. ``image`` is optional for text-only models (``siglip-text``, ``clip-text``).
        """
        fields: Dict[str, str] = {"model": model}
        effective_prompt = normalize_prompt(model, prompt)  # exactly what predict() would send
        if effective_prompt is not None:
            fields["prompt"] = effective_prompt
        cr = None
        if image is not None:
            image_bytes, filename, cr = self._prepare(model, image, resize, jpeg, jpeg_quality)
        else:
            image_bytes, filename = None, ""
        if cr is not None and cr.resized:
            box = _resize.scale_boxes(_normalize_list(box), cr) if box is not None else None
            point = _resize.scale_points(_normalize_list(point), cr) if point is not None else None
        box_str = _serialize_boxes(box)
        if box_str:
            fields["box"] = box_str
        point_str = _serialize_points(point)
        if point_str:
            fields["point"] = point_str
        body, content_type = _build_multipart(fields, image_bytes, filename)
        res = PreprocessResult.from_json(self._post_raw("/api/preprocess", body, content_type))
        res.client_resize = cr
        return res

    def tokenize(self, model: str, text: str) -> "Any":
        """Token ids the server feeds ``model`` for ``text`` (padded exactly as served).

        Shorthand for ``preprocess(model, prompt=text)`` returning the ``input_ids`` array, e.g. to
        compare with ``transformers.AutoTokenizer(...)(text, padding="max_length")``.
        """
        res = self.preprocess(model, prompt=text)
        if "input_ids" not in res.inputs:
            raise VisionServeError(f"{model} takes no token ids (inputs: {sorted(res.inputs)})")
        return res.inputs["input_ids"]

    # ------------------------------------------------------------------ #
    # Client-side resizing
    # ------------------------------------------------------------------ #
    def useful_side(self, model: str) -> "tuple[Optional[int], Optional[int]]":
        """The server's client-resize hint for ``model``: ``(max_useful_side,
        max_useful_short_side)`` (see :class:`ModelInfo`), ``(None, None)`` for no hint.

        Fetched with ONE ``GET /api/models`` and cached for the client's lifetime; a model the
        cache does not know triggers a refresh (at most every few seconds). If the listing fails
        (an unreachable or old server), there is no hint: the image is sent at full resolution
        and predict() reports the server's own error, if any.
        """
        with self._hints_lock:
            if model in self._hints:
                return self._hints[model]
            stale = time.monotonic() - self._hints_fetched > _HINT_REFRESH_S
        if stale:
            try:
                self.list_models()
            except (VisionServeError, ValueError, TypeError):
                with self._hints_lock:
                    self._hints_fetched = time.monotonic()
        with self._hints_lock:
            return self._hints.get(model, (None, None))

    def _prepare(
        self,
        model: str,
        image: ImageInput,
        resize: Optional[ResizeOption],
        jpeg: Optional[bool],
        jpeg_quality: Optional[int],
        *,
        roi: Any = None,
        full_res: bool = False,
    ) -> "tuple[bytes, str, Optional[ClientResize]]":
        """The upload for ``image``: (bytes, filename, what was done or None)."""
        # Validated here too: the attributes may be reassigned after construction.
        mode = _resize.check_resize(self.resize if resize is None else resize)
        use_jpeg = bool(self.jpeg if jpeg is None else jpeg)
        quality = _resize.check_quality(self.jpeg_quality if jpeg_quality is None else jpeg_quality)
        max_side = max_short = max_scale = None
        reason = "hint"
        if not full_res and mode != _resize.RESIZE_OFF:
            if mode == _resize.RESIZE_AUTO:
                max_side, max_short = self.useful_side(model)
                if _is_loopback(self.host):
                    max_scale = _LOOPBACK_MAX_SCALE
            else:
                max_side, reason = int(mode), "resize=%d" % int(mode)
        roi_box = _single_box(roi) if roi is not None else None
        return _resize.prepare_upload(
            image, max_side=max_side, max_short_side=max_short, jpeg=use_jpeg, quality=quality,
            roi=roi_box, encode_plain=_encode_image, ndarray_to_pil=_ndarray_to_pil,
            reason=reason, max_scale=max_scale,
        )

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
            try:
                raw = e.read()
            except (OSError, http.client.HTTPException):
                raw = b""
            message = _extract_error(raw) or e.reason or "HTTP error"
            raise VisionServeError(
                "%s %s -> %s: %s" % (method, path, e.code, message), status=e.code,
                retry_after=_retry_after(e.headers),
            )
        except urllib_error.URLError as e:
            raise VisionServeError(
                "failed to reach VisionServe at %s: %s" % (url, e.reason)
            )
        except (socket.timeout, TimeoutError) as e:
            raise VisionServeError(
                "%s %s: no answer from VisionServe at %s within %ss (%s)"
                % (method, path, self.host, self.timeout, e or "timed out")
            )
        except (http.client.HTTPException, ConnectionError, OSError) as e:
            # RemoteDisconnected, IncompleteRead, BadStatusLine, ConnectionResetError, ...
            raise VisionServeError(
                "%s %s: connection to VisionServe at %s failed: %s: %s"
                % (method, path, self.host, type(e).__name__, e)
            )
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            raise VisionServeError("invalid JSON response from %s: %s" % (url, e))


# How long an unknown model name waits before it triggers another GET /api/models (seconds).
_HINT_REFRESH_S = 5.0

# The loopback rule of resize="auto": with the server on this machine the upload costs nothing,
# so a photo is shrunk only when that at least halves its sides; a milder shrink (GroundingDINO's
# 1600 px hint on a 4000 x 3000 photo: scale 0.53) measured slower than sending it whole, because
# the client must decode the full photo. Remote servers keep shrinking to the hint.
_LOOPBACK_MAX_SCALE = 0.5


def _is_loopback(host: str) -> bool:
    """True when ``host`` (the client's base URL) names this machine: ``localhost``, 127.0.0.0/8
    or ``::1``. Only literal addresses and the name ``localhost`` count (no DNS lookup)."""
    import ipaddress
    from urllib.parse import urlsplit

    try:
        name = urlsplit(host if "://" in host else "http://" + host).hostname or ""
    except ValueError:
        return False
    name = name.lower().rstrip(".")
    if name == "localhost" or name.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def _single_box(roi: Any) -> List[float]:
    """A ``roi`` as one ``[x, y, w, h]`` list of floats (numpy accepted)."""
    boxes = _normalize_list(roi)
    if len(boxes) != 1 or len(boxes[0]) != 4:
        raise ValueError("roi must be one box [x,y,w,h], got %r" % (roi,))
    return [float(v) for v in boxes[0]]


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
        return p.read_bytes(), _safe_filename(p.name)

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


def _ndarray_to_pil(arr):
    """A numpy image as a PIL image — the ONE ndarray rule, shared by :meth:`Client.predict`
    uploads and :func:`visionserve.visualize.draw`.

    HWC ``uint8`` is used as is; float is taken as ``[0, 1]`` and scaled to ``uint8``; any other
    dtype is clipped to ``[0, 255]``. Grayscale ``(H, W)`` / ``(H, W, 1)`` becomes RGB;
    ``(H, W, 3)`` / ``(H, W, 4)`` keep their channels.
    """
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

    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    if a.ndim == 2:  # grayscale -> RGB
        a = np.stack([a, a, a], axis=-1)
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError(
            "unsupported ndarray shape %r; expected (H,W), (H,W,1), (H,W,3) or (H,W,4)"
            % (arr.shape,)
        )
    return Image.fromarray(np.ascontiguousarray(a))


def _ndarray_to_png(arr) -> bytes:
    img = _ndarray_to_pil(arr)
    buf = io.BytesIO()
    # PNG (lossless): the server must see exactly these pixels, or preprocess() comparisons and
    # predict() on a frame differ from the array the caller holds. compress_level=1 keeps the
    # encode cheap; pass already-encoded JPEG bytes if bandwidth matters more than exactness.
    img.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


# ---------------------------------------------------------------------- #
# Prompt / box / point serialization (server string formats)
# ---------------------------------------------------------------------- #
def _plain(v: Any) -> Any:
    """numpy arrays / scalars -> Python lists / numbers (recursively); anything else unchanged."""
    if isinstance(v, (str, bytes)):
        return v
    if hasattr(v, "tolist") and callable(v.tolist):  # numpy ndarray or numpy scalar
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def _is_scalar_seq(seq: Any) -> bool:
    """True if seq looks like a flat sequence of numbers, e.g. [x, y, w, h]."""
    return (
        isinstance(seq, (list, tuple))
        and len(seq) > 0
        and all(isinstance(v, numbers.Real) for v in seq)
    )


def _normalize_list(values: Any) -> List[Sequence[float]]:
    """Normalize a single tuple or a list of tuples into a list of tuples. numpy input (an
    ``(4,)`` / ``(N, 4)`` array, numpy scalars) is converted to Python numbers first."""
    if values is None:
        return []
    values = _plain(values)
    if _is_scalar_seq(values):
        return [values]  # a single box/point
    return [list(v) if isinstance(v, (list, tuple)) else v for v in values]  # list of boxes/points


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


def _fmt_num(v: Any) -> str:
    """Format a number without a trailing ``.0`` for integers (server parses floats). numpy
    scalars are converted first (``repr(np.float64(1.5))`` is ``"np.float64(1.5)"`` on numpy 2)."""
    v = _plain(v)
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, numbers.Integral):
        return str(int(v))
    if not isinstance(v, numbers.Real):
        raise TypeError("expected a number, got %r" % (v,))
    f = float(v)
    if f.is_integer():
        return str(int(f))
    return repr(f)


def _as_int(name: str, v: Any) -> int:
    """An integer form field (the server uses Atoi). Integral floats are accepted (2.0 -> 2)."""
    v = _plain(v)
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise TypeError("%s must be an integer, got %r" % (name, v))
    f = float(v)
    if not f.is_integer():
        raise ValueError("%s must be an integer, got %r" % (name, v))
    return int(f)


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
    if np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.bool_:
        if arr.size and (int(arr.min()) < 0 or int(arr.max()) > 65535):
            raise ValueError(
                "integer depth is sent as uint16 but ranges over [%d, %d]; values outside "
                "[0, 65535] would wrap around — rescale it, or pass a float32 array"
                % (int(arr.min()), int(arr.max()))
            )
        arr = np.ascontiguousarray(arr.astype("<u2"))
        dtype = "uint16"
    else:
        arr = np.ascontiguousarray(arr.astype("<f4"))
        dtype = "float32"
    h, w = int(arr.shape[0]), int(arr.shape[1])
    return arr.tobytes(), h, w, dtype


def _build_multipart(
    fields: Dict[str, str], image_bytes: Optional[bytes], filename: str, extra_files=None
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
            ('Content-Disposition: form-data; name="%s"; filename="%s"'
             % (field, _safe_filename(fname))).encode("utf-8")
            + crlf
        )
        out.write(b"Content-Type: application/octet-stream" + crlf)
        out.write(crlf)
        out.write(data + crlf)

    if image_bytes is not None:  # optional for text-only requests (/api/preprocess on a text tower)
        _write_file("image", image_bytes, filename)
    for field, data, fname in (extra_files or []):
        _write_file(field, data, fname)

    out.write(b"--" + boundary.encode() + b"--" + crlf)

    content_type = "multipart/form-data; boundary=%s" % boundary
    return out.getvalue(), content_type


def _safe_filename(name: str) -> str:
    """A multipart ``filename="..."`` value: quotes, backslashes and line breaks would end the
    header (or inject another); the server ignores the name anyway."""
    s = re.sub(r'["\\\r\n]', "_", str(name or "image"))
    return s or "image"


# Models that REQUIRE a text prompt (the server rejects an empty one): GroundingDINO and the
# pipelines built on it. "gdino-siglip" (and -sam) runs GroundingDINO first; rfdetr-gdino-* does
# NOT need a prompt (no prompt = everything RF-DETR knows), so it is excluded.
_OPEN_VOCAB_MODELS = ("grounding-dino", "grounded-sam", "grasp-gd")
_OPEN_VOCAB_RE = re.compile(r"(?<!rfdetr-)gdino-siglip")
# CLIP / SigLIP text and image towers: each "."-separated phrase is one label, and a comma is
# part of the label ("a photo of a cat, sitting"), so commas must not become phrase separators.
_EMBED_RE = re.compile(r"clip|siglip")
_DETECTOR_RE = re.compile(r"gdino|grounding|grounded|textalign|grasp")


def _is_open_vocab_model(model: str) -> bool:
    """True when the model requires a text prompt and should default to 'object'."""
    m = str(model).lower()
    return any(k in m for k in _OPEN_VOCAB_MODELS) or bool(_OPEN_VOCAB_RE.search(m))


def _is_embedding_model(model: str) -> bool:
    m = str(model).lower()
    return bool(_EMBED_RE.search(m)) and not _DETECTOR_RE.search(m)


def normalize_prompt(model: str, prompt: Optional[str]) -> Optional[str]:
    """The prompt string :meth:`Client.predict`, :meth:`Client.preprocess` and
    :meth:`Client.tokenize` send for ``model`` (one rule, so a preprocess() check sees what
    predict() feeds). ``None`` when no prompt field is sent.

    * empty / ``None``: ``"object."`` for models that require a prompt (GroundingDINO family),
      else ``None``;
    * CLIP / SigLIP towers: sent verbatim (the server splits phrases on ``"."`` only);
    * everything else: ``","`` and ``"|"`` become the ``"."`` phrase separator and a single
      phrase gets a trailing ``"."`` (``"cat, remote"`` -> ``"cat. remote"``, ``"cat"`` -> ``"cat."``).
    """
    if prompt is None or not str(prompt).strip():
        return "object." if _is_open_vocab_model(model) else None
    text = str(prompt)
    if _is_embedding_model(model):
        return text
    text = text.replace(",", ".").replace("|", ".")
    if "." not in text:
        text += "."
    return text


def _retry_after(headers: Any) -> Optional[float]:
    """The ``Retry-After`` header as seconds (the server sends an integer), or ``None`` when it is
    absent or not a non-negative number (the HTTP-date form is not used by VisionServe)."""
    value = headers.get("Retry-After") if headers is not None else None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if 0 <= seconds < float("inf") else None


def _extract_error(raw: bytes) -> Optional[str]:
    """Pull the ``error`` field from a server ErrorResponse JSON body, if present."""
    try:
        d = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return raw.decode("utf-8", errors="replace") if raw else None
    if isinstance(d, dict) and "error" in d:
        return str(d["error"])
    return None


class PreprocessResult:
    """Response of :meth:`Client.preprocess`.

    Attributes:
        inputs: ``{onnx_input_name: numpy.ndarray}`` — bit-exact copies of the served tensors.
        roles:  ``{onnx_input_name: session_role}`` (``"model"`` for single-session models).
        meta:   ``{"orig_width", "orig_height", "scale_x", "scale_y", "pad_x", "pad_y"}`` mapping
                model-input pixels back to the original image (``input = orig * scale + pad``),
                or ``None`` when the model's preprocessing has no such mapping. "Original" is the
                image the server received: the SENT one when ``client_resize`` is set.
        client_resize: what the client did to the image before sending it, or ``None`` (the
                default for preprocess(): the image is sent as given).
    """

    def __init__(self, model: str, inputs: Dict[str, Any], roles: Dict[str, str], meta: Optional[Dict[str, Any]]):
        self.model, self.inputs, self.roles, self.meta = model, inputs, roles, meta
        self.client_resize: Optional[ClientResize] = None  # set by Client.preprocess

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "PreprocessResult":
        try:
            import numpy as np
        except ImportError as e:  # pragma: no cover - numpy is in the [images] extra
            raise VisionServeError("preprocess() needs numpy: pip install visionserve[images]") from e
        inputs, roles = {}, {}
        for it in data.get("inputs") or []:
            dtype = {"float32": np.float32, "int64": np.int64}[it["dtype"]]
            arr = np.frombuffer(base64.b64decode(it["data"]), dtype=dtype).reshape(it["shape"])
            inputs[it["name"]] = arr
            roles[it["name"]] = it.get("role", "model")
        return cls(data.get("model", ""), inputs, roles, data.get("meta"))

    def __repr__(self) -> str:
        shapes = {k: tuple(v.shape) for k, v in self.inputs.items()}
        return f"PreprocessResult(model={self.model!r}, inputs={shapes}, meta={self.meta})"
