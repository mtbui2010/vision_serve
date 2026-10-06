"""Orbbec adapter (``pip install 'visionserve[orbbec]'``: ``pyorbbecsdk2``, Apache-2.0, the official
Python binding of Orbbec SDK v2; imported as ``pyorbbecsdk``).

Colour + depth (Gemini, Femto, Astra series), depth aligned to colour by the SDK's
``AlignFilter`` (software depth-to-colour), ``depth_scale`` from each depth frame (the SDK reports
millimetres per unit; the frame carries metres per unit).
"""

from __future__ import annotations

import io
import time
from typing import Any, Optional

from .frame import BaseSource, Frame, _np, require

_INSTALL = "pip install 'visionserve[orbbec]'  (pyorbbecsdk2, Orbbec SDK v2)"


def yuyv_to_rgb(buf: Any, width: int, height: int) -> Any:
    """Packed YUYV (YUY2) 4:2:2 bytes as an ``(H, W, 3)`` uint8 RGB array (BT.601, limited range)."""
    np = _np()
    a = np.frombuffer(bytes(buf) if not hasattr(buf, "dtype") else buf.tobytes(), dtype=np.uint8)
    a = a[: width * height * 2].reshape(height, width // 2, 4).astype(np.float32)
    y = np.empty((height, width), np.float32)
    y[:, 0::2], y[:, 1::2] = a[:, :, 0], a[:, :, 2]
    u = np.repeat(a[:, :, 1], 2, axis=1) - 128.0
    v = np.repeat(a[:, :, 3], 2, axis=1) - 128.0
    c = (y - 16.0) * 1.164
    rgb = np.stack([c + 1.596 * v, c - 0.392 * u - 0.813 * v, c + 2.017 * u], axis=-1)
    return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)


def color_to_rgb(fmt_name: str, data: Any, width: int, height: int) -> Any:
    """An Orbbec colour frame's bytes as ``(H, W, 3)`` uint8 RGB, by its format name
    (``RGB``, ``BGR``, ``MJPG``, ``YUYV``)."""
    np = _np()
    fmt = fmt_name.upper().rsplit(".", 1)[-1]
    if fmt == "RGB":
        return np.frombuffer(_bytes(data), np.uint8)[: width * height * 3].reshape(height, width, 3).copy()
    if fmt == "BGR":
        bgr = np.frombuffer(_bytes(data), np.uint8)[: width * height * 3].reshape(height, width, 3)
        return np.ascontiguousarray(bgr[:, :, ::-1])
    if fmt in ("MJPG", "MJPEG"):
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError("decoding the camera's MJPG frames needs pillow: pip install 'visionserve[images]'") from e
        return np.asarray(Image.open(io.BytesIO(_bytes(data))).convert("RGB"))
    if fmt in ("YUYV", "YUY2"):
        return yuyv_to_rgb(data, width, height)
    raise ValueError("Orbbec colour format %s is not supported: pick an RGB, BGR, MJPG or YUYV profile" % fmt_name)


def _bytes(data: Any) -> bytes:
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    np = _np()
    return np.asarray(data).tobytes()


class OrbbecSource(BaseSource):
    """Frames from an Orbbec RGB-D camera (Orbbec SDK v2).

    Args:
        index: which connected camera (default 0, the first).
        width, height, fps: the colour mode to ask for in RGB (``None`` = the camera's
            default colour profile, converted to RGB from BGR / MJPG / YUYV).
        depth: also stream depth, aligned to colour (default ``True``).

    ``timestamp`` is the colour frame's device timestamp in seconds; ``intrinsics`` are the
    colour camera's when the SDK reports them.
    """

    def __init__(
        self,
        index: int = 0,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        depth: bool = True,
    ):
        super().__init__()
        ob = require("pyorbbecsdk", _INSTALL)
        self._ob = ob
        self.index = int(index)
        self.has_depth = bool(depth)
        devices = ob.Context().query_devices()
        n = devices.get_count()
        if self.index >= n:
            raise OSError("Orbbec: camera %d requested, %d connected" % (self.index, n))
        self._pipe = ob.Pipeline(devices.get_device_by_index(self.index))
        cfg = ob.Config()
        color_profiles = self._pipe.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        color_profile = None
        if width or height or fps:
            try:
                color_profile = color_profiles.get_video_stream_profile(int(width or 0), int(height or 0), ob.OBFormat.RGB, int(fps or 0))
            except Exception as e:  # noqa: BLE001 - the SDK raises its own OBError
                raise OSError("Orbbec: no RGB colour profile %sx%s@%s: %s" % (width, height, fps, e)) from e
        else:
            color_profile = color_profiles.get_default_video_stream_profile()
        cfg.enable_stream(color_profile)
        if self.has_depth:
            depth_profiles = self._pipe.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
            cfg.enable_stream(depth_profiles.get_default_video_stream_profile())
            if hasattr(cfg, "set_frame_aggregate_output_mode"):
                cfg.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        try:
            self._pipe.start(cfg)
        except Exception as e:  # noqa: BLE001
            raise OSError("Orbbec: could not start camera %d: %s" % (self.index, e)) from e
        self._started = True
        self._align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM) if self.has_depth else None
        self.intrinsics = self._intrinsics(color_profile)

    def _intrinsics(self, profile: Any) -> Optional[dict]:
        i = None
        try:
            i = profile.get_intrinsic()
        except Exception:  # noqa: BLE001 - older SDKs: camera param instead
            try:
                i = self._pipe.get_camera_param().rgb_intrinsic
            except Exception:  # noqa: BLE001
                return None
        try:
            out = {"fx": float(i.fx), "fy": float(i.fy), "cx": float(i.cx), "cy": float(i.cy)}
            if getattr(i, "width", 0) and getattr(i, "height", 0):
                out.update(width=int(i.width), height=int(i.height))
            return out
        except (AttributeError, TypeError, ValueError):
            return None

    def read(self, timeout: Optional[float] = None) -> Optional[Frame]:
        if self._closed:
            return None
        np = _np()
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._closed:
                return None
            remaining = 1000 if deadline is None else int((deadline - time.monotonic()) * 1000)
            if deadline is not None and remaining <= 0:
                raise TimeoutError("Orbbec: no frame within %ss" % timeout)
            frames = self._pipe.wait_for_frames(max(1, min(remaining, 1000)))
            if frames is None:
                continue
            if self._align is not None:
                frames = self._align.process(frames)
                if not frames:
                    continue
                frames = frames.as_frame_set()
            c = frames.get_color_frame()
            d = frames.get_depth_frame() if self.has_depth else None
            if c is None or (self.has_depth and d is None):
                continue
            w, h = int(c.get_width()), int(c.get_height())
            color = color_to_rgb(str(c.get_format()), c.get_data(), w, h)
            depth = scale = None
            if d is not None:
                dw, dh = int(d.get_width()), int(d.get_height())
                depth = np.frombuffer(_bytes(d.get_data()), dtype="<u2")[: dw * dh].reshape(dh, dw).astype(np.uint16)
                scale = float(d.get_depth_scale()) * 0.001  # the SDK: millimetres per unit
            ts = c.get_timestamp_us() / 1e6 if hasattr(c, "get_timestamp_us") else c.get_timestamp() / 1e3
            return Frame(color=color, depth=depth, depth_scale=scale, timestamp=float(ts),
                         frame_id=self._next_id(), intrinsics=dict(self.intrinsics) if self.intrinsics else None)

    def _release(self) -> None:
        if getattr(self, "_started", False):
            self._started = False
            try:
                self._pipe.stop()
            except Exception:  # noqa: BLE001
                pass
