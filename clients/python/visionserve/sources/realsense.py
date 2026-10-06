"""Intel RealSense adapter (``pip install 'visionserve[realsense]'``: ``pyrealsense2``, Apache-2.0).

Colour (``rgb8``) and depth (``z16``) streams; depth is aligned to colour with ``rs.align`` so a
depth pixel sits under the colour pixel it measures, and ``depth_scale`` comes from the device's
depth sensor (0.001 m per unit on the D400 series).
"""

from __future__ import annotations

from typing import Optional, Tuple

from .frame import BaseSource, Frame, _np, require

_INSTALL = "pip install 'visionserve[realsense]'  (pyrealsense2; on Jetson / ARM build librealsense with its Python bindings)"


class RealSenseSource(BaseSource):
    """Frames from an Intel RealSense camera.

    Args:
        serial: the camera's serial number (``None`` = the first one found).
        width, height, fps: the colour stream's mode (default 640 x 480 at 30 fps).
        depth: also stream depth, aligned to colour (default ``True``).
        depth_size: ``(width, height)`` of the depth stream before alignment (default: the colour
            size); after alignment depth always has the colour image's size.

    ``timestamp`` is the device timestamp in seconds; ``intrinsics`` are the colour stream's
    (they hold for the aligned depth too).
    """

    def __init__(
        self,
        serial: Optional[str] = None,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        depth: bool = True,
        depth_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        rs = require("pyrealsense2", _INSTALL)
        self._rs = rs
        self.serial = serial or None
        self.has_depth = bool(depth)
        self._pipe = rs.pipeline()
        cfg = rs.config()
        if self.serial:
            cfg.enable_device(str(self.serial))
        cfg.enable_stream(rs.stream.color, int(width), int(height), rs.format.rgb8, int(fps))
        if self.has_depth:
            dw, dh = depth_size or (width, height)
            cfg.enable_stream(rs.stream.depth, int(dw), int(dh), rs.format.z16, int(fps))
        try:
            profile = self._pipe.start(cfg)
        except RuntimeError as e:
            raise OSError("RealSense: could not start %s: %s" % (self.serial or "the camera", e)) from e
        self._started = True
        self.depth_scale: Optional[float] = None
        self._align = None
        if self.has_depth:
            self.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
            self._align = rs.align(rs.stream.color)
        self.intrinsics: Optional[dict] = None
        try:
            vsp = profile.get_stream(rs.stream.color).as_video_stream_profile()
            i = vsp.get_intrinsics()
            self.intrinsics = {"fx": float(i.fx), "fy": float(i.fy), "cx": float(i.ppx), "cy": float(i.ppy),
                               "width": int(i.width), "height": int(i.height)}
        except (RuntimeError, AttributeError):
            self.intrinsics = None

    def read(self, timeout: Optional[float] = None) -> Optional[Frame]:
        if self._closed:
            return None
        np = _np()
        ms = 5000 if timeout is None else max(1, int(timeout * 1000))
        while True:
            try:
                frames = self._pipe.wait_for_frames(ms)
            except RuntimeError as e:
                if timeout is not None:
                    raise TimeoutError("RealSense: no frame within %ss (%s)" % (timeout, e)) from e
                if self._closed:
                    return None
                continue  # no timeout requested: keep waiting
            if self._align is not None:
                frames = self._align.process(frames)
            c = frames.get_color_frame()
            d = frames.get_depth_frame() if self.has_depth else None
            if not c or (self.has_depth and not d):
                continue  # an incomplete set (startup): wait for the next one
            color = np.array(np.asanyarray(c.get_data()), dtype=np.uint8, copy=True)
            depth = np.array(np.asanyarray(d.get_data()), dtype=np.uint16, copy=True) if d else None
            return Frame(
                color=color,
                depth=depth,
                depth_scale=self.depth_scale if depth is not None else None,
                timestamp=float(frames.get_timestamp()) / 1000.0,
                frame_id=self._next_id(),
                intrinsics=dict(self.intrinsics) if self.intrinsics else None,
            )

    def _release(self) -> None:
        if getattr(self, "_started", False):
            self._started = False
            try:
                self._pipe.stop()
            except RuntimeError:
                pass
