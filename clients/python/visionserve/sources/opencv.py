"""OpenCV adapter: webcams, video files, RTSP / HTTP streams (``pip install 'visionserve[opencv]'``).

OpenCV (``opencv-python-headless``) is Apache-2.0. Its wheels bundle FFmpeg (LGPL-2.1, dynamically
linked) for decoding; it is an optional extra the user installs, never a dependency of the core
client.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional, Union

from .frame import BaseSource, Frame, _np, require

log = logging.getLogger("visionserve.sources")

_INSTALL = "pip install 'visionserve[opencv]'  (opencv-python-headless)"


def _is_stream_url(spec: Any) -> bool:
    return isinstance(spec, str) and spec.lower().startswith(("rtsp://", "rtsps://", "http://", "https://", "rtmp://", "udp://", "tcp://"))


def bgr_to_rgb(img: Any) -> Any:
    """An OpenCV image (BGR, BGRA or grayscale) as a contiguous ``(H, W, 3)`` uint8 RGB array."""
    np = _np()
    a = np.asarray(img)
    if a.dtype != np.uint8:
        if a.dtype == np.uint16:  # 16-bit video (rare): keep the high byte
            a = (a >> 8).astype(np.uint8)
        else:
            a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        return np.ascontiguousarray(np.stack([a, a, a], axis=-1))
    if a.ndim == 3 and a.shape[2] == 1:
        return np.ascontiguousarray(np.repeat(a, 3, axis=2))
    if a.ndim == 3 and a.shape[2] in (3, 4):
        return np.ascontiguousarray(a[:, :, 2::-1])
    raise ValueError("unsupported OpenCV frame shape %r" % (a.shape,))


class OpenCVSource(BaseSource):
    """Frames from ``cv2.VideoCapture``: a camera index (``0``), a device (``/dev/video0``), a video
    file, or a stream URL (``rtsp://``, ``http(s)://``).

    Args:
        spec: what to open: an int index, a path or URL, or an already-open ``cv2.VideoCapture``
            (it is released by :meth:`close`).
        width, height, fps: ask a CAMERA for this mode (``CAP_PROP_FRAME_WIDTH`` / ``HEIGHT`` /
            ``FPS``); the camera may pick the nearest it supports. Ignored by files and streams.
        backend: an OpenCV API preference (e.g. ``cv2.CAP_V4L2``, ``cv2.CAP_FFMPEG``,
            ``cv2.CAP_GSTREAMER``); ``None`` = OpenCV's choice.
        realtime: pace a video FILE at its own frame rate, like a camera (default ``True`` for
            files): with :meth:`~visionserve.Client.watch` dropping stale frames, a file then
            plays like a live feed. ``False`` reads as fast as OpenCV decodes.
        reconnect: when a STREAM stops delivering frames, release it and open it again after
            ``reconnect_delay`` seconds (default: on for ``rtsp://`` / ``http(s)://`` URLs).
            ``max_reconnects`` bounds the attempts in a row (``None`` = keep trying).

    ``read(timeout)`` honours ``timeout`` only between reconnect attempts (``TimeoutError``):
    ``VideoCapture.read`` itself blocks, so a stalled RTSP stream blocks until OpenCV's own
    timeout. ``timestamp`` is the position in the file for a
    video file, ``time.time()`` otherwise.
    """

    def __init__(
        self,
        spec: Union[int, str, Any] = 0,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        backend: Optional[int] = None,
        realtime: Optional[bool] = None,
        reconnect: Optional[bool] = None,
        reconnect_delay: float = 1.0,
        max_reconnects: Optional[int] = None,
    ):
        super().__init__()
        self._cv2 = require("cv2", _INSTALL)
        self._cap = None
        if isinstance(spec, str) and spec.strip().isdigit():
            spec = int(spec.strip())
        self.spec = spec
        self._backend = backend
        self._mode = (width, height, fps)
        self._is_file = isinstance(spec, (str, os.PathLike)) and not _is_stream_url(spec) \
            and not str(spec).startswith("/dev/") and os.path.isfile(str(spec))
        self.realtime = self._is_file if realtime is None else bool(realtime)
        self.reconnect = _is_stream_url(spec) if reconnect is None else bool(reconnect)
        self.reconnect_delay = float(reconnect_delay)
        self.max_reconnects = max_reconnects
        self._t0: Optional[float] = None  # monotonic time of the first frame (realtime pacing)
        self._failures = 0  # reconnect attempts in a row
        if hasattr(spec, "read") and hasattr(spec, "grab") and hasattr(spec, "isOpened"):
            self._cap = spec  # an open cv2.VideoCapture
            self.spec = "<VideoCapture>"
        else:
            self._open()

    def _open(self) -> None:
        cv2 = self._cv2
        spec = self.spec
        if isinstance(spec, os.PathLike):
            spec = os.fspath(spec)
        cap = cv2.VideoCapture(spec) if self._backend is None else cv2.VideoCapture(spec, self._backend)
        if not cap.isOpened():
            cap.release()
            raise OSError("OpenCV could not open %r (no such camera, file or stream, or no backend for it)" % (self.spec,))
        w, h, fps = self._mode
        if not self._is_file:
            if w:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
            if h:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
            if fps:
                cap.set(cv2.CAP_PROP_FPS, float(fps))
        self._cap = cap

    def read(self, timeout: Optional[float] = None) -> Optional[Frame]:
        if self._closed or (self._cap is None and not self.reconnect):
            return None
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            ok, img = self._cap.read() if self._cap is not None else (False, None)
            if ok and img is not None:
                self._failures = 0
                break
            if not self.reconnect:
                return None  # end of file / camera gone
            self._failures += 1
            if self.max_reconnects is not None and self._failures > self.max_reconnects:
                return None
            log.warning("OpenCVSource: %s stopped delivering frames; reconnecting in %.1fs", self.spec, self.reconnect_delay)
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            time.sleep(self.reconnect_delay)
            if self._closed:
                return None
            try:
                self._open()
            except OSError as e:
                log.warning("OpenCVSource: reconnect failed: %s", e)
            if deadline is not None and time.monotonic() >= deadline:
                # Still reconnecting: let the caller check whether to stop; the next read()
                # carries on (a stalled read inside OpenCV itself cannot be interrupted).
                raise TimeoutError("OpenCVSource: %s is reconnecting" % (self.spec,))
        ts = time.time()
        if self._is_file:
            ts = float(self._cap.get(self._cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
            if self.realtime:
                self._pace(ts)
        return Frame(color=bgr_to_rgb(img), timestamp=ts, frame_id=self._next_id())

    def _pace(self, media_t: float) -> None:
        """Sleep until the wall clock reaches the frame's position in the file."""
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now - media_t
            return
        wait = self._t0 + media_t - now
        if wait > 0:
            time.sleep(wait)

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
