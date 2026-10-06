"""The frame-source interface shared by every camera / video adapter.

A *source* hands out :class:`Frame` objects: an RGB colour image, optionally a depth image
ALIGNED to it (same height and width, pixel for pixel), and when the device knows them, the
colour camera's intrinsics. Vendor SDKs (OpenCV, GStreamer, RealSense, Orbbec, ROS 2) appear only
inside the thin adapters of this package, each imported lazily, so ``import visionserve`` never
needs any of them.

Write your own source by giving any object a ``read(timeout=None)`` method that returns a
:class:`Frame` (or an ``(H, W, 3)`` uint8 RGB array) and ``None`` at the end of the stream;
:meth:`visionserve.Client.watch` accepts it as is.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Optional

try:  # Python 3.8+: typing.Protocol
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover - Python < 3.8 is not supported by the package
    Protocol = object  # type: ignore

    def runtime_checkable(c):  # type: ignore
        return c


def _np():
    try:
        import numpy as np
    except ImportError as e:  # pragma: no cover - numpy is in the [images] extra
        raise ImportError(
            "camera and video sources need numpy: pip install 'visionserve[images]'"
        ) from e
    return np


@dataclass
class Frame:
    """One frame from a camera or video.

    Attributes:
        color: ``(H, W, 3)`` ``uint8`` RGB image (RGB, not OpenCV's BGR).
        depth: ``(H, W)`` depth image ALIGNED to ``color`` (same size, pixel for pixel), or
            ``None``. ``uint16`` raw sensor units (metres = value × ``depth_scale``) or
            ``float32`` / ``float64`` metres. ``0`` (and NaN / inf for floats) = no reading.
            Adapters align depth to colour with the vendor SDK (``rs.align``, Orbbec's
            ``AlignFilter``, an ``aligned_depth_to_color`` ROS topic).
        depth_scale: metres per ``depth`` unit for a ``uint16`` depth (RealSense D4xx: 0.001,
            L515: 0.00025, ROS ``16UC1``: 0.001); ``None`` for float depth in metres.
        timestamp: capture time in seconds: the device / message clock when the source has one
            (RealSense, Orbbec, ROS header stamp), the position in the file for a video file,
            else ``time.time()`` when the frame was read.
        frame_id: the frame's index in the stream (0, 1, 2, ...): every frame the source
            produced counts, so a gap between two results' ``frame_id`` is the number of frames
            :meth:`~visionserve.Client.watch` skipped to stay real time. ``-1`` = not numbered
            by the source (``watch`` then numbers it).
        intrinsics: the colour camera's pinhole intrinsics ``{"fx", "fy", "cx", "cy"}`` in
            pixels (``width`` / ``height`` too when known), or ``None``. They also describe the
            aligned depth, so :func:`visionserve.backproject` can turn a pixel + depth into 3-D.
    """

    color: Any
    depth: Optional[Any] = None
    depth_scale: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    frame_id: int = -1
    intrinsics: Optional[Dict[str, float]] = None

    @property
    def height(self) -> int:
        return int(self.color.shape[0])

    @property
    def width(self) -> int:
        return int(self.color.shape[1])

    @property
    def has_depth(self) -> bool:
        return self.depth is not None

    def check(self) -> "Frame":
        """Validate the frame (raises ``ValueError`` / ``TypeError``) and return it."""
        np = _np()
        c = self.color
        if not isinstance(c, np.ndarray) or c.ndim != 3 or c.shape[2] != 3 or c.dtype != np.uint8:
            raise ValueError(
                "Frame.color must be an (H, W, 3) uint8 RGB array, got %s"
                % (_describe(c),)
            )
        if c.shape[0] == 0 or c.shape[1] == 0:
            raise ValueError("Frame.color is empty: %r" % (c.shape,))
        if self.depth is not None:
            d = self.depth
            if not isinstance(d, np.ndarray) or d.ndim != 2:
                raise ValueError("Frame.depth must be an (H, W) array, got %s" % (_describe(d),))
            if d.shape != c.shape[:2]:
                raise ValueError(
                    "Frame.depth is %dx%d but Frame.color is %dx%d: depth must be ALIGNED to the "
                    "colour image (same size; align it with the camera SDK, e.g. rs.align or an "
                    "aligned_depth_to_color topic)" % (d.shape[1], d.shape[0], c.shape[1], c.shape[0])
                )
            if d.dtype not in (np.uint16, np.float32, np.float64):
                raise ValueError(
                    "Frame.depth must be uint16 (with depth_scale) or float32/float64 metres, got %s"
                    % d.dtype
                )
        if self.depth_scale is not None:
            s = self.depth_scale
            if isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) or s <= 0:
                raise ValueError("Frame.depth_scale must be a positive number, got %r" % (s,))
        return self

    def camera_intrinsics(self) -> Optional[Any]:
        """``intrinsics`` as a :class:`visionserve.CameraIntrinsics` (for
        :func:`visionserve.grasp_distances`, :func:`visionserve.object_distances`, ...), or
        ``None`` when the source gave none."""
        if not self.intrinsics:
            return None
        from ..postprocess import CameraIntrinsics

        i = self.intrinsics
        return CameraIntrinsics(fx=float(i["fx"]), fy=float(i["fy"]), cx=float(i["cx"]), cy=float(i["cy"]))

    def depth_meters(self) -> Optional[Any]:
        """The depth as a ``float32`` ``(H, W)`` array in metres, NaN where there is no reading
        (0, negative, NaN or inf), or ``None`` without depth. A ``uint16`` depth without a
        ``depth_scale`` is taken as millimetres (0.001), the common sensor convention."""
        if self.depth is None:
            return None
        np = _np()
        d = np.asarray(self.depth)
        if d.dtype.kind in "ui":
            scale = 0.001 if self.depth_scale is None else float(self.depth_scale)
        else:
            scale = 1.0 if self.depth_scale is None else float(self.depth_scale)
        m = d.astype(np.float32) * np.float32(scale)
        m[~np.isfinite(m) | (m <= 0)] = np.nan
        return m


def _describe(a: Any) -> str:
    shape = getattr(a, "shape", None)
    dtype = getattr(a, "dtype", None)
    if shape is not None:
        return "%s %s" % (tuple(shape), dtype)
    return type(a).__name__


@runtime_checkable
class FrameSource(Protocol):
    """What :meth:`visionserve.Client.watch` reads frames from.

    ``read(timeout)`` blocks until the next frame and returns it, returns ``None`` at the end of
    the stream (a file ended, a camera was unplugged), and raises ``TimeoutError`` when
    ``timeout`` seconds passed without a frame (the stream may still continue; sources that
    cannot wait with a timeout, like OpenCV, ignore it). ``close()`` releases the device; it is
    safe to call twice. Every source is a context manager.
    """

    has_depth: bool

    def read(self, timeout: Optional[float] = None) -> Optional[Frame]:  # pragma: no cover
        ...

    def close(self) -> None:  # pragma: no cover
        ...


class BaseSource:
    """Shared plumbing for the adapters: frame numbering, context manager, idempotent close."""

    has_depth = False

    def __init__(self) -> None:
        self._count = 0
        self._closed = False

    def _next_id(self) -> int:
        n = self._count
        self._count += 1
        return n

    def read(self, timeout: Optional[float] = None) -> Optional[Frame]:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._release()

    def _release(self) -> None:  # pragma: no cover - adapters override
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __iter__(self) -> Iterator[Frame]:
        """Iterate frames until the end of the stream (``for frame in source``)."""
        while True:
            f = self.read()
            if f is None:
                return
            yield f

    def __del__(self) -> None:  # best effort; close() explicitly or use ``with``
        try:
            self.close()
        except Exception:  # noqa: BLE001 - never raise from a finalizer
            pass


class IterSource(BaseSource):
    """A source over any iterable of :class:`Frame` objects or ``(H, W, 3)`` uint8 RGB arrays
    (a list, a generator, a NumPy stack): synthetic streams, frames you already decoded, tests.

    ``has_depth`` is what you say it is (frames carry their own ``depth``); ``fps`` paces
    ``read()`` like a live camera (``None`` = as fast as the iterable yields).
    """

    def __init__(self, frames: Iterable[Any], *, has_depth: bool = False, fps: Optional[float] = None):
        super().__init__()
        self._it = iter(frames)
        self.has_depth = bool(has_depth)
        if fps is not None and (not isinstance(fps, (int, float)) or not fps > 0):
            raise ValueError("fps must be a positive number, got %r" % (fps,))
        self._period = 1.0 / fps if fps else 0.0
        self._next_t: Optional[float] = None

    def read(self, timeout: Optional[float] = None) -> Optional[Frame]:
        if self._closed:
            return None
        if self._period:
            now = time.monotonic()
            if self._next_t is not None and now < self._next_t:
                time.sleep(self._next_t - now)
            self._next_t = max(now, self._next_t or now) + self._period
        try:
            item = next(self._it)
        except StopIteration:
            return None
        frame = as_frame(item)
        if frame.frame_id < 0:
            frame.frame_id = self._next_id()
        return frame


def as_frame(item: Any) -> Frame:
    """``item`` as a :class:`Frame`: a Frame is returned as is, an ``(H, W, 3)`` uint8 RGB array
    is wrapped (``timestamp`` = now, not numbered)."""
    if isinstance(item, Frame):
        return item
    np = _np()
    if isinstance(item, np.ndarray):
        return Frame(color=item)
    raise TypeError(
        "a source must return a visionserve.sources.Frame or an (H, W, 3) uint8 RGB numpy array, "
        "got %s" % type(item).__name__
    )


def require(module: str, install: str) -> Any:
    """Import a vendor module lazily, or raise ``ImportError`` naming how to install it."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError("this source needs the %r module: %s" % (module, install)) from e


__all__ = ["Frame", "FrameSource", "BaseSource", "IterSource", "as_frame", "require"]
