"""Camera and video sources for :meth:`visionserve.Client.watch`.

One interface (:class:`FrameSource`: ``read(timeout) -> Frame | None``, ``close()``) over many
cameras; vendor SDKs live only in the thin, lazily imported adapters:

=============================================  ========================  ==========================
``spec`` given to :func:`open_source`           adapter                   needs
=============================================  ========================  ==========================
``0``, ``"0"``, ``"/dev/video0"``, a file path,  :class:`OpenCVSource`     ``visionserve[opencv]``
``"rtsp://..."``, ``"http(s)://..."``
``"gst:<pipeline>"``                            :class:`GStreamerSource`  ``gst-launch-1.0`` (system)
``"realsense"`` / ``"realsense:<serial>"``      :class:`RealSenseSource`  ``visionserve[realsense]``
``"orbbec"`` / ``"orbbec:<index>"``             :class:`OrbbecSource`     ``visionserve[orbbec]``
``"ros2:<color_topic>[,<depth_topic>]"``        :class:`ROS2Source`       a sourced ROS 2 install
any object with ``read()``                      used as is                --
=============================================  ========================  ==========================
"""

from __future__ import annotations

import os
from typing import Any, Tuple

from .frame import BaseSource, Frame, FrameSource, IterSource, as_frame

__all__ = [
    "Frame",
    "FrameSource",
    "BaseSource",
    "IterSource",
    "as_frame",
    "open_source",
    "OpenCVSource",
    "GStreamerSource",
    "RealSenseSource",
    "OrbbecSource",
    "ROS2Source",
]

_LAZY = {
    "OpenCVSource": ".opencv",
    "GStreamerSource": ".gstreamer",
    "RealSenseSource": ".realsense",
    "OrbbecSource": ".orbbec",
    "ROS2Source": ".ros2",
}


def __getattr__(name: str) -> Any:  # PEP 562: the adapters import nothing until used
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def _is_video_capture(obj: Any) -> bool:
    return all(hasattr(obj, a) for a in ("read", "grab", "retrieve", "isOpened"))


def _open(spec: Any, **opts: Any) -> Tuple[Any, bool]:
    """``(source, owned)``: owned is False when ``spec`` already was a source (the caller keeps
    ownership and closes it)."""
    from . import opencv

    if isinstance(spec, bool):
        raise TypeError("source must be a camera index, path, URL or spec string, got %r" % (spec,))
    if _is_video_capture(spec):
        return opencv.OpenCVSource(spec, **opts), True
    if hasattr(spec, "read") and callable(spec.read):
        if opts:
            raise TypeError("options %s apply only when opening a source from a spec" % sorted(opts))
        return spec, False
    if isinstance(spec, int):
        return opencv.OpenCVSource(spec, **opts), True
    if isinstance(spec, os.PathLike):
        spec = os.fspath(spec)
    if not isinstance(spec, str):
        raise TypeError("unsupported source %r: give a camera index, a path, a URL, a spec string "
                        "(gst:, realsense, orbbec, ros2:) or an object with read()" % (spec,))
    s = spec.strip()
    low = s.lower()
    if low.startswith("gst:"):
        from .gstreamer import GStreamerSource

        return GStreamerSource(s[4:], **opts), True
    if low == "realsense" or low.startswith("realsense:"):
        from .realsense import RealSenseSource

        serial = s.split(":", 1)[1].strip() if ":" in s else None
        return RealSenseSource(serial or None, **opts), True
    if low == "orbbec" or low.startswith("orbbec:"):
        from .orbbec import OrbbecSource

        idx = s.split(":", 1)[1].strip() if ":" in s else ""
        if idx and not idx.isdigit():
            raise ValueError("orbbec:<index> takes a camera index, got %r" % idx)
        return OrbbecSource(int(idx or 0), **opts), True
    if low.startswith("ros2:"):
        from .ros2 import ROS2Source

        topics = [t.strip() for t in s[5:].split(",")]
        if not topics[0] or len(topics) > 2:
            raise ValueError("ros2 spec is ros2:<color_topic>[,<depth_topic>], got %r" % spec)
        return ROS2Source(topics[0], topics[1] if len(topics) == 2 and topics[1] else None, **opts), True
    if s.isdigit():
        return opencv.OpenCVSource(int(s), **opts), True
    if not opencv._is_stream_url(s) and not s.startswith("/dev/") and not os.path.exists(s):
        raise FileNotFoundError("no such video file or device: %r (a camera is an index like 0, a stream a "
                                "rtsp:// or http(s):// URL)" % spec)
    return opencv.OpenCVSource(s, **opts), True


def open_source(spec: Any, **opts: Any) -> Any:
    """Open a camera or video as a :class:`FrameSource` (see the module table for ``spec``).

    ``opts`` go to the adapter (``width``, ``height``, ``fps``, ``reconnect``, ...; see each
    class). An object that already has ``read()`` is returned as is; a ``cv2.VideoCapture`` is
    wrapped in :class:`OpenCVSource` (BGR becomes RGB). Use the result as a context manager, or
    call ``close()``.
    """
    return _open(spec, **opts)[0]
