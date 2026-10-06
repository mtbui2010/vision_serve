"""ROS 2 adapter: subscribe to an image topic (and an aligned depth topic).

``rclpy``, ``sensor_msgs`` and ``message_filters`` come with the ROS 2 install (Apache-2.0), not
from PyPI: ``source /opt/ros/<distro>/setup.bash`` before running Python. No ``cv_bridge``:
``sensor_msgs/Image`` is decoded with NumPy (``rgb8``, ``bgr8``, ``rgba8``, ``bgra8``, ``mono8``;
depth ``16UC1`` / ``mono16`` in millimetres per REP 118, ``32FC1`` in metres), and
``sensor_msgs/CompressedImage`` (a colour topic ending in ``/compressed``) with Pillow.

With a depth topic the two are paired by ``message_filters.ApproximateTimeSynchronizer``. The
depth topic must already be aligned to the colour image (same size): the RealSense ROS driver
publishes ``/camera/aligned_depth_to_color/image_raw`` with ``align_depth.enable:=true``;
Orbbec's driver has ``depth_registration:=true``.
"""

from __future__ import annotations

import io
import threading
import time
from typing import Any, Optional, Tuple

from .frame import BaseSource, Frame, _np, require

_INSTALL = ("rclpy is part of ROS 2, not PyPI: source your ROS 2 setup first "
            "(e.g. source /opt/ros/humble/setup.bash) and use the system Python it targets")

# encoding -> (channels, numpy dtype, kind)
_ENCODINGS = {
    "rgb8": (3, "u1", "rgb"), "bgr8": (3, "u1", "bgr"),
    "rgba8": (4, "u1", "rgb"), "bgra8": (4, "u1", "bgr"),
    "mono8": (1, "u1", "mono"), "8uc1": (1, "u1", "mono"), "8uc3": (3, "u1", "bgr"),
    "16uc1": (1, "u2", "depth_mm"), "mono16": (1, "u2", "depth_mm"),
    "32fc1": (1, "f4", "depth_m"),
}


def image_msg_to_array(msg: Any) -> Tuple[Any, str]:
    """A ``sensor_msgs/Image`` as ``(array, kind)``: kind ``rgb`` (``(H, W, 3)`` uint8 RGB),
    ``depth_mm`` (``(H, W)`` uint16 millimetres) or ``depth_m`` (``(H, W)`` float32 metres).
    Honours ``step`` (row padding) and ``is_bigendian``."""
    np = _np()
    enc = str(msg.encoding).lower()
    if enc not in _ENCODINGS:
        raise ValueError("ROS image encoding %r is not supported (use rgb8, bgr8, rgba8, bgra8, mono8, 16UC1, 32FC1)" % msg.encoding)
    ch, base, kind = _ENCODINGS[enc]
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    item = np.dtype(base).itemsize
    row_bytes = w * ch * item
    if step < row_bytes:
        raise ValueError("ROS image step %d is smaller than width x channels x %d bytes = %d" % (step, item, row_bytes))
    try:
        buf = np.frombuffer(msg.data, dtype=np.uint8)
    except (TypeError, ValueError):
        buf = np.asarray(msg.data, dtype=np.uint8)
    if buf.size < h * step:
        raise ValueError("ROS image has %d bytes, expected height x step = %d" % (buf.size, h * step))
    rows = buf[: h * step].reshape(h, step)[:, :row_bytes]
    order = ">" if getattr(msg, "is_bigendian", False) and item > 1 else "<"
    pix = np.ascontiguousarray(rows).view(order + base).astype(base if item == 1 else "=" + base)
    pix = pix.reshape(h, w, ch) if ch > 1 else pix.reshape(h, w)
    if kind == "rgb":
        return np.ascontiguousarray(pix[:, :, :3]), "rgb"
    if kind == "bgr":
        return np.ascontiguousarray(pix[:, :, 2::-1]), "rgb"
    if kind == "mono":
        return np.ascontiguousarray(np.stack([pix, pix, pix], axis=-1)), "rgb"
    if kind == "depth_mm":
        return pix.astype(np.uint16), kind
    return pix.astype(np.float32), kind


def compressed_msg_to_rgb(msg: Any) -> Any:
    """A ``sensor_msgs/CompressedImage`` (JPEG / PNG) as ``(H, W, 3)`` uint8 RGB."""
    np = _np()
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("decoding a CompressedImage topic needs pillow: pip install 'visionserve[images]'") from e
    data = bytes(msg.data) if not isinstance(msg.data, bytes) else msg.data
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def _stamp(msg: Any) -> float:
    s = msg.header.stamp
    return float(s.sec) + float(s.nanosec) * 1e-9


class ROS2Source(BaseSource):
    """Frames from ROS 2 topics.

    Args:
        color_topic: a ``sensor_msgs/Image`` topic, or a ``sensor_msgs/CompressedImage`` topic
            when it ends in ``/compressed``.
        depth_topic: an aligned depth ``sensor_msgs/Image`` topic (``16UC1`` / ``32FC1``), or
            ``None`` for colour only.
        camera_info_topic: a ``sensor_msgs/CameraInfo`` topic for ``Frame.intrinsics`` (e.g.
            ``/camera/color/camera_info``), or ``None``.
        slop: the synchroniser's tolerance between colour and depth stamps, seconds.
        queue_size: the synchroniser's queue length.
        node_name: the name of the node this source creates.

    The source runs its own ``rclpy`` context and executor in a background thread, so it does
    not disturb a node you already run. It keeps only the LATEST frame: a frame you did not
    read before the next one arrived is dropped (and counted in ``frame_id``). QoS is
    ``qos_profile_sensor_data`` (best effort), which works with reliable publishers too.
    ``timestamp`` is the message's header stamp.
    """

    def __init__(
        self,
        color_topic: str,
        depth_topic: Optional[str] = None,
        *,
        camera_info_topic: Optional[str] = None,
        slop: float = 0.05,
        queue_size: int = 10,
        node_name: str = "visionserve_watch",
    ):
        super().__init__()
        rclpy = require("rclpy", _INSTALL)
        msgs = require("sensor_msgs.msg", _INSTALL)
        qos_mod = require("rclpy.qos", _INSTALL)
        executors = require("rclpy.executors", _INSTALL)
        context_mod = require("rclpy.context", _INSTALL)
        if not color_topic:
            raise ValueError("ROS2Source needs a colour topic")
        self.color_topic, self.depth_topic = color_topic, depth_topic or None
        self.has_depth = self.depth_topic is not None
        self._cond = threading.Condition()
        self._latest: Optional[Frame] = None
        self._error: Optional[BaseException] = None
        self.intrinsics: Optional[dict] = None
        self._compressed = color_topic.rstrip("/").endswith("/compressed")
        color_type = msgs.CompressedImage if self._compressed else msgs.Image
        qos = qos_mod.qos_profile_sensor_data

        self._rclpy = rclpy
        self._ctx = context_mod.Context()
        rclpy.init(context=self._ctx)
        self._node = rclpy.create_node(node_name, context=self._ctx)
        self._subs = []
        if self.has_depth:
            mf = require("message_filters", _INSTALL)
            cs = mf.Subscriber(self._node, color_type, color_topic, qos_profile=qos)
            ds = mf.Subscriber(self._node, msgs.Image, self.depth_topic, qos_profile=qos)
            self._sync = mf.ApproximateTimeSynchronizer([cs, ds], int(queue_size), float(slop))
            self._sync.registerCallback(self._on_pair)
            self._subs += [cs, ds]
        else:
            self._subs.append(self._node.create_subscription(color_type, color_topic, self._on_color, qos))
        if camera_info_topic:
            self._subs.append(self._node.create_subscription(msgs.CameraInfo, camera_info_topic, self._on_info, qos))
        self._executor = executors.SingleThreadedExecutor(context=self._ctx)
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._spin, daemon=True, name="vs-ros2-spin")
        self._thread.start()

    # -- callbacks (executor thread) ------------------------------------------------------------ #
    def _spin(self) -> None:
        try:
            self._executor.spin()
        except Exception as e:  # noqa: BLE001 - ExternalShutdownException etc. end the spin
            if not self._closed:
                self._fail(e)

    def _fail(self, e: BaseException) -> None:
        with self._cond:
            self._error = e
            self._cond.notify_all()

    def _color(self, msg: Any) -> Any:
        if self._compressed:
            return compressed_msg_to_rgb(msg)
        arr, kind = image_msg_to_array(msg)
        if kind != "rgb":
            raise ValueError("colour topic %s carries depth (%s)" % (self.color_topic, msg.encoding))
        return arr

    def _publish(self, color: Any, depth: Any, scale: Optional[float], stamp: float) -> None:
        with self._cond:
            intr = dict(self.intrinsics) if self.intrinsics else None
            self._latest = Frame(color=color, depth=depth, depth_scale=scale, timestamp=stamp,
                                 frame_id=self._next_id(), intrinsics=intr)
            self._cond.notify_all()

    def _on_color(self, msg: Any) -> None:
        try:
            self._publish(self._color(msg), None, None, _stamp(msg))
        except Exception as e:  # noqa: BLE001 - surface in read()
            self._fail(e)

    def _on_pair(self, cmsg: Any, dmsg: Any) -> None:
        try:
            color = self._color(cmsg)
            depth, kind = image_msg_to_array(dmsg)
            if kind == "rgb":
                raise ValueError("depth topic %s carries a colour image (%s)" % (self.depth_topic, dmsg.encoding))
            scale = 0.001 if kind == "depth_mm" else None
            if depth.shape != color.shape[:2]:
                raise ValueError(
                    "depth %s is %dx%d but colour %s is %dx%d: subscribe to a depth topic ALIGNED to the "
                    "colour image (e.g. /camera/aligned_depth_to_color/image_raw)"
                    % (self.depth_topic, depth.shape[1], depth.shape[0], self.color_topic, color.shape[1], color.shape[0]))
            self._publish(color, depth, scale, _stamp(cmsg))
        except Exception as e:  # noqa: BLE001
            self._fail(e)

    def _on_info(self, msg: Any) -> None:
        k = list(msg.k)
        if len(k) == 9 and k[0] > 0:
            with self._cond:
                self.intrinsics = {"fx": float(k[0]), "fy": float(k[4]), "cx": float(k[2]), "cy": float(k[5]),
                                   "width": int(msg.width), "height": int(msg.height)}

    # -- reading -------------------------------------------------------------------------------- #
    def read(self, timeout: Optional[float] = None) -> Optional[Frame]:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while True:
                if self._error is not None:
                    e, self._error = self._error, None
                    raise e
                if self._closed:
                    return None
                if self._latest is not None:
                    f, self._latest = self._latest, None
                    return f
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("no message on %s within %ss" % (self.color_topic, timeout))
                self._cond.wait(remaining if remaining is not None else 0.5)

    def _release(self) -> None:
        with self._cond:
            self._cond.notify_all()
        try:
            self._executor.shutdown()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._rclpy.shutdown(context=self._ctx)
        except Exception:  # noqa: BLE001
            pass
        self._thread.join(timeout=2)
