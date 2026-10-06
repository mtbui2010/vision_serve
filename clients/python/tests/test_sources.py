"""Frame-source adapters.

* RealSense, Orbbec and ROS 2: no hardware here, so their vendor modules are replaced by fakes in
  ``sys.modules`` that follow the documented APIs (pyrealsense2, pyorbbecsdk v2, rclpy +
  message_filters); the tests check what the adapter does with what the SDK hands it (alignment,
  units, colour order, row padding, endianness, latest-frame, timeouts, shutdown).
* OpenCV and GStreamer: REAL, on a generated video file and ``videotestsrc``; skipped when cv2 /
  ``gst-launch-1.0`` is missing.
"""

import array
import io
import shutil
import sys
import threading
import time
import types

import numpy as np
import pytest
from PIL import Image

from visionserve.sources import Frame, IterSource, open_source


# --------------------------------------------------------------------------------------------
# RealSense (mocked pyrealsense2)
# --------------------------------------------------------------------------------------------
def fake_realsense(framesets, depth_scale=0.001):
    rs = types.ModuleType("pyrealsense2")
    rs.log = {"stopped": 0, "aligned": 0, "streams": [], "device": None, "align_to": None}

    class stream:
        color, depth = "color", "depth"

    class format:
        rgb8, z16 = "rgb8", "z16"

    class config:
        def enable_device(self, s):
            rs.log["device"] = s

        def enable_stream(self, *a):
            rs.log["streams"].append(a)

    class _Intr:
        fx, fy, ppx, ppy, width, height = 615.0, 616.0, 321.5, 239.5, 8, 6

    class _Profile:
        def get_device(self):
            return types.SimpleNamespace(first_depth_sensor=lambda: types.SimpleNamespace(get_depth_scale=lambda: depth_scale))

        def get_stream(self, s):
            vsp = types.SimpleNamespace(get_intrinsics=lambda: _Intr())
            return types.SimpleNamespace(as_video_stream_profile=lambda: vsp)

    class _Frame:
        def __init__(self, arr):
            self.arr = arr

        def get_data(self):
            return self.arr

        def __bool__(self):
            return self.arr is not None

    class _FrameSet:
        def __init__(self, color, depth, ts_ms):
            self.c, self.d, self.ts = _Frame(color), _Frame(depth), ts_ms

        def get_color_frame(self):
            return self.c

        def get_depth_frame(self):
            return self.d

        def get_timestamp(self):
            return self.ts

    queue = [_FrameSet(*f) for f in framesets]

    class pipeline:
        def start(self, cfg):
            return _Profile()

        def wait_for_frames(self, ms):
            if not queue:
                raise RuntimeError("Frame didn't arrive within %d" % ms)
            return queue.pop(0)

        def stop(self):
            rs.log["stopped"] += 1

    class align:
        def __init__(self, to):
            rs.log["align_to"] = to

        def process(self, fs):
            rs.log["aligned"] += 1
            return fs

    rs.stream, rs.format, rs.config, rs.pipeline, rs.align = stream, format, config, pipeline, align
    return rs


def test_realsense_colour_aligned_depth_scale_intrinsics(monkeypatch):
    color = np.random.default_rng(0).integers(0, 255, (6, 8, 3), dtype=np.uint8)
    depth = np.arange(48, dtype=np.uint16).reshape(6, 8)
    rs = fake_realsense([(color, depth, 1500.0)], depth_scale=0.00025)
    monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
    from visionserve.sources.realsense import RealSenseSource

    with open_source("realsense:123456", width=8, height=6, fps=15) as src:
        assert isinstance(src, RealSenseSource) and src.has_depth
        f = src.read()
        assert (f.color == color).all() and f.color.dtype == np.uint8
        assert f.depth.dtype == np.uint16 and (f.depth == depth).all()
        assert f.depth_scale == 0.00025 and f.timestamp == 1.5 and f.frame_id == 0
        assert f.intrinsics == {"fx": 615.0, "fy": 616.0, "cx": 321.5, "cy": 239.5, "width": 8, "height": 6}
        assert f.check() is f
        assert rs.log["aligned"] == 1 and rs.log["align_to"] == "color" and rs.log["device"] == "123456"
        assert ("color", 8, 6, "rgb8", 15) in rs.log["streams"] and ("depth", 8, 6, "z16", 15) in rs.log["streams"]
        with pytest.raises(TimeoutError):
            src.read(timeout=0.05)
        color[:] = 0  # the frame owns its pixels (the SDK reuses its buffers)
        assert f.color.any()
    src.close()
    assert rs.log["stopped"] == 1


def test_realsense_colour_only(monkeypatch):
    rs = fake_realsense([(np.zeros((6, 8, 3), np.uint8), None, 0.0)])
    monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
    from visionserve.sources.realsense import RealSenseSource

    src = RealSenseSource(depth=False)
    f = src.read()
    assert not src.has_depth and f.depth is None and f.depth_scale is None
    assert rs.log["aligned"] == 0 and len(rs.log["streams"]) == 1
    src.close()


# --------------------------------------------------------------------------------------------
# Orbbec (mocked pyorbbecsdk v2)
# --------------------------------------------------------------------------------------------
class _Enum:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "OBFormat." + self.name


def fake_orbbec(framesets, rgb_ok=True):
    ob = types.ModuleType("pyorbbecsdk")
    ob.log = {"stopped": 0, "aligned": 0, "enabled": [], "profile_req": None}
    ob.OBSensorType = types.SimpleNamespace(COLOR_SENSOR="color", DEPTH_SENSOR="depth")
    ob.OBFormat = types.SimpleNamespace(RGB=_Enum("RGB"), BGR=_Enum("BGR"), MJPG=_Enum("MJPG"), YUYV=_Enum("YUYV"), Y16=_Enum("Y16"))
    ob.OBStreamType = types.SimpleNamespace(COLOR_STREAM="color_stream")
    ob.OBFrameAggregateOutputMode = types.SimpleNamespace(FULL_FRAME_REQUIRE="full")

    class _Profile:
        def __init__(self, kind):
            self.kind = kind

        def get_intrinsic(self):
            return types.SimpleNamespace(fx=500.0, fy=501.0, cx=4.0, cy=3.0, width=8, height=6)

    class _ProfileList:
        def __init__(self, kind):
            self.kind = kind

        def get_default_video_stream_profile(self):
            return _Profile(self.kind)

        def get_video_stream_profile(self, w, h, fmt, fps):
            ob.log["profile_req"] = (w, h, fmt.name, fps)
            if not rgb_ok:
                raise RuntimeError("OBError: no such profile")
            return _Profile(self.kind)

    class Context:
        def query_devices(self):
            return types.SimpleNamespace(get_count=lambda: 1, get_device_by_index=lambda i: "dev%d" % i)

    class Config:
        def enable_stream(self, p):
            ob.log["enabled"].append(p.kind)

        def set_frame_aggregate_output_mode(self, m):
            ob.log["aggregate"] = m

    queue = list(framesets)

    class Pipeline:
        def __init__(self, dev):
            ob.log["device"] = dev

        def get_stream_profile_list(self, kind):
            return _ProfileList(kind)

        def start(self, cfg):
            pass

        def stop(self):
            ob.log["stopped"] += 1

        def wait_for_frames(self, ms):
            if not queue:
                time.sleep(ms / 1000.0)
                return None
            return queue.pop(0)

    class AlignFilter:
        def __init__(self, align_to_stream):
            ob.log["align_to"] = align_to_stream

        def process(self, frames):
            ob.log["aligned"] += 1
            return frames

    ob.Context, ob.Config, ob.Pipeline, ob.AlignFilter = Context, Config, Pipeline, AlignFilter
    return ob


class _ObFrame:
    def __init__(self, fmt, data, w, h, scale=None, ts_us=2_000_000):
        self.fmt, self.data, self.w, self.h, self.scale, self.ts = fmt, data, w, h, scale, ts_us

    def get_format(self):
        return _Enum(self.fmt)

    def get_data(self):
        return self.data

    def get_width(self):
        return self.w

    def get_height(self):
        return self.h

    def get_depth_scale(self):
        return self.scale

    def get_timestamp_us(self):
        return self.ts


class _ObFrameSet:
    def __init__(self, c, d):
        self.c, self.d = c, d

    def __bool__(self):
        return True

    def as_frame_set(self):
        return self

    def get_color_frame(self):
        return self.c

    def get_depth_frame(self):
        return self.d


def _yuyv(rgb_like_pairs, w, h):
    """YUYV bytes from per-pixel (Y, U, V) with U/V shared by pixel pairs."""
    out = bytearray()
    for _ in range(h):
        for (y0, u, y1, v) in rgb_like_pairs[: w // 2]:
            out += bytes([y0, u, y1, v])
    return bytes(out)


def test_orbbec_formats_depth_scale_and_intrinsics(monkeypatch):
    w, h = 4, 2
    rgb = np.arange(w * h * 3, dtype=np.uint8).reshape(h, w, 3)
    depth = np.full((h, w), 750, np.uint16)
    jpg = io.BytesIO()
    Image.fromarray(np.full((h, w, 3), (200, 10, 10), np.uint8)).save(jpg, "JPEG", quality=100, subsampling=0)
    yuyv = _yuyv([(235, 128, 16, 128), (81, 90, 81, 240)], w, h)  # white, black, then BT.601 red x2
    sets = [
        _ObFrameSet(_ObFrame("RGB", rgb.tobytes(), w, h), _ObFrame("Y16", depth.tobytes(), w, h, scale=1.0)),
        _ObFrameSet(_ObFrame("BGR", rgb[:, :, ::-1].tobytes(), w, h), _ObFrame("Y16", depth.tobytes(), w, h, scale=0.1)),
        _ObFrameSet(_ObFrame("MJPG", jpg.getvalue(), w, h), _ObFrame("Y16", depth.tobytes(), w, h, scale=1.0)),
        _ObFrameSet(_ObFrame("YUYV", yuyv, w, h), _ObFrame("Y16", depth.tobytes(), w, h, scale=1.0)),
    ]
    ob = fake_orbbec(sets)
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", ob)
    src = open_source("orbbec:0")
    f = src.read()
    assert (f.color == rgb).all() and (f.depth == depth).all() and f.depth.dtype == np.uint16
    assert f.depth_scale == pytest.approx(0.001)  # the SDK says 1 mm per unit
    assert f.timestamp == 2.0 and f.intrinsics == {"fx": 500.0, "fy": 501.0, "cx": 4.0, "cy": 3.0, "width": 8, "height": 6}
    f = src.read()
    assert (f.color == rgb).all() and f.depth_scale == pytest.approx(0.0001)  # BGR -> RGB; 0.1 mm
    f = src.read()
    assert abs(f.color.astype(int) - (200, 10, 10)).max() <= 3
    f = src.read()
    assert f.color[0, 0].tolist() == [255, 255, 255] and f.color[0, 1].tolist() == [0, 0, 0]
    assert abs(f.color[0, 2].astype(int) - (255, 0, 0)).max() <= 3
    assert f.check() is f and f.frame_id == 3
    assert ob.log["aligned"] == 4 and ob.log["align_to"] == "color_stream" and ob.log["enabled"] == ["color", "depth"]
    with pytest.raises(TimeoutError):
        src.read(timeout=0.05)
    src.close()
    src.close()
    assert ob.log["stopped"] == 1


def test_orbbec_rgb_mode_request_and_errors(monkeypatch):
    ob = fake_orbbec([], rgb_ok=False)
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", ob)
    from visionserve.sources.orbbec import OrbbecSource, color_to_rgb

    with pytest.raises(OSError, match="no RGB colour profile"):
        OrbbecSource(width=1280, height=720, fps=30)
    assert ob.log["profile_req"] == (1280, 720, "RGB", 30)
    with pytest.raises(OSError, match="2 requested"):
        OrbbecSource(2)
    with pytest.raises(ValueError, match="NV12"):
        color_to_rgb("OBFormat.NV12", b"", 2, 2)


# --------------------------------------------------------------------------------------------
# ROS 2 (mocked rclpy, sensor_msgs, message_filters)
# --------------------------------------------------------------------------------------------
class _Msg:
    pass


def image_msg(arr, encoding, *, pad=0, bigendian=False, stamp=(10, 500_000_000), as_array=True):
    m = _Msg()
    a = np.asarray(arr)
    h, w = a.shape[:2]
    if bigendian:
        a = a.astype(a.dtype.newbyteorder(">"))
    row = a.reshape(h, -1).view(np.uint8).reshape(h, -1)
    rows = np.concatenate([row, np.full((h, pad), 0xEE, np.uint8)], axis=1) if pad else row
    m.height, m.width, m.encoding, m.is_bigendian = h, w, encoding, int(bigendian)
    m.step = rows.shape[1]
    raw = rows.tobytes()
    m.data = array.array("B", raw) if as_array else raw
    m.header = types.SimpleNamespace(stamp=types.SimpleNamespace(sec=stamp[0], nanosec=stamp[1]))
    return m


def fake_ros(monkeypatch):
    log = {"subs": [], "sync": None, "shutdown": 0, "destroyed": 0, "node": None}
    rclpy = types.ModuleType("rclpy")
    context = types.ModuleType("rclpy.context")
    qos = types.ModuleType("rclpy.qos")
    executors = types.ModuleType("rclpy.executors")
    msgs = types.ModuleType("sensor_msgs.msg")
    mf = types.ModuleType("message_filters")

    class Context:
        pass

    class Node:
        def create_subscription(self, typ, topic, cb, q):
            log["subs"].append((typ.__name__, topic, cb, q))
            return object()

        def destroy_node(self):
            log["destroyed"] += 1

    def create_node(name, context=None):
        log["node"] = (name, context)
        return Node()

    class SingleThreadedExecutor:
        def __init__(self, context=None):
            self.ev = threading.Event()

        def add_node(self, n):
            pass

        def spin(self):
            self.ev.wait()

        def shutdown(self):
            self.ev.set()

    class Subscriber:
        def __init__(self, node, typ, topic, qos_profile=None):
            log["subs"].append((typ.__name__, topic, None, qos_profile))

    class ApproximateTimeSynchronizer:
        def __init__(self, subs, queue_size, slop):
            log["sync"] = {"n": len(subs), "queue": queue_size, "slop": slop}

        def registerCallback(self, cb):
            log["sync"]["cb"] = cb

    for name in ("Image", "CompressedImage", "CameraInfo"):
        setattr(msgs, name, type(name, (), {}))
    rclpy.init = lambda context=None: None
    rclpy.shutdown = lambda context=None: log.__setitem__("shutdown", log["shutdown"] + 1)
    rclpy.create_node = create_node
    context.Context = Context
    qos.qos_profile_sensor_data = "SENSOR_QOS"
    executors.SingleThreadedExecutor = SingleThreadedExecutor
    mf.Subscriber, mf.ApproximateTimeSynchronizer = Subscriber, ApproximateTimeSynchronizer
    for name, mod in (("rclpy", rclpy), ("rclpy.context", context), ("rclpy.qos", qos),
                      ("rclpy.executors", executors), ("sensor_msgs", types.ModuleType("sensor_msgs")),
                      ("sensor_msgs.msg", msgs), ("message_filters", mf)):
        monkeypatch.setitem(sys.modules, name, mod)
    return log


def test_image_msg_decoding():
    from visionserve.sources.ros2 import image_msg_to_array

    rgb = np.random.default_rng(1).integers(0, 255, (3, 5, 3), dtype=np.uint8)
    out, kind = image_msg_to_array(image_msg(rgb[:, :, ::-1], "bgr8", pad=3))  # padded rows
    assert kind == "rgb" and (out == rgb).all()
    out, _ = image_msg_to_array(image_msg(rgb, "rgb8", as_array=False))
    assert (out == rgb).all()
    rgba = np.concatenate([rgb, np.full((3, 5, 1), 9, np.uint8)], axis=2)
    assert (image_msg_to_array(image_msg(rgba[:, :, [2, 1, 0, 3]], "bgra8"))[0] == rgb).all()
    mono, _ = image_msg_to_array(image_msg(rgb[:, :, 0], "mono8"))
    assert mono.shape == (3, 5, 3) and (mono[:, :, 2] == rgb[:, :, 0]).all()
    mm = np.array([[0, 1000, 65535], [1, 2, 3]], np.uint16)
    for big in (False, True):
        out, kind = image_msg_to_array(image_msg(mm, "16UC1", pad=2, bigendian=big))
        assert kind == "depth_mm" and out.dtype == np.uint16 and (out == mm).all()
    metres = np.array([[np.nan, 0.5], [np.inf, 2.25]], np.float32)
    out, kind = image_msg_to_array(image_msg(metres, "32FC1", bigendian=True))
    assert kind == "depth_m" and out.dtype == np.float32
    assert np.isnan(out[0, 0]) and out[0, 1] == 0.5 and np.isinf(out[1, 0]) and out[1, 1] == 2.25
    with pytest.raises(ValueError, match="encoding"):
        image_msg_to_array(image_msg(rgb, "yuv422"))
    short = image_msg(rgb, "rgb8")
    short.data = short.data[:-1]
    with pytest.raises(ValueError, match="bytes"):
        image_msg_to_array(short)


def test_ros2_colour_and_aligned_depth(monkeypatch):
    log = fake_ros(monkeypatch)
    from visionserve.sources.ros2 import ROS2Source

    src = open_source("ros2:/camera/color/image_raw,/camera/aligned_depth_to_color/image_raw",
                      camera_info_topic="/camera/color/camera_info")
    assert isinstance(src, ROS2Source) and src.has_depth
    assert log["sync"]["n"] == 2 and log["sync"]["slop"] == 0.05
    assert [s[1] for s in log["subs"]] == ["/camera/color/image_raw", "/camera/aligned_depth_to_color/image_raw",
                                           "/camera/color/camera_info"]
    assert all(s[3] == "SENSOR_QOS" for s in log["subs"])
    info_cb = log["subs"][2][2]
    info = types.SimpleNamespace(k=[600.0, 0, 320.0, 0, 601.0, 240.0, 0, 0, 1], width=640, height=480)
    info_cb(info)
    cb = log["sync"]["cb"]
    rgb = np.zeros((2, 3, 3), np.uint8)
    rgb[..., 0] = 200
    mm = np.full((2, 3), 800, np.uint16)
    for i in range(3):  # three pairs before a read: only the latest is kept
        cb(image_msg(rgb[:, :, ::-1], "bgr8", stamp=(10 + i, 0)), image_msg(mm, "16UC1"))
    f = src.read(timeout=1)
    assert f.frame_id == 2 and f.timestamp == 12.0
    assert (f.color == rgb).all() and (f.depth == mm).all() and f.depth_scale == 0.001
    assert f.intrinsics == {"fx": 600.0, "fy": 601.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}
    with pytest.raises(TimeoutError):
        src.read(timeout=0.05)
    cb(image_msg(rgb, "rgb8"), image_msg(np.full((2, 3), 0.8, np.float32), "32FC1"))
    f = src.read(timeout=1)
    assert f.depth.dtype == np.float32 and f.depth_scale is None and f.check() is f
    cb(image_msg(rgb, "rgb8"), image_msg(np.zeros((1, 3), np.uint16), "16UC1"))  # not aligned
    with pytest.raises(ValueError, match="ALIGNED"):
        src.read(timeout=1)
    src.close()
    assert log["destroyed"] == 1 and log["shutdown"] == 1
    assert not [t for t in threading.enumerate() if t.name == "vs-ros2-spin" and t.is_alive()]
    assert src.read() is None


def test_ros2_compressed_colour_only(monkeypatch):
    log = fake_ros(monkeypatch)
    from visionserve.sources.ros2 import ROS2Source

    src = ROS2Source("/camera/color/image_raw/compressed")
    assert not src.has_depth and log["subs"][0][0] == "CompressedImage"
    buf = io.BytesIO()
    Image.fromarray(np.full((4, 6, 3), (0, 0, 250), np.uint8)).save(buf, "PNG")
    msg = _Msg()
    msg.data = array.array("B", buf.getvalue())
    msg.header = types.SimpleNamespace(stamp=types.SimpleNamespace(sec=1, nanosec=250_000_000))
    log["subs"][0][2](msg)
    f = src.read(timeout=1)
    assert f.color.shape == (4, 6, 3) and f.color[0, 0].tolist() == [0, 0, 250] and f.timestamp == 1.25
    src.close()


# --------------------------------------------------------------------------------------------
# OpenCV (REAL: a generated video file)
# --------------------------------------------------------------------------------------------
def fake_cv2(sessions):
    """A cv2 stand-in: each VideoCapture() takes the next session, a list of frames (BGR) it
    delivers before its read() fails; a session None cannot be opened."""
    cv = types.ModuleType("cv2")
    cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT, cv.CAP_PROP_FPS, cv.CAP_PROP_POS_MSEC = 3, 4, 5, 0
    cv.opened = 0

    class VideoCapture:
        def __init__(self, spec, *a):
            self.frames = sessions.pop(0) if sessions else None
            cv.opened += 1

        def isOpened(self):
            return self.frames is not None

        def read(self):
            if self.frames:
                return True, self.frames.pop(0)
            return False, None

        def set(self, *a):
            return True

        def get(self, prop):
            return 0.0

        def release(self):
            pass

    cv.VideoCapture = VideoCapture
    return cv


def test_opencv_stream_reconnects_and_honours_timeout(monkeypatch):
    bgr = np.zeros((4, 4, 3), np.uint8)
    bgr[..., 2] = 255  # red in BGR
    cv = fake_cv2([[bgr, bgr], [bgr], None, None, None])
    monkeypatch.setitem(sys.modules, "cv2", cv)
    from visionserve.sources.opencv import OpenCVSource

    src = OpenCVSource("rtsp://cam/stream", reconnect_delay=0.01)
    assert src.reconnect and not src.realtime
    got = [src.read(timeout=1) for _ in range(3)]  # 2 frames, a dropped stream, 1 frame
    assert [f.frame_id for f in got] == [0, 1, 2] and (got[2].color[0, 0] == (255, 0, 0)).all()
    with pytest.raises(TimeoutError):  # the stream stays down: read() returns to its caller
        src.read(timeout=0.02)
    src.close()
    cv2 = fake_cv2([[bgr], None])
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    src = OpenCVSource("rtsp://cam/stream", reconnect_delay=0.01, max_reconnects=1)
    assert src.read() is not None and src.read() is None  # gave up after one failed reconnect
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2([[bgr], None]))
    src = OpenCVSource("/dev/video9")
    assert not src.reconnect and src.read() is not None and src.read() is None  # a camera that stops ends the stream
    with pytest.raises(OSError, match="could not open"):
        OpenCVSource(3)


def _cv2_or_skip():
    try:
        import cv2  # noqa: F401
    except ImportError:
        pytest.skip("opencv not installed")
    return sys.modules["cv2"]


def _write_video(path, colours, w=64, h=48, fps=20):
    cv = _cv2_or_skip()
    vw = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not vw.isOpened():
        pytest.skip("this OpenCV build cannot write MJPG")
    for rgb in colours:
        vw.write(np.full((h, w, 3), rgb[::-1], np.uint8))  # OpenCV writes BGR
    vw.release()


def test_opencv_reads_a_video_file_as_rgb(tmp_path):
    path = tmp_path / "clip.avi"
    colours = [(220, 20, 20), (20, 220, 20), (20, 20, 220)] * 3
    _write_video(path, colours)
    with open_source(str(path), realtime=False) as src:
        got = list(src)
    assert len(got) == len(colours)
    for f, rgb in zip(got, colours):
        assert f.color.shape == (48, 64, 3) and f.color.dtype == np.uint8
        assert abs(f.color.reshape(-1, 3).mean(0) - rgb).max() < 12  # RGB, not BGR
    assert [f.frame_id for f in got] == list(range(len(colours)))
    assert got[1].timestamp == pytest.approx(0.05, abs=1e-3)  # position in the file
    assert src.read() is None


def test_opencv_realtime_pacing_and_wrapping_a_capture(tmp_path):
    cv = _cv2_or_skip()
    path = tmp_path / "clip.avi"
    _write_video(path, [(100, 100, 100)] * 6, fps=20)
    t0 = time.monotonic()
    with open_source(path) as src:  # a file plays at its own rate by default
        n = len(list(src))
    assert n == 6 and time.monotonic() - t0 >= 0.2  # 5 frame periods of 50 ms
    cap = cv.VideoCapture(str(path))
    with open_source(cap, realtime=False) as src:
        assert len(list(src)) == 6
    assert not cap.isOpened()  # the wrapper released it


def test_watch_a_video_file_end_to_end_with_a_stub(tmp_path):
    from test_watch import Stub

    from visionserve import Client

    path = tmp_path / "clip.avi"
    _write_video(path, [(i * 20, 0, 0) for i in range(8)])
    s = Stub()
    try:
        out = list(Client(s.url).watch(str(path), "det", drop_frames=False))
    finally:
        s.close()
    assert [r.frame_id for r in out] == list(range(8))


# --------------------------------------------------------------------------------------------
# GStreamer (REAL: gst-launch-1.0 videotestsrc)
# --------------------------------------------------------------------------------------------
needs_gst = pytest.mark.skipif(shutil.which("gst-launch-1.0") is None, reason="gst-launch-1.0 not installed")


@needs_gst
def test_gstreamer_videotestsrc_frames_rgb_and_padded_rows():
    with open_source("gst:videotestsrc num-buffers=10 pattern=red", width=65, height=7) as src:
        got = list(src)
    assert len(got) == 10 and [f.frame_id for f in got] == list(range(10))
    f = got[0]
    assert f.color.shape == (7, 65, 3)  # 65 * 3 = 195-byte rows, padded to 196 by GStreamer
    assert (f.color == (255, 0, 0)).all()


@needs_gst
def test_gstreamer_native_size_from_caps_and_timeout():
    from visionserve.sources.gstreamer import GStreamerSource

    src = GStreamerSource("videotestsrc pattern=blue is-live=true")
    f = src.read(timeout=10)
    assert f.color.shape == (240, 320, 3) and (f.color == (0, 0, 255)).all()
    src.close()
    assert src.read() is None


@needs_gst
def test_gstreamer_errors_and_restart():
    from visionserve.sources.gstreamer import GStreamerSource, build_command

    with pytest.raises(RuntimeError, match="no element"):
        GStreamerSource("nosuchelement ! fakesink").read(timeout=10)
    src = GStreamerSource("videotestsrc num-buffers=3", width=16, height=8, restart=True, restart_delay=0.05)
    assert [src.read(timeout=10).frame_id for _ in range(7)] == list(range(7))  # 3 + 3 + 1 across restarts
    src.close()
    cmd = build_command("rtspsrc location=rtsp://x latency=0 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv !", 3)
    assert cmd[:2] == ["gst-launch-1.0", "-v"] and cmd.count("!") == 7  # the trailing "!" is dropped
    assert cmd[-10:] == ["nvvidconv", "!", "videoconvert", "!", "video/x-raw,format=RGB", "!",
                         "fdsink", "name=vs_sink", "fd=3", "sync=false"]
    with pytest.raises(ValueError):
        build_command("videotestsrc", 3, width=10)


@needs_gst
def test_watch_a_gstreamer_pipeline_with_a_stub():
    from test_watch import Stub

    from visionserve import Client

    s = Stub()
    try:
        out = list(Client(s.url).watch("gst:videotestsrc num-buffers=5", "det", drop_frames=False))
    finally:
        s.close()
    assert [r.frame_id for r in out] == list(range(5))


def test_iter_source_paces_and_frames():
    src = IterSource([np.zeros((2, 2, 3), np.uint8)] * 4, fps=50)
    t0 = time.monotonic()
    got = list(src)
    assert len(got) == 4 and time.monotonic() - t0 >= 0.05
    assert [f.frame_id for f in got] == [0, 1, 2, 3]
    with pytest.raises(TypeError):
        list(IterSource(["nope"]))
    assert isinstance(got[0], Frame)
