"""Client.watch(): latest-frame dropping, fps cap, in_flight, stop conditions, shutdown, the depth
policy and depth units, the tracker, and the `watch` CLI. Offline: a stub HTTP server stands in for
VisionServe and frames come from in-memory sources (no camera, no cv2 needed).
"""

import http.server
import io
import json
import threading
import time

import numpy as np
import pytest
from PIL import Image

import visionserve.watch as watch_mod
from visionserve import Client, Detection, IoUTracker, VisionServeError
from visionserve.sources import Frame, IterSource, open_source
from visionserve.track import box_iou
from visionserve.watch import depth_for_upload


def _parse_multipart(body, ctype):
    boundary = ctype.split("boundary=")[1].encode()
    out = {}
    for part in body.split(b"--" + boundary)[1:-1]:
        head, _, data = part[2:-2].partition(b"\r\n\r\n")
        name = head.decode().split('name="')[1].split('"')[0]
        out[name] = data
    return out


class Stub:
    """/api/models -> MODELS; /api/predict answers one detection whose class is the frame's
    pixel value (so a result names the frame it came from) and records every request."""

    MODELS = [
        {"name": "det", "task": "detection", "license": "Apache-2.0", "state": "loaded",
         "max_useful_side": None, "max_useful_short_side": 1120, "accepts_depth": False},
        {"name": "bg", "task": "segmentation", "license": "Apache-2.0", "state": "loaded",
         "max_useful_side": None, "max_useful_short_side": None, "accepts_depth": True},
        {"name": "old", "task": "detection", "license": "MIT", "state": "loaded"},  # predates accepts_depth
    ]

    def __init__(self, delay=0.0):
        self.delay = delay
        self.requests = []
        self.models_calls = 0
        self.active = 0
        self.max_active = 0
        self.fail = []  # queue of (status, headers) to answer before succeeding
        self.lock = threading.Lock()
        self.box = None  # callable(frame_value) -> bbox
        stub = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj, headers=None):
                payload = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                with stub.lock:
                    stub.models_calls += 1
                self._send(200, stub.MODELS)

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                fields = _parse_multipart(body, self.headers["Content-Type"])
                with stub.lock:
                    stub.active += 1
                    stub.max_active = max(stub.max_active, stub.active)
                    fail = stub.fail.pop(0) if stub.fail else None
                try:
                    if stub.delay:
                        time.sleep(stub.delay)
                    if fail is not None:
                        self._send(fail[0], {"error": "stub says %d" % fail[0]}, fail[1])
                        return
                    img = Image.open(io.BytesIO(fields["image"]))
                    value = int(np.asarray(img)[0, 0, 0])
                    with stub.lock:
                        stub.requests.append({"fields": fields, "value": value, "size": img.size})
                    bbox = stub.box(value) if stub.box else [1, 1, 4, 4]
                    self._send(200, {"task": "detection", "model": fields["model"].decode(), "duration_ms": 1.0,
                                     "detections": [{"bbox": bbox, "class": str(value), "conf": 0.9}]})
                finally:
                    with stub.lock:
                        stub.active -= 1

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = "http://127.0.0.1:%d" % self.srv.server_address[1]

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def stub():
    s = Stub()
    yield s
    s.close()


def frames(n, w=32, h=24, depth=None, scale=None):
    """n frames whose pixel value is their index (mod 256)."""
    for i in range(n):
        d = None
        if depth is not None:
            d = depth(i) if callable(depth) else depth
        yield Frame(color=np.full((h, w, 3), i % 256, np.uint8), depth=d, depth_scale=scale, frame_id=i)


def endless(w=32, h=24):
    i = 0
    while True:
        yield np.full((h, w, 3), i % 256, np.uint8)
        i += 1


class Closable(IterSource):
    closed_calls = 0

    def _release(self):
        type(self).closed_calls += 1


def _threads(prefix):
    return [t for t in threading.enumerate() if t.name.startswith(prefix) and t.is_alive()]


# --------------------------------------------------------------------------------------------
# frames in, results out
# --------------------------------------------------------------------------------------------
def test_every_frame_in_order_without_dropping(stub):
    c = Client(stub.url)
    out = list(c.watch(IterSource(frames(6)), "det", drop_frames=False))
    assert [r.frame_id for r in out] == list(range(6))
    # Each result answers its own frame: the stub names the detection by the frame's pixels.
    assert [r.detections[0].cls for r in out] == [str(i) for i in range(6)]
    assert all(r.timestamp is not None and r.frame is None for r in out)


def test_latest_frame_wins_while_a_request_runs():
    s = Stub(delay=0.05)
    try:
        c = Client(s.url)
        src = IterSource(frames(60), fps=200)  # a frame every 5 ms, a request every 50 ms
        out = list(c.watch(src, "det"))
    finally:
        s.close()
    ids = [r.frame_id for r in out]
    assert ids == sorted(set(ids)), ids  # in order, no frame twice
    assert 3 <= len(out) < 40, ids  # most frames skipped ...
    assert max(b - a for a, b in zip(ids, ids[1:])) > 1  # ... in gaps
    assert ids[-1] >= 50  # and the last results are recent frames, not a backlog


def test_fps_caps_the_request_rate(stub):
    c = Client(stub.url)
    t0 = time.monotonic()
    out = list(c.watch(IterSource(endless(), fps=100), "det", fps=10, duration=1.0))
    dt = time.monotonic() - t0
    assert 8 <= len(out) <= 12, len(out)
    assert dt < 2.0


def test_in_flight_runs_requests_concurrently_in_order():
    s = Stub(delay=0.15)
    try:
        c = Client(s.url)
        out = list(c.watch(IterSource(frames(9)), "det", in_flight=3, drop_frames=False))
        assert s.max_active == 3
        assert [r.frame_id for r in out] == list(range(9))
        s.max_active = 0
        list(c.watch(IterSource(frames(4)), "det", drop_frames=False))
        assert s.max_active == 1  # the default never overlaps
    finally:
        s.close()


def test_max_frames_and_duration_stop_sending(stub):
    c = Client(stub.url)
    out = list(c.watch(IterSource(endless()), "det", max_frames=5))
    assert len(out) == 5 and len(stub.requests) == 5
    t0 = time.monotonic()
    out = list(c.watch(IterSource(endless(), fps=50), "det", duration=0.3))
    assert 0.25 < time.monotonic() - t0 < 1.5
    assert out


def test_return_frames_and_ndarray_sources(stub):
    c = Client(stub.url)
    out = list(c.watch(IterSource([np.full((8, 8, 3), 7, np.uint8)] * 2), "det", return_frames=True, drop_frames=False))
    assert [r.frame_id for r in out] == [0, 1]  # numbered by watch when the source does not
    assert out[0].frame.color[0, 0, 0] == 7


def test_a_source_object_without_a_timeout_parameter(stub):
    class Plain:
        def __init__(self):
            self.n = 0

        def read(self):
            self.n += 1
            return None if self.n > 3 else np.zeros((4, 4, 3), np.uint8)

    assert len(list(Client(stub.url).watch(Plain(), "det", drop_frames=False))) == 3


# --------------------------------------------------------------------------------------------
# shutdown and errors
# --------------------------------------------------------------------------------------------
def test_break_stops_the_reader_and_closes_an_owned_source(stub, monkeypatch):
    Closable.closed_calls = 0
    src = Closable(endless(), fps=200)
    monkeypatch.setattr(watch_mod, "_open_source", lambda spec: (src, True))
    gen = Client(stub.url).watch("anything", "det")
    for i, _ in enumerate(gen):
        if i == 2:
            break
    gen.close()
    time.sleep(0.1)
    assert Closable.closed_calls == 1
    assert not _threads("visionserve-watch-reader")


def test_a_source_you_pass_stays_open(stub):
    Closable.closed_calls = 0
    src = Closable(frames(3))
    list(Client(stub.url).watch(src, "det", drop_frames=False))
    assert Closable.closed_calls == 0
    src.close()
    assert Closable.closed_calls == 1


def test_interrupt_inside_the_loop_cleans_up(stub, monkeypatch):
    Closable.closed_calls = 0
    src = Closable(endless(), fps=100)
    monkeypatch.setattr(watch_mod, "_open_source", lambda spec: (src, True))
    c = Client(stub.url)

    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(c, "predict", boom)
    with pytest.raises(KeyboardInterrupt):
        list(c.watch("x", "det"))
    time.sleep(0.1)
    assert Closable.closed_calls == 1
    assert not _threads("visionserve-watch-reader")


def test_server_errors_surface(stub):
    stub.fail = [(400, {})]
    with pytest.raises(VisionServeError) as e:
        list(Client(stub.url).watch(IterSource(frames(3)), "det", drop_frames=False))
    assert e.value.status == 400


def test_503_is_retried_once_after_retry_after(stub):
    stub.fail = [(503, {"Retry-After": "0"})]
    out = list(Client(stub.url).watch(IterSource(frames(2)), "det", drop_frames=False))
    assert len(out) == 2
    stub.fail = [(503, {"Retry-After": "0"}), (503, {"Retry-After": "0"})]
    with pytest.raises(VisionServeError) as e:
        list(Client(stub.url).watch(IterSource(frames(2)), "det", drop_frames=False))
    assert e.value.status == 503


def test_source_errors_surface(stub):
    def bad():
        yield np.zeros((4, 4, 3), np.uint8)
        yield np.zeros((4, 4), np.uint8)  # not RGB

    with pytest.raises(ValueError, match="RGB"):
        list(Client(stub.url).watch(IterSource(bad()), "det", drop_frames=False))


def test_arguments_are_checked_before_the_loop_starts(stub):
    c = Client(stub.url)
    for kw in ({"fps": 0}, {"fps": -1}, {"max_frames": 1.5}, {"duration": float("nan")},
               {"depth": "sometimes"}, {"in_flight": 0}, {"in_flight": 1000}):
        with pytest.raises(ValueError):
            c.watch(IterSource([]), "det", **kw)
    with pytest.raises(TypeError):
        c.watch(IterSource([]), "det", depth_dtype=None, image=1)
    with pytest.raises(TypeError):
        c.watch(IterSource([]), "det", track="yes")


# --------------------------------------------------------------------------------------------
# depth: policy, alignment, units
# --------------------------------------------------------------------------------------------
def _depth_frames(n=2, w=40, h=30):
    d = (np.arange(w * h, dtype=np.uint16).reshape(h, w) + 500)
    return IterSource(frames(n, w, h, depth=d, scale=0.001), has_depth=True)


def test_auto_sends_depth_only_to_models_that_read_it(stub):
    c = Client(stub.url)
    list(c.watch(_depth_frames(), "bg", drop_frames=False))
    assert all("depth" in r["fields"] for r in stub.requests)
    f = stub.requests[0]["fields"]
    assert f["depth_dtype"] == b"uint16" and f["depth_width"] == b"40" and f["depth_height"] == b"30"
    stub.requests.clear()
    list(c.watch(_depth_frames(), "det", drop_frames=False))
    list(c.watch(_depth_frames(), "old", drop_frames=False))  # a server that does not say: colour only
    assert stub.requests and not any("depth" in r["fields"] for r in stub.requests)
    assert stub.models_calls == 1  # one GET /api/models for all of it


def test_always_and_never(stub):
    c = Client(stub.url)
    list(c.watch(_depth_frames(), "det", depth="always", drop_frames=False))
    assert all("depth" in r["fields"] for r in stub.requests)
    stub.requests.clear()
    list(c.watch(_depth_frames(), "bg", depth="never", drop_frames=False))
    assert not any("depth" in r["fields"] for r in stub.requests)
    with pytest.raises(ValueError, match="without depth"):
        list(c.watch(IterSource(frames(1)), "bg", depth="always"))


def test_colour_is_never_resized_apart_from_its_depth(stub, monkeypatch):
    monkeypatch.setattr("visionserve.client._is_loopback", lambda host: False)
    c = Client(stub.url)
    src = IterSource(frames(1, 400, 300, depth=np.ones((300, 400), np.uint16), scale=0.001), has_depth=True)
    list(c.watch(src, "bg", resize=100))  # a forced shrink is overridden: depth is aligned to full size
    assert stub.requests[0]["size"] == (400, 300)
    stub.requests.clear()
    list(c.watch(IterSource(frames(1, 400, 300)), "det", resize=100))  # no depth: shrunk as asked
    assert stub.requests[0]["size"] == (100, 75)


def test_misaligned_depth_is_refused(stub):
    src = IterSource(frames(1, 40, 30, depth=np.ones((15, 20), np.uint16)), has_depth=True)
    with pytest.raises(ValueError, match="ALIGNED"):
        list(Client(stub.url).watch(src, "bg"))


def test_depth_units_on_the_wire(stub):
    """uint16 goes up unchanged (the server reads value/65535, 0 = no reading); float depth goes
    up as float32 METRES (x depth_scale) with NaN / inf / <= 0 as 0, the server's no-reading."""
    c = Client(stub.url)
    mm = np.array([[0, 1000], [65535, 1234]], np.uint16)
    list(c.watch(IterSource([Frame(np.zeros((2, 2, 3), np.uint8), mm, 0.001, frame_id=0)]), "bg"))
    f = stub.requests[-1]["fields"]
    assert f["depth_dtype"] == b"uint16"
    assert (np.frombuffer(f["depth"], "<u2").reshape(2, 2) == mm).all()

    metres = np.array([[np.nan, np.inf], [-1.0, 0.75]], np.float32)
    list(c.watch(IterSource([Frame(np.zeros((2, 2, 3), np.uint8), metres, frame_id=0)]), "bg"))
    f = stub.requests[-1]["fields"]
    assert f["depth_dtype"] == b"float32"
    assert np.frombuffer(f["depth"], "<f4").tolist() == [0.0, 0.0, 0.0, 0.75]

    mm_float = np.array([[500.0, 0.0]], np.float64)  # float millimetres with a scale
    sent = depth_for_upload(Frame(np.zeros((1, 2, 3), np.uint8), mm_float, 0.001))
    assert sent.dtype == np.float32 and sent.tolist() == [[0.5, 0.0]]


def test_frame_check_and_depth_meters():
    good = Frame(np.zeros((2, 3, 3), np.uint8), np.array([[0, 1000, 2000], [3, 4, 5]], np.uint16), 0.001)
    assert good.check() is good and good.width == 3 and good.height == 2 and good.has_depth
    m = good.depth_meters()
    assert np.isnan(m[0, 0]) and m[0, 1] == pytest.approx(1.0) and m.dtype == np.float32
    assert Frame(np.zeros((1, 1, 3), np.uint8), np.array([[np.inf]], np.float32)).depth_meters()[0, 0] != 0
    for bad in (
        Frame(np.zeros((2, 3), np.uint8)),
        Frame(np.zeros((2, 3, 3), np.float32)),
        Frame(np.zeros((2, 3, 3), np.uint8), np.zeros((2, 3), np.int32)),
        Frame(np.zeros((2, 3, 3), np.uint8), np.zeros((3, 2), np.uint16)),
        Frame(np.zeros((2, 3, 3), np.uint8), np.zeros((2, 3), np.uint16), depth_scale=0),
    ):
        with pytest.raises(ValueError):
            bad.check()


# --------------------------------------------------------------------------------------------
# tracker
# --------------------------------------------------------------------------------------------
def _dets(*boxes, cls="a"):
    return [Detection(bbox=list(b), cls=cls, conf=0.9) for b in boxes]


def test_tracker_keeps_ids_of_moving_objects_and_starts_new_ones():
    t = IoUTracker(iou_threshold=0.3, max_age=1)
    assert t.update(_dets([0, 0, 10, 10], [100, 100, 10, 10])) == [1, 2]
    assert t.update(_dets([102, 101, 10, 10], [2, 1, 10, 10])) == [2, 1]  # order of detections does not matter
    assert t.update(_dets([50, 50, 10, 10])) == [3]  # a new object
    assert t.update(_dets([3, 2, 10, 10])) == [1]  # missed once (max_age 1): still alive
    t.update([])
    t.update([])
    assert t.update(_dets([3, 2, 10, 10])) == [4]  # gone for longer than max_age: a new id


def test_tracker_matches_by_class_and_best_iou_first():
    t = IoUTracker()
    t.update(_dets([0, 0, 10, 10], cls="cat"))
    assert t.update(_dets([0, 0, 10, 10], cls="dog")) == [2]
    t = IoUTracker()
    t.update(_dets([0, 0, 10, 10]))
    ids = t.update(_dets([4, 0, 10, 10], [1, 0, 10, 10]))  # the closer box keeps the id
    assert ids == [2, 1]
    assert box_iou([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(50 / 150)
    with pytest.raises(ValueError):
        IoUTracker(iou_threshold=0)


def test_watch_track_adds_track_ids(stub):
    stub.box = lambda v: [10 + 2 * v, 10, 20, 20]  # one object drifting right
    # The stub names each detection after its frame, so match across classes here.
    tracker = IoUTracker(match_class=False)
    out = list(Client(stub.url).watch(IterSource(frames(5)), "det", track=tracker, drop_frames=False))
    assert [r.detections[0].track_id for r in out] == [1] * 5
    assert out[0].detections[0].to_json()["track_id"] == 1
    assert Detection.from_json(out[0].detections[0].to_json()).track_id == 1


# --------------------------------------------------------------------------------------------
# hints, open_source, CLI
# --------------------------------------------------------------------------------------------
def test_accepts_depth_hint(stub):
    c = Client(stub.url)
    assert c.accepts_depth("bg") is True and c.accepts_depth("det") is False and c.accepts_depth("old") is None
    assert [m.accepts_depth for m in c.list_models()] == [False, True, None]


def test_open_source_dispatch(monkeypatch, tmp_path):
    import visionserve.sources.gstreamer as gst
    import visionserve.sources.opencv as ocv
    import visionserve.sources.orbbec as ob
    import visionserve.sources.realsense as rs
    import visionserve.sources.ros2 as ros

    calls = []

    def fake(name):
        return lambda *a, **k: calls.append((name, a, k)) or name

    monkeypatch.setattr(ocv, "OpenCVSource", fake("cv"))
    monkeypatch.setattr(gst, "GStreamerSource", fake("gst"))
    monkeypatch.setattr(rs, "RealSenseSource", fake("rs"))
    monkeypatch.setattr(ob, "OrbbecSource", fake("ob"))
    monkeypatch.setattr(ros, "ROS2Source", fake("ros"))
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    assert open_source(0) == "cv" and calls[-1][1] == (0,)
    assert open_source(" 2 ") == "cv" and calls[-1][1] == (2,)
    assert open_source(str(video), realtime=False) == "cv" and calls[-1][2] == {"realtime": False}
    assert open_source(video) == "cv"
    assert open_source("rtsp://cam/stream") == "cv"
    assert open_source("gst:videotestsrc ! queue", width=8, height=8) == "gst" and calls[-1][1] == ("videotestsrc ! queue",)
    assert open_source("realsense") == "rs" and calls[-1][1] == (None,)
    assert open_source("realsense:0123") == "rs" and calls[-1][1] == ("0123",)
    assert open_source("orbbec:1") == "ob" and calls[-1][1] == (1,)
    assert open_source("ros2:/c/image_raw,/d/image_raw") == "ros" and calls[-1][1] == ("/c/image_raw", "/d/image_raw")
    assert open_source("ros2:/c/image_raw") == "ros" and calls[-1][1] == ("/c/image_raw", None)
    src = IterSource([])
    assert open_source(src) is src
    with pytest.raises(FileNotFoundError):
        open_source(str(tmp_path / "missing.mp4"))
    with pytest.raises(ValueError):
        open_source("orbbec:x")
    with pytest.raises(TypeError):
        open_source(3.5)
    with pytest.raises(TypeError):
        open_source(src, width=3)


def test_missing_vendor_package_names_the_extra(monkeypatch):
    import sys

    from visionserve.sources.opencv import OpenCVSource
    from visionserve.sources.orbbec import OrbbecSource
    from visionserve.sources.realsense import RealSenseSource
    from visionserve.sources.ros2 import ROS2Source

    for mod in ("cv2", "pyrealsense2", "pyorbbecsdk", "rclpy"):
        monkeypatch.setitem(sys.modules, mod, None)  # None in sys.modules = the import fails
    with pytest.raises(ImportError, match=r"visionserve\[opencv\]"):
        OpenCVSource(0)
    with pytest.raises(ImportError, match=r"visionserve\[realsense\]"):
        RealSenseSource()
    with pytest.raises(ImportError, match=r"visionserve\[orbbec\]"):
        OrbbecSource()
    with pytest.raises(ImportError, match="ROS 2"):
        ROS2Source("/image")


def test_cli_watch_prints_lines_and_json(stub, monkeypatch, capsys):
    from visionserve import cli

    monkeypatch.setattr(watch_mod, "_open_source", lambda spec: (IterSource(frames(3)), True))
    assert cli.main(["--host", stub.url, "watch", "cam", "--model", "det", "--every-frame", "--track"]) == 0
    out = capsys.readouterr()
    lines = out.out.strip().splitlines()
    assert len(lines) == 3 and lines[0].startswith("frame 0") and "#1" in lines[0]
    assert "3 frames" in out.err
    monkeypatch.setattr(watch_mod, "_open_source", lambda spec: (IterSource(frames(2)), True))
    assert cli.main(["--host", stub.url, "watch", "cam", "-m", "det", "--every-frame", "--json", "--max-frames", "1"]) == 0
    rows = [json.loads(x) for x in capsys.readouterr().out.strip().splitlines()]
    assert len(rows) == 1 and rows[0]["frame_id"] == 0 and rows[0]["detections"][0]["class"] == "0"
