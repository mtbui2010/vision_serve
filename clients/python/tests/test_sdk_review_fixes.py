"""Regression tests for the SDK review findings (S1-S7 + perf). Offline: a stub HTTP server or
direct calls; numpy + pillow required."""
import http.server
import io
import json
import os
import socket
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from visionserve import Client, Detection, Mask, Result, VisionServeError  # noqa: E402
from visionserve import postprocess as pp  # noqa: E402
from visionserve.client import _build_multipart, _encode_depth, _encode_image, _serialize_boxes  # noqa: E402
from visionserve.client import _serialize_points  # noqa: E402


# --------------------------------------------------------------------------------------------
# a server that records the multipart fields it receives
# --------------------------------------------------------------------------------------------

def _parse_multipart(body: bytes, ctype: str):
    boundary = ctype.split("boundary=")[1].encode()
    out = {}
    for part in body.split(b"--" + boundary)[1:-1]:
        head, _, data = part.strip(b"\r\n").partition(b"\r\n\r\n")
        h = head.decode()
        name = h.split('name="')[1].split('"')[0]
        out[name] = (h, data)
    return out


class _Recorder:
    def __init__(self, reply=None, mode="ok"):
        self.reply = reply or {"task": "detection", "model": "m", "duration_ms": 1.0}
        self.mode = mode
        self.fields = None

        rec = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                rec.fields = _parse_multipart(body, self.headers["Content-Type"])
                if rec.mode == "disconnect":
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
                if rec.mode in ("busy", "bad"):  # the server's 503 + Retry-After, and a plain 400
                    payload = json.dumps({"error": "queue full" if rec.mode == "busy" else "bad box"}).encode()
                    self.send_response(503 if rec.mode == "busy" else 400)
                    if rec.mode == "busy":
                        self.send_header("Retry-After", "1")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                payload = json.dumps(rec.reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                if rec.mode == "short":  # promise more than we send -> IncompleteRead
                    self.send_header("Content-Length", str(len(payload) + 100))
                    self.end_headers()
                    self.wfile.write(payload[:10])
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = do_POST

        self.srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = "http://127.0.0.1:%d" % self.srv.server_address[1]

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def rec():
    r = _Recorder()
    yield r
    r.close()


def _img():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)


def _field(rec, k):
    return rec.fields[k][1].decode()


# --------------------------------------------------------------------------------------------
# S1 depth helpers
# --------------------------------------------------------------------------------------------

def _depth_result(w, h, fn):
    ys, xs = np.mgrid[0:h, 0:w]
    m = fn(xs, ys).astype(np.float32)
    return Result(task="depth", model="midas", depth_map=m.ravel().tolist(), depth_width=w, depth_height=h)


def test_s1_relative_depth_result_maps_boxes_to_model_resolution():
    # image 640x480, model map 64x48: left half 0.2, right half 0.9 (relative, larger = closer)
    depth = _depth_result(64, 48, lambda x, y: np.where(x < 32, 0.2, 0.9))
    det = Result(task="detection", model="d", detections=[Detection([400, 100, 200, 300], "cup", 0.9)])
    # the box lies in the RIGHT half of the image; with unscaled indexing it falls off the 64x48 map
    assert pp.get_depth_at_detection(depth, det, image_size=(640, 480)) == [pytest.approx(0.9)]
    with pytest.raises(ValueError, match="image_size"):
        pp.get_depth_at_detection(depth, det)
    with pytest.raises(ValueError, match="relative"):
        pp.get_depth_at_detection(depth, det, image_size=(640, 480), depth_scale=1.0)


def test_s1_metric_helpers_refuse_relative_depth():
    depth = _depth_result(8, 6, lambda x, y: x / 7.0)
    det = Result(task="detection", model="d", detections=[Detection([1, 1, 2, 2], "cup", 0.9)])
    from visionserve.types import Grasp
    with pytest.raises(ValueError, match="RELATIVE"):
        pp.object_distances(depth, det, [500, 500, 4, 3])
    with pytest.raises(ValueError, match="RELATIVE"):
        pp.grasp_distances(depth, [Grasp(2, 2, 0, 5, 0.5)], [500, 500, 4, 3])
    with pytest.raises(ValueError, match="RELATIVE"):
        pp.select_target_object(det, depth_result=depth, intrinsics=[500, 500, 4, 3], target_distance=0.5)
    with pytest.raises(ValueError, match="RELATIVE"):
        pp.select_target_grasp([Grasp(2, 2, 0, 5, 0.5)], depth_result=depth, intrinsics=[500, 500, 4, 3],
                               target_distance=0.5)


def test_s1_metric_sensor_array_still_works_and_can_be_scaled():
    mm = np.full((48, 64), 1000, np.uint16)
    mm[:, 32:] = 2000
    det = Result(task="detection", model="d", detections=[Detection([40, 10, 10, 10], "cup", 0.9)])
    assert pp.get_depth_at_detection(mm, det) == [pytest.approx(2.0)]
    # a half-resolution sensor array, boxes in a 128x96 image
    det2 = Result(task="detection", model="d", detections=[Detection([80, 20, 20, 20], "cup", 0.9)])
    assert pp.get_depth_at_detection(mm, det2, image_size=(128, 96)) == [pytest.approx(2.0)]
    d = pp.object_distances(mm, det, [500, 500, 32, 24])
    assert d[0] == pytest.approx(2.0, rel=0.05)


# --------------------------------------------------------------------------------------------
# S2 numpy boxes / points / numbers
# --------------------------------------------------------------------------------------------

def test_s2_numpy_boxes_points_and_scalars(rec):
    assert _serialize_boxes(np.array([1, 2, 3, 4])) == "1,2,3,4"
    assert _serialize_boxes(np.array([[1.5, 2, 3, 4], [5, 6, 7, 8]], np.float32)) == "1.5,2,3,4;5,6,7,8"
    assert _serialize_boxes([np.float64(1.5), np.float32(2.0), np.int64(3), 4]) == "1.5,2,3,4"
    assert _serialize_points(np.array([[10, 20, 1], [30, 40, 0]])) == "10,20,1;30,40,0"
    Client(rec.url).predict("m", _img(), box=np.array([1, 2, 3, 4]), box_threshold=np.float64(0.35),
                            roi=np.array([0.1, 0.2, 0.5, 0.5]))
    assert _field(rec, "box") == "1,2,3,4"
    assert _field(rec, "box_threshold") == "0.35"
    assert _field(rec, "roi") == "0.1,0.2,0.5,0.5"


# --------------------------------------------------------------------------------------------
# S3 network errors
# --------------------------------------------------------------------------------------------

def test_s3_timeout_is_a_visionserve_error():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)  # accepts the TCP connection, never answers
    try:
        with pytest.raises(VisionServeError):
            Client("http://127.0.0.1:%d" % s.getsockname()[1], timeout=0.3).health()
    finally:
        s.close()


@pytest.mark.parametrize("mode", ["disconnect", "short"])
def test_s3_dropped_connection_is_a_visionserve_error(mode):
    r = _Recorder(mode=mode)
    try:
        with pytest.raises(VisionServeError):
            Client(r.url, timeout=5).predict("m", _img())
    finally:
        r.close()


@pytest.mark.parametrize("mode,status,retry_after", [("busy", 503, 1.0), ("bad", 400, None)])
def test_s3_status_and_retry_after_on_the_error(mode, status, retry_after):
    r = _Recorder(mode=mode)
    try:
        with pytest.raises(VisionServeError) as ei:
            Client(r.url, timeout=5).predict("m", _img())
        assert ei.value.status == status
        assert ei.value.retry_after == retry_after
    finally:
        r.close()


def test_s3_python_m_visionserve_exits_nonzero_on_error():
    import subprocess
    root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens there: the request fails to connect
    p = subprocess.run([sys.executable, "-m", "visionserve", "--host", "http://127.0.0.1:%d" % port, "health"],
                       env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 1, p.stderr
    assert "error:" in p.stderr
    p = subprocess.run([sys.executable, "-m", "visionserve", "--version"], env=env, capture_output=True,
                       text=True, timeout=60)
    assert p.returncode == 0


def test_s3_retry_after_parsing():
    from visionserve.client import _retry_after
    assert _retry_after({"Retry-After": "2"}) == 2.0
    for bad in ({}, {"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}, {"Retry-After": "-1"},
                {"Retry-After": "nan"}, {"Retry-After": "inf"}, None):
        assert _retry_after(bad) is None
    assert VisionServeError("x").retry_after is None


# --------------------------------------------------------------------------------------------
# S4 ndarray images are lossless
# --------------------------------------------------------------------------------------------

def test_s4_ndarray_sent_lossless_png():
    a = _img()
    data, name = _encode_image(a)
    assert name == "image.png" and data[:8] == b"\x89PNG\r\n\x1a\n"
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(data)).convert("RGB")), a)


# --------------------------------------------------------------------------------------------
# S5 group_by_class with masks
# --------------------------------------------------------------------------------------------

def test_s5_group_by_class_with_masks():
    r = Result(task="open_vocab", model="grounded-sam",
               detections=[Detection([1, 2, 3, 4], "cup", 0.9), Detection([5, 6, 7, 8], "mug", 0.8)],
               masks=[Mask("0 1", [1, 2, 3, 4], 0.9), Mask("0 1", [5, 6, 7, 8], 0.8), Mask("0 1", [0, 0, 1, 1], 0.5)])
    g = r.group_by_class()
    assert sorted(g) == ["", "cup", "mug"]
    assert len(g["cup"].masks) == 1 and g["cup"].masks[0].bbox == [1, 2, 3, 4]
    assert len(g[""].masks) == 1 and not g[""].detections
    sam_only = Result(task="segmentation", model="mobile-sam", masks=[Mask("0 1", [0, 0, 1, 1], 0.9)])
    assert list(sam_only.group_by_class()) == [""]


# --------------------------------------------------------------------------------------------
# S6 one prompt rule for predict / preprocess / tokenize
# --------------------------------------------------------------------------------------------

def test_s6_prompt_normalised_the_same_everywhere(rec):
    rec.reply = {"model": "m", "inputs": [], "meta": None}
    c = Client(rec.url)
    c.predict("grounding-dino", _img(), prompt="cat, remote")
    sent_predict = _field(rec, "prompt")
    c.preprocess("grounding-dino", _img(), prompt="cat, remote")
    assert _field(rec, "prompt") == sent_predict == "cat. remote"
    c.preprocess("grounding-dino", _img())  # no prompt: the same default predict() sends
    assert _field(rec, "prompt") == "object."


def test_s6_clip_and_siglip_labels_keep_their_commas(rec):
    c = Client(rec.url)
    c.predict("clip", _img(), prompt="a photo of a cat, sitting. a dog")
    assert _field(rec, "prompt") == "a photo of a cat, sitting. a dog"
    from visionserve.client import normalize_prompt
    assert normalize_prompt("siglip-text", "red, round fruit") == "red, round fruit"
    assert normalize_prompt("rfdetr-gdino", "cat, dog") == "cat. dog"


# --------------------------------------------------------------------------------------------
# S7
# --------------------------------------------------------------------------------------------

def test_s7_quote_in_filename_does_not_break_the_header(tmp_path):
    p = tmp_path / 'a"b.png'
    Image.fromarray(_img()).save(p)
    data, name = _encode_image(p)
    body, ctype = _build_multipart({"model": "m"}, data, name)
    h, payload = _parse_multipart(body, ctype)["image"]
    assert 'filename="a_b.png"' in h and payload == data


def test_s7_int_fields_sent_as_ints_and_new_fields(rec):
    Client(rec.url).predict("rfdetr-textalign", _img(), dilate=2.0, grid_size=np.int64(16), claim_threshold=0.4,
                            crop_temp=0.02, template_name="shelf")
    assert _field(rec, "dilate") == "2" and _field(rec, "grid_size") == "16"
    assert _field(rec, "claim_threshold") == "0.4" and _field(rec, "crop_temp") == "0.02"
    assert _field(rec, "template_name") == "shelf"
    with pytest.raises(ValueError):
        Client(rec.url).predict("m", _img(), dilate=2.5)


def test_s7_hint_kept():
    r = Result.from_json({"task": "detection", "model": "m", "hint": "install TensorRT"})
    assert r.hint == "install TensorRT"


def test_s7_gdino_siglip_gets_the_default_prompt(rec):
    Client(rec.url).predict("gdino-siglip", _img())
    assert _field(rec, "prompt") == "object."
    Client(rec.url).predict("rfdetr-gdino-siglip", _img())  # no prompt = everything RF-DETR knows
    assert "prompt" not in rec.fields


def test_s7_uint16_depth_overflow_refused():
    with pytest.raises(ValueError, match="65535"):
        _encode_depth(np.array([[0, 70000]], np.int32))
    with pytest.raises(ValueError, match="65535"):
        _encode_depth(np.array([[-1, 5]], np.int16))
    raw, h, w, dt = _encode_depth(np.array([[0, 65535]], np.int32))
    assert dt == "uint16" and np.frombuffer(raw, "<u2").tolist() == [0, 65535]


def test_s7_draw_accepts_pathlib_and_single_channel(tmp_path):
    from visionserve.visualize import draw
    p = tmp_path / "x.png"
    Image.fromarray(_img()).save(p)
    r = Result(task="detection", model="m", detections=[Detection([1, 1, 5, 5], "cup", 0.9)])
    assert draw(r, p).size == (32, 24)
    assert draw(r, _img()[:, :, :1]).size == (32, 24)


def test_s7_utils_imports_without_cv2_and_helpers_exist(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)  # simulate "OpenCV not installed"
    sys.modules.pop("visionserve.utils", None)
    import visionserve.utils as u
    assert len(u.n2colormap(3)) == 3 and u._as_numpy([1, 2]).tolist() == [1, 2]
    with pytest.raises(ImportError, match="OpenCV"):
        u.show_mask_on_rgb(np.zeros((4, 4, 3), np.uint8), np.ones((4, 4)))


def test_s7_version_single_source():
    import re

    import visionserve
    toml = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', toml, re.M)
    if m:  # a static version must match the package
        assert m.group(1) == visionserve.__version__
    else:
        assert 'attr = "visionserve.__version__"' in toml


# --------------------------------------------------------------------------------------------
# perf rewrites keep the exact behaviour
# --------------------------------------------------------------------------------------------

def test_perf_rle_numpy_matches_reference():
    rng = np.random.default_rng(1)
    m = rng.random((13, 17)) > 0.6
    flat = m.ravel(order="F")
    counts, prev, run = [], False, 0
    for v in flat:
        if v == prev:
            run += 1
        else:
            counts.append(run)
            prev, run = v, 1
    counts.append(run)
    assert np.array_equal(Mask(" ".join(map(str, counts)), [0, 0, 1, 1], 1).to_ndarray(17, 13), m)
    with pytest.raises(ValueError):
        Mask("3 -1 219", [0, 0, 1, 1], 1).to_ndarray(17, 13)


def test_perf_depth_colormap_matches_scalar_ramp():
    from visionserve.visualize import _draw_depth, _turbo_colour
    vals = np.linspace(0, 1, 97).tolist()
    r = Result(task="depth", model="d", depth_map=vals, depth_width=97, depth_height=1)
    got = np.asarray(_draw_depth(r, Image))[0]
    want = np.array([_turbo_colour(v) for v in vals], np.uint8)
    assert np.array_equal(got, want)
    assert r.depth_array().dtype == np.float32 and r.depth_array().shape == (1, 97)
