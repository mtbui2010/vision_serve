"""Client-side resizing (Client(resize="auto", jpeg=True)): the scaling math, what is uploaded for
each input kind, prompt scaling, result mapping, and the passthrough cases. Offline: a stub
server answers /api/models with a hint and records each /api/predict upload."""
import http.server
import io
import json
import math
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from visionserve import Client, Grasp, Mask, ModelInfo, Result  # noqa: E402
from visionserve.client import _encode_image, _ndarray_to_pil  # noqa: E402
from visionserve.resize import (  # noqa: E402
    ClientResize,
    check_quality,
    check_resize,
    map_result,
    prepare_upload,
    roi_region,
    scale_points,
    target_size,
)


# --------------------------------------------------------------------------------------------
# stub server
# --------------------------------------------------------------------------------------------
def _parse_multipart(body, ctype):
    boundary = ctype.split("boundary=")[1].encode()
    out = {}
    for part in body.split(b"--" + boundary)[1:-1]:
        head, _, data = part[2:-2].partition(b"\r\n\r\n")  # strip the CRLFs around the part
        name = head.decode().split('name="')[1].split('"')[0]
        out[name] = data
    return out


class _Stub:
    """/api/models -> MODELS; /api/predict records the multipart fields and answers REPLY."""

    MODELS = [
        {"name": "rf-detr", "task": "detection", "license": "Apache-2.0", "state": "loaded",
         "max_useful_side": None, "max_useful_short_side": 1120},
        {"name": "rt-detr", "task": "detection", "license": "Apache-2.0", "state": "loaded",
         "max_useful_side": 1280, "max_useful_short_side": None},
        {"name": "mobile-sam", "task": "segmentation", "license": "Apache-2.0", "state": "loaded",
         "max_useful_side": None, "max_useful_short_side": None},
        {"name": "old", "task": "detection", "license": "MIT", "state": "loaded"},  # predates the hint
    ]

    def __init__(self, reply=None):
        self.reply = reply or {"task": "detection", "model": "m", "duration_ms": 1.0}
        self.models_calls = 0
        self.fields = None
        stub = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj):
                payload = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                stub.models_calls += 1
                self._send(stub.MODELS)

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                stub.fields = _parse_multipart(body, self.headers["Content-Type"])
                self._send(stub.reply)

        self.srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = "http://127.0.0.1:%d" % self.srv.server_address[1]

    def field(self, k):
        return self.fields[k].decode()

    def image(self):
        return Image.open(io.BytesIO(self.fields["image"]))

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def stub(monkeypatch):
    """A stub server the client treats as REMOTE (the loopback rule has its own tests)."""
    monkeypatch.setattr("visionserve.client._is_loopback", lambda host: False)
    s = _Stub()
    yield s
    s.close()


def _photo(w, h, fmt="JPEG", **kw):
    """A smooth colour gradient (JPEG-friendly) as encoded bytes."""
    ys, xs = np.mgrid[0:h, 0:w]
    a = np.stack([xs * 255 // max(w - 1, 1), ys * 255 // max(h - 1, 1), (xs + ys) % 256], -1).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, fmt, **kw)
    return buf.getvalue()


def _prep(image, **kw):
    kw.setdefault("max_side", None)
    kw.setdefault("max_short_side", None)
    kw.setdefault("jpeg", True)
    kw.setdefault("quality", 90)
    return prepare_upload(image, encode_plain=_encode_image, ndarray_to_pil=_ndarray_to_pil, **kw)


# --------------------------------------------------------------------------------------------
# scaling math
# --------------------------------------------------------------------------------------------
def test_target_size_short_and_long_side():
    assert target_size(4000, 3000, max_short_side=1120) == (1493, 1120)  # 4000*1120/3000 = 1493.3
    assert target_size(3000, 4000, max_short_side=1120) == (1120, 1493)
    assert target_size(4000, 3000, max_side=1280) == (1280, 960)
    # a panorama keeps its rows under the short-side rule (the model squashes both axes)
    assert target_size(10000, 1000, max_short_side=1120) == (10000, 1000)
    assert target_size(10000, 1000, max_side=1280) == (1280, 128)


def test_target_size_never_enlarges_and_rounds_half_up():
    assert target_size(640, 480, max_short_side=1120) == (640, 480)
    assert target_size(1120, 1500, max_short_side=1120) == (1120, 1500)  # exactly at the bound
    assert target_size(10, 5, max_side=3) == (3, 2)  # 5 * 0.3 = 1.5 -> 2 (half up)
    assert target_size(10, 5, max_side=4) == (4, 2)
    assert target_size(1000, 1, max_side=10) == (10, 1)  # never below 1 px
    assert target_size(100, 100) == (100, 100)  # no hint


def test_target_size_bounds_the_roi_not_the_image():
    # a 1000 px ROI of a 4000x3000 photo: the model sees the ROI, so the bound applies to it
    assert target_size(4000, 3000, max_short_side=1120, region=(1000, 1000)) == (4000, 3000)
    assert target_size(4000, 3000, max_short_side=500, region=(1000, 1000)) == (2000, 1500)


def test_roi_region_follows_the_servers_clamp():
    assert roi_region([100, 50, 300, 200], 4000, 3000) == (300, 200)
    assert roi_region([0.5, 0.5, 0.25, 0.5], 4000, 3000) == (1000, 1500)  # fractions
    assert roi_region([3900, 0, 500, 100], 4000, 3000) == (100, 100)  # clamped to the image
    assert roi_region([0, 0, 0, 10], 4000, 3000) is None  # degenerate: whole image
    assert roi_region(None, 4000, 3000) is None


def test_option_validation():
    assert check_resize("auto") == "auto" and check_resize("OFF") == "off"
    assert check_resize(1280) == 1280 and check_resize("1280") == 1280
    for bad in (0, -5, True, 1.5, "big", None):
        with pytest.raises(ValueError):
            check_resize(bad)
    assert check_quality(90) == 90
    for bad in (0, 101, 9.5, True):
        with pytest.raises(ValueError):
            check_quality(bad)
    with pytest.raises(ValueError):
        Client(resize="sometimes")


# --------------------------------------------------------------------------------------------
# what is uploaded
# --------------------------------------------------------------------------------------------
def test_no_hint_sends_the_input_exactly_as_before(tmp_path):
    big = _photo(3000, 2000)
    assert _prep(big) == (big, "image.png", None)
    p = tmp_path / "x.jpg"
    p.write_bytes(big)
    assert _prep(str(p)) == (big, "x.jpg", None)
    arr = np.zeros((40, 60, 3), np.uint8)
    data, name, cr = _prep(arr)
    assert cr is None and (data, name) == _encode_image(arr)  # lossless PNG, as before


def test_small_jpeg_is_sent_untouched():
    small = _photo(640, 480)
    assert _prep(small, max_short_side=1120) == (small, "image.png", None)


def test_large_jpeg_is_shrunk_to_the_hint_as_jpeg():
    data, name, cr = _prep(_photo(3000, 2000), max_short_side=1120)
    sent = Image.open(io.BytesIO(data))
    assert sent.format == "JPEG" and sent.size == (1680, 1120) and name == "image.jpg"
    assert cr == ClientResize(3000, 2000, 1680, 1120, 90) and cr.resized
    assert cr.scale_x == pytest.approx(0.56) and cr.scale_y == pytest.approx(0.56)


def test_jpeg_false_sends_png_and_leaves_small_inputs_alone():
    small_png = _photo(300, 200, "PNG")
    assert _prep(small_png, max_side=1000, jpeg=False) == (small_png, "image.png", None)
    data, _, cr = _prep(_photo(3000, 2000), max_side=1000, jpeg=False)
    sent = Image.open(io.BytesIO(data))
    assert sent.format == "PNG" and sent.size == (1000, 667) and cr.jpeg_quality is None


def test_what_is_not_shrunk_goes_out_as_before():
    small_png = _photo(300, 200, "PNG")
    assert _prep(small_png, max_side=1000) == (small_png, "image.png", None)  # not re-encoded
    arr = np.zeros((40, 60, 3), np.uint8)
    assert _prep(arr, max_side=1000) == (*_encode_image(arr), None)  # lossless PNG, as before
    pil = Image.fromarray(arr)
    assert _prep(pil, max_side=1000) == (*_encode_image(pil), None)


def test_exif_rotation_is_applied_before_resizing():
    # Stored 3000x2000 with the left half red; orientation 6 = rotate 90 degrees clockwise to
    # display, so the server would see 2000x3000 with the red half at the TOP.
    a = np.zeros((2000, 3000, 3), np.uint8)
    a[:, :1500] = (255, 0, 0)
    a[:, 1500:] = (0, 0, 255)
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", exif=exif, quality=95)
    data, _, cr = _prep(buf.getvalue(), max_short_side=1000)
    sent = np.asarray(Image.open(io.BytesIO(data)))
    assert (cr.original_width, cr.original_height) == (2000, 3000)
    assert sent.shape[:2] == (1500, 1000)
    assert sent[100, 500, 0] > 200 and sent[100, 500, 2] < 60  # top: red
    assert sent[1400, 500, 2] > 200 and sent[1400, 500, 0] < 60  # bottom: blue
    assert Image.open(io.BytesIO(data)).getexif().get(0x0112) in (None, 1)  # not rotated twice


def test_exif_orientation_of_a_small_jpeg_is_left_to_the_server():
    exif = Image.Exif()
    exif[0x0112] = 6
    raw = _photo(400, 300, exif=exif)
    assert _prep(raw, max_short_side=1000) == (raw, "image.png", None)


def test_png_alpha_is_dropped_keeping_the_colour():
    a = np.zeros((2000, 2000, 4), np.uint8)
    a[..., :3] = (10, 200, 30)
    a[..., 3] = 0  # fully transparent: the server's tensor reads the RGB anyway
    buf = io.BytesIO()
    Image.fromarray(a, "RGBA").save(buf, "PNG")
    data, _, cr = _prep(buf.getvalue(), max_side=500, jpeg=False)
    sent = Image.open(io.BytesIO(data))
    assert sent.mode == "RGB" and sent.size == (500, 500)
    assert np.asarray(sent)[250, 250].tolist() == [10, 200, 30]


def test_16bit_gray_png_keeps_the_high_byte_like_the_server():
    a = np.full((1200, 1200), 0x8040, np.uint16)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    data, _, cr = _prep(buf.getvalue(), max_side=600, jpeg=False)
    assert np.asarray(Image.open(io.BytesIO(data)))[300, 300].tolist() == [0x80] * 3


def test_pil_and_ndarray_inputs_are_resized():
    arr = np.zeros((2000, 3000, 3), np.uint8)
    for img in (arr, Image.fromarray(arr)):
        data, _, cr = _prep(img, max_side=600)
        assert Image.open(io.BytesIO(data)).size == (600, 400) and cr.resized


def test_unreadable_bytes_go_out_untouched():
    junk = b"not an image at all"
    assert _prep(junk, max_side=100) == (junk, "image.png", None)


# --------------------------------------------------------------------------------------------
# prompts and results
# --------------------------------------------------------------------------------------------
def test_max_scale_keeps_a_mild_shrink_whole():
    raw = _photo(3000, 2000)
    # 1120 / 2000 = 0.56 > 0.5: sent as given, and the record says why
    data, name, cr = _prep(raw, max_short_side=1120, max_scale=0.5)
    assert data == raw and not cr.resized and cr.jpeg_quality is None
    assert (cr.original_width, cr.sent_width) == (3000, 3000)
    assert cr.reason == "loopback: scale 0.56 > 0.5, sent as is"
    # 900 / 2000 = 0.45: shrunk
    data, _, cr = _prep(raw, max_short_side=900, max_scale=0.5)
    assert cr.resized and cr.sent_height == 900 and cr.reason == "hint"
    arr = np.zeros((2000, 3000, 3), np.uint8)
    data, _, cr = _prep(arr, max_short_side=1120, max_scale=0.5)
    assert (data, "image.png") == _encode_image(arr) and cr.reason.startswith("loopback")


def test_loopback_rule_through_the_client():
    s = _Stub()
    try:
        c = Client(s.url)  # 127.0.0.1: the loopback rule applies
        big = _photo(3000, 2000)
        res = c.predict("rf-detr", big)  # hint 1120 short: scale 0.56 > 0.5
        assert s.fields["image"] == big and not res.client_resize.resized
        assert res.client_resize.reason.startswith("loopback")
        res = c.predict("rf-detr", _photo(4000, 2400))  # 1120 / 2400 = 0.47: shrunk
        assert s.image().size == (1867, 1120) and res.client_resize.reason == "hint"
        res = c.predict("rf-detr", big, resize=1500)  # an explicit size ignores the rule
        assert s.image().size == (1500, 1000) and res.client_resize.reason == "resize=1500"
    finally:
        s.close()


def test_points_keep_their_label():
    cr = ClientResize(1000, 500, 500, 250, 90)
    assert scale_points([[100, 50, 0], [10, 20]], cr) == [[50, 25, 0], [5, 10]]


def test_map_result_back_to_original_pixels():
    cr = ClientResize(4000, 3000, 1493, 1120, 90)
    sx, sy = 1493 / 4000, 1120 / 3000
    theta = 0.7
    res = Result(task="detection", model="m",
                 detections=[_det([149.3, 112.0, 14.93, 11.2])],
                 masks=[Mask(rle="0 %d" % (1493 * 1120), bbox=[149.3, 112.0, 14.93, 11.2], conf=0.9)],
                 grasps=[Grasp(x=149.3, y=112.0, theta=theta, width=10.0, quality=0.5)])
    map_result(res, cr)
    assert res.detections[0].bbox == pytest.approx([400, 300, 40, 30])
    assert res.masks[0].bbox == pytest.approx([400, 300, 40, 30])
    g = res.grasps[0]
    assert (g.x, g.y) == pytest.approx((400, 300))
    # the jaw vector (10 cos t, 10 sin t) in sent pixels, mapped per axis
    vx, vy = 10 * math.cos(theta) / sx, 10 * math.sin(theta) / sy
    assert g.width == pytest.approx(math.hypot(vx, vy)) and g.theta == pytest.approx(math.atan2(vy, vx))
    assert res.client_resize is cr
    # the client-side record does not leak into the wire form or equality
    assert "client_resize" not in res.to_json()
    assert Result.from_json(res.to_json()) == res


def _det(bbox):
    from visionserve import Detection

    return Detection(bbox=bbox, cls="cat", conf=0.9)


def test_mask_decoded_at_the_sent_size_returns_the_original_size():
    # sent 4x2, foreground = the right half (columns 2..3); original 8x4.
    sent = np.zeros((2, 4), bool)
    sent[:, 2:] = True
    flat = sent.flatten(order="F")
    runs, cur, n = [], False, 0
    for v in flat:
        if v == cur:
            n += 1
        else:
            runs.append(n)
            cur, n = v, 1
    runs.append(n)
    m = Mask(rle=" ".join(map(str, runs)), bbox=[2, 0, 2, 2], conf=1.0)
    map_result(Result(task="segmentation", model="m", masks=[m]), ClientResize(8, 4, 4, 2, 90))
    full = m.to_ndarray(8, 4)
    assert full.shape == (4, 8) and full[:, 4:].all() and not full[:, :4].any()
    assert m.to_ndarray(4, 2).tolist() == sent.tolist()  # the sent size still decodes as received
    assert m.bbox == [4, 0, 4, 4]


# --------------------------------------------------------------------------------------------
# through the client
# --------------------------------------------------------------------------------------------
def test_predict_shrinks_scales_prompts_and_maps_back(stub):
    stub.reply = {"task": "detection", "model": "rf-detr", "duration_ms": 1.0,
                  "detections": [{"bbox": [56, 56, 112, 56], "class": "cat", "conf": 0.9}]}
    c = Client(stub.url)
    res = c.predict("rf-detr", _photo(3000, 2000), box=[300, 300, 600, 300], point=[[100, 200, 1]],
                    roi=[0, 0, 3000, 2000])
    assert stub.image().size == (1680, 1120)
    assert stub.field("box") == "168,168,336,168"
    assert [float(v) for v in stub.field("point").split(",")] == pytest.approx([56, 112, 1])
    assert [float(v) for v in stub.field("roi").split(",")] == pytest.approx([0, 0, 1680, 1120])
    assert res.detections[0].bbox == pytest.approx([100, 100, 200, 100])
    assert res.client_resize == ClientResize(3000, 2000, 1680, 1120, 90)


def test_hint_is_fetched_once_and_refreshed_for_an_unknown_model(stub):
    c = Client(stub.url)
    c.predict("rf-detr", _photo(64, 48))
    c.predict("rt-detr", _photo(64, 48))
    assert stub.models_calls == 1
    assert c.useful_side("rt-detr") == (1280, None)
    c._hints_fetched = float("-inf")  # past the refresh delay
    assert c.useful_side("not-listed") == (None, None)
    assert stub.models_calls == 2


def test_list_models_reads_the_hint(stub):
    infos = {m.name: m for m in Client(stub.url).list_models()}
    assert infos["rf-detr"] == ModelInfo("rf-detr", "detection", "Apache-2.0", "loaded", None, 1120)
    assert infos["old"].max_useful_side is None and infos["old"].max_useful_short_side is None


def test_models_without_a_hint_get_the_original_bytes(stub):
    big = _photo(3000, 2000)
    for model in ("mobile-sam", "old", "unknown-model"):
        res = Client(stub.url).predict(model, big)
        assert stub.fields["image"] == big and res.client_resize is None


def test_resize_off_and_per_call_overrides(stub):
    big = _photo(3000, 2000)
    Client(stub.url, resize="off").predict("rf-detr", big)
    assert stub.fields["image"] == big and stub.models_calls == 0  # no listing needed
    c = Client(stub.url)
    c.predict("rf-detr", big, resize="off")
    assert stub.fields["image"] == big
    c.predict("rf-detr", big, resize=500, jpeg=False)
    assert stub.image().size == (500, 333) and stub.image().format == "PNG"
    res = Client(stub.url, resize=1000, jpeg_quality=70).predict("mobile-sam", big)  # int: any model
    assert stub.image().size == (1000, 667) and res.client_resize.jpeg_quality == 70


def test_full_resolution_options_disable_resizing(stub):
    big = _photo(3000, 2000)
    c = Client(stub.url)
    for kw in ({"depth": np.zeros((2000, 3000), np.uint16)}, {"dilate": 3}, {"gripper_max": 80.0},
               {"template_name": "mug"}):
        res = c.predict("rf-detr", big, **kw)
        assert stub.fields["image"] == big and res.client_resize is None, kw


def test_listing_failure_means_full_resolution(stub):
    stub.MODELS = "not a list"
    big = _photo(3000, 2000)
    res = Client(stub.url).predict("rf-detr", big)
    assert stub.fields["image"] == big and res.client_resize is None


def test_preprocess_sends_the_image_as_given_by_default(stub):
    stub.reply = {"model": "rf-detr", "inputs": [], "meta": None}
    big = _photo(3000, 2000)
    c = Client(stub.url)
    res = c.preprocess("rf-detr", big)
    assert stub.fields["image"] == big and res.client_resize is None
    res = c.preprocess("rf-detr", big, resize="auto", box=[300, 300, 600, 300])
    assert stub.image().size == (1680, 1120) and stub.field("box") == "168,168,336,168"
    assert res.client_resize.sent_width == 1680


def test_cli_flags_reach_the_upload(stub, tmp_path, capsys):
    from visionserve import cli as vs_cli

    p = tmp_path / "big.jpg"
    p.write_bytes(_photo(3000, 2000))
    rc = vs_cli.main(["--host", stub.url, "predict", "rf-detr", str(p), "--compact"])
    assert rc == 0 and stub.image().size == (1680, 1120)
    assert "sent 1680x1120 of 3000x2000 as JPEG q90" in capsys.readouterr().err
    rc = vs_cli.main(["--host", stub.url, "predict", "rf-detr", str(p), "--resize", "800", "--no-jpeg", "--quiet"])
    assert rc == 0 and stub.image().size == (800, 533) and stub.image().format == "PNG"
    rc = vs_cli.main(["--host", stub.url, "predict", "rf-detr", str(p), "--resize", "off", "--quiet"])
    assert rc == 0 and stub.fields["image"] == p.read_bytes()
    with pytest.raises(SystemExit):
        vs_cli.main(["--host", stub.url, "predict", "rf-detr", str(p), "--resize", "sometimes"])
    rc = vs_cli.main(["--host", stub.url, "predict", "rf-detr", str(p), "--jpeg-quality", "0", "--quiet"])
    assert rc == 1 and "jpeg_quality" in capsys.readouterr().err
