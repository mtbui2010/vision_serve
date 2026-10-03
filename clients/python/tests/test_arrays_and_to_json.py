"""SDK tests for the binary-friendly array encoding, Result.to_json, and the consolidated
grasp / ndarray helpers.

The server side of ``encoding=base64`` is pkg/api/encoding.go: depth_map_base64 /
embeddings_base64 (+ embeddings_shape) carry the row-major little-endian float32 bytes.
"""

import base64
import builtins
import dataclasses
import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visionserve import Client, FloatArray, Result  # noqa: E402
from visionserve import cli as vs_cli  # noqa: E402
from visionserve import types as vs_types  # noqa: E402
from visionserve.client import _ndarray_to_png  # noqa: E402
from visionserve.types import Classification, Detection, Grasp, Mask  # noqa: E402

# Values a float32 round trip must keep bit for bit, including those JSON numbers cannot carry.
AWKWARD = np.array(
    [0.0, -0.0, 0.1, 1 / 3, 3.4e38, -3.4e38, 1e-45, 1e-40, np.inf, -np.inf, np.nan, 0.5],
    dtype=np.float32,
)


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def same_bits(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


# --------------------------------------------------------------------------- #
# base64 decode
# --------------------------------------------------------------------------- #
def test_base64_depth_decodes_bit_exact_and_zero_copy():
    depth = AWKWARD.reshape(3, 4)
    res = Result.from_json({"task": "depth", "model": "midas", "depth_map_base64": b64(depth),
                            "depth_width": 4, "depth_height": 3, "duration_ms": 1.5})
    assert isinstance(res.depth_map, FloatArray)
    arr = res.depth_array()
    assert arr.shape == (3, 4) and arr.dtype == np.float32
    assert same_bits(arr, depth)
    assert np.shares_memory(arr, res.depth_map.array), "depth_array() must not copy"
    arr[0, 0] = 7.0  # writable (not a read-only view of the decoded bytes)


def test_float_array_behaves_like_the_json_list():
    vals = np.array([0.5, 0.25, -1.0], np.float32)
    res = Result.from_json({"task": "depth", "model": "m", "depth_map_base64": b64(vals),
                            "depth_width": 3, "depth_height": 1})
    d = res.depth_map
    plain = Result.from_json({"task": "depth", "model": "m", "depth_map": vals.tolist(),
                              "depth_width": 3, "depth_height": 1})
    # Everything list code does keeps working.
    assert len(d) == 3 and bool(d) and d[1] == 0.25 and isinstance(d[0], float)
    assert list(d) == [0.5, 0.25, -1.0] and d == [0.5, 0.25, -1.0] and d[1:] == [0.25, -1.0]
    assert max(d) == 0.5 and 0.25 in d
    assert res == plain, "a base64 Result equals the JSON-number Result with the same values"
    assert not FloatArray(np.zeros(0, np.float32))


def test_base64_embeddings_decode_to_rows():
    emb = AWKWARD.reshape(2, 6)
    res = Result.from_json({"task": "embed", "model": "clip", "embeddings_base64": b64(emb),
                            "embeddings_shape": [2, 6]})
    assert len(res.embeddings) == 2 and len(res.embeddings[0]) == 6
    assert same_bits(res.embeddings_array(), emb)
    assert np.shares_memory(res.embeddings_array(), res.embeddings.array)
    assert same_bits(np.asarray(res.embeddings[1]), emb[1])  # how convert/verify.py reads a row
    # The JSON-number form gives the same arrays through the same accessors.
    plain = Result.from_json({"task": "embed", "model": "clip", "embeddings": [[0.5, 0.25]]})
    assert plain.embeddings_array().tolist() == [[0.5, 0.25]]
    assert Result.from_json({"task": "embed", "model": "m"}).embeddings_array() is None


def test_base64_shape_mismatch_is_an_error():
    with pytest.raises(ValueError):
        Result.from_json({"depth_map_base64": b64(np.zeros(5, np.float32)), "depth_width": 2, "depth_height": 2})
    with pytest.raises(ValueError):
        Result.from_json({"embeddings_base64": b64(np.zeros(5, np.float32)), "embeddings_shape": [2, 2]})
    with pytest.raises(ValueError):
        Result.from_json({"embeddings_base64": b64(np.zeros(4, np.float32))})  # no shape


def test_base64_decodes_without_numpy(monkeypatch):
    """Without numpy the SDK still decodes (stdlib array): plain lists, exact float32 values."""
    real_import = builtins.__import__

    def no_numpy(name, *a, **k):
        if name == "numpy" or name.startswith("numpy."):
            raise ImportError("no numpy")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_numpy)
    vals = np.array([0.1, -2.5, 1e-40], np.float32)
    emb = np.array([[1, 2], [3, 4.5]], np.float32)
    res = Result.from_json({"depth_map_base64": b64(vals), "depth_width": 3, "depth_height": 1,
                            "embeddings_base64": b64(emb), "embeddings_shape": [2, 2]})
    assert type(res.depth_map) is list and res.depth_map == vals.tolist()
    assert res.embeddings == [[1.0, 2.0], [3.0, 4.5]]


# --------------------------------------------------------------------------- #
# to_json
# --------------------------------------------------------------------------- #
def full_result():
    return Result(
        task="grasp", model="grasp-gd", device="gpu:0", hint="install TensorRT",
        detections=[Detection(bbox=[1.0, 2.0, 3.0, 4.0], cls="cup", conf=0.9)],
        masks=[Mask(rle="0 3 1", bbox=[1.0, 2.0, 3.0, 4.0], conf=0.8)],
        grasps=[Grasp(x=2, y=3, theta=0.5, width=10, quality=0.7, cls="cup", conf=0.9),
                Grasp(x=5, y=6, theta=1.0, width=8, quality=0.4)],
        classifications=[Classification(cls="cat", conf=0.6)],
        depth_map=[0.5, 0.25, 0.125, 1.0], depth_width=2, depth_height=2,
        embeddings=[[0.5, -0.5], [0.25, 0.75]], duration_ms=12.5,
    )


def test_to_json_round_trip_both_encodings():
    r = full_result()
    for enc in ("json", "base64"):
        wire = r.to_json(encoding=enc)
        json.dumps(wire)  # serialisable as is
        back = Result.from_json(json.loads(json.dumps(wire)))
        assert back == r, enc
    assert "depth_map" not in r.to_json(encoding="base64") and "depth_map" in r.to_json()


def test_to_json_is_the_server_wire_shape():
    """Key names, key ORDER (pkg/api/types.go field order) and omitempty match the server."""
    wire = {
        "task": "segmentation", "model": "grounded-sam", "device": "cpu",
        "detections": [{"bbox": [1.5, 2, 3, 4], "class": "cat", "conf": 0.9}],
        "masks": [{"rle": "1 2 3", "bbox": [1.5, 2, 3, 4], "conf": 0.8}],
        "grasps": [{"x": 1, "y": 2, "theta": 0.1, "width": 5, "quality": 0.5}],
        "embeddings": [[0.1, 0.2]],
        "depth_map": [0.1, 0.2], "depth_width": 2, "depth_height": 1,
        "duration_ms": 3.25,
    }
    out = Result.from_json(wire).to_json()
    assert out == wire and list(out) == list(wire)
    assert list(out["grasps"][0]) == ["x", "y", "theta", "width", "quality"]  # class/conf omitted


def test_to_json_from_a_base64_result_gives_the_float32_numbers():
    vals = np.array([0.1, 1 / 3], np.float32)
    r = Result.from_json({"task": "depth", "model": "m", "depth_map_base64": b64(vals),
                          "depth_width": 2, "depth_height": 1})
    out = r.to_json()
    assert out["depth_map"] == vals.tolist() and same_bits(out["depth_map"], vals)
    back = Result.from_json(r.to_json(encoding="base64"))
    assert same_bits(back.depth_array(), vals.reshape(1, 2))


def test_to_json_ragged_embeddings_stay_numbers():
    r = Result(task="embed", model="m", embeddings=[[1.0, 2.0], [3.0]])
    out = r.to_json(encoding="base64")
    assert out["embeddings"] == [[1.0, 2.0], [3.0]] and "embeddings_base64" not in out
    with pytest.raises(ValueError):
        r.to_json(encoding="msgpack")


# --------------------------------------------------------------------------- #
# Client: asks for base64 by default (numpy present), opt-out flag, transparent decode
# --------------------------------------------------------------------------- #
class _Server:
    def __init__(self, answer):
        self.answer, self.bodies = answer, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                outer.bodies.append(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                body = json.dumps(outer.answer).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_client_requests_base64_by_default_and_decodes_it():
    depth = np.arange(6, dtype=np.float32) / 7
    srv = _Server({"task": "depth", "model": "midas", "depth_map_base64": b64(depth),
                   "depth_width": 3, "depth_height": 2, "duration_ms": 1})
    try:
        res = Client(srv.url).predict("midas", b"\x89PNG fake")
        assert b'name="encoding"\r\n\r\nbase64' in srv.bodies[-1]
        assert same_bits(res.depth_array(), depth.reshape(2, 3))

        Client(srv.url, base64_arrays=False).predict("midas", b"\x89PNG fake")
        assert b'name="encoding"' not in srv.bodies[-1]
    finally:
        srv.close()


def test_client_default_follows_numpy_availability(monkeypatch):
    assert Client().base64_arrays is True
    monkeypatch.setattr("visionserve.client._have_numpy", lambda: False)
    assert Client().base64_arrays is False
    assert Client(base64_arrays=True).base64_arrays is True


def test_cli_prints_to_json_and_asks_for_numbers(capsys):
    wire = {"task": "embed", "model": "clip", "embeddings": [[0.1, 0.2]], "duration_ms": 2}
    srv = _Server(wire)
    try:
        img = os.path.join(os.path.dirname(__file__), "_cli_input.png")
        with open(img, "wb") as f:
            f.write(b"\x89PNG fake")
        try:
            assert vs_cli.main(["--host", srv.url, "predict", "clip", img, "--compact", "--quiet"]) == 0
        finally:
            os.remove(img)
        assert json.loads(capsys.readouterr().out) == wire
        assert b'name="encoding"' not in srv.bodies[-1]
    finally:
        srv.close()


# --------------------------------------------------------------------------- #
# Consolidated helpers
# --------------------------------------------------------------------------- #
def test_one_grasp_grouping_rule():
    from visionserve.visualize import _grasps_per_object

    dets = [Detection(bbox=[0, 0, 100, 100], cls="box", conf=0.9),
            Detection(bbox=[10, 10, 20, 20], cls="cup", conf=0.8)]
    grasps = [Grasp(x=15, y=15, theta=0, width=5, quality=q / 10, cls="cup") for q in range(5)]
    grasps += [Grasp(x=80, y=80, theta=0, width=5, quality=q / 10, cls="box") for q in range(5)]
    grasps += [Grasp(x=500, y=500, theta=0, width=5, quality=q / 10, cls="far") for q in range(3)]
    for dets_ in (dets, []):
        res = Result(task="grasp", model="g", detections=dets_, grasps=list(grasps))
        for k in (None, 0, 1, 2, 9):
            assert _grasps_per_object(res, k) == res.filter_grasps(k).grasps
    res = Result(task="grasp", model="g", detections=dets, grasps=list(grasps))
    kept = res.filter_grasps(2).grasps
    # smallest containing bbox wins (the cup inside the box); outside every bbox -> by label
    assert [(g.cls, g.quality) for g in kept] == [("cup", 0.4), ("cup", 0.3), ("box", 0.4), ("box", 0.3),
                                                  ("far", 0.2), ("far", 0.1)]


@pytest.mark.parametrize("arr", [
    np.arange(12, dtype=np.uint8).reshape(3, 4),
    np.arange(12, dtype=np.uint8).reshape(3, 4, 1),
    np.linspace(0, 1, 36, dtype=np.float32).reshape(3, 4, 3),
    (np.arange(48, dtype=np.uint16) * 9).reshape(3, 4, 4),
    np.ones((3, 4), bool),
])
def test_one_ndarray_rule_for_upload_and_drawing(arr):
    from PIL import Image

    from visionserve.visualize import _open_image

    uploaded = np.asarray(Image.open(io.BytesIO(_ndarray_to_png(arr))))
    drawn = np.asarray(_open_image(arr, Image))
    assert uploaded.shape == drawn.shape and np.array_equal(uploaded, drawn)
    assert uploaded.ndim == 3 and uploaded.shape[2] in (3, 4)


def test_one_ndarray_rule_rejects_the_same_shapes():
    from PIL import Image

    from visionserve.visualize import _open_image

    for bad in (np.zeros(5, np.uint8), np.zeros((2, 2, 2), np.uint8), np.zeros((1, 2, 2, 3), np.uint8)):
        with pytest.raises(ValueError):
            _ndarray_to_png(bad)
        with pytest.raises(ValueError):
            _open_image(bad, Image)


def test_group_by_class_and_replace_keep_working_with_float_arrays():
    r = Result.from_json({"task": "depth", "model": "m", "depth_map_base64": b64(np.ones(4, np.float32)),
                          "depth_width": 2, "depth_height": 2})
    assert dataclasses.replace(r, depth_map=[]).depth_array() is None
    assert list(r.group_by_class()) == []  # no detections/masks: nothing to group
    assert vs_types._plain_floats(r.depth_map) == [1.0, 1.0, 1.0, 1.0]
