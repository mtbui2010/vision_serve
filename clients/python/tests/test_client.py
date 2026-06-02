"""Deterministic tests for the VisionServe Python client.

These DO NOT require the Go server: a tiny mock HTTP server (stdlib ``http.server``
in a background thread) returns canned JSON and records the requests it receives, so
we can validate both request building and response parsing offline.

Run with the label env python:
    /home/trung/miniconda3/envs/label/bin/python3 -m pytest clients/python/tests -v
or as a self-test:
    /home/trung/miniconda3/envs/label/bin/python3 clients/python/tests/test_client.py
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# Make the package importable when run directly (not just via pytest -e install).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visionserve import Client, Mask, Result  # noqa: E402
from visionserve.client import (  # noqa: E402
    _serialize_boxes,
    _serialize_points,
    _build_multipart,
)


# --------------------------------------------------------------------------- #
# Reference Go encoder (column-major, starts background) — ported from
# internal/models/mobilesam/postprocess.go : encodeRLEColumnMajor.
# Used to verify our decoder is a faithful inverse.
# --------------------------------------------------------------------------- #
def encode_rle_column_major(bin_rows, h, w):
    """bin_rows: 2D list [h][w] of bool. Returns space-separated counts string."""
    counts = []
    prev = False  # runs start with background
    run = 0
    for x in range(w):
        for y in range(h):
            v = bool(bin_rows[y][x])
            if v == prev:
                run += 1
            else:
                counts.append(run)
                prev = v
                run = 1
    counts.append(run)
    return " ".join(str(c) for c in counts)


# --------------------------------------------------------------------------- #
# Mock server
# --------------------------------------------------------------------------- #
class _MockState:
    last_request = None  # dict: {path, method, headers, body}


def _make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def _read_body(self):
            length = int(self.headers.get("Content-Length", 0))
            return self.rfile.read(length) if length else b""

        def _send(self, code, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            state.last_request = {"path": self.path, "method": "GET", "headers": dict(self.headers), "body": b""}
            if self.path == "/api/health":
                self._send(200, {"status": "ok"})
            elif self.path == "/api/models":
                self._send(200, [
                    {"name": "rf-detr", "task": "detection", "license": "Apache-2.0", "state": "loaded"},
                    {"name": "mobile-sam", "task": "segmentation", "license": "Apache-2.0", "state": "available"},
                ])
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            body = self._read_body()
            state.last_request = {
                "path": self.path,
                "method": "POST",
                "headers": dict(self.headers),
                "body": body,
            }
            if self.path == "/api/load":
                req = json.loads(body)
                self._send(200, {"model": req["model"], "state": "loaded"})
            elif self.path == "/api/unload":
                req = json.loads(body)
                self._send(200, {"model": req["model"], "state": "unloaded"})
            elif self.path == "/api/predict":
                self._send(200, {
                    "task": "detection",
                    "model": "rf-detr",
                    "detections": [
                        {"bbox": [10.0, 20.0, 30.0, 40.0], "class": "cat", "conf": 0.91},
                    ],
                    "masks": [
                        {"rle": "1 2 3", "bbox": [0, 0, 2, 3], "conf": 0.8},
                    ],
                    "duration_ms": 12.5,
                })
            else:
                self._send(404, {"error": "not found"})

    return Handler


class MockServer:
    def __enter__(self):
        self.state = _MockState()
        self.httpd = HTTPServer(("127.0.0.1", 0), _make_handler(self.state))
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.host = "http://127.0.0.1:%d" % self.port
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_health_and_models():
    with MockServer() as srv:
        c = Client(host=srv.host)
        assert c.health() == {"status": "ok"}
        models = c.list_models()
        assert [m.name for m in models] == ["rf-detr", "mobile-sam"]
        assert models[0].state == "loaded"
        ps = c.ps()
        assert [m.name for m in ps] == ["rf-detr"]


def test_load_unload():
    with MockServer() as srv:
        c = Client(host=srv.host)
        assert c.load("rf-detr") == {"model": "rf-detr", "state": "loaded"}
        body = json.loads(srv.state.last_request["body"])
        assert body == {"model": "rf-detr"}
        assert c.unload("rf-detr")["state"] == "unloaded"


def test_predict_multipart_request_and_parse():
    with MockServer() as srv:
        c = Client(host=srv.host)
        res = c.predict(
            "rf-detr",
            b"\x89PNG fake bytes",
            prompt="cat. remote.",
            box=[10, 20, 30, 40],
            point=[5, 6, 1],
        )
        # response parsing
        assert isinstance(res, Result)
        assert res.task == "detection"
        assert res.detections[0].cls == "cat"
        assert res.detections[0].bbox == [10.0, 20.0, 30.0, 40.0]
        assert abs(res.duration_ms - 12.5) < 1e-9

        # request building: multipart content-type + fields present
        req = srv.state.last_request
        assert req["path"] == "/api/predict"
        ct = req["headers"]["Content-Type"]
        assert ct.startswith("multipart/form-data; boundary=")
        raw = req["body"].decode("utf-8", errors="replace")
        assert 'name="model"' in raw and "rf-detr" in raw
        assert 'name="prompt"' in raw and "cat. remote." in raw
        assert 'name="box"' in raw and "10,20,30,40" in raw
        assert 'name="point"' in raw and "5,6,1" in raw
        assert 'name="image"; filename=' in raw


def test_serialize_boxes_and_points():
    # single box and list of boxes
    assert _serialize_boxes([1, 2, 3, 4]) == "1,2,3,4"
    assert _serialize_boxes([[1, 2, 3, 4], [5, 6, 7, 8]]) == "1,2,3,4;5,6,7,8"
    # float that is integer-valued prints without .0
    assert _serialize_boxes([1.0, 2.5, 3, 4]) == "1,2.5,3,4"
    # points 2 and 3 wide
    assert _serialize_points([5, 6]) == "5,6"
    assert _serialize_points([[5, 6, 1], [7, 8, 0]]) == "5,6,1;7,8,0"


def test_multipart_builder_shape():
    body, ct = _build_multipart({"model": "x"}, b"IMG", "image.png")
    assert ct.startswith("multipart/form-data; boundary=")
    assert b'name="model"' in body
    assert b'name="image"; filename="image.png"' in body
    assert b"IMG" in body


def test_rle_roundtrip_known():
    """A small hand-built RLE round-trips through Mask.to_ndarray()."""
    import numpy as np

    # 3x2 (H=3, W=2) mask. Lay it out as rows [h][w]:
    #   col0: [1,0,1]   col1: [0,0,1]
    # Column-major read order (x outer, y inner): 1,0,1, 0,0,1
    #   -> runs starting background: bg0(0)=0? sequence is [1,0,1,0,0,1]
    # Build counts via the reference Go encoder to be exact.
    h, w = 3, 2
    rows = [
        [1, 0],
        [0, 0],
        [1, 1],
    ]
    rle = encode_rle_column_major(rows, h, w)
    m = Mask(rle=rle, bbox=[0, 0, w, h], conf=1.0)
    arr = m.to_ndarray(width=w, height=h)
    expected = np.array(rows, dtype=bool)
    assert arr.shape == (h, w)
    assert np.array_equal(arr, expected), (arr.tolist(), expected.tolist(), rle)


def test_rle_roundtrip_random():
    """Random masks: encode (reference Go algo) -> decode -> must equal original."""
    import numpy as np

    rng = np.random.default_rng(1234)
    for _ in range(50):
        h = int(rng.integers(1, 9))
        w = int(rng.integers(1, 9))
        mask = rng.integers(0, 2, size=(h, w)).astype(bool)
        rows = mask.astype(int).tolist()
        rle = encode_rle_column_major(rows, h, w)
        decoded = Mask(rle=rle, bbox=[0, 0, w, h], conf=1.0).to_ndarray(width=w, height=h)
        assert np.array_equal(decoded, mask), (h, w, rle)


def test_rle_all_background_and_all_foreground():
    import numpy as np

    h, w = 4, 3
    # all background -> single run "12"
    m0 = Mask(rle="%d" % (h * w), bbox=[0, 0, w, h], conf=1.0)
    assert not m0.to_ndarray(w, h).any()
    # all foreground -> "0 12" (background run 0, then foreground 12)
    m1 = Mask(rle="0 %d" % (h * w), bbox=[0, 0, w, h], conf=1.0)
    assert m1.to_ndarray(w, h).all()


def test_rle_bad_sum_raises():
    import pytest

    m = Mask(rle="1 2", bbox=[0, 0, 5, 5], conf=1.0)
    with pytest.raises(ValueError):
        m.to_ndarray(5, 5)


# --------------------------------------------------------------------------- #
# Self-test entry point (works without pytest installed)
# --------------------------------------------------------------------------- #
def _run_self_test():
    failures = []
    tests = [
        test_health_and_models,
        test_load_unload,
        test_predict_multipart_request_and_parse,
        test_serialize_boxes_and_points,
        test_multipart_builder_shape,
        test_rle_roundtrip_known,
        test_rle_roundtrip_random,
        test_rle_all_background_and_all_foreground,
    ]
    for t in tests:
        try:
            t()
            print("PASS %s" % t.__name__)
        except Exception as e:  # noqa: BLE001
            failures.append((t.__name__, e))
            print("FAIL %s: %r" % (t.__name__, e))

    # bad-sum test without pytest:
    try:
        Mask(rle="1 2", bbox=[0, 0, 5, 5], conf=1.0).to_ndarray(5, 5)
        failures.append(("test_rle_bad_sum_raises", "did not raise"))
        print("FAIL test_rle_bad_sum_raises: did not raise")
    except ValueError:
        print("PASS test_rle_bad_sum_raises")

    if failures:
        print("\n%d FAILURE(S)" % len(failures))
        sys.exit(1)
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    _run_self_test()
