"""`visionserve predict` (the Python CLI) passes every Client.predict option to the server.

Offline: a stub HTTP server records the multipart request. numpy + pillow required.
"""
import http.server
import inspect
import json
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from visionserve import Client  # noqa: E402
from visionserve import cli as vs_cli  # noqa: E402


class _Stub:
    """Answers every POST with a small detection result and keeps the multipart parts it got."""

    def __init__(self):
        self.parts = None
        stub = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                boundary = self.headers["Content-Type"].split("boundary=")[1].encode()
                stub.parts = {}
                for part in body.split(b"--" + boundary)[1:-1]:
                    head, _, data = part[2:-2].partition(b"\r\n\r\n")  # strip the CRLFs around the part
                    stub.parts[head.decode().split('name="')[1].split('"')[0]] = data
                payload = json.dumps({"task": "detection", "model": "m", "duration_ms": 1.0}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = "http://127.0.0.1:%d" % self.srv.server_address[1]

    def field(self, name):
        return self.parts[name].decode()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def stub():
    s = _Stub()
    yield s
    s.close()


@pytest.fixture
def image(tmp_path):
    p = tmp_path / "img.png"
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(p)  # 4 wide, 3 high
    return p


def _predict(stub, image, *flags):
    return vs_cli.main(["--host", stub.url, "predict", "background", str(image), "--quiet", "--compact", *flags])


def test_every_predict_keyword_has_a_cli_flag():
    """A new Client.predict keyword must get a CLI flag (or be listed here with the reason)."""
    keywords = set(inspect.signature(Client.predict).parameters) - {"self", "model", "image"}
    parser = vs_cli.build_parser()
    predict = next(a for a in parser._subparsers._group_actions[0].choices.values() if a.prog.endswith("predict"))
    dests = {a.dest for a in predict._actions}
    assert sorted(keywords - dests) == []


def test_textalign_and_template_flags_are_sent(stub, image, capsys):
    rc = _predict(stub, image, "--claim-threshold", "0.05", "--crop-temp", "0.02", "--template-name", "dog")
    assert rc == 0, capsys.readouterr().err
    assert stub.field("claim_threshold") == "0.05"
    assert stub.field("crop_temp") == "0.02"
    assert stub.field("template_name") == "dog"
    assert "depth" not in stub.parts


def test_depth_npy_uint16(stub, image, tmp_path, capsys):
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 65535]], dtype=np.uint16)
    np.save(tmp_path / "d.npy", arr)
    assert _predict(stub, image, "--method", "depth", "--depth", str(tmp_path / "d.npy")) == 0, capsys.readouterr().err
    assert (stub.field("depth_dtype"), stub.field("depth_width"), stub.field("depth_height")) == ("uint16", "4", "3")
    assert np.frombuffer(stub.parts["depth"], "<u2").tolist() == arr.ravel().tolist()


def test_depth_16bit_png(stub, image, tmp_path, capsys):
    arr = (np.arange(12, dtype=np.uint16).reshape(3, 4) * 1000 + 1)
    Image.fromarray(arr).save(tmp_path / "d.png")
    assert _predict(stub, image, "--depth", str(tmp_path / "d.png")) == 0, capsys.readouterr().err
    assert stub.field("depth_dtype") == "uint16"
    assert np.frombuffer(stub.parts["depth"], "<u2").reshape(3, 4).tolist() == arr.tolist()


def test_depth_raw_float32_with_size(stub, image, tmp_path, capsys):
    arr = np.linspace(0.5, 3.0, 6, dtype="<f4").reshape(2, 3)
    (tmp_path / "d.bin").write_bytes(arr.tobytes())
    rc = _predict(stub, image, "--depth", str(tmp_path / "d.bin"), "--depth-dtype", "float32",
                  "--depth-width", "3", "--depth-height", "2")
    assert rc == 0, capsys.readouterr().err
    assert (stub.field("depth_dtype"), stub.field("depth_width"), stub.field("depth_height")) == ("float32", "3", "2")
    assert np.frombuffer(stub.parts["depth"], "<f4").tolist() == arr.ravel().tolist()


def test_depth_raw_defaults_to_uint16_at_the_image_size(stub, image, tmp_path, capsys):
    (tmp_path / "d.raw").write_bytes(np.arange(12, dtype="<u2").tobytes())  # the image is 4x3
    assert _predict(stub, image, "--depth", str(tmp_path / "d.raw")) == 0, capsys.readouterr().err
    assert (stub.field("depth_dtype"), stub.field("depth_width"), stub.field("depth_height")) == ("uint16", "4", "3")


def test_depth_raw_size_follows_exif_rotation(tmp_path):
    # Pixels stored 4x3 with EXIF "rotate 90": the server sees (and sizes the depth to) 3x4.
    p = tmp_path / "rot.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(p, exif=exif)
    assert vs_cli._image_size(str(p)) == (3, 4)


@pytest.mark.parametrize(
    "flags, message",
    [
        (["--depth", "{raw}"], "has 10 bytes, but 4x3 uint16 needs 24"),
        (["--depth", "{raw}", "--depth-width", "5"], "go together"),
        (["--depth", "{raw}", "--depth-width", "0", "--depth-height", "5"], "must be > 0"),
        (["--depth", "{npy}", "--depth-dtype", "float32"], "--depth-dtype is for raw depth files"),
        (["--depth-width", "3"], "--depth-width needs --depth"),
        (["--depth", "{missing}"], "No such file"),
    ],
)
def test_depth_mistakes_are_errors_before_any_request(stub, image, tmp_path, capsys, flags, message):
    paths = {"raw": tmp_path / "d.raw", "npy": tmp_path / "d.npy", "missing": tmp_path / "nope.bin"}
    paths["raw"].write_bytes(b"\0" * 10)
    np.save(paths["npy"], np.zeros((3, 4), dtype=np.uint16))
    flags = [f.format(**paths) for f in flags]
    assert _predict(stub, image, *flags) == 1
    assert message in capsys.readouterr().err
    assert stub.parts is None  # nothing was sent
