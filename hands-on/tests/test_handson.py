"""Light tests for hands-on/handson.py. No real server or GPU is needed.

Run from the repository root:  python -m pytest hands-on/tests -q
"""

import json
import os
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import handson  # noqa: E402
from visionserve import Classification, Detection, Grasp, Mask, Result  # noqa: E402


class _FakeServer:
    """A tiny HTTP server that answers /api/health and /api/models like VisionServe."""

    def __init__(self, models):
        self.models = models
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path == "/api/health":
                    body = {"status": "ok"}
                elif self.path == "/api/models":
                    body = outer.models
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


def _model(name, state):
    return {"name": name, "task": "detection", "license": "Apache-2.0", "state": state}


# ----------------------------------------------------------------- connect / host
def test_host_url_order(monkeypatch):
    monkeypatch.delenv("VS_HOST", raising=False)
    assert handson.host_url() == "http://127.0.0.1:11435"
    monkeypatch.setenv("VS_HOST", "http://10.0.0.5:9000/")
    assert handson.host_url() == "http://10.0.0.5:9000"
    assert handson.host_url("localhost:1234") == "http://localhost:1234"


def test_connect_ok(capsys):
    with _FakeServer([]) as srv:
        client = handson.connect(srv.url)
    assert client.host == srv.url
    assert "Connected" in capsys.readouterr().out


def test_connect_uses_env(monkeypatch):
    with _FakeServer([]) as srv:
        monkeypatch.setenv("VS_HOST", srv.url)
        assert handson.connect().host == srv.url


def test_connect_fails_with_help(capsys):
    with pytest.raises(RuntimeError):
        handson.connect("http://127.0.0.1:9")  # nothing listens on port 9
    out = capsys.readouterr().out
    assert "docker run" in out and "visionserve serve" in out


# ----------------------------------------------------------------- ensure_model
def test_ensure_model_installed_does_not_pull(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(handson, "run_cli", lambda *a, **k: calls.append(a) or "")
    with _FakeServer([_model("rf-detr", "available")]) as srv:
        assert handson.ensure_model(handson.Client(srv.url), "rf-detr") is True
    assert calls == []
    assert "installed" in capsys.readouterr().out


def test_ensure_model_pulls_missing(monkeypatch):
    with _FakeServer([_model("midas", "not_downloaded")]) as srv:
        calls = []

        def fake_cli(*a, **k):
            calls.append(a)
            srv.models = [_model("midas", "available")]  # the pull "worked"
            return ""

        monkeypatch.setattr(handson, "run_cli", fake_cli)
        monkeypatch.delenv("HANDSON_NO_PULL", raising=False)
        assert handson.ensure_model(handson.Client(srv.url), "midas") is True
    assert calls == [("pull", "midas")]


def test_ensure_model_unknown_model_pulls(monkeypatch):
    calls = []
    monkeypatch.setattr(handson, "run_cli", lambda *a, **k: calls.append(a) or "")
    monkeypatch.delenv("HANDSON_NO_PULL", raising=False)
    with _FakeServer([]) as srv:
        assert handson.ensure_model(handson.Client(srv.url), "scrfd") is False  # still missing
    assert calls == [("pull", "scrfd")]


def test_ensure_model_no_pull(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(handson, "run_cli", lambda *a, **k: calls.append(a) or "")
    monkeypatch.setenv("HANDSON_NO_PULL", "1")
    with _FakeServer([_model("midas", "not_downloaded")]) as srv:
        assert handson.ensure_model(handson.Client(srv.url), "midas") is False
    assert calls == []
    assert "NOT downloaded" in capsys.readouterr().out


# ----------------------------------------------------------------- photos
def test_photo_keys_exist():
    for key in handson.PHOTOS:
        p = handson.photo(key)
        assert Path(p).is_file(), p
        assert Path(p).name == key + ".jpg"


def test_photo_unknown_key():
    with pytest.raises(KeyError) as e:
        handson.photo("zebra")
    assert "cat" in str(e.value)


def test_photo_relative_inside_cwd(monkeypatch):
    monkeypatch.chdir(HERE.parent)
    assert handson.photo("cat") == Path("images/cat.jpg")


# ----------------------------------------------------------------- run_cli
def _fake_binary(tmp_path, code):
    exe = tmp_path / "fakevs"
    exe.write_text(
        "#!%s\nimport sys\nprint('args', ' '.join(sys.argv[1:]))\n"
        "print('to stderr', file=sys.stderr)\nsys.exit(%d)\n" % (sys.executable, code))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe


@pytest.mark.skipif(os.name == "nt", reason="shebang scripts")
def test_run_cli_exit_codes(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("VISIONSERVE_CLI", raising=False)
    monkeypatch.setenv("VISIONSERVE_BIN", str(_fake_binary(tmp_path, 0)))
    assert handson.run_cli("list", "--x") == "args list --x\n"
    out = capsys.readouterr().out
    assert "$ visionserve list --x" in out and "to stderr" in out and "(exit code 0)" in out

    monkeypatch.setenv("VISIONSERVE_BIN", str(_fake_binary(tmp_path, 1)))
    assert "args check" in handson.run_cli("check", echo=False)  # FAIL verdict: no error
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("VISIONSERVE_BIN", str(_fake_binary(tmp_path, 2)))
    with pytest.raises(RuntimeError):
        handson.run_cli("check")
    assert handson.run_cli("check", check=False, echo=False).startswith("args")


@pytest.mark.skipif(os.name == "nt", reason="shebang scripts")
def test_run_cli_prefix_command(tmp_path, monkeypatch, capsys):
    exe = _fake_binary(tmp_path, 0)
    monkeypatch.setenv("VISIONSERVE_CLI", "%s %s extra" % (sys.executable, exe))
    assert handson.run_cli("pull", "midas") == "args extra pull midas\n"
    assert "pull midas" in capsys.readouterr().out


def test_run_cli_missing_binary(monkeypatch, capsys):
    monkeypatch.delenv("VISIONSERVE_CLI", raising=False)
    monkeypatch.setenv("VISIONSERVE_BIN", "/no/such/visionserve")
    with pytest.raises(RuntimeError):
        handson.run_cli("list")
    assert "VISIONSERVE_BIN" in capsys.readouterr().out


def test_find_cli_skips_python_script(tmp_path, monkeypatch):
    script = tmp_path / "visionserve"
    script.write_text("#!/usr/bin/env python\n")
    script.chmod(0o755)
    monkeypatch.delenv("VISIONSERVE_CLI", raising=False)
    monkeypatch.delenv("VISIONSERVE_BIN", raising=False)
    monkeypatch.setattr(handson, "REPO", tmp_path / "no-repo")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert handson._find_cli() is None


# ----------------------------------------------------------------- drawing
def _rle_box(w, h, x0, y0, x1, y1):
    """Column-major RLE of a filled rectangle [x0, x1) x [y0, y1)."""
    import numpy as np

    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    flat = m.flatten(order="F")
    runs, cur, n = [], False, 0
    for v in flat:
        if v != cur:
            runs.append(n)
            cur, n = v, 0
        n += 1
    runs.append(n)
    return " ".join(map(str, runs))


def _full_result():
    w, h = 64, 48
    return Result(
        task="grasp", model="fake",
        detections=[Detection(bbox=[10, 10, 20, 15], cls="cup", conf=0.9)],
        masks=[Mask(rle=_rle_box(w, h, 10, 10, 30, 25), bbox=[10, 10, 20, 15], conf=0.95)],
        grasps=[Grasp(x=20, y=17, theta=0.3, width=12, quality=0.8, cls="cup", conf=0.9)],
        classifications=[Classification("cup", 0.7), Classification("mug", 0.2)],
        depth_map=[float(i % 7) for i in range(8 * 6)], depth_width=8, depth_height=6,
        duration_ms=12.0, device="cpu")


def test_draw_returns_same_size_and_colours_mask():
    import numpy as np

    img = np.zeros((48, 64, 3), np.uint8)
    out = handson.draw(img, _full_result())
    assert out.size == (64, 48)
    arr = np.asarray(out)
    assert arr[17, 22].sum() > 0  # inside the mask: coloured
    assert arr[45, 60].sum() == 0  # far away: untouched


def test_show_and_side_by_side_run(tmp_path):
    import numpy as np
    from PIL import Image

    p = tmp_path / "x.png"
    Image.fromarray(np.full((48, 64, 3), 128, np.uint8)).save(p)
    res = _full_result()
    handson.show(p, res)
    handson.show(p, res, title="t", show_masks=False)
    handson.show(np.zeros((48, 64), np.float32))
    handson.show_side_by_side([p, (p, res), (p, None)], titles=["a", "b"])


def test_table_and_json(capsys):
    handson.table([["cat", 0.91234], ["dog", 0.5]], ["class", "conf"])
    out = capsys.readouterr().out
    assert "class" in out and "cat" in out and "0.912" in out
    handson.print_json(_full_result(), max_items=1)
    out = capsys.readouterr().out
    assert '"bbox": [10, 10, 20, 15]' in out
    assert "... 47 more" in out  # the depth map is shortened


def test_show_html_report(tmp_path):
    p = tmp_path / "r.html"
    p.write_text("<h1>Report</h1>")
    handson.show_html_report(p, height=300)
