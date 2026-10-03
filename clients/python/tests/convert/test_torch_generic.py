"""Tests for the `torchscript` and `pytorch` converter formats (families/torch_generic.py).

Offline and fast: tiny synthetic modules, run through the real CLI path with --dry-run
(export -> parity -> I/O contract -> manifest), plus the refusals (AGPL pickle, wrong label count,
parity failure, wrong detection output count, unsafe pickle).
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # clients/python

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from visionserve.convert import cli  # noqa: E402
from visionserve.convert.common import onnx_io  # noqa: E402


@pytest.fixture(autouse=True)
def _only_torch(monkeypatch):
    monkeypatch.setattr(cli, "FORMATS", {k: v for k, v in cli.FORMATS.items() if k in ("torchscript", "pytorch")})


def run_cli(tmp_path, fmt, src, name, *extra):
    models = tmp_path / "models"
    rc = cli.main([fmt, str(src), "--name", name, "--dry-run", "--models", str(models), *extra])
    return rc, models / ".convert-dry-run" / name


# --------------------------------------------------------------------------------------------
# tiny models (the same classes are written into build scripts for the `pytorch` format)
# --------------------------------------------------------------------------------------------

MODELS_SRC = textwrap.dedent('''
    import torch
    from torch import nn

    class TinyClassifier(nn.Module):
        def __init__(self, n=5):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, stride=2, padding=1)
            self.bn = nn.BatchNorm2d(8)
            self.fc = nn.Linear(8, n)

        def forward(self, x):
            x = torch.relu(self.bn(self.conv(x)))
            return self.fc(x.mean((2, 3)))

    class TinyDetr(nn.Module):
        """DETR-shaped head: returns a dict {pred_logits [1,Q,C], pred_boxes [1,Q,4] sigmoid cxcywh}."""
        def __init__(self, q=10, c=3, n_out=2):
            super().__init__()
            self.q, self.n_out = q, n_out
            self.conv = nn.Conv2d(3, 16, 4, stride=4)
            self.cls = nn.Linear(16, c)
            self.box = nn.Linear(16, 4)

        def forward(self, x):
            f = self.conv(x).flatten(2).transpose(1, 2)[:, : self.q]   # [1,Q,16]
            out = {"pred_logits": self.cls(f), "pred_boxes": self.box(f).sigmoid()}
            if self.n_out == 3:
                out["aux"] = f
            return out

    class Noisy(nn.Module):
        """Output depends on torch.rand: the ONNX graph cannot reproduce it -> parity must fail."""
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 4)

        def forward(self, x):
            return self.fc(x.mean((2, 3))) + torch.rand(1, 4) * 10
''')


@pytest.fixture
def models_mod(tmp_path):
    p = tmp_path / "tiny_models.py"
    p.write_text(MODELS_SRC)
    ns: dict = {}
    exec(compile(MODELS_SRC, str(p), "exec"), ns)
    return ns


def script(tmp_path, body: str) -> Path:
    p = tmp_path / "build.py"
    p.write_text(MODELS_SRC + "\n" + textwrap.dedent(body))
    return p


def labels_file(tmp_path, n) -> Path:
    p = tmp_path / "labels.txt"
    p.write_text("\n".join(f"c{i}" for i in range(n)) + "\n")
    return p


def randomize_bn(m):
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.5, 0.5)
            mod.running_var.uniform_(0.5, 2.0)
    return m


# --------------------------------------------------------------------------------------------
# torchscript
# --------------------------------------------------------------------------------------------

def test_torchscript_classifier(tmp_path, models_mod):
    m = randomize_bn(models_mod["TinyClassifier"](5)).eval()
    src = tmp_path / "cls.pt"
    torch.jit.script(m).save(str(src))
    rc, out = run_cli(tmp_path, "torchscript", src, "tiny-cls", "--task", "classification", "--input", "64x48",
                      "--license", "MIT", "--labels", str(labels_file(tmp_path, 5)))
    assert rc == 0
    man = (out / "manifest.yaml").read_text()
    for line in ("task: classification", "architecture: efficientnet", "license: MIT", "width: 64",
                 "height: 48", "letterbox: false", "type: classification", "labels: labels.txt"):
        assert line in man, line
    ins, outs = onnx_io(out / "model.onnx")
    assert ins[0][1] == [1, 3, 48, 64]
    assert outs[0][1] == [1, 5]
    assert (out / "labels.txt").read_text().split() == [f"c{i}" for i in range(5)]


def test_torchscript_traced_detection(tmp_path, models_mod):
    m = models_mod["TinyDetr"](q=10, c=3).eval()

    class AsTuple(torch.nn.Module):  # tracing wants tensors out, not a dict
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            o = self.inner(x)
            return o["pred_boxes"], o["pred_logits"]

    src = tmp_path / "det.pt"
    torch.jit.trace(AsTuple(m).eval(), torch.randn(1, 3, 64, 64)).save(str(src))
    rc, out = run_cli(tmp_path, "torchscript", src, "tiny-det", "--task", "detection", "--input", "64x64",
                      "--license", "Apache-2.0", "--labels", str(labels_file(tmp_path, 3)))
    assert rc == 0
    man = (out / "manifest.yaml").read_text()
    assert "architecture: rt-detr" in man and "box_format: cxcywh" in man and "task: detection" in man
    _, outs = onnx_io(out / "model.onnx")
    assert sorted(o[1][-1] for o in outs) == [3, 4]
    assert all(o[1][:2] == [1, 10] for o in outs)


def test_torchscript_agpl_marker_refused(tmp_path, models_mod, capsys):
    m = models_mod["TinyClassifier"](5).eval()
    src = tmp_path / "yolo.pt"
    torch.jit.script(m).save(str(src), _extra_files={"meta.txt": "ultralytics.nn.tasks.DetectionModel"})
    rc, out = run_cli(tmp_path, "torchscript", src, "bad", "--task", "classification", "--input", "32x32",
                      "--license", "MIT")
    assert rc == 2
    assert "ultralytics" in capsys.readouterr().err
    assert not out.exists()


def test_not_torchscript_refused(tmp_path, models_mod, capsys):
    src = tmp_path / "sd.pt"
    torch.save(models_mod["TinyClassifier"](5).state_dict(), src)
    rc, _ = run_cli(tmp_path, "torchscript", src, "x", "--task", "classification", "--input", "32x32",
                    "--license", "MIT")
    assert rc == 2
    assert "pytorch" in capsys.readouterr().err  # points the user at the right format


# --------------------------------------------------------------------------------------------
# pytorch (build script + weights)
# --------------------------------------------------------------------------------------------

def test_pytorch_state_dict_classifier(tmp_path, models_mod):
    m = randomize_bn(models_mod["TinyClassifier"](7))
    w = tmp_path / "w.pth"
    torch.save({"epoch": 3, "state_dict": {"module." + k: v for k, v in m.state_dict().items()}}, w)
    s = script(tmp_path, "def build_model():\n    return TinyClassifier(7)\n")
    rc, out = run_cli(tmp_path, "pytorch", w, "pt-cls", "--script", str(s), "--task", "classification",
                      "--input", "32x32", "--license", "BSD-3-Clause")
    assert rc == 0
    man = (out / "manifest.yaml").read_text()
    assert "license: BSD-3-Clause" in man and "architecture: efficientnet" in man

    # the exported graph carries the TRAINED weights, not build_model()'s fresh init
    import numpy as np
    import onnxruntime as ort
    from visionserve.convert.common import IMAGENET_MEAN, IMAGENET_STD, sample_image, to_nchw
    x = to_nchw(sample_image(32, 32, seed=5), IMAGENET_MEAN, IMAGENET_STD)
    got = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    with torch.no_grad():
        want = m.eval()(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(got, want, atol=1e-4)


def test_pytorch_build_model_loads_weights_itself(tmp_path, models_mod):
    m = models_mod["TinyDetr"](q=12, c=5)
    w = tmp_path / "det.bin"
    torch.save(m.state_dict(), w)
    s = script(tmp_path, """
        def build_model(weights_path):
            m = TinyDetr(q=12, c=5)
            m.load_state_dict(torch.load(weights_path, weights_only=True))
            return m
    """)
    rc, out = run_cli(tmp_path, "pytorch", s, "pt-det", "--weights", str(w), "--task", "detection",
                      "--input", "64x64", "--license", "Apache-2.0", "--labels", str(labels_file(tmp_path, 5)))
    assert rc == 0
    _, outs = onnx_io(out / "model.onnx")
    assert [o[1] for o in outs] == [[1, 12, 5], [1, 12, 4]]  # dict order: pred_logits, pred_boxes
    assert "architecture: rt-detr" in (out / "manifest.yaml").read_text()


def test_pytorch_agpl_marker_refused(tmp_path, models_mod, capsys):
    w = tmp_path / "best.pt"
    torch.save({"model_class": "ultralytics.nn.tasks.DetectionModel",
                "state_dict": models_mod["TinyClassifier"](5).state_dict()}, w)
    s = script(tmp_path, "def build_model():\n    return TinyClassifier(5)\n")
    rc, _ = run_cli(tmp_path, "pytorch", w, "bad", "--script", str(s), "--task", "classification",
                    "--input", "32x32", "--license", "MIT")
    assert rc == 2
    assert "AGPL" in capsys.readouterr().err


def test_wrong_label_count_refused(tmp_path, models_mod, capsys):
    w = tmp_path / "w.pth"
    torch.save(models_mod["TinyClassifier"](5).state_dict(), w)
    s = script(tmp_path, "def build_model():\n    return TinyClassifier(5)\n")
    rc, out = run_cli(tmp_path, "pytorch", w, "badlabels", "--script", str(s), "--task", "classification",
                      "--input", "32x32", "--license", "MIT", "--labels", str(labels_file(tmp_path, 4)))
    assert rc == 2
    assert "5 logits but 4 labels" in capsys.readouterr().err
    assert not out.exists()


def test_parity_failure_refused(tmp_path, capsys):
    s = script(tmp_path, "def build_model():\n    return Noisy()\n")
    rc, out = run_cli(tmp_path, "pytorch", s, "noisy", "--task", "classification", "--input", "16x16",
                      "--license", "MIT")
    assert rc == 2
    err = capsys.readouterr().err
    assert "parity" in err and "Not installed" in err
    assert not out.exists()


def test_detection_needs_exactly_two_outputs(tmp_path, capsys):
    s = script(tmp_path, "def build_model():\n    return TinyDetr(n_out=3)\n")
    rc, _ = run_cli(tmp_path, "pytorch", s, "det3", "--task", "detection", "--input", "64x64", "--license", "MIT")
    assert rc == 2
    assert "EXACTLY two outputs" in capsys.readouterr().err


def test_state_dict_mismatch_refused(tmp_path, models_mod, capsys):
    w = tmp_path / "w.pth"
    torch.save(models_mod["TinyClassifier"](9).state_dict(), w)
    s = script(tmp_path, "def build_model():\n    return TinyClassifier(5)\n")
    rc, _ = run_cli(tmp_path, "pytorch", w, "mismatch", "--script", str(s), "--task", "classification",
                    "--input", "32x32", "--license", "MIT")
    assert rc == 2
    assert "do not fit" in capsys.readouterr().err


def test_pickled_module_needs_unsafe_flag(tmp_path, models_mod, capsys):
    import types
    mod = types.ModuleType("tiny_models_pickled")  # a real importable module so the class pickles
    exec(MODELS_SRC, mod.__dict__)
    sys.modules["tiny_models_pickled"] = mod
    try:
        w = tmp_path / "whole.pt"
        torch.save(mod.TinyClassifier(5), w)
        s = script(tmp_path, "def build_model():\n    return TinyClassifier(5)\n")
        base = ["--script", str(s), "--task", "classification", "--input", "32x32", "--license", "MIT"]
        rc, _ = run_cli(tmp_path, "pytorch", w, "whole", *base)
        assert rc == 2
        assert "--unsafe-pickle" in capsys.readouterr().err
        rc, _ = run_cli(tmp_path, "pytorch", w, "whole", *base, "--unsafe-pickle")
        assert rc == 0
    finally:
        sys.modules.pop("tiny_models_pickled", None)


def test_agpl_license_refused(tmp_path, models_mod, capsys):
    src = tmp_path / "cls.pt"
    torch.jit.script(models_mod["TinyClassifier"](5).eval()).save(str(src))
    rc, _ = run_cli(tmp_path, "torchscript", src, "x", "--task", "classification", "--input", "32x32",
                    "--license", "AGPL-3.0")
    assert rc == 2
    assert "not allowed" in capsys.readouterr().err
