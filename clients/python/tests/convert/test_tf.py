"""Offline tests for the TensorFlow family (SavedModel / Keras / TFLite -> ONNX).

Run inside the TF venv (convert/requirements-tf.txt):  python -m pytest convert/tests/test_tf.py
Every model is built on the fly; nothing is downloaded. --dry-run means no Go binary is needed.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
tf = pytest.importorskip("tensorflow")
pytest.importorskip("tf2onnx")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from visionserve.convert import cli  # noqa: E402

K = tf.keras
H, W = 40, 48          # non-square on purpose: catches H/W swaps in the NCHW transpose
RAW = ["--mean", "0,0,0", "--std", "0.00392157,0.00392157,0.00392157"]


# ------------------------------------------------------------------------------------------------
# model factories
# ------------------------------------------------------------------------------------------------

def _classifier(n_classes=5, softmax=False, rescaling=False, dynamic=True):
    tf.random.set_seed(0)
    inp = K.Input((None, None, 3) if dynamic else (H, W, 3), name="image")
    x = K.layers.Rescaling(1 / 127.5, offset=-1)(inp) if rescaling else inp
    x = K.layers.Conv2D(8, 3, strides=2, padding="same", activation="relu")(x)
    x = K.layers.Conv2D(8, 3, padding="same", activation="relu")(x)
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(n_classes, activation="softmax" if softmax else None, name="logits")(x)
    return K.Model(inp, x)


def _detector(q=6, c=3, sigmoid_logits=False, extra_output=False):
    tf.random.set_seed(1)
    inp = K.Input((H, W, 3), name="image")
    x = K.layers.Conv2D(8, 3, strides=2, padding="same", activation="relu")(inp)
    x = K.layers.GlobalAveragePooling2D()(x)
    logits = K.layers.Reshape((q, c), name="pred_logits")(
        K.layers.Dense(q * c, activation="sigmoid" if sigmoid_logits else None)(x))
    boxes = K.layers.Reshape((q, 4), name="pred_boxes")(K.layers.Dense(q * 4, activation="sigmoid")(x))
    outs = {"pred_logits": logits, "pred_boxes": boxes}
    if extra_output:
        outs["aux_feat"] = K.layers.Dense(7, name="aux_feat")(x)
    return K.Model(inp, outs)


def _tflite(model, path, quant=None):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if quant:
        def rep():
            for _ in range(4):
                yield [np.random.rand(1, H, W, 3).astype(np.float32) * 255]
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep
        if quant == "int8-io":
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type = tf.uint8
            conv.inference_output_type = tf.uint8
    path.write_bytes(conv.convert())
    return path


def _run(tmp_path, fmt, src, *extra, task="classification", name="m", license="Apache-2.0"):
    argv = [fmt, str(src), "--name", name, "--task", task, "--input", f"{W}x{H}",
            "--models", str(tmp_path / "models"), "--dry-run"]
    if license:
        argv += ["--license", license]
    return cli.main(argv + list(extra))


def _installed(tmp_path, name="m"):
    d = tmp_path / "models" / ".convert-dry-run" / name
    manifest = (d / "manifest.yaml").read_text()
    onnx_path = d / f"{name}.onnx"
    return manifest, onnx_path


def _io(onnx_path):
    m = onnx.load(str(onnx_path))
    dims = lambda v: [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]  # noqa: E731
    return m, [(i.name, dims(i)) for i in m.graph.input], [(o.name, dims(o)) for o in m.graph.output]


def _nhwc_vs_nchw(onnx_path, reference_fn):
    """Independent end-to-end check: same pixels in both layouts give the same answer."""
    x = np.random.default_rng(3).random((1, H, W, 3), np.float32)
    got = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(
        None, {"input": x.transpose(0, 3, 1, 2).copy()})
    return got, reference_fn(x)


# ------------------------------------------------------------------------------------------------
# positive paths
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("ext", [".keras", ".h5"])
def test_keras_classifier(tmp_path, ext):
    m = _classifier()
    src = tmp_path / f"cls{ext}"
    m.save(src)
    assert _run(tmp_path, "keras", src) == 0
    manifest, path = _installed(tmp_path)
    assert "architecture: efficientnet" in manifest and "license: Apache-2.0" in manifest
    assert f"width: {W}" in manifest and f"height: {H}" in manifest
    _, ins, outs = _io(path)
    assert ins == [("input", [1, 3, H, W])]
    assert outs == [("logits", [1, 5])]
    got, ref = _nhwc_vs_nchw(path, lambda x: m(x).numpy())
    np.testing.assert_allclose(got[0], ref, atol=1e-4)


def test_savedmodel_classifier_and_signature(tmp_path):
    m = _classifier()
    sm = tmp_path / "sm"
    tf.saved_model.save(m, str(sm))
    assert _run(tmp_path, "tensorflow", sm) == 0
    _, path = _installed(tmp_path)
    _, ins, outs = _io(path)
    assert ins == [("input", [1, 3, H, W])] and outs[0][1] == [1, 5]
    # an unknown signature is refused with the list of available ones
    assert _run(tmp_path, "tensorflow", sm, "--signature", "nope") == 2


def test_tflite_float_classifier(tmp_path):
    m = _classifier(dynamic=False)
    src = _tflite(m, tmp_path / "cls.tflite")
    assert _run(tmp_path, "tflite", src) == 0
    _, path = _installed(tmp_path)
    _, ins, outs = _io(path)
    assert ins == [("input", [1, 3, H, W])] and outs[0][1][-1] == 5


def test_tflite_int8_internals_float_io(tmp_path):
    """Float-I/O TFLite with int8 internals (Optimize.DEFAULT + representative data): tf2onnx maps
    the (de)quantize ops, and parity against the TFLite interpreter is what vouches for it."""
    m = _classifier(dynamic=False)
    src = _tflite(m, tmp_path / "dq.tflite", quant="int8-internal")
    assert _run(tmp_path, "tflite", src, "--tolerance", "5e-2") == 0


def test_trailing_softmax_is_removed(tmp_path, capsys):
    m = _classifier(softmax=True)
    src = tmp_path / "sm.keras"
    m.save(src)
    assert _run(tmp_path, "keras", src) == 0
    manifest, path = _installed(tmp_path)
    g, _, outs = _io(path)
    assert g.graph.node[-1].op_type != "Softmax"
    assert "Softmax removed" in manifest
    got, ref = _nhwc_vs_nchw(path, lambda x: m(x).numpy())
    e = np.exp(got[0] - got[0].max(-1, keepdims=True))
    np.testing.assert_allclose(e / e.sum(-1, keepdims=True), ref, atol=1e-4)


@pytest.mark.parametrize("fmt", ["keras", "tensorflow", "tflite"])
def test_preprocessing_layer_warns(tmp_path, capsys, fmt):
    m = _classifier(rescaling=True, dynamic=fmt != "tflite")
    if fmt == "keras":
        src = tmp_path / "pre.keras"
        m.save(src)
    elif fmt == "tensorflow":
        src = tmp_path / "pre_sm"
        tf.saved_model.save(m, str(src))
    else:
        src = _tflite(m, tmp_path / "pre.tflite")
    assert _run(tmp_path, fmt, src) == 0
    assert "contains its own preprocessing" in capsys.readouterr().err
    # with raw-pixel normalisation there is nothing to warn about
    assert _run(tmp_path, fmt, src, *RAW) == 0
    assert "contains its own preprocessing" not in capsys.readouterr().err


@pytest.mark.parametrize("fmt", ["keras", "tensorflow"])
def test_detection_two_outputs(tmp_path, fmt):
    m = _detector(sigmoid_logits=True)
    if fmt == "keras":
        src = tmp_path / "det.keras"
        m.save(src)
    else:
        src = tmp_path / "det_sm"
        tf.saved_model.save(m, str(src))
    assert _run(tmp_path, fmt, src, task="detection") == 0
    manifest, path = _installed(tmp_path)
    assert "architecture: rt-detr" in manifest and "Sigmoid removed" in manifest
    _, _, outs = _io(path)
    shapes = sorted(o[1] for o in outs)
    assert shapes == [[1, 6, 3], [1, 6, 4]]


def test_detection_pick_outputs(tmp_path):
    m = _detector(extra_output=True)
    src = tmp_path / "det3.keras"
    m.save(src)
    assert _run(tmp_path, "keras", src, task="detection") == 2          # 3 outputs: ambiguous
    assert _run(tmp_path, "keras", src, "--outputs", "pred_logits,pred_boxes", task="detection") == 0
    _, _, outs = _io(_installed(tmp_path)[1])
    assert [o[0] for o in outs] == ["pred_logits", "pred_boxes"]


def test_tflite_detection_output_names(tmp_path):
    m = _detector(extra_output=True)
    src = _tflite(m, tmp_path / "det.tflite")
    assert _run(tmp_path, "tflite", src, "--outputs", "pred_boxes,pred_logits", task="detection") == 0
    _, _, outs = _io(_installed(tmp_path)[1])
    assert [(o[0], o[1][1:]) for o in outs] == [("pred_boxes", [6, 4]), ("pred_logits", [6, 3])]


def test_labels_count_checked(tmp_path):
    m = _classifier()
    src = tmp_path / "cls.keras"
    m.save(src)
    labels = tmp_path / "labels.txt"
    labels.write_text("a\nb\nc\nd\ne\n")
    assert _run(tmp_path, "keras", src, "--labels", str(labels)) == 0
    labels.write_text("a\nb\n")
    assert _run(tmp_path, "keras", src, "--labels", str(labels)) == 2


# ------------------------------------------------------------------------------------------------
# refusals
# ------------------------------------------------------------------------------------------------

def test_quantized_tflite_refused(tmp_path, capsys):
    m = _classifier(dynamic=False)
    src = _tflite(m, tmp_path / "q.tflite", quant="int8-io")
    assert _run(tmp_path, "tflite", src) == 2
    assert "quantized" in capsys.readouterr().err


def test_wrong_output_count_refused(tmp_path, capsys):
    m = _detector()
    src = tmp_path / "det.keras"
    m.save(src)
    assert _run(tmp_path, "keras", src, task="classification") == 2
    err = capsys.readouterr().err
    assert "needs exactly 1 output" in err and "pred_boxes" in err


def test_wrong_output_shape_refused(tmp_path, capsys):
    """Two outputs but rank-2, no [1,Q,4] boxes tensor -> the rt-detr contract refuses it."""
    tf.random.set_seed(0)
    inp = K.Input((H, W, 3))
    x = K.layers.GlobalAveragePooling2D()(inp)
    m = K.Model(inp, {"a": K.layers.Dense(3)(x), "b": K.layers.Dense(4)(x)})
    src = tmp_path / "bad.keras"
    m.save(src)
    assert _run(tmp_path, "keras", src, task="detection") == 2
    assert "needs outputs boxes [1,Q,4]" in capsys.readouterr().err


@pytest.mark.parametrize("lic", [None, "AGPL-3.0"])
def test_license_refused(tmp_path, capsys, lic):
    m = _classifier()
    src = tmp_path / "cls.keras"
    m.save(src)
    assert _run(tmp_path, "keras", src, license=lic) == 2
    assert "license" in capsys.readouterr().err


def test_fixed_input_size_mismatch_refused(tmp_path, capsys):
    m = _classifier(dynamic=False)
    src = tmp_path / "fixed.keras"
    m.save(src)
    argv = ["keras", str(src), "--name", "m", "--task", "classification", "--input", "64x64",
            "--license", "MIT", "--models", str(tmp_path / "models"), "--dry-run"]
    assert cli.main(argv) == 2
    assert "fixed to" in capsys.readouterr().err
