"""Offline tests for convert/precision.py: FP16, INT8 QDQ, sensitivity, mixed-precision search and the
metric. Tiny hand-built ONNX graphs + onnxruntime/onnx/numpy/PIL only — no torch, no checkpoint, no
network. Each test states the behaviour it guards; none pins "whatever the code printed once"."""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # clients/python

from visionserve.convert import precision as P  # noqa: E402
from visionserve.convert.common import Bundle, ConvertError  # noqa: E402

IMN_MEAN, IMN_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _importable(mod, attr=None):
    try:
        m = __import__(mod, fromlist=["_"])
        return attr is None or hasattr(m, attr)
    except Exception:  # noqa: BLE001
        return False


# The converter image's TensorFlow venv (onnxruntime 1.18, ml_dtypes 0.3) lacks parts of this; the
# PyTorch venv and a normal environment have all of it. A test needing a missing piece is skipped there,
# and test_check_environment_* pins that the converter refuses the feature up front instead.
needs_int4 = pytest.mark.skipif(not _importable("onnxruntime.quantization.matmul_nbits_quantizer"),
                                reason="onnxruntime has no MatMulNBitsQuantizer (TensorFlow venv)")
needs_fp4 = pytest.mark.skipif(not _importable("ml_dtypes", "float4_e2m1fn"),
                               reason="ml_dtypes < 0.5 has no float4 (TensorFlow venv)")


# --------------------------------------------------------------------------------------------
# fixtures: tiny graphs
# --------------------------------------------------------------------------------------------

def _model(nodes, inputs, outputs, inits):
    g = helper.make_graph(nodes, "g", inputs, outputs, [numpy_helper.from_array(v, k) for k, v in inits.items()])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    return m


def _vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def two_branch(path, rng):
    """out = a @ Wa + b @ Wb (nodes named mm_a / mm_b, then an unnamed Add)."""
    nodes = [helper.make_node("MatMul", ["a", "Wa"], ["ya"], name="mm_a"),
             helper.make_node("MatMul", ["b", "Wb"], ["yb"], name="mm_b"),
             helper.make_node("Add", ["ya", "yb"], ["out"])]
    m = _model(nodes, [_vi("a", [1, 16]), _vi("b", [1, 16])], [_vi("out", [1, 8])],
               {"Wa": rng.normal(size=(16, 8)).astype(np.float32), "Wb": rng.normal(size=(16, 8)).astype(np.float32)})
    onnx.save(m, str(path))
    return path


def image_mlp(path, rng, side=8):
    """[1,3,side,side] -> flatten -> MatMul -> Relu -> MatMul -> [1,4]; the shape of a toy classifier."""
    n = 3 * side * side
    nodes = [helper.make_node("Flatten", ["input"], ["f"], axis=1),
             helper.make_node("MatMul", ["f", "W1"], ["h"], name="fc1"),
             helper.make_node("Relu", ["h"], ["r"], name="relu"),
             helper.make_node("MatMul", ["r", "W2"], ["out"], name="fc2")]
    m = _model(nodes, [_vi("input", [1, 3, side, side])], [_vi("out", [1, 4])],
               {"W1": (rng.normal(size=(n, 32)) / math.sqrt(n)).astype(np.float32),
                "W2": (rng.normal(size=(32, 4)) / math.sqrt(32)).astype(np.float32)})
    onnx.save(m, str(path))
    return path


def write_images(d: Path, n=12, side=8, seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, (side, side, 3), dtype=np.uint8)).save(d / f"i{i}.png")
    return d


def run(path, feeds):
    return P._session(path).run(None, feeds)


# --------------------------------------------------------------------------------------------
# metric
# --------------------------------------------------------------------------------------------

def _detr(boxes, scores_by_class):
    """(boxes [Q,4] cxcywh, logits [Q,C]) where scores_by_class[q] = (class, prob)."""
    q, c = len(boxes), 3
    logits = np.full((q, c), -8.0)
    for i, (k, p) in enumerate(scores_by_class):
        if k is not None:
            logits[i, k] = math.log(p / (1 - p))
    return np.array(boxes, float), logits


A = _detr([[.3, .3, .2, .2], [.7, .7, .2, .2], [.5, .5, .1, .1]], [(0, .9), (1, .8), (None, 0)])


def test_detection_error_zero_for_identical_and_blind_to_query_order():
    assert P.detection_error(A, A) == 0.0
    perm = [2, 1, 0]
    shuffled = (A[0][perm], A[1][perm])      # a DETR's top-K reshuffles queries: the SET is unchanged
    assert P.detection_error(A, shuffled) == 0.0


def test_detection_error_grows_with_shift_class_flip_and_extra_or_missing_detection():
    shifted = (A[0] + np.array([[.05, 0, 0, 0]] * 3), A[1])
    e_shift = P.detection_error(A, shifted)
    assert 0 < e_shift < 0.5
    flipped = (A[0], A[1][:, [1, 0, 2]])                    # classes 0 <-> 1
    assert P.detection_error(A, flipped) == 1.0             # same-class matching only
    missing = (A[0], A[1].copy())
    missing[1][1] = -8.0                                     # one of two confident detections vanishes
    assert P.detection_error(A, missing) == pytest.approx(0.5)
    assert P.detection_error(missing, A) == pytest.approx(0.5)   # symmetric for an extra one


def test_detr_head_is_recognised_when_the_class_count_is_four():
    """A 3-class fine-tune has 3 + N/A = 4 logits: the logits tensor looks like a box tensor."""
    boxes = np.array([[[.3, .3, .2, .2], [.7, .7, .2, .2]]], np.float32)
    logits = np.full((1, 2, 4), -8.0, np.float32)
    logits[0, 0, 1], logits[0, 1, 2] = 3.0, 3.0
    perm = [1, 0]
    assert P.output_error([boxes, logits], [boxes[:, perm], logits[:, perm]]) == 0.0   # set compare, not L2


def test_detection_error_empty_and_nan_cases():
    none = _detr([[.5, .5, .1, .1]], [(None, 0)])
    assert P.detection_error(none, none) == 0.0
    assert P.detection_error(none, A) == 1.0
    bad = (A[0].copy(), A[1].copy())
    bad[1][0, 0] = np.nan
    assert P.detection_error(A, bad) == math.inf


def test_output_error_uses_detection_error_for_detr_heads_and_l2_otherwise():
    ref = [A[0][None], A[1][None]]
    perm = [2, 0, 1]
    got = [A[0][perm][None], A[1][perm][None]]
    assert P.output_error(ref, got) == 0.0                   # order-blind for a DETR head
    x = np.ones((1, 4), np.float32)
    assert P.output_error([x], [x * 1.1]) == pytest.approx(0.1, rel=1e-5)
    assert P.output_error([x], [x * np.nan]) == math.inf


# --------------------------------------------------------------------------------------------
# graph helpers
# --------------------------------------------------------------------------------------------

def test_ensure_node_names_names_every_node_uniquely_and_keeps_existing_names(tmp_path):
    m = onnx.load(str(two_branch(tmp_path / "m.onnx", np.random.default_rng(0))))
    assert P.ensure_node_names(m) == 1                       # only the Add was unnamed
    names = [n.name for n in m.graph.node]
    assert names[:2] == ["mm_a", "mm_b"] and len(set(names)) == 3 and all(names)


def test_batched_weight_matmuls_flags_non_2d_constants_only(tmp_path):
    nodes = [helper.make_node("MatMul", ["x", "W2"], ["y"], name="ok"),
             helper.make_node("MatMul", ["y", "W4"], ["z"], name="batched")]
    m = _model(nodes, [_vi("x", [1, 4])], [_vi("z", [1, 8, 3, 3])],
               {"W2": np.ones((4, 4), np.float32), "W4": np.ones((1, 8, 4, 3), np.float32)})
    assert P.batched_weight_matmuls(m) == {"batched"}


# --------------------------------------------------------------------------------------------
# FP16
# --------------------------------------------------------------------------------------------

def test_fp16_halves_the_file_keeps_float32_io_and_stays_close(tmp_path):
    rng = np.random.default_rng(1)
    src = image_mlp(tmp_path / "m.onnx", rng)
    dst = P.convert_fp16(src, tmp_path / "m16.onnx")
    assert dst.stat().st_size < 0.6 * src.stat().st_size
    m = onnx.load(str(dst))
    assert [i.type.tensor_type.elem_type for i in m.graph.input] == [TensorProto.FLOAT]
    assert [o.type.tensor_type.elem_type for o in m.graph.output] == [TensorProto.FLOAT]
    x = {"input": rng.normal(size=(1, 3, 8, 8)).astype(np.float32)}
    a, b = run(src, x)[0], run(dst, x)[0]
    assert np.linalg.norm(a - b) / np.linalg.norm(a) < 5e-3


def test_fp16_keep_node_and_op_stay_float32_and_unknown_names_are_refused(tmp_path):
    src = image_mlp(tmp_path / "m.onnx", np.random.default_rng(1))
    dst = P.convert_fp16(src, tmp_path / "m16.onnx", keep_nodes=["fc2"])
    m = onnx.load(str(dst))
    fc2 = next(n for n in m.graph.node if n.name == "fc2")
    w = next(i for i in m.graph.initializer if i.name in fc2.input and i.name != "r")
    assert w.data_type == TensorProto.FLOAT                  # the kept node's weight is untouched
    with pytest.raises(ConvertError, match="no such node"):
        P.convert_fp16(src, tmp_path / "x.onnx", keep_nodes=["nope"])
    with pytest.raises(ConvertError, match="no Softmax"):
        P.convert_fp16(src, tmp_path / "x.onnx", keep_ops=["Softmax"])


def test_fp16_keeps_ops_the_cpu_ep_cannot_run_in_half_precision(tmp_path):
    """ScatterElements(reduction=add) has no FP16 kernel on the CPU EP: a model converted with it in
    half precision would fail to load/run in the converter's own verification."""
    nodes = [helper.make_node("ScatterElements", ["d", "i", "u"], ["y"], reduction="add", axis=1, name="sc")]
    m = _model(nodes, [_vi("d", [1, 4])], [_vi("y", [1, 4])],
               {"i": np.array([[0, 1]], np.int64), "u": np.ones((1, 2), np.float32)})
    m.opset_import[0].version = 18
    src = tmp_path / "s.onnx"
    onnx.save(m, str(src))
    dst = P.convert_fp16(src, tmp_path / "s16.onnx")
    x = {"d": np.zeros((1, 4), np.float32)}
    assert np.allclose(run(dst, x)[0], run(src, x)[0])          # runs on CPU and matches


def test_a_reduced_model_that_cannot_run_gives_an_actionable_error(tmp_path):
    class Boom:
        def run(self, *_):
            raise RuntimeError("[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : no kernel for Foo\nmore")
    with pytest.raises(ConvertError, match="cannot run on ONNX Runtime's CPU EP.*--keep-fp-op"):
        P.run_all(Boom(), [{}], "reduced model")


# --------------------------------------------------------------------------------------------
# INT8
# --------------------------------------------------------------------------------------------

def _feeds(rng, n=16, side=8):
    return [{"input": rng.normal(size=(1, 3, side, side)).astype(np.float32)} for _ in range(n)]


def test_int8_is_qdq_smaller_and_close_to_fp32_and_keep_node_is_not_quantized(tmp_path):
    rng = np.random.default_rng(2)
    src = image_mlp(tmp_path / "m.onnx", rng)
    feeds = _feeds(rng)
    dst = P.quantize_int8(src, tmp_path / "q.onnx", feeds, "minmax", workdir=tmp_path)
    ops = [n.op_type for n in onnx.load(str(dst)).graph.node]
    assert "QuantizeLinear" in ops and "DequantizeLinear" in ops
    x = _feeds(np.random.default_rng(99), 8)
    err = P.compare_models(src, dst, x)
    assert err["err_mean"] < 0.1 and err["non_finite_images"] == 0
    # keeping fc2 in float leaves fewer Q/DQ pairs than quantizing both layers
    dst2 = P.quantize_int8(src, tmp_path / "q2.onnx", feeds, "minmax", keep_nodes=["fc2"], workdir=tmp_path)
    n_q = lambda p: sum(n.op_type == "QuantizeLinear" for n in onnx.load(str(p)).graph.node)  # noqa: E731
    assert n_q(dst2) < n_q(dst)


def test_int8_leaves_batched_constant_matmul_in_float(tmp_path):
    """ORT's QLinearMatMul cannot run a per-channel zero point on a [1,H,Q,D] constant."""
    rng = np.random.default_rng(3)
    nodes = [helper.make_node("MatMul", ["x", "W"], ["y"], name="plain"),
             helper.make_node("MatMul", ["y", "B"], ["z"], name="batched")]
    m = _model(nodes, [_vi("input", [1, 2, 4, 6])], [_vi("z", [1, 2, 4, 5])],
               {"W": rng.normal(size=(6, 6)).astype(np.float32), "B": rng.normal(size=(1, 2, 6, 5)).astype(np.float32)})
    m.graph.node[0].input[0] = "input"
    src = tmp_path / "m.onnx"
    onnx.save(m, str(src))
    feeds = [{"input": rng.normal(size=(1, 2, 4, 6)).astype(np.float32)} for _ in range(8)]
    dst = P.quantize_int8(src, tmp_path / "q.onnx", feeds, "minmax", workdir=tmp_path)
    run(dst, feeds[0])                                       # loads and runs: no QLinearMatMul failure


# --------------------------------------------------------------------------------------------
# sensitivity
# --------------------------------------------------------------------------------------------

def test_sensitivity_ranks_the_layer_with_an_activation_outlier_first(tmp_path):
    """min-max INT8 on an input that carries one huge value crushes every other value of that input
    to ~0, so the layer reading it must be the most sensitive; the layer on a well-behaved input is not."""
    rng = np.random.default_rng(4)
    src = two_branch(tmp_path / "m.onnx", rng)
    feeds = []
    for _ in range(6):
        a = rng.normal(size=(1, 16)).astype(np.float32)
        a[0, 0] = 400.0
        feeds.append({"a": a, "b": rng.normal(size=(1, 16)).astype(np.float32)})
    scores = P.measure_sensitivity(src, feeds, "int8", method="minmax")
    assert [s.name for s in scores] == ["mm_a", "mm_b"] and [s.rank for s in scores] == [1, 2]
    assert scores[0].output_err > 5 * scores[1].output_err
    by = {s.name: s for s in scores}
    assert by["mm_a"].act_outlier_ratio > by["mm_b"].act_outlier_ratio     # the stat that explains it
    assert P.select_keep(scores, 1) == ["mm_a"] and P.select_keep(scores, 0) == []


def test_sensitivity_does_not_modify_the_source_model(tmp_path):
    src = two_branch(tmp_path / "m.onnx", np.random.default_rng(5))
    before = src.read_bytes()
    rng = np.random.default_rng(6)
    P.measure_sensitivity(src, [{"a": rng.normal(size=(1, 16)).astype(np.float32),
                                 "b": rng.normal(size=(1, 16)).astype(np.float32)}], "fp16")
    assert src.read_bytes() == before


def test_sensitivity_payload_is_json_serialisable_even_with_inf(tmp_path):
    s = P.LayerScore("n", "MatMul", math.inf, -math.inf, None, 1.0, 2.0, 0, 1)
    payload = P.sensitivity_payload([s], "int8", "minmax", 1, kept=["n"])
    import json
    out = json.loads(P.save_sensitivity(payload, tmp_path).read_text())
    assert out["layers"][0]["output_err"] is None and out["kept_in_float"] == ["n"]


# --------------------------------------------------------------------------------------------
# mixed-precision search
# --------------------------------------------------------------------------------------------

def test_search_keep_finds_the_fewest_layers_meeting_the_target():
    ranked = [f"L{i}" for i in range(10)]
    sens = {"layers": [{"name": n} for n in ranked]}
    calls = []

    def build(keep):                       # error falls linearly as more layers are kept in float
        calls.append(len(keep))
        return Path("m.onnx"), {"err_mean": 1.0 - 0.1 * len(keep)}
    keep, dst, extra = P._search_keep(build, sens, [], 0.35)
    assert keep == ranked[:7] and extra["err_mean"] == pytest.approx(0.3)
    assert len(calls) < 10                 # doubling + bisection, not a linear scan


def test_search_keep_respects_pinned_nodes_and_refuses_an_unreachable_target():
    sens = {"layers": [{"name": "a"}, {"name": "b"}]}
    keep, _, _ = P._search_keep(lambda k: (Path("m"), {"err_mean": 0.0 if "pin" in k else 1.0}), sens, ["pin"], 0.5)
    assert keep == ["pin"]
    with pytest.raises(ConvertError, match="cannot be met"):
        P._search_keep(lambda k: (Path("m"), {"err_mean": 0.9}), sens, [], 0.1)


# --------------------------------------------------------------------------------------------
# apply() on a bundle, and the argument checks
# --------------------------------------------------------------------------------------------

def _args(**kw):
    a = dict(precision="fp32", formats="int8,fp16", sens_formats=None, int4_algo="rtn", int4_block=32, calib=None, images=None, calib_n=32, calib_method="minmax", sensitivity=False,
             sens_images=4, keep_fp_top=0, keep_fp_node=[], keep_fp_op=[], max_output_err=0.0, eval=None,
             threshold=None)
    a.update(kw)
    return types.SimpleNamespace(**a)


def _bundle(path, side=8):
    return Bundle(name="t", task="classification", architecture="efficientnet", license="Apache-2.0",
                  width=side, height=side, onnx={"model": str(path)}, mean=IMN_MEAN, std=IMN_STD)


def test_apply_is_a_noop_without_a_precision_flag(tmp_path):
    src = image_mlp(tmp_path / "m.onnx", np.random.default_rng(7))
    b = _bundle(src)
    res = P.apply([b], _args(), tmp_path)
    assert not res.active and b.onnx["model"] == str(src)


def test_apply_fp16_and_int8_replace_the_model_and_grade_the_result(tmp_path):
    src = image_mlp(tmp_path / "m.onnx", np.random.default_rng(8))
    cal = write_images(tmp_path / "cal")
    for prec in ("fp16", "int8"):
        b = _bundle(src)
        res = P.apply([b], _args(precision=prec, calib=str(cal)), tmp_path / prec)
        assert b.onnx["model"].endswith(f"-{prec}.onnx") and Path(b.onnx["model"]).exists()
        row = next(r for r in res.rows if r.title.startswith(prec))
        assert row.status in ("PASS", "WARN") and row.metrics["err_mean"] < 0.25
        assert any("not measured on labelled data" in r.summary for r in res.rows)   # no --eval: said so
        assert any(f"precision {prec}" in n for n in b.notes)


def test_apply_with_eval_does_not_warn_about_unmeasured_accuracy(tmp_path):
    src = image_mlp(tmp_path / "m.onnx", np.random.default_rng(8))
    cal = write_images(tmp_path / "cal")
    res = P.apply([_bundle(src)], _args(precision="fp16", calib=str(cal), eval="x.json"), tmp_path)
    assert not any("not measured" in r.summary for r in res.rows)


def test_apply_fails_a_reduction_that_breaks_the_model(tmp_path):
    """A reduced model far from FP32 must FAIL the P row (and so be uninstalled), not just be reported."""
    src = image_mlp(tmp_path / "m.onnx", np.random.default_rng(9))
    cal = write_images(tmp_path / "cal")
    res = P.apply([_bundle(src)], _args(precision="int8", calib=str(cal), threshold=["p_err_fail=1e-9",
                                                                                       "p_err_warn=1e-10"]),
                  tmp_path)
    assert next(r for r in res.rows if r.title.startswith("int8")).status == "FAIL"


def test_apply_sensitivity_writes_payload_and_keep_top_marks_layers(tmp_path):
    src = image_mlp(tmp_path / "m.onnx", np.random.default_rng(10))
    cal = write_images(tmp_path / "cal")
    res = P.apply([_bundle(src)], _args(precision="int8", calib=str(cal), keep_fp_top=1), tmp_path)
    assert res.sensitivity and len(res.sensitivity["layers"]) == 2
    assert res.sensitivity["kept_in_float"] == [res.sensitivity["layers"][0]["name"]]


def test_apply_refuses_multi_session_bundles(tmp_path):
    src = image_mlp(tmp_path / "m.onnx", np.random.default_rng(11))
    b = _bundle(src)
    b.onnx = {"encoder": str(src), "decoder": str(src)}
    with pytest.raises(ConvertError, match="single-session"):
        P.apply([b], _args(precision="fp16"), tmp_path)


def test_validate_precision_args():
    P.validate_precision_args(_args())                                         # defaults are fine
    with pytest.raises(ConvertError, match="calibration images"):
        P.validate_precision_args(_args(precision="int8"))
    with pytest.raises(ConvertError, match="--keep-fp"):
        P.validate_precision_args(_args(keep_fp_op=["Softmax"]))
    with pytest.raises(ConvertError, match="--max-output-err needs"):
        P.validate_precision_args(_args(max_output_err=0.1, calib="."))
    with pytest.raises(ConvertError, match="finite"):
        P.validate_precision_args(_args(precision="fp16", max_output_err=float("nan"), calib="."))
    with pytest.raises(ConvertError, match="DIRECTORY"):
        P.validate_precision_args(_args(precision="int8", calib=__file__))


def test_cli_parser_exposes_the_precision_flags_and_defaults_to_fp32():
    from visionserve.convert import cli
    ns = cli.build_parser().parse_args(["rfdetr", "x.pth", "--name", "n"])
    assert ns.precision == "fp32" and ns.sensitivity is False and ns.keep_fp_top == 0 and ns.max_output_err == 0
    ns = cli.build_parser().parse_args(["rfdetr", "x.pth", "--name", "n", "--precision", "int8", "--calib", "d",
                                        "--keep-fp-op", "Softmax", "--keep-fp-op", "LayerNormalization",
                                        "--max-output-err", "0.05"])
    assert ns.precision == "int8" and ns.keep_fp_op == ["Softmax", "LayerNormalization"] and ns.max_output_err == 0.05


# --------------------------------------------------------------------------------------------
# INT4 / FP8 / FP4: round trips, builders, mixed assignment
# --------------------------------------------------------------------------------------------

@needs_fp4
def test_weight_fakequant_formats_have_the_error_ordering_of_their_bit_widths():
    rng = np.random.default_rng(20)
    w = rng.normal(size=(64, 16)).astype(np.float32)
    rel = lambda f: float(np.linalg.norm(P.weight_fakequant(w, 1, f) - w) / np.linalg.norm(w))  # noqa: E731
    e = {f: rel(f) for f in ("fp16", "int8", "fp8", "int4", "fp4")}
    assert e["fp16"] < e["int8"] < e["fp8"] < e["int4"]       # 11 > 8 > 4 effective mantissa/levels
    assert e["fp4"] > e["int8"]
    assert e["fp16"] < 1e-3 and e["int4"] < 0.2


def test_int4_blocks_run_along_k_and_a_block_of_equal_values_is_exact():
    w = np.zeros((64, 3), np.float32)
    w[:32] = 0.5                                              # one K-block per column: a single magnitude
    w[32:] = np.linspace(-1, 1, 32)[:, None]
    q = P.weight_fakequant(w, 1, "int4")
    assert np.allclose(q[:32], 0.5, atol=1e-6) or np.abs(q[:32] - 0.5).max() <= 0.5 / 8   # within one step
    assert np.abs(q[32:] - w[32:]).max() <= 1.0 / 8 + 1e-6    # step = absmax / 8


def test_fp8_roundtrip_matches_onnx_runtime_cast_and_saturates():
    import onnxruntime as ort
    x = np.array([[0.1, 1.0, 3.3, 100.0, 450.0, 1000.0]], np.float32)
    g = helper.make_graph([helper.make_node("Cast", ["x"], ["h"], to=TensorProto.FLOAT8E4M3FN, saturate=1),
                           helper.make_node("Cast", ["h"], ["y"], to=TensorProto.FLOAT)], "g",
                          [_vi("x", [1, 6])], [_vi("y", [1, 6])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 19)])
    m.ir_version = 9
    ref = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    mine = P._fp8_roundtrip(x, 448.0)                         # scale 1: same grid
    assert np.allclose(mine, ref)
    assert mine[0, -1] == 448.0                               # saturates, never NaN


@needs_fp4
def test_fp4_values_sit_on_the_e2m1_grid_per_block():
    rng = np.random.default_rng(21)
    w = rng.normal(size=(16, 4)).astype(np.float32)
    q = P.weight_fakequant(w, 1, "fp4")
    grid = np.array([0, .5, 1, 1.5, 2, 3, 4, 6])
    amax = np.abs(w).max(axis=0)
    scaled = np.abs(q) / (amax / 6.0)
    assert np.abs(scaled[..., None] - grid).min(axis=-1).max() < 1e-5


def test_int4_eligible_is_a_2d_float_constant_b_operand_only(tmp_path):
    nodes = [helper.make_node("MatMul", ["x", "W2"], ["y"], name="ok"),
             helper.make_node("MatMul", ["y", "W4"], ["z"], name="batched"),
             helper.make_node("MatMul", ["x", "y2"], ["u"], name="dynamic")]
    m = _model(nodes, [_vi("x", [1, 4]), _vi("y2", [4, 4])], [_vi("z", [1, 8, 3, 3]), _vi("u", [1, 4])],
               {"W2": np.ones((4, 4), np.float32), "W4": np.ones((1, 8, 4, 3), np.float32)})
    assert P.int4_eligible(m) == ["ok"]


def _mixed_fixture(tmp_path, seed=30):
    rng = np.random.default_rng(seed)
    src = image_mlp(tmp_path / "m.onnx", rng)
    return src, _feeds(rng), rng


@needs_int4
def test_build_mixed_puts_int4_int8_and_fp16_in_one_model_that_runs(tmp_path):
    src, feeds, rng = _mixed_fixture(tmp_path)
    dst = P.build_mixed(src, tmp_path / "mix.onnx", {"fc1": "int4", "fc2": "int8"}, feeds, "minmax",
                        fp16_rest=True, workdir=tmp_path)
    ops = {n.op_type for n in onnx.load(str(dst)).graph.node}
    assert "MatMulNBits" in ops and "QuantizeLinear" in ops and "Cast" in ops   # all three formats present
    err = P.compare_models(src, dst, _feeds(np.random.default_rng(99), 8))
    assert err["err_mean"] < 0.3 and err["non_finite_images"] == 0
    assert dst.stat().st_size < src.stat().st_size


def test_build_mixed_fp32_layer_stays_float_and_unbuildable_requests_are_refused(tmp_path):
    src, feeds, _ = _mixed_fixture(tmp_path)
    dst = P.build_mixed(src, tmp_path / "mix.onnx", {"fc1": "fp32", "fc2": "fp32"}, feeds, "minmax",
                        fp16_rest=False, workdir=tmp_path)
    x = {"input": feeds[0]["input"]}
    assert np.allclose(run(src, x)[0], run(dst, x)[0], atol=1e-6)               # nothing changed
    with pytest.raises(ConvertError, match="sensitivity-only"):
        P.build_mixed(src, tmp_path / "x.onnx", {"fc1": "fp8"}, feeds, workdir=tmp_path)
    with pytest.raises(ConvertError, match="unknown node"):
        P.build_mixed(src, tmp_path / "x.onnx", {"nope": "int8"}, feeds, workdir=tmp_path)


def test_assign_formats_takes_the_most_aggressive_format_within_tau():
    mk = lambda n, **e: P.LayerScore(n, "MatMul", 0, 0, None, 0, 0, errs=e)  # noqa: E731
    scores = [mk("a", int4=0.30, int8=0.02, fp16=0.001), mk("b", int4=0.01, int8=0.005, fp16=0.0),
              mk("c", int8=0.5, fp16=0.2), mk("d")]
    ladder = ["int4", "int8", "fp16"]
    assert P.assign_formats(scores, ladder, 0.05) == {"a": "int8", "b": "int4", "c": "fp32", "d": "fp32"}
    assert P.assign_formats(scores, ladder, 0.0) == {"a": "fp32", "b": "fp16", "c": "fp32", "d": "fp32"}
    assert P.assign_formats(scores, ladder, 1.0)["c"] == "int8"
    assert P.assign_formats(scores, ["int8"], 0.05, keep=["a"]) == {"a": "fp32", "b": "int8", "c": "fp32", "d": "fp32"}


def test_search_assignment_finds_the_largest_tau_meeting_the_target():
    mk = lambda n, e: P.LayerScore(n, "MatMul", 0, 0, None, 0, 0, errs={"int8": e})  # noqa: E731
    scores = [mk(f"L{i}", 0.01 * (i + 1)) for i in range(10)]          # layer i costs 0.01 (i+1) alone
    calls = []

    def build(asg):                                                     # model error = sum of its int8 layers
        n8 = [k for k, v in asg.items() if v == "int8"]
        calls.append(len(n8))
        return Path("m"), {"err_mean": sum(scores[int(k[1:])].errs["int8"] for k in n8)}
    asg, _, extra, tau, steps = P.search_assignment(scores, ["int8"], build, 0.20)
    n8 = sorted(k for k, v in asg.items() if v == "int8")
    assert n8 == ["L0", "L1", "L2", "L3"] and extra["err_mean"] == pytest.approx(0.10) or extra["err_mean"] <= 0.20
    assert extra["err_mean"] <= 0.20
    assert len(calls) <= 6                                              # bisection, not a scan of 11 taus
    with pytest.raises(ConvertError, match="cannot be met"):
        P.search_assignment(scores, ["int8"], lambda a: (Path("m"), {"err_mean": 1.0}), 0.1)


@needs_fp4
def test_sensitivity_measures_every_format_per_layer_and_marks_simulated_ones(tmp_path):
    rng = np.random.default_rng(31)
    src = two_branch(tmp_path / "m.onnx", rng)
    feeds = [{"a": rng.normal(size=(1, 16)).astype(np.float32), "b": rng.normal(size=(1, 16)).astype(np.float32)}
             for _ in range(6)]
    scores = P.measure_sensitivity(src, feeds, formats=["int4", "int8", "fp16", "fp8", "fp4"], method="minmax")
    for s in scores:
        assert set(s.errs) == {"int4", "int8", "fp16", "fp8", "fp4"}     # both MatMuls take every format
        assert s.errs["fp16"] < s.errs["int8"] < s.errs["int4"]
        assert s.errs["fp16"] < s.errs["fp8"] and s.errs["int8"] < s.errs["fp4"]
    p = P.sensitivity_payload(scores, "int4", "minmax", 6, formats=["int4", "fp8"])
    assert p["simulated_formats"] == ["fp8"] and p["layers"][0]["errs"]["int4"] is not None


def test_sensitivity_skips_formats_a_layer_cannot_take(tmp_path):
    """int4 / fp4 are weight-only and need a 2-D constant weight: a Conv, or a MatMul of two
    activations, simply has no entry for them."""
    nodes = [helper.make_node("MatMul", ["a", "b"], ["y"], name="dyn")]
    m = _model(nodes, [_vi("a", [1, 8]), _vi("b", [8, 4])], [_vi("y", [1, 4])], {})
    src = tmp_path / "d.onnx"
    onnx.save(m, str(src))
    rng = np.random.default_rng(32)
    feeds = [{"a": rng.normal(size=(1, 8)).astype(np.float32), "b": rng.normal(size=(8, 4)).astype(np.float32)}]
    s = P.measure_sensitivity(src, feeds, formats=["int4", "int8"], method="minmax")
    assert set(s[0].errs) == {"int8"}


@needs_int4
def test_apply_int4_and_mixed_on_a_bundle(tmp_path):
    src = image_mlp(tmp_path / "m.onnx", np.random.default_rng(33))
    cal = write_images(tmp_path / "cal")
    b = _bundle(src)
    r4 = P.apply([b], _args(precision="int4", calib=str(cal)), tmp_path / "i4")
    assert any(n.op_type == "MatMulNBits" for n in onnx.load(b.onnx["model"]).graph.node)
    assert next(x for x in r4.rows if x.title.startswith("int4")).metrics["err_mean"] < 0.3
    b = _bundle(src)
    rm = P.apply([b], _args(precision="mixed", formats="int4,int8,fp16", calib=str(cal), max_output_err=0.2),
                 tmp_path / "mx")
    asg = rm.sensitivity["assignment"]
    assert set(asg) == {"fc1", "fc2"} and set(asg.values()) <= {"int4", "int8", "fp16", "fp32"}
    row = next(x for x in rm.rows if x.title.startswith("mixed"))
    assert row.metrics["err_mean"] <= 0.2 and "layers" in row.summary


@needs_int4
def test_validate_precision_args_for_mixed_and_formats():
    P.validate_precision_args(_args(precision="mixed", formats="int4,fp16", max_output_err=0.1, calib=str(Path(__file__).parent)))
    with pytest.raises(ConvertError, match="--max-output-err"):
        P.validate_precision_args(_args(precision="mixed", formats="fp16"))
    with pytest.raises(ConvertError, match="--formats"):
        P.validate_precision_args(_args(precision="mixed", formats="fp8", max_output_err=0.1))
    with pytest.raises(ConvertError, match="calibration images"):
        P.validate_precision_args(_args(precision="mixed", formats="int8", max_output_err=0.1))
    with pytest.raises(ConvertError, match="calibration images"):                 # mixed measures sensitivity
        P.validate_precision_args(_args(precision="mixed", formats="int4,fp16", max_output_err=0.1))
    with pytest.raises(ConvertError, match="--sens-formats"):
        P.validate_precision_args(_args(precision="fp16", sens_formats="int3"))
    with pytest.raises(ConvertError, match="power of two"):
        P.validate_precision_args(_args(int4_block=30))


@needs_int4
def test_quantize_int4_touches_only_the_requested_matmuls(tmp_path):
    """ORT's MatMulNBitsQuantizer ignores nodes_to_include in some builds; the others must stay MatMul."""
    src = image_mlp(tmp_path / "m.onnx", np.random.default_rng(34))
    m = P.quantize_int4(P._load(src), ["fc1"], 32, "rtn")
    kinds = {n.name: n.op_type for n in m.graph.node if "fc" in n.name}
    assert kinds == {"fc1_Q4": "MatMulNBits", "fc2": "MatMul"}


def test_cli_parser_exposes_mixed_and_int4_flags():
    from visionserve.convert import cli
    ns = cli.build_parser().parse_args(["rfdetr", "x.pth", "--name", "n", "--precision", "mixed", "--formats",
                                        "int4,int8,fp16", "--sens-formats", "int4,fp8,fp4", "--int4-algo", "hqq",
                                        "--int4-block", "64", "--max-output-err", "0.05", "--calib", "d"])
    assert (ns.precision, ns.formats, ns.sens_formats, ns.int4_algo, ns.int4_block) == \
        ("mixed", "int4,int8,fp16", "int4,fp8,fp4", "hqq", 64)
    d = cli.build_parser().parse_args(["rfdetr", "x.pth", "--name", "n"])
    assert d.formats == "int8,fp16" and d.sens_formats is None and d.int4_algo == "rtn" and d.int4_block == 32


def test_check_environment_names_the_missing_piece_before_any_work(monkeypatch):
    """A venv without a feature (the TensorFlow one: onnxruntime 1.18) must fail up front with a message
    that says what is missing, not halfway through an export."""
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("onnxruntime.quantization.matmul_nbits_quantizer") or name == "ml_dtypes":
            raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ConvertError, match=r"cannot run: INT4 \(MatMulNBitsQuantizer\).*TensorFlow"):
        P.check_environment(_args(precision="int4"))
    with pytest.raises(ConvertError, match="FP4 simulation"):
        P.check_environment(_args(precision="fp16", sensitivity=True, sens_formats="int8,fp4"))
    P.check_environment(_args(precision="fp16"))                    # fp16 alone needs none of the missing pieces
    P.check_environment(_args())                                    # fp32, no sensitivity: nothing to check


def test_mixed_with_int8_and_fp16_only_needs_nothing_from_the_int4_stack(tmp_path):
    """--formats int8,fp16 must work where MatMulNBits does not exist (the TensorFlow venv)."""
    src, feeds, _ = _mixed_fixture(tmp_path, seed=40)
    dst = P.build_mixed(src, tmp_path / "mix.onnx", {"fc1": "int8", "fc2": "fp16"}, feeds, "minmax",
                        fp16_rest=True, workdir=tmp_path)
    ops = {n.op_type for n in onnx.load(str(dst)).graph.node}
    assert "QuantizeLinear" in ops and "MatMulNBits" not in ops
    assert P.compare_models(src, dst, _feeds(np.random.default_rng(41), 4))["err_mean"] < 0.3


def _mlp_with_bias(path, rng, side=8):
    """flatten -> MatMul -> Add(bias) -> Relu -> MatMul -> Add(bias): the shape of a transformer's linear layers."""
    n = 3 * side * side
    nodes = [helper.make_node("Flatten", ["input"], ["f"], axis=1),
             helper.make_node("MatMul", ["f", "W1"], ["h0"], name="fc1"),
             helper.make_node("Add", ["h0", "b1"], ["h"], name="add1"),
             helper.make_node("Relu", ["h"], ["r"], name="relu"),
             helper.make_node("MatMul", ["r", "W2"], ["o0"], name="fc2"),
             helper.make_node("Add", ["o0", "b2"], ["out"], name="add2")]
    m = _model(nodes, [_vi("input", [1, 3, side, side])], [_vi("out", [1, 4])],
               {"W1": (rng.normal(size=(n, 32)) / math.sqrt(n)).astype(np.float32),
                "b1": rng.normal(size=(32,)).astype(np.float32),
                "W2": (rng.normal(size=(32, 4)) / math.sqrt(32)).astype(np.float32),
                "b2": rng.normal(size=(4,)).astype(np.float32)})
    onnx.save(m, str(path))
    return path


@needs_int4
@pytest.mark.parametrize("assign", [{"fc1": "int4", "fc2": "fp16"}, {"fc1": "int8", "fc2": "int4"},
                                    {"fc1": "int4", "fc2": "int8"}, {"fc1": "fp16", "fc2": "int4"}])
def test_mixed_with_fp16_rest_loads_when_a_quantized_matmul_feeds_a_bias_add(tmp_path, assign):
    """Regression: with INT4 + the FP16 rest, the Add after a MatMulNBits read an FP32 tensor and an FP16
    bias ('Type parameter (T) of Optype (Add) bound to different types') and the model failed to load."""
    rng = np.random.default_rng(50)
    src = _mlp_with_bias(tmp_path / "m.onnx", rng)
    feeds = _feeds(rng)
    dst = P.build_mixed(src, tmp_path / "mix.onnx", assign, feeds, "minmax", fp16_rest=True, workdir=tmp_path)
    err = P.compare_models(src, dst, _feeds(np.random.default_rng(51), 6))      # loads and runs
    assert err["err_mean"] < 0.3 and err["non_finite_images"] == 0
