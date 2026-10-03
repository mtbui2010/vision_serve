"""Shared-core checks that every family relies on."""
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from visionserve.convert.common import (Bundle, ConvertError, canonical_license, check_contract, parity,
                                        resolve_license)


def _identity_onnx(tmp_path):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    g = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g", [x], [y])
    p = tmp_path / "id.onnx"
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), str(p))
    return p


def test_parity_rejects_nan_the_reference_does_not_have(tmp_path):
    # NaN - x is NaN and `NaN > tol` is False: a naive max|d| check would PASS this graph.
    p = _identity_onnx(tmp_path)
    feed = {"x": np.array([[1.0, np.nan, 3.0]], np.float32)}
    with pytest.raises(ConvertError, match="NaN/inf"):
        parity(p, feed, [np.array([[1.0, 2.0, 3.0]])], tol=1e-3)


def test_parity_accepts_matching_infinities(tmp_path):
    # e.g. GroundingDINO's masked logits: -inf in both graphs at the same positions.
    p = _identity_onnx(tmp_path)
    feed = {"x": np.array([[1.0, -np.inf, 3.0]], np.float32)}
    assert parity(p, feed, [np.array([[1.0, -np.inf, 3.0]])], tol=1e-3) == 0.0


def test_parity_rejects_real_difference(tmp_path):
    p = _identity_onnx(tmp_path)
    with pytest.raises(ConvertError, match="differs"):
        parity(p, {"x": np.ones((1, 3), np.float32)}, [np.zeros((1, 3))], tol=1e-3)


def test_license_gate():
    assert canonical_license("apache-2.0") == "Apache-2.0"
    for bad in ("AGPL-3.0", "GPL-3.0", "", None, "cc-by-nc-4.0"):
        with pytest.raises(ConvertError):
            canonical_license(bad)
    with pytest.raises(ConvertError):   # a copyleft card cannot be laundered by --license
        resolve_license("MIT", "agpl-3.0", "model card")
    with pytest.raises(ConvertError):   # two permissive ids must still agree
        resolve_license("MIT", "apache-2.0", "model card")


def _depth_onnx(tmp_path, h, w):
    x = helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, h, w])
    y = helper.make_tensor_value_info("predicted_depth", TensorProto.FLOAT, [1, h, w])
    g = helper.make_graph([helper.make_node("ReduceMean", ["pixel_values"], ["predicted_depth"], axes=[1],
                                            keepdims=0)], "g", [x], [y])
    p = tmp_path / f"depth_{h}_{w}.onnx"
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), str(p))
    return p


def test_bundle_renders_keep_aspect_only_when_set(tmp_path):
    p = _depth_onnx(tmp_path, "height", "width")
    b = Bundle(name="d", task="depth", architecture="midas", license="Apache-2.0", width=518, height=518,
               onnx={"model": str(p)}, keep_aspect=True, multiple_of=14)
    m = b.render_manifest()
    assert "  keep_aspect: true\n  multiple_of: 14\n" in m
    plain = Bundle(name="d", task="depth", architecture="midas", license="MIT", width=256, height=256,
                   onnx={"model": str(p)}).render_manifest()
    assert "keep_aspect" not in plain and "multiple_of" not in plain


def test_contract_keep_aspect_needs_dynamic_hw(tmp_path):
    dyn = _depth_onnx(tmp_path, "height", "width")
    fixed = _depth_onnx(tmp_path, 518, 518)
    kw = dict(name="d", task="depth", architecture="midas", license="MIT", width=518, height=518)
    check_contract(Bundle(onnx={"model": str(dyn)}, keep_aspect=True, multiple_of=14, **kw))
    check_contract(Bundle(onnx={"model": str(fixed)}, **kw))           # squash: a fixed graph is fine
    with pytest.raises(ConvertError, match="dynamic height/width"):
        check_contract(Bundle(onnx={"model": str(fixed)}, keep_aspect=True, multiple_of=14, **kw))
