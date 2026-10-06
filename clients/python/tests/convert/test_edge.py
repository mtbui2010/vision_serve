"""Offline tests for convert/edge.py (visionserve sensitivity / optimize) and edgereport.py.

A tiny hand-built image model installed in a temporary registry, random photos, onnxruntime on CPU.
The server-side measurement (latency, served mAP) needs the Go binary and is exercised by the real
runs in the guide, not here; the decision rule it feeds is tested on its own."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("yaml")
PIL = pytest.importorskip("PIL")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # clients/python

from visionserve.convert import edge as E  # noqa: E402
from visionserve.convert.common import ConvertError  # noqa: E402
from visionserve.convert.edgereport import FAIL, PASS, WARN, Report, Row, Table, bar_chart  # noqa: E402

needs_int4 = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("onnxruntime.quantization.matmul_nbits_quantizer"),
    reason="onnxruntime has no MatMulNBitsQuantizer")


# --------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------

def tiny_classifier(path: Path, seed: int = 0) -> None:
    """[1,3,32,32] -> Conv(8) -> Relu -> GlobalAveragePool -> Flatten -> MatMul [8,32] -> Relu -> MatMul [32,4]."""
    rng = np.random.default_rng(seed)
    inits = {"W": rng.normal(0, 0.5, (8, 3, 3, 3)).astype(np.float32), "B": np.zeros(8, np.float32),
             "M1": rng.normal(0, 0.5, (8, 32)).astype(np.float32), "M2": rng.normal(0, 0.5, (32, 4)).astype(np.float32)}
    nodes = [helper.make_node("Conv", ["input", "W", "B"], ["c"], name="conv", pads=[1, 1, 1, 1]),
             helper.make_node("Relu", ["c"], ["r"], name="relu"),
             helper.make_node("GlobalAveragePool", ["r"], ["g"], name="gap"),
             helper.make_node("Flatten", ["g"], ["f"], name="flat"),
             helper.make_node("MatMul", ["f", "M1"], ["h"], name="fc1"),
             helper.make_node("Relu", ["h"], ["h2"], name="relu2"),
             helper.make_node("MatMul", ["h2", "M2"], ["logits"], name="fc2")]
    g = helper.make_graph(nodes, "tiny", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])],
                          [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4])],
                          [numpy_helper.from_array(v, k) for k, v in inits.items()])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, str(path))


MANIFEST = """# a hand-written manifest with a comment
name: tiny
task: classification
license: Apache-2.0
architecture: efficientnet
model_file: tiny.onnx
sha256: 0000000000000000000000000000000000000000000000000000000000000000
input:
  width: 32
  height: 32
  layout: NCHW
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
postprocess:
  type: classification
  max_detections: 4
labels: labels.txt
runtime:
  prefer: [cuda, cpu]
"""


@pytest.fixture()
def registry(tmp_path):
    reg = tmp_path / "models"
    d = reg / "tiny-dir"           # the directory name need not be the model name
    d.mkdir(parents=True)
    tiny_classifier(d / "tiny.onnx")
    tiny_classifier(d / "unused-old.onnx", seed=1)
    (d / "manifest.yaml").write_text(MANIFEST)
    (d / "labels.txt").write_text("a\nb\nc\nd\n")
    imgs = tmp_path / "photos"
    imgs.mkdir()
    rng = np.random.default_rng(3)
    from PIL import Image
    for i in range(8):
        Image.fromarray(rng.integers(0, 255, (40 + i, 50, 3), dtype=np.uint8)).save(imgs / f"p{i}.png")
    return reg, imgs


# --------------------------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------------------------

def test_presets_are_consistent():
    assert set(E.PRESETS) == {"jetson-orin", "jetson-thor", "cuda", "cpu"}
    for p in E.PRESETS.values():
        assert set(p.candidates) <= {"fp16", "int8", "mixed", "int4"}
        assert set(p.mixed_formats) <= {"int4", "int8", "fp16"}
        assert p.host_ep in ("gpu", "cpu")
    # honesty: Thor is not claimed as tested, and FP16 is not offered for a CPU target
    assert "UNTESTED" in E.PRESETS["jetson-thor"].verified
    assert "fp16" not in E.PRESETS["cpu"].candidates
    assert len(E.preset_table_rows()) == len(E.PRESETS)


# --------------------------------------------------------------------------------------------
# the installed model
# --------------------------------------------------------------------------------------------

def test_find_model_by_manifest_name(registry):
    reg, _ = registry
    m = E.find_model(reg, "tiny")
    assert m.dir.name == "tiny-dir" and m.onnx.name == "tiny.onnx" and m.labels == ["a", "b", "c", "d"]
    assert m.input_shape == [1, 3, 32, 32] and m.role_key == "model_file"
    with pytest.raises(ConvertError, match="not found"):
        E.find_model(reg, "nope")


def test_multi_session_models_are_refused(tmp_path):
    d = tmp_path / "sam"
    d.mkdir()
    (d / "manifest.yaml").write_text("name: sam\nfiles:\n  encoder: e.onnx\n  decoder: d.onnx\n")
    with pytest.raises(ConvertError, match="2 ONNX sessions"):
        E.find_model(tmp_path, "sam")


def test_stage_variant_rewrites_only_the_model_file(registry, tmp_path):
    import yaml
    reg, _ = registry
    m = E.find_model(reg, "tiny")
    src = tmp_path / "built.onnx"
    tiny_classifier(src, seed=7)
    v = E.Variant("fp16", path=src, size=src.stat().st_size)
    d = E.stage_variant(m, v, "tiny-fp16", tmp_path / "out", ["precision: fp16 (test)"])
    doc = yaml.safe_load((d / "manifest.yaml").read_text())
    assert doc["name"] == "tiny-fp16" and doc["model_file"] == "tiny-fp16.onnx"
    from visionserve.convert.common import sha256
    assert doc["sha256"] == sha256(d / "tiny-fp16.onnx")
    # preprocessing, labels, postprocess and runtime are the source's
    src_doc = yaml.safe_load(MANIFEST)
    for k in ("input", "postprocess", "labels", "runtime", "license", "architecture", "task"):
        assert doc[k] == src_doc[k], k
    assert (d / "labels.txt").read_text() == "a\nb\nc\nd\n"
    assert sorted(p.name for p in d.glob("*.onnx")) == ["tiny-fp16.onnx"]     # no stale graphs copied
    assert "precision: fp16 (test)" in (d / "manifest.yaml").read_text().splitlines()[1]


def test_stage_variant_files_map(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    tiny_classifier(d / "m.onnx")
    (d / "manifest.yaml").write_text(MANIFEST.replace("model_file: tiny.onnx", "files:\n  model: m.onnx").replace(
        "sha256: " + "0" * 64, "sha256:\n  model: " + "0" * 64))
    (d / "labels.txt").write_text("a\nb\nc\nd\n")
    m = E.find_model(tmp_path, "tiny")
    assert m.role_key == "files.model"
    v = E.Variant("int8", path=d / "m.onnx")
    import yaml
    out = E.stage_variant(m, v, "tiny-int8", tmp_path / "o", [])
    doc = yaml.safe_load((out / "manifest.yaml").read_text())
    assert doc["files"] == {"model": "m-int8.onnx"} and len(doc["sha256"]["model"]) == 64


# --------------------------------------------------------------------------------------------
# sensitivity, end to end on the tiny model
# --------------------------------------------------------------------------------------------

def test_sensitivity_json_and_saved_scores(registry, tmp_path, capsys):
    reg, imgs = registry
    save = tmp_path / "s.json"
    report = tmp_path / "s.html"
    code = E.main(["sensitivity", "tiny", "--models", str(reg), "--images", str(imgs), "--formats", "int8,fp16",
                   "--sens-images", "4", "--save", str(save), "--json", "--report", str(report)])
    out = capsys.readouterr().out
    doc = json.loads(out)                    # stdout is ONE JSON object
    assert set(doc) == {"verdict", "reason", "summary", "details"}
    assert doc["verdict"] in (PASS, WARN) and code == 0
    assert doc["summary"]["layers"] == 3          # conv, fc1, fc2
    assert doc["summary"]["formats"] == ["int8", "fp16"]   # the order asked for: int8 ranks
    names = [l["name"] for l in doc["details"]["layers"]]
    assert sorted(names) == ["conv", "fc1", "fc2"]
    # the saved scores load back and keep the ranking
    scores = E.scores_from_json(json.loads(save.read_text()))
    assert [s.name for s in sorted(scores, key=lambda s: s.rank)] == names
    assert all({"int8", "fp16"} <= set(s.errs) for s in scores)
    html = report.read_text()
    assert "data:image/png;base64," in html and "<script" not in html


def test_sensitivity_verdict_names_the_worst_layer(registry, tmp_path, capsys):
    reg, imgs = registry
    # threshold far below any INT8 error: every layer is "sensitive" -> WARN naming the worst
    code = E.main(["sensitivity", "tiny", "--models", str(reg), "--images", str(imgs), "--sens-images", "2",
                   "--threshold", "1e-9"])
    out = capsys.readouterr().out
    first = out.splitlines()[0]
    assert code == 0 and first.startswith("WARN: 3 of 3 layers are sensitive to INT8; the worst is ")
    assert "keep it in FP32" in first


def test_usage_errors_exit_2(registry, capsys):
    reg, imgs = registry
    assert E.main(["sensitivity", "tiny", "--models", str(reg), "--images", str(imgs), "--formats", "int9"]) == 2
    assert E.main(["sensitivity", "nope", "--models", str(reg), "--images", str(imgs)]) == 2
    assert E.main(["optimize", "tiny", "--models", str(reg), "--images", str(imgs), "--target", "mars"]) == 2
    assert E.main(["optimize", "tiny", "--models", str(reg), "--images", str(imgs), "--target", "cpu",
                   "--max-drop", "nan"]) == 2
    assert E.main(["optimize", "tiny", "--models", str(reg), "--images", str(imgs), "--target", "cpu",
                   "--tensorrt"]) == 2
    assert E.main(["bogus"]) == 2


def test_converter_cli_dispatches(registry, capsys):
    from visionserve.convert import cli
    reg, imgs = registry
    assert cli.main(["sensitivity", "nope", "--models", str(reg), "--images", str(imgs)]) == 2
    assert "not found" in capsys.readouterr().err


# --------------------------------------------------------------------------------------------
# building the candidates
# --------------------------------------------------------------------------------------------

@needs_int4
def test_build_variants_for_cpu(registry, tmp_path):
    reg, imgs = registry
    m = E.find_model(reg, "tiny")
    feeds = E.calibration_feeds(m, imgs, 8)

    class A:
        calib_method, sens_images, max_output_err, gpu = "minmax", 4, 0.5, False
    vs, scores = E.build_variants(m, E.PRESETS["cpu"], feeds, A, tmp_path)
    assert [v.fmt for v in vs] == ["fp32", "int8", "mixed", "int4"]
    for v in vs[1:]:
        assert not v.error, v.error
        assert v.path.is_file() and v.err_mean is not None and math.isfinite(v.err_mean)
    assert scores is not None                      # mixed measured the sensitivity it needed
    assert set(vs[2].assignment.values()) <= {"int8", "fp32"}   # cpu's ladder has no fp16
    assert "MatMulNBits" in {n.op_type for n in onnx.load(str(vs[3].path)).graph.node}


# --------------------------------------------------------------------------------------------
# the decision
# --------------------------------------------------------------------------------------------

class _Args:
    max_output_err, max_drop = 0.05, 1.0


def _v(fmt, size, err, lat, dev="gpu:0", mAP=None, ep="default"):
    return E.Variant(fmt, size=size, err_mean=err, err_max=err, lat_p50=lat, device=dev, map=mAP, ep=ep)


def test_decide_fastest_measured_within_budget_smaller_wins_ties():
    orin = E.PRESETS["jetson-orin"]
    vs = [_v("fp32", 100, 0, 20), _v("fp16", 50, 0.01, 12), _v("int8", 25, 0.30, 30), _v("mixed", 40, 0.04, 12.5)]
    pick, rule, st, _ = E.decide(vs, orin, _Args, have_labels=False)
    assert pick.fmt == "mixed"          # 12.5 is within 10 % of 12 and the file is smaller
    assert st["int8"].startswith("over budget") and st["mixed"] == "recommended ✓"
    assert "fastest measured" in rule


def test_decide_a_reduced_variant_slower_than_fp32_is_not_recommended_for_speed():
    """Measured: the FP16 RF-DETR nano ran 3.5x slower than FP32 on ONNX Runtime's CUDA EP. Smaller is
    not faster; FP32 competes on speed."""
    orin = E.PRESETS["jetson-orin"]
    vs = [_v("fp32", 100, 0, 10), _v("fp16", 50, 0.01, 22), _v("mixed", 49, 0.04, 25)]
    pick, _, st, _ = E.decide(vs, orin, _Args, have_labels=False)
    assert pick.fmt == "fp32" and st["fp16"] == "within budget"


def test_decide_falls_back_to_the_preset_order_when_latency_is_not_comparable():
    orin = E.PRESETS["jetson-orin"]
    # measured on the CPU (no --gpu): the CPU's ordering says nothing about the Jetson's CUDA EP
    vs = [_v("fp32", 100, 0, 20, "cpu"), _v("fp16", 50, 0.01, 90, "cpu"), _v("mixed", 40, 0.04, 15, "cpu")]
    pick, rule, _, _ = E.decide(vs, orin, _Args, have_labels=False)
    assert pick.fmt == "mixed" and "smallest variant" in rule      # CPU timings say nothing about the GPU


def test_decide_accuracy_budget_and_tensorrt_rows():
    orin = E.PRESETS["jetson-orin"]
    vs = [_v("fp32", 100, 0, 20, mAP=50.0), _v("fp16", 50, 0.01, 12, mAP=48.5), _v("mixed", 40, 0.03, 13, mAP=49.6),
          _v("int8", 25, 0.04, 5, "gpu:0+trt", mAP=49.5, ep="tensorrt")]
    pick, _, st, trt = E.decide(vs, orin, _Args, have_labels=True)
    assert "mAP −1.50" in st["fp16"]
    # TensorRT is opt-in on the target: the recommendation comes from the default EP's rows, and the
    # TensorRT row (inside the budget, measured under TensorRT) is a separate, flagged alternative
    assert pick.fmt == "mixed" and pick.ep == "default"
    assert trt.label == "int8 (TensorRT)" and "alternative" in st["int8 (TensorRT)"]
    # without labels a TensorRT row is not even an alternative
    for v in vs:
        v.map = None
    pick, _, st, trt = E.decide(vs, orin, _Args, have_labels=False)
    assert pick.ep == "default" and trt is None and "needs --labels" in st["int8 (TensorRT)"]


def test_decide_keeps_fp32_when_nothing_fits():
    vs = [_v("fp32", 100, 0, 20), _v("int8", 25, 0.3, 10)]
    pick, _, st, _ = E.decide(vs, E.PRESETS["cpu"], _Args, have_labels=False)
    assert pick.fmt == "fp32" and st["int8"].startswith("over budget")


def test_optimize_report_verdicts(registry):
    reg, imgs = registry
    m = E.find_model(reg, "tiny")

    class A(_Args):
        images, gpu, models, labels, target = str(imgs), True, str(reg), None, "jetson-orin"
    orin = E.PRESETS["jetson-orin"]
    vs = [_v("fp32", 100, 0, 20), _v("fp16", 50, 0.01, 12)]
    pick, rule, st, _ = E.decide(vs, orin, A, False)
    rep = E.optimize_report(m, orin, vs, pick, rule, st, A, None, 8, 60, None, None)
    assert rep.verdict == WARN and "accuracy NOT measured" in rep.reason and "2.0× smaller" in rep.reason
    assert "THIS host" in rep.tables[0].header[4]
    assert any("bench tiny-fp16" in s for s in rep.next_steps)
    doc = rep.to_json()
    assert doc["summary"]["recommended"] == "tiny-fp16"
    json.dumps(doc, allow_nan=False)
    # nothing built -> FAIL
    vs = [_v("fp32", 100, 0, 20), E.Variant("fp16", error="boom")]
    pick, rule, st, _ = E.decide(vs, orin, A, False)
    rep = E.optimize_report(m, orin, vs, pick, rule, st, A, None, 8, 60, None, None)
    assert rep.verdict == FAIL and rep.exit_code == 1


# --------------------------------------------------------------------------------------------
# report format (shared with the Go commands)
# --------------------------------------------------------------------------------------------

def test_report_formats():
    r = Report("t", WARN, "one sentence", [Row("x", "x label", float("nan")), Row("y", "y", 3, note="n")],
               [Table("tbl", ["a", "b"], [["1", "2"]], highlight=0)], next_steps=["do it"], details={"k": float("inf")})
    txt = r.text()
    assert txt.splitlines()[0] == "WARN: one sentence"
    assert json.loads(r.json_text()) == {"verdict": "WARN", "reason": "one sentence",
                                         "summary": {"x": None, "y": 3}, "details": {"k": None, "findings": []}}
    h = r.html()
    assert 'class="banner warn"' in h and "prefers-color-scheme: dark" in h and 'class="rec"' in h
    assert "<script" not in h and "<link" not in h and "max-width:960px" in h
    assert r.exit_code == 0 and Report("t", FAIL, "x").exit_code == 1
    assert bar_chart(["a", "b"], [0.1, None], mark=0.05, log=True).startswith(b"\x89PNG")


def test_optimize_without_a_server_fails_with_one_json_object(registry, capsys, monkeypatch):
    """The whole optimize path offline: candidates build, then no VisionServe binary to measure them
    with -> FAIL (exit 1) with the reason, and stdout is still ONE JSON object (the INT8 quantizer's
    progress prints must not leak into it)."""
    reg, imgs = registry
    monkeypatch.setenv("VISIONSERVE_BIN", "/nonexistent/visionserve")
    code = E.main(["optimize", "tiny", "--models", str(reg), "--images", str(imgs), "--target", "jetson-orin",
                   "--calib-n", "4", "--sens-images", "2", "--max-output-err", "0.5", "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert code == 1 and doc["verdict"] == FAIL and "could not be measured" in doc["reason"]
    assert [v["format"] for v in doc["details"]["variants"]] == ["fp32", "fp16", "int8", "mixed"]


def _score(name, rank, **errs):
    from visionserve.convert.precision import LayerScore
    first = next(iter(errs.values()))
    return LayerScore(name, "MatMul", first, 20.0, None, 1.0, 1.0, rank=rank, errs=dict(errs))


def test_complete_scores_measures_only_what_the_file_lacks(registry, monkeypatch):
    """A --sensitivity file from `sensitivity --formats int8` and a ladder int8+fp16: fp16 is
    measured (only fp16), merged per layer, the int8 scores reused; a layer the file did not
    score but fp16 applies to is added after the file's layers."""
    reg, _ = registry
    m = E.find_model(reg, "tiny")
    calls = []

    def fake_measure(m_, feeds, formats, method, gpu):
        calls.append((list(formats), len(feeds), method, gpu))
        return [_score("fc1", 1, fp16=0.001), _score("conv", 2, fp16=0.002), _score("fc2", 3, fp16=0.003)], "CPU"
    monkeypatch.setattr(E, "measure", fake_measure)
    given = [_score("conv", 1, int8=0.1), _score("fc1", 2, int8=0.2)]
    v = E.Variant("mixed")
    merged, measured = E.complete_scores(m, given, ("int8", "fp16"), [{}, {}], "minmax", True, v, "s.json")
    assert calls == [(["fp16"], 2, "minmax", True)] and measured == ("fp16",) and not v.warn
    by = {s.name: s for s in merged}
    assert by["conv"].errs == {"int8": 0.1, "fp16": 0.002} and by["fc1"].errs == {"int8": 0.2, "fp16": 0.001}
    assert by["conv"].rank == 1 and by["fc1"].rank == 2 and by["fc2"].rank == 3 and by["fc2"].errs == {"fp16": 0.003}
    assert given[0].errs == {"int8": 0.1}                       # the caller's scores are not mutated
    # Complete already: nothing measured.
    calls.clear()
    assert E.complete_scores(m, merged, ("int8", "fp16"), [{}], "minmax", False, v, "s.json") == (merged, ())
    assert calls == []


def test_complete_scores_that_cannot_measure_skip_mixed_with_the_fix(registry, monkeypatch):
    reg, _ = registry
    m = E.find_model(reg, "tiny")

    def broken(*a, **k):
        raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(E, "measure", broken)
    v = E.Variant("mixed")
    with pytest.raises(ConvertError):
        E.complete_scores(m, [_score("conv", 1, int8=0.1)], ("int8", "fp16"), [{}], "minmax", False, v, "s.json")
    assert v.warn.startswith("mixed skipped: s.json has no fp16 scores") and "CUDA out of memory" in v.warn
    assert "`visionserve sensitivity tiny --formats int8,fp16 --save FILE`" in v.warn
    # Scores of another model: refused rather than merged into nothing.
    monkeypatch.setattr(E, "measure", lambda *a, **k: ([_score("conv", 1, fp16=0.002)], "CPU"))
    v = E.Variant("mixed")
    with pytest.raises(ConvertError):
        E.complete_scores(m, [_score("other/layer", 1, int8=0.1)], ("int8", "fp16"), [{}], "minmax", False, v, "x.json")
    assert "are not this model's" in v.warn


def test_build_variants_skips_only_mixed_when_measuring_fails(registry, tmp_path, monkeypatch):
    reg, imgs = registry
    m = E.find_model(reg, "tiny")
    feeds = E.calibration_feeds(m, imgs, 4)
    monkeypatch.setattr(E, "measure", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no GPU")))

    class A:
        calib_method, sens_images, max_output_err, gpu, sensitivity = "minmax", 2, 0.5, False, "int8.json"
    vs, _ = E.build_variants(m, E.PRESETS["jetson-orin"], feeds, A, tmp_path, [_score("conv", 1, int8=0.1)])
    by = {v.fmt: v for v in vs}
    assert by["mixed"].error and by["mixed"].warn and "--formats int8,fp16" in by["mixed"].warn
    assert not by["int8"].error and by["int8"].path.is_file()   # the other candidates still built


def test_optimize_with_an_int8_only_sensitivity_file_measures_fp16(registry, tmp_path, capsys, monkeypatch):
    """`optimize --sensitivity FILE` with a file from `sensitivity --formats int8` used to fail the
    mixed candidate ("the sensitivity scores do not cover fp16"); it now measures fp16 itself (real
    precision code on the tiny model) and says so."""
    reg, imgs = registry
    save = tmp_path / "int8.json"
    assert E.main(["sensitivity", "tiny", "--models", str(reg), "--images", str(imgs), "--formats", "int8",
                   "--sens-images", "2", "--save", str(save), "--json"]) == 0
    capsys.readouterr()
    assert all(set(s["errs"]) == {"int8"} for s in json.loads(save.read_text())["layers"])
    monkeypatch.setenv("VISIONSERVE_BIN", "/nonexistent/visionserve")   # no server: the build is what we test
    E.main(["optimize", "tiny", "--models", str(reg), "--images", str(imgs), "--target", "jetson-orin",
            "--calib-n", "4", "--sens-images", "2", "--max-output-err", "0.5", "--sensitivity", str(save), "--json"])
    cap = capsys.readouterr()
    doc = json.loads(cap.out)
    mixed = next(v for v in doc["details"]["variants"] if v["format"] == "mixed")
    assert not mixed["error"], mixed["error"]
    assert "fp16 sensitivity measured here, the rest reused from" in mixed["note"]
    assert any("has no fp16 scores: the mixed candidate measured them here" in f["text"]
               for f in doc["details"]["findings"])
    assert "measuring fp16 here for the mixed candidate" in cap.err


def test_keep_fp32_report_offers_the_smaller_variant(registry):
    reg, imgs = registry
    m = E.find_model(reg, "tiny")

    class A(_Args):
        images, gpu, models, labels, target = str(imgs), True, str(reg), None, "jetson-orin"
    orin = E.PRESETS["jetson-orin"]
    vs = [_v("fp32", 100, 0, 10), _v("fp16", 50, 0.01, 22)]
    pick, rule, st, _ = E.decide(vs, orin, A, False)
    rep = E.optimize_report(m, orin, vs, pick, rule, st, A, None, 8, 60, None, None)
    assert rep.verdict == WARN and rep.reason.startswith("keep FP32 for speed") and "2.0× smaller but 2.2× slower" in rep.reason
    assert any("--install-format fp16" in n for n in rep.next_steps)
    assert any("bench tiny-fp16" in n for n in rep.next_steps)


def test_install_format_is_validated_before_measuring(registry):
    reg, imgs = registry
    base = ["optimize", "tiny", "--models", str(reg), "--images", str(imgs), "--target", "jetson-orin"]
    assert E.main(base + ["--install-format", "fp16"]) == 2                 # needs --install
    assert E.main(base + ["--install", "--install-format", "int4"]) == 2     # not a jetson-orin candidate
