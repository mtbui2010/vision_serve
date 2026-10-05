"""Tests for the `rfdetr` converter family (families/rfdetr.py).

Offline: a randomly initialised RF-DETR Nano (built by the installed `rfdetr`, no download: patch 16
never loads DINOv2 hub weights) is saved as a training checkpoint and run through the real CLI path
with --dry-run. The real 22-class tabletop fine-tune on the NAS is converted when present (skipped
otherwise), and its ONNX is checked against ground-truth boxes to pin the label mapping.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # clients/python

torch = pytest.importorskip("torch")
pytest.importorskip("rfdetr")
ort = pytest.importorskip("onnxruntime")

from visionserve.convert import cli  # noqa: E402
from visionserve.convert.common import IMAGENET_MEAN, IMAGENET_STD, ConvertError, onnx_io, to_nchw  # noqa: E402
from visionserve.convert.families import rfdetr as fam  # noqa: E402

REAL_CKPT = Path("/mnt/nas/huggingface/trung_w6/ovd-adapt-weights/ckpt/final/checkpoint_best_total.pth")
ETRI = Path("/home/trung/trung_workdir/etri_simple")
OFFICIAL = Path.home() / ".roboflow" / "models"


@pytest.fixture(autouse=True)
def _only_rfdetr(monkeypatch):
    monkeypatch.setattr(cli, "FORMATS", {"rfdetr": cli.FORMATS["rfdetr"]})


def run_cli(tmp_path, src, name, *extra):
    models = tmp_path / "models"
    rc = cli.main(["rfdetr", str(src), "--name", name, "--dry-run", "--models", str(models), *extra])
    return rc, models / ".convert-dry-run" / name


# --------------------------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------------------------

def test_labels_finetune_extra_slot_is_last():
    names = [f"c{i}" for i in range(22)]
    assert fam.labels_for(names, 23) == names + ["N/A"]
    assert fam.labels_for(names, 22) == names


def test_labels_coco_are_indexed_by_category_id():
    from rfdetr.assets.coco_classes import COCO_CLASS_NAMES
    out = fam.labels_for(list(COCO_CLASS_NAMES), 91)
    assert len(out) == 91 and out[0] == "N/A" and out[1] == "person" and out[90] == "toothbrush"
    assert out[12] == "N/A"  # a gap in the COCO id space
    # same as the shipped models/rf-detr/coco91.txt wherever that file names a class
    shipped = Path(__file__).resolve().parents[4] / "models/rf-detr/coco91.txt"
    if shipped.exists():
        ref = shipped.read_text().splitlines()
        assert all(a == b for a, b in zip(out, ref) if a != "N/A")


def test_labels_count_mismatch_refused():
    with pytest.raises(ConvertError):
        fam.labels_for(["a", "b"], 5)
    with pytest.raises(ConvertError):
        fam.labels_for(["a", "b", "c"], 2)
    assert fam.labels_for(None, 5) is None


# --------------------------------------------------------------------------------------------
# variant detection
# --------------------------------------------------------------------------------------------

def fake_sd(patch, dim, grid, dec):
    sd = {fam._PE_KEY: torch.empty(1, grid * grid + 1, dim), fam._PATCH_KEY: torch.empty(dim, 3, patch, patch),
          "class_embed.bias": torch.empty(4)}
    for i in range(dec):
        sd[f"transformer.decoder.layers.{i}.norm1.weight"] = torch.empty(1)
    return sd


@pytest.mark.parametrize("variant", sorted(fam._SIGNATURES))
def test_variant_from_shapes(variant):
    sd = fake_sd(*fam._SIGNATURES[variant])
    v, how = fam.detect_variant(sd, {}, None, "checkpoint.pth")
    assert v == variant and "state_dict shapes" in how


def test_variant_precedence_and_refusals():
    sd = fake_sd(*fam._SIGNATURES["small"])
    # metadata that contradicts the shapes loses: those weights can only be built as small
    v, how = fam.detect_variant(sd, {}, "RFDETRMedium", "c.pth")
    assert v == "small" and "overrides checkpoint model_name='RFDETRMedium'" in how
    assert fam.detect_variant(sd, {}, "RFDETRMedium", "c.pth", forced="nano")[0] == "nano"
    assert fam.detect_variant(sd, {"pretrain_weights": "rf-detr-base.pth"}, None, "c.pth")[0] == "small"
    # metadata that agrees with the shapes is what the log names
    assert fam.detect_variant(sd, {}, "RFDETRSmall", "c.pth") == ("small", "checkpoint model_name='RFDETRSmall'")
    # no readable shapes: metadata, then the file name
    assert fam.detect_variant({"class_embed.bias": 0}, {}, None, "rf-detr-nano.pth")[0] == "nano"
    assert fam.detect_variant({"class_embed.bias": 0}, {}, "RFDETRBase", "rf-detr-nano.pth")[0] == "base"
    # custom resolution changes only the PE grid
    assert fam.detect_variant(fake_sd(16, 384, 40, 2), {}, None, "c.pth")[0] == "nano"
    with pytest.raises(ConvertError, match="Platform Model License"):
        fam.detect_variant(sd, {}, "RFDETRXLarge", "c.pth")
    with pytest.raises(ConvertError, match="segmentation"):
        fam.detect_variant(sd, {}, "RFDETRSegSmall", "c.pth")
    with pytest.raises(ConvertError, match="--variant"):
        fam.detect_variant({"class_embed.bias": 0}, {}, None, "checkpoint.pth")


def test_official_base_checkpoint_is_base_despite_its_small_pretrain_hint():
    """rfdetr's own rf-detr-base.pth: args.pretrain_weights names the PRETRAINING checkpoint
    ('lwdetr_dinov2_small_o365_checkpoint.pth', reads as "small"), no model_name, and a patch-14
    DINOv2 backbone (37x37 PE grid, 3 decoder layers) — Base. Following the hint failed the build
    with a patch_size 14 vs 16 mismatch."""
    sd = fake_sd(14, 384, 37, 3)
    args = {"pretrain_weights": "lwdetr_dinov2_small_o365_checkpoint.pth", "encoder": "dinov2_windowed_small",
            "resolution": 560, "dec_layers": 3}
    v, how = fam.detect_variant(sd, args, None, "rf-detr-base.pth")
    assert v == "base"
    assert "state_dict shapes" in how and "lwdetr_dinov2_small_o365_checkpoint.pth" in how
    # the same shapes under a different file name: still base (the shapes decide, not the name)
    assert fam.detect_variant(sd, args, None, "checkpoint.pth")[0] == "base"


def test_ambiguous_shapes_metadata_decides():
    """Medium and Large have the same weights except the PE grid (which follows the training
    resolution), so on shapes that fit both, model_name / args / file name pick."""
    medium_at_704 = fake_sd(16, 384, 44, 4)  # PE grid = Large's default
    assert fam.detect_variant(medium_at_704, {}, None, "c.pth")[0] == "large"  # exact PE grid, nothing else
    assert fam.detect_variant(medium_at_704, {}, "RFDETRMedium", "c.pth")[0] == "medium"
    assert fam.detect_variant(medium_at_704, {"pretrain_weights": "rf-detr-medium.pth"}, None, "c.pth")[0] == "medium"
    large_at_576 = fake_sd(16, 384, 36, 4)  # PE grid = Medium's default
    v, how = fam.detect_variant(large_at_576, {}, "RFDETRLarge", "c.pth")
    assert v == "large" and "medium or large" in how
    # a hint the shapes rule out is skipped; the next one that fits decides
    assert fam.detect_variant(large_at_576, {"pretrain_weights": "rf-detr-large.pth"}, "RFDETRNano", "c.pth")[0] \
        == "large"
    odd = fake_sd(16, 384, 40, 4)  # neither default grid
    assert fam.detect_variant(odd, {}, None, "rf-detr-large-ft.pth")[0] == "large"
    assert fam.detect_variant(odd, {"pretrain_weights": "rf-detr-medium.pth"}, None, "c.pth")[0] == "medium"
    with pytest.raises(ConvertError, match="--variant medium or --variant large"):
        fam.detect_variant(odd, {}, None, "c.pth")
    with pytest.raises(ConvertError, match="--variant medium or --variant large"):
        fam.detect_variant(odd, {}, "RFDETRSmall", "c.pth")  # the only hint is ruled out


def test_explicit_variant_beats_shapes_and_metadata():
    sd = fake_sd(14, 384, 37, 3)  # base shapes
    args = {"pretrain_weights": "lwdetr_dinov2_small_o365_checkpoint.pth"}
    assert fam.detect_variant(sd, args, "RFDETRSmall", "rf-detr-small.pth", forced="medium") == ("medium", "--variant")
    # even on shapes that fit nothing (the strict load then reports the mismatch)
    assert fam.detect_variant(fake_sd(16, 512, 32, 6), {}, None, "c.pth", forced="nano")[0] == "nano"


@pytest.mark.parametrize("variant", ["nano", "small", "medium", "base"])
def test_official_checkpoints_detected_without_variant(variant):
    """rfdetr's own COCO checkpoints (downloaded to ~/.roboflow/models by rfdetr) carry no model_name;
    each must be detected as its own size. Skipped where rfdetr has not downloaded them."""
    p = OFFICIAL / f"rf-detr-{variant}.pth"
    if not p.is_file():
        pytest.skip(f"{p} not downloaded")
    sd, args, name = fam._split_checkpoint(fam._load_checkpoint(p))
    assert fam.detect_variant(sd, args, name, p.name)[0] == variant


@pytest.mark.parametrize("sig", [(16, 512, 32, 3), (8, 384, 32, 3), (16, 384, 32, 6), (14, 768, 37, 6)])
def test_unknown_shapes_refused(sig):
    with pytest.raises(ConvertError, match="fit no RF-DETR variant") as e:
        fam.detect_variant(fake_sd(*sig), {}, None, "rf-detr-small.pth")
    assert str(sig) in str(e.value)
    # metadata naming a known variant does not rescue them: those weights would not load
    with pytest.raises(ConvertError, match="its metadata says small"):
        fam.detect_variant(fake_sd(*sig), {}, "RFDETRSmall", "c.pth")


# --------------------------------------------------------------------------------------------
# a tiny real RF-DETR (Nano, random weights) through the CLI
# --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nano_ckpt(tmp_path_factory):
    from rfdetr.variants import RFDETRNano
    torch.manual_seed(0)
    m = RFDETRNano(pretrain_weights=None, num_classes=3, device="cpu")
    d = tmp_path_factory.mktemp("nano")
    p = d / "checkpoint_best_total.pth"
    torch.save({"model": m.model.model.state_dict(), "model_name": "RFDETRNano", "rfdetr_version": "1.7.1",
                "args": {"class_names": ["mug", "bowl", "spoon"], "group_detr": 13, "resolution": None}}, p)
    return p


def test_nano_checkpoint_converts(tmp_path, nano_ckpt):
    rc, out = run_cli(tmp_path, nano_ckpt, "tiny-nano")
    assert rc == 0
    man = (out / "manifest.yaml").read_text()
    for line in ("task: detection", "architecture: rf-detr", "license: Apache-2.0", "width: 384", "height: 384",
                 "letterbox: false", "type: detr", "box_format: cxcywh", "conf_threshold: 0.5",
                 "max_detections: 300", "mean: [0.485, 0.456, 0.406]", "labels: labels.txt"):
        assert line in man, line
    assert (out / "labels.txt").read_text().splitlines() == ["mug", "bowl", "spoon", "N/A"]
    ins, outs = onnx_io(out / "model.onnx")
    assert ins[0][1] == [1, 3, 384, 384]
    assert [o[1] for o in outs] == [[1, 300, 4], [1, 300, 4]]  # C = 3 classes + 1


def test_nano_query_feats_and_resolution(tmp_path, nano_ckpt):
    rc, out = run_cli(tmp_path, nano_ckpt, "tiny-nano-qf", "--query-feats", "--resolution", "320",
                      "--license", "apache-2.0")
    assert rc == 0
    _, outs = onnx_io(out / "model.onnx")
    assert [o[0] for o in outs] == ["dets", "labels", "query_feats"]
    assert outs[2][1] == [1, 300, 256]
    assert "width: 320" in (out / "manifest.yaml").read_text()


def test_bad_resolution_refused(tmp_path, nano_ckpt, capsys):
    rc, _ = run_cli(tmp_path, nano_ckpt, "x", "--resolution", "330")
    assert rc == 2
    assert "divisible" in capsys.readouterr().err


def test_wrong_variant_refused(tmp_path, nano_ckpt, capsys):
    rc, _ = run_cli(tmp_path, nano_ckpt, "x", "--variant", "small")
    assert rc == 2
    err = capsys.readouterr().err
    assert "do not fit RF-DETR small" in err and "--variant nano" in err


def test_wrong_labels_file_refused(tmp_path, nano_ckpt, capsys):
    lab = tmp_path / "labels.txt"
    lab.write_text("a\nb\n")
    rc, out = run_cli(tmp_path, nano_ckpt, "x", "--labels", str(lab))
    assert rc == 2
    assert "2 class names for a head with 4 logits" in capsys.readouterr().err
    assert not out.exists()


def test_agpl_marker_refused(tmp_path, capsys):
    p = tmp_path / "best.pth"
    torch.save({"model": {"class_embed.bias": torch.zeros(3)},
                "args": {"source": "ultralytics.nn.tasks.DetectionModel"}}, p)
    rc, _ = run_cli(tmp_path, p, "x")
    assert rc == 2
    assert "AGPL" in capsys.readouterr().err


def test_agpl_license_refused(tmp_path, nano_ckpt, capsys):
    rc, _ = run_cli(tmp_path, nano_ckpt, "x", "--license", "AGPL-3.0")
    assert rc == 2
    assert "not allowed" in capsys.readouterr().err


def test_detr_parity_ignores_only_low_score_boundary_queries(tmp_path):
    import onnx
    from onnx import TensorProto, helper
    # identity graph: 2 inputs -> 2 outputs, so "ONNX output" == feeds and we control the difference
    g = helper.make_graph([helper.make_node("Identity", ["b"], ["B"]), helper.make_node("Identity", ["l"], ["L"])],
                          "id", [helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 100, 4]),
                                 helper.make_tensor_value_info("l", TensorProto.FLOAT, [1, 100, 3])],
                          [helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 100, 4]),
                           helper.make_tensor_value_info("L", TensorProto.FLOAT, [1, 100, 3])])
    path = tmp_path / "id.onnx"
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), path)
    rng = np.random.default_rng(0)
    b = rng.random((1, 100, 4), dtype=np.float32)
    lo = np.full((1, 100, 3), -2.0, np.float32)
    lo[0, :10] = 3.0  # 10 confident queries
    ref = [b.copy(), lo.copy()]
    b2 = b.copy()
    b2[0, 95] += 0.3  # query 95 is a different proposal in the two runs (top-K near-tie) ...
    lo2 = lo.copy()
    lo2[0, 95] = -5.0  # ... and scores < SCORE_FLOOR in both
    ref[1][0, 95] = -4.0
    fam.detr_parity(path, {"b": b2, "l": lo2}, ref, 1e-3)  # tolerated: score < 0.05 both sides
    b3 = b.copy()
    b3[0, 3] += 0.3  # a confident query moved: a real bug
    with pytest.raises(ConvertError, match="Not installed"):
        fam.detr_parity(path, {"b": b3, "l": lo}, [b, lo], 1e-3)


def _identity_pair(tmp_path, q=100, c=3):
    """An identity ONNX graph (b, l) -> (B, L): the 'ONNX output' is the feed, so a test sets it."""
    import onnx
    from onnx import TensorProto, helper
    g = helper.make_graph([helper.make_node("Identity", ["b"], ["B"]), helper.make_node("Identity", ["l"], ["L"])],
                          "id", [helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, q, 4]),
                                 helper.make_tensor_value_info("l", TensorProto.FLOAT, [1, q, c])],
                          [helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, q, 4]),
                           helper.make_tensor_value_info("L", TensorProto.FLOAT, [1, q, c])])
    path = tmp_path / "id.onnx"
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), path)
    return path


def _queries(q=100, c=3, confident=10, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.random((1, q, 4), dtype=np.float32)
    lo = rng.uniform(-7, -4, (1, q, c)).astype(np.float32)  # sigmoid < 0.02
    lo[0, :confident, 0] = rng.uniform(0, 4, confident)      # sigmoid 0.5 .. 0.98
    return b, lo


def test_detr_parity_passes_top_k_near_ties(tmp_path):
    """What the official COCO checkpoints do on the noise image (Nano: 100 of 300 queries differ,
    all scoring < 0.02): many LOW-score queries differ, every confident one matches. Not a defect."""
    path = _identity_pair(tmp_path)
    b, lo = _queries()
    rng = np.random.default_rng(1)
    b2, lo2 = b.copy(), lo.copy()
    near = rng.choice(np.arange(10, 100), 40, replace=False)  # 40% of the queries, all low-score
    b2[0, near] = rng.random((40, 4))
    lo2[0, near] = rng.uniform(-7, -4, (40, 3))
    err = fam.detr_parity(path, {"b": b2, "l": lo2}, [b, lo], 1e-3)
    assert err <= 1e-3
    # Two confident queries that trade places in the top-K order are still the same two queries.
    b3, lo3 = b.copy(), lo.copy()
    b3[0, [2, 7]], lo3[0, [2, 7]] = b[0, [7, 2]], lo[0, [7, 2]]
    fam.detr_parity(path, {"b": b3, "l": lo3}, [b, lo], 1e-3)


@pytest.mark.parametrize("bug", ["confident box moved", "confident score changed", "query appears",
                                 "swapped box coords", "shifted logits", "low-score majority differs",
                                 "swapped box coords, no confident query", "shifted logits, no confident query"])
def test_detr_parity_still_fails_real_export_bugs(tmp_path, bug):
    """A real export bug fails however few confident queries the input has (the noise image of a
    COCO checkpoint has almost none): rule 2 still sees every query move."""
    path = _identity_pair(tmp_path)
    b, lo = _queries(confident=0 if "no confident" in bug else 10)
    bug = bug.split(",")[0]
    g_b, g_lo = b.copy(), lo.copy()
    if bug == "confident box moved":
        g_b[0, 3] += 0.01
    elif bug == "confident score changed":
        g_lo[0, 3, 0] += 0.2
    elif bug == "query appears":  # a low-score query becomes confident in the ONNX run only
        g_lo[0, 50, 1] = 2.0
    elif bug == "swapped box coords":  # cx,cy,w,h -> cy,cx,h,w
        g_b = g_b[..., [1, 0, 3, 2]].copy()
    elif bug == "shifted logits":
        g_lo += 0.5
    elif bug == "low-score majority differs":  # every confident query fine, 60% of the rest moved
        rows = np.arange(10, 70)
        g_b[0, rows] += 0.05
    with pytest.raises(ConvertError, match="Not installed"):
        fam.detr_parity(path, {"b": g_b, "l": g_lo}, [b, lo], 1e-3)


def test_detr_parity_non_finite_must_agree(tmp_path):
    path = _identity_pair(tmp_path)
    b, lo = _queries()
    lo[0, 50, 2] = -np.inf  # the same -inf in both runs: compared as equal, then left out
    fam.detr_parity(path, {"b": b, "l": lo}, [b, lo], 1e-3)
    g_lo = lo.copy()
    g_lo[0, 60, 1] = np.nan  # a NaN only the ONNX graph produces
    with pytest.raises(ConvertError, match="NaN/inf"):
        fam.detr_parity(path, {"b": b, "l": g_lo}, [b, lo], 1e-3)


def test_detr_parity_fails_a_wrong_normalisation_in_the_graph(tmp_path):
    """A small linear 'detector' exported to ONNX, fed an input normalised differently from the
    reference's (ImageNet mean/std vs /255 only): the outputs move everywhere, confident or not."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    rng = np.random.default_rng(0)
    q, f, c = 100, 3, 3
    wb, wl = rng.normal(0, 0.5, (f, 4)).astype(np.float32), rng.normal(0, 2, (f, c)).astype(np.float32)
    wl[:, 0] += 3.0
    g = helper.make_graph(
        [helper.make_node("MatMul", ["x", "wb"], ["zb"]), helper.make_node("Sigmoid", ["zb"], ["B"]),
         helper.make_node("MatMul", ["x", "wl"], ["L"])], "lin",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, q, f])],
        [helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, q, 4]),
         helper.make_tensor_value_info("L", TensorProto.FLOAT, [1, q, c])],
        [numpy_helper.from_array(wb, "wb"), numpy_helper.from_array(wl, "wl")])
    path = tmp_path / "lin.onnx"
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), path)
    px = rng.random((1, q, f)).astype(np.float32)  # "pixels" in 0..1
    mean, std = np.array(IMAGENET_MEAN, np.float32), np.array(IMAGENET_STD, np.float32)
    right = (px - mean) / std
    ref = [1 / (1 + np.exp(-(right @ wb))), right @ wl]
    fam.detr_parity(path, {"x": right}, ref, 1e-3)  # same normalisation: passes
    with pytest.raises(ConvertError, match="Not installed"):
        fam.detr_parity(path, {"x": px}, ref, 1e-3)


# --------------------------------------------------------------------------------------------
# two-stage proposal near-ties (parity_run)
# --------------------------------------------------------------------------------------------

def test_check_proposal_tie():
    rng = np.random.default_rng(0)
    s = rng.normal(-4, 1, 50)
    s[[7, 9]] = -1.0  # proposals 7 and 9 tie (both in the top 5)
    i_r = np.argsort(-s, kind="stable")[:5]
    s_g = s + rng.uniform(-1e-6, 1e-6, 50)  # ORT's float noise
    assert fam.check_proposal_tie(s, i_r, s_g, i_r, 1e-3) == []
    i_g = i_r.copy()
    a, b = int(np.nonzero(i_r == 7)[0][0]), int(np.nonzero(i_r == 9)[0][0])
    i_g[[a, b]] = i_r[[b, a]]  # the tied pair in the other order
    assert fam.check_proposal_tie(s[None], i_r[None], s_g[None], i_g[None], 1e-3) == sorted([a, b])
    # a tie AT the cut: ORT keeps the 6th proposal, which ties with PyTorch's 5th
    s2 = s.copy()
    sixth = int(np.argsort(-s2, kind="stable")[5])
    s2[sixth] = s2[i_r[4]]
    i_g2 = i_r.copy()
    i_g2[4] = sixth
    assert fam.check_proposal_tie(s2, i_r, s2, i_g2, 1e-3) == [4]


@pytest.mark.parametrize("bug", ["encoder scores differ", "ascending top-K", "wrong proposal kept", "out of range"])
def test_check_proposal_tie_refuses_real_bugs(bug):
    rng = np.random.default_rng(0)
    s = rng.normal(-4, 1, 50)
    i_r = np.argsort(-s)[:5]
    s_g, i_g = s.copy(), i_r.copy()
    if bug == "encoder scores differ":
        s_g = s + 0.05
        i_g[[0, 1]] = i_r[[1, 0]]
    elif bug == "ascending top-K":
        i_g = np.argsort(s)[:5]
    elif bug == "wrong proposal kept":
        i_g[2] = np.argsort(-s)[30]
    else:
        i_g[0] = 50
    with pytest.raises(ConvertError, match="proposal selection.*Not installed"):
        fam.check_proposal_tie(s, i_r, s_g, i_g, 1e-3)


def test_topk_tap_records_forces_and_restores():
    orig = torch.topk
    x = torch.tensor([[0.1, 0.9, 0.5, 0.7]])
    with fam._topk_tap() as calls:
        idx = torch.topk(x, 2, dim=1)[1]
    assert idx.tolist() == [[1, 3]] and len(calls) == 1 and calls[0][1].tolist() == [[1, 3]]
    with fam._topk_tap(force=torch.tensor([[3, 1]])) as calls:
        v, idx = torch.topk(x, 2, dim=1)
        other = torch.topk(x, 1, dim=1)[1]  # a call of another shape is left alone
    assert idx.tolist() == [[3, 1]] and v[0].tolist() == pytest.approx([0.7, 0.9]) and other.tolist() == [[1]]
    assert torch.topk is orig


class _TwoStage(torch.nn.Module):
    """A toy two-stage DETR: score N proposals, keep the top K, add slot r's own learned query to the
    r-th kept proposal (as RF-DETR's query_feat / refpoint_embed), decode boxes + logits."""

    def __init__(self, n=40, k=8, c=3, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.k = k
        self.score = torch.nn.Linear(1, c)  # proposals' scores depend on feature 0 only: ties are easy
        self.slot = torch.nn.Parameter(torch.randn(k, 4, generator=g))
        self.box, self.cls = torch.nn.Linear(4, 4), torch.nn.Linear(4, c)
        with torch.no_grad():
            self.score.weight.copy_(torch.tensor([[1.0], [0.5], [-1.0]]))
            self.score.bias.zero_()
            self.cls.bias.fill_(3.0)  # every query confident
        self.register_buffer("bump", torch.zeros(1, n))  # added to the selection scores only

    def forward(self, x):
        s = self.score(x[..., :1]).max(-1)[0] + self.bump
        idx = torch.topk(s, self.k, dim=1)[1]
        q = torch.gather(x, 1, idx.unsqueeze(-1).repeat(1, 1, 4)) + self.slot
        return torch.sigmoid(self.box(q)), self.cls(q)


def _two_stage_case(tmp_path, onnx_change=None, bump=1e-6):
    """(reference module, ONNX path, input): proposals 2 and 5 tie exactly, and the exported graph
    breaks the tie the other way (+`bump` on the one PyTorch ranks second)."""
    torch.manual_seed(0)
    ref = _TwoStage().eval()
    x = torch.rand(1, 40, 4)
    x[0, :, 0] *= 0.5
    x[0, [2, 5], 0] = 0.9  # tied, and in the top K
    with torch.no_grad():
        idx = torch.topk(ref.score(x[..., :1]).max(-1)[0], ref.k, dim=1)[1][0].tolist()
    second = max((2, 5), key=idx.index)
    dut = copy.deepcopy(ref)
    dut.bump[0, second] = bump
    if onnx_change:
        onnx_change(dut)
    path = tmp_path / "two_stage.onnx"
    torch.onnx.export(dut, (x,), str(path), input_names=["x"], output_names=["dets", "labels"], opset_version=17,
                      dynamo=False)
    return ref, path, x.numpy()


def test_parity_run_reruns_reference_on_onnx_proposal_order(tmp_path):
    ref, path, x = _two_stage_case(tmp_path)
    with pytest.raises(ConvertError, match="Not installed"):  # what detr_parity alone says
        with torch.no_grad():
            r = [t.numpy() for t in ref(torch.from_numpy(x))]
        fam.detr_parity(path, {"x": x}, r, 1e-3)
    from visionserve.convert import common
    common.PARITY_RECORDS.clear()
    assert fam.parity_run(ref, path, "x", x, 1e-3, " toy", workdir=tmp_path) <= 1e-3
    assert common.PARITY_RECORDS[-1]["proposal_rows"] == 2
    common.PARITY_RECORDS.clear()


@pytest.mark.parametrize("case", ["slot embedding shifted", "box head changed", "selection scores moved",
                                  "no workdir"])
def test_parity_run_still_fails_real_export_bugs(tmp_path, case):
    """The re-run on ONNX's proposal order excuses the swap, nothing else: a real bug alongside the
    tie, or a 'tie' that is not one, still fails."""
    change = {"slot embedding shifted": lambda m: m.slot.data.add_(0.05),
              "box head changed": lambda m: m.box.bias.data.add_(0.1)}.get(case)
    ref, path, x = _two_stage_case(tmp_path, change, bump=5.0 if case == "selection scores moved" else 1e-6)
    with pytest.raises(ConvertError, match="Not installed"):
        fam.parity_run(ref, path, "x", x, 1e-3, " toy", workdir=None if case == "no workdir" else tmp_path)


def test_proposal_probe_needs_exactly_one_topk(tmp_path):
    ref, path, x = _two_stage_case(tmp_path)
    probe = fam.proposal_probe(path, tmp_path / "probe.onnx")
    outs = ort.InferenceSession(str(probe), providers=["CPUExecutionProvider"]).run(None, {"x": x})
    assert len(outs) == 4 and outs[-2].shape == (1, 40) and outs[-1].shape == (1, 8)
    plain = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"x": x})
    assert all(np.array_equal(a, b) for a, b in zip(plain, outs[:2]))
    assert fam.proposal_probe(_identity_pair(tmp_path), tmp_path / "p2.onnx") is None  # no TopK


def test_parity_inputs_add_the_first_photo(tmp_path):
    from PIL import Image
    (tmp_path / "b.jpg").write_bytes(b"")  # sorted after a.png; never opened
    Image.new("RGB", (40, 30), (200, 10, 10)).save(tmp_path / "a.png")
    got = fam.parity_inputs(32, str(tmp_path))
    assert [(lab, real) for lab, _, real in got] == [("seed 0", False), ("seed 1", False), ("a.png", True)]
    assert all(x.shape == (1, 3, 32, 32) and x.dtype == np.float32 for _, x, _ in got)
    assert [lab for lab, _, _ in fam.parity_inputs(32)] == ["seed 0", "seed 1"]


def test_tier_a_row_names_the_photo_and_counts_the_near_ties():
    from visionserve.convert.verify import tier_a_results
    runs = [{"what": "rfdetr[seed 0]", "max_rel_diff": 9.9e-4, "tol": 1e-3, "low_score_differ": 100,
             "real_input": False},
            {"what": "rfdetr[a.jpg]", "max_rel_diff": 5e-5, "tol": 1e-3, "low_score_differ": 0, "real_input": True}]
    (row,) = tier_a_results(runs, ["m"])
    assert row.title == "ONNX vs framework parity (synthetic + photo)"
    assert "100 low-score queries differ" in row.summary
    (row,) = tier_a_results(runs[:1], ["m"])
    assert row.title == "ONNX vs framework parity (synthetic input)"


# --------------------------------------------------------------------------------------------
# the real tabletop fine-tune (NAS)
# --------------------------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_CKPT.exists(), reason="NAS checkpoint not mounted")
def test_real_tabletop_checkpoint(tmp_path):
    rc, out = run_cli(tmp_path, REAL_CKPT, "tt-rfdetr")
    assert rc == 0
    man = (out / "manifest.yaml").read_text()
    assert "width: 512" in man and "letterbox: false" in man and "architecture: rf-detr" in man
    labels = (out / "labels.txt").read_text().splitlines()
    assert len(labels) == 23 and labels[0] == "remote" and labels[-1] == "N/A"

    # Pin the label mapping on real data: predictions that overlap a ground-truth box (IoU > 0.5) must
    # carry that box's name. (0-based + trailing N/A scored 137/141 when this was worked out; a 1-based
    # shift scores ~1/141.)
    anns = sorted((ETRI / "annotations").glob("*.json"))[:12] if ETRI.exists() else []
    if not anns:
        return
    from PIL import Image
    sess = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])

    def iou(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)

    right = wrong = 0
    for f in anns:
        d = json.loads(f.read_text())
        im = Image.open(ETRI / "images" / d["image"]).convert("RGB")
        W, H = im.size
        x = to_nchw(np.asarray(im.resize((512, 512), Image.BILINEAR)), IMAGENET_MEAN, IMAGENET_STD)
        boxes, logits = sess.run(None, {"input": x})
        p = 1 / (1 + np.exp(-logits[0]))
        for q in np.nonzero(p.max(-1) > 0.5)[0]:
            cx, cy, w, h = boxes[0, q]
            bx = [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
            gt = max(d["annotations"], key=lambda a: iou(a["bbox"], bx), default=None)
            if gt is not None and iou(gt["bbox"], bx) > 0.5:
                if labels[int(p[q].argmax())] == gt["category"]:
                    right += 1
                else:
                    wrong += 1
    assert right >= 20 and right / (right + wrong) > 0.9, (right, wrong)
