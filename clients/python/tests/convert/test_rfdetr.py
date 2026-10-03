"""Tests for the `rfdetr` converter family (families/rfdetr.py).

Offline: a randomly initialised RF-DETR Nano (built by the installed `rfdetr`, no download: patch 16
never loads DINOv2 hub weights) is saved as a training checkpoint and run through the real CLI path
with --dry-run. The real 22-class tabletop fine-tune on the NAS is converted when present (skipped
otherwise), and its ONNX is checked against ground-truth boxes to pin the label mapping.
"""
from __future__ import annotations

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
    assert fam.detect_variant(sd, {}, "RFDETRMedium", "c.pth")[0] == "medium"      # model_name beats shapes
    assert fam.detect_variant(sd, {}, "RFDETRMedium", "c.pth", forced="nano")[0] == "nano"
    assert fam.detect_variant(sd, {"pretrain_weights": "rf-detr-base.pth"}, None, "c.pth")[0] == "base"
    assert fam.detect_variant({"class_embed.bias": 0}, {}, None, "rf-detr-nano.pth")[0] == "nano"
    # custom resolution changes only the PE grid
    assert fam.detect_variant(fake_sd(16, 384, 40, 2), {}, None, "c.pth")[0] == "nano"
    with pytest.raises(ConvertError, match="Platform Model License"):
        fam.detect_variant(sd, {}, "RFDETRXLarge", "c.pth")
    with pytest.raises(ConvertError, match="segmentation"):
        fam.detect_variant(sd, {}, "RFDETRSegSmall", "c.pth")
    with pytest.raises(ConvertError, match="--variant"):
        fam.detect_variant({"class_embed.bias": 0}, {}, None, "checkpoint.pth")


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
    lo2[0, 95] = -5.0  # ... and scores < BOUNDARY_SCORE in both
    ref[1][0, 95] = -4.0
    fam.detr_parity(path, {"b": b2, "l": lo2}, ref, 1e-3)  # tolerated: score < 0.05 both sides
    b3 = b.copy()
    b3[0, 3] += 0.3  # a confident query moved: a real bug
    with pytest.raises(ConvertError, match="Not installed"):
        fam.detr_parity(path, {"b": b3, "l": lo}, [b, lo], 1e-3)


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
