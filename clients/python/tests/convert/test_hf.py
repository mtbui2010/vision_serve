"""Tests for the `hf` converter family (clients/python/visionserve/convert/families/hf.py).

Offline: every checkpoint is a TINY randomly-initialised transformers config saved to tmp_path, run
through the real CLI path (`visionserve-convert hf ... --dry-run`): export -> parity ->
I/O contract -> manifest. Real-checkpoint conversions (siglip, grounding-dino) are slow and need the
HF cache or network; they run only with VS_CONVERT_SLOW=1.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # clients/python

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
ort = pytest.importorskip("onnxruntime")

from visionserve.convert import cli  # noqa: E402
from visionserve.convert.common import onnx_io  # noqa: E402
from visionserve.convert.families import hf  # noqa: E402

REPO = Path(__file__).resolve().parents[4]


@pytest.fixture(autouse=True)
def _only_hf(monkeypatch):
    # The CLI imports every family to build its parser; the other families (and their frameworks)
    # are not this file's concern.
    monkeypatch.setattr(cli, "FORMATS", {"hf": cli.FORMATS["hf"]})


def run_cli(tmp_path, src, name, *extra):
    models = tmp_path / "models"
    rc = cli.main(["hf", str(src), "--name", name, "--dry-run", "--models", str(models), *extra])
    return rc, models / ".convert-dry-run"


def card(d: Path, license_line: str | None):
    body = "# model\n"
    if license_line is not None:
        body = f"---\n{license_line}\ntags:\n- vision\n---\n" + body
    (d / "README.md").write_text(body)


def manifest(out: Path, name: str) -> str:
    return (out / name / "manifest.yaml").read_text()


# --------------------------------------------------------------------------------------------
# tiny checkpoints
# --------------------------------------------------------------------------------------------

@pytest.fixture
def tiny_vit(tmp_path):
    from transformers import AutoModelForImageClassification, ViTConfig, ViTImageProcessor
    torch.manual_seed(0)
    d = tmp_path / "tiny-vit"
    cfg = ViTConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=37,
                    image_size=32, patch_size=8, num_labels=3,
                    id2label={0: "cat", 1: "dog", 2: "remote control"},
                    label2id={"cat": 0, "dog": 1, "remote control": 2})
    AutoModelForImageClassification.from_config(cfg).save_pretrained(d)
    ViTImageProcessor(size={"height": 32, "width": 32}, image_mean=[0.5, 0.5, 0.5],
                      image_std=[0.5, 0.5, 0.5]).save_pretrained(d)
    card(d, "license: apache-2.0")
    return d


@pytest.fixture
def tiny_dpt(tmp_path):
    from transformers import AutoModelForDepthEstimation, DPTConfig, DPTImageProcessor
    torch.manual_seed(0)
    d = tmp_path / "tiny-dpt"
    cfg = DPTConfig(hidden_size=32, num_hidden_layers=4, num_attention_heads=2, intermediate_size=37,
                    image_size=32, patch_size=8, backbone_out_indices=[0, 1, 2, 3],
                    neck_hidden_sizes=[8, 8, 16, 16], fusion_hidden_size=16, head_in_index=-1)
    AutoModelForDepthEstimation.from_config(cfg).save_pretrained(d)
    DPTImageProcessor(size={"height": 32, "width": 32}, keep_aspect_ratio=True,
                      ensure_multiple_of=8).save_pretrained(d)
    card(d, "license:\n- mit")
    return d


@pytest.fixture
def tiny_depth_anything(tmp_path):
    """Depth Anything layout (DINOv2 backbone, DPTImageProcessor keep_aspect_ratio + multiple 14):
    unlike DPT's own ViT it accepts any HxW that is a multiple of the patch size."""
    from transformers import AutoModelForDepthEstimation, DepthAnythingConfig, Dinov2Config, DPTImageProcessor
    torch.manual_seed(0)
    d = tmp_path / "tiny-da"
    bb = Dinov2Config(hidden_size=32, num_hidden_layers=4, num_attention_heads=2, intermediate_size=37,
                      image_size=28, patch_size=14, out_features=["stage1", "stage2", "stage3", "stage4"],
                      reshape_hidden_states=False)
    cfg = DepthAnythingConfig(backbone_config=bb, reassemble_hidden_size=32, patch_size=14,
                              neck_hidden_sizes=[8, 8, 16, 16], fusion_hidden_size=16, head_hidden_size=8)
    model = AutoModelForDepthEstimation.from_config(cfg)
    # HF's 0.02-std init makes the head output a near-constant map; torch's default init gives a
    # spatially varying one, so parity at other sizes compares real content.
    for mod in model.modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
            mod.reset_parameters()
    model.save_pretrained(d)
    DPTImageProcessor(size={"height": 28, "width": 28}, keep_aspect_ratio=True, ensure_multiple_of=14,
                      resample=3, image_mean=[0.485, 0.456, 0.406],
                      image_std=[0.229, 0.224, 0.225]).save_pretrained(d)
    card(d, "license: apache-2.0")
    return d


def make_rtdetr(d: Path, n_classes: int = 5) -> Path:
    from transformers import AutoModelForObjectDetection, RTDetrConfig, RTDetrImageProcessor, RTDetrResNetConfig
    torch.manual_seed(0)
    bb = RTDetrResNetConfig(embedding_size=8, hidden_sizes=[8, 16, 16, 16], depths=[1, 1, 1, 1],
                            layer_type="basic", out_features=["stage2", "stage3", "stage4"])
    cfg = RTDetrConfig(backbone_config=bb, encoder_in_channels=[16, 16, 16], d_model=32,
                       encoder_hidden_dim=32, encoder_ffn_dim=64, decoder_ffn_dim=64, encoder_layers=1,
                       decoder_layers=1, num_queries=10, num_feature_levels=3, decoder_n_points=2,
                       decoder_in_channels=[32, 32, 32], anchor_image_size=None, num_denoising=0,
                       num_labels=n_classes, id2label={i: f"c{i}" for i in range(n_classes)},
                       label2id={f"c{i}": i for i in range(n_classes)}, use_pretrained_backbone=False)
    m = AutoModelForObjectDetection.from_config(cfg)
    # HF initialises the encoder score head to a constant prior, so every anchor scores exactly the
    # same and top-k query selection is an arbitrary tie-break (torch and ORT pick different
    # anchors). Real checkpoints never tie; jitter the weights so the tiny one does not either.
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.2 * torch.randn_like(p))
    m.save_pretrained(d)
    RTDetrImageProcessor(size={"height": 64, "width": 64}).save_pretrained(d)
    card(d, "license: apache-2.0")
    return d


@pytest.fixture
def tiny_rtdetr(tmp_path):
    return make_rtdetr(tmp_path / "tiny-rtdetr")


# --------------------------------------------------------------------------------------------
# positive: export + parity + contract + manifest
# --------------------------------------------------------------------------------------------

def test_classification_vit(tmp_path, tiny_vit, capsys):
    rc, out = run_cli(tmp_path, tiny_vit, "tiny-vit")
    err = capsys.readouterr().err
    assert rc == 0, err
    m = manifest(out, "tiny-vit")
    assert "architecture: efficientnet" in m and "task: classification" in m
    assert "license: Apache-2.0" in m
    assert "width: 32" in m and "height: 32" in m and "letterbox: false" in m
    assert "mean: [0.5, 0.5, 0.5]" in m and "std: [0.5, 0.5, 0.5]" in m
    assert "labels: labels.txt" in m
    assert (out / "tiny-vit" / "labels.txt").read_text().splitlines() == ["cat", "dog", "remote control"]
    assert "parity (classification)" in err
    ins, outs = onnx_io(out / "tiny-vit" / "model.onnx")
    assert ins[0][1] == [1, 3, 32, 32] and outs[0][1] == [1, 3]


def test_input_override_rejected_by_fixed_position_embeddings(tmp_path, tiny_vit, capsys):
    # ViT's position embeddings fix the resolution: one clear line, not two exporter traces.
    rc, _ = run_cli(tmp_path, tiny_vit, "tiny-vit48", "--input", "48x48")
    err = capsys.readouterr().err
    assert rc == 2 and "rejects the input" in err and "48x48" in err, err


def test_depth_input_override(tmp_path, tiny_dpt, capsys):
    rc, out = run_cli(tmp_path, tiny_dpt, "tiny-dpt40", "--input", "40x40")
    assert rc == 0, capsys.readouterr().err
    m = manifest(out, "tiny-dpt40")
    assert "width: 40" in m and "height: 40" in m
    _, outs = onnx_io(out / "tiny-dpt40" / "model.onnx")
    assert len(outs[0][1]) == 3  # [1,H,W]; the tracer may leave H,W symbolic


def test_depth_dpt(tmp_path, tiny_dpt, capsys):
    rc, out = run_cli(tmp_path, tiny_dpt, "tiny-dpt")
    err = capsys.readouterr().err
    assert rc == 0, err
    m = manifest(out, "tiny-dpt")
    assert "architecture: midas" in m and "task: depth" in m and "license: MIT" in m
    assert "type: depth" in m
    ins, outs = onnx_io(out / "tiny-dpt" / "model.onnx")
    assert len(outs) == 1 and len(outs[0][1]) == 3
    assert "parity (depth 32x32)" in err
    # DPT's own ViT reassembles a SQUARE patch grid, so the processor's keep-aspect resize cannot be
    # served: squash, with the divergence still documented.
    assert "keep_aspect" not in m and "HF keeps the aspect ratio" in m
    assert "rejects non-square inputs" in err


def test_depth_anything_keeps_aspect(tmp_path, tiny_depth_anything, capsys):
    rc, out = run_cli(tmp_path, tiny_depth_anything, "tiny-da")
    err = capsys.readouterr().err
    assert rc == 0, err
    m = manifest(out, "tiny-da")
    assert "  keep_aspect: true" in m and "  multiple_of: 14" in m
    assert "width: 28" in m and "height: 28" in m
    # Served exactly as the processor does: no longer a documented divergence.
    assert "HF keeps the aspect ratio" not in m and "HF resamples" not in m
    # Parity at the traced size AND at a landscape + a portrait keep-aspect size.
    for size in ("28x28", "56x28", "28x42"):
        assert f"parity (depth {size})" in err, err
    ins, outs = onnx_io(out / "tiny-da" / "model.onnx")
    assert all(isinstance(d, str) for d in ins[0][1][2:]), ins  # dynamic H/W
    sess = ort.InferenceSession(str(out / "tiny-da" / "model.onnx"), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"pixel_values": np.zeros((1, 3, 70, 42), np.float32)})[0]
    assert got.shape == (1, 70, 42)


def test_dpt_keep_aspect_size_matches_transformers():
    """reference.dpt_keep_aspect_size (= the Go rule) against the real HF function on many sizes."""
    from transformers.models.dpt.image_processing_dpt import get_resize_output_image_size
    from visionserve.convert.reference import dpt_keep_aspect_size
    rng = np.random.default_rng(0)
    sizes = [(int(w), int(h)) for w, h in rng.integers(16, 4000, size=(300, 2))]
    sizes += [(1036, 1050), (1036, 1078), (200, 201), (200, 203), (518, 518), (1, 1)]
    for tw, th, m in ((518, 518, 14), (384, 384, 32), (518, 392, 14), (100, 100, 1)):
        for w, h in sizes:
            s = get_resize_output_image_size(torch.zeros(3, h, w), (th, tw), True, m)
            if s.width == 0 or s.height == 0:   # HF degenerates (then fails); ours clamps to m
                continue
            assert dpt_keep_aspect_size(w, h, tw, th, m) == (s.width, s.height), (w, h, tw, th, m)


def test_depth_input_must_respect_multiple(tmp_path, tiny_dpt, capsys):
    rc, _ = run_cli(tmp_path, tiny_dpt, "bad", "--input", "36x36")
    assert rc == 2 and "multiple of 8" in capsys.readouterr().err


def test_rt_detr(tmp_path, tiny_rtdetr, capsys):
    rc, out = run_cli(tmp_path, tiny_rtdetr, "tiny-rtdetr")
    err = capsys.readouterr().err
    assert rc == 0, err
    m = manifest(out, "tiny-rtdetr")
    assert "architecture: rt-detr" in m and "task: detection" in m
    assert "box_format: cxcywh" in m and "letterbox: false" in m
    # RTDetrImageProcessor: do_normalize=False, rescale 1/255 -> identity normalisation
    assert "mean: [0, 0, 0]" in m and "std: [1, 1, 1]" in m
    assert (out / "tiny-rtdetr" / "labels.txt").read_text().split() == ["c0", "c1", "c2", "c3", "c4"]
    _, outs = onnx_io(out / "tiny-rtdetr" / "model.onnx")
    assert sorted(o[1][-1] for o in outs) == [4, 5]


def test_rt_detr_four_classes_refused(tmp_path, capsys):
    # logits [1,Q,4] would be indistinguishable from pred_boxes [1,Q,4] in the Go decoder
    rc, _ = run_cli(tmp_path, make_rtdetr(tmp_path / "rt4", 4), "rt4")
    assert rc == 2 and "exactly 4 classes" in capsys.readouterr().err


# --------------------------------------------------------------------------------------------
# negative: license gate and unsupported model types
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("lic", ["agpl-3.0", "gpl-3.0", "cc-by-nc-4.0", "other"])
def test_copyleft_card_refused_even_with_permissive_flag(tmp_path, tiny_vit, capsys, lic):
    card(tiny_vit, f"license: {lic}")
    rc, _ = run_cli(tmp_path, tiny_vit, "x", "--license", "Apache-2.0")
    err = capsys.readouterr().err
    assert rc == 2 and "not allowed" in err, err


def test_card_flag_conflict_refused(tmp_path, tiny_vit, capsys):
    rc, _ = run_cli(tmp_path, tiny_vit, "x", "--license", "MIT")
    err = capsys.readouterr().err
    assert rc == 2 and "contradicts" in err


def test_no_card_requires_license(tmp_path, tiny_vit, capsys):
    (tiny_vit / "README.md").unlink()
    rc, _ = run_cli(tmp_path, tiny_vit, "x")
    assert rc == 2 and "--license is required" in capsys.readouterr().err
    rc, out = run_cli(tmp_path, tiny_vit, "x", "--license", "bsd-3-clause")
    assert rc == 0 and "license: BSD-3-Clause" in manifest(out, "x")


@pytest.mark.parametrize("mt,arch,needle", [
    ("detr", "DetrForObjectDetection", "no-object"),
    ("owlv2", "Owlv2ForObjectDetection", "OWLv2"),
    ("llama", "LlamaForCausalLM", "not supported"),
])
def test_unsupported_model_type_refused(tmp_path, capsys, mt, arch, needle):
    d = tmp_path / mt
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": mt, "architectures": [arch]}))
    card(d, "license: apache-2.0")
    rc, _ = run_cli(tmp_path, d, "x")
    err = capsys.readouterr().err
    assert rc == 2 and repr(mt) in err and needle in err and "VisionServe architecture decodes" in err, err


def test_not_a_checkpoint(tmp_path, capsys):
    d = tmp_path / "empty"
    d.mkdir()
    card(d, "license: mit")
    rc, _ = run_cli(tmp_path, d, "x")
    assert rc == 2 and "config.json" in capsys.readouterr().err


# --------------------------------------------------------------------------------------------
# units
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("front,want", [
    ("license: apache-2.0", "apache-2.0"),
    ("license: 'mit'", "mit"),
    ("license:\n- bsd-3-clause", "bsd-3-clause"),
    ("license: [agpl-3.0]", "agpl-3.0"),
    ("tags: [x]", None),
])
def test_card_license_parser(tmp_path, front, want):
    card(tmp_path, front)
    assert hf.card_license(tmp_path / "README.md") == want


def test_card_license_ambiguous_list(tmp_path):
    card(tmp_path, "license:\n- mit\n- gpl-3.0")
    with pytest.raises(hf.ConvertError):
        hf.card_license(tmp_path / "README.md")


def test_export_only_removes_its_own_sidecars(tmp_path):
    # Regression: the single-file re-save once deleted every file sharing the ONNX file's stem.
    for name in ("model_notes.py", "model.log", "other.onnx"):
        (tmp_path / name).write_text("keep me")
    hf._export(torch.nn.Linear(3, 2).eval(), (torch.zeros(1, 3),), tmp_path / "model.onnx", ["x"], ["y"],
               None, 17)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.log", "model.onnx", "model_notes.py", "other.onnx"]


def test_gdino_parity_tolerates_tail_swaps_not_decodable_drift(tmp_path):
    """gdino_parity: -inf padding and swapped low-score tail queries are fine; a decodable query
    that moved is not."""
    import onnx
    from onnx import TensorProto, helper
    q, L, T = 6, 4, 8
    rl = np.full((1, q, T), -np.inf, np.float32)
    rl[0, :, :L] = -6.0                      # tail: sigmoid ~ 0.0025
    rl[0, 0, 1] = 2.0                        # one decodable query (sigmoid 0.88)
    rb = np.random.default_rng(0).uniform(0.1, 0.9, (1, q, 4)).astype(np.float32)

    def const_graph(logits, boxes, path):
        nodes = [helper.make_node("Constant", [], ["logits"], value=helper.make_tensor(
                     "l", TensorProto.FLOAT, logits.shape, logits.flatten().tolist())),
                 helper.make_node("Constant", [], ["pred_boxes"], value=helper.make_tensor(
                     "b", TensorProto.FLOAT, boxes.shape, boxes.flatten().tolist())),
                 helper.make_node("Identity", ["input_ids"], ["ids_out"])]
        g = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, L])],
                              [helper.make_tensor_value_info("logits", TensorProto.FLOAT, logits.shape),
                               helper.make_tensor_value_info("pred_boxes", TensorProto.FLOAT, boxes.shape),
                               helper.make_tensor_value_info("ids_out", TensorProto.INT64, [1, L])])
        onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), str(path))

    feeds = {"input_ids": np.zeros((1, L), np.int64)}
    swapped = rb.copy()
    swapped[0, [4, 5]] = swapped[0, [5, 4]] + 0.3   # tail queries replaced by different anchors
    const_graph(rl, swapped, tmp_path / "ok.onnx")
    assert hf.gdino_parity(tmp_path / "ok.onnx", [("p", feeds, (rl, rb))], 1e-3) < 1e-6
    moved = rb.copy()
    moved[0, 0] += 0.01                            # the decodable query's box moved
    const_graph(rl, moved, tmp_path / "bad.onnx")
    with pytest.raises(hf.ConvertError, match="decodable"):
        hf.gdino_parity(tmp_path / "bad.onnx", [("p", feeds, (rl, rb))], 1e-3)


def _gdino_ids(rng, n_phrases):
    ids = [101]
    for _ in range(n_phrases):
        ids += list(rng.integers(2000, 3000, rng.integers(1, 4))) + [1012]
    return ids + [102]


def test_gdino_mask_matches_transformers_and_repo_script():
    """The vendored exportable mask equals transformers' own implementation for any phrase count,
    and the repo's corrected export script (models/grounding-dino/export_gdino_fixedmask.py)."""
    import importlib.util
    import transformers.models.grounding_dino.modeling_grounding_dino as gd
    script = REPO / "models" / "grounding-dino" / "export_gdino_fixedmask.py"
    ref_script = None
    if script.exists():
        spec = importlib.util.spec_from_file_location("gdfix", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ref_script = mod.exportable_masks
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 5, 12):
        ids = torch.tensor([_gdino_ids(rng, n)])
        am, pos = hf.gdino_exportable_masks(ids)
        am_hf, pos_hf = gd.generate_masks_with_special_tokens_and_transfer_map(ids)
        assert torch.equal(am, am_hf.bool()) and torch.equal(pos, pos_hf), n
        if ref_script is not None:
            am2, pos2 = ref_script(ids)
            assert torch.equal(am, am2) and torch.equal(pos, pos2)


# --------------------------------------------------------------------------------------------
# slow: real checkpoints (HF cache or network)
# --------------------------------------------------------------------------------------------

slow = pytest.mark.skipif(os.environ.get("VS_CONVERT_SLOW") != "1", reason="set VS_CONVERT_SLOW=1")


@slow
def test_siglip_real(tmp_path, capsys):
    rc, out = run_cli(tmp_path, "google/siglip-base-patch16-224", "sg")
    err = capsys.readouterr().err
    assert rc == 0, err
    mi, mt = manifest(out, "sg-image"), manifest(out, "sg-text")
    assert "architecture: siglip-image" in mi and "mean: [0.5, 0.5, 0.5]" in mi
    assert "architecture: siglip-text" in mt and "width: 64" in mt
    assert (out / "sg-text" / "tokenizer.json").exists()
    _, o = onnx_io(out / "sg-image" / "model.onnx")
    assert o[0][1][1] == 768
    i, o = onnx_io(out / "sg-text" / "model.onnx")
    assert i[0][1][1] == 64 and o[0][1][1] == 768


@slow
def test_grounding_dino_real(tmp_path, capsys):
    rc, out = run_cli(tmp_path, "IDEA-Research/grounding-dino-tiny", "gd")
    err = capsys.readouterr().err
    assert rc == 0, err
    m = manifest(out, "gd")
    assert "architecture: grounding-dino" in m and "conf_threshold: 0.3" in m and "text_threshold: 0.25" in m
    assert "files:\n  model: model.onnx" in m
    assert (out / "gd" / "vocab.txt").exists()
