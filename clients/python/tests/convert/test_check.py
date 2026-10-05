"""Offline tests for `visionserve check` (convert/check.py, convert/htmlreport.py): the verdict and
the plain-language cause/fix text, the HTML page (structure, self-contained), the labels loaders,
the server pre-flight, and the whole command against an in-process fake server."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # clients/python

from PIL import Image  # noqa: E402

from visionserve.convert import check as ck  # noqa: E402
from visionserve.convert import htmlreport as hr  # noqa: E402
from visionserve.convert.report import FAIL, INFO, PASS, SKIP, WARN, Report, TierResult  # noqa: E402

IMN_MEAN, IMN_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

LEGACY = """name: det
task: detection
license: Apache-2.0
architecture: {arch}
model_file: model.onnx
input:
  width: 32
  height: 32
  layout: NCHW
  letterbox: {letterbox}
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
postprocess:
  conf_threshold: 0.5
labels: labels.txt
"""


def _photo(seed=0, w=64, h=40):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    a = np.stack([x * 4 % 256, y * 6 % 256, (x + y) * 3 % 256], -1).astype(np.float32)
    return Image.fromarray(np.clip(a + rng.normal(0, 8, a.shape), 0, 255).astype(np.uint8))


def _registry(tmp_path, arch="rf-detr", letterbox="false", block=None):
    d = tmp_path / "models" / "det"
    d.mkdir(parents=True)
    text = LEGACY.format(arch=arch, letterbox=letterbox)
    if block:
        text += block
    (d / "manifest.yaml").write_text(text)
    (d / "labels.txt").write_text("N/A\ncat\ndog\n")
    return tmp_path / "models"


def _run(tmp_path, codes=None, status=FAIL, block=False, b1_source="recipe", implied=None, b2=None, c=None,
         arch="rf-detr", letterbox="false"):
    """A CheckRun with a hand-made report: B1 with the given diagnosis codes. A server that
    letterboxes (code letterbox_server) has a manifest that says so."""
    padded = "letterbox_server" in (codes or {})
    if padded:
        letterbox = "true"
    resize = "letterbox" if padded else "squash"
    reg = _registry(tmp_path, arch=arch, letterbox=letterbox, block=f"preprocess:\n  resize: {resize}\n" if block else None)
    yaml = pytest.importorskip("yaml")  # noqa: F841
    inst = ck.load_installed(reg, "det")
    rep = Report(models=["det"], task="detection", architecture="rf-detr",
                 thresholds={"b1_mean_warn": 2.0, "b1_mean_fail": 8.0, "b2_iou": 0.5})
    ml = {PASS: 0.4, WARN: 3.1, FAIL: 52.0}[status]
    rep.add(TierResult("B1", "preprocessing", status, f"mean|Δ| {ml}", model="det",
                       metrics={"images": 5, "mean_levels": ml, "diagnosis": dict(codes or {})}))
    rep.add(b2 or TierResult("B2", "outputs", SKIP, "no reference model was given. Pass --checkpoint", model="det"))
    rep.add(c or TierResult("C", "accuracy", SKIP, "no --labels: accuracy not measured", model="det"))
    args = SimpleNamespace(images="photos/", labels=None, reference=None, checkpoint=None, max_map_drop=1.0)
    plan = ck.CheckPlan(SimpleNamespace(description="ref", short_pre="the rfdetr package's preprocessing"), None,
                        b1_source)
    run = ck.CheckRun(args, inst, "http://127.0.0.1:1", rep, plan=plan, images=[object()] * 5, images_total=5)
    run.implied_norm = implied
    return run


# --------------------------------------------------------------------------------------------
# verdict, causes and fixes
# --------------------------------------------------------------------------------------------

def test_letterbox_fail_names_cause_and_legacy_fix(tmp_path):
    out = ck.summarise(_run(tmp_path, {"letterbox_server": 5}))
    assert out["verdict"] == FAIL
    assert "letterboxes" in out["reason"] and "training stretches" in out["reason"]
    b1 = out["summary"]["checks"][0]
    assert b1["status"] == FAIL and "52.0 gray levels" in b1["finding"]
    assert "`input.letterbox: false`" in b1["fix"] and "manifest.yaml" in b1["fix"]
    steps = " ".join(out["summary"]["next_steps"])
    assert "input.letterbox: false" in steps and "Restart `visionserve serve`" in steps
    assert "--checkpoint" in steps and "--labels" in steps  # what would enable B2 and C


def test_a_fix_never_asks_for_a_value_already_set(tmp_path):
    """SCRFD's manifest already says `letterbox: true` (its top-left pad): a reference that
    letterboxes centred must not be answered with "set `input.letterbox: true`"."""
    pytest.importorskip("yaml")
    out = ck.summarise(_run(tmp_path / "a", {"letterbox_reference": 5}, arch="scrfd", letterbox="true"))
    b1 = out["summary"]["checks"][0]
    assert "while the server pastes it at the top-left" in b1["cause"]
    assert b1["fix"].startswith("`input.letterbox: true` is already set in") and " set `" not in b1["fix"]
    assert "scrfd cannot serve letterbox (it serves top left pad)" in b1["fix"]
    assert not any("Restart" in s for s in out["summary"]["next_steps"])  # nothing to edit
    # The same rule for the other manifest fixes: geometry already squash, mean/std already declared.
    sq = _run(tmp_path / "b", {"letterbox_server": 5})
    sq.inst.doc["input"]["letterbox"] = False  # a server that pads although the manifest says squash
    assert ck.causes_for({"letterbox_server": 5}, sq)[0]["fix"].startswith("`input.letterbox: false` is already set")
    nm = ck.summarise(_run(tmp_path / "c", {"normalisation": 5}, implied=(IMN_MEAN, IMN_STD)))
    assert "is already set in" in nm["summary"]["checks"][0]["fix"]
    crop = ck.summarise(_run(tmp_path / "d", {"crop": 5}, arch="rf-detr"))["summary"]["checks"][0]["fix"]
    assert "set `input.crop: center`" in crop


def test_letterbox_fix_follows_a_preprocess_block(tmp_path):
    out = ck.summarise(_run(tmp_path, {"letterbox_server": 5}, block=True))
    assert "`preprocess.resize: squash`" in out["summary"]["checks"][0]["fix"]


def test_normalisation_fix_gives_recovered_values(tmp_path):
    out = ck.summarise(_run(tmp_path, {"normalisation": 5}, implied=([0.5, 0.5, 0.5], [0.501, 0.5, 0.499])))
    b1 = out["summary"]["checks"][0]
    assert "0.5 mean/std" in b1["cause"]
    assert "`input.normalize.mean` / `std` to [0.5, 0.5, 0.5] / [0.5, 0.5, 0.5]" in b1["fix"]


def test_channel_swap_hides_its_normalisation_side_effect(tmp_path):
    out = ck.summarise(_run(tmp_path, {"normalisation": 5, "channel_swap": 5}))
    b1 = out["summary"]["checks"][0]
    assert "BGR" in b1["cause"] and [c["code"] for c in b1["causes"]] == ["channel_swap"]


def test_shape_and_resample_causes(tmp_path):
    run = _run(tmp_path, {"shape": 2}, status=FAIL)
    run.b1_shapes = ([3, 32, 32], [3, 24, 32])
    b1 = ck.summarise(run)["summary"]["checks"][0]
    assert "[3, 32, 32]" in b1["cause"] and "[3, 24, 32]" in b1["cause"]
    w = ck.summarise(_run(tmp_path / "w", {"resample": 5}, status=WARN))
    assert w["verdict"] == WARN and "resize-filter" in w["summary"]["checks"][0]["cause"]


def test_pass_with_manifest_reference_does_not_claim_training_fidelity(tmp_path):
    out = ck.summarise(_run(tmp_path, {}, status=PASS, b1_source="manifest"))
    assert out["verdict"] == PASS
    assert "as its manifest declares" in out["reason"] and "--reference" in out["reason"]
    assert "not that the manifest matches your training" in out["summary"]["checks"][0]["finding"]


def test_pass_with_training_reference(tmp_path):
    out = ck.summarise(_run(tmp_path, {}, status=PASS))
    assert out["reason"].startswith("det behaves like its training pipeline on 5 photos")
    assert "0.4 gray levels" in out["reason"] and "not compared" in out["reason"]


def test_b2_and_c_rows_point_back_to_b1(tmp_path):
    b2 = TierResult("B2", "outputs", WARN, "x", model="det", metrics={
        "matched": 21, "unmatched_ref": 5, "unmatched_srv": 2, "matched_frac": 0.8, "mean_dconf": 0.045,
        "mean_box_px": 1.6})
    c = TierResult("C", "accuracy", FAIL, "x", model="det", metrics={
        "kind": "detection", "images": 200, "map_served": 40.9, "map_ref": 44.1, "delta_map": -3.2,
        "conf_threshold": 0.5})
    out = ck.summarise(_run(tmp_path, {"letterbox_server": 5}, b2=b2, c=c))
    rows = {r["tier"]: r for r in out["summary"]["checks"]}
    assert "21 of 26 boxes match" in rows["B2"]["finding"]
    assert "mAP 40.9, the original model 44.1 (-3.2 points" in rows["C"]["finding"]
    assert rows["B2"]["cause"] == rows["C"]["cause"] == "the preprocessing difference found in B1"


def test_served_only_accuracy_is_info_and_does_not_fail(tmp_path):
    c = TierResult("C", "served accuracy", INFO, "x", model="det", metrics={
        "kind": "detection", "served_only": True, "images": 200, "map_served": 47.6, "map50_served": 59.1,
        "conf_threshold": 0.5})
    out = ck.summarise(_run(tmp_path, {}, status=PASS, c=c))
    assert out["verdict"] == PASS
    assert "mAP 47.6 (mAP50 59.1)" in out["summary"]["checks"][2]["finding"]


def test_everything_skipped_is_a_warning(tmp_path):
    run = _run(tmp_path, {}, status=PASS)
    run.report.tiers[0] = TierResult("B1", "preprocessing", SKIP, "no image input", model="det")
    out = ck.summarise(run)
    assert out["verdict"] == WARN and "nothing" in out["reason"]


def test_text_starts_with_the_verdict_line(tmp_path):
    out = ck.summarise(_run(tmp_path, {"letterbox_server": 5}))
    text = ck.render_text(out)
    first = text.splitlines()[0]
    assert re.match(r"^(PASS|WARN|FAIL): \S", first) and first.startswith("FAIL: det ")
    assert "Next steps" in text and "Details (the converter's tier report)" in text
    path = str(out["summary"]["manifest"])
    assert any(path in ln for ln in text.splitlines()), "a path is never wrapped mid-way"


# --------------------------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------------------------

def _html(tmp_path, reason_extra=""):
    run = _run(tmp_path, {"letterbox_server": 5})
    out = ck.summarise(run)
    out["reason"] += reason_extra
    img = _photo()
    figs = [{"title": "What the model sees", "caption": "c",
             "panels": [(img, "reference"), (img, "server"), (hr.heatmap_image(np.random.rand(40, 64) * 20, 8.0), "Δ")]},
            {"title": "Boxes", "caption": "c", "panels": [(hr.draw_boxes(
                img, [{"cls": "cat", "conf": 0.9, "bbox": [2, 2, 20, 20]}],
                [{"cls": "cat", "conf": 0.8, "bbox": [3, 3, 20, 20]}]), "b")]}]
    return out, hr.render_html(out, figs)


def test_html_is_one_self_contained_file(tmp_path):
    out, page = _html(tmp_path)
    for sid in ('id="verdict"', 'id="summary"', 'id="details"', 'id="next-steps"'):
        assert sid in page
    assert 'class="banner fail"' in page and "prefers-color-scheme: dark" in page
    assert "max-width:960px" in page and "system-ui" in page and "nth-child(even)" in page
    assert page.count("data:image/jpeg;base64,") == 4
    assert not re.search(r'(src|href)\s*=\s*["\']?\s*(https?:)?//', page), "nothing may be fetched"
    assert "<link" not in page and "<script" not in page
    assert "<code>input.letterbox: false</code>" in page


def test_html_escapes_text(tmp_path):
    _, page = _html(tmp_path, reason_extra=' <script>alert("x")</script>')
    assert "<script>" not in page and "&lt;script&gt;" in page


def test_heatmap_and_tensor_image():
    im = hr.heatmap_image(np.array([[0.0, 4.0], [8.0, 100.0]]), 8.0)
    a = np.asarray(im)
    assert a.shape == (2, 2, 3) and tuple(a[0, 0]) == (0, 0, 4) and tuple(a[1, 0]) == tuple(a[1, 1])
    t = ((np.full((3, 2, 2), 0.5) - np.array(IMN_MEAN)[:, None, None]) / np.array(IMN_STD)[:, None, None])
    assert np.asarray(hr.tensor_image(t, IMN_MEAN, IMN_STD)).max() in (127, 128)


# --------------------------------------------------------------------------------------------
# manifest, labels, server pre-flight
# --------------------------------------------------------------------------------------------

def test_load_installed_reads_block_and_finds_by_manifest_name(tmp_path):
    pytest.importorskip("yaml")
    reg = _registry(tmp_path, letterbox="true")
    (reg / "det").rename(reg / "some-dir")
    inst = ck.load_installed(reg, "det")
    assert inst.bundle.letterbox and inst.bundle.labels == ["N/A", "cat", "dog"] and not inst.uses_block
    with pytest.raises(ck.SetupError, match="not installed"):
        ck.load_installed(reg, "missing")


def test_load_installed_resolves_the_architecture(tmp_path):
    """The manifest reference is what the Go model applies (spec.resolve_arch): SCRFD's legacy
    letterbox is its top-left pad and its normalize is in 0..255 units; the bundle carries the
    mean/std in [0,1] units, which B1's gray-level arithmetic assumes."""
    pytest.importorskip("yaml")
    d = tmp_path / "models" / "scrfd"
    d.mkdir(parents=True)
    (d / "manifest.yaml").write_text(LEGACY.format(arch="scrfd", letterbox="true").replace(
        "[0.485, 0.456, 0.406]", "[127.5, 127.5, 127.5]").replace("[0.229, 0.224, 0.225]", "[128.0, 128.0, 128.0]"))
    inst = ck.load_installed(tmp_path / "models", "det")
    assert inst.spec.resize == "top_left_pad" and inst.spec.rescale is False
    assert np.allclose(inst.bundle.mean, [0.5] * 3) and np.allclose(inst.bundle.std, [128 / 255] * 3)
    d2 = tmp_path / "m2" / "sam"
    d2.mkdir(parents=True)
    (d2 / "manifest.yaml").write_text(LEGACY.format(arch="mobile-sam", letterbox="true"))
    assert ck.load_installed(tmp_path / "m2", "det").fixed_by_export
    (d2 / "manifest.yaml").write_text(LEGACY.format(arch="mobile-sam", letterbox="true")
                                      + "preprocess:\n  resize: long_side_pad\n")
    with pytest.raises(ck.SetupError, match="fixed by its export"):
        ck.load_installed(tmp_path / "m2", "det")


def test_load_labels_coco_subset_and_csv(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    _photo(1).save(imgs / "a.jpg")
    _photo(2).save(imgs / "b.jpg")
    coco = {"images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "zz.jpg"}, {"id": 3, "file_name": "b.jpg"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 5, 5]},
                            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [1, 1, 5, 5]}],
            "categories": [{"id": 1, "name": "cat"}]}
    f = tmp_path / "ann.json"
    f.write_text(json.dumps(coco))
    work = tmp_path / "w"
    work.mkdir()
    ev = ck.load_labels(f, imgs, 200, work)
    assert ev.kind == "detection" and [i for _, i in ev.items] == [1, 3]  # zz.jpg is not in --images
    assert str(f) in ev.source and str(work) not in ev.source
    c = tmp_path / "l.csv"
    c.write_text("image,label\na.jpg,cat\nb.jpg,dog\nmissing.jpg,cat\n")
    ev = ck.load_labels(c, imgs, 200, work)
    assert ev.kind == "classification" and [g for _, g in ev.items] == ["cat", "dog"]
    with pytest.raises(ck.SetupError):
        ck.load_labels(tmp_path / "nope.txt", imgs, 200, work)


def test_host_paths_maps_container_prefixes_back():
    m = {"/root/.models": "/home/u/reg", "/in/1": "/home/u/data", "/in/12": "/x"}
    out = ck.host_paths({"fix": "in /root/.models/det/manifest.yaml set ...", "l": ["/in/1/photos", "/in/12/a",
                                                                                    "/in/123"]}, m)
    assert out == {"fix": "in /home/u/reg/det/manifest.yaml set ...", "l": ["/home/u/data/photos", "/x/a", "/in/123"]}
    assert ck.host_paths({"a": "/root/.models"}, {}) == {"a": "/root/.models"}


def test_connect_preflight_messages():
    with pytest.raises(ck.SetupError, match="visionserve serve --models /reg"):
        ck.connect("http://x", "det", Path("/reg"), health=lambda u: False)
    with pytest.raises(ck.SetupError, match="does not serve 'det'"):
        ck.connect("http://x", "det", Path("/reg"), health=lambda u: True, listed=lambda u: ["a"],
                   registered=lambda u, n: False)
    ck.connect("http://x", "det", Path("/reg"), health=lambda u: True, listed=lambda u: [],
               registered=lambda u, n: True)  # installed after the server started


def test_usage_errors_exit_2_and_json_stays_one_object(tmp_path, capsys):
    from visionserve.convert import cli
    assert cli.main(["check", "det", "--images", str(tmp_path / "nope")]) == 2
    assert ck.main(["det", "--images", str(tmp_path / "nope"), "--json"]) == 2
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["verdict"] == "ERROR" and "--images" in out["reason"] and set(out) == {"verdict", "reason", "summary",
                                                                                      "details"}


# --------------------------------------------------------------------------------------------
# the whole command against a fake server
# --------------------------------------------------------------------------------------------

class _FakeServer:
    """Answers /api/preprocess with the manifest's OWN declared preprocessing (what the Go server
    does) and /api/load with success."""
    spec = None

    def __init__(self, url, timeout=0):
        self.url = url

    def load(self, name):
        return {"model": name, "state": "loaded"}

    def preprocess(self, name, image, prompt=None):
        from visionserve.convert.spec import apply_spec
        with Image.open(image) as im:
            x, meta = apply_spec(im.convert("RGB"), _FakeServer.spec)
        return SimpleNamespace(inputs={"input": x}, meta=meta)


@pytest.fixture
def served(monkeypatch, tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("yaml")
    from onnx import TensorProto, helper

    def make(arch, letterbox):
        reg = _registry(tmp_path, arch=arch, letterbox=letterbox)
        g = helper.make_graph([helper.make_node("Identity", ["input"], ["out"])], "g",
                              [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])],
                              [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3, 32, 32])])
        onnx.save(helper.make_model(g), str(reg / "det" / "model.onnx"))
        photos = tmp_path / "photos"
        photos.mkdir(exist_ok=True)
        for i in range(3):
            _photo(i).save(photos / f"p{i}.png")
        _FakeServer.spec = ck.load_installed(reg, "det").spec
        import visionserve.client
        monkeypatch.setattr(visionserve.client, "Client", _FakeServer)
        monkeypatch.setattr(ck, "connect", lambda *a, **k: None)
        return reg, photos
    return make


def test_check_end_to_end_wrong_letterbox_fails_with_cause(served, tmp_path, capsys):
    reg, photos = served("rf-detr", "true")  # RF-DETR is trained squashed: the recipe reference knows it
    report = tmp_path / "r.html"
    code = ck.main(["det", "--images", str(photos), "--models", str(reg), "--server", "http://fake",
                    "--report", str(report)])
    text = capsys.readouterr().out
    assert code == 1, text
    assert text.startswith("FAIL: det does not see photos the way it was trained: the server letterboxes")
    assert "`input.letterbox: false`" in text
    page = report.read_text()
    assert 'class="banner fail"' in page and "data:image/jpeg;base64," in page


def test_check_end_to_end_manifest_reference_passes(served, capsys):
    reg, photos = served("efficientnet", "false")  # no known recipe: compare with the declared spec
    code = ck.main(["det", "--images", str(photos), "--models", str(reg), "--server", "http://fake", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["verdict"] == PASS
    assert out["summary"]["reference"]["kind"] == "manifest"
    assert [c["status"] for c in out["summary"]["checks"]] == [PASS, SKIP, SKIP]
