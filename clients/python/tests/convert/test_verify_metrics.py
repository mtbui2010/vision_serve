"""Offline tests for the verification tiers' pure parts: B2 metrics, B1 diagnosis, the report,
tier-C helpers, the generic reference/decoder and the server discovery logic. No network, no GPU,
no framework model (numpy + PIL + pycocotools only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # clients/python

from visionserve.convert import diagnose as dg  # noqa: E402
from visionserve.convert import metrics as mt  # noqa: E402
from visionserve.convert import report as rp  # noqa: E402
from visionserve.convert import serverctl  # noqa: E402
from visionserve.convert.reference import (ManifestReference, UserReference, decode_outputs,  # noqa: E402
                                           dpt_keep_aspect_size, manifest_meta, manifest_preprocess)

IMN_MEAN, IMN_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def det(cls, conf, x, y, w, h):
    return {"cls": cls, "conf": conf, "bbox": [x, y, w, h]}


# --------------------------------------------------------------------------------------------
# B2 metrics
# --------------------------------------------------------------------------------------------

def test_iou_basic():
    assert mt.iou_xywh([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert mt.iou_xywh([0, 0, 10, 10], [10, 10, 5, 5]) == 0.0
    assert abs(mt.iou_xywh([0, 0, 10, 10], [5, 0, 10, 10]) - 50 / 150) < 1e-12


def test_match_is_greedy_by_iou_and_class_aware():
    ref = [det("cat", 0.9, 0, 0, 10, 10), det("dog", 0.8, 100, 100, 10, 10)]
    srv = [det("cat", 0.9, 1, 0, 10, 10), det("cat", 0.7, 0, 0, 10, 10), det("cow", 0.8, 100, 100, 10, 10)]
    pairs, ur, us = mt.match_detections(ref, srv, 0.5)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]   # the exact box wins over the shifted one
    assert ur == [1] and us == [0, 2]                  # dog vs cow never match: different class


def test_compare_detections_metrics_and_boundary_flips():
    ref = [det("cat", 0.90, 0, 0, 100, 100), det("dog", 0.52, 300, 300, 50, 50)]
    srv = [det("cat", 0.88, 2, 0, 100, 100)]
    m = mt.compare_detections([(ref, srv, (640, 480))], conf_threshold=0.5, boundary=0.05)
    assert m["matched"] == 1 and m["boundary_ref"] == 1 and m["unmatched_ref"] == 0
    assert m["matched_frac"] == 1.0                    # the 0.52 dog is a threshold flip, not a miss
    assert abs(m["mean_dconf"] - 0.02) < 1e-9
    assert abs(m["max_box_px"] - 2.0) < 1e-9 and abs(m["mean_box_px"] - 1.0) < 1e-9
    # without a threshold the same dog IS a disagreement
    m2 = mt.compare_detections([(ref, srv, (640, 480))], conf_threshold=None)
    assert m2["matched_frac"] == 0.5


def test_compare_detections_vacuous_and_confusions():
    assert mt.compare_detections([([], [], (10, 10))], 0.5)["matched_frac"] is None
    m = mt.compare_detections([([det("cat", 0.9, 0, 0, 10, 10)], [det("dog", 0.9, 0, 0, 10, 10)], (100, 100))], 0.5)
    assert m["matched_frac"] == 0.0 and m["class_confusions"] == [("cat", "dog")]


def test_compare_detections_accepts_sdk_objects():
    from visionserve.types import Detection
    m = mt.compare_detections([([det("a", 0.9, 0, 0, 5, 5)], [Detection([0, 0, 5, 5], "a", 0.9)], (10, 10))], 0.5)
    assert m["matched"] == 1


def test_compare_classification():
    ref = {"cat": 0.7, "dog": 0.2, "cow": 0.1}
    m = mt.compare_classification([(ref, [("cat", 0.68), ("dog", 0.25)]), (ref, [("dog", 0.6), ("cat", 0.3)])])
    assert m["n"] == 2 and m["top1_agree"] == 0.5
    assert abs(m["max_dprob"] - 0.4) < 1e-9


def test_cosine_pearson_resize():
    a = np.array([1.0, 2.0, 3.0])
    assert abs(mt.cosine(a, 2 * a) - 1.0) < 1e-12
    with pytest.raises(ValueError):
        mt.cosine(a, np.ones(4))
    m = np.random.default_rng(0).random((20, 30))
    assert abs(mt.pearson(m, 3 * m + 7) - 1.0) < 1e-9   # invariant to the Go min-max normalisation
    assert mt.resize_map(m, 15, 10).shape == (10, 15)
    p = mt.percentile_ms([1, 2, 3, 4, 100])
    assert p["p50_ms"] == 3 and p["n"] == 5


# --------------------------------------------------------------------------------------------
# B1 diagnosis
# --------------------------------------------------------------------------------------------

def _photo(h=96, w=128, seed=0):
    """Smooth, non-constant RGB in [0,1] (HWC) — enough structure for correlations."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w] / max(h, w)
    base = np.stack([np.sin(6 * x + 1) * np.cos(4 * y), np.cos(5 * x * y + 2), np.sin(3 * (x + y))], -1)
    blobs = sum(np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 0.01)[..., None] * rng.random(3)
                for cx, cy in rng.random((6, 2)))
    img = base * 0.3 + blobs * 0.5 + 0.4
    return np.clip(img, 0, 1)


def _nchw(img, mean=IMN_MEAN, std=IMN_STD):
    x = (img - np.asarray(mean)) / np.asarray(std)
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def test_identical_tensors_have_no_diagnosis():
    s = _nchw(_photo())
    c = dg.compare_tensors(s, s, IMN_STD)
    assert c["same_shape"] and c["max_levels"] == 0
    assert dg.diagnose(s, s, {"pad_x": 0, "pad_y": 0}, IMN_MEAN, IMN_STD) == ([], [])


def test_missing_normalisation_is_diagnosed():
    img = _photo()
    codes, msgs = dg.diagnose(_nchw(img), _nchw(img, [0, 0, 0], [1, 1, 1]), {"pad_x": 0, "pad_y": 0},
                              IMN_MEAN, IMN_STD)
    assert codes == ["normalisation"] and "NO mean/std normalisation" in msgs[0]
    c = dg.compare_tensors(_nchw(img), _nchw(img, [0, 0, 0], [1, 1, 1]), IMN_STD)
    assert c["mean_levels"] > 20  # far beyond the default fail threshold


def test_other_mean_std_is_reported_with_the_implied_values():
    img = _photo()
    codes, msgs = dg.diagnose(_nchw(img), _nchw(img, [0.5] * 3, [0.5] * 3), None, IMN_MEAN, IMN_STD)
    assert codes == ["normalisation"] and "mean = std = 0.5" in msgs[0]


def test_missing_div255_is_diagnosed():
    img = _photo()
    ref = _nchw(img * 255.0, [0, 0, 0], [1, 1, 1])
    codes, msgs = dg.diagnose(_nchw(img, [0, 0, 0], [1, 1, 1]), ref, None, [0, 0, 0], [1, 1, 1])
    assert codes == ["scale_255"] and "/255" in msgs[0]


def test_bgr_swap_is_diagnosed():
    img = _photo()
    codes, msgs = dg.diagnose(_nchw(img, [0, 0, 0], [1, 1, 1]), _nchw(img[..., ::-1], [0, 0, 0], [1, 1, 1]),
                              None, [0, 0, 0], [1, 1, 1])
    assert "channel_swap" in codes and "RGB/BGR" in msgs[codes.index("channel_swap")]


def test_server_letterbox_vs_reference_squash():
    img = _photo(96, 128)
    from PIL import Image
    pil = Image.fromarray((img * 255).astype(np.uint8))
    lb, meta = manifest_preprocess(pil, 64, 64, IMN_MEAN, IMN_STD, letterbox=True)
    sq, _ = manifest_preprocess(pil, 64, 64, IMN_MEAN, IMN_STD, letterbox=False)
    assert meta["pad_y"] > 0
    codes, msgs = dg.diagnose(lb, sq, meta, IMN_MEAN, IMN_STD)
    assert codes[0] == "letterbox_server" and "LETTERBOXES" in msgs[0]
    # and the other way round, with the server's meta saying it did not pad
    codes, _ = dg.diagnose(sq, lb, {"pad_x": 0, "pad_y": 0}, IMN_MEAN, IMN_STD)
    assert codes[0] == "letterbox_reference"


def test_dark_edge_rows_are_not_mistaken_for_server_padding():
    img = _photo()
    img[:5] = 0.0  # a black band at the top of the PHOTO
    s = _nchw(img)
    r = _nchw(img, [0, 0, 0], [1, 1, 1])
    codes, _ = dg.diagnose(s, r, {"pad_x": 0, "pad_y": 0}, IMN_MEAN, IMN_STD)
    assert "letterbox_server" not in codes and "letterbox_reference" not in codes


def test_centre_crop_is_diagnosed():
    from PIL import Image
    img = (_photo(120, 160) * 255).astype(np.uint8)
    pil = Image.fromarray(img)
    srv, _ = manifest_preprocess(pil, 64, 64, IMN_MEAN, IMN_STD)
    crop = pil.crop((24, 15, 136, 105))  # central 70% x 75%
    ref, _ = manifest_preprocess(crop, 64, 64, IMN_MEAN, IMN_STD)
    codes, msgs = dg.diagnose(srv, ref, {"pad_x": 0, "pad_y": 0}, IMN_MEAN, IMN_STD)
    assert codes == ["crop"] and "CENTRE CROP" in msgs[0]


def test_shape_mismatch_recognises_an_aspect_kept_reference():
    s = np.zeros((1, 3, 80, 80), np.float32)
    s[0, :, 10:20, 10:20] = 1
    r = np.zeros((1, 3, 80, 120), np.float32)
    r[0, :, 10:20, 10:20] = 1
    codes, msgs = dg.diagnose(s, r, None, IMN_MEAN, IMN_STD, orig_size=(600, 400))
    assert codes[0] == "shape" and "KEEPS the aspect ratio" in msgs[0]
    assert not dg.compare_tensors(s, r)["same_shape"]


# --------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------

def test_thresholds_parse_and_refuse_unknown_keys():
    t = rp.parse_thresholds(["b1_mean_fail=10", {"b2_iou": 0.6}])
    assert t["b1_mean_fail"] == 10.0 and t["b2_iou"] == 0.6 and t["b1_mean_warn"] == 2.0
    with pytest.raises(ValueError, match="unknown threshold"):
        rp.parse_thresholds(["b1_mean_fial=10"])
    with pytest.raises(ValueError):
        rp.parse_thresholds(["b1_mean_fail"])


def test_grading_and_overall_status():
    assert rp.grade_low(1, 2, 8) == rp.PASS and rp.grade_low(3, 2, 8) == rp.WARN and rp.grade_low(9, 2, 8) == rp.FAIL
    assert rp.grade_high(0.99, 0.95, 0.8) == rp.PASS and rp.grade_high(0.5, 0.95, 0.8) == rp.FAIL
    assert rp.grade_high(None, 1, 0) == rp.SKIP
    r = rp.Report(models=["m"])
    r.add(rp.TierResult("A", "parity", rp.PASS, "ok"))
    r.add(rp.TierResult("speed", "speed", rp.INFO, "1 ms"))
    r.add(rp.TierResult("B2", "outputs", rp.SKIP, "no images"))
    assert r.status == rp.PASS and r.ok
    r.add(rp.TierResult("C", "acc", rp.ERROR, "boom"))
    assert r.status == rp.ERROR and r.ok          # a broken check is loud but does not uninstall
    r.add(rp.TierResult("B1", "pre", rp.FAIL, "bad"))
    assert r.status == rp.FAIL and not r.ok


def test_report_table_and_json(tmp_path):
    r = rp.Report(models=["m"], task="detection", architecture="rf-detr")
    r.add(rp.TierResult("B1", "preprocessing vs x", rp.PASS, "mean 0.3", model="m",
                        metrics={"a": np.float32(1.5), "arr": np.arange(3)}, notes=["hello"]))
    t = r.table()
    assert "overall: PASS" in t and "B1" in t and "[B1] hello" in t
    p = r.save(tmp_path / "x" / "convert-report.json")
    d = json.loads(p.read_text())
    assert d["status"] == "PASS" and d["tiers"][0]["metrics"]["arr"] == [0, 1, 2]


def test_annotate_manifest_is_comment_only_and_replaces_previous_block(tmp_path):
    yaml = pytest.importorskip("yaml")
    m = tmp_path / "manifest.yaml"
    m.write_text("# Generated by x\n# second\nname: m\ntask: detection\n")
    r = rp.Report(models=["m"])
    r.add(rp.TierResult("A", "parity", rp.PASS, "1e-6"))
    rp.annotate_manifest(m, r.manifest_comment_lines("m"))
    rp.annotate_manifest(m, r.manifest_comment_lines("m"))   # re-run: replaced, not stacked
    text = m.read_text()
    assert text.count("# verify ") == 1 and text.count("#   A parity") == 1
    assert text.startswith("# Generated by x\n# second\n# verify ")
    assert yaml.safe_load(text) == {"name": "m", "task": "detection"}
    with pytest.raises(ValueError):
        rp.annotate_manifest(m, ["name: injected"])


# --------------------------------------------------------------------------------------------
# reference / decoder
# --------------------------------------------------------------------------------------------

def _bundle(**kw):
    d = dict(name="m", task="detection", architecture="rt-detr", width=100, height=100, letterbox=False,
             mean=IMN_MEAN, std=IMN_STD, labels=["a", "b"], postprocess={"conf_threshold": 0.5, "max_detections": 10})
    d.update(kw)
    return SimpleNamespace(**d)


def test_decode_detection_maps_to_original_coordinates_squash_and_letterbox():
    logits = np.full((1, 2, 2), -10.0, np.float32)
    logits[0, 0, 1] = 5.0                       # query 0 -> class b
    boxes = np.array([[[0.5, 0.5, 0.2, 0.4], [0.1, 0.1, 0.1, 0.1]]], np.float32)
    b = _bundle()
    p = decode_outputs([logits, boxes], b, manifest_meta(b, 200, 400))   # squash: sx=0.5, sy=0.25
    (d,) = p.detections
    assert d["cls"] == "b" and abs(d["conf"] - 1 / (1 + np.exp(-5))) < 1e-6
    np.testing.assert_allclose(d["bbox"], [80, 120, 40, 160], atol=1e-4)
    b2 = _bundle(letterbox=True)
    meta = manifest_meta(b2, 200, 100)          # scale 0.5, pad_y 25
    assert meta["pad_y"] == 25 and meta["scale_x"] == 0.5
    (d,) = decode_outputs([boxes, logits], b2, meta).detections   # output order does not matter
    np.testing.assert_allclose(d["bbox"], [80, 10, 40, 80], atol=1e-4)  # y = (30 - 25) / 0.5


def test_decode_classification_depth_embed():
    b = _bundle(task="classification", architecture="efficientnet", labels=["x", "y"])
    p = decode_outputs([np.array([[0.0, np.log(3.0)]], np.float32)], b, {})
    assert abs(p.probs["y"] - 0.75) < 1e-6
    p = decode_outputs([np.zeros((1, 4, 5), np.float32)], _bundle(task="depth", architecture="midas"), {})
    assert p.depth.shape == (4, 5)
    p = decode_outputs([np.ones((1, 8), np.float32)], _bundle(task="embed", architecture="clip"), {})
    assert p.embedding.shape == (8,)


def test_manifest_and_user_references(tmp_path):
    from PIL import Image
    pil = Image.fromarray((_photo(60, 80) * 255).astype(np.uint8))
    b = _bundle(task="classification", architecture="efficientnet", width=32, height=32)
    fwd = lambda feeds: [np.array([[1.0, 0.0]], np.float32)]  # noqa: E731
    ref = ManifestReference(b, "input", fwd)
    x = ref.preprocess(pil)["input"]
    assert x.shape == (1, 3, 32, 32) and x.dtype == np.float32
    assert ref.predict(pil).probs["a"] > 0.7 and ref.kind == "manifest"
    # a user transform returning CHW gets a batch dim; a predict() returning a list of probs is mapped to labels
    u = UserReference(b, "input", lambda im: np.zeros((3, 32, 32)), lambda im: [0.1, 0.9])
    assert u.preprocess(pil)["input"].shape == (1, 3, 32, 32)
    assert u.predict(pil).probs == {"a": 0.1, "b": 0.9}
    with pytest.raises(ValueError):
        UserReference(b, "input", lambda im: np.zeros((32, 32))).preprocess(pil)


def test_load_reference_script(tmp_path):
    from visionserve.convert.reference import load_reference_script
    f = tmp_path / "ref.py"
    f.write_text("import numpy as np\ndef preprocess(img):\n    return np.zeros((3, 4, 4), np.float32)\n")
    s = load_reference_script(f)
    assert callable(s["preprocess"]) and s["predict"] is None
    g = tmp_path / "bad.py"
    g.write_text("x = 1\n")
    with pytest.raises(ValueError, match="must define preprocess"):
        load_reference_script(g)


# --------------------------------------------------------------------------------------------
# server discovery (no sockets except free_port; health/popen/which are fakes)
# --------------------------------------------------------------------------------------------

class FakeProc:
    def __init__(self, rc=None):
        self.rc, self.terminated = rc, False

    @property
    def returncode(self):
        return self.rc

    def poll(self):
        return self.rc

    def terminate(self):
        self.terminated, self.rc = True, 0

    def wait(self, timeout=None):
        return self.rc

    def kill(self):
        self.rc = -9


def _acq(tmp_path, **kw):
    logs = []
    kw.setdefault("env", {"VISIONSERVE_BIN": "visionserve", "ORT_DYLIB_PATH": "/x.so"})
    kw.setdefault("sleep", lambda s: None)
    h = serverctl.acquire(["m"], tmp_path / "models", workdir=tmp_path, log=logs.append, **kw)
    return h, "\n".join(logs)


def test_no_server_flag_skips_with_reason(tmp_path):
    h, log = _acq(tmp_path, no_server=True)
    assert h is None and "--no-server" in log


def test_running_server_with_the_model_is_used(tmp_path):
    h, log = _acq(tmp_path, url="http://h:1", health=lambda u: True, listed=lambda u: ["m", "x"])
    assert h is not None and not h.temporary and h.url == "http://h:1"


def test_model_listed_late_after_install_is_not_another_registry(tmp_path):
    """A server that rescans /api/models at most once a second lists a model just installed only
    on the next try: asked again after ~1.1 s, it is used, with no 'different registry' warning."""
    answers, slept = [[], ["m"]], []
    h, log = _acq(tmp_path, url="http://h:1", health=lambda u: True, listed=lambda u: answers.pop(0),
                  sleep=slept.append)
    assert h is not None and not h.temporary and "different registry" not in log and "WARNING" not in log
    assert slept == [serverctl.LIST_RETRY_SECONDS] and serverctl.LIST_RETRY_SECONDS > 1.0
    # Listed at once: no wait.
    slept.clear()
    _acq(tmp_path, url="http://h:1", health=lambda u: True, listed=lambda u: ["m"], sleep=slept.append)
    assert slept == []


def test_server_on_another_registry_is_not_used(tmp_path):
    started = []

    def popen(cmd, **kw):
        started.append(cmd)
        return FakeProc()
    health = lambda u: True  # noqa: E731 — both the user's server and the temporary one answer
    h, log = _acq(tmp_path, url="http://h:1", health=health, listed=lambda u: ["other"], popen=popen,
                  which=lambda b: "/usr/bin/visionserve")
    assert "different registry" in log and h is not None and h.temporary
    cmd = started[0]
    assert cmd[1] == "serve" and cmd[cmd.index("--models") + 1] == str(tmp_path / "models")
    assert cmd[cmd.index("--idle-unload-seconds") + 1] == "0" and cmd[cmd.index("--addr") + 1].startswith("127.0.0.1:")
    h.close()


def test_no_binary_means_skip_with_reason(tmp_path):
    h, log = _acq(tmp_path, health=lambda u: False, which=lambda b: None)
    assert h is None and "SKIPPED" in log and "VISIONSERVE_BIN" in log


def test_temporary_server_that_exits_reports_its_log(tmp_path):
    def popen(cmd, stdout=None, **kw):
        stdout.write(b"ORT: cannot load libonnxruntime.so\n")
        stdout.flush()
        return FakeProc(rc=1)
    h, log = _acq(tmp_path, health=lambda u: False, popen=popen, which=lambda b: "/bin/vs")
    assert h is None and "exited with code 1" in log and "libonnxruntime" in log


def test_temporary_server_timeout_is_stopped(tmp_path):
    procs = []

    def popen(cmd, **kw):
        procs.append(FakeProc())
        return procs[-1]
    h, log = _acq(tmp_path, health=lambda u: False, popen=popen, which=lambda b: "/bin/vs", start_timeout=0.05)
    assert h is None and "did not answer" in log and procs[0].terminated


def test_missing_ort_dylib_is_warned(tmp_path):
    h, log = _acq(tmp_path, health=lambda u: u.startswith("http://127.0.0.1"), popen=lambda c, **k: FakeProc(),
                  which=lambda b: "/bin/vs", env={"VISIONSERVE_BIN": "visionserve"})
    assert "ORT_DYLIB_PATH is not set" in log and h is not None
    h.close()


def test_phrase_in_merged_label_for_grounding_dino():
    from visionserve.convert.metrics import compare_detections, phrase_in_merged_label
    assert phrase_in_merged_label("cup water bottle", "water bottle")
    assert phrase_in_merged_label("cup water bottle", "cup")
    assert not phrase_in_merged_label("cup water bottle", "bottle cap")  # whole phrase, in order
    assert not phrase_in_merged_label("cupboard", "cup")                # whole words only
    ref = [{"cls": "cup water bottle", "conf": 0.6, "bbox": [0, 0, 10, 10]}]
    srv = [{"cls": "cup", "conf": 0.6, "bbox": [0, 0, 10, 10]}]
    strict = compare_detections([(ref, srv, (100, 100))])
    merged = compare_detections([(ref, srv, (100, 100))], label_match=phrase_in_merged_label)
    assert strict["matched"] == 0 and merged["matched"] == 1 and merged["matched_via_merged_label"] == 1


# --------------------------------------------------------------------------------------------
# keep_aspect (depth-anything-v2): DPT's rule, identical to internal/imageproc.DPTKeepAspectSize
# --------------------------------------------------------------------------------------------

# (w, h, tw, th, multiple) -> HF get_resize_output_image_size(keep_aspect_ratio=True); the same
# table as internal/imageproc/keepaspect_test.go (ties at .5 are Python's half-to-even round).
DPT_SIZES = [
    ((848, 480, 518, 518, 14), (910, 518)), ((480, 848, 518, 518, 14), (518, 910)),
    ((640, 427, 518, 518, 14), (518, 350)), ((1, 1, 518, 518, 14), (518, 518)),
    ((10000, 7000, 518, 518, 14), (742, 518)), ((640, 480, 384, 384, 32), (512, 384)),
    ((640, 480, 518, 392, 14), (518, 392)), ((200, 201, 100, 100, 1), (100, 100)),
    ((200, 203, 100, 100, 1), (100, 102)), ((1036, 1050, 518, 518, 14), (518, 532)),
    ((1036, 1078, 518, 518, 14), (518, 532)), ((600, 200, 300, 300, 1), (900, 300)),
]


@pytest.mark.parametrize("args,want", DPT_SIZES)
def test_dpt_keep_aspect_size(args, want):
    assert dpt_keep_aspect_size(*args) == want


def test_dpt_keep_aspect_size_never_zero():
    assert dpt_keep_aspect_size(3000, 20, 518, 518, 14) == (518, 14)   # HF: (518, 0), then crashes
    assert dpt_keep_aspect_size(640, 480, 518, 518, 0) == (691, 518)   # multiple 0 = 1


def test_manifest_preprocess_keep_aspect():
    from PIL import Image
    pil = Image.fromarray((np.random.default_rng(0).random((480, 848, 3)) * 255).astype(np.uint8))
    x, meta = manifest_preprocess(pil, 518, 518, IMN_MEAN, IMN_STD, keep_aspect=True, multiple_of=14)
    assert x.shape == (1, 3, 518, 910)
    assert meta == {"orig_width": 848, "orig_height": 480, "scale_x": 910 / 848, "scale_y": 518 / 480,
                    "pad_x": 0, "pad_y": 0}
    # bicubic, exactly PIL's
    want = np.asarray(pil.resize((910, 518), Image.BICUBIC), np.float32) / 255.0
    want = ((want - np.asarray(IMN_MEAN, np.float32)) / np.asarray(IMN_STD, np.float32)).transpose(2, 0, 1)
    np.testing.assert_allclose(x[0], want, atol=1e-6)
    b = _bundle(task="depth", architecture="midas", width=518, height=518, keep_aspect=True, multiple_of=14)
    assert manifest_meta(b, 848, 480) == meta
    ref = ManifestReference(b, "pixel_values")
    assert ref.preprocess(pil)["pixel_values"].shape == (1, 3, 518, 910)
    assert "keep-aspect" in ref.description


def test_diagnose_quiet_on_matching_keep_aspect_tensors():
    """Server and reference both keep the aspect (910x518 for 848x480): no shape/geometry finding
    (a different filter on a noisy image may still read as a per-channel gain — not geometry)."""
    from PIL import Image
    from visionserve.convert.common import sample_image
    pil = Image.fromarray(sample_image(848, 480, seed=1))   # smooth gradients + mild noise, photo-like
    srv, meta = manifest_preprocess(pil, 518, 518, IMN_MEAN, IMN_STD, keep_aspect=True, multiple_of=14)
    ref = np.asarray(pil.resize((910, 518), Image.BILINEAR), np.float32) / 255.0   # another filter
    ref = ((ref - np.asarray(IMN_MEAN, np.float32)) / np.asarray(IMN_STD, np.float32)).transpose(2, 0, 1)[None]
    codes, _ = dg.diagnose(srv, ref, meta, IMN_MEAN, IMN_STD, pil.size)
    geometry = {"shape", "letterbox_server", "letterbox_reference", "crop", "flip", "spatial"}
    assert not set(codes) & geometry, codes
    codes, _ = dg.diagnose(srv, srv.copy(), meta, IMN_MEAN, IMN_STD, pil.size)
    assert codes == [], codes
