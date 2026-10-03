"""Offline tests for the verification FLOW: tier C helpers, tiers B1/B2 with fake references and a
fake server, and the whole CLI / Python-API pipeline (export -> A -> install -> B/C -> report ->
uninstall on FAIL) with `visionserve pull` and the HTTP server replaced by in-process fakes.

The fake server is honest: it runs the INSTALLED ONNX on the manifest's preprocessing, so a test
reference that disagrees with the manifest really produces a measurable difference."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # clients/python

from PIL import Image  # noqa: E402

from visionserve.convert import evaluate as ev  # noqa: E402
from visionserve.convert import report as rp  # noqa: E402
from visionserve.convert import verify as vf  # noqa: E402
from visionserve.convert.reference import Prediction, Reference, manifest_preprocess  # noqa: E402
from visionserve.types import Classification, Detection, Result  # noqa: E402

IMN_MEAN, IMN_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _img(seed=0, w=64, h=48):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    a = np.stack([x * 4 % 256, y * 5 % 256, (x + y) * 3 % 256], -1).astype(np.float32)
    return Image.fromarray(np.clip(a + rng.normal(0, 10, a.shape), 0, 255).astype(np.uint8))


def _write_images(d: Path, n=3):
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        p = d / f"im{i}.png"
        _img(i).save(p)
        out.append(p)
    return out


# --------------------------------------------------------------------------------------------
# tier C helpers
# --------------------------------------------------------------------------------------------

def _coco(tmp_path):
    imgs = _write_images(tmp_path / "images", 2)
    d = {"images": [{"id": i + 1, "file_name": p.name, "width": 64, "height": 48} for i, p in enumerate(imgs)],
         "annotations": [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [5, 5, 20, 20]},
                         {"id": 2, "image_id": 2, "category_id": 7, "bbox": [30, 10, 10, 20]}],
         "categories": [{"id": 3, "name": "water_bottle"}, {"id": 7, "name": "motorcycle"}, {"id": 9, "name": "kite"}]}
    (tmp_path / "annotations").mkdir()
    f = tmp_path / "annotations" / "instances_val.json"
    f.write_text(json.dumps(d))
    return f


def test_load_coco_finds_images_and_restricts_to_max(tmp_path):
    f = _coco(tmp_path)
    e = ev.load_eval(f)                       # images found in ../images
    assert e.kind == "detection" and len(e.items) == 2 and e.coco["annotations"][0]["area"] == 400
    e1 = ev.load_eval(f, max_n=1)
    assert len(e1.items) == 1 and len(e1.coco["annotations"]) == 1
    with pytest.raises(ValueError, match="--eval-images"):
        _no_images(tmp_path)


def _no_images(tmp_path):
    d = json.loads((tmp_path / "annotations" / "instances_val.json").read_text())
    d["images"][0]["file_name"] = "missing.png"
    g = tmp_path / "x.json"
    g.write_text(json.dumps(d))
    ev.load_eval(g)


def test_load_roboflow_dir_and_imagefolder(tmp_path):
    rf = tmp_path / "ds" / "test"
    _write_images(rf, 1)
    (rf / "_annotations.coco.json").write_text(json.dumps(
        {"images": [{"id": 0, "file_name": "im0.png"}], "annotations": [], "categories": [{"id": 1, "name": "a"}]}))
    assert ev.load_eval(tmp_path / "ds").kind == "detection"
    for c in ("cat", "dog"):
        _write_images(tmp_path / "folder" / c, 3)
    e = ev.load_eval(tmp_path / "folder", max_n=4)
    assert e.kind == "classification" and [g for _, g in e.items] == ["cat", "dog", "cat", "dog"]


def test_map_labels_by_name_and_alias():
    cats = {3: "water_bottle", 7: "motorcycle", 9: "kite"}
    aliased = []
    m, unmapped, uncovered = ev.map_labels(["Water Bottle", "motorbike", "N/A", "zebra"], cats, aliased)
    assert m == {"Water Bottle": 3, "motorbike": 7} and unmapped == ["zebra"] and uncovered == ["kite"]
    assert aliased == ["motorbike->motorcycle"]


def test_coco_map_perfect_and_empty(tmp_path):
    pytest.importorskip("pycocotools")
    e = ev.load_eval(_coco(tmp_path))
    perfect = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 0.9}
               for a in e.coco["annotations"]]
    assert ev.coco_map(e.coco, perfect, [1, 2], [3, 7]) == pytest.approx((100.0, 100.0))
    assert ev.coco_map(e.coco, [], [1, 2], [3, 7]) == (0.0, 0.0)


def test_grade_drop():
    assert ev.grade_drop(0.1, 0.5) == rp.PASS and ev.grade_drop(0.3, 0.5) == rp.WARN and ev.grade_drop(0.6, 0.5) == rp.FAIL
    assert ev.grade_drop(-2.0, 0.5) == rp.PASS


class _Ref(Reference):
    def __init__(self, dets_by_size=None, probs=None):
        self.dets, self.probs = dets_by_size, probs
        self.description = "fake reference"

    def predict(self, pil, prompt=None):
        if self.probs is not None:
            return Prediction(probs=self.probs)
        return Prediction(detections=list(self.dets))


class _Client:
    def __init__(self, dets=None, cls=None):
        self.dets, self.cls = dets or [], cls or []

    def predict(self, name, src, prompt=None, max_grasps_per_object=None):
        return Result(task="detection", model=name, detections=[Detection(d["bbox"], d["cls"], d["conf"])
                                                                 for d in self.dets],
                      classifications=[Classification(c, p) for c, p in self.cls], device="cpu")


def test_tier_c_detection_reference_vs_served(tmp_path):
    pytest.importorskip("pycocotools")
    e = ev.load_eval(_coco(tmp_path))
    gt = {"cls": "water bottle", "conf": 0.9, "bbox": [5, 5, 20, 20]}
    plan = SimpleNamespace(b2=_Ref([gt]))
    b = SimpleNamespace(name="m", task="detection", labels=["water bottle", "motorbike"],
                        postprocess={"conf_threshold": 0.5})
    good = ev.tier_c(plan, _Client([gt]), b, e, 0.5)
    assert good.status == rp.PASS and good.metrics["delta_map"] == 0.0
    bad = ev.tier_c(plan, _Client([{**gt, "bbox": [40, 30, 20, 15]}]), b, e, 0.5)
    assert bad.status == rp.FAIL and bad.metrics["map_served"] < bad.metrics["map_ref"]
    assert any("kite" in n for n in bad.notes)  # the category nobody can detect is named


def test_tier_c_classification(tmp_path):
    for c in ("cat", "dog"):
        _write_images(tmp_path / "f" / c, 2)
    e = ev.load_eval(tmp_path / "f")
    b = SimpleNamespace(name="m", task="classification", labels=["cat", "dog", "cow"])
    plan = SimpleNamespace(b2=_Ref(probs={"cat": 0.6, "dog": 0.3, "cow": 0.1}))
    r = ev.tier_c(plan, _Client(cls=[("dog", 0.5), ("cat", 0.4)]), b, e, 0.5)
    assert r.metrics["top1_ref"] == 50.0 and r.metrics["top1_served"] == 50.0 and r.status == rp.PASS
    assert any("top-2" in n for n in r.notes)


# --------------------------------------------------------------------------------------------
# tiers B1 / B2 with fakes
# --------------------------------------------------------------------------------------------

class _PreRef(Reference):
    kind = "user"
    description = "fake preprocess"

    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def preprocess(self, pil, prompt=None):
        return {"input": manifest_preprocess(pil, 32, 32, self.mean, self.std)[0]}


class _PreClient:
    def preprocess(self, name, src, prompt=None):
        pil = src if isinstance(src, Image.Image) else Image.open(src).convert("RGB")
        x, meta = manifest_preprocess(pil, 32, 32, IMN_MEAN, IMN_STD)
        return SimpleNamespace(inputs={"input": x}, meta=meta)


def _plan(ref, **b):
    bundle = SimpleNamespace(name="m", task="classification", architecture="efficientnet", width=32, height=32,
                             mean=IMN_MEAN, std=IMN_STD, letterbox=False, labels=["a"], postprocess={}, **b)
    return vf.Plan(bundle, "input", ref, ref, None, "x.onnx")


def test_b1_pass_and_fail_with_diagnosis(tmp_path):
    th = rp.parse_thresholds([])
    imgs = vf.load_images(_write_images(tmp_path, 2))
    ok = vf.tier_b1(_plan(_PreRef(IMN_MEAN, IMN_STD)), _PreClient(), "m", imgs, th)
    assert ok.status == rp.PASS and ok.metrics["mean_levels"] < 1e-3
    bad = vf.tier_b1(_plan(_PreRef(None, None)), _PreClient(), "m", imgs, th)
    assert bad.status == rp.FAIL and bad.metrics["diagnosis"] == {"normalisation": 2}
    assert any("[2/2 images]" in n and "NO mean/std" in n for n in bad.notes)


def test_b1_official_documented_divergence_is_a_warning(tmp_path):
    ref = _PreRef(None, None)
    ref.kind, ref.known_divergence = "official", "HF centre-crops"
    r = vf.tier_b1(_plan(ref), _PreClient(), "m", vf.load_images(_write_images(tmp_path, 1)), rp.parse_thresholds([]))
    assert r.status == rp.WARN and any("documented divergence" in n for n in r.notes)


def test_b2_skips_without_real_images_and_grades_detections(tmp_path):
    th = rp.parse_thresholds([])
    d = {"cls": "a", "conf": 0.9, "bbox": [1, 1, 10, 10]}
    plan = _plan(_Ref([d]))
    plan.bundle.task, plan.bundle.postprocess = "detection", {"conf_threshold": 0.5}
    r = vf.tier_b2(plan, _Client([d]), "m", [vf.synthetic_image()], th)
    assert r.status == rp.SKIP and "--images" in r.summary
    imgs = vf.load_images(_write_images(tmp_path, 2))
    assert vf.tier_b2(plan, _Client([d]), "m", imgs, th).status == rp.PASS
    r = vf.tier_b2(plan, _Client([{**d, "cls": "b"}]), "m", imgs, th)
    assert r.status == rp.FAIL and any("a->b" in n for n in r.notes)
    plan0 = _plan(_Ref([]))
    plan0.bundle.task = "detection"
    assert vf.tier_b2(plan0, _Client([]), "m", imgs, th).status == rp.WARN  # vacuous, said so


def test_spread_is_deterministic_and_even():
    assert vf.spread(list(range(10)), 3) == [0, 4, 9]
    assert vf.spread([1, 2], 8) == [1, 2]


# --------------------------------------------------------------------------------------------
# the whole pipeline, offline: tiny torch classifier, fake `visionserve pull`, fake server
# --------------------------------------------------------------------------------------------

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")


class TinyCls(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.fc = torch.nn.Linear(8, 3)

    def forward(self, x):
        return self.fc(torch.relu(self.conv(x)).mean((2, 3)))


class FakeServerClient:
    """Runs the INSTALLED model the way the Go server would: manifest preprocessing + ORT + softmax."""

    def __init__(self, models_dir):
        self.models = Path(models_dir)
        self.loaded = []

    def _man(self, name):
        import yaml
        return yaml.safe_load((self.models / name / "manifest.yaml").read_text())

    def load(self, name):
        self.loaded.append(name)
        return {"model": name, "state": "loaded"}

    def preprocess(self, name, src, prompt=None):
        m = self._man(name)
        pil = src if isinstance(src, Image.Image) else Image.open(src).convert("RGB")
        i = m["input"]
        x, meta = manifest_preprocess(pil, i["width"], i["height"], i["normalize"]["mean"], i["normalize"]["std"],
                                      i["letterbox"])
        return SimpleNamespace(inputs={"input": x}, meta=meta)

    def predict(self, name, src, prompt=None, max_grasps_per_object=None):
        import onnxruntime as ort
        m = self._man(name)
        if isinstance(src, (bytes, bytearray)):
            import io
            src = Image.open(io.BytesIO(src)).convert("RGB")
        x = self.preprocess(name, src).inputs["input"]
        logits = ort.InferenceSession(str(self.models / name / m["model_file"]),
                                      providers=["CPUExecutionProvider"]).run(None, {"input": x})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        labels = (self.models / name / "labels.txt").read_text().split()
        top = np.argsort(-p)[:5]
        return Result(task="classification", model=name, device="cpu",
                      classifications=[Classification(labels[k], float(p[k])) for k in top])


@pytest.fixture
def offline(monkeypatch, tmp_path):
    from visionserve.convert import cli, serverctl

    def fake_install(bundles, staging, models_dir, force, dry_run):
        for b in bundles:
            d = b.write(staging)
            if not dry_run:
                dst = Path(models_dir) / b.name
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(d, dst)

    clients = []

    def fake_acquire(names, models_dir, **kw):
        c = FakeServerClient(models_dir)
        clients.append(c)
        return serverctl.ServerHandle("http://fake", temporary=True, client=c)

    monkeypatch.setattr(cli, "install", fake_install)
    monkeypatch.setattr(serverctl, "acquire", fake_acquire)
    imgs = tmp_path / "imgs"
    _write_images(imgs, 3)
    labels = tmp_path / "labels.txt"
    labels.write_text("a\nb\nc\n")
    return SimpleNamespace(models=tmp_path / "models", imgs=imgs, labels=labels, clients=clients, tmp=tmp_path)


def test_api_export_in_memory_module_with_user_preprocess_passes(offline):
    pytest.importorskip("yaml")
    from visionserve.convert import export

    def train_tf(pil):  # exactly the declared spec -> B1 ~0
        return manifest_preprocess(pil, 32, 32, IMN_MEAN, IMN_STD)[0][0]

    r = export(TinyCls(), name="tiny", task="classification", input=32, labels=["a", "b", "c"],
               license="MIT", preprocess=train_tf, images=offline.imgs, models_dir=offline.models, verify=3)
    assert r.ok and r.status == rp.PASS, r.table()
    assert r.tier("A").status == rp.PASS and r.tier("B1").metrics["mean_levels"] < 1e-3
    assert r.tier("B2").metrics["top1_agree"] == 1.0
    assert "your preprocess" in r.tier("B1").title
    d = offline.models / "tiny"
    assert json.loads((d / "convert-report.json").read_text())["status"] == "PASS"
    assert "# verify " in (d / "manifest.yaml").read_text()
    assert offline.clients[0].loaded == ["tiny"]


def test_cli_wrong_reference_script_fails_and_uninstalls(offline, tmp_path):
    pytest.importorskip("yaml")
    from visionserve.convert import cli
    script = tmp_path / "build.py"
    script.write_text("import sys\nsys.path.insert(0, %r)\nfrom test_verify_flow import TinyCls\n"
                      "def build_model():\n    return TinyCls()\n" % str(Path(__file__).parent))
    ref = tmp_path / "ref.py"
    ref.write_text("import numpy as np\ndef preprocess(img):\n"
                   "    return np.asarray(img.convert('RGB').resize((32, 32)), np.float32).transpose(2, 0, 1) / 255.0\n")
    argv = ["pytorch", str(script), "--name", "tiny2", "--license", "MIT", "--task", "classification",
            "--input", "32x32", "--labels", str(offline.labels), "--models", str(offline.models),
            "--images", str(offline.imgs), "--reference-script", str(ref)]
    assert cli.main(argv) == 1
    assert not (offline.models / "tiny2").exists()
    rep = json.loads((offline.models / ".convert-failed" / "tiny2" / "convert-report.json").read_text())
    b1 = next(t for t in rep["tiers"] if t["tier"] == "B1")
    assert rep["uninstalled"] and b1["status"] == "FAIL" and "normalisation" in b1["metrics"]["diagnosis"]
    # --keep-on-fail keeps it
    assert cli.main(argv + ["--keep-on-fail", "--force"]) == 1
    assert (offline.models / "tiny2" / "convert-report.json").is_file()


def test_no_server_skips_tiers_with_a_reason(offline, monkeypatch):
    from visionserve.convert import export, serverctl
    monkeypatch.setattr(serverctl, "acquire", lambda *a, **k: None)
    r = export(TinyCls(), name="tiny3", task="classification", input=32, labels=["a", "b", "c"],
               license="MIT", models_dir=offline.models, bench=True)
    assert r.ok and r.tier("B1").status == rp.SKIP and "no usable VisionServe server" in r.tier("B1").summary
    assert r.tier("speed").status == rp.SKIP


def test_bad_threshold_is_refused_before_converting(offline):
    from visionserve.convert import cli
    rc = cli.main(["pytorch", "x.py", "--name", "t", "--license", "MIT", "--task", "classification", "--input",
                   "32x32", "--models", str(offline.models), "--threshold", "nope=1"])
    assert rc == 2 and not offline.models.exists()


def test_infer_format(tmp_path):
    from visionserve.convert.api import infer_format
    from visionserve.convert.common import ConvertError
    (tmp_path / "hf").mkdir()
    (tmp_path / "hf" / "config.json").write_text("{}")
    assert infer_format(tmp_path / "hf") == "hf"
    assert infer_format("PekingU/rtdetr_r50vd") == "hf"
    (tmp_path / "m.tflite").write_bytes(b"x")
    assert infer_format(tmp_path / "m.tflite") == "tflite"
    ts = tmp_path / "m.pt"
    torch.jit.save(torch.jit.script(TinyCls()), str(ts))
    assert infer_format(ts) == "torchscript"
    sd = tmp_path / "w.pth"
    torch.save(TinyCls().state_dict(), str(sd))
    with pytest.raises(ConvertError, match="format='pytorch'"):
        infer_format(sd)
