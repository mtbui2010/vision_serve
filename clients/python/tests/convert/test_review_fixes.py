"""Regression tests for the converter review findings (C1-C8). Each test reproduces the bug it is
named after; all are offline (fake `visionserve pull`, fake server)."""
from __future__ import annotations

import io
import json
import shutil
import sys
import textwrap
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # clients/python

from visionserve.convert import cli, serverctl  # noqa: E402
from visionserve.convert import evaluate as ev  # noqa: E402
from visionserve.convert import report as rp  # noqa: E402
from visionserve.convert.common import ConvertError, license_scan  # noqa: E402

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")


class TinyCls(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.fc = torch.nn.Linear(8, 3)

    def forward(self, x):
        return self.fc(torch.relu(self.conv(x)).mean((2, 3)))


TINY_SRC = textwrap.dedent('''
    import torch

    class DetectionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.conv = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
            self.fc = torch.nn.Linear(8, 3)

        def forward(self, x):
            return self.fc(torch.relu(self.conv(x)).mean((2, 3)))
''')


@pytest.fixture
def fake_ultralytics(tmp_path, monkeypatch):
    """A package literally named `ultralytics` whose class is a harmless tiny classifier: what the
    gate must catch is WHERE the class comes from, not what it computes."""
    root = tmp_path / "pkgs"
    (root / "ultralytics" / "nn").mkdir(parents=True)
    (root / "ultralytics" / "__init__.py").write_text("")
    (root / "ultralytics" / "nn" / "__init__.py").write_text("")
    (root / "ultralytics" / "nn" / "tasks.py").write_text(TINY_SRC)
    monkeypatch.syspath_prepend(str(root))
    import importlib
    mod = importlib.import_module("ultralytics.nn.tasks")
    yield mod
    for k in [k for k in sys.modules if k == "ultralytics" or k.startswith("ultralytics.")]:
        del sys.modules[k]


def _fake_install(bundles, staging, models_dir, force, dry_run):
    """`visionserve pull` stand-in: Go overwrites the directory in place under --force."""
    for b in bundles:
        d = b.write(staging)
        if dry_run:
            continue
        dst = Path(models_dir) / b.name
        if dst.exists() and not force:
            raise ConvertError(f"visionserve refused {b.name}: already exists")
        shutil.copytree(d, dst, dirs_exist_ok=True)


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "install", _fake_install)
    monkeypatch.setattr(serverctl, "acquire", lambda *a, **k: None)
    monkeypatch.setattr(serverctl, "_health", lambda *a, **k: False)
    return SimpleNamespace(models=tmp_path / "models", tmp=tmp_path)


def _argv(name, models, *extra, src="x.py", fmt="pytorch"):
    return [fmt, str(src), "--name", name, "--license", "MIT", "--task", "classification", "--input", "32x32",
            "--models", str(models), "--no-server", *extra]


def _build_script(tmp_path, body):
    s = tmp_path / "build.py"
    s.write_text(body)
    return s


# --------------------------------------------------------------------------------------------
# C1: license gate on in-memory modules, --script builds, TF/Keras/TFLite metadata
# --------------------------------------------------------------------------------------------

def test_c1_api_in_memory_ultralytics_module_refused(offline, fake_ultralytics):
    from visionserve.convert import export
    with pytest.raises(ConvertError, match="AGPL"):
        export(fake_ultralytics.DetectionModel(), name="yolo", task="classification", input=32,
               license="MIT", server=False, models_dir=offline.models)
    assert not (offline.models / "yolo").exists()


def test_c1_api_subclass_of_ultralytics_class_refused(offline, fake_ultralytics):
    from visionserve.convert import export

    class Mine(fake_ultralytics.DetectionModel):  # defined here, but IS an Ultralytics model
        pass
    with pytest.raises(ConvertError, match="AGPL"):
        export(Mine(), name="yolo2", task="classification", input=32, license="MIT", server=False,
               models_dir=offline.models)


def test_c1_script_build_model_ultralytics_refused(offline, fake_ultralytics, tmp_path, capsys):
    s = _build_script(tmp_path, "from ultralytics.nn.tasks import DetectionModel\n"
                                "def build_model():\n    return DetectionModel()\n")
    assert cli.main(_argv("yolo3", offline.models, src=s)) == 2
    assert "AGPL" in capsys.readouterr().err
    assert not (offline.models / "yolo3").exists()


def test_c1_permissive_in_memory_module_still_passes(offline):
    from visionserve.convert import export
    r = export(TinyCls(), name="fine", task="classification", input=32, license="MIT", server=False,
               models_dir=offline.models)
    assert r.ok and (offline.models / "fine" / "manifest.yaml").is_file()


def _tf_args(src, fmt):
    return SimpleNamespace(input="32x32", mean="0.485,0.456,0.406", std="0.229,0.224,0.225", license="MIT",
                           source=str(src), format=fmt, outputs=None, name="x", task="classification",
                           signature="serving_default", opset=17, tolerance=1e-3)


ULTRA_META = "author: Ultralytics\nlicense: AGPL-3.0 License (https://ultralytics.com/license)\nstride: 32\n"


def test_c1_tf_savedmodel_with_ultralytics_metadata_refused(tmp_path):
    from visionserve.convert.families import tf as tff
    d = tmp_path / "yolov8n_saved_model"
    d.mkdir()
    (d / "saved_model.pb").write_bytes(b"\x08\x01" * 100)
    (d / "metadata.yaml").write_text(ULTRA_META)
    with pytest.raises(ConvertError, match="AGPL"):
        tff.convert(_tf_args(d, "tensorflow"), tmp_path)


def test_c1_tflite_with_appended_metadata_zip_refused(tmp_path):
    from visionserve.convert.families import tf as tff
    p = tmp_path / "yolov8n_float32.tflite"
    p.write_bytes(b"TFL3" + bytes(range(256)) * 64)
    with zipfile.ZipFile(p, "a", zipfile.ZIP_DEFLATED) as z:  # what Ultralytics' exporter appends
        z.writestr("metadata.json", json.dumps({"author": "Ultralytics", "license": "AGPL-3.0 License"}))
    with pytest.raises(ConvertError, match="AGPL"):
        tff.convert(_tf_args(p, "tflite"), tmp_path)


def test_c1_keras_zip_config_with_ultralytics_module_refused(tmp_path):
    p = tmp_path / "m.keras"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("config.json", json.dumps({"class_name": "Functional", "config": {"layers": [
            {"module": "ultralytics.nn.modules", "class_name": "TFConv"}]}}))
        z.writestr("model.weights.h5", b"\x89HDF\r\n" + b"\0" * 64)
    with pytest.raises(ConvertError, match="AGPL"):
        license_scan(source=p)


# --------------------------------------------------------------------------------------------
# C2: TorchScript scan reads the whole archive; a path containing "ultralytics" is not a marker
# --------------------------------------------------------------------------------------------

def _repack_with_padding(src: Path, dst: Path, pad_bytes: int) -> None:
    """Same archive, with a big stored entry FIRST so the code/ entries sit past `pad_bytes`."""
    with zipfile.ZipFile(src) as zin:
        top = zin.namelist()[0].split("/", 1)[0]
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as zout:
            zout.writestr(f"{top}/data/zz_padding", np.random.default_rng(0).bytes(pad_bytes))
            for info in zin.infolist():
                zout.writestr(info, zin.read(info))


def test_c2_torchscript_code_past_8mb_refused(tmp_path, fake_ultralytics):
    raw = tmp_path / "raw.pt"
    torch.jit.script(fake_ultralytics.DetectionModel().eval()).save(str(raw))
    with zipfile.ZipFile(raw) as z:
        assert any("ultralytics" in n for n in z.namelist())  # code/__torch__/ultralytics/nn/tasks.py
    big = tmp_path / "model.pt"
    _repack_with_padding(raw, big, 9 << 20)
    with pytest.raises(ConvertError, match="AGPL"):
        license_scan(source=big)


def test_c2_torchscript_compressed_entries_refused(tmp_path, fake_ultralytics):
    raw = tmp_path / "raw.pt"
    torch.jit.script(fake_ultralytics.DetectionModel().eval()).save(str(raw))
    z2 = tmp_path / "deflated.pt"
    with zipfile.ZipFile(raw) as zin, zipfile.ZipFile(z2, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            zout.writestr(info.filename, zin.read(info))
    with pytest.raises(ConvertError, match="AGPL"):
        license_scan(source=z2)


def test_c2_torchscript_loaded_module_qualified_name_refused(offline, fake_ultralytics, tmp_path, capsys):
    p = tmp_path / "m.pt"
    torch.jit.script(fake_ultralytics.DetectionModel().eval()).save(str(p))
    rc = cli.main(_argv("ts", offline.models, src=p, fmt="torchscript"))
    assert rc == 2 and "AGPL" in capsys.readouterr().err


def test_c2_file_named_ultralytics_is_not_a_marker(offline, tmp_path):
    p = tmp_path / "ultralytics_baseline.pt"  # archive entries are "ultralytics_baseline/..."
    torch.jit.script(TinyCls().eval()).save(str(p))
    license_scan(source=p)
    assert cli.main(_argv("baseline", offline.models, "--dry-run", src=p, fmt="torchscript")) == 0


def test_c2_checkpoint_with_a_path_string_is_not_a_marker(tmp_path):
    p = tmp_path / "ck.pth"
    torch.save({"model": TinyCls().state_dict(),
                "args": {"dataset_dir": "/data/ultralytics/coco", "output_dir": "/runs/ultralytics.v2/x"}}, p)
    license_scan(source=p)
    legacy = tmp_path / "legacy.pth"
    torch.save({"args": {"output_dir": "/home/me/ultralytics_runs"}}, legacy, _use_new_zipfile_serialization=False)
    license_scan(source=legacy)


def test_c2_dotted_class_name_string_and_legacy_pickle_still_refused(tmp_path, fake_ultralytics):
    p = tmp_path / "w.pth"
    torch.save({"model_class": "ultralytics.nn.tasks.DetectionModel"}, p)
    with pytest.raises(ConvertError, match="AGPL"):
        license_scan(source=p)
    legacy = tmp_path / "legacy.pt"  # a whole pickled Ultralytics module, old (non-zip) format
    torch.save(fake_ultralytics.DetectionModel(), legacy, _use_new_zipfile_serialization=False)
    with pytest.raises(ConvertError, match="AGPL"):
        license_scan(source=legacy)


# --------------------------------------------------------------------------------------------
# C3: model card with a BOM, LICENSE file
# --------------------------------------------------------------------------------------------

def test_c3_card_license_with_bom_and_crlf():
    from visionserve.convert.families import hf
    d = Path(__import__("tempfile").mkdtemp())
    (d / "README.md").write_bytes(b"\xef\xbb\xbf---\r\nlicense: agpl-3.0\r\n---\r\n# model\r\n")
    assert hf.card_license(d / "README.md") == "agpl-3.0"


def test_c3_hf_local_dir_with_bom_card_cannot_be_laundered(tmp_path):
    from visionserve.convert.families import hf
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "vit", "architectures": ["ViTForImageClassification"]}))
    (d / "README.md").write_bytes("﻿---\nlicense: agpl-3.0\n---\n".encode("utf-8"))
    args = SimpleNamespace(source=str(d), license="MIT", revision=None)
    with pytest.raises(ConvertError, match="not allowed"):
        hf.convert(args, tmp_path / "w")


def test_c3_hf_local_dir_with_gpl_license_file_refused(tmp_path):
    from visionserve.convert.families import hf
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "vit", "architectures": ["ViTForImageClassification"]}))
    (d / "LICENSE").write_text("                    GNU AFFERO GENERAL PUBLIC LICENSE\n                       Version 3\n")
    args = SimpleNamespace(source=str(d), license="MIT", revision=None)
    with pytest.raises(ConvertError, match="license text"):
        hf.convert(args, tmp_path / "w")


def test_c3_permissive_license_file_and_prose_card_pass(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "LICENSE").write_text("Apache License\nVersion 2.0, January 2004\n")
    (d / "README.md").write_text("---\nlicense: apache-2.0\n---\nUnlike YOLO (AGPL-3.0), this model is Apache.\n")
    license_scan(source=d)


# --------------------------------------------------------------------------------------------
# C4-C8: install transaction, server reuse, name validation, exception safety
# --------------------------------------------------------------------------------------------

class _FailingServer:
    """A server whose predict disagrees with the reference: B2 FAILs."""

    def __init__(self, models_dir):
        self.models = Path(models_dir)

    def load(self, name):
        return {"model": name, "state": "loaded"}

    def preprocess(self, name, src, prompt=None):
        raise RuntimeError("no preprocess")

    def predict(self, name, src, prompt=None, max_grasps_per_object=None):
        from visionserve.types import Classification, Result
        return Result(task="classification", model=name, classifications=[Classification("zzz", 1.0)])


def _images(d: Path, n=2):
    from PIL import Image
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.fromarray(np.full((20, 24, 3), 40 * i, np.uint8)).save(d / f"i{i}.png")
    return d


def _script_cls(tmp_path):
    return _build_script(tmp_path, "import sys\nsys.path.insert(0, %r)\nfrom test_review_fixes import TinyCls\n"
                                   "def build_model():\n    return TinyCls()\n" % str(Path(__file__).parent))


def _old_model(models, name):
    d = models / name
    d.mkdir(parents=True)
    (d / "manifest.yaml").write_text(f"name: {name}\n# OLD WORKING VERSION\n")
    (d / "model.onnx").write_bytes(b"old weights")
    return d


def test_c5_failed_force_replacement_restores_previous_model(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "install", _fake_install)
    monkeypatch.setattr(serverctl, "_health", lambda *a, **k: False)
    monkeypatch.setattr(serverctl, "acquire", lambda names, models_dir, **kw: serverctl.ServerHandle(
        "http://fake", temporary=True, client=_FailingServer(models_dir)))
    models = tmp_path / "models"
    _old_model(models, "m")
    labels = tmp_path / "labels.txt"
    labels.write_text("a\nb\nc\n")
    argv = ["pytorch", str(_script_cls(tmp_path)), "--name", "m", "--license", "MIT", "--task", "classification",
            "--input", "32x32", "--labels", str(labels), "--models", str(models), "--force",
            "--images", str(_images(tmp_path / "imgs"))]
    assert cli.main(argv) == 1
    assert (models / "m" / "manifest.yaml").read_text() == "name: m\n# OLD WORKING VERSION\n"
    assert (models / "m" / "model.onnx").read_bytes() == b"old weights"
    assert not (models / ".convert-backup").exists()
    rep = json.loads((models / ".convert-failed" / "m" / "convert-report.json").read_text())
    assert rep["uninstalled"] and rep["restored"] == ["m"]


def test_c5_successful_force_replacement_drops_backup(offline, tmp_path):
    _old_model(offline.models, "m2")
    argv = _argv("m2", offline.models, "--force", src=_script_cls(tmp_path))
    assert cli.main(argv) == 0
    assert "OLD" not in (offline.models / "m2" / "manifest.yaml").read_text()
    assert not (offline.models / ".convert-backup").exists()


def test_c6_name_outside_registry_refused_before_any_write(offline, tmp_path, capsys):
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("precious")
    for bad in (str(victim), "../victim", "a/b", ".hidden", "x\nname: evil", ""):
        assert cli.main(_argv(bad, offline.models, "--dry-run", src=_script_cls(tmp_path))) == 2
        assert "not a valid model name" in capsys.readouterr().err
    assert (victim / "keep.txt").read_text() == "precious"


def test_c6_bundle_write_refuses_bad_name(tmp_path):
    from visionserve.convert.common import Bundle
    b = Bundle(name=str(tmp_path / "abs"), task="classification", architecture="efficientnet", license="MIT",
               width=8, height=8, onnx={})
    with pytest.raises(ConvertError, match="valid model name"):
        b.write(tmp_path / "staging")


def test_c7_images_is_a_file_refused_before_install(offline, tmp_path, capsys):
    f = tmp_path / "one.png"
    f.write_bytes(b"x")
    assert cli.main(_argv("m3", offline.models, "--images", str(f), src=_script_cls(tmp_path))) == 2
    assert "DIRECTORY" in capsys.readouterr().err
    assert not (offline.models / "m3").exists()


def test_c7_corrupt_image_after_install_uninstalls_with_report(offline, tmp_path, monkeypatch):
    monkeypatch.setattr(serverctl, "acquire", lambda names, models_dir, **kw: serverctl.ServerHandle(
        "http://fake", temporary=True, client=_FailingServer(models_dir)))
    imgs = tmp_path / "bad"
    imgs.mkdir()
    (imgs / "broken.jpg").write_bytes(b"not a jpeg")
    argv = [a for a in _argv("m4", offline.models, "--images", str(imgs), src=_script_cls(tmp_path))
            if a != "--no-server"]
    assert cli.main(argv) == 1
    assert not (offline.models / "m4").exists()
    rep = json.loads((offline.models / ".convert-failed" / "m4" / "convert-report.json").read_text())
    assert any(t["tier"] == "verify" and "aborted" in t["summary"] for t in rep["tiers"])


def test_c8_partial_multi_bundle_install_rolled_back(monkeypatch, tmp_path):
    from visionserve.convert.common import Bundle
    models = tmp_path / "models"
    _old_model(models, "pair-text")  # exists, no --force: the 2nd pull is refused

    def two_bundles(args, work):
        p = work / "m.onnx"
        torch.onnx.export(TinyCls().eval(), (torch.zeros(1, 3, 32, 32),), str(p), dynamo=False, opset_version=17)
        return [Bundle(name=args.name, task="embed", architecture="clip", license="MIT", width=32, height=32,
                       onnx={"model": str(p)}),
                Bundle(name=f"{args.name}-text", task="embed", architecture="clip", license="MIT", width=32,
                       height=32, onnx={"model": str(p)})]
    fam = SimpleNamespace(convert=two_bundles, add_arguments=lambda p, f: None, help_for=lambda f: f)
    monkeypatch.setattr(cli, "_family", lambda modname: fam)
    monkeypatch.setattr(cli, "install", _fake_install)
    monkeypatch.setattr(serverctl, "_health", lambda *a, **k: False)
    assert cli.main(["hf", "x", "--name", "pair", "--models", str(models), "--no-server"]) == 2
    assert not (models / "pair").exists()                       # the first bundle was rolled back
    assert "OLD WORKING" in (models / "pair-text" / "manifest.yaml").read_text()  # not ours: untouched


def test_c4_running_server_that_had_the_model_is_not_reused(tmp_path):
    started, logs = [], []

    class P:
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    def popen(cmd, **kw):
        started.append(cmd)
        return P()
    bin_ = tmp_path / "visionserve"
    bin_.write_text("#!/bin/sh\n")
    up = {"http://localhost:11435"}
    health = lambda u, *a, **k: u in up or bool(started)  # noqa: E731
    h = serverctl.acquire(["m"], tmp_path, health=health, listed=lambda u: ["m"], popen=popen,
                          env={"VISIONSERVE_BIN": str(bin_)}, log=logs.append, workdir=tmp_path, fresh=True)
    assert h is not None and h.temporary and started, logs
    h.close()
    # and the check that decides it: the server listed the name BEFORE the install
    assert serverctl.registered_on_server(["m", "n"], health=health, listed=lambda u: ["m"],
                                          log=logs.append) == ["m"]
    assert serverctl.registered_on_server(["m"], no_server=True) == []


def test_c4_cli_asks_for_a_fresh_server_when_the_name_was_registered(offline, tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(serverctl, "_health", lambda *a, **k: True)
    monkeypatch.setattr(serverctl, "_listed", lambda *a, **k: ["m5"])
    monkeypatch.setattr(serverctl, "acquire", lambda names, models_dir, **kw: seen.update(kw) or None)
    _old_model(offline.models, "m5")
    argv = [a for a in _argv("m5", offline.models, "--force", src=_script_cls(tmp_path)) if a != "--no-server"]
    assert cli.main(argv) == 0
    assert seen.get("fresh") is True


def test_c8_ctrl_c_while_temporary_server_starts_kills_it(tmp_path):
    killed = []

    class P:
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            killed.append("terminate")

        def wait(self, timeout=None):
            return 0

    def sleep(_):
        raise KeyboardInterrupt
    bin_ = tmp_path / "visionserve"
    bin_.write_text("")
    with pytest.raises(KeyboardInterrupt):
        serverctl.start_temporary(tmp_path, workdir=tmp_path, health=lambda u: False, popen=lambda *a, **k: P(),
                                  env={"VISIONSERVE_BIN": str(bin_)}, sleep=sleep, log=lambda m: None)
    assert killed == ["terminate"]


def test_c8_parity_accepts_nan_at_the_same_positions(tmp_path):
    from onnx import TensorProto, helper

    from visionserve.convert.common import parity
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    g = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g", [x], [y])
    p = tmp_path / "id.onnx"
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), str(p))
    v = np.array([[1.0, np.nan, 3.0]], np.float32)
    assert parity(p, {"x": v}, [v.astype(np.float64)], tol=1e-3) == 0.0
    with pytest.raises(ConvertError, match="NaN/inf"):  # a NaN where the original has +inf still fails
        parity(p, {"x": v}, [np.array([[1.0, np.inf, 3.0]])], tol=1e-3)


def test_c8_coco_annotation_id_zero_counts(tmp_path):
    from PIL import Image
    pytest.importorskip("pycocotools")
    Image.fromarray(np.zeros((48, 64, 3), np.uint8)).save(tmp_path / "a.png")
    d = {"images": [{"id": 0, "file_name": "a.png", "width": 64, "height": 48}],
         "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "bbox": [5, 5, 20, 20]}],
         "categories": [{"id": 1, "name": "cup"}]}
    f = tmp_path / "_annotations.coco.json"
    f.write_text(json.dumps(d))
    e = ev.load_eval(f)
    m, m50 = ev.coco_map(e.coco, [{"image_id": 0, "category_id": 1, "bbox": [5, 5, 20, 20], "score": 0.9}],
                         [0], [1])
    assert m == pytest.approx(100.0) and m50 == pytest.approx(100.0)


def test_c8_api_threshold_list_is_not_dropped(offline):
    from visionserve.convert import export
    with pytest.raises(ConvertError, match="unknown threshold"):
        export(TinyCls(), name="t1", task="classification", input=32, license="MIT", server=False,
               models_dir=offline.models, threshold=["nope=1"])
    r = export(TinyCls(), name="t2", task="classification", input=32, license="MIT", server=False,
               models_dir=offline.models, threshold=["b1_mean_fail=11"])
    assert r.thresholds["b1_mean_fail"] == 11.0


def test_c8_verify_zero_refused(offline, tmp_path, capsys):
    assert cli.main(_argv("v0", offline.models, "--verify", "0", src=_script_cls(tmp_path))) == 2
    assert "--verify 0" in capsys.readouterr().err


def test_c8_external_data_location_escaping_staging_refused(tmp_path):
    from onnx import TensorProto, helper

    from visionserve.convert.common import _copy_onnx
    w = helper.make_tensor("w", TensorProto.FLOAT, [3], [1.0, 2.0, 3.0])
    w.data_location = TensorProto.EXTERNAL
    w.ClearField("float_data")
    e = w.external_data.add()
    e.key, e.value = "location", "../../outside.bin"
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])
    g = helper.make_graph([helper.make_node("Add", ["x", "w"], ["y"])], "g", [x], [y], initializer=[w])
    src = tmp_path / "src"
    src.mkdir()
    p = src / "m.onnx"
    onnx.save(helper.make_model(g), str(p))
    (tmp_path / "outside.bin").write_bytes(b"\0" * 12)
    dst = tmp_path / "a" / "b" / "staging"
    dst.mkdir(parents=True)
    with pytest.raises(ConvertError, match="escapes"):
        _copy_onnx(p, dst)
    assert not (tmp_path / "a" / "outside.bin").exists()


def test_c8_abbreviated_flags_refused(capsys):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["pytorch", "x.py", "--name", "n", "--task", "classification", "--input",
                                       "8x8", "--imag", "/somewhere"])


def test_c8_tier_c_open_vocab_reference_labels_lowercased_and_merged(tmp_path):
    from PIL import Image

    from visionserve.convert.reference import Prediction
    from visionserve.types import Detection, Result
    pytest.importorskip("pycocotools")
    Image.fromarray(np.zeros((48, 64, 3), np.uint8)).save(tmp_path / "a.png")
    d = {"images": [{"id": 1, "file_name": "a.png", "width": 64, "height": 48}],
         "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 2, 20, 20]},
                         {"id": 2, "image_id": 1, "category_id": 2, "bbox": [30, 10, 20, 20]}],
         "categories": [{"id": 1, "name": "Water_Bottle"}, {"id": 2, "name": "cup"}]}
    f = tmp_path / "_annotations.coco.json"
    f.write_text(json.dumps(d))
    e = ev.load_eval(f)

    class Ref:
        kind = "official"

        def predict(self, pil, prompt=None):  # what transformers' GroundingDINO post-process returns
            return Prediction(detections=[{"cls": "water _ bottle", "conf": 0.9, "bbox": [2, 2, 20, 20]},
                                          {"cls": "cup water", "conf": 0.8, "bbox": [30, 10, 20, 20]}])

    class Srv:
        def predict(self, name, src, prompt=None, max_grasps_per_object=None):
            return Result(task="open_vocab", model=name, detections=[
                Detection([2, 2, 20, 20], "Water_Bottle", 0.9), Detection([30, 10, 20, 20], "cup", 0.8)])

    b = SimpleNamespace(name="gd", task="open_vocab", labels=None, postprocess={"conf_threshold": 0.3})
    prompt = "Water_Bottle. cup."
    t = ev.tier_c(SimpleNamespace(b2=Ref()), Srv(), b, e, 1.0, prompt)
    assert t.metrics["n_dets_ref"] == 2, t.metrics
    assert t.metrics["map_ref"] == pytest.approx(100.0) and t.status == rp.PASS
    assert ev.open_vocab_label("cup water bottle", {"cup": 1, "water bottle": 2}) == "water bottle"
    assert ev.open_vocab_label("kite", {"cup": 1}) is None
