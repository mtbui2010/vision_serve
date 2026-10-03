"""Every converter format runs THE content license gate (common.license_scan), before any work.

The gate on the declared license (resolve_license) cannot see an Ultralytics model whose user typed
--license MIT; license_scan reads the checkpoint itself. A family that forgot to call it would
silently let AGPL weights through, so this enumerates every format the CLI offers (cli.FORMATS — a
new format without a case here fails) and checks, with the scan replaced by a recorder:

  * the scan is invoked, on the checkpoint the user gave (or the in-memory module);
  * it runs BEFORE the checkpoint is loaded: the recorder raises, and the dummy checkpoints below
    are not loadable, so reaching any loader would raise a different error instead.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # clients/python

from visionserve.convert import cli, common  # noqa: E402


class _Scanned(Exception):
    """Raised by the recorder: the conversion stops at the gate, before any heavy work."""


@pytest.fixture
def recorder(monkeypatch):
    """Replace license_scan everywhere a family can reach it: common (looked up at call time by
    tf.py's lazy import) and every family module that imported the name."""
    calls = []
    stop_on = {"kind": "any"}

    def fake(source=None, module=None):
        calls.append({"source": None if source is None else Path(source), "module": module})
        if stop_on["kind"] == "any" or (stop_on["kind"] == "module" and module is not None):
            raise _Scanned()

    # Import the families BEFORE patching common: a family imported (or re-imported, if another test
    # dropped it from sys.modules) while common is patched would bind the fake for good.
    families = []
    for modname in sorted(set(cli.FORMATS.values())):
        try:
            families.append(cli._family(modname))
        except ImportError:
            continue
    monkeypatch.setattr(common, "license_scan", fake)
    for mod in families:
        if hasattr(mod, "license_scan"):
            monkeypatch.setattr(mod, "license_scan", fake)
    return calls, stop_on


def _args(fmt, source, *extra):
    args = cli.build_parser().parse_args([fmt, str(source), "--name", "t", *extra])
    if getattr(args, "_unavailable", None):
        pytest.skip(args._unavailable)
    return args


GENERIC = ["--task", "classification", "--input", "32x32", "--license", "MIT"]


def _case_rfdetr(tmp):
    pytest.importorskip("torch")
    ck = tmp / "checkpoint_best_total.pth"
    ck.write_bytes(b"not a checkpoint")
    return _args("rfdetr", tmp, "--license", "Apache-2.0"), ck


def _case_hf(tmp):
    src = tmp / "hfmodel"
    src.mkdir()
    (src / "config.json").write_text("{}")
    return _args("hf", src, "--license", "Apache-2.0"), src


def _case_torchscript(tmp):
    pytest.importorskip("torch")
    src = tmp / "model.pt"
    src.write_bytes(b"not torchscript")
    return _args("torchscript", src, *GENERIC), src


def _case_pytorch(tmp):
    pytest.importorskip("torch")
    weights = tmp / "weights.pth"
    weights.write_bytes(b"not a state dict")
    return _args("pytorch", weights, "--script", str(_SCRIPT(tmp)), *GENERIC), weights


def _case_tf(fmt, name):
    def case(tmp):
        src = tmp / name
        if name.endswith("/"):
            src.mkdir()
            (src / "saved_model.pb").write_bytes(b"not a graph")
        else:
            src.write_bytes(b"not a model")
        return _args(fmt, src, *GENERIC), src
    return case


def _SCRIPT(tmp):
    p = tmp / "build.py"
    p.write_text(textwrap.dedent("""
        import torch
        def build_model():
            return torch.nn.Linear(2, 2)
    """))
    return p


CASES = {
    "rfdetr": _case_rfdetr,
    "hf": _case_hf,
    "torchscript": _case_torchscript,
    "pytorch": _case_pytorch,
    "tensorflow": _case_tf("tensorflow", "saved/"),
    "keras": _case_tf("keras", "model.keras"),
    "tflite": _case_tf("tflite", "model.tflite"),
}


def test_every_format_has_a_case():
    assert set(CASES) == set(cli.FORMATS), "add a license_scan case for the new format"


@pytest.mark.parametrize("fmt", sorted(CASES))
def test_format_scans_the_checkpoint_before_loading_it(fmt, tmp_path, recorder):
    calls, _ = recorder
    args, expected = CASES[fmt](tmp_path)
    fam = cli._family(cli.FORMATS[fmt])
    with pytest.raises(_Scanned):
        fam.convert(args, tmp_path / "work")
    assert calls, f"{fmt}: license_scan was never called"
    assert calls[0]["source"] == expected, f"{fmt}: scanned {calls[0]['source']}, not the checkpoint {expected}"


def test_pytorch_also_scans_the_built_module(tmp_path, recorder):
    """--script code can build an Ultralytics class without any file saying so: the module the
    script returns is scanned too (after the weights file, before the weights are loaded)."""
    calls, stop_on = recorder
    stop_on["kind"] = "module"
    args, weights = _case_pytorch(tmp_path)
    with pytest.raises(_Scanned):
        cli._family(cli.FORMATS["pytorch"]).convert(args, tmp_path / "work")
    assert [c["source"] for c in calls] == [weights, None]
    assert calls[1]["module"] is not None and type(calls[1]["module"]).__name__ == "Linear"


def test_python_api_module_is_scanned(tmp_path, recorder):
    """visionserve.convert.export(<nn.Module>): no file at all, the classes are the evidence."""
    torch = pytest.importorskip("torch")
    calls, _ = recorder
    args = _args("pytorch", "in-memory-module", *GENERIC)
    args.module = torch.nn.Linear(2, 2)
    with pytest.raises(_Scanned):
        cli._family(cli.FORMATS["pytorch"]).convert(args, tmp_path / "work")
    assert calls and calls[0]["module"] is args.module and calls[0]["source"] is None
