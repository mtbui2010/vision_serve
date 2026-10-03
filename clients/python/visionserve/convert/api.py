"""Python API: `visionserve.convert.export(...)` — the CLI's convert + install + verify, from code.

    from visionserve.convert import export

    # an in-memory PyTorch model, with YOUR training transform as the tier-B1 reference
    report = export(model, name="my-cls", task="classification", input="224x224",
                    labels=["cat", "dog"], license="Apache-2.0",
                    preprocess=train_transform,          # PIL -> CHW tensor/ndarray
                    images="val/images", bench=True)
    print(report)            # the table
    assert report.ok         # False when a check FAILED (the model was then uninstalled)

    # a checkpoint path or hub id, format inferred where possible
    export("runs/checkpoint_best_total.pth", name="tabletop", images="data/images", eval="data/coco.json")
    export("PekingU/rtdetr_r50vd", name="rtdetr", eval="coco/annotations/instances_val2017.json")

Conversion refusals (license, parity, I/O contract, `visionserve pull`) raise ConvertError. A
verification FAIL does not raise: it returns the Report with .ok == False.
"""
from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

from .common import ConvertError
from .constants import HUB_ID_RE


def default_models_dir() -> Path:
    """Same resolution as the Go CLI: $VISIONSERVE_MODELS, else ~/.visionserve/models."""
    env = os.environ.get("VISIONSERVE_MODELS")
    return Path(env) if env else Path.home() / ".visionserve" / "models"


def infer_format(source) -> str:
    """Best-effort format of a checkpoint path or hub id; raises when ambiguous."""
    p = Path(str(source))
    if p.is_dir():
        if (p / "config.json").is_file():
            return "hf"
        if (p / "saved_model.pb").is_file():
            return "tensorflow"
        if any(p.glob("checkpoint*.pth")):
            return "rfdetr"
        raise ConvertError(f"cannot infer the format of directory {p}; pass format=")
    if not p.exists():
        if HUB_ID_RE.match(str(source)):
            return "hf"
        raise ConvertError(f"{source}: no such file or directory (and not a hub id org/name)")
    suf = p.suffix.lower()
    if suf in (".keras", ".h5"):
        return "keras"
    if suf == ".tflite":
        return "tflite"
    if suf in (".pt", ".pth"):
        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as z:
                names = z.namelist()
            if any(n.endswith("/constants.pkl") or "/code/" in n for n in names):
                return "torchscript"
        try:
            from .families.rfdetr import _load_checkpoint, _split_checkpoint
            _split_checkpoint(_load_checkpoint(p))
            return "rfdetr"
        except Exception:  # noqa: BLE001
            pass
        raise ConvertError(f"{p.name}: not TorchScript and not an RF-DETR checkpoint; for plain PyTorch weights "
                           "pass format='pytorch', script='build.py' (or pass the nn.Module itself)")
    raise ConvertError(f"cannot infer the format of {p.name}; pass format= (one of rfdetr, hf, torchscript, "
                       "pytorch, tensorflow, keras, tflite)")


def _wxh(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, int):
        return f"{v}x{v}"
    if isinstance(v, (tuple, list)):
        return f"{int(v[0])}x{int(v[1])}"
    return str(v)


def export(model_or_path, *, format: Optional[str] = None, name: str, task: Optional[str] = None,
           input: Union[str, int, Sequence[int], None] = None, labels: Union[str, Sequence[str], None] = None,
           license: Optional[str] = None, images=None, eval=None, bench: bool = False,
           preprocess: Optional[Callable] = None, server: Union[str, bool, None] = None,
           models_dir=None, **kw):
    """Convert `model_or_path` to ONNX, install it as `name`, verify it, and return the Report.

    model_or_path  a torch.nn.Module (exported through the `pytorch` path: task= and input= are
                   required), or a checkpoint path / HF hub id (format= or inferred from it).
    task, input    generic formats only: classification|detection|depth|embed and "WxH" / 224 / (w, h).
    labels         a list of class names or a labels.txt path.
    images         dir of real images for tier B.  eval: COCO json / ImageFolder dir for tier C.
    preprocess     YOUR training transform, PIL -> CHW tensor/ndarray: the tier-B1 reference (the
                   strongest check: it compares what the server feeds with what the model learned on).
    server         a server URL, or False to skip server-side checks; default: $VISIONSERVE_HOST /
                   localhost:11435, else a temporary local server.
    models_dir     the registry; default $VISIONSERVE_MODELS or ~/.visionserve/models.
    **kw           any other CLI flag, underscores for dashes: variant="small", mean=[...],
                   std=[...], letterbox=True, keep_on_fail=True, max_map_drop=1.0, eval_max=100,
                   bench_iters=20, reference_script="ref.py", threshold={"b1_mean_fail": 10}, ...
    """
    from . import cli
    module = None
    try:
        import torch
        if isinstance(model_or_path, torch.nn.Module):
            module = model_or_path
    except ImportError:
        pass
    tmp = tempfile.TemporaryDirectory(prefix="vsconvert-api-")
    try:
        if module is not None:
            fmt = "pytorch"
            if not task or input is None:
                raise ConvertError("exporting an in-memory nn.Module needs task= and input=")
            source = "in-memory-module"
        else:
            source = str(model_or_path)
            fmt = format or infer_format(source)
        argv = [fmt, source, "--name", name, "--models", str(models_dir or default_models_dir())]
        if license:
            argv += ["--license", license]
        if labels is not None:
            if isinstance(labels, (list, tuple)):
                lp = Path(tmp.name) / "labels.txt"
                lp.write_text("\n".join(str(x) for x in labels) + "\n")
                argv += ["--labels", str(lp)]
            else:
                argv += ["--labels", str(labels)]
        if task:
            argv += ["--task", task]
        if input is not None:
            argv += ["--input", _wxh(input)]
        if images:
            argv += ["--images", str(images)]
        if eval:
            argv += ["--eval", str(eval)]
        if bench:
            argv += ["--bench"]
        if server is False:
            argv += ["--no-server"]
        elif server:
            argv += ["--server", str(server)]
        given = [kw.pop(k) for k in ("threshold", "thresholds") if k in kw]
        for thresholds in given:
            if thresholds is None:
                continue
            if isinstance(thresholds, dict):
                items = [f"{k}={v}" for k, v in thresholds.items()]
            elif isinstance(thresholds, str):
                items = [thresholds]
            elif isinstance(thresholds, (list, tuple)):
                items = [str(x) for x in thresholds]
            else:
                raise ConvertError(f"threshold= takes a dict {{key: value}} or a list of 'key=value' strings, "
                                   f"got {type(thresholds).__name__}")
            for it in items:
                argv += ["--threshold", it]
        for k, v in kw.items():
            flag = "--" + k.replace("_", "-")
            if v is True:
                argv.append(flag)
            elif v is False or v is None:
                continue
            elif isinstance(v, (list, tuple)):
                argv += [flag, ",".join(str(x) for x in v)]
            else:
                argv += [flag, str(v)]
        parser = cli.build_parser()
        try:
            args = parser.parse_args(argv)
        except SystemExit as e:  # argparse error: turn it into an exception, not an exit
            raise ConvertError(f"invalid arguments {argv[2:]}: see the message above") from e
        if getattr(args, "_unavailable", None):
            raise ConvertError(f"this Python environment cannot run {args._unavailable}")
        if module is not None:
            args.module = module
            if getattr(args, "script", None) is None:
                args.script = None
        args.preprocess_fn = preprocess
        return cli.run(args)
    finally:
        tmp.cleanup()
