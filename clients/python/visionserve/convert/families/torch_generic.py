"""TorchScript (.pt/.pth from torch.jit.save) and plain PyTorch (build script + weights) -> ONNX.

Neither format says what the model IS, so the task / input size / normalisation come from the generic
flags (cli.add_generic_arguments) and select a Go architecture whose I/O contract is checked.

  torchscript SOURCE                      torch.jit.load(SOURCE)
  pytorch SOURCE --script build.py        SOURCE = the weights file; build.py defines build_model()
                                          or build_model(weights_path). With no argument, the weights
                                          are loaded here: torch.load(weights_only=True), a `state_dict`/
                                          `model`/`model_state_dict`/`ema` sub-dict is unwrapped,
                                          a `module.` prefix is stripped, load_state_dict(strict=True).
                                          With one argument, build_model loads them itself.
                                          (--weights W may be given instead of / in addition to SOURCE.)

Exporter: the LEGACY TorchScript-based `torch.onnx.export(..., dynamo=False)`. It is the only one of
the two that takes a ScriptModule, it emits opset 17 directly (dynamo targets opset 18+ and down-
converts), and it is what rfdetr itself uses on torch 2.10. For the `pytorch` format, if the legacy
exporter fails the dynamo exporter is tried once (it needs `onnxscript`). Either way the graph has
static shapes [1,3,H,W] and is checked against the original by the parity run before anything is
installed.

Outputs: whatever the module returns is flattened in order (tensor -> 1 output; tuple/list -> in order;
dict -> values in insertion order). For --task detection the Go rt-detr decoder needs EXACTLY two:
boxes [1,Q,4] (cxcywh, normalised) and logits [1,Q,C] (sigmoid), in either order.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
from pathlib import Path

from ..cli import add_generic_arguments, generic_bundle, parse_floats, parse_wxh
from ..common import (ConvertError, license_scan, log, onnx_io, parity, resolve_license, sample_image,
                      to_nchw)

_SUBDICT_KEYS = ("state_dict", "model_state_dict", "model", "ema", "model_ema", "net", "weights")


def help_for(fmt: str) -> str:
    if fmt == "torchscript":
        return "TorchScript module (torch.jit.save) -> classification / detection / depth / embed"
    return "PyTorch weights + a build script defining build_model() -> classification / detection / depth / embed"


def add_arguments(p: argparse.ArgumentParser, fmt: str) -> None:
    add_generic_arguments(p)
    if fmt == "pytorch":
        p.add_argument("--script", help="Python file defining build_model() -> torch.nn.Module, or "
                                        "build_model(weights_path) if it loads the weights itself. "
                                        "(If omitted, SOURCE may be the .py file and --weights the checkpoint.)")
        p.add_argument("--weights", help="checkpoint to load (default: SOURCE)")
        p.add_argument("--unsafe-pickle", action="store_true",
                       help="allow torch.load(weights_only=False) when the checkpoint pickles whole objects "
                            "(e.g. torch.save(model)). This RUNS CODE from the checkpoint; only for files you trust")


# --------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------

def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location(f"vs_user_build_{path.stem}", str(path))
    if spec is None or spec.loader is None:
        raise ConvertError(f"--script {path}: not a loadable Python file")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001 — user code
        raise ConvertError(f"--script {path.name} failed to import: {type(e).__name__}: {e}")
    fn = getattr(mod, "build_model", None)
    if not callable(fn):
        raise ConvertError(f"--script {path.name} must define build_model() -> torch.nn.Module")
    return fn


def _torch_load(path: Path, unsafe: bool):
    import torch
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        if not unsafe:
            raise ConvertError(f"{path.name} is not a plain-weights checkpoint ({str(e).splitlines()[0]}). "
                               "Re-save it as a state_dict (torch.save(model.state_dict(), ...)), or pass "
                               "--unsafe-pickle if you trust the file (it runs code from the checkpoint).")
        log(f"torch: warning: loading {path.name} with weights_only=False (--unsafe-pickle)")
        return torch.load(str(path), map_location="cpu", weights_only=False)


def _state_dict_of(obj):
    import torch
    if isinstance(obj, torch.nn.Module):
        return obj.state_dict()
    if not isinstance(obj, dict):
        raise ConvertError(f"checkpoint holds a {type(obj).__name__}, not a state_dict")
    if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
        sd = obj
    else:
        for k in _SUBDICT_KEYS:
            if k in obj:
                return _state_dict_of(obj[k])
        raise ConvertError(f"cannot find a state_dict in the checkpoint (top-level keys: {sorted(obj)[:12]})")
    if sd and all(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd


def _pytorch_module(args):
    import torch
    src = Path(args.source)
    script = Path(args.script) if args.script else (src if src.suffix == ".py" else None)
    weights = Path(args.weights) if args.weights else (src if src.suffix != ".py" else None)
    if script is None:
        raise ConvertError("pytorch: --script build.py is required (it defines build_model())")
    if weights is not None:
        if not weights.is_file():
            raise ConvertError(f"weights {weights}: no such file")
        license_scan(source=weights)
    build = _load_script(script)
    try:
        n_params = len(inspect.signature(build).parameters)
    except (TypeError, ValueError):
        n_params = 0
    try:
        model = build(str(weights) if weights else None) if n_params >= 1 else build()
    except ConvertError:
        raise
    except Exception as e:  # noqa: BLE001 — user code
        raise ConvertError(f"build_model() raised {type(e).__name__}: {e}")
    if not isinstance(model, torch.nn.Module):
        raise ConvertError(f"build_model() returned {type(model).__name__}, not a torch.nn.Module")
    license_scan(module=model)  # an Ultralytics class built by the script: no file says so
    if weights is not None and n_params == 0:
        sd = _state_dict_of(_torch_load(weights, args.unsafe_pickle))
        try:
            model.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            raise ConvertError(f"weights do not fit build_model()'s module: {str(e).splitlines()[0]} ...")
        log(f"torch: loaded {len(sd)} tensors from {weights.name} (strict)")
    elif weights is None:
        log("torch: warning: no weights file — exporting build_model() as returned")
    return model, script, weights


# --------------------------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------------------------

def _flatten(out):
    import torch
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, dict):
        out = list(out.values())
    if isinstance(out, (list, tuple)):
        flat = []
        for o in out:
            flat += _flatten(o)
        return flat
    raise ConvertError(f"model returns a {type(out).__name__}; expected tensors (a tensor, tuple, list or dict)")


def _wrap(model):
    """A plain nn.Module whose forward returns a flat tuple of tensors (what the exporter names)."""
    import torch

    class Flat(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return tuple(_flatten(self.inner(x)))

    return Flat(model).eval()


def _export(module, x, path: Path, opset: int, allow_dynamo: bool, n_out: int):
    import torch
    names = ["output"] if n_out == 1 else [f"output_{i}" for i in range(n_out)]
    kw = dict(input_names=["input"], output_names=names, opset_version=opset, do_constant_folding=True,
              dynamic_axes=None)
    try:
        torch.onnx.export(module, (x,), str(path), dynamo=False, **kw)
        return "legacy (TorchScript-based, dynamo=False)"
    except Exception as e:  # noqa: BLE001
        if not allow_dynamo:
            raise ConvertError(f"ONNX export failed: {type(e).__name__}: {e}")
        log(f"torch: legacy exporter failed ({type(e).__name__}: {str(e).splitlines()[0]}); trying dynamo")
        first = e
    try:
        torch.onnx.export(module, (x,), str(path), dynamo=True, external_data=False, **kw)
        return "dynamo (torch.export)"
    except Exception as e:  # noqa: BLE001
        raise ConvertError(f"ONNX export failed with both exporters: legacy: {first}; dynamo: {e}")


def convert(args, workdir: Path):
    import torch

    resolve_license(args.license, None, args.source)  # license gate first: no work on a refused model
    w, h = parse_wxh(args.input)
    mean, std = parse_floats(args.mean), parse_floats(args.std)
    src = Path(args.source)
    if getattr(args, "module", None) is not None:
        # Python API: visionserve.convert.export(<torch.nn.Module>, ...) — no file, no unpickling.
        import copy
        license_scan(module=args.module)  # no file to scan: the classes themselves are the evidence
        model = copy.deepcopy(args.module).cpu()
        described = f"in-memory {type(args.module).__name__} (Python API)"
    elif args.format == "torchscript":
        if not src.is_file():
            raise ConvertError(f"{src}: no such file")
        license_scan(source=src)
        try:
            model = torch.jit.load(str(src), map_location="cpu")
        except Exception as e:  # noqa: BLE001
            raise ConvertError(f"{src.name} is not a TorchScript archive (torch.jit.load: {str(e).splitlines()[0]}). "
                               "For a state_dict + model code use the `pytorch` format with --script.")
        license_scan(module=model)  # qualified names of the scripted classes
        described = f"torchscript {src.name}"
    else:
        model, script, weights = _pytorch_module(args)
        described = f"pytorch {script.name}" + (f" + {weights.name}" if weights else "")
    model.eval()
    flat = _wrap(model)

    x = to_nchw(sample_image(w, h, seed=0), mean, std)
    with torch.no_grad():
        try:
            ref = [t.detach().cpu().numpy() for t in flat(torch.from_numpy(x))]
        except ConvertError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ConvertError(f"the model does not run on a [1,3,{h},{w}] float input: {type(e).__name__}: {e}")
    log(f"torch: outputs on [1,3,{h},{w}]: {[tuple(r.shape) for r in ref]}")
    if args.task == "detection" and len(ref) != 2:
        raise ConvertError(f"detection: the rt-detr decoder takes EXACTLY two outputs, boxes [1,Q,4] (cxcywh, "
                           f"normalised) and logits [1,Q,C]; the model returns {len(ref)}: "
                           f"{[tuple(r.shape) for r in ref]}")

    if args.task == "detection" and all(r.ndim == 3 and r.shape[-1] == 4 for r in ref):
        log("torch: warning: both outputs end in 4 (a 4-class head?) — the decoder takes the FIRST as boxes; "
            "make sure the model returns boxes first")

    onnx_path = workdir / "model.onnx"
    # A ScriptModule is handed to the exporter as is (it cannot be called from inside a trace of a
    # Python wrapper); the exporter flattens its tuple/list/dict outputs in the same order as _flatten.
    target = model if isinstance(model, torch.jit.ScriptModule) else flat
    how = _export(target, torch.from_numpy(x), onnx_path, args.opset,
                  args.format == "pytorch" or getattr(args, "module", None) is not None, len(ref))
    ins, outs = onnx_io(onnx_path)
    log(f"torch: exported with the {how} exporter, opset {args.opset}; inputs {[(n, s) for n, s, _ in ins]} "
        f"outputs {[(n, s) for n, s, _ in outs]}")
    if len(ins) != 1:
        raise ConvertError(f"exported graph has {len(ins)} inputs {[n for n, _, _ in ins]}; expected one image")

    for seed in (0, 1):
        xs = to_nchw(sample_image(w, h, seed=seed), mean, std)
        with torch.no_grad():
            r = [t.detach().cpu().numpy() for t in flat(torch.from_numpy(xs))]
        if args.task == "detection":  # a DETR's queries are a top-K set: see rfdetr.detr_parity
            from .rfdetr import detr_parity
            detr_parity(onnx_path, {ins[0][0]: xs}, r, args.tolerance, f" torch[seed {seed}]")
        else:
            parity(onnx_path, {ins[0][0]: xs}, r, args.tolerance, f" torch[seed {seed}]")

    notes = [described, f"task {args.task}, input {w}x{h} {'letterboxed' if args.letterbox else 'squashed'}; "
                        f"{how} ONNX exporter, opset {args.opset}",
             f"parity vs PyTorch checked at tolerance {args.tolerance:g}"]
    b = generic_bundle(args, onnx_path, notes=notes)
    from ..reference import TorchModuleReference
    b.reference = TorchModuleReference(flat, ins[0][0])
    return [b]
