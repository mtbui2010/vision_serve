"""TensorFlow family: SavedModel directories, Keras (.keras / .h5) and TFLite (.tflite) -> ONNX.

All three go through tf2onnx at the requested opset (default 17). What this module owns:

  * LAYOUT. TF graphs are NHWC; every VisionServe architecture feeds NCHW [1,3,H,W] float32.
    tf2onnx's `inputs_as_nchw` inserts the transpose inside the graph, so the ONNX input is NCHW
    and the Go side needs no special case. Batch is fixed to 1 and H,W to --input.
  * SIGNATURE / OUTPUT SELECTION. SavedModels are called through a signature (--signature,
    default serving_default). Graphs with more outputs than the task's contract needs are refused
    unless --outputs names the ones to keep (the rest are pruned from the graph).
  * TRAILING ACTIVATIONS. The Go decoders apply softmax (classification) / sigmoid (detection)
    themselves, so a graph that already ends in Softmax / Sigmoid would be squashed twice (ranking
    survives, scores do not). Such a trailing op is cut off, and the cut graph is re-checked:
    softmax/sigmoid(ONNX logits) must reproduce the original model's output.
  * QUANTIZED TFLITE. A uint8/int8 INPUT cannot satisfy the float NCHW contract -> refused.
    Float-I/O models with quantized internals are converted and trusted only if parity passes.
  * PARITY. The original (tf.function / Keras model / tf.lite.Interpreter) runs on the NHWC
    sample, ORT runs on the NCHW version of the SAME normalised pixels (common.to_nchw).

TensorFlow and tf2onnx are imported lazily: cli.build_parser imports every family.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from ..cli import add_generic_arguments, generic_bundle, parse_floats
from ..common import ConvertError, log, parity, sample_image, to_nchw
from ..constants import parse_wxh

_HELP = {
    "tensorflow": "TensorFlow SavedModel directory -> ONNX (tf2onnx)",
    "keras": "Keras model (.keras / .h5, Keras 2 format) -> ONNX (tf2onnx)",
    "tflite": "TFLite flatbuffer (.tflite, float input) -> ONNX (tf2onnx)",
}

_RAW_PIXELS_HINT = ("the graph normalises pixels itself, so VisionServe must feed raw 0-255 values: "
                    "pass --mean 0,0,0 --std 0.00392157,0.00392157,0.00392157")

_EPILOG = """\
Input layout: TF graphs are NHWC; the ONNX is exported with an NCHW [1,3,H,W] float32 input
(tf2onnx inputs_as_nchw), batch 1, H,W from --input.

Preprocessing: --mean/--std describe what VisionServe does BEFORE the graph (x/255, then
(x-mean)/std). Many Keras models (keras.applications with include_preprocessing, or any
Rescaling / Normalization layer) normalise internally and expect raw 0-255 pixels. For those
pass:  --mean 0,0,0 --std 0.00392157,0.00392157,0.00392157
The converter warns when it finds such layers but cannot know which convention a model was
trained with - check the model's documentation.

Outputs: classification needs exactly one [1,C] output; detection needs a DETR-style pair,
logits [1,Q,C] + boxes [1,Q,4] (cxcywh, normalised, NMS-free - TF Object Detection API models
with post-NMS yxyx outputs do NOT fit). Pick outputs with --outputs a,b (names as listed in the
error message when there are too many). A trailing Softmax/Sigmoid is removed because VisionServe
applies it itself.

License: TensorFlow checkpoints carry no license, so --license is required.
"""


def help_for(fmt: str) -> str:
    return _HELP[fmt]


def add_arguments(p: argparse.ArgumentParser, fmt: str) -> None:
    add_generic_arguments(p)
    p.formatter_class = argparse.RawDescriptionHelpFormatter
    p.epilog = _EPILOG
    if fmt == "tensorflow":
        p.add_argument("--signature", default="serving_default",
                       help="SavedModel signature to export (default serving_default)")
    p.add_argument("--outputs", help="comma-separated output names to keep, in order (others are dropped); "
                                     "required when the graph has more outputs than the task needs")


# --------------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------------

def convert(args, workdir: Path):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    w, h = parse_wxh(args.input)
    mean, std = parse_floats(args.mean), parse_floats(args.std)
    # License before any heavy work (CLAUDE.md rule 1); generic_bundle re-resolves it.
    from ..common import resolve_license
    resolve_license(args.license, None, args.source)

    src = Path(args.source)
    if not src.exists():
        raise ConvertError(f"{src} does not exist")
    # AGPL content gate before TensorFlow is even imported: Ultralytics TF exports carry their
    # license in metadata.yaml (SavedModel) / an appended metadata.json (TFLite).
    from ..common import license_scan
    license_scan(source=src)
    want = [s.strip() for s in args.outputs.split(",") if s.strip()] if getattr(args, "outputs", None) else None

    if args.format == "tflite":
        src_model = _TFLite(src, w, h)
    elif args.format == "keras":
        src_model = _Keras(src, w, h)
    else:
        src_model = _SavedModel(src, getattr(args, "signature", "serving_default"), w, h)

    keep = _select_outputs(args.task, src_model.outputs, want)
    if src_model.has_preprocessing and not _is_raw_pixels(mean, std):
        log(f"WARNING: {src.name} contains its own preprocessing ({src_model.has_preprocessing}); "
            f"{_RAW_PIXELS_HINT}. Current --mean {args.mean} --std {args.std} would normalise twice.")

    raw = workdir / f"{args.name}.full.onnx"
    src_model.export(keep, args.opset, raw)
    model = _finalize(raw, keep, src_model.onnx_output_names(keep), w, h)

    # Parity on the graph as exported (before any activation is cut).
    img = sample_image(w, h)
    nchw = to_nchw(img, mean, std)
    nhwc = np.ascontiguousarray(nchw.transpose(0, 2, 3, 1))
    ref = src_model.run(nhwc, keep)
    in_name = model.graph.input[0].name
    import onnx
    onnx.save(model, str(raw))
    err = parity(raw, {in_name: nchw}, ref, args.tolerance, f" [{args.format}]")

    out = workdir / f"{args.name}.onnx"
    notes = [f"{args.format} {src.name} (tf2onnx, opset {args.opset}, NHWC->NCHW input)",
             f"outputs: {', '.join(keep)}; parity max|d|/scale = {err:.2e}"]
    cut = _strip_trailing_activation(model, args.task)
    if cut:
        onnx.save(model, str(out))
        err2 = _check_cut(out, in_name, nchw, ref, cut, args.tolerance)
        notes.append(f"trailing {cut} removed (VisionServe applies it); re-check max|d| = {err2:.2e}")
    else:
        onnx.save(model, str(out))
    onnx.checker.check_model(str(out))
    if src_model.has_preprocessing:
        notes.append(f"graph contains preprocessing: {src_model.has_preprocessing}")
    return [generic_bundle(args, out, notes=notes)]


def _is_raw_pixels(mean, std) -> bool:
    return all(abs(m) < 1e-6 for m in mean) and all(abs(s - 1 / 255) < 1e-4 for s in std)


def _select_outputs(task: str, outputs: dict, want):
    """outputs: name -> shape (list, None = dynamic). Returns the names to keep, in order."""
    listing = ", ".join(f"{n} {list(s) if s is not None else '?'}" for n, s in outputs.items())
    if want:
        missing = [n for n in want if n not in outputs]
        if missing:
            raise ConvertError(f"--outputs {missing} not in the graph; it has: {listing}")
        keep = want
    else:
        keep = list(outputs)
    need = {"classification": 1, "detection": 2, "depth": 1, "embed": 1}[task]
    if len(keep) != need:
        how = "pick them with --outputs" if not want else "--outputs must name exactly that many"
        raise ConvertError(f"--task {task} needs exactly {need} output(s) "
                           f"({'logits [1,Q,C] + boxes [1,Q,4]' if task == 'detection' else '[1,C]' if task == 'classification' else 'see --help'}); "
                           f"{'the graph has' if not want else 'you selected'} {len(keep)}: {listing}. {how}.")
    return keep


# --------------------------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------------------------

def _tf():
    try:
        import tensorflow as tf  # noqa: WPS433
    except ImportError as e:
        raise ConvertError(f"TensorFlow is not installed in this environment ({e}); "
                           "install convert/requirements-tf.txt") from e
    import logging
    logging.getLogger("tf2onnx").setLevel(logging.WARNING)
    tf.get_logger().setLevel("ERROR")
    return tf


def _check_image_spec(shape, dtype, w, h, what):
    """shape: NHWC with None for dynamic dims."""
    if dtype not in ("float32",):
        raise ConvertError(f"{what}: input dtype is {dtype}; VisionServe feeds float32 NCHW. "
                           "uint8/int8 image inputs (quantized or TF-Hub style) cannot be served.")
    if shape is None or len(shape) != 4 or shape[-1] not in (3, None):
        raise ConvertError(f"{what}: input shape {shape} is not an NHWC RGB image [N,H,W,3]")
    if shape[0] not in (1, None):
        raise ConvertError(f"{what}: input batch is fixed to {shape[0]}; VisionServe feeds batch 1")
    if (shape[1] not in (None, h)) or (shape[2] not in (None, w)):
        raise ConvertError(f"{what}: graph input is fixed to {shape[2]}x{shape[1]} (WxH) but --input is {w}x{h}")


class _FunctionSource:
    """Shared by SavedModel and Keras: both become a tf.function returning a tuple of outputs."""
    has_preprocessing = ""
    outputs: dict

    def _fn(self, keep):
        raise NotImplementedError

    def export(self, keep, opset, path):
        tf = _tf()
        import tf2onnx
        spec = [tf.TensorSpec((1, self.h, self.w, 3), tf.float32, name="input")]
        try:
            tf2onnx.convert.from_function(self._fn(keep), input_signature=spec, opset=opset,
                                          inputs_as_nchw=["input:0"], output_path=str(path))
        except Exception as e:  # tf2onnx raises many types; surface them as a refusal
            raise ConvertError(f"tf2onnx could not convert the graph: {type(e).__name__}: {e}") from e

    def run(self, nhwc, keep):
        tf = _tf()
        return [np.asarray(t) for t in self._fn(keep)(tf.constant(nhwc))]

    def onnx_output_names(self, keep):
        return None  # tf2onnx keeps the tuple order of the tf.function outputs


class _SavedModel(_FunctionSource):
    def __init__(self, path: Path, signature: str, w: int, h: int):
        tf = _tf()
        self.w, self.h = w, h
        if not (path.is_dir() and (path / "saved_model.pb").exists()):
            raise ConvertError(f"{path} is not a SavedModel directory (no saved_model.pb)")
        self.obj = tf.saved_model.load(str(path))
        sigs = dict(self.obj.signatures)
        if signature not in sigs:
            raise ConvertError(f"signature {signature!r} not found; available: {sorted(sigs) or 'none'} "
                               "(pick one with --signature)")
        self.sig = sigs[signature]
        ins = self.sig.structured_input_signature[1]
        if len(ins) != 1:
            raise ConvertError(f"signature {signature!r} has {len(ins)} inputs {sorted(ins)}; "
                               "VisionServe feeds exactly one image")
        (self.in_key, spec), = ins.items()
        _check_image_spec(spec.shape.as_list() if spec.shape.rank is not None else None,
                          spec.dtype.name, w, h, f"signature {signature!r} input {self.in_key!r}")
        outs = self.sig.structured_outputs
        self.outputs = {k: (v.shape.as_list() if v.shape.rank is not None else None) for k, v in outs.items()}
        # Layer names survive in the node names of the signature's function library (the top
        # graph is only a StatefulPartitionedCall).
        gd = self.sig.graph.as_graph_def()
        names = " ".join([n.name for n in gd.node] + [f.signature.name for f in gd.library.function]
                         + [n.name for f in gd.library.function for n in f.node_def]).lower()
        self.has_preprocessing = ", ".join(k for k in ("rescaling", "normalization") if k in names)

    def _fn(self, keep):
        tf = _tf()
        sig, key = self.sig, self.in_key
        return tf.function(lambda x: tuple(sig(**{key: x})[k] for k in keep))


class _Keras(_FunctionSource):
    def __init__(self, path: Path, w: int, h: int):
        tf = _tf()
        self.w, self.h = w, h
        if path.suffix.lower() not in (".keras", ".h5", ".hdf5"):
            raise ConvertError(f"{path.name}: expected a .keras or .h5 file (for a SavedModel directory use "
                               "the 'tensorflow' format)")
        try:
            self.model = tf.keras.models.load_model(str(path), compile=False)
        except Exception as e:
            raise ConvertError(
                f"could not load {path.name} with Keras {tf.keras.__version__}: {type(e).__name__}: {e}. "
                "Custom layers are not supported; a .keras file saved by Keras 3 (TF >= 2.16) cannot be read "
                "by Keras 2 - re-export it as a SavedModel (model.export(dir)) and use the 'tensorflow' format."
            ) from e
        m = self.model
        if len(m.inputs) != 1:
            raise ConvertError(f"{path.name} has {len(m.inputs)} inputs; VisionServe feeds exactly one image")
        _check_image_spec(m.inputs[0].shape.as_list(), m.inputs[0].dtype.name, w, h, path.name)
        # Dict-output models are called -> dict keyed by the dict keys (NOT m.output_names, which
        # are layer names); list/single-output models -> positional, named by m.output_names.
        if isinstance(m.output, dict):
            self.names = list(m.output)
            self.outputs = {n: m.output[n].shape.as_list() for n in self.names}
        else:
            self.names = list(m.output_names)
            self.outputs = {n: t.shape.as_list() for n, t in zip(self.names, m.outputs)}
        found = set()
        for layer in _walk_layers(m):
            cls = type(layer).__name__
            if cls in ("Rescaling", "Normalization"):
                found.add(f"{cls} layer {layer.name!r}")
        self.has_preprocessing = ", ".join(sorted(found))

    def _fn(self, keep):
        tf = _tf()
        m, names = self.model, self.names

        def f(x):
            y = m(x, training=False)
            if isinstance(y, dict):
                d = y
            else:
                d = dict(zip(names, y if isinstance(y, (list, tuple)) else [y]))
            return tuple(d[k] for k in keep)
        return tf.function(f)


def _walk_layers(model):
    for layer in getattr(model, "layers", []):
        yield layer
        if hasattr(layer, "layers"):
            yield from _walk_layers(layer)


class _TFLite:
    def __init__(self, path: Path, w: int, h: int):
        tf = _tf()
        self.path, self.w, self.h = path, w, h
        self.it = tf.lite.Interpreter(model_path=str(path))
        ins = self.it.get_input_details()
        if len(ins) != 1:
            raise ConvertError(f"{path.name} has {len(ins)} inputs; VisionServe feeds exactly one image")
        d = ins[0]
        dtype = np.dtype(d["dtype"]).name
        if dtype != "float32":
            q = d.get("quantization", (0.0, 0))
            raise ConvertError(f"{path.name}: input is {dtype} (quantized, scale={q[0]:g} zero_point={q[1]}). "
                               "VisionServe feeds float32 NCHW, which a quantized input cannot accept. "
                               "Convert the float model (or a float-I/O TFLite export) instead.")
        sig = [int(v) if v >= 0 else None for v in d.get("shape_signature", d["shape"])]
        _check_image_spec(sig, dtype, w, h, path.name)
        self.in_tensor = d["name"]
        self.in_index = d["index"]
        self.dynamic = any(v is None for v in sig)
        # Friendly output names come from the signature runner when there is one; otherwise the
        # raw tensor names (e.g. StatefulPartitionedCall:0).
        self.outputs, self.tensor_of, self.index_of = {}, {}, {}
        sigs = self.it.get_signature_list()
        if len(sigs) == 1:
            runner = self.it.get_signature_runner(next(iter(sigs)))
            for k, od in runner.get_output_details().items():
                self._add_out(k, od)
            # keep the flatbuffer's output order for stable listings
            order = {od["name"]: i for i, od in enumerate(self.it.get_output_details())}
            self.outputs = dict(sorted(self.outputs.items(), key=lambda kv: order.get(self.tensor_of[kv[0]], 0)))
        else:
            for od in self.it.get_output_details():
                self._add_out(od["name"], od)
        names = " ".join(t["name"].lower() for t in self.it.get_tensor_details())
        self.has_preprocessing = ", ".join(k for k in ("rescaling", "normalization") if k in names)

    def _add_out(self, k, od):
        s = od.get("shape_signature", od["shape"])
        self.outputs[k] = [int(v) if v >= 0 else None for v in s]
        self.tensor_of[k] = od["name"]
        self.index_of[k] = od["index"]

    def export(self, keep, opset, path):
        import tf2onnx
        outs = {o["name"]: o for o in self.it.get_output_details()}
        for k in keep:
            dt = np.dtype(outs[self.tensor_of[k]]["dtype"]).name
            if dt != "float32":
                raise ConvertError(f"output {k!r} is {dt} (quantized); VisionServe decodes float32 outputs")
        try:
            tf2onnx.convert.from_tflite(str(self.path), opset=opset, inputs_as_nchw=[self.in_tensor],
                                        output_names=[self.tensor_of[k] for k in keep], output_path=str(path))
        except Exception as e:
            raise ConvertError(f"tf2onnx could not convert {self.path.name}: {type(e).__name__}: {e}") from e

    def run(self, nhwc, keep):
        it = self.it
        if self.dynamic:
            it.resize_tensor_input(self.in_index, list(nhwc.shape))
        it.allocate_tensors()
        it.set_tensor(self.in_index, nhwc)
        it.invoke()
        return [np.array(it.get_tensor(self.index_of[k])) for k in keep]

    def onnx_output_names(self, keep):
        return [self.tensor_of[k] for k in keep]


# --------------------------------------------------------------------------------------------
# ONNX graph surgery
# --------------------------------------------------------------------------------------------

def _finalize(path: Path, keep, onnx_names, w: int, h: int):
    """Fix the input to [1,3,H,W] named 'input', order the outputs like `keep` (matching by
    `onnx_names` when given, else positionally) and name them after `keep`."""
    import onnx
    m = onnx.load(str(path))
    g = m.graph
    init = {t.name for t in g.initializer}
    ins = [i for i in g.input if i.name not in init]
    if len(ins) != 1:
        raise ConvertError(f"converted graph has {len(ins)} inputs, expected 1")
    _rename(g, ins[0].name, "input")
    dims = ins[0].type.tensor_type.shape.dim
    for d, v in zip(dims, (1, 3, h, w)):
        d.Clear()
        d.dim_value = v
    if len(g.output) != len(keep):
        raise ConvertError(f"converted graph has {len(g.output)} outputs, expected {len(keep)} ({keep})")
    if onnx_names:
        by = {o.name: o for o in g.output}
        if set(by) != set(onnx_names):
            raise ConvertError(f"converted graph outputs {sorted(by)} do not match the selected {onnx_names}")
        ordered = [by[n] for n in onnx_names]
        copies = [type(o)() for o in ordered]
        for c, o in zip(copies, ordered):
            c.CopyFrom(o)
        del g.output[:]
        g.output.extend(copies)
    for o, k in zip(list(g.output), keep):
        new = _safe_name(k)
        if o.name != new:
            _rename(g, o.name, new)
    for o in g.output:  # batch is 1
        d = o.type.tensor_type.shape.dim
        if len(d) and not d[0].HasField("dim_value"):
            d[0].Clear()
            d[0].dim_value = 1
    try:
        m = onnx.shape_inference.infer_shapes(m)
    except Exception as e:  # shapes are cosmetic here; parity is the real check
        log(f"  note: ONNX shape inference failed ({e}); output shapes may stay symbolic")
    return m


def _safe_name(k: str) -> str:
    return "".join(c if c.isalnum() or c in "_-." else "_" for c in k)


def _rename(g, old: str, new: str) -> None:
    if old == new:
        return
    taken = {v for n in g.node for v in list(n.input) + list(n.output)} | {i.name for i in g.input}
    if new in taken:
        raise ConvertError(f"cannot rename {old!r} to {new!r}: name already used in the graph")
    for n in g.node:
        for i, v in enumerate(n.input):
            if v == old:
                n.input[i] = new
        for i, v in enumerate(n.output):
            if v == old:
                n.output[i] = new
    for coll in (g.input, g.output, g.value_info):
        for v in coll:
            if v.name == old:
                v.name = new


def _strip_trailing_activation(m, task: str):
    """Cut a final Softmax (classification) / Sigmoid (detection logits). Returns the op name cut,
    or None. The graph output keeps its name and shape (softmax/sigmoid preserve shape)."""
    g = m.graph
    producer = {o: n for n in g.node for o in n.output}
    if task == "classification":
        targets, op = [g.output[0]], "Softmax"
    elif task == "detection":
        targets = [o for o in g.output if len(o.type.tensor_type.shape.dim) == 3
                   and o.type.tensor_type.shape.dim[2].dim_value != 4]
        op = "Sigmoid"
    else:
        return None
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers[i] = consumers.get(i, 0) + 1
    fixed = {i.name for i in g.input} | {t.name for t in g.initializer}
    cut = None
    for o in targets:
        n = producer.get(o.name)
        # Sigmoid is elementwise, so it may sit behind reshapes (Dense(sigmoid) -> Reshape).
        while (n is not None and op == "Sigmoid" and n.op_type in _SHAPE_OPS
               and consumers.get(n.input[0], 0) == 1):
            n = producer.get(n.input[0])
        if n is None or n.op_type != op or n.input[0] in fixed:
            continue
        if op == "Softmax":
            axis = next((a.i for a in n.attribute if a.name == "axis"), -1)
            if axis not in (-1, len(o.type.tensor_type.shape.dim) - 1):
                continue
        out, src = n.output[0], n.input[0]
        g.node.remove(n)
        if out == o.name:   # activation is the graph output: its input takes over the output name
            old, new = src, out
        else:               # activation behind shape ops: its consumer reads the input directly
            old, new = out, src
        for other in g.node:
            for i, v in enumerate(other.output):
                if v == old:
                    other.output[i] = new
            for i, v in enumerate(other.input):
                if v == old:
                    other.input[i] = new
        cut = op
    if cut:
        del g.value_info[:]  # stale after rewiring; shapes of inputs/outputs are unchanged
        _prune(g)
    return cut


_SHAPE_OPS = {"Reshape", "Transpose", "Identity", "Squeeze", "Unsqueeze", "Flatten"}


def _prune(g) -> None:
    needed = {o.name for o in g.output}
    keep = []
    for n in reversed(list(g.node)):
        if any(o in needed for o in n.output):
            keep.append(n)
            needed.update(i for i in n.input if i)
            for a in n.attribute:  # subgraphs (If/Loop) reference outer names
                for sub in ([a.g] if a.HasField("g") else []) + list(a.graphs):
                    for sn in sub.node:
                        needed.update(sn.input)
    keep.reverse()
    del g.node[:]
    g.node.extend(keep)
    inits = [t for t in g.initializer if t.name in needed]
    del g.initializer[:]
    g.initializer.extend(inits)


def _check_cut(path, in_name, nchw, ref, cut, tol) -> float:
    import onnxruntime as ort
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {in_name: nchw})
    worst = 0.0
    for g, r in zip(got, ref):
        g, r = np.asarray(g, np.float64), np.asarray(r, np.float64)
        if cut == "Softmax":
            e = np.exp(g - g.max(-1, keepdims=True))
            act = e / e.sum(-1, keepdims=True)
        elif g.ndim == 3 and g.shape[-1] != 4:
            act = 1 / (1 + np.exp(-g))
        else:
            act = g
        worst = max(worst, float(np.abs(act - r).max()) / max(1.0, float(np.abs(r).max())))
    log(f"  parity [{cut} removed]: {cut.lower()}(onnx) vs original max|d|/scale = {worst:.2e}")
    if worst > tol:
        raise ConvertError(f"after removing the trailing {cut}, {cut.lower()}(ONNX) differs from the original "
                           f"by {worst:.2e} (> {tol:g})")
    from ..common import record_parity
    record_parity(f" [{cut} removed]", worst, tol)
    return worst
