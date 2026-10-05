"""Reduced precision for a converted ONNX model: FP16, INT8 (QDQ) and per-layer sensitivity.

    --precision fp16            ONNX FP32 -> FP16 weights/activations; float32 inputs/outputs are kept,
                                so the Go side and the manifest do not change
    --precision int8 --calib D  static INT8 in QDQ form (per-channel weights, symmetric int8), the form
                                TensorRT and ONNX Runtime both read; scales come from images in D
    --sensitivity               measure how much each MatMul/Gemm/Conv hurts the model's outputs when ONLY
                                that layer is reduced; writes <model>/sensitivity.json
    --keep-fp-top N             mixed precision: leave the N most sensitive layers in float
    --keep-fp-node NAME / --keep-fp-op OP   the same by hand

Everything here is OPT-IN: without --precision the converter's output is byte-for-byte what it was.
The ONNX file is the only thing changed; preprocessing, labels and the manifest's I/O stay as they
were, because inputs and outputs remain float32.

Accuracy is NOT decided here. This module compares the reduced model with the FP32 ONNX (a row in the
report) and the normal tiers then compare the INSTALLED model with the original framework (B1/B2) and
on labelled data (--eval, tier C, --max-map-drop). A reduced model without --eval is reported as
unmeasured.

Sensitivity method (one-at-a-time fake quantization). The graph is run on the calibration images once
with every candidate layer's inputs exposed, to learn each activation's range. Then, for each layer
alone, its weight is replaced by its quantize->dequantize round trip and a QuantizeLinear ->
DequantizeLinear (or Cast fp16 -> fp32) is put on its activation inputs; the model's outputs are
compared with the FP32 outputs. The error is a model-output distance (detection_error for a DETR head: 1 - matched IoU of the detections as a set; relative L2 otherwise), averaged over images, so layers
rank by how much of the model's output they destroy. It is a proxy: a layer that moves the logits of
queries nobody keeps can rank high. Use it to choose WHAT to try keeping in float, and let tier C
decide whether the result is acceptable. Cost: one ORT session per layer (graph optimisation off).
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

from .common import ConvertError, log, onnx_io
from .constants import IMAGE_EXT
from .report import INFO, SKIP, WARN, TierResult, grade_low, parse_thresholds

PRECISIONS = ("fp32", "fp16", "int8", "int4", "mixed")
METHODS = ("minmax", "entropy", "percentile")
QUANT_OPS = ("MatMul", "Gemm", "Conv")
_PERCENTILE = 99.99          # activation range used for the fake-quant scale (percentile / entropy)
_MIN_OPSET_QDQ = 13
# Ops with no FP16 kernel on ORT's CPU EP (the converter image verifies on CPU): converted models would
# fail to run there. ScatterElements(reduction=add) is what a deformable cross-attention exports to.
FP16_CPU_UNSUPPORTED = ("ScatterElements",)
DET_FLOOR = 0.3              # detections scoring below this (sigmoid) do not count in detection_error


# --------------------------------------------------------------------------------------------
# options
# --------------------------------------------------------------------------------------------

def add_precision_arguments(p) -> None:
    g = p.add_argument_group("precision (opt-in; the default output stays FP32 and bit-identical)")
    g.add_argument("--precision", choices=PRECISIONS, default="fp32",
                   help="fp16: half precision weights+activations; int8: static QDQ (needs --calib); int4: "
                        "weight-only 4-bit MatMul (MatMulNBits); mixed: a format per layer from --formats, "
                        "chosen by sensitivity to meet --max-output-err. Float32 inputs/outputs are kept. "
                        "Single-session image models only")
    g.add_argument("--formats", default="int8,fp16", metavar="LIST",
                   help="with --precision mixed: the formats a layer may take, from int4,int8,fp16 (fp32 is "
                        "always the fallback). Each layer gets the most aggressive one it tolerates; "
                        "needs --max-output-err (default int8,fp16)")
    g.add_argument("--sens-formats", metavar="LIST",
                   help="formats --sensitivity measures per layer, from int4,int8,fp16,fp8,fp4 (default: the "
                        "target precision). fp8/fp4 are SIMULATED: they show what that hardware would cost in "
                        "accuracy, nothing is built")
    g.add_argument("--int4-algo", choices=INT4_ALGOS, default="rtn",
                   help="INT4 weight quantizer: rtn (round to nearest, no dependencies) or hqq (needs torch)")
    g.add_argument("--int4-block", type=int, default=32, metavar="N", help="INT4 block size along K (default 32)")
    g.add_argument("--ep", choices=EP_CHOICES, default="auto",
                   help="ONNX Runtime provider for the precision step's runs (sensitivity, calibration "
                        "comparison): auto = CUDA when available, else CPU; cuda = require it. The GPU "
                        "converter image (`visionserve convert --gpu`) has CUDA; the default image does not")
    g.add_argument("--calib", metavar="DIR", help="images for INT8 calibration and for --sensitivity "
                   "(default: --images). Use images like the ones the model will see")
    g.add_argument("--calib-n", type=int, default=32, metavar="N", help="calibration images used (default 32)")
    g.add_argument("--calib-method", choices=METHODS, default="percentile",
                   help="INT8 activation range: minmax | entropy | percentile (default; clips rare outliers, "
                        "usually safer for transformers)")
    g.add_argument("--sensitivity", action="store_true",
                   help="measure each MatMul/Gemm/Conv's sensitivity to the target precision (--precision, "
                        "or int8 when that is fp32) and write sensitivity.json next to the model")
    g.add_argument("--sens-images", type=int, default=8, metavar="N",
                   help="images used per layer when measuring sensitivity (default 8; cost ~ layers x N runs)")
    g.add_argument("--max-output-err", type=float, default=0.0, metavar="E",
                   help="mixed precision, automatic: keep the fewest of the most sensitive layers in float32 "
                        "such that the reduced model's output error vs FP32 (the P row) is <= E "
                        "(implies --sensitivity; each step re-builds the model, INT8 ~30 s)")
    g.add_argument("--keep-fp-top", type=int, default=0, metavar="N",
                   help="mixed precision: keep the N most sensitive layers in float32 (implies --sensitivity)")
    g.add_argument("--keep-fp-node", action="append", default=[], metavar="NAME",
                   help="keep this node in float32 (repeatable)")
    g.add_argument("--keep-fp-op", action="append", default=[], metavar="OP",
                   help="keep every node of this op type in float32, e.g. Softmax or LayerNormalization (repeatable)")


def check_environment(args) -> None:
    """Refuse, BEFORE the export, a precision feature this Python environment cannot run. The
    converter image has two venvs: the PyTorch one (onnxruntime 1.26, onnx 1.21) has everything; the
    TensorFlow one (onnxruntime 1.18, numpy 1.26: tf2onnx pins them) may lack parts. A missing piece must
    be an error that says what to do, not a stack trace halfway through a long run."""
    prec = getattr(args, "precision", "fp32")
    want = set()
    if prec in ("fp16", "int8", "int4"):
        want.add(prec)
    if prec == "mixed":
        want |= set(parse_formats(getattr(args, "formats", "int8,fp16"), allowed=DEPLOYABLE))
    if getattr(args, "sens_formats", None):
        want |= set(parse_formats(args.sens_formats))
    if prec == "fp32" and not want and not getattr(args, "sensitivity", False):
        return
    missing = []

    def need(mod, why):
        try:
            __import__(mod, fromlist=["_"])
        except Exception:  # noqa: BLE001 — ImportError or an ABI error from a half-installed wheel
            missing.append(why)
    try:
        import onnxruntime
        ort_v = onnxruntime.__version__
    except ImportError:
        raise ConvertError("--precision needs onnxruntime in this Python environment")
    if "fp16" in want or prec == "mixed":
        need("onnxruntime.transformers.float16", "FP16 (onnxruntime.transformers.float16)")
    if "int4" in want:
        need("onnxruntime.quantization.matmul_nbits_quantizer", "INT4 (MatMulNBitsQuantizer)")
    if getattr(args, "int4_algo", "rtn") == "hqq" and ("int4" in want or prec in ("int4", "mixed")):
        try:
            from onnxruntime.quantization.matmul_nbits_quantizer import HQQWeightOnlyQuantConfig  # noqa: F401
            import torch  # noqa: F401
        except Exception:  # noqa: BLE001
            missing.append("--int4-algo hqq (HQQWeightOnlyQuantConfig and torch)")
    if "fp4" in want:
        try:
            import ml_dtypes
            ml_dtypes.float4_e2m1fn  # noqa: B018
        except Exception:  # noqa: BLE001
            missing.append("FP4 simulation (ml_dtypes >= 0.5)")
    if "fp8" in want:
        need("ml_dtypes", "FP8 simulation (ml_dtypes)")
    if missing:
        raise ConvertError(
            f"this Python environment (onnxruntime {ort_v}) cannot run: {'; '.join(missing)}. In the converter "
            "image the TensorFlow formats (tensorflow, keras, tflite) run in an older environment; convert "
            "the model to FP32 ONNX first and apply the precision step with a PyTorch-side format, or drop "
            "the option")


def validate_precision_args(args) -> None:
    """Everything checkable before the (slow) export."""
    prec = getattr(args, "precision", "fp32")
    wants_calib = (prec in ("int8", "mixed") or getattr(args, "sensitivity", False) or getattr(args, "keep_fp_top", 0) > 0
                   or getattr(args, "max_output_err", 0) > 0)
    if getattr(args, "max_output_err", 0) < 0 or not math.isfinite(getattr(args, "max_output_err", 0)):
        raise ConvertError("--max-output-err must be a finite number >= 0")
    if prec == "mixed":
        try:
            ladder = parse_formats(args.formats, allowed=DEPLOYABLE)
        except ConvertError as e:
            raise ConvertError(f"--formats: {e}")
        if not [f for f in ladder if f != "fp32"]:
            raise ConvertError("--formats needs at least one of int4, int8, fp16 for --precision mixed")
        if not getattr(args, "max_output_err", 0) > 0:
            raise ConvertError("--precision mixed needs --max-output-err E: the target the layer assignment "
                               "is searched against")
        if "int8" in ladder and not (getattr(args, "calib", None) or getattr(args, "images", None)):
            raise ConvertError("--formats with int8 needs calibration images: pass --calib DIR (or --images DIR)")
    if getattr(args, "sens_formats", None):
        try:
            parse_formats(args.sens_formats)
        except ConvertError as e:
            raise ConvertError(f"--sens-formats: {e}")
    if getattr(args, "int4_block", 32) < 8 or getattr(args, "int4_block", 32) & (getattr(args, "int4_block", 32) - 1):
        raise ConvertError("--int4-block must be a power of two >= 8")
    if getattr(args, "max_output_err", 0) > 0 and prec == "fp32":
        raise ConvertError("--max-output-err needs --precision fp16 or int8")
    if getattr(args, "keep_fp_top", 0) < 0:
        raise ConvertError("--keep-fp-top must be >= 0")
    if getattr(args, "calib_n", 1) < 1 or getattr(args, "sens_images", 1) < 1:
        raise ConvertError("--calib-n and --sens-images must be at least 1")
    if prec == "fp32" and (args.keep_fp_node or args.keep_fp_op or args.keep_fp_top):
        raise ConvertError("--keep-fp-* only makes sense with --precision fp16 or int8")
    check_environment(args)
    d = getattr(args, "calib", None) or getattr(args, "images", None)
    if wants_calib and not d:
        raise ConvertError("INT8 and --sensitivity need calibration images: pass --calib DIR (or --images DIR)")
    if d and not Path(d).is_dir():
        raise ConvertError(f"--calib {d}: expected a DIRECTORY of images")


# --------------------------------------------------------------------------------------------
# calibration data
# --------------------------------------------------------------------------------------------

def load_calibration(spec, input_name: str, input_shape: Sequence, directory, n: int) -> List[dict]:
    """Up to n images of `directory`, preprocessed exactly as the server will (spec.apply_spec) ->
    [{input_name: float32 [1,...]}]."""
    from PIL import Image
    from .spec import apply_spec
    files = sorted(f for f in Path(directory).rglob("*") if f.suffix.lower() in IMAGE_EXT)
    if not files:
        raise ConvertError(f"{directory}: no images ({', '.join(sorted(IMAGE_EXT))}) to calibrate with")
    if len(files) < 8:
        log(f"precision: warning: only {len(files)} calibration image(s); ranges will be unreliable")
    feeds = []
    for f in files[:n]:
        try:
            with Image.open(f) as im:
                x, _ = apply_spec(im, spec)
        except Exception as e:  # noqa: BLE001 — a corrupt image must not abort a calibration
            log(f"precision: skipping {f.name}: {e}")
            continue
        x = np.ascontiguousarray(x, dtype=np.float32)   # apply_spec already returns the batched tensor
        if [int(d) for d in x.shape] != [int(d) for d in input_shape]:
            raise ConvertError(f"calibration tensor {tuple(x.shape)} != model input {tuple(input_shape)}; "
                               "precision supports static-shape image models only")
        feeds.append({input_name: x})
    if not feeds:
        raise ConvertError(f"{directory}: no readable images")
    return feeds


# --------------------------------------------------------------------------------------------
# graph helpers
# --------------------------------------------------------------------------------------------

def ensure_node_names(model) -> int:
    """Give every unnamed node a unique name (ORT's node include/exclude lists work by name).
    Returns how many were named."""
    used = {n.name for n in model.graph.node if n.name}
    k = 0
    for i, n in enumerate(model.graph.node):
        if n.name:
            continue
        name = f"{n.op_type}__{i}"
        while name in used:
            name += "_"
        n.name = name
        used.add(name)
        k += 1
    return k


def batched_weight_matmuls(model) -> set:
    """MatMul nodes whose constant operand is not a plain 2-D matrix (a folded [1,heads,Q,D] tensor in
    an exported DETR decoder). ORT's QLinearMatMul cannot run a per-channel zero point on those, so
    they are left in float (and not scored)."""
    init = {i.name: len(i.dims) for i in model.graph.initializer}
    return {n.name for n in model.graph.node if n.op_type == "MatMul"
            and any(init.get(t, 2) != 2 for t in n.input)}


def drop_duplicate_producers(model) -> int:
    """ORT's FP16 pass can emit the same Cast twice (same input, output name and attributes) when
    several blocked consumers read one tensor. Keep the first, drop the repeats. A node that redefines
    a tensor with DIFFERENT content is a real error and is left for ORT to report."""
    made, drop = {}, []
    for i, n in enumerate(model.graph.node):
        key = (n.op_type, tuple(n.input), tuple(n.output), tuple(sorted(a.SerializeToString() for a in n.attribute)))
        if n.output and all(o in made for o in n.output) and all(made[o] == key for o in n.output):
            drop.append(i)
            continue
        for o in n.output:
            made.setdefault(o, key)
    for i in reversed(drop):
        del model.graph.node[i]
    return len(drop)


def dedupe_node_names(model) -> int:
    """Rename nodes that share a name. Returns how many were renamed."""
    seen, k = set(), 0
    for n in model.graph.node:
        if n.name in seen:
            i = 1
            while f"{n.name}__{i}" in seen:
                i += 1
            n.name = f"{n.name}__{i}"
            k += 1
        seen.add(n.name)
    return k


def _load(path):
    import onnx
    m = onnx.load(str(path))
    ensure_node_names(m)
    return m


def _save(model, path) -> Path:
    import onnx
    try:
        onnx.save(model, str(path))
    except ValueError as e:  # protobuf's 2 GB limit
        raise ConvertError(f"{Path(path).name}: cannot be saved as a single file ({e})")
    return Path(path)


def _rel_l2(ref: Sequence[np.ndarray], got: Sequence[np.ndarray]) -> float:
    """Mean over outputs of ||got - ref|| / ||ref||. inf when got has NaN/Inf the reference lacks."""
    errs = []
    for r, g in zip(ref, got):
        r, g = np.asarray(r, np.float64), np.asarray(g, np.float64)
        if r.shape != g.shape:
            raise ConvertError(f"output shape {g.shape} != reference {r.shape}")
        if not np.isfinite(g).all() and np.isfinite(r).all():
            return math.inf
        den = float(np.linalg.norm(r))
        errs.append(float(np.linalg.norm(g - r)) / den if den > 0 else float(np.linalg.norm(g - r)))
    return float(np.mean(errs)) if errs else 0.0


def _detr_pair(outs):
    """(boxes [Q,4] cxcywh, logits [Q,C]) when `outs` looks like a DETR head, else None. The boxes
    are the FIRST [1,Q,4] output (the Go rf-detr contract puts dets before labels); the logits are the
    next [1,Q,C] output — C may itself be 4 (a 3-class fine-tune has 3 + N/A logits)."""
    cand = [np.asarray(o) for o in outs if np.asarray(o).ndim == 3 and np.asarray(o).shape[0] == 1]
    boxes = next((o for o in cand if o.shape[2] == 4), None)
    if boxes is None:
        return None
    logits = next((o for o in cand if o is not boxes and o.shape[1] == boxes.shape[1] and o.shape[2] >= 2), None)
    if logits is None:
        return None
    return boxes[0].astype(np.float64), logits[0].astype(np.float64)


def _cxcywh_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[N,4] x [M,4] cxcywh -> IoU [N,M]."""
    ax0, ay0, ax1, ay1 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2, a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
    bx0, by0, bx1, by1 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    iw = np.clip(np.minimum(ax1[:, None], bx1[None]) - np.maximum(ax0[:, None], bx0[None]), 0, None)
    ih = np.clip(np.minimum(ay1[:, None], by1[None]) - np.maximum(ay0[:, None], by0[None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-12)


def detection_error(ref, got, floor: float = DET_FLOOR, min_iou: float = 0.3) -> float:
    """1 - (sum of IoU over one-to-one same-class matches) / max(#ref, #got), over the detections
    scoring >= floor on either side. 0 = the same detections, 1 = nothing in common. Unlike a
    per-query difference it is blind to the ORDER of the queries, which a DETR's top-K selection
    reshuffles on the slightest numeric change."""
    (rb, rl), (gb, gl) = ref, got
    if not (np.isfinite(gb).all() and np.isfinite(gl).all()) and np.isfinite(rb).all():
        return math.inf

    def dets(b, l):
        p = 1.0 / (1.0 + np.exp(-np.clip(l, -50, 50)))
        cls, sc = p.argmax(1), p.max(1)
        keep = np.where(sc >= floor)[0]
        return b[keep], cls[keep], sc[keep]
    rbx, rc, rs = dets(rb, rl)
    gbx, gc, gs = dets(gb, gl)
    n = max(len(rs), len(gs))
    if n == 0:
        return 0.0
    if not len(rs) or not len(gs):
        return 1.0
    iou = _cxcywh_iou(rbx, gbx) * (rc[:, None] == gc[None])
    total, used = 0.0, set()
    for i in np.argsort(-rs):                  # greedy by reference confidence
        cand = [(iou[i, j], j) for j in range(len(gs)) if j not in used and iou[i, j] >= min_iou]
        if cand:
            v, j = max(cand)
            used.add(j)
            total += float(v)
    return round(max(0.0, 1.0 - total / n), 12)   # IoU of a box with itself is 1 - 1e-16


def output_error(ref: Sequence[np.ndarray], got: Sequence[np.ndarray]) -> float:
    """The model-level distance between two runs of the same model, 0 = identical. A DETR-style head
    (boxes + class logits) is compared as a SET of detections (detection_error); anything else by
    relative L2 of the raw outputs."""
    a, b = _detr_pair(ref), _detr_pair(got)
    if a is not None and b is not None:
        return detection_error(a, b)
    return _rel_l2(ref, got)


def sqnr_db(err: float) -> float:
    return math.inf if err <= 0 else (-math.inf if math.isinf(err) else -20.0 * math.log10(err))


EP_CHOICES = ("auto", "cpu", "cuda")
CPU_EP = ("CPUExecutionProvider",)


GPU_MIN_IMAGES = 16     # below this a CUDA session costs more to create than it saves (see resolve_providers)


def resolve_providers(choice: str = "auto", n_images: Optional[int] = None) -> tuple:
    """--ep -> ONNX Runtime providers for one stage that runs `n_images` images through a fresh session.

    cpu = CPU. cuda = CUDA, and an error when this onnxruntime has none (a run meant for the GPU must not
    silently take hours on the CPU). auto = CUDA only when it pays: measured on RF-DETR base (RTX A6000),
    a CUDA session takes 3.1 s to create against 0.29 s on CPU, and then runs an image in 10 ms against
    183 ms, so it wins from about 16 images per session on. Sensitivity makes a session per layer and
    format, so with its default 8 images the CPU is faster. CPU is always last."""
    import onnxruntime as ort
    have = ort.get_available_providers()
    if choice == "cpu":
        return CPU_EP
    if "CUDAExecutionProvider" not in have:
        if choice == "cuda":
            raise ConvertError("--ep cuda: this onnxruntime has no CUDAExecutionProvider "
                               f"(available: {', '.join(have)}). Use the GPU converter image "
                               "(visionserve-convert:*-gpu) with `visionserve convert --gpu`")
        return CPU_EP
    if choice == "auto" and n_images is not None and n_images < GPU_MIN_IMAGES:
        return CPU_EP
    return ("CUDAExecutionProvider", "CPUExecutionProvider")


def _session(model_or_path, optimise: bool = True, providers=CPU_EP):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if not optimise:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    data = model_or_path if isinstance(model_or_path, (str, bytes)) else str(model_or_path)
    return ort.InferenceSession(data, so, providers=list(providers))


def run_all(sess, feeds: Sequence[dict], what: str = "model") -> List[list]:
    try:
        return [sess.run(None, f) for f in feeds]
    except Exception as e:  # noqa: BLE001 — ORT raises RuntimeException / Fail / InvalidArgument
        raise ConvertError(f"the {what} cannot run on ONNX Runtime's CPU EP: {str(e).splitlines()[0][:300]}. "
                           "Keep the offending op in float32 with --keep-fp-op OP (or --keep-fp-node NAME)")


def compare_models(ref_path, test_path, feeds, providers=CPU_EP) -> dict:
    """The reduced model against the FP32 one on the calibration images."""
    a, b = _session(ref_path, providers=providers), _session(test_path, providers=providers)
    errs = [output_error(ra, rb) for ra, rb in zip(run_all(a, feeds, 'FP32 model'), run_all(b, feeds, 'reduced model'))]
    finite = [e for e in errs if math.isfinite(e)]
    return {"images": len(errs), "err_mean": float(np.mean(finite)) if finite else math.inf,
            "err_max": max(errs) if errs else 0.0, "non_finite_images": len(errs) - len(finite)}


# --------------------------------------------------------------------------------------------
# FP16
# --------------------------------------------------------------------------------------------

def _keep_sets(model, keep_nodes, keep_ops):
    names = {n.name for n in model.graph.node}
    unknown = sorted(set(keep_nodes) - names)
    if unknown:
        raise ConvertError(f"--keep-fp-node: no such node(s): {', '.join(unknown[:5])}"
                           + (" ..." if len(unknown) > 5 else ""))
    ops = {n.op_type for n in model.graph.node}
    unknown_ops = sorted(set(keep_ops) - ops)
    if unknown_ops:
        raise ConvertError(f"--keep-fp-op: the model has no {', '.join(unknown_ops)} node "
                           f"(op types present: {', '.join(sorted(ops)[:12])} ...)")
    return set(keep_nodes), set(keep_ops)


def convert_fp16(src, dst, keep_nodes=(), keep_ops=()) -> Path:
    """FP32 -> FP16 with float32 I/O kept. Nodes in keep_nodes / ops in keep_ops stay float32 (the
    converter inserts Casts around them)."""
    from onnxruntime.transformers.float16 import DEFAULT_OP_BLOCK_LIST, convert_float_to_float16
    m = _load(src)
    nodes, ops = _keep_sets(m, keep_nodes, keep_ops)
    m16 = convert_float_to_float16(m, keep_io_types=True,
                                   op_block_list=list(DEFAULT_OP_BLOCK_LIST) + sorted(ops | set(FP16_CPU_UNSUPPORTED)),
                                   node_block_list=sorted(nodes))
    drop_duplicate_producers(m16)
    dedupe_node_names(m16)
    return _save(m16, dst)


# --------------------------------------------------------------------------------------------
# INT8 (static, QDQ)
# --------------------------------------------------------------------------------------------

class _Reader:
    def __init__(self, feeds):
        self._it = iter(feeds)

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        pass


def quantize_int8(src, dst, feeds, method="percentile", keep_nodes=(), keep_ops=(), workdir=None) -> Path:
    """Static INT8 in QDQ form: per-channel symmetric int8 weights, symmetric int8 activations.
    Only MatMul/Gemm/Conv are quantized; everything else (LayerNorm, Softmax, GELU, ...) stays float,
    which is where transformers are fragile. keep_* leave further nodes in float32."""
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    work = Path(workdir or Path(dst).parent)
    m = _load(src)
    if m.opset_import[0].version < _MIN_OPSET_QDQ:
        raise ConvertError(f"INT8 QDQ needs opset >= {_MIN_OPSET_QDQ}; the model has {m.opset_import[0].version}")
    nodes, ops = _keep_sets(m, keep_nodes, keep_ops)
    named = _save(m, work / "named.onnx")
    exclude = sorted(nodes | {n.name for n in m.graph.node if n.op_type in ops} | batched_weight_matmuls(m))
    del m
    cm = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
          "percentile": CalibrationMethod.Percentile}[method]
    try:
        quantize_static(str(named), str(dst), _Reader(feeds), quant_format=QuantFormat.QDQ,
                        op_types_to_quantize=list(QUANT_OPS), per_channel=True,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                        nodes_to_exclude=exclude, calibrate_method=cm,
                        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True})
    except ConvertError:
        raise
    except Exception as e:  # noqa: BLE001 — ORT raises assorted types
        raise ConvertError(f"INT8 quantization failed: {type(e).__name__}: {e}")
    named.unlink(missing_ok=True)
    return Path(dst)


# --------------------------------------------------------------------------------------------
# INT4 (weight-only) and mixed per-layer assignment
# --------------------------------------------------------------------------------------------

DEPLOYABLE = ("int4", "int8", "fp16", "fp32")      # formats a built model can contain
SIMULATED = ("fp8", "fp4")                         # sensitivity only: no ONNX Runtime 1.26 kernel path (see docs)
_ORDER = ("int4", "fp4", "int8", "fp8", "fp16", "fp32")   # most aggressive first
INT4_ALGOS = ("rtn", "hqq")


def int4_eligible(model) -> List[str]:
    """MatMul nodes whose B operand is a plain 2-D float constant (what MatMulNBits can store)."""
    init = {i.name: i for i in model.graph.initializer}
    out = []
    for n in model.graph.node:
        if n.op_type == "MatMul" and len(n.input) == 2 and n.input[1] in init and len(init[n.input[1]].dims) == 2 \
                and init[n.input[1]].data_type == 1 and n.input[0] not in init:
            out.append(n.name)
    return out


def quantize_int4(model, include, block_size: int = 32, algo: str = "rtn"):
    """Weight-only INT4 (blockwise, symmetric) for the MatMul nodes in `include`, as MatMulNBits.
    Activations stay float: no calibration data, and no activation outliers to fight. Returns the
    quantized ModelProto."""
    include = set(include)
    if not include:             # nothing to do: also works where MatMulNBitsQuantizer does not exist
        return model
    from onnxruntime.quantization.matmul_nbits_quantizer import (DefaultWeightOnlyQuantConfig,
                                                                 HQQWeightOnlyQuantConfig, MatMulNBitsQuantizer)
    # `nodes_to_include` is not honoured by every ORT build; excluding everything else always is.
    exclude = sorted(n.name for n in model.graph.node if n.op_type == "MatMul" and n.name not in include)
    if algo not in INT4_ALGOS:
        raise ConvertError(f"--int4-algo {algo!r}: expected one of {', '.join(INT4_ALGOS)}")
    try:
        cfg = (HQQWeightOnlyQuantConfig(block_size=block_size, bits=4) if algo == "hqq" else
               DefaultWeightOnlyQuantConfig(block_size=block_size, is_symmetric=True, bits=4))
        q = MatMulNBitsQuantizer(model, bits=4, block_size=block_size, is_symmetric=(algo != "hqq"),
                                 nodes_to_exclude=exclude, algo_config=cfg)
        q.process()
    except ImportError as e:     # hqq needs torch
        raise ConvertError(f"--int4-algo {algo} needs {e.name or 'a missing package'} in this environment")
    except Exception as e:  # noqa: BLE001
        raise ConvertError(f"INT4 quantization failed: {type(e).__name__}: {e}")
    return q.model.model


def build_mixed(src, dst, assign: dict, feeds, method="percentile", int4_algo="rtn", int4_block=32,
                fp16_rest=True, workdir=None) -> Path:
    """One model with a format per layer. `assign` maps a MatMul/Gemm/Conv node name to one of
    int4 | int8 | fp16 | fp32. Every other op follows `fp16_rest` (half precision when True).

    Order matters (see the comment below): INT8 static QDQ, then FP16 over everything not quantized or
    kept FP32, then INT4 (MatMulNBits, weight-only). ONNX Runtime can still fuse each Q-op-DQ group,
    as the node stays float32 in between."""
    import onnx
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    work = Path(workdir or Path(dst).parent)
    m = _load(src)
    names = {n.name: n for n in m.graph.node}
    bad = sorted(k for k in assign if k not in names)
    if bad:
        raise ConvertError(f"assignment names unknown node(s): {', '.join(bad[:5])}")
    unknown_fmt = sorted({v for v in assign.values() if v not in DEPLOYABLE})
    if unknown_fmt:
        raise ConvertError(f"cannot build format(s) {', '.join(unknown_fmt)}: buildable formats are "
                           f"{', '.join(DEPLOYABLE)} (fp8/fp4 are sensitivity-only)")
    int4 = [k for k, v in assign.items() if v == "int4"]
    int8 = [k for k, v in assign.items() if v == "int8"]
    fp32 = [k for k, v in assign.items() if v == "fp32"]
    ok4 = set(int4_eligible(m))
    if set(int4) - ok4:
        raise ConvertError("int4 needs a MatMul with a 2-D constant weight; not eligible: "
                           + ", ".join(sorted(set(int4) - ok4)[:5]))
    if m.opset_import[0].version < _MIN_OPSET_QDQ and int8:
        raise ConvertError(f"INT8 QDQ needs opset >= {_MIN_OPSET_QDQ}; the model has {m.opset_import[0].version}")
    # Order: INT8 (calibrated on the clean float32 graph), then FP16, then INT4.
    #  * INT8 first: ORT's quantizer fails on a graph that already mixes FP16 and FP32 regions.
    #  * FP16 second, with every quantized or kept layer blocked (Casts surround them) and Q/DQ left alone.
    #  * INT4 last: MatMulNBits is a contrib op the FP16 pass cannot type, so its FP32 output must not
    #    meet an FP16 bias Add; swapping it in after the pass, onto an already blocked FP32 MatMul with
    #    Casts around it, keeps every type right.
    cur = work / "mixed-step0.onnx"
    _save(m, cur)
    if int8:
        try:
            quantize_static(str(cur), str(work / "mixed-step1.onnx"), _Reader(feeds), quant_format=QuantFormat.QDQ,
                            op_types_to_quantize=list(QUANT_OPS), per_channel=True, activation_type=QuantType.QInt8,
                            weight_type=QuantType.QInt8, nodes_to_quantize=sorted(int8),
                            calibrate_method={"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
                                              "percentile": CalibrationMethod.Percentile}[method],
                            extra_options={"ActivationSymmetric": True, "WeightSymmetric": True})
        except Exception as e:  # noqa: BLE001
            raise ConvertError(f"INT8 quantization failed: {type(e).__name__}: {e}")
        cur = work / "mixed-step1.onnx"
    m = onnx.load(str(cur))
    ensure_node_names(m)
    if fp16_rest:
        from onnxruntime.transformers.float16 import DEFAULT_OP_BLOCK_LIST, convert_float_to_float16
        m = convert_float_to_float16(
            m, keep_io_types=True,
            op_block_list=list(DEFAULT_OP_BLOCK_LIST) + list(FP16_CPU_UNSUPPORTED) + [
                "QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"],
            node_block_list=sorted(set(int4) | set(int8) | set(fp32)))
        drop_duplicate_producers(m)
        dedupe_node_names(m)
    m = quantize_int4(m, int4, int4_block, int4_algo)
    cur = _save(m, work / "mixed-step2.onnx")
    del m
    shutil_copy(cur, dst)
    for f in ("mixed-step0.onnx", "mixed-step1.onnx", "mixed-step2.onnx"):
        (work / f).unlink(missing_ok=True)
    return Path(dst)


def shutil_copy(a, b):
    import shutil
    shutil.copyfile(str(a), str(b))


# --------------------------------------------------------------------------------------------
# sensitivity
# --------------------------------------------------------------------------------------------

@dataclasses.dataclass
class LayerScore:
    name: str
    op: str
    output_err: float            # model-output distance (see output_err) when only this layer is reduced
    sqnr_db: float               # -20 log10(output_err): higher = safer
    weight_sqnr_db: Optional[float]   # quantization error of the weight alone (None: no weight)
    act_absmax: float            # largest |activation| input over the calibration images
    act_outlier_ratio: float     # absmax / 99.9th percentile of |x|: large = a few huge values dominate
    weight_elems: int = 0
    rank: int = 0
    # every format measured for this layer: format -> output error. output_err is the entry of the
    # FIRST format (the one the ranking uses); a format a layer cannot take (INT4 on a Conv) is absent.
    errs: dict = dataclasses.field(default_factory=dict)

    def to_dict(self):
        d = dataclasses.asdict(self)
        d["errs"] = {k: (None if not math.isfinite(v) else v) for k, v in self.errs.items()}
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in d.items()}


INT4_BLOCK = 32
FP4_BLOCK = 16
_FP8_MAX = 448.0


def _fp8_roundtrip(x: np.ndarray, amax: float) -> np.ndarray:
    """Per-tensor scaled float8 e4m3fn round trip (scale = amax/448, saturating)."""
    import ml_dtypes
    scale = max(float(amax), 1e-12) / _FP8_MAX
    y = np.clip(np.asarray(x, np.float32) / scale, -_FP8_MAX, _FP8_MAX)
    return (y.astype(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale).astype(np.float32)


def _blockwise(w: np.ndarray, block: int, fn) -> np.ndarray:
    """Apply fn(blocks [Kb,block,N]) -> same shape, along axis 0 of a 2-D [K,N] weight (the MatMul
    reduction axis, which is how MatMulNBits and NVFP4 group weights)."""
    k, n = w.shape
    pad = (-k) % block
    wp = np.pad(w, ((0, pad), (0, 0))) if pad else w
    out = fn(wp.reshape(-1, block, n)).reshape(-1, n)
    return out[:k].astype(np.float32)


def weight_fakequant(w: np.ndarray, axis: int, fmt: str) -> np.ndarray:
    """The weight after a round trip through `fmt`: fp16 | int8 (per-channel) | fp8 (per-tensor e4m3) |
    int4 (blockwise 32 along K, symmetric, ORT's rule) | fp4 (e2m1 blockwise 16 along K, NVFP4-style).
    int4 / fp4 expect a 2-D [K,N] MatMul weight."""
    w = np.asarray(w, np.float32)
    if fmt == "fp16":
        return w.astype(np.float16).astype(np.float32)
    if fmt == "int8":
        red = tuple(i for i in range(w.ndim) if i != axis)
        amax = np.abs(w).max(axis=red, keepdims=True) if w.ndim > 1 else np.abs(w).max(keepdims=True)
        scale = np.maximum(amax, 1e-12) / 127.0
        return (np.clip(np.round(w / scale), -127, 127) * scale).astype(np.float32)
    if fmt == "fp8":
        return _fp8_roundtrip(w, float(np.abs(w).max()))
    if fmt == "int4":
        def q4(b):
            sc = np.maximum(np.abs(b).max(axis=1, keepdims=True), 1e-12) / 8.0
            return np.clip(np.round(b / sc), -8, 7) * sc
        return _blockwise(w, INT4_BLOCK, q4)
    if fmt == "fp4":
        import ml_dtypes

        def q_fp4(b):
            sc = np.maximum(np.abs(b).max(axis=1, keepdims=True), 1e-12) / 6.0
            y = np.clip(b / sc, -6.0, 6.0)
            return y.astype(ml_dtypes.float4_e2m1fn).astype(np.float32) * sc
        return _blockwise(w, FP4_BLOCK, q_fp4)
    raise ConvertError(f"unknown format {fmt!r}")


def _weight_axis(node, w_index: int, ndim: int) -> int:
    """The output-channel axis of node's weight, for per-channel quantization."""
    if node.op_type == "Conv":
        return 0
    if node.op_type == "Gemm":
        trans = {a.name: a.i for a in node.attribute}.get("transB" if w_index == 1 else "transA", 0)
        return 0 if (w_index == 1 and trans) else ndim - 1
    return ndim - 1     # MatMul: [..., K, N] -> N


def collect_activation_stats(model, targets, feeds, providers=CPU_EP):
    """name -> {absmax, p, p999} for every non-weight input of the target nodes, over `feeds`."""
    import onnx
    init = {i.name for i in model.graph.initializer}
    tensors = sorted({t for n in targets for t in n.input if t and t not in init})
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    have = {o.name for o in probe.graph.output}
    for t in tensors:
        if t not in have:
            probe.graph.output.append(onnx.ValueInfoProto(name=t))
    sess = _session(probe.SerializeToString(), optimise=False, providers=providers)
    names = [o.name for o in sess.get_outputs()]
    stats = {t: {"absmax": 0.0, "p": 0.0, "p999": 0.0} for t in tensors}
    for f in feeds:
        for name, val in zip(names, sess.run(None, f)):
            if name not in stats or val.dtype.kind != "f":
                continue
            a = np.abs(val.astype(np.float32, copy=False)).ravel()
            if not a.size:
                continue
            s = stats[name]
            s["absmax"] = max(s["absmax"], float(a.max()))
            s["p"] = max(s["p"], float(np.percentile(a, _PERCENTILE)))
            s["p999"] = max(s["p999"], float(np.percentile(a, 99.9)))
    return stats


def parse_formats(text, allowed=_ORDER) -> List[str]:
    """'int4,int8,fp16' -> ['int4', 'int8', 'fp16'] (unique, most aggressive first)."""
    items = [t.strip().lower() for t in (text.split(",") if isinstance(text, str) else text) if t.strip()]
    bad = [t for t in items if t not in allowed]
    if bad:
        raise ConvertError(f"unknown format(s) {', '.join(bad)}; choose from {', '.join(allowed)}")
    return [f for f in _ORDER if f in items]


def _applicable(fmt: str, node, skip: set, ok4: set) -> bool:
    if fmt in ("int4", "fp4"):
        return node.name in ok4
    if fmt in ("int8", "fp8"):
        return node.name not in skip
    return True        # fp16


def _act_nodes(helper, TensorProto, numpy_helper, fmt, t, new, stats, method, trial):
    """Nodes (and initializers appended to `trial`) that round-trip activation `t` through `fmt`."""
    s = stats[t]
    if fmt == "fp16":
        return [helper.make_node("Cast", [t], [f"{new}__h"], to=TensorProto.FLOAT16, name=f"{new}__c1"),
                helper.make_node("Cast", [f"{new}__h"], [new], to=TensorProto.FLOAT, name=f"{new}__c2")]
    rng = s["absmax"] if method == "minmax" else s["p"]
    if fmt == "int8":
        scale = np.float32(max(rng, 1e-8) / 127.0)
        trial.graph.initializer.append(numpy_helper.from_array(np.array(scale, np.float32), f"{new}__s"))
        trial.graph.initializer.append(numpy_helper.from_array(np.array(0, np.int8), f"{new}__z"))
        return [helper.make_node("QuantizeLinear", [t, f"{new}__s", f"{new}__z"], [f"{new}__q"], name=f"{new}__Q"),
                helper.make_node("DequantizeLinear", [f"{new}__q", f"{new}__s", f"{new}__z"], [new], name=f"{new}__DQ")]
    if fmt == "fp8":            # needs opset >= 19 (Cast to float8)
        sc = np.float32(max(rng, 1e-8) / _FP8_MAX)
        trial.graph.initializer.append(numpy_helper.from_array(np.array(1.0 / sc, np.float32), f"{new}__inv"))
        trial.graph.initializer.append(numpy_helper.from_array(np.array(sc, np.float32), f"{new}__sc"))
        return [helper.make_node("Mul", [t, f"{new}__inv"], [f"{new}__a"], name=f"{new}__m1"),
                helper.make_node("Cast", [f"{new}__a"], [f"{new}__f8"], to=TensorProto.FLOAT8E4M3FN, saturate=1,
                                 name=f"{new}__c1"),
                helper.make_node("Cast", [f"{new}__f8"], [f"{new}__b"], to=TensorProto.FLOAT, name=f"{new}__c2"),
                helper.make_node("Mul", [f"{new}__b", f"{new}__sc"], [new], name=f"{new}__m2")]
    return []                   # int4 / fp4: weight-only, activations untouched


def measure_sensitivity(src, feeds: Sequence[dict], fmt: str = "int8", method: str = "percentile",
                        progress: Optional[Callable[[int, int, str], None]] = None,
                        formats: Optional[Sequence[str]] = None, providers=CPU_EP) -> List[LayerScore]:
    """Rank every MatMul/Gemm/Conv by how much reducing ONLY it to a format changes the model's outputs.

    `formats` (default [fmt]) measures several formats per layer: int8 | fp16 | int4 | fp8 | fp4. The
    first one orders the ranking; every layer's `errs` carries all of them. int4 / fp4 are weight-only
    and only exist for a MatMul with a 2-D constant weight. fp8 / fp4 are SIMULATED round trips (no
    ONNX Runtime 1.26 kernel path builds them); the numbers say what such hardware would cost in
    accuracy, not that you can deploy it. Most sensitive first."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    fmts = parse_formats(list(formats) if formats else [fmt])
    if not fmts:
        raise ConvertError("no sensitivity format given")
    base = _load(src)
    if ("int8" in fmts) and base.opset_import[0].version < _MIN_OPSET_QDQ:
        raise ConvertError(f"sensitivity needs opset >= {_MIN_OPSET_QDQ}; the model has {base.opset_import[0].version}")
    skip = batched_weight_matmuls(base)
    ok4 = set(int4_eligible(base))
    targets = [n for n in base.graph.node if n.op_type in QUANT_OPS]
    if not targets:
        raise ConvertError("the model has no MatMul/Gemm/Conv to measure")
    init = {i.name: i for i in base.graph.initializer}
    ref_outs = run_all(_session(base.SerializeToString(), optimise=False, providers=providers), feeds)
    stats = collect_activation_stats(base, targets, feeds, providers)
    out_names = [o.name for o in base.graph.output]
    base19 = None
    if "fp8" in fmts and base.opset_import[0].version < 19:
        base19 = onnx.version_converter.convert_version(base, 19)
        ensure_node_names(base19)
    total = sum(1 for f in fmts for n in targets if _applicable(f, n, skip, ok4))
    done = 0
    by_name = {}
    for f in fmts:
        model_f = base19 if (f == "fp8" and base19 is not None) else base
        for node in targets:
            if not _applicable(f, node, skip, ok4):
                continue
            if progress:
                progress(done, total, f"{f}:{node.name}")
            done += 1
            trial = onnx.ModelProto()
            trial.CopyFrom(model_f)
            tn = next(n for n in trial.graph.node if n.name == node.name)
            added, w_sqnr, w_elems, amax, outlier = [], None, 0, 0.0, 0.0
            for idx, t in enumerate(list(tn.input)):
                if not t:
                    continue
                if t in init:
                    w = numpy_helper.to_array(init[t])
                    if w.dtype.kind != "f" or w.ndim < 1:
                        continue
                    fq = weight_fakequant(w, _weight_axis(node, idx, w.ndim), f)
                    e = float(np.linalg.norm(fq - w) / max(float(np.linalg.norm(w)), 1e-30))
                    w_sqnr, w_elems = sqnr_db(e), int(w.size)
                    new = f"{t}__fq{done}"
                    trial.graph.initializer.append(numpy_helper.from_array(fq, new))
                    tn.input[idx] = new
                    continue
                st = stats.get(t)
                if st is None:
                    continue
                amax = max(amax, st["absmax"])
                if st["p999"] > 0:
                    outlier = max(outlier, st["absmax"] / st["p999"])
                new = f"{t}__fq{done}"
                nodes = _act_nodes(helper, TensorProto, numpy_helper, f, t, new, stats, method, trial)
                if nodes:
                    added += nodes
                    tn.input[idx] = new
            pos = next(i for i, n in enumerate(trial.graph.node) if n.name == node.name)
            for j, n in enumerate(added):
                trial.graph.node.insert(pos + j, n)
            got = run_all(_session(trial.SerializeToString(), optimise=False, providers=providers), feeds)
            errs = [output_error(r[:len(out_names)], g[:len(out_names)]) for r, g in zip(ref_outs, got)]
            err = math.inf if any(math.isinf(e) for e in errs) else float(np.mean(errs))
            sc = by_name.get(node.name)
            if sc is None:
                sc = by_name[node.name] = LayerScore(node.name, node.op_type, err, sqnr_db(err), w_sqnr, amax,
                                                     outlier, w_elems)
            sc.errs[f] = err
            if f == fmts[0]:
                sc.output_err, sc.sqnr_db, sc.weight_sqnr_db = err, sqnr_db(err), w_sqnr
            del trial
    scores = list(by_name.values())
    first = fmts[0]
    scores.sort(key=lambda s: (first not in s.errs, -(s.errs.get(first, 0.0) if math.isfinite(s.errs.get(first, 0.0))
                                                         else 1e30), s.name))
    for i, s_ in enumerate(scores, 1):
        s_.rank = i
    return scores


def select_keep(scores: Sequence[LayerScore], top: int = 0) -> List[str]:
    """Names of the `top` most sensitive layers."""
    return [s.name for s in sorted(scores, key=lambda s: s.rank)[:max(0, top)]]


def sensitivity_payload(scores: Sequence[LayerScore], fmt: str, method: str, images: int, kept=(),
                        formats: Optional[Sequence[str]] = None) -> dict:
    return {"format": fmt, "formats": list(formats or [fmt]),
            "simulated_formats": [f for f in (formats or []) if f in SIMULATED], "calib_method": method, "images": images, "metric": "model output distance (detection_error for DETR heads, relative L2 otherwise), "
            "mean over images, one layer reduced at a time", "kept_in_float": list(kept),
            "layers": [s.to_dict() for s in sorted(scores, key=lambda s: s.rank)]}


def save_sensitivity(payload: dict, directory) -> Path:
    p = Path(directory) / "sensitivity.json"
    p.write_text(json.dumps(payload, indent=1) + "\n")
    return p


def format_top(scores: Sequence[LayerScore], n: int = 12) -> str:
    rows = ["  rank  op        err        SQNR dB  w-SQNR dB  outlier  layer"]
    for s in sorted(scores, key=lambda s: s.rank)[:n]:
        w = "    -" if s.weight_sqnr_db is None else f"{s.weight_sqnr_db:7.1f}"
        rows.append(f"  {s.rank:4d}  {s.op:8s}  {s.output_err:10.4g}  {s.sqnr_db:7.1f}  {w:>9}  "
                    f"{s.act_outlier_ratio:7.1f}  {s.name[-60:]}")
    return "\n".join(rows)


def format_matrix(scores: Sequence[LayerScore], formats: Sequence[str], n: int = 12) -> str:
    head = "  rank  " + "".join(f"{f:>9}" for f in formats) + "  layer"
    rows = [head]
    for s in sorted(scores, key=lambda s: s.rank)[:n]:
        cells = "".join(f"{s.errs[f]:9.3g}" if f in s.errs else f"{'-':>9}" for f in formats)
        rows.append(f"  {s.rank:4d}  {cells}  {s.name[-50:]}")
    return "\n".join(rows)


# --------------------------------------------------------------------------------------------
# glue for cli.run
# --------------------------------------------------------------------------------------------

@dataclasses.dataclass
class PrecisionResult:
    rows: List[TierResult] = dataclasses.field(default_factory=list)
    sensitivity: Optional[dict] = None

    @property
    def active(self) -> bool:
        return bool(self.rows) or self.sensitivity is not None


def assign_formats(scores: Sequence[LayerScore], ladder: Sequence[str], tau: float, keep: Sequence[str] = ()) -> dict:
    """node -> format. Each layer takes the most aggressive format in `ladder` whose OWN measured error
    is <= tau; a layer none of them fits (or that has no measurement) stays fp32. `keep` is forced fp32."""
    order = [f for f in _ORDER if f in ladder and f != "fp32"]
    out = {}
    for sc in scores:
        pick = "fp32"
        for f in order:
            e = sc.errs.get(f)
            if e is not None and math.isfinite(e) and e <= tau:
                pick = f
                break
        out[sc.name] = "fp32" if sc.name in keep else pick
    return out


def search_assignment(scores: Sequence[LayerScore], ladder: Sequence[str], build, target: float, keep=()):
    """Largest tau (the most aggressive assignment) whose BUILT model has error <= target.

    Candidate taus are the measured per-layer errors, so every step changes at least one layer.
    Bisection on the sorted candidates, assuming that a larger tau never makes the model better
    (the error is noisy, so the answer is a good assignment, not a proven optimum).
    build(assign) -> (dst, extra). Returns (assign, dst, extra, tau, steps)."""
    cands = sorted({e for sc in scores for f, e in sc.errs.items()
                    if f in ladder and e is not None and math.isfinite(e)} | {0.0})
    steps = []

    def at(i):
        tau = cands[i]
        asg = assign_formats(scores, ladder, tau, keep)
        dst, extra = build(asg)
        err = extra.get("err_mean", math.inf)
        steps.append((tau, err))
        cnt = {f: sum(1 for v in asg.values() if v == f) for f in DEPLOYABLE if any(v == f for v in asg.values())}
        log(f"precision: tau {tau:.4g} -> {cnt} -> output error {err:.4g} (target {target:g})")
        return asg, dst, extra, tau, err
    top = at(len(cands) - 1)
    if top[4] <= target:
        return top[:4] + (steps,)
    lo, hi, best = 0, len(cands) - 1, None
    base = at(0)
    if base[4] > target:
        raise ConvertError(f"--max-output-err {target:g} cannot be met even with every layer at its safest "
                           f"(error {base[4]:.4g}); the cause is outside the MatMul/Gemm/Conv layers")
    best = base
    while hi - lo > 1:
        mid = (lo + hi) // 2
        r = at(mid)
        if r[4] <= target:
            lo, best = mid, r
        else:
            hi = mid
    return best[:4] + (steps,)


def _sens_formats(args, prec: str) -> List[str]:
    if getattr(args, "sens_formats", None):
        fm = parse_formats(args.sens_formats)
    elif prec == "mixed":
        fm = parse_formats(args.formats, allowed=DEPLOYABLE)
    elif prec in ("int8", "fp16", "int4"):
        fm = [prec]
    else:
        fm = ["int8"]
    if prec == "mixed":     # the ladder must be measured whatever else was asked for
        fm = parse_formats(set(fm) | set(parse_formats(args.formats, allowed=DEPLOYABLE)))
    return [f for f in fm if f != "fp32"] or ["int8"]


def apply(bundles, args, workdir: Path) -> PrecisionResult:
    """Reduce the precision of the converted bundle in place (bundle.onnx['model'] is replaced) and/or
    measure sensitivity. A no-op unless --precision / --sensitivity / --keep-fp-* was given."""
    prec = getattr(args, "precision", "fp32")
    want_sens = (getattr(args, "sensitivity", False) or getattr(args, "keep_fp_top", 0) > 0
                 or getattr(args, "max_output_err", 0) > 0 or prec == "mixed")
    res = PrecisionResult()
    if prec == "fp32" and not want_sens:
        return res
    if len(bundles) != 1 or list(bundles[0].onnx) != ["model"]:
        raise ConvertError("--precision / --sensitivity support single-session models only (one ONNX "
                           "file, role 'model') for now; this conversion produces "
                           f"{[list(b.onnx) for b in bundles]}")
    b = bundles[0]
    src = Path(b.onnx["model"])
    ins, _ = onnx_io(src)
    if len(ins) != 1:
        raise ConvertError(f"--precision supports single-input models; this one has {[n for n, _, _ in ins]}")
    in_name, in_shape = ins[0][0], ins[0][1]
    out_dir = Path(workdir) / "precision"
    out_dir.mkdir(parents=True, exist_ok=True)

    ep = getattr(args, "ep", "auto")

    def pick(n):
        prov = resolve_providers(ep, n)
        log(f"precision: {n} image(s) per session -> ONNX Runtime {prov[0].replace('ExecutionProvider', '')}")
        return prov
    calib_dir = getattr(args, "calib", None) or getattr(args, "images", None)
    n_cal = max(1, int(getattr(args, "calib_n", 32)))
    feeds = (load_calibration(b.spec(), in_name, in_shape, calib_dir, n_cal) if calib_dir and
             (want_sens or prec in ("int8", "mixed") or prec != "fp32") else [])
    method = getattr(args, "calib_method", "percentile")
    keep_nodes = list(getattr(args, "keep_fp_node", []) or [])
    keep_ops = list(getattr(args, "keep_fp_op", []) or [])
    int4_algo, int4_block = getattr(args, "int4_algo", "rtn"), int(getattr(args, "int4_block", 32))
    ladder = parse_formats(args.formats, allowed=DEPLOYABLE) if prec == "mixed" else []

    scores = []
    if want_sens:
        sfmts = _sens_formats(args, prec)
        sfeeds = feeds[:max(1, int(getattr(args, "sens_images", 8)))]
        log(f"precision: measuring {', '.join(sfmts)} sensitivity of every MatMul/Gemm/Conv on {len(sfeeds)} image(s) ...")

        def progress(i, n, name):
            if i == 0 or (i + 1) * 10 // n != i * 10 // n:
                log(f"  sensitivity {i + 1}/{n} ({name.split(':')[0]})")

        scores = measure_sensitivity(src, sfeeds, sfmts[0], method, progress, formats=sfmts,
                                     providers=pick(len(sfeeds)))
        top = int(getattr(args, "keep_fp_top", 0))
        picked = select_keep(scores, top)
        keep_nodes += [n for n in picked if n not in keep_nodes]
        res.sensitivity = sensitivity_payload(scores, sfmts[0], method, len(sfeeds), picked, formats=sfmts)
        log("precision: most sensitive layers (" + sfmts[0] + "):\n" + format_top(scores))
        if len(sfmts) > 1:
            log("precision: error per format for the top layers:\n" + format_matrix(scores, sfmts))
        worst = scores[0]
        res.rows.append(TierResult("P", f"{'/'.join(sfmts)} sensitivity ({len(scores)} layers)", INFO,
                                   f"worst layer {worst.name[-40:]} (err {worst.output_err:.3g})"
                                   + (f"; {len(picked)} kept in float" if picked else "")
                                   + ("; fp8/fp4 simulated, not buildable" if set(sfmts) & set(SIMULATED) else ""),
                                   model=b.name, metrics={"top": [x.to_dict() for x in scores[:5]]}))

    if prec != "fp32":
        thr = parse_thresholds(getattr(args, "threshold", None))
        cmp_feeds = feeds
        if not cmp_feeds and getattr(args, "images", None):
            cmp_feeds = load_calibration(b.spec(), in_name, in_shape, args.images, 8)
        cmp_providers = pick(len(cmp_feeds)) if cmp_feeds else CPU_EP
        model0 = _load(src)
        eligible4 = set(int4_eligible(model0))
        skip8 = batched_weight_matmuls(model0)
        layer_names = [n.name for n in model0.graph.node if n.op_type in QUANT_OPS]
        op_keep = {n.name for n in model0.graph.node if n.op_type in set(keep_ops)}
        del model0

        def check(dst):
            return compare_models(src, dst, cmp_feeds, cmp_providers) if cmp_feeds else {}

        def build(keep):
            dst = out_dir / f"{src.stem}-{prec}.onnx"
            if prec == "fp16":
                convert_fp16(src, dst, keep, keep_ops)
            elif prec == "int8":
                quantize_int8(src, dst, feeds, method, keep, keep_ops, workdir=out_dir)
            else:   # int4: eligible MatMuls only, everything else float32
                asg = {n: ("int4" if n in eligible4 and n not in keep and n not in op_keep else "fp32")
                       for n in layer_names}
                build_mixed(src, dst, asg, feeds, method, int4_algo, int4_block, fp16_rest=False, workdir=out_dir)
            return dst, check(dst)

        def build_asg(asg):
            dst = out_dir / f"{src.stem}-mixed.onnx"
            build_mixed(src, dst, asg, feeds, method, int4_algo, int4_block, fp16_rest="fp16" in ladder,
                        workdir=out_dir)
            return dst, check(dst)

        target = float(getattr(args, "max_output_err", 0) or 0)
        assignment, tau = None, None
        if prec == "mixed":
            forced = set(keep_nodes) | op_keep
            assignment, dst, extra, tau, _ = search_assignment(scores, ladder, build_asg, target, forced)
            keep_nodes = [n for n, f in assignment.items() if f == "fp32"]
        elif target > 0:
            keep_nodes, dst, extra = _search_keep(build, res.sensitivity, keep_nodes, target)
        else:
            dst, extra = build(keep_nodes)
        if res.sensitivity is not None:
            res.sensitivity["kept_in_float"] = list(keep_nodes)
            if assignment is not None:
                res.sensitivity["assignment"] = assignment
                res.sensitivity["tau"] = tau
        b.onnx["model"] = str(dst)
        size0, size1 = src.stat().st_size, dst.stat().st_size
        if assignment is not None:
            cnt = {f: sum(1 for v in assignment.values() if v == f) for f in DEPLOYABLE}
            kept = ", layers " + " ".join(f"{f}={c}" for f, c in cnt.items() if c)
        else:
            kept = (f", {len(keep_nodes)} node(s)" + (f" + every {'/'.join(keep_ops)}" if keep_ops else "")
                    + " kept in float" if keep_nodes or keep_ops else "")
        line = f"{prec} ({size1 / 1e6:.1f} MB vs {size0 / 1e6:.1f} MB FP32{kept})"
        if extra:
            status = grade_low(extra["err_mean"], thr["p_err_warn"], thr["p_err_fail"])
            if extra["non_finite_images"]:
                status = WARN if status == "PASS" else status
            res.rows.append(TierResult(
                "P", f"{prec} vs FP32 ONNX", status,
                f"{line}; output error mean {extra['err_mean']:.3g}, max {extra['err_max']:.3g} "
                f"on {extra['images']} image(s)", model=b.name,
                metrics={**extra, "size_bytes": size1, "fp32_size_bytes": size0, "kept_in_float": len(keep_nodes)}))
        else:
            res.rows.append(TierResult("P", f"{prec} vs FP32 ONNX", SKIP, line + "; no images to compare on",
                                       model=b.name))
        detail = {"int8": f", INT8 QDQ per-channel, {method} calibration on {len(feeds)} image(s)",
                  "int4": f", INT4 weight-only MatMulNBits ({int4_algo}, block {int4_block})",
                  "mixed": f", per-layer {'/'.join(ladder)} (sensitivity-driven, target {target:g})"
                           + (f", INT4 {int4_algo} block {int4_block}" if "int4" in ladder else "")
                           + (f", INT8 {method} calibration" if "int8" in ladder else "")}.get(prec, "")
        b.notes.append(f"precision {prec}{detail}{kept} — output-changing: accuracy must be measured (--eval)")
        if not getattr(args, "eval", None):
            res.rows.append(TierResult("P", "accuracy of the reduced model", WARN,
                                       "not measured on labelled data (no --eval); only agreement with the "
                                       "FP32 model was checked", model=b.name))
    return res


def _search_keep(build, sens: dict, base_keep: Sequence[str], target: float):
    """Fewest top-sensitivity layers to keep in float so that build(keep)'s error <= target.
    Doubles N until the target is met, then bisects. -> (keep list, dst, extra)."""
    ranked = [l["name"] for l in sens["layers"]]
    pinned = list(base_keep)

    def at(n):
        keep = pinned + [x for x in ranked[:n] if x not in pinned]
        dst, extra = build(keep)
        err = extra.get("err_mean", math.inf)
        log(f"precision: keep {n:3d} most sensitive in float -> output error {err:.4g} (target {target:g})")
        return keep, dst, extra, err
    lo, hi, best = 0, None, None
    n = 0
    while True:
        r = at(n)
        if r[3] <= target:
            hi, best = n, r
            break
        lo = n
        if n >= len(ranked):
            raise ConvertError(f"--max-output-err {target:g} cannot be met even with every layer in float "
                               f"(error {r[3]:.4g}); the cause is outside the MatMul/Gemm/Conv layers")
        n = min(len(ranked), max(2, n * 2))
    while hi - lo > 1:
        mid = (lo + hi) // 2
        r = at(mid)
        if r[3] <= target:
            hi, best = mid, r
        else:
            lo = mid
    return best[0], best[1], best[2]
