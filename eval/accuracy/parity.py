"""Export numerical parity: PyTorch checkpoint vs ONNX Runtime export (W6 Part A).

Feeds identical inputs to (i) the original PyTorch model and (ii) ORT on the shipped .onnx
export, then reports per-output max-abs error, relative error, and cosine similarity, and asserts
agreement with ``numpy.testing.assert_allclose(rtol=1e-3, atol=1e-5)``. One row per model makes a
silently-broken export visible.

The ORT side is fully implemented (it only needs the .onnx file). The PyTorch side needs the
original architecture + checkpoint, which is model-specific; that loader is a TODO you fill per
model. We deliberately do NOT fabricate a torch model — random weights would produce a meaningless
"parity" number.

Usage:

    python -m eval.accuracy.parity \
        --onnx ~/.visionserve/models/mobilenet-v3/model.onnx \
        --torch-ckpt /path/to/mobilenet_v3.pth \
        --model mobilenet-v3 --n 16
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("onnxruntime required; pip install onnxruntime-gpu==1.20.1") from exc


@dataclass
class ParityRow:
    model: str
    output_name: str
    max_abs_err: float
    max_rel_err: float
    cosine_sim: float
    shape: tuple[int, ...]
    passed: bool
    note: str = ""


@dataclass
class ParityReport:
    model: str
    rtol: float
    atol: float
    rows: list[ParityRow] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.rows) and bool(self.rows)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def ort_outputs(onnx_path: str, feeds: dict[str, np.ndarray],
                providers: Optional[list[str]] = None) -> dict[str, np.ndarray]:
    """Run the ORT export and return {output_name: array}."""
    sess = ort.InferenceSession(
        onnx_path, providers=providers or ["CPUExecutionProvider"]
    )
    names = [o.name for o in sess.get_outputs()]
    outs = sess.run(names, feeds)
    return dict(zip(names, outs))


def load_torch_model(model: str, ckpt: str, device: str = "cpu") -> Callable[[Any], Any]:
    """Return a callable mapping a torch input tensor -> a dict/list/tensor of outputs.

    TODO: requires the original architecture + checkpoint for ``model``. Examples:

        # mobilenet-v3 (torchvision)
        import torch, torchvision
        m = torchvision.models.mobilenet_v3_large(weights=None)
        m.load_state_dict(torch.load(ckpt, map_location=device)); m.eval().to(device)
        return lambda x: m(x)

        # RF-DETR / MobileSAM / GroundingDINO are NOT in torchvision — import their repo's model
        # definition and load the matching checkpoint. Output names must line up with the ONNX
        # graph's output order for a meaningful comparison.

    Do not substitute random weights.
    """
    raise NotImplementedError(
        f"load_torch_model is a TODO for model={model!r}: supply the architecture + checkpoint "
        f"loader (ckpt={ckpt!r}). See the docstring for examples."
    )


def make_inputs(onnx_path: str, n: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Build a batch of random inputs matching the ONNX graph's input spec.

    Random *inputs* (not random weights) are valid for numerical parity: we compare two
    implementations of the SAME function on the SAME inputs. For task-relevant ranges you may
    instead feed real preprocessed images (see eval/baselines/preprocess.py).
    """
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    feeds: dict[str, np.ndarray] = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        if n > 1 and shape:
            shape[0] = n
        dtype = np.float32 if "float" in (inp.type or "") else np.int64
        if dtype == np.float32:
            feeds[inp.name] = rng.standard_normal(shape).astype(np.float32)
        else:
            # integer inputs (e.g. token ids) — fill with small non-negative ints
            feeds[inp.name] = rng.integers(0, 1, size=shape, dtype=np.int64)
            # TODO: requires model-specific token/attention inputs for GroundingDINO-style graphs.
    return feeds


def compare(
    model: str,
    onnx_path: str,
    torch_ckpt: Optional[str],
    n: int = 16,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    device: str = "cpu",
) -> ParityReport:
    """Run both sides on identical inputs and build a parity report."""
    feeds = make_inputs(onnx_path, n)
    ort_out = ort_outputs(onnx_path, feeds)

    report = ParityReport(model=model, rtol=rtol, atol=atol)

    # --- PyTorch side (TODO loader) -------------------------------------------------------
    torch_out: Optional[dict[str, np.ndarray]] = None
    if torch_ckpt:
        import torch  # type: ignore

        fn = load_torch_model(model, torch_ckpt, device)  # raises NotImplementedError until filled
        # The single float input is assumed to be the image tensor; adapt for multi-input graphs.
        first = next(iter(feeds.values()))
        with torch.no_grad():
            t_out = fn(torch.from_numpy(first).to(device))
        torch_out = _torch_to_numpy_dict(t_out, list(ort_out.keys()))

    for name, ref in ort_out.items():
        ref = np.asarray(ref)
        if torch_out is None:
            report.rows.append(ParityRow(
                model=model, output_name=name, max_abs_err=float("nan"),
                max_rel_err=float("nan"), cosine_sim=float("nan"),
                shape=tuple(ref.shape), passed=False,
                note="TODO: no torch checkpoint provided — ORT-only run, parity not computed",
            ))
            continue
        got = np.asarray(torch_out[name])
        abs_err = float(np.max(np.abs(got - ref)))
        denom = np.maximum(np.abs(ref), 1e-12)
        rel_err = float(np.max(np.abs(got - ref) / denom))
        cos = _cosine(got, ref)
        try:
            np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol)
            passed = True
            note = ""
        except AssertionError as e:  # noqa: BLE001
            passed = False
            note = str(e).splitlines()[0]
        report.rows.append(ParityRow(
            model=model, output_name=name, max_abs_err=abs_err, max_rel_err=rel_err,
            cosine_sim=cos, shape=tuple(ref.shape), passed=passed, note=note,
        ))
    return report


def _torch_to_numpy_dict(out: Any, names: list[str]) -> dict[str, np.ndarray]:
    """Normalize a torch model's output into {onnx_output_name: ndarray}, in ONNX output order."""
    import torch  # type: ignore

    if isinstance(out, dict):
        return {n: out[n].detach().cpu().numpy() for n in names if n in out}
    if isinstance(out, (list, tuple)):
        return {names[i]: o.detach().cpu().numpy() for i, o in enumerate(out) if i < len(names)}
    if isinstance(out, torch.Tensor):
        return {names[0]: out.detach().cpu().numpy()}
    raise TypeError(f"unsupported torch output type {type(out)!r}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="ORT export (same file VisionServe serves)")
    ap.add_argument("--torch-ckpt", default=None, help="original PyTorch checkpoint (TODO loader)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=16, help="batch size for the parity inputs")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="optional JSON report path")
    args = ap.parse_args(argv)

    report = compare(args.model, args.onnx, args.torch_ckpt, n=args.n,
                     rtol=args.rtol, atol=args.atol, device=args.device)
    payload = {
        "model": report.model, "rtol": report.rtol, "atol": report.atol,
        "passed": report.passed,
        "rows": [vars(r) for r in report.rows],
    }
    text = json.dumps(payload, indent=2, default=str)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    # exit non-zero if any row failed parity (so CI can gate on it) — but not for the
    # TODO/no-checkpoint case which simply has nothing to assert.
    return 0 if report.passed or not args.torch_ckpt else 1


if __name__ == "__main__":
    raise SystemExit(main())
