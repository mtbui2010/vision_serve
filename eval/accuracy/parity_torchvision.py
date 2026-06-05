"""Export parity for torchvision classifiers (W6 Part A): PyTorch vs the shipped ONNX.

Feeds identical random inputs to (i) the torchvision pretrained model and (ii) ORT on the
shipped .onnx, and reports max-abs / max-rel error + cosine similarity on the logits, asserting
agreement. This is the numerical-fidelity check that complements the end-to-end Top-1 result:
if these two implement the same function on the same input, the ONNX export is faithful.

Random *inputs* (not weights) are valid — we compare two implementations of the same function.
A FAIL here would mean the shipped ONNX was NOT exported from torchvision IMAGENET1K_V1 weights
(useful to know), not necessarily a bug.

Run (in the `label` env, which has torch+torchvision):
    python -m eval.accuracy.parity_torchvision --model mobilenet-v3 \
        --onnx models/mobilenet-v3/model.onnx --out eval/results/parity_mobilenet-v3.json
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

# torchvision constructor + pretrained weights enum, per VisionServe model name.
TORCHVISION_MODELS = {
    "mobilenet-v3": ("mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
    "efficientnet-b0": ("efficientnet_b0", "EfficientNet_B0_Weights"),
}


def _cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=list(TORCHVISION_MODELS))
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import onnxruntime as ort  # type: ignore
    import torch  # type: ignore
    import torchvision  # type: ignore

    ctor_name, weights_enum_name = TORCHVISION_MODELS[args.model]
    ctor = getattr(torchvision.models, ctor_name)
    weights = getattr(torchvision.models, weights_enum_name).IMAGENET1K_V1
    tmodel = ctor(weights=weights).eval()

    # ONNX session + input spec. The shipped exports use a FIXED batch dim (e.g. [1,3,224,224]),
    # so we cannot batch n samples; instead we run n independent batch-1 inputs and aggregate.
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out_names = [o.name for o in sess.get_outputs()]
    base_shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]  # batch->1 if dynamic
    rng = np.random.default_rng(0)

    onnx_all, torch_all = [], []
    for _ in range(args.n):
        x = rng.standard_normal(base_shape).astype(np.float32)
        onnx_all.append(sess.run(out_names, {inp.name: x})[0])
        with torch.no_grad():
            torch_all.append(tmodel(torch.from_numpy(x)).cpu().numpy())
    onnx_logits = np.concatenate(onnx_all, axis=0)
    torch_logits = np.concatenate(torch_all, axis=0)

    abs_err = float(np.max(np.abs(torch_logits - onnx_logits)))
    denom = np.maximum(np.abs(onnx_logits), 1e-12)
    rel_err = float(np.max(np.abs(torch_logits - onnx_logits) / denom))
    cos = _cosine(torch_logits, onnx_logits)
    try:
        np.testing.assert_allclose(torch_logits, onnx_logits, rtol=args.rtol, atol=args.atol)
        passed, note = True, ""
    except AssertionError as e:
        passed, note = False, str(e).splitlines()[0]

    # also: do the two agree on the predicted class for each sample? (argmax match rate)
    argmax_match = float(np.mean(onnx_logits.argmax(-1) == torch_logits.argmax(-1)))

    report = {
        "model": args.model, "onnx": args.onnx,
        "torchvision": ctor_name + " (IMAGENET1K_V1)",
        "n": args.n, "rtol": args.rtol, "atol": args.atol,
        "max_abs_err": abs_err, "max_rel_err": rel_err, "cosine_sim": cos,
        "argmax_match_rate": argmax_match, "passed": passed, "note": note,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
