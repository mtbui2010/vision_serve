#!/usr/bin/env python3
"""Export the fast-path distilled head (head.bin, VSTXALN1) as head.onnx.

WHY (vision_serve BUGS_TO_FIX.md #4, ovd-edge FINDINGS §15): the head is a 300 x 768 x 256 kernel
per request. In scalar Go it cost 56 ms direct and 19 ms folded; on the detector's provider it is
a rounding error. CLAUDE.md rule 2: all inference goes through ONNX Runtime.

WHAT THE GRAPH COMPUTES -- exactly the served Go math of internal/models/hybrid/fastpath.go,
written in its DIRECT form (the folded form is an optimisation of the same function, and the
reason for it -- scalar Go -- is gone):

    z      = f @ P^T                      [B, Q, d_text]   (P: row-major [d_text, d_feat])
    norm   = ||z||                        [B, Q, 1]        (Go: ProjNorm, sqrt(f^T P^T P f))
    denom  = norm if norm > 0 else 1      (Go: `inv := 1; if norm > 0 { inv = 1/norm }`)
    logits = Scale * (z @ T^T) / denom + Bias      [B, Q, C]   RAW logits, no sigmoid

Scale is applied EXACTLY ONCE (bug #7: textalign.Fold folds Scale in, and the fast path once
multiplied by it again, costing 9.2 mAP). Here nothing is folded, so the graph applies it once.

    inputs       query_feats  float32 [batch, queries, d_feat]  the detector's `query_feats`
                 text_embeds  float32 [classes, d_text]         L2-normalised SigLIP text rows,
                                                                UNSCALED (per request: an input)
    output       logits       float32 [batch, queries, classes]
    initializers P^T [d_feat, d_text], scale [], bias []        (fixed trained weights)

Usage:
    python3 export_head_onnx.py                 # head.bin -> head.onnx next to this script, verify
    python3 export_head_onnx.py --check-only    # verify an existing head.onnx (needs only
                                                # onnxruntime; use with an onnxruntime-gpu env to
                                                # also check the CUDA EP)

Verification: runs head.onnx through ORT and prints the max abs diff against a numpy
reimplementation of the Go FOLDED path (textalign.Fold + textalign.ProjNorm), in float32 as Go
does it, plus the float64 direct reference.
"""
import argparse
import hashlib
import os
import struct
import sys

import numpy as np

MAGIC = b"VSTXALN1"
HEADER = 32


def read_head(path):
    raw = open(path, "rb").read()
    if len(raw) < HEADER or raw[:8] != MAGIC:
        sys.exit(f"{path}: not a VSTXALN1 file")
    d_text, d_feat = struct.unpack_from("<II", raw, 8)
    scale, bias = struct.unpack_from("<ff", raw, 16)
    flags, reserved = struct.unpack_from("<II", raw, 24)
    want = HEADER + 4 * d_text * d_feat
    if len(raw) != want:
        sys.exit(f"{path}: size {len(raw)} != {want}")
    if scale == 0:
        sys.exit(f"{path}: scale is 0")
    P = np.frombuffer(raw, dtype="<f4", offset=HEADER).reshape(d_text, d_feat).astype(np.float32)
    return dict(P=P, scale=np.float32(scale), bias=np.float32(bias), flags=flags,
                d_text=d_text, d_feat=d_feat, sha256=hashlib.sha256(raw).hexdigest())


def build(head, out_path, graph_name="fastpath_head", source="head.bin (VSTXALN1)",
          producer="visionserve/export_head_onnx.py"):
    """Write the head graph. models/rfdetr-textalign-dec1-siglip/export_head_onnx.py reuses it for
    textalign's proj.bin: the same VSTXALN1 object and the same function, with its own labels."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    d_text, d_feat = head["d_text"], head["d_feat"]
    inits = [
        numpy_helper.from_array(np.ascontiguousarray(head["P"].T), "P_T"),  # [d_feat, d_text]
        numpy_helper.from_array(np.array(head["scale"], dtype=np.float32), "scale"),
        numpy_helper.from_array(np.array(head["bias"], dtype=np.float32), "bias"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one"),
        numpy_helper.from_array(np.array([-1], dtype=np.int64), "last_axis"),
    ]
    nodes = [
        helper.make_node("MatMul", ["query_feats", "P_T"], ["z"]),              # [B,Q,dT]
        helper.make_node("Mul", ["z", "z"], ["z2"]),
        helper.make_node("ReduceSum", ["z2", "last_axis"], ["ss"], keepdims=1),  # [B,Q,1]
        helper.make_node("Sqrt", ["ss"], ["norm"]),
        helper.make_node("Greater", ["norm", "zero"], ["pos"]),
        helper.make_node("Where", ["pos", "norm", "one"], ["denom"]),
        helper.make_node("Transpose", ["text_embeds"], ["text_T"], perm=[1, 0]),  # [dT,C]
        helper.make_node("MatMul", ["z", "text_T"], ["dots"]),                  # [B,Q,C]
        helper.make_node("Div", ["dots", "denom"], ["cos"]),
        helper.make_node("Mul", ["cos", "scale"], ["scaled"]),  # Scale exactly once (bug #7)
        helper.make_node("Add", ["scaled", "bias"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes, graph_name,
        inputs=[
            helper.make_tensor_value_info("query_feats", TensorProto.FLOAT,
                                          ["batch", "queries", d_feat]),
            helper.make_tensor_value_info("text_embeds", TensorProto.FLOAT,
                                          ["classes", d_text]),
        ],
        outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT,
                                               ["batch", "queries", "classes"])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)],
                              producer_name=producer)
    model.ir_version = 8
    for k, v in [("source", source),("source_sha256", head["sha256"]),
                 ("d_text", str(d_text)), ("d_feat", str(d_feat)),
                 ("scale", repr(float(head["scale"]))), ("bias", repr(float(head["bias"]))),
                 ("flags", str(head["flags"]))]:
        e = model.metadata_props.add()
        e.key, e.value = k, v
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, out_path)
    print(f"wrote {out_path}: P [{d_text}x{d_feat}] scale={float(head['scale']):.6f} "
          f"bias={float(head['bias']):.6f} flags={head['flags']} sha256({source.split()[0]})={head['sha256'][:16]}")


def go_folded(head, feats, text):
    """float32 reimplementation of fastpath.go: textalign.Fold + ProjNorm + dot*inv + Bias."""
    P, scale, bias = head["P"], head["scale"], head["bias"]
    W = ((text * scale).astype(np.float32) @ P).astype(np.float32)          # Fold: [C, d_feat]
    G = (P.astype(np.float64).T @ P.astype(np.float64)).astype(np.float32)  # Gram, f64-accum
    f = feats[0]
    ss = np.einsum("qi,ij,qj->q", f.astype(np.float64), G.astype(np.float64), f.astype(np.float64))
    norm = np.where(ss > 0, np.sqrt(np.maximum(ss, 0)), 0).astype(np.float32)
    inv = np.where(norm > 0, 1 / np.where(norm > 0, norm, 1), 1).astype(np.float32)
    return ((f @ W.T) * inv[:, None] + bias).astype(np.float32)[None]


def direct_f64(head, feats, text):
    P = head["P"].astype(np.float64)
    z = feats[0].astype(np.float64) @ P.T
    n = np.linalg.norm(z, axis=-1, keepdims=True)
    n = np.where(n > 0, n, 1)
    return (float(head["scale"]) * (z / n) @ text.astype(np.float64).T + float(head["bias"]))[None]


def verify(head, onnx_path):
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    Q, C = 300, 5
    feats = rng.standard_normal((1, Q, head["d_feat"])).astype(np.float32)
    feats[0, 7] = 0  # the degenerate row: Go gives exactly Bias there
    text = rng.standard_normal((C, head["d_text"])).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)

    ref_go = go_folded(head, feats, text)
    ref64 = direct_f64(head, feats, text)
    print(f"numpy Go-folded vs float64 direct: max abs diff {np.abs(ref_go - ref64).max():.3e}")
    ok = True
    for prov in ["CPUExecutionProvider", "CUDAExecutionProvider"]:
        if prov not in ort.get_available_providers():
            print(f"{prov}: not available in this onnxruntime build, skipped")
            continue
        s = ort.InferenceSession(onnx_path, providers=[prov])
        if s.get_providers()[0] != prov:
            print(f"{prov}: fell back to {s.get_providers()[0]}, skipped")
            continue
        out = s.run(["logits"], {"query_feats": feats, "text_embeds": text})[0]
        assert out.shape == (1, Q, C), out.shape
        d_go = np.abs(out - ref_go).max()
        d64 = np.abs(out - ref64).max()
        rank = (out[0].argmax(0) == ref_go[0].argmax(0)).all()
        print(f"{prov}: ORT vs numpy Go-folded max abs diff {d_go:.3e}; vs float64 direct "
              f"{d64:.3e}; logit range [{out.min():.3f}, {out.max():.3f}]; "
              f"degenerate row == bias: {np.allclose(out[0, 7], head['bias'])}; "
              f"per-word argmax query identical: {rank}")
        ok &= d_go < 1e-3 and np.allclose(out[0, 7], head["bias"])
    if not ok:
        sys.exit("VERIFY FAILED")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=os.path.join(here, "head.bin"))
    ap.add_argument("--out", default=os.path.join(here, "head.onnx"))
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    head = read_head(a.head)
    if not a.check_only:
        build(head, a.out)
    verify(head, a.out)


if __name__ == "__main__":
    main()
