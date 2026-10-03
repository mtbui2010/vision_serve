#!/usr/bin/env python3
"""Export textalign's projection (proj.bin, VSTXALN1) as head.onnx: method "exact" on ORT.

WHY: method "exact" (the default when a request names no method) scores 300 queries as
a·cos(P f, t) + b. In scalar Go that is ~22 ms per request, almost all of it the per-query
||P f|| (a 256x256 quadratic form per query, internal/models/textalign/head.go). CLAUDE.md rule 2
puts inference in ONNX Runtime; this is the same move BUGS_TO_FIX.md #4 made for the fast-path
head in ../rfdetr-gdino-fastpath.

IT IS THE SAME GRAPH. textalign's exact head and the fast-path head are the same function of the
same VSTXALN1 object:

    logits = Scale * <t, P f> / ||P f|| + Bias      (t: L2-normalised text row, UNSCALED)

so this script reuses ../rfdetr-gdino-fastpath/export_head_onnx.py's builder and verifier and only
changes the defaults and the labels. Graph contract (internal/models/textalign/headonnx.go):

    inputs   query_feats  float32 [batch, queries, d_feat]   the detector's `query_feats`
             text_embeds  float32 [classes, d_text]          per request, so an INPUT, not weights
    output   logits       float32 [batch, queries, classes]  RAW (no sigmoid), Scale applied once

The degenerate query (f = 0) gives exactly Bias in both the graph and the Go path. Go's guard is
"Gram-form ||P f||^2 <= 0", the graph's "sum of z^2 == 0"; they can only part when rounding makes
a positive ||P f||^2 look <= 0, which real detector features never approach (see the comment on the
contract in internal/models/textalign/headonnx.go for the measurement).

Only method "exact" runs on the graph. "folded", "gated" and "dual" still read proj.bin in Go
(they skip the norm, which is the expensive part), so proj.bin stays the source of truth and
head.onnx MUST be re-exported whenever proj.bin changes. The server checks one query per request
against proj.bin and refuses a head.onnx that disagrees with it.

Usage (from the repo root, with any Python that has numpy + onnx + onnxruntime):
    python3 models/rfdetr-textalign-dec1-siglip/export_head_onnx.py
        # proj.bin -> head.onnx next to this script, then verify
    python3 models/rfdetr-textalign-dec1-siglip/export_head_onnx.py \\
        --proj models/rfdetr-textalign-dec1/proj.bin --out models/rfdetr-textalign-dec1/head.onnx
        # any other textalign model directory
    python3 models/rfdetr-textalign-dec1-siglip/export_head_onnx.py --check-only

Then enable it in that directory's manifest.yaml:   files:  head: head.onnx
                                       together with  runtime:  threads: {head: 1}
The shipped rfdetr-textalign-* manifests already do; head.onnx itself is not committed and not on
the HF catalog, so this script is how a checkout gets it. The export is deterministic (same proj.bin
and same onnx package -> byte-identical head.onnx).
"""
import argparse
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.join(HERE, "..", "rfdetr-gdino-fastpath", "export_head_onnx.py")


def shared():
    spec = importlib.util.spec_from_file_location("fastpath_export_head_onnx", SHARED)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=os.path.join(HERE, "proj.bin"))
    ap.add_argument("--out", default=None, help="default: head.onnx next to --proj")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.proj)), "head.onnx")

    fp = shared()
    head = fp.read_head(a.proj)
    if not a.check_only:
        fp.build(head, out, graph_name="textalign_exact_head",
                 source=f"{os.path.basename(a.proj)} (VSTXALN1)",
                 producer="visionserve/textalign export_head_onnx.py")
    fp.verify(head, out)


if __name__ == "__main__":
    main()
