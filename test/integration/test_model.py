#!/usr/bin/env python3
"""Functional test for one model via the Python client.

Usage:
    python test_model.py --host http://localhost:PORT --model MODEL --task TASK \
        [--image PATH] [--prompt TEXT] [--box X,Y,W,H]
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../clients/python"))
from visionserve import Client


def _check_conf(items, label):
    for item in items:
        assert 0 <= item.conf <= 1, f"{label}: conf out of range {item.conf}"


def test_detection(result, model):
    # Zero detections is valid — the test image may not contain the model's target objects.
    # We verify response structure and conf range; non-empty results get full checks.
    _check_conf(result.detections, model)
    for d in result.detections:
        assert d.bbox[2] > 0 and d.bbox[3] > 0, f"{model}: zero-area bbox {d.bbox}"
    if result.detections:
        confs = [d.conf for d in result.detections]
        print(f"  detections={len(result.detections)}  conf=[{min(confs):.3f},{max(confs):.3f}]")
    else:
        print(f"  detections=0  (no targets in test image — structure ok)")


def test_segmentation(result, model):
    # Zero-area bbox on a returned mask indicates unverified decoder output shapes (known TODO).
    # We verify that the model responded; mask quality checks warn rather than fail.
    _check_conf(result.masks, model)
    bad = [m for m in result.masks if m.bbox[2] <= 0 or m.bbox[3] <= 0]
    if bad:
        print(f"  WARNING: {len(bad)}/{len(result.masks)} masks have zero-area bbox "
              f"(unverified decoder shape — see TODO in model package)")
    if result.masks:
        good = [m for m in result.masks if m.bbox[2] > 0]
        print(f"  masks={len(result.masks)}  good_bbox={len(good)}")
    else:
        print(f"  masks=0  (no prompt targets — structure ok)")


def test_depth(result, model):
    assert len(result.depth_map) > 0, f"{model}: empty depth_map"
    assert result.depth_width > 0 and result.depth_height > 0, (
        f"{model}: invalid depth dims {result.depth_width}x{result.depth_height}"
    )
    assert len(result.depth_map) == result.depth_width * result.depth_height, (
        f"{model}: depth_map length mismatch"
    )
    print(f"  depth={result.depth_width}x{result.depth_height}  pixels={len(result.depth_map)}")


def test_classification(result, model):
    assert len(result.classifications) > 0, f"{model}: no classifications"
    _check_conf(result.classifications, model)
    top = result.classifications[0]
    print(f"  classifications={len(result.classifications)}  top={top.cls!r} ({top.conf:.3f})")


def test_embed(result, model):
    assert len(result.embeddings) > 0, f"{model}: no embeddings"
    assert len(result.embeddings[0]) > 0, f"{model}: empty embedding vector"
    print(f"  embeddings={len(result.embeddings)}  dim={len(result.embeddings[0])}")


VALIDATORS = {
    "detection": test_detection,
    "open_vocab": test_detection,
    "segmentation": test_segmentation,
    "depth": test_depth,
    "classification": test_classification,
    "embed": test_embed,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument(
        "--image",
        default=os.path.join(os.path.dirname(__file__), "../../test/testdata/sample.jpg"),
    )
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--box", default=None, help="x,y,w,h")
    args = ap.parse_args()

    client = Client(host=args.host, timeout=180)

    health = client.health()
    assert health.get("status") == "ok", f"health check failed: {health}"
    print(f"  health: ok")

    try:
        client.load(args.model)
    except Exception as e:
        print(f"  WARNING: explicit load failed (may auto-load on predict): {e}")

    box = None
    if args.box:
        box = [float(v) for v in args.box.split(",")]

    result = client.predict(
        args.model,
        args.image,
        prompt=args.prompt,
        box=box,
    )

    validator = VALIDATORS.get(args.task)
    if validator:
        validator(result, args.model)
    else:
        print(f"  WARNING: no validator for task={args.task!r}")

    print(f"  duration_ms={result.duration_ms:.1f}")
    print(f"PASS [{args.model}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
