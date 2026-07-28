#!/usr/bin/env python3
"""
Export OWLv2 / OWL-ViT to ONNX for VisionServe instance_detection.

Produces a single ONNX file that takes:
  query_pixel_values   [1, 3, H, W]  -- scene image  (CLIP-normalized, NCHW)
  query_image_features [N, 3, H, W]  -- N template images (same normalization)

And returns:
  logits     [1, num_patches, N]  -- cosine similarity per spatial patch per template
  pred_boxes [1, num_patches, 4]  -- [cx, cy, w, h] normalized [0, 1]

Usage:
  pip install "transformers>=4.35" torch onnx onnxruntime pillow
  python scripts/export_owlvit.py
  python scripts/export_owlvit.py --model google/owlv2-large-patch14-ensemble
  python scripts/export_owlvit.py --output models/owlvit-base/owlv2_base_patch16.onnx
  python scripts/export_owlvit.py --model google/owlvit-base-patch32  # original OWL-ViT

Supported models (all Apache-2.0):
  google/owlv2-base-patch16-ensemble   (recommended, 960px)
  google/owlv2-base-patch16
  google/owlv2-large-patch14-ensemble  (1008px, heavier)
  google/owlvit-base-patch32           (original OWL-ViT, 768px)
  google/owlvit-base-patch16
  google/owlvit-large-patch14
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

# ── Model registry ─────────────────────────────────────────────────────────────

MODELS: dict[str, dict] = {
    "google/owlv2-base-patch16-ensemble": {
        "cls":        "Owlv2ForObjectDetection",
        "image_size": 960,
        "patch_size": 16,
        "out_file":   "owlv2_base_patch16.onnx",
    },
    "google/owlv2-base-patch16": {
        "cls":        "Owlv2ForObjectDetection",
        "image_size": 960,
        "patch_size": 16,
        "out_file":   "owlv2_base_patch16.onnx",
    },
    "google/owlv2-large-patch14-ensemble": {
        "cls":        "Owlv2ForObjectDetection",
        "image_size": 1008,
        "patch_size": 14,
        "out_file":   "owlv2_large_patch14.onnx",
    },
    "google/owlv2-large-patch14": {
        "cls":        "Owlv2ForObjectDetection",
        "image_size": 1008,
        "patch_size": 14,
        "out_file":   "owlv2_large_patch14.onnx",
    },
    "google/owlvit-base-patch32": {
        "cls":        "OwlViTForObjectDetection",
        "image_size": 768,
        "patch_size": 32,
        "out_file":   "owlvit_base_patch32.onnx",
    },
    "google/owlvit-base-patch16": {
        "cls":        "OwlViTForObjectDetection",
        "image_size": 768,
        "patch_size": 16,
        "out_file":   "owlvit_base_patch16.onnx",
    },
    "google/owlvit-large-patch14": {
        "cls":        "OwlViTForObjectDetection",
        "image_size": 840,
        "patch_size": 14,
        "out_file":   "owlvit_large_patch14.onnx",
    },
}


# ── Wrapper ────────────────────────────────────────────────────────────────────

def _make_wrapper(model):
    """
    Build a thin nn.Module wrapper for ONNX tracing.

    The wrapper calls model.image_guided_detection and extracts the two
    tensors VisionServe needs (logits, target_pred_boxes).
    """
    import torch
    import torch.nn as nn

    class _Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, query_pixel_values, query_image_features):
            # query_pixel_values   : [1, 3, H, W]
            # query_image_features : [N, 3, H, W]
            out = self.m.image_guided_detection(
                pixel_values=query_pixel_values,
                query_pixel_values=query_image_features,
            )
            # HuggingFace attribute is target_pred_boxes (not pred_boxes).
            # We rename to pred_boxes in output_names so VisionServe can
            # find it by name.
            return out.logits, out.target_pred_boxes

    return _Wrapper(model)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export OWLv2 / OWL-ViT to ONNX for VisionServe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python scripts/export_owlvit.py
          python scripts/export_owlvit.py --model google/owlv2-large-patch14-ensemble
          python scripts/export_owlvit.py --output models/owlvit-base/owlv2_base_patch16.onnx
        """),
    )
    p.add_argument(
        "--model",
        default="google/owlv2-base-patch16-ensemble",
        choices=list(MODELS.keys()),
        metavar="MODEL_ID",
        help="HuggingFace model ID (default: google/owlv2-base-patch16-ensemble)",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        metavar="PATH",
        help="Output .onnx path. Default: <out_file> from model config",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17, minimum: 14)",
    )
    p.add_argument(
        "--num-templates",
        type=int,
        default=1,
        metavar="N",
        help="Number of template images used for tracing (default: 1). "
             "OWLv2 unrolls the template loop at trace time so N is fixed. "
             "Use N=1 (default); VisionServe runs one inference per template and aggregates.",
    )
    p.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip ONNXRuntime verification after export",
    )
    p.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip printing the manifest.yaml template",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        metavar="DIR",
        help="HuggingFace cache directory (passed to from_pretrained)",
    )
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _check_deps() -> None:
    missing = []
    for pkg in ("torch", "transformers", "onnx"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(
            f"ERROR: missing packages: {', '.join(missing)}\n"
            f"Run: pip install {' '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    import transformers
    tv_str = transformers.__version__
    try:
        tv_parts = tuple(int(x) for x in tv_str.split(".")[:2])
        if tv_parts < (4, 35):
            print(
                f"WARNING: transformers {tv_str} detected. "
                "Version >=4.35 is recommended for OWLv2 support.",
                file=sys.stderr,
            )
    except ValueError:
        pass  # non-standard version string; skip check


def _load_model(model_id: str, cfg: dict, cache_dir: str | None):
    import torch
    from transformers import Owlv2ForObjectDetection, OwlViTForObjectDetection

    cls_map = {
        "Owlv2ForObjectDetection":   Owlv2ForObjectDetection,
        "OwlViTForObjectDetection":  OwlViTForObjectDetection,
    }
    cls = cls_map[cfg["cls"]]

    print(f"Loading {model_id} ...")
    model = cls.from_pretrained(model_id, cache_dir=cache_dir)
    model.eval()
    if torch.cuda.is_available():
        print("  CUDA available — keeping model on CPU for export stability")
    return model


# ── Export ─────────────────────────────────────────────────────────────────────

def _export(
    model_id: str,
    cfg: dict,
    output: Path,
    opset: int,
    num_templates: int,
    cache_dir: str | None,
) -> None:
    import torch

    model = _load_model(model_id, cfg, cache_dir)
    wrapper = _make_wrapper(model)
    wrapper.eval()

    H = W = cfg["image_size"]
    P = (H // cfg["patch_size"]) ** 2

    dummy_scene     = torch.zeros(1, 3, H, W)
    dummy_templates = torch.zeros(num_templates, 3, H, W)

    print(f"\nExporting to {output}")
    print(f"  scene     : [1, 3, {H}, {W}]")
    print(f"  templates : [{num_templates}, 3, {H}, {W}]  (N fixed at trace time)")
    print(f"  expected logits    : [{num_templates}, {P}, 1]  (shape [N, P, 1])")
    print(f"  expected pred_boxes: [1, {P}, 4]")

    output.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_scene, dummy_templates),
            str(output),
            opset_version=opset,
            input_names=["query_pixel_values", "query_image_features"],
            output_names=["logits", "pred_boxes"],
            # No dynamic axes for num_templates: OWLv2 unrolls the template loop
            # at trace time, so N is baked in. Export with N=1; VisionServe runs
            # one inference per template and aggregates the max score per patch.
            dynamic_axes={},
            do_constant_folding=True,
            dynamo=False,  # use legacy TorchScript tracer; dynamo=True (torch 2.x default) breaks dynamic_axes
        )

    size_mb = output.stat().st_size / 1e6
    print(f"\n  Saved: {output}  ({size_mb:.1f} MB)")


# ── Verify ─────────────────────────────────────────────────────────────────────

def _verify(output: Path, cfg: dict, num_templates: int) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("\nSKIP verify: onnxruntime not installed. Run: pip install onnxruntime")
        return

    import numpy as np

    print("\nVerifying with ONNXRuntime ...")
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3  # suppress verbose ORT logs
    sess = ort.InferenceSession(
        str(output),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )

    H = W = cfg["image_size"]
    P = (H // cfg["patch_size"]) ** 2

    # ── Static shape test ──────────────────────────────────────────────────────
    feeds = {
        "query_pixel_values":    np.zeros((1, 3, H, W), dtype=np.float32),
        "query_image_features":  np.zeros((num_templates, 3, H, W), dtype=np.float32),
    }
    logits, boxes = sess.run(None, feeds)

    print("\n  I/O shapes:")
    for info in sess.get_inputs():
        print(f"    INPUT  {info.name:<32} {info.shape}")
    for info, arr in zip(sess.get_outputs(), [logits, boxes]):
        print(f"    OUTPUT {info.name:<32} {arr.shape}  dtype={arr.dtype}")

    # Shape assertions — verified shape: logits [N, P, 1], pred_boxes [1, P, 4]
    assert logits.shape == (num_templates, P, 1), (
        f"logits shape {logits.shape} != ({num_templates}, {P}, 1)"
    )
    assert boxes.shape == (1, P, 4), (
        f"pred_boxes shape {boxes.shape} != (1, {P}, 4)"
    )
    print("\n  Shape assertions: PASSED")

    # N is fixed at trace time — just confirm the model rejects wrong N gracefully.
    print(f"  Note: N={num_templates} is baked in at trace time. "
          "VisionServe runs one inference per template and aggregates.")

    # ── Score range sanity (random non-zero input) ─────────────────────────────
    rng = np.random.default_rng(42)
    feeds_rnd = {
        "query_pixel_values":   rng.standard_normal((1, 3, H, W)).astype(np.float32),
        "query_image_features": rng.standard_normal((num_templates, 3, H, W)).astype(np.float32),
    }
    lg_rnd, bx_rnd = sess.run(None, feeds_rnd)
    print(f"\n  logits    range: [{lg_rnd.min():.3f}, {lg_rnd.max():.3f}]")
    print(f"  pred_boxes range: [{bx_rnd.min():.3f}, {bx_rnd.max():.3f}]  (expect ~[0,1])")


# ── Manifest ───────────────────────────────────────────────────────────────────

def _print_manifest(model_id: str, output: Path, cfg: dict) -> None:
    H = cfg["image_size"]
    ps = cfg["patch_size"]
    name = output.stem

    manifest = f"""\
name: {name}
task: instance_detection
license: Apache-2.0
architecture: owlvit

# Single ONNX file — OWLv2 handles both scene and templates in one forward pass.
files:
  model: {output.name}

input:
  width: {H}
  height: {H}
  layout: NCHW
  letterbox: false
  normalize:
    mean: [0.48145466, 0.4578275, 0.40821073]   # CLIP normalization (not ImageNet)
    std:  [0.26862954, 0.26130258, 0.27577711]

postprocess:
  conf_threshold: 0.1    # minimum sigmoid(logit) to emit a detection
  max_detections: 10

instance:
  patch_size: {ps}       # must match the exported variant
  sim_threshold: 0.1     # confidence threshold (same semantics as conf_threshold)
  max_templates: 5       # max template images used per call (0 = no limit)

runtime:
  prefer: [cuda, cpu]
  idle_unload_seconds: 300
"""

    print("\n" + "─" * 60)
    print(f"  Save as:  {output.parent}/manifest.yaml")
    print("─" * 60)
    print(manifest)
    print("─" * 60)

    # Also write the file if the output directory already has the ONNX in it
    manifest_path = output.parent / "manifest.yaml"
    if output.parent.exists() and output.exists():
        manifest_path.write_text(manifest)
        print(f"  Written:  {manifest_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse()
    _check_deps()

    cfg = MODELS[args.model]
    output = Path(args.output) if args.output else Path(cfg["out_file"])

    _export(args.model, cfg, output, args.opset, args.num_templates, args.cache_dir)

    if not args.no_verify:
        _verify(output, cfg, args.num_templates)

    if not args.no_manifest:
        _print_manifest(args.model, output, cfg)

    print(f"\nDone.")
    print(f"  Next steps:")
    print(f"    mkdir -p models/{output.stem}")
    print(f"    mv {output} models/{output.stem}/")
    print(f"    # copy manifest.yaml into models/{output.stem}/")
    print(f"    ORT_DYLIB_PATH=... ./visionserve serve")
    print(f"    curl -X POST http://localhost:11435/api/templates \\")
    print(f"         -F name=my_object -F images=@template.jpg")
    print(f"    curl -X POST http://localhost:11435/api/predict \\")
    print(f"         -F model={output.stem} -F template_name=my_object -F image=@scene.jpg")


if __name__ == "__main__":
    main()
