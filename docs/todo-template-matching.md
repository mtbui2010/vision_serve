# TODO: Template Matching Approaches

Template (instance) detection is tracked under `api.TaskInstanceDetection`.  All models
in this category implement `models.PipelineModel` and read their template images from
`models.Prompt.TemplateImages` (populated by the lifecycle manager from the template store).

---

## Implemented

- [x] **OWL-ViT / OWLv2** (one-shot detection) — `internal/models/owlvit/`
  - License: Apache-2.0 (Google Research)
  - Architecture: single ONNX session, image-conditioned; scene + N templates → boxes
  - ONNX export: see `docs/owlvit-export.md`
  - Best for: general one-shot detection, works across viewpoints, GPU-friendly

---

## Planned

### SiamRPN / SiamFC (Siamese tracking-as-detection)

- **Architecture**: template encoder + query encoder + cross-correlation head → bbox regression
- **License**: MIT (open academic implementations; verify each checkpoint)
- **ONNX strategy**: split into template encoder + query+correlation head (two roles)
- **Best for**: real-time detection at the same viewpoint (tracking-style), very fast on CPU
- **TODO**: `internal/models/siamrpn/`
- **Open question**: verify tensor shapes from a real ONNX export; siamrpn and siamfc
  have different head architectures.

### LoFTR (detector-free keypoint matching)

- **Architecture**: template + query → dense feature maps → self/cross-attention → keypoints
  → RANSAC homography → bounding box from matched corners
- **License**: MIT (zju3dv/LoFTR)
- **ONNX strategy**: single session (encoder + transformer backbone); RANSAC post-process in Go
- **Best for**: textured rigid objects (PCBs, books, tools) with large viewpoint changes
- **TODO**: `internal/models/loftr/`
- **Open question**: ONNX export of the full LoFTR pipeline needs an indoor/outdoor
  checkpoint audit; runtime RANSAC must be pure-Go (no OpenCV).

### DINOv2 dense similarity (zero-shot)

- **Architecture**: template → patch embedding; query → dense patch embeddings → cosine
  similarity heatmap → peak detection → bounding box
- **License**: Apache-2.0 (Meta AI — DINOv2 already on the VisionServe roadmap as encoder)
- **ONNX strategy**: reuse DINOv2 encoder (single role); similarity + peak detection in Go
- **No explicit bbox**: requires cluster/peak detection post-process (e.g. connected
  components on the thresholded heatmap)
- **Best for**: zero-shot (no fine-tuning), any object with strong DINOv2 feature response
- **TODO**: `internal/models/dinosim/` (or as a variant of the embed model)

### ORB + RANSAC (classical, no ONNX)

- **Architecture**: ORB keypoint extraction + descriptor matching + RANSAC homography →
  template corners projected into scene → bounding box
- **License**: BSD (OpenCV ORB) or implement pure-Go ORB from scratch
- **ONNX strategy**: none — classical algorithm; implement directly in Go image ops
- **Best for**: edge devices with no GPU, rigid texture-rich objects under mild viewpoint
  change, ultra-low latency
- **Constraint**: avoid cgo/OpenCV unless strictly necessary (CLAUDE.md rule #4); a
  pure-Go ORB may be the right path
- **TODO**: `internal/models/orbmatch/`
