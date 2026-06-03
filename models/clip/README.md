# CLIP Image Encoder

**License:** MIT (OpenAI)  
**Architecture:** ViT-B/32  
**Output:** 512-dimensional L2-normalised embedding

## What is CLIP?

CLIP (Contrastive Language-Image Pre-Training) is a model from OpenAI that learns visual concepts
from natural language supervision. It can encode both images and text into a shared embedding space,
enabling zero-shot tasks without task-specific training data.

## V1: Image Encoder Only

This implementation provides the **image encoder only** (ViT-B/32 vision transformer).
The result is a 512-d float32 vector, L2-normalised so that cosine similarity equals a plain dot product.

Text encoder is deferred to v2 — it requires a BPE tokenizer which adds significant complexity.

## Getting the ONNX File

### Option A — pull with VisionServe (recommended)

```bash
make pull MODEL=clip    # downloads visual.onnx from HuggingFace
```

### Option B — download manually

```bash
hf download khasinski/clip-ViT-B-32-onnx visual.onnx \
    --local-dir models/clip/
mv models/clip/visual.onnx models/clip/model.onnx
```

HuggingFace repo: [`khasinski/clip-ViT-B-32-onnx`](https://huggingface.co/khasinski/clip-ViT-B-32-onnx)  
File: `visual.onnx` → saved locally as `model.onnx`

Place it at `models/clip/model.onnx`.

## Verified I/O

| Tensor | Shape | Dtype | Notes |
|--------|-------|-------|-------|
| `input` (in) | `[batch, 3, 224, 224]` | float32 | NCHW, CLIP-normalized |
| `output` (out) | `[batch, 512]` | float32 | L2-normalized embedding |

## Use Cases

- **Zero-shot image classification:** Compare the image embedding against a library of
  precomputed text embeddings (e.g., "a photo of a cat", "a photo of a dog"). The class
  with the highest dot product score wins.
- **Image similarity / visual search:** Embed query and database images; find nearest neighbours.
- **Cross-modal retrieval:** Embed images and text independently; rank by similarity.

## Preprocessing

CLIP uses a slightly different ImageNet normalization than standard models:

```
mean = [0.48145466, 0.4578275, 0.40821073]
std  = [0.26862954, 0.26130258, 0.27577711]
```

Input is resized to 224×224 with squash (no letterbox) before normalization. This matches
how CLIP was originally trained.

## Note on Text Encoder

The text encoder is not yet implemented. Encoding text requires a BPE (Byte-Pair Encoding)
tokenizer, which is planned for v2. In the meantime, you can precompute text embeddings
externally (e.g., using the Python `transformers` library) and store them for comparison.

## Performance

Measured on NVIDIA RTX A6000 (48 GB VRAM), VisionServe Go HTTP server, 20 warm requests.

| Metric | Value |
|--------|-------|
| p50 latency (end-to-end HTTP) | 33 ms |
| p95 latency | 69 ms |
| Inference only (srv p50) | 12 ms |
| Throughput | 27.9 RPS |
| VRAM (GPU) | 810 MB |
| ONNX size | 335 MB |
| Cold-start | 5.8 s |

Fastest end-to-end in the catalog. Inference is 12 ms; the 21 ms overhead is Go preprocess (resize+normalize+CHW).
