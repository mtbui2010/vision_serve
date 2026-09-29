# clip-text — CLIP ViT-B/32 TEXT tower

The missing half of [`../clip`](../clip), which ships the **image** tower only. Same
checkpoint (`openai/clip-vit-base-patch32`, **MIT**), same 512-d embedding space, so a
text embedding from here and an image embedding from `clip` can be compared with a plain
dot product (both are L2-normalised in Go).

| | |
|---|---|
| architecture | `clip-text` (PipelineModel, single session, text-prompted) |
| input | `input_ids` `[batch, 77]` int64 |
| output | `text_embeds` `[batch, 512]` float32 |
| tokenizer | pure Go BPE — [`internal/models/clip/tokenizer.go`](../../internal/models/clip/tokenizer.go) |
| files | `model.onnx` (244 MB), `vocab.json`, `merges.txt` |

## Why

Two things needed text embeddings and were blocked on them:

1. **A text-aligned detection head.** It scores object query `i` as
   `cosine(P · f_i, text_embedding(class))`, so it needs an embedding per class name to
   build its output matrix. (`f_i` comes from
   [`../rfdetr-small-etri-qf`](../rfdetr-small-etri-qf)'s `query_feats`.)
2. **Pseudo-label verification** — crop a candidate region, embed it with `clip`, embed
   the candidate label text here, and keep the label only if the cosine is convincing.

## Same embedding space — verified, not assumed

This is the check that catches a wrong projection, wrong pooling (CLIP pools at the
`<|endoftext|>` position) or missing normalisation. Both numbers below come from actual
runs, not from the model card.

**1. The shipped image tower *is* this checkpoint's vision tower.**
`../clip/model.onnx` vs HuggingFace `openai/clip-vit-base-patch32`
`CLIPVisionModelWithProjection` on 8 real ETRI images:

- per-image cosine = **1.000000** (all 8)
- max |Δ| on the normalised vectors = **1.7e-06**

So the two towers provably come from one checkpoint, and their spaces are the same.

**2. Zero-shot crop → class name, end to end.** 373 ground-truth object crops from 60
ETRI tabletop images, 22 candidate classes, prompt `"a photo of a {class}"`, argmax
cosine:

| | |
|---|---|
| **top-1** | **61.9 %** (231/373) |
| top-5 | 93.0 % |
| chance | 4.5 % |
| mean cosine, correct class | 0.282 |
| mean cosine, all pairs | 0.229 |

Measured twice and identical both ways: once in Python (onnxruntime), once **through the
Go server** (`POST /api/predict` to `clip` for the crops and to `clip-text` for the class
names) — which also confirms the pure-Go tokenizer agrees with HuggingFace.

Bare class names instead of the template score 57.6 %. Residual errors are dominated by
genuinely ambiguous label pairs in this dataset (`coffee` vs `coffee can`, `beer can` vs
the other cans) — not by a space mismatch, which would have pinned accuracy at chance.

## Usage

```bash
# The image argument is IGNORED (this is the text branch) — the CLI always wants one.
visionserve run --models ./models clip-text any.png --prompt "cup. remote. water bottle."
```

```bash
curl -s localhost:11435/api/predict -H 'Content-Type: application/json' -d '{
  "model": "clip-text",
  "image_base64": "<any 1x1 png>",
  "prompt": "a photo of a cup. a photo of a remote."
}'
```

The prompt is split on `"."` into phrases (the same convention GroundingDINO uses) and
`embeddings[i]` corresponds to phrase `i` — that ordering is the contract a class matrix
relies on. `clip.SplitPhrases` reproduces it in Go.

In-process (the expected path for training a text-aligned head):

```go
res, err := mgr.PredictPrompt("clip-text", nil, models.Prompt{Text: "cup. remote."})
// res.Embeddings[0] = 512-d L2-normalised vector for "cup"
```

## Provenance / how it was exported

- Source: `openai/clip-vit-base-patch32` (**MIT**), HuggingFace
  `CLIPTextModelWithProjection` — i.e. text transformer + `text_projection`, which is the
  half that lands in the shared space. `CLIPTextModel` alone would **not**.
- Exported with `torch.onnx.export(..., dynamo=True)` (torch 2.10, transformers 5.9).
  The legacy TorchScript exporter is what fails on some CLIP/GDINO ops; the dynamo path
  exported cleanly on the first try. Opset 18.
- Wrapper module takes `input_ids` only and returns `.text_embeds`. **No `attention_mask`
  input by design**: the text transformer is causally masked and pooling happens at the
  first `<|endoftext|>`, so padding after it cannot influence the pooled vector. Passing a
  mask would be a no-op — one fewer input to get wrong.
- Dynamic batch axis; verified at batch 1, 2 and 3.
- ONNX output vs PyTorch: max |Δ| = **3.3e-06**.
- Saved without external data (`save_as_external_data=False`) so `model.onnx` is one
  self-contained file the `sha256` in `manifest.yaml` can pin.
- `vocab.json` / `merges.txt` copied verbatim from the same HuggingFace repo.

Tokenizer assets, for reference:

```
vocab.json  sha256 5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363
merges.txt  sha256 f526393189112391ce6f9795d4695f704121ce452c3aad1f5335cc41337eba85
```

## The tokenizer is the risky part

CLIP does **not** use WordPiece. `internal/models/clip/tokenizer.go` reimplements the
byte-pair encoder in pure Go (no Python at runtime, per CLAUDE.md): collapse whitespace →
lowercase → CLIP's split regex (note digits split one at a time: `"42"` → `4`,`2`) →
GPT-2 byte-level encoding → BPE merges with the `</w>` end-of-word suffix → wrap with
`<|startoftext|>`/`<|endoftext|>` → pad to 77 with `<|endoftext|>`.

It was cross-checked against `transformers` on **471 strings** (class names × 6 prompt
templates, plus contractions, punctuation, digits, emoji, CJK, accented and decomposed
text, embedded tabs/newlines and literal special tokens): **0 mismatches**.
`tokenizer_test.go` pins the exact ids for a representative subset.

One known deviation: HuggingFace applies Unicode **NFC** normalisation first and we do
not (it would add a `golang.org/x/text` dependency). Already-NFC input — which all ASCII
is — is unaffected; decomposed input (`"e"` + U+0301 rather than `"é"`) can differ.

## Publishing so `visionserve pull` can fetch it

`model.onnx` is **not committed** (244 MB). To make this model pullable:

1. Upload `model.onnx`, `vocab.json` and `merges.txt` to a public HuggingFace repo (or
   point at `openai/clip-vit-base-patch32` for the two tokenizer files, which are already
   there — the ONNX itself is not).
2. Point `source_url` in `manifest.yaml` at the uploaded ONNX (`sha256.model` already pins
   these exact bytes).
3. Add an entry to [`internal/catalog/catalog.go`](../../internal/catalog/catalog.go):

```go
{
    Name:         "clip-text",
    Task:         "embed",
    License:      "MIT",
    Architecture: "clip-text",
    Description:  "CLIP ViT-B/32 text tower — 512-d text embeddings in the clip image-tower space",
    HFRepo:       "<your-org>/clip-vit-base-patch32-text-onnx",
    Files: []File{
        {Role: "model", HFFilename: "model.onnx", LocalFilename: "model.onnx"},
        {Role: "vocab", HFFilename: "vocab.json", LocalFilename: "vocab.json"},
        {Role: "merges", HFFilename: "merges.txt", LocalFilename: "merges.txt"},
    },
    InputWidth: 77, InputHeight: 1, InputLayout: "NCHW",
    PostprocessType:   "embed",
    RuntimePrefer:     []string{"cuda", "cpu"},
    IdleUnloadSeconds: 300,
    Verified:          true,
},
```

As with `rfdetr-small-etri-qf`, the catalog entry is deliberately **not** added yet:
`visionserve list` would advertise the model as pullable and every `pull` would fail
until the repo exists.

## Reproduce

Export + both verification runs:
`scripts/export_onnx_clip_text.py`, `scripts/verify_clip_text.py`.
