# siglip-text — SigLIP text tower (Apache-2.0)

The text branch of `google/siglip-base-patch16-224`, served by `internal/models/siglip`
(architecture `siglip-text`). It exists so a head B distilled from SigLIP can run without Python.

## Why a second text tower

Head B is a student of whichever model produced its distillation targets, and on words it has
never seen it cannot beat that teacher. Measured on five names held out of every trained
component (`paper/FINDINGS-2026-08.md` §9–§10):

| | held-out 5 | base 17 | COCO macro |
|---|---|---|---|
| CLIP-crop (today's teacher) | 65.3 | 72.4 | 66.4 |
| **SigLIP-crop** | **85.3** | 78.8 | **73.7** |
| head B distilled from CLIP, COCO crops | 46.7 | 54.1 | 77.2 |
| **head B distilled from SigLIP, in-domain crops** | **69.3** | 64.0 | 20.5 |
| **…plus supervision and a COCO mixture** | 62.7 | **78.8** | 72.9 |

Raising the teacher is the only lever that moves the ceiling on unseen words.

The projection side needed no change: `d_text` is a field of `proj.bin`, and
`internal/models/textalign/dim_test.go` exercises Fold, ProjNorm and the gated decode at 512, 768
and 1152. The tokenizer was the only missing piece.

## The tokenizer, and its boundary

SigLIP uses SentencePiece **Unigram** (32 000 pieces, byte fallback) where CLIP uses byte-level
BPE, so `internal/models/siglip/tokenizer.go` is a fresh implementation: normalise → Metaspace →
Viterbi over piece log-probabilities → `</s>`.

**Verified token-for-token against the reference on 525 strings** — every COCO and ETRI class
name, all ten prompt templates applied to them, plus empty/whitespace/punctuation-only/very long
shapes. `go test ./internal/models/siglip/` reports `525 strings, 0 mismatches`. Regenerate the
reference with `scratchpad/ovclean/dump_siglip_tokens.py`.

Two things worth knowing before touching it:

- **The two reference tokenizers disagree.** `tokenizer.json`'s normalizer omits `/`, `<` and `>`
  from the punctuation it strips; the slow `SiglipTokenizer` strips all of Python's
  `string.punctuation`. On `"a photo of a N/A."` the fast path keeps the slash and the slow path
  drops it — 16 of the 525 strings. We follow the **slow** one, because
  `AutoProcessor.from_pretrained(...).tokenizer` returns it when sentencepiece is installed and
  it is therefore what produced the embeddings head B was distilled against.
- **Non-ASCII input is refused, not guessed.** The reference normalizer ends with a Precompiled
  charsmap (a compiled trie shipped as base64) that this package does not implement. It is the
  identity on ASCII, so the implementation is exact there and returns an error elsewhere rather
  than emitting a plausible-but-different sequence.

## Getting the weights

Not committed (`*.onnx` is ignored). Export the text branch to ONNX with a static
`input_ids [N, 64]` input and a `[N, 768]` output; `tokenizer.json` is already here, copied from
the HuggingFace snapshot.

The manifest must declare `license: apache-2.0` — confirmed from the model card via the Hub API,
and inside the CLAUDE.md allowlist.

## Status

The tokenizer and the model wrapper are implemented, registered and tested. **No ONNX weights and
no manifest are committed yet**, so `visionserve list` does not advertise this model — a pull that
cannot succeed is worse than an absent entry. The head that would use it
(`scratchpad/ovclean/sigsup/heads/mix_*`) is trained but not exported either.
