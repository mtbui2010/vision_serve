# Engineering rules — learned from bugs we actually shipped or nearly shipped

Every rule below exists because breaking it caused a real bug in this repository (refactor
2026-10 and the reviews around it). Each row names the guard that now catches a regression.
CLAUDE.md carries the one-line version; this file is the "why" and "where".

## 1. Inputs and limits

| Rule | The bug it prevents | Guard |
|---|---|---|
| Bound **every** external input before spending memory on it: bytes, decoded pixels (read the header with `image.DecodeConfig`), multipart parts, prompt tokens, cache entries. | A template upload could decode hundreds of GB before the store's limit was checked. | `TestTemplatesPixelBoundCheckedBeforeDecoding` |
| Bounds-check every length or offset read from untrusted bytes (model files, uploads). | The ONNX header reader panicked on a malformed file. | `TestONNXHeaderMalformed` |
| Validate floats as finite (`NaN`/`Inf`) wherever config or requests carry them. | `pad: .nan` passed manifest validation; NMS panicked on non-finite boxes. | `TestValidateRejectsNonFinite`, `TestNonFiniteBoxesOnTheGridPath`, `TestParsePromptRejectsNonFinite` |
| Cache keys are fixed-size (hash), and caches are bounded (LRU). | The text-embedding cache keyed on the raw word: long prompts grew RAM without bound. | `internal/pipeline/textembed_test.go` |
| Never hold a scarce slot (admission, session, lock) while waiting on the client or on I/O. Probe, then take the slot when the work can start. | A client trickling its upload held an admission slot; 32 slow uploads made a model answer 503 to everyone for 2 minutes. | `TestSlowUploadHoldsNoSlot`, `TestRefusedMultipartDoesNotReadUpload` |

## 2. Errors and status codes

| Rule | The bug it prevents | Guard |
|---|---|---|
| An error caused by the caller is typed (`lifecycle.ErrInvalidRequest`, `models.BadPrompt`, …) and maps to 4xx in **one** place (`server/errors.go statusOf`). Never let a caller mistake surface as 500. | Missing prompts, prompts for the wrong model, `/api/preprocess` refusals all answered 500. | `TestStatusOf`, `TestBadPrompt`, `TestPreprocessModelRefusalIsInvalidRequest` |
| Wrap with `%w`; never compare error strings. Tests that pin error text exist only where clients parse it. | — | `go vet`, review |
| No panics in normal paths; a malformed input is an error value. | Header parser and NMS panics above. | race + fuzz-style table tests |

## 3. Concurrency and lifecycle

| Rule | The bug it prevents | Guard |
|---|---|---|
| A shared in-flight operation (singleflight load) is cancelled only by its owner (Unload/Close), never by one waiter; a waiter that joins a cancelled call must not inherit its error. | A Load right after Unload joined the cancelled load and answered 500. | `TestLoadAfterUnloadDuringLoadLoadsAgain`, `internal/lifecycle/cancel_test.go` |
| Every wait honours the request `ctx`; never *start* work for a client that has left. Release every slot/lease on every path (`defer`). | Requests whose client had gone still waited for and ran inference. | `TestAdmitRefusesADoneContext`, `TestClientGoneWhileWaitingInRuntime` |
| ORT sessions are created and run on their own OS-locked worker goroutine; never call a session from an arbitrary goroutine under a mutex. | CUDA EP allocates per OS thread; the mutex version leaked GPU handles on every call. | `internal/engine` tests |
| Tests never depend on process-global state (warn-once maps, env); they must pass under `-count=5` and `-race`. | Warn-once tests passed once and failed on the second run. | `go test -race -count=5` |

## 4. Geometry and model I/O

| Rule | The bug it prevents | Guard |
|---|---|---|
| Record the **true per-axis** scale and pad of every resize, and map results back with it. Test extreme aspect ratios (10000×10). | SCRFD used one `det_scale` for both axes: faces on wide images shifted up to 4 px, panoramas completely. | `TestPostprocess_ExtremePanoramaMapsBack` |
| Carry an object's **identity**, not its position, across filtering and sorting. | `/api/explain` used the index of a sorted, filtered detection as a raw query index and explained the wrong object. | `internal/lifecycle/explain_query_test.go`, `internal/explain/target_test.go` |
| A manifest setting a model cannot honour is a load error, never silently ignored. | SAM/PaddleOCR silently ignored a `preprocess:` block. | `TestFixedByExportArchitecturesRefuseABlock` |
| Verify real tensor shapes before writing decode logic; pin output selection against the reference rule. | — | `TestSplitRFMatchesReference` |
| A derived artifact (exported head, cache, generated manifest) is checked against its source at load. | A `head.onnx` exported from a different `proj.bin` would have scored silently wrong. | `TestExactHeadONNXMatchesGoORT`, stale-projection guard |

## 5. Two implementations of one rule

| Rule | The bug it prevents | Guard |
|---|---|---|
| Any rule implemented in both Go and Python (preprocessing, manifest parsing, licence allowlist) has **one** shared fixture set and a sync test on both sides. | Python read YAML booleans, null keys and short normalisation lists differently from Go. | `test_spec_from_manifest_reads_yaml_like_go`, `test_odd_legacy_normalize_matches_go`, `internal/vision/preprocess/sync_test.go` |
| One code path per concern: a request option resolved by one endpoint is resolved by the shared helper for all. | `/api/preprocess` did not resolve `template_name` like `/api/predict`. | `TestPreprocessResolvesTemplateName` |
| Never write a test that pins today's output without a reference; a test must encode the *correct* behaviour. | A Python test pinned the Go/Python divergence itself. | review |

## 6. Files, installs, destructive operations

| Rule | The bug it prevents | Guard |
|---|---|---|
| Validate the input of a destructive operation before touching anything, and never remove the last good copy. | `pull` of a mistyped name still cleaned the registry; stale-staging cleanup could delete the only install. | `TestPullRemovesStaleStaging`, `TestInstallRemovesStaleStagingOnly` |
| Set permissions explicitly after `MkdirTemp`/`CreateTemp` when the result is published. | Models installed from a folder were `0700`. | `internal/catalog/local_test.go` |
| Tracked config uses relative paths only. | A shipped manifest pointed at `/home/<user>/…` and could not load elsewhere. | review |

## 7. Accuracy, performance and defaults

| Rule | The bug it prevents |
|---|---|
| Any change of execution provider, precision, thread/kernel settings or preprocessing re-measures **accuracy** (held-out protocol), not only latency. A quoted number records the EP it was measured on. | TensorRT was 1.5× faster and 6.8 mAP worse; it sat first in every chain. A preprocessing mismatch cost RF-DETR ~2 mAP silently. |
| A change that moves outputs or quoted numbers is **opt-in** until the numbers are re-measured and updated (`VISIONSERVE_DETERMINISTIC`, `--tensorrt`, SDK `base64_arrays`). | Defaults that silently changed results or return types. |
| Measure end to end (whole request, peak RSS) before optimising; find the real cause first. | Automask was slow from ORT thread oversubscription (4 pooled sessions × all cores spinning), not from the code the proposal targeted. |
| Size thread pools explicitly: a pooled or small session must not get ORT's default spinning pool. | Pools spun 4× the cores; a 1 ms head session made its detector 3× slower on CPU. |
| Golden/equivalence checks run on CPU (deterministic). GPU comparisons use a tolerance or `VISIONSERVE_DETERMINISTIC=1`. | GPU reductions flip last bits depending on load; masks moved by boundary pixels. |

## 8. Hygiene

- Environment variables are `VISIONSERVE_*`, warn once on a bad value, and cap absurd values.
- Platform-specific symbols live in build-tagged files (`staticcheck` flags them otherwise).
- Docs change in the **same** commit as the behaviour. Numbers in docs say where they were
  measured. Stale claims (TensorRT-first, "Run locks a mutex") cost more than missing ones.
- Every client package builds in CI (`tsc`, `pytest`), not only the Go server.
- Every ```mermaid diagram is parsed in CI (`website/tools/check_mermaid.mjs`): a `;` inside a
  sequence-diagram message silently broke the README's diagram on GitHub.
- Shell helpers: `pgrep -f PATTERN` matches its own command line — never loop on it.

## Before you merge

1. `gofmt -l`, `go vet ./...`, `staticcheck ./...` clean; `go test -race ./...` green
   (add `-count=5` for anything concurrent).
2. Golden harness: CPU outputs bit-identical, or the diff explained and quantified.
3. Anything that can move accuracy: held-out protocol on GPU, numbers in the PR.
4. Python (`pytest`), JS (`npm test`, `tsc`) and the converter image if their side changed.
5. Docs and manifests updated; `docs/refactor-proposal.md` §6 lists what stays open.
