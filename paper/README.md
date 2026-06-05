# VisionServe: A Lean, License-Safe Inference Server for Computer Vision

This directory contains the LaTeX source for the VisionServe system paper, ready for
arXiv submission.

## What the paper covers

The paper describes the design and implementation of VisionServe: a single Go binary
that serves computer-vision models locally without accounts, API keys, or cloud
dependencies. It covers:

- **System design** — the unified `Model` / `PipelineModel` interface, lifecycle
  manager, ONNX Runtime integration, and execution-provider fallback chain.
- **Model catalog** — 16 permissive-licensed models spanning detection, segmentation,
  open-vocabulary detection, depth estimation, classification, image embeddings, face
  detection, and OCR.
- **Benchmarks** — end-to-end latency, throughput, VRAM, and cold-start numbers for
  all models on an NVIDIA RTX A6000 GPU.
- **License safety** — why AGPL copyleft is strictly forbidden and how the registry
  enforces the Apache-2.0 / MIT / BSD allowlist.

## How to compile

```bash
# Option A: the texpdf conda env (recommended — no container, no full TeX Live)
# One-time setup: a self-contained `tectonic` engine that fetches only the
# packages this document needs on first run and caches them.
conda create -n texpdf -c conda-forge tectonic

# Build from the repo root (runs the build inside texpdf via `conda run`):
make pdf
# ...or inside paper/ with the env active:
conda activate texpdf && make pdf
# Output: paper/main.pdf. (tectonic prints a benign "main.bbl changed" warning on
# bibtex docs; the PDF and citations are correct.)

# Option B: Overleaf
# Upload this entire paper/ directory as a new project.
# neurips_2024.sty is pre-installed on Overleaf.

# Option C: raw system TeX fallback
# Download neurips_2024.sty from https://neurips.cc/Conferences/2024/PaperInformation/StyleFiles
# Place it in paper/, then:
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Before submitting to arXiv

The affiliation (`Korea Electronics Technology Institute (KETI)`) and GitHub URL
(`https://github.com/mtbui2010/visionserve`) are already filled in.

Before submitting, update the arXiv ID placeholder `XXXX.XXXXX` once you have it.

## Section map

| File | Contents |
|------|----------|
| `sections/abstract.tex` | One-paragraph summary of contributions and results |
| `sections/introduction.tex` | Motivation, problem statement, and paper contributions |
| `sections/related.tex` | Related work: serving systems, SAM variants, DETR-family detectors |
| `sections/design.tex` | Architecture: Model interface, lifecycle manager, pipeline, EP fallback |
| `sections/catalog.tex` | The 16-model catalog: tasks, licenses, sources, ONNX I/O shapes |
| `sections/evaluation.tex` | Latency / throughput / VRAM benchmarks and accuracy reference numbers |
| `sections/applications.tex` | Example end-to-end use cases (edge robotics, Grounded-SAM, OCR) |
| `sections/limitations.tex` | Current limitations and open issues |
| `sections/conclusion.tex` | Summary and future directions |

## Citation

```bibtex
@misc{visionserve2026,
  title={VisionServe: A Lean, License-Safe Inference Server for Computer Vision},
  author={Bui, Trung Minh},
  year={2026},
  eprint={XXXX.XXXXX},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

> The arXiv ID (`XXXX.XXXXX`) is a placeholder and will be updated after submission.
