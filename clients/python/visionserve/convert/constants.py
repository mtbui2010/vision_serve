"""Names and small parsers that several converter modules need — one copy, so they cannot drift.

Standard library only: the CLI imports this before any framework, the TensorFlow venv has no
torch, and common.py imports it (so it must not import common at module level).
"""
from __future__ import annotations

import re

# Image files that verification (tier B, --images) and evaluation (tier C, ImageFolder --eval)
# pick up from a directory, by lower-cased suffix.
IMAGE_EXT = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})

# A HuggingFace hub id, "org/name" (what the hf format downloads when no such local path exists).
HUB_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")

# Text towers: architectures whose ONNX takes token ids, not an image (no image preprocessing to
# verify, nothing image-shaped to benchmark).
TEXT_ARCHS = ("siglip-text", "clip-text")

# Architectures that label a detection with a PROMPT PHRASE rather than a fixed class name, so a
# served label is compared with the reference's by phrase containment, not equality.
OPEN_VOCAB_ARCHS = ("grounding-dino", "grounded-sam", "gdino-siglip", "rfdetr-gdino")


def parse_wxh(s: str):
    """"224x224" -> (224, 224). Raises ConvertError for anything else."""
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except ValueError:
        from .common import ConvertError  # lazy: common imports this module
        raise ConvertError(f"--input must look like 224x224, got {s!r}")
