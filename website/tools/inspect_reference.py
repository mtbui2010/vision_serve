"""Example `--reference-script` for the converter's tier B1 (website/docs/guides/inspect.md).

It is the preprocessing an RF-DETR model saw in training, written the way a training script
would write it: squash the photo to 560 x 560 (the aspect ratio is NOT kept), scale to 0..1,
normalise with the ImageNet mean/std, channels first. Tier B1 compares the tensor this builds
with the tensor the server builds (POST /api/preprocess) for the same photo.
"""
import numpy as np
from PIL import Image

SIZE = 560
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(pil):
    """PIL image -> float32 array [3, SIZE, SIZE]."""
    x = np.asarray(pil.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)
