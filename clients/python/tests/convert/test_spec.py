"""convert/spec.py — the manifest's preprocessing as data, the Python twin of vision/preprocess.

  * manifest_preprocess / manifest_meta / ManifestReference (tier B1, B2) give exactly what they
    gave before the Spec existed, for every legacy mode (frozen copies below);
  * the new modes follow the Go semantics (pad domains, rescale, layouts);
  * Bundle keeps rendering the LEGACY fields by default; preprocess_block=True adds a block that
    resolves to the same spec without contradicting them.

Resolution and geometry against Go itself: tests/test_go_python_sync.py (shared corpora).
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # clients/python

from visionserve.convert import common, spec  # noqa: E402
from visionserve.convert.reference import (ManifestReference, bundle_spec, manifest_meta,  # noqa: E402
                                           manifest_preprocess)

IMN_MEAN, IMN_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CLIP_MEAN, CLIP_STD = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]


# ---- FROZEN: reference.py before the Spec (do not fix) ---------------------------------------

def _old_dpt(w, h, tw, th, multiple=1):
    m = int(multiple) if multiple and multiple > 0 else 1
    sw, sh = tw / w, th / h
    if abs(1 - sw) < abs(1 - sh):
        sh = sw
    else:
        sw = sh
    return max(m, round(sw * w / m) * m), max(m, round(sh * h / m) * m)


def _old_manifest_preprocess(pil, width, height, mean=None, std=None, letterbox=False, crop=None,
                             keep_aspect=False, multiple_of=0):
    from PIL import Image
    img = pil.convert("RGB")
    ow, oh = img.size
    if keep_aspect:
        nw, nh = _old_dpt(ow, oh, width, height, multiple_of)
        img = img.resize((nw, nh), Image.BICUBIC)
        meta = {"orig_width": ow, "orig_height": oh, "scale_x": nw / ow, "scale_y": nh / oh, "pad_x": 0, "pad_y": 0}
    elif crop == "center":
        if ow <= oh:
            rw, rh = width, int(width * oh / ow)
        else:
            rw, rh = int(height * ow / oh), height
        rw, rh = max(rw, width), max(rh, height)
        ox, oy = (rw - width) // 2, (rh - height) // 2
        img = img.resize((rw, rh), Image.BICUBIC).crop((ox, oy, ox + width, oy + height))
        meta = {"orig_width": ow, "orig_height": oh, "scale_x": rw / ow, "scale_y": rh / oh, "pad_x": -ox, "pad_y": -oy}
    elif letterbox:
        s = min(width / ow, height / oh)
        nw, nh = max(1, int(ow * s + 0.5)), max(1, int(oh * s + 0.5))
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        px, py = (width - nw) // 2, (height - nh) // 2
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
        img, meta = canvas, {"orig_width": ow, "orig_height": oh, "scale_x": s, "scale_y": s, "pad_x": px, "pad_y": py}
    else:
        img = img.resize((width, height), Image.BILINEAR)
        meta = {"orig_width": ow, "orig_height": oh, "scale_x": width / ow, "scale_y": height / oh,
                "pad_x": 0, "pad_y": 0}
    x = np.asarray(img, np.float32) / 255.0
    if mean is not None and std is not None:
        x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]).astype(np.float32), meta


def _old_manifest_meta(bundle, ow, oh):
    if getattr(bundle, "keep_aspect", False):
        nw, nh = _old_dpt(ow, oh, bundle.width, bundle.height, getattr(bundle, "multiple_of", 0))
        return {"orig_width": ow, "orig_height": oh, "scale_x": nw / ow, "scale_y": nh / oh, "pad_x": 0, "pad_y": 0}
    if bundle.letterbox:
        s = min(bundle.width / ow, bundle.height / oh)
        nw, nh = int(ow * s + 0.5), int(oh * s + 0.5)
        return {"orig_width": ow, "orig_height": oh, "scale_x": s, "scale_y": s,
                "pad_x": (bundle.width - nw) // 2, "pad_y": (bundle.height - nh) // 2}
    return {"orig_width": ow, "orig_height": oh, "scale_x": bundle.width / ow, "scale_y": bundle.height / oh,
            "pad_x": 0, "pad_y": 0}


# ---- inputs ------------------------------------------------------------------------------------

SIZES = [(1, 1), (2, 1), (3, 5), (17, 31), (31, 17), (97, 173), (224, 224), (300, 500), (640, 480), (1500, 200)]


def _images():
    from PIL import Image
    rng = np.random.default_rng(0)
    for w, h in SIZES:
        rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        yield f"{w}x{h}/RGB", Image.fromarray(rgb)
    w, h = 97, 61
    rgba = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
    yield "RGBA", Image.fromarray(rgba, "RGBA")
    yield "L", Image.fromarray(rgba[..., 0], "L")
    yield "P", Image.fromarray(rgba[..., :3]).convert("P", palette=Image.ADAPTIVE)
    yield "I;16", Image.fromarray(rng.integers(0, 65536, (h, w), dtype=np.uint16))
    yield "CMYK", Image.fromarray(rgba, "RGBA").convert("CMYK")


LEGACY_CASES = [
    dict(width=64, height=64, mean=IMN_MEAN, std=IMN_STD),
    dict(width=48, height=32, mean=IMN_MEAN, std=IMN_STD, letterbox=True),
    dict(width=32, height=32, mean=CLIP_MEAN, std=CLIP_STD, crop="center"),
    dict(width=42, height=28, crop="center"),
    dict(width=70, height=70, mean=IMN_MEAN, std=IMN_STD, keep_aspect=True, multiple_of=14),
    dict(width=40, height=30, keep_aspect=True, letterbox=True),   # keep_aspect wins, as in Go
]


def test_manifest_preprocess_is_unchanged_for_legacy_fields():
    for name, pil in _images():
        for kw in LEGACY_CASES:
            got, gmeta = manifest_preprocess(pil, **kw)
            want, wmeta = _old_manifest_preprocess(pil, **kw)
            assert got.dtype == want.dtype and got.shape == want.shape, (name, kw)
            assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), (name, kw)
            assert gmeta == wmeta, (name, kw)


def test_odd_legacy_normalize_matches_go():
    """Legacy normalize lists the Go normalizer accepts: a missing mean entry is 0, a missing or
    zero std entry is 1, extras are ignored. The pre-Spec Python reference skipped normalisation
    when only one of mean/std was given (or broadcast / crashed on other lengths), so tier B1
    compared the server against a tensor it never fed."""
    from PIL import Image
    pil = Image.new("RGB", (4, 3), (51, 102, 204))
    p = np.array([51, 102, 204], np.float32) / np.float32(255)
    cases = [
        (dict(mean=IMN_MEAN), (p - np.array(IMN_MEAN, np.float32)) / np.float32(1)),
        (dict(std=IMN_STD), p / np.array(IMN_STD, np.float32)),
        (dict(mean=[0.5], std=[0.25]), (p - np.array([0.5, 0, 0], np.float32)) / np.array([0.25, 1, 1], np.float32)),
        (dict(mean=[0.5, 0.5, 0.5, 9], std=[0.25, 0, 0.5, 9]),
         (p - np.float32(0.5)) / np.array([0.25, 1, 0.5], np.float32)),
    ]
    for kw, want in cases:
        x, _ = manifest_preprocess(pil, width=4, height=3, **kw)
        got = x[0, :, 0, 0]
        assert np.array_equal(got.view(np.uint32), want.astype(np.float32).view(np.uint32)), (kw, got, want)


def test_manifest_reference_b1_is_unchanged():
    """ManifestReference.preprocess (tier B1's manifest reference) feeds the same bits as before."""
    from types import SimpleNamespace
    for kw in LEGACY_CASES:
        b = SimpleNamespace(name="m", task="classification", architecture="efficientnet", layout="NHWC",
                            letterbox=kw.get("letterbox", False), crop=kw.get("crop"),
                            keep_aspect=kw.get("keep_aspect", False), multiple_of=kw.get("multiple_of", 0),
                            width=kw["width"], height=kw["height"], mean=kw.get("mean"), std=kw.get("std"))
        ref = ManifestReference(b, "input")
        for name, pil in _images():
            got = ref.preprocess(pil)["input"]
            want, _ = _old_manifest_preprocess(pil, **kw)
            assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), (name, kw)


def test_manifest_meta_is_unchanged():
    from types import SimpleNamespace
    bundles = [SimpleNamespace(width=100, height=100, letterbox=False),
               SimpleNamespace(width=100, height=80, letterbox=True),
               SimpleNamespace(width=518, height=518, letterbox=False, keep_aspect=True, multiple_of=14),
               SimpleNamespace(width=224, height=224, letterbox=False, crop="center")]  # box decoders ignore crop
    for b in bundles:
        for w, h in SIZES + [(848, 480), (200, 400)]:
            assert manifest_meta(b, w, h) == _old_manifest_meta(b, w, h), (vars(b), w, h)


def test_reference_descriptions_are_unchanged():
    from types import SimpleNamespace

    def d(**kw):
        kw.setdefault("letterbox", False)
        return ManifestReference(SimpleNamespace(width=8, height=8, **kw), "x").description

    assert "(numpy/PIL: squash bilinear to 8x8" in d()
    assert "letterbox bilinear" in d(letterbox=True)
    assert "centre crop bicubic" in d(crop="center")
    assert "keep-aspect bicubic (multiple of 14) around" in d(keep_aspect=True, multiple_of=14)


# ---- the modes that have no legacy spelling ---------------------------------------------------

def _solid(w, h, rgb):
    from PIL import Image
    return Image.fromarray(np.full((h, w, 3), rgb, np.uint8))


def test_top_left_pad_places_the_image_at_the_origin_and_pads_pixels():
    s = spec.Spec(resize="top_left_pad", width=64, height=64, rescale=False,
                  mean=[127.5] * 3, std=[128.0] * 3)
    x, meta = spec.apply_spec(_solid(64, 32, (255, 128, 0)), s)
    assert x.shape == (1, 3, 64, 64) and meta == {"orig_width": 64, "orig_height": 32, "scale_x": 1.0,
                                                  "scale_y": 1.0, "pad_x": 0, "pad_y": 0}
    np.testing.assert_allclose(x[0, :, 0, 0], [(255 - 127.5) / 128, (128 - 127.5) / 128, -127.5 / 128], atol=1e-6)
    np.testing.assert_allclose(x[0, :, 40, 10], [-127.5 / 128] * 3, atol=1e-6)  # black pixel pad, normalised


def test_long_side_pad_pads_the_normalised_tensor():
    s = spec.Spec(resize="long_side_pad", width=32, height=32, mean=IMN_MEAN, std=IMN_STD)
    x, meta = spec.apply_spec(_solid(40, 20, (10, 20, 30)), s)
    assert x.shape == (1, 3, 32, 32) and meta["scale_x"] == 32 / 40
    assert (x[0, :, 16:, :] == 0).all()                       # SAM: zeros AFTER normalisation
    np.testing.assert_allclose(x[0, 0, 0, 0], (10 / 255 - 0.485) / 0.229, atol=1e-6)
    s = spec.Spec(resize="long_side_pad", width=960, height=960, multiple_of=32, no_upscale=True)
    x, meta = spec.apply_spec(_solid(100, 50, (1, 2, 3)), s)  # never upscaled, padded to 128x64
    assert x.shape == (1, 3, 64, 128) and meta["scale_x"] == 1.0


def test_long_side_raw_hwc_and_none():
    x, meta = spec.apply_spec(_solid(64, 48, (100, 150, 200)),
                              spec.Spec(resize="long_side", width=128, height=128, rescale=False, layout="HWC"))
    assert x.shape == (96, 128, 3) and meta["scale_x"] == 2.0
    np.testing.assert_array_equal(x[0, 0], [100, 150, 200])   # raw 0..255 (MobileSAM)
    x, meta = spec.apply_spec(_solid(5, 3, (51, 102, 255)), spec.Spec(resize="none"))
    assert x.shape == (1, 3, 3, 5) and meta["scale_x"] == 1.0
    np.testing.assert_allclose(x[0, :, 1, 1], [0.2, 0.4, 1.0], atol=1e-7)


def test_layouts_are_transposes():
    pil = next(p for n, p in _images() if n.startswith("97x173"))
    base = spec.Spec(resize="squash", width=20, height=10, mean=IMN_MEAN, std=IMN_STD)
    nchw, _ = spec.apply_spec(pil, base)
    nhwc, _ = spec.apply_spec(pil, dataclasses.replace(base, layout="NHWC"))
    hwc, _ = spec.apply_spec(pil, dataclasses.replace(base, layout="HWC"))
    assert nhwc.shape == (1, 10, 20, 3) and hwc.shape == (10, 20, 3)
    assert np.array_equal(nchw[0].transpose(1, 2, 0), nhwc[0]) and np.array_equal(nhwc[0], hwc)
    # a LEGACY spec is always NCHW, whatever input.layout says (no architecture read it)
    legacy, _ = spec.apply_spec(pil, dataclasses.replace(base, layout="NHWC", legacy=True))
    assert legacy.shape == (1, 3, 10, 20)


def test_gray_letterbox_pad():
    s = spec.Spec(resize="letterbox", width=16, height=16, pad=114)
    x, meta = spec.apply_spec(_solid(16, 8, (0, 0, 0)), s)
    assert meta["pad_y"] == 4
    np.testing.assert_allclose(x[0, :, 0, 0], [114 / 255] * 3, atol=1e-7)
    np.testing.assert_allclose(x[0, :, 8, 8], [0.0] * 3, atol=1e-7)


# ---- Bundle rendering ----------------------------------------------------------------------------

def _bundle(tmp_path, **kw):
    p = tmp_path / "m.onnx"
    p.write_bytes(b"onnx")
    d = dict(name="m", task="detection", architecture="rf-detr", license="Apache-2.0", width=560, height=560,
             onnx={"model": str(p)}, mean=IMN_MEAN, std=IMN_STD)
    d.update(kw)
    return common.Bundle(**d)


@pytest.mark.parametrize("kw", [{}, {"letterbox": True}, {"crop": "center", "architecture": "clip", "task": "embed"},
                                {"keep_aspect": True, "multiple_of": 14, "architecture": "depth-anything-v2",
                                 "task": "depth", "width": 518, "height": 518}])
def test_bundle_renders_legacy_fields_by_default_and_a_consistent_block_on_request(tmp_path, kw):
    yaml = pytest.importorskip("yaml")
    b = _bundle(tmp_path, **kw)
    plain = b.render_manifest()
    assert "preprocess:" not in plain and "  letterbox: " in plain
    with_block = b.render_manifest(preprocess_block=True)
    assert with_block.startswith(plain.split("\npostprocess:")[0].split("\nlabels:")[0].split("\nruntime:")[0])
    doc = yaml.safe_load(with_block)
    assert doc["input"] == yaml.safe_load(plain)["input"]     # the legacy fields are untouched
    resolved = spec.spec_from_manifest(doc)                    # the Go rules: no conflict
    assert dataclasses.replace(resolved, legacy=True) == dataclasses.replace(b.spec(), mean=resolved.mean,
                                                                             std=resolved.std)
    np.testing.assert_allclose(resolved.mean, b.spec().mean, rtol=1e-5)
    assert bundle_spec(b) == b.spec()


def test_spec_from_manifest_reads_yaml_like_go():
    """Booleans as yaml.v3 decodes them (YAML 1.1 spellings, quoted or not; "false" quoted, numbers
    and others are errors), null keys absent, non-finite numbers refused."""
    def resolve(inp, pre):
        return spec.spec_from_manifest({"input": dict(width=8, height=8, **inp), "preprocess": pre})

    assert resolve({"letterbox": "n"}, {}).resize == ""          # y/n arrive as str from PyYAML
    assert resolve({"letterbox": "y"}, {}).resize == "letterbox"
    assert resolve({}, {"resize": "long_side", "rescale": "no"}).rescale is False
    assert resolve({"letterbox": None}, {"resize": "letterbox"}).resize == "letterbox"
    assert resolve({"keep_aspect": None}, {"resize": "keep_aspect"}).resize == "keep_aspect"
    assert resolve({"multiple_of": None}, {"resize": "keep_aspect", "multiple_of": 14}).multiple_of == 14
    for inp, pre in [({"letterbox": "false"}, {}), ({"letterbox": 1}, {}), ({}, {"rescale": 0}),
                     ({}, {"resize": False}),                                  # `resize: no`
                     ({}, {"resize": "letterbox", "pad": float("nan")}),
                     ({}, {"mean": [float("nan"), 0.5, 0.5], "std": [0.2, 0.2, 0.2]}),
                     ({}, {"mean": [0.5, 0.5, 0.5], "std": [float("inf"), 0.2, 0.2]}),
                     ({}, {"resize": "long_side_pad", "pad": float("inf")})]:
        with pytest.raises(spec.SpecError):
            resolve(inp, pre)
