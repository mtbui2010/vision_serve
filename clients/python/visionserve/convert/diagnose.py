"""Tier B1: compare the tensor the SERVER feeds the model with the tensor the REFERENCE
preprocessing builds for the same image, and when they differ, say WHY.

A wrong preprocessing step does not crash anything: the model runs, returns plausible boxes, and is
quietly worse. The usual culprits each leave a recognisable fingerprint in the two tensors:

  letterbox vs squash   constant padding rows/columns on one side only (or the server's meta pad)
  centre crop           the reference correlates with a central crop of the server image
  mean/std mismatch     a per-channel affine relation ref = a*srv + b with a high r^2
  /255 missing          the same relation with a ~= 255 (or 1/255)
  RGB/BGR swap          reference channel 0 correlates best with server channel 2
  resampling filter     high r^2, a ~= 1, b ~= 0, small residual

Differences are reported in GRAY LEVELS (0-255 pixel units): the tensor difference times the
manifest's std*255, so the number means the same thing whatever the normalisation.

numpy + PIL only; unit-tested offline.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def _chw(t) -> np.ndarray:
    a = np.asarray(t, np.float64)
    if a.ndim == 4:
        if a.shape[0] != 1:
            raise ValueError(f"expected batch 1, got {a.shape}")
        a = a[0]
    if a.ndim != 3:
        raise ValueError(f"expected an image tensor [1,C,H,W] or [C,H,W], got {a.shape}")
    return a


def _vec(v, c, default):
    if v is None:
        return np.full(c, default, np.float64)
    v = np.asarray(v, np.float64).ravel()
    return np.resize(v, c)


def compare_tensors(srv, ref, std=None) -> dict:
    """Element-wise difference of two image tensors of the SAME shape, in tensor units and in
    gray levels. Returns {"same_shape": False, ...} when the shapes differ."""
    s, r = _chw(srv), _chw(ref)
    out = {"shape_srv": list(s.shape), "shape_ref": list(r.shape), "same_shape": s.shape == r.shape}
    if not out["same_shape"]:
        return out
    d = np.abs(s - r)
    lv = d * _vec(std, s.shape[0], 1.0)[:, None, None] * 255.0
    out.update(mean_abs=float(d.mean()), max_abs=float(d.max()), mean_levels=float(lv.mean()),
               max_levels=float(lv.max()), p99_levels=float(np.percentile(lv, 99)),
               per_channel_mean_levels=[float(x) for x in lv.mean((1, 2))])
    return out


# --------------------------------------------------------------------------------------------
# fingerprints
# --------------------------------------------------------------------------------------------

def pad_extent(t, tol: float = 1e-4) -> Tuple[int, int, int, int]:
    """(top, bottom, left, right): rows/columns at each edge that are constant (per channel) and
    equal to each other — what a letterbox's padding looks like."""
    a = _chw(t)

    def run(lines):
        n, v0 = 0, None
        for ln in lines:  # ln: [C, L]
            if ln.std(axis=1).max() > tol:
                break
            v = ln[:, 0]
            if v0 is None:
                v0 = v
            elif np.abs(v - v0).max() > tol:
                break
            n += 1
        return n

    _, h, w = a.shape
    return (run(a[:, i, :] for i in range(h)), run(a[:, i, :] for i in range(h - 1, -1, -1)),
            run(a[:, :, j] for j in range(w)), run(a[:, :, j] for j in range(w - 1, -1, -1)))


def affine_fit(s: np.ndarray, r: np.ndarray):
    """Least squares r = a*s + b on flattened values -> (a, b, r2)."""
    s, r = s.ravel(), r.ravel()
    if s.size > 200_000:  # subsample deterministically; the fit does not need every pixel
        idx = np.linspace(0, s.size - 1, 200_000).astype(np.int64)
        s, r = s[idx], r[idx]
    vs = s.var()
    if vs < 1e-12:
        return 0.0, float(r.mean()), 0.0
    a = float(((s - s.mean()) * (r - r.mean())).mean() / vs)
    b = float(r.mean() - a * s.mean())
    res = r - (a * s + b)
    vr = r.var()
    r2 = float(1.0 - res.var() / vr) if vr > 1e-12 else (1.0 if res.var() < 1e-12 else 0.0)
    return a, b, r2


def _corr(x, y) -> float:
    x, y = x.ravel() - x.mean(), y.ravel() - y.mean()
    d = np.sqrt((x @ x) * (y @ y))
    return float(x @ y / d) if d > 0 else 0.0


def _resize2d(m: np.ndarray, w: int, h: int) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.fromarray(np.asarray(m, np.float32), mode="F").resize((w, h), Image.BILINEAR),
                      np.float64)


def _small(g: np.ndarray, side: int = 64) -> np.ndarray:
    h, w = g.shape
    k = side / max(h, w)
    return _resize2d(g, max(8, int(round(w * k))), max(8, int(round(h * k)))) if k < 1 else g


def crop_search(srv, ref, fracs=None):
    """Find the centred crop (fx, fy) of one tensor that best explains the other.
    Returns (base_corr, best_corr, fx, fy, direction) with direction 'ref_is_crop' (the reference
    shows only the centre of what the server sees) or 'srv_is_crop'."""
    fracs = fracs or [round(1.0 - 0.05 * i, 2) for i in range(15)]  # 1.0 .. 0.30
    gs, gr = _small(_chw(srv).mean(0)), _small(_chw(ref).mean(0))
    h, w = gr.shape
    if gs.shape != gr.shape:
        gs = _resize2d(gs, w, h)
    base = _corr(gs, gr)
    best = (base, 1.0, 1.0, "none")
    for direction, big, small in (("ref_is_crop", gs, gr), ("srv_is_crop", gr, gs)):
        H, W = big.shape
        for fy in fracs:
            for fx in fracs:
                if fx == 1.0 and fy == 1.0:
                    continue
                cw, ch = max(4, int(round(W * fx))), max(4, int(round(H * fy)))
                x0, y0 = (W - cw) // 2, (H - ch) // 2
                c = _resize2d(big[y0:y0 + ch, x0:x0 + cw], W, H)
                v = _corr(c, small)
                if v > best[0]:
                    best = (v, fx, fy, direction)
    return base, best[0], best[1], best[2], best[3]


def _implied_norm(a, b, mean_s, std_s):
    """Server = (x - m_s)/s_s; reference = a*server + b = (x - m_r)/s_r  =>  s_r = s_s/a,
    m_r = m_s - b*s_r (x in [0,1])."""
    a = np.where(np.abs(a) < 1e-9, 1e-9, a)
    s_r = std_s / a
    m_r = mean_s - b * s_r
    return m_r, s_r


def _fmt3(v) -> str:
    return "[" + ", ".join(f"{float(x):.3g}" for x in v) + "]"


def diagnose(srv, ref, meta: Optional[dict] = None, mean: Optional[Sequence[float]] = None,
             std: Optional[Sequence[float]] = None, orig_size: Optional[Tuple[int, int]] = None
             ) -> Tuple[List[str], List[str]]:
    """Explain a difference between the server tensor and the reference tensor.

    mean/std: what the MANIFEST declares (the server applies x/255, then (x-mean)/std).
    meta: the server's preprocess meta (pad_x/pad_y > 0 means the server letterboxed).
    orig_size: (width, height) of the original image, to recognise an aspect-kept reference.
    Returns (codes, messages); codes are stable identifiers for tests and the JSON report."""
    s, r = _chw(srv), _chw(ref)
    c = s.shape[0]
    mean_s, std_s = _vec(mean, c, 0.0), _vec(std, c, 1.0)
    codes, msgs = [], []

    if s.shape != r.shape:
        codes.append("shape")
        m = f"shape: server {list(s.shape)} vs reference {list(r.shape)}"
        if orig_size and r.shape[1:] != s.shape[1:]:
            ow, oh = orig_size
            ar_o, ar_r, ar_s = ow / oh, r.shape[2] / r.shape[1], s.shape[2] / s.shape[1]
            if abs(ar_r - ar_o) / ar_o < 0.03 and abs(ar_s - ar_o) / ar_o >= 0.03:
                m += (f" — the reference KEEPS the aspect ratio ({r.shape[2]}x{r.shape[1]} for a {ow}x{oh} image) "
                      f"while the server SQUASHES to {s.shape[2]}x{s.shape[1]}")
        msgs.append(m)
        if s.shape[0] == r.shape[0]:  # global per-channel statistics survive a geometry change
            a = np.array([r[k].std() / s[k].std() if s[k].std() > 0 else 1.0 for k in range(c)])
            b = np.array([r[k].mean() - a[k] * s[k].mean() for k in range(c)])
            _norm_messages(a, b, np.ones(c), mean_s, std_s, codes, msgs, approx=True)
        return codes, msgs

    # ---- letterbox vs squash ----
    sp = pad_extent(s)
    rp = pad_extent(r)
    # Constant edge rows count as padding only where the OTHER tensor is not constant too (a dark
    # band in the photo is constant in both); the server's meta, when given, is authoritative.
    if meta is not None:
        srv_pad = bool(meta.get("pad_x") or meta.get("pad_y"))
    else:
        srv_pad = any(a >= 2 and a > b + 1 for a, b in zip(sp, rp))
    ref_pad = any(b >= 2 and b > a + 1 for a, b in zip(sp, rp))
    if srv_pad and not ref_pad:
        px = (meta or {}).get("pad_x", (sp[2] + sp[3]) // 2)
        py = (meta or {}).get("pad_y", (sp[0] + sp[1]) // 2)
        codes.append("letterbox_server")
        msgs.append(f"geometry: the server LETTERBOXES (keeps aspect, pads x={px} y={py}) but the reference "
                    "SQUASHES the whole frame — set `letterbox: false` in the manifest unless the model was "
                    "trained letterboxed")
    elif ref_pad and not srv_pad:
        codes.append("letterbox_reference")
        msgs.append(f"geometry: the reference LETTERBOXES (constant padding top/bottom/left/right = {list(rp)}) "
                    "but the server SQUASHES (letterbox: false) — set `letterbox: true` if the model was "
                    "trained that way")

    # ---- channel order ----
    if c == 3:
        cm = np.array([[_corr(r[i], s[j]) for j in range(3)] for i in range(3)])
        perm = cm.argmax(1)
        if sorted(perm.tolist()) == [0, 1, 2] and perm.tolist() != [0, 1, 2] and \
                cm[np.arange(3), perm].mean() > np.diag(cm).mean() + 0.02:
            codes.append("channel_swap")
            msgs.append(f"channel order: reference channels {list(range(3))} match server channels "
                        f"{perm.tolist()} — {'RGB/BGR swap' if perm.tolist() == [2, 1, 0] else 'channels permuted'}")
            s = s[perm]

    # ---- per-channel affine relation ----
    fits = [affine_fit(s[k], r[k]) for k in range(c)]
    a = np.array([f[0] for f in fits])
    b = np.array([f[1] for f in fits])
    r2 = np.array([f[2] for f in fits])
    if r2.min() >= 0.97:
        _norm_messages(a, b, r2, mean_s, std_s, codes, msgs, approx=False)
        if not any(k in codes for k in ("scale_255", "scale_inv255", "normalisation", "channel_swap",
                                        "letterbox_server", "letterbox_reference")):
            lv = np.abs(s - r) * std_s[:, None, None] * 255.0
            if lv.mean() > 0.25:
                codes.append("resample")
                msgs.append(f"small residual (mean {lv.mean():.2f} gray levels, r^2 {r2.min():.4f}, no offset or "
                            "scale): resampling filter / antialiasing or JPEG-decoder differences")
        return codes, msgs

    # ---- spatial mismatch: crop? flip? ----
    if "letterbox_server" in codes or "letterbox_reference" in codes:
        return codes, msgs
    base, best, fx, fy, direction = crop_search(s, r)
    flip = _corr(_small(s.mean(0)), _small(r.mean(0)[:, ::-1]))
    if flip > max(best, base) + 0.05 and flip > 0.9:
        codes.append("flip")
        msgs.append(f"geometry: the reference is horizontally FLIPPED relative to the server (corr {flip:.3f})")
    elif best >= 0.85 and best - base >= 0.05 and direction != "none":
        codes.append("crop")
        who, other = (("reference", "server")) if direction == "ref_is_crop" else ("server", "reference")
        msgs.append(f"geometry: the {who} shows only the central ~{fx:.0%} x {fy:.0%} of what the {other} sees "
                    f"(corr {base:.3f} -> {best:.3f} after cropping) — a CENTRE CROP (e.g. HF do_center_crop / "
                    "torchvision CenterCrop) that VisionServe's squash does not do")
    else:
        codes.append("spatial")
        msgs.append(f"spatial mismatch (per-channel r^2 {r2.min():.3f}, image corr {base:.3f}): a different "
                    "resize geometry (crop / aspect / interpolation) or a different image")
    return codes, msgs


def _norm_messages(a, b, r2, mean_s, std_s, codes, msgs, approx: bool):
    k = float(np.median(a))
    how = "global channel statistics suggest" if approx else "per-channel fit ref = a*server + b"
    if 150 < k < 400:
        codes.append("scale_255")
        msgs.append(f"scale: the reference is ~{k:.0f}x the server tensor ({how}) — the reference works in 0-255 "
                    "while the server divides by 255: a missing /255 (ToTensor) on one side")
        return
    if 1 / 400 < k < 1 / 150:
        codes.append("scale_inv255")
        msgs.append(f"scale: the reference is ~1/{1 / k:.0f} of the server tensor ({how}) — the reference divides by "
                    "255 twice, or the model expects raw 0-255 (manifest mean 0 / std 0.00392157)")
        return
    tol_a, tol_b = (0.08, 0.08) if approx else (0.02, 0.02)
    if np.all(np.abs(a - 1) < tol_a) and np.all(np.abs(b) < tol_b):
        return
    m_r, s_r = _implied_norm(a, b, mean_s, std_s)
    codes.append("normalisation")
    if np.allclose(m_r, 0, atol=0.03) and np.allclose(s_r, 1, atol=0.05):
        what = "NO mean/std normalisation (just /255)"
    elif np.allclose(m_r, IMAGENET[0], atol=0.02) and np.allclose(s_r, IMAGENET[1], atol=0.02):
        what = "ImageNet mean/std"
    elif np.allclose(m_r, 0.5, atol=0.02) and np.allclose(s_r, 0.5, atol=0.02):
        what = "mean = std = 0.5"
    else:
        what = f"mean ~ {_fmt3(m_r)}, std ~ {_fmt3(s_r)}"
    msgs.append(f"normalisation: the reference looks like {what} ({how}: a={_fmt3(a)}, b={_fmt3(b)}"
                + ("" if approx else f", r^2>={float(np.min(r2)):.3f}")
                + f"); the manifest declares mean {_fmt3(mean_s)}, std {_fmt3(std_s)}")
