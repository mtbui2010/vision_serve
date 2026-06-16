"""Visualization helpers for VisionServe prediction results.

Requires **Pillow** (``pip install pillow`` or ``pip install 'visionserve[images]'``).
Pillow is imported lazily so that the rest of the SDK can be used without it.

Public API::

    from visionserve.visualize import draw

    annotated = draw(result, "photo.jpg")
    annotated.save("out.jpg")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

if TYPE_CHECKING:
    from .types import Result


# ---------------------------------------------------------------------------
# Fixed colour palette — identical to the Go server's overlay palette.
# ---------------------------------------------------------------------------
_PALETTE: List[Tuple[int, int, int]] = [
    (255, 59, 59),
    (255, 165, 0),
    (50, 205, 50),
    (0, 191, 255),
    (238, 130, 238),
    (255, 215, 0),
    (0, 255, 127),
    (255, 99, 71),
]


def _colour(idx: int) -> Tuple[int, int, int]:
    return _PALETTE[idx % len(_PALETTE)]


# Highlight colour for the selected target box (matches the target-grasp red).
_TARGET_BOX_COLOUR: Tuple[int, int, int] = (255, 0, 0)


def _target_bbox(target_box: Any) -> Optional[Tuple[float, float, float, float]]:
    """Resolve *target_box* to a ``(x, y, w, h)`` tuple, or ``None``.

    Accepts a ``Detection`` / ``Mask`` (anything with a ``.bbox``) or a raw
    ``[x, y, w, h]`` sequence.
    """
    if target_box is None:
        return None
    bbox = getattr(target_box, "bbox", target_box)
    try:
        vals = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    return (vals[0], vals[1], vals[2], vals[3]) if len(vals) == 4 else None


def _is_target_item(item: Any, target_box: Any, target_bbox: Any) -> bool:
    """True if *item* (a Detection/Mask) is the selected target — by object identity
    (``is``) or by matching bounding box."""
    if target_box is None:
        return False
    if item is target_box:
        return True
    if target_bbox is not None:
        try:
            return tuple(float(v) for v in item.bbox) == target_bbox
        except (TypeError, ValueError):
            return False
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draw(
    result: "Result",
    image: Any,
    *,
    alpha: float = 0.45,
    max_grasps_per_object: Optional[int] = 3,
    target_grasp: Optional[Any] = None,
    target_box: Optional[Any] = None,
) -> Any:
    """Draw *result* predictions on *image* and return an annotated PIL image.

    Args:
        result: :class:`~visionserve.Result` from :meth:`~visionserve.Client.predict`.
        image:  Input image — one of:

                * ``PIL.Image.Image`` (used directly, not modified in-place),
                * ``str`` file path,
                * ``bytes`` raw encoded image data.
        alpha:  Opacity for mask colour overlays (0.0 = transparent, 1.0 = opaque).
        max_grasps_per_object: For a ``grasp`` result, draw at most this many
                highest-quality grasps PER object (grouped by the detection/mask
                whose bbox contains each grasp centre) instead of all of them —
                the analytic search can return hundreds per object. ``None`` or
                ``<= 0`` draws every grasp. Defaults to 3.
        target_grasp: A :class:`~visionserve.Grasp` instance (identity comparison via
                ``is``) to highlight in red. All other grasps are drawn in the
                quality colour (red→yellow→green). ``None`` disables highlighting.
        target_box: The selected target object to highlight in red with a thicker
                outline (e.g. the result of :func:`~visionserve.select_target_object`).
                Accepts a ``Detection`` / ``Mask`` (matched by identity or bbox) or a
                raw ``[x, y, w, h]`` box. If it matches one of this result's own
                detections/masks that item is highlighted in place; otherwise the box
                is drawn as a standalone red rectangle (handy when the box came from a
                separate detect call). ``None`` disables it.

    Returns:
        Annotated ``PIL.Image.Image``.

    Raises:
        ImportError: if Pillow is not installed.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ImportError(
            "visionserve.visualize.draw() requires Pillow. "
            "Install with: pip install pillow  (or pip install 'visionserve[images]')"
        ) from exc

    img = _open_image(image, Image)

    # Dispatch to the correct renderer based on the task reported in the result.
    task = (result.task or "").lower()

    if task == "depth":
        return _draw_depth(result, Image)

    # Start with a working copy so we don't mutate the caller's image.
    img = img.convert("RGBA")

    # Does the target box correspond to one of this result's own detections/masks?
    target_bbox = _target_bbox(target_box)
    target_matched = target_box is not None and any(
        _is_target_item(it, target_box, target_bbox)
        for it in (list(result.detections) + list(result.masks))
    )

    if result.detections:
        img = _draw_detections(result, img, ImageDraw, ImageFont, target_box)

    if result.masks:
        img = _draw_masks(result, img, alpha, Image, ImageDraw, ImageFont, target_box)

    if result.grasps:
        img = _draw_grasps(result, img, ImageDraw, ImageFont, max_grasps_per_object, target_grasp)

    if result.classifications:
        img = _draw_classifications(result, img, ImageDraw, ImageFont)

    # A target box that is not one of this result's own items (e.g. selected on a
    # separate detection result) is drawn as a standalone red highlight.
    if target_bbox is not None and not target_matched:
        x, y, w, h = target_bbox
        _draw_box(ImageDraw.Draw(img), x, y, w, h, _TARGET_BOX_COLOUR, thickness=4)

    return img.convert("RGB")


# ---------------------------------------------------------------------------
# Detections / open-vocab
# ---------------------------------------------------------------------------

def _draw_detections(
    result: "Result", img: Any, ImageDraw: Any, ImageFont: Any, target_box: Any = None
) -> Any:
    draw_ctx = ImageDraw.Draw(img)
    target_bbox = _target_bbox(target_box)
    target_i: Optional[int] = None
    for i, det in enumerate(result.detections):
        # Defer the target so it is drawn last (on top) in the highlight colour.
        if target_i is None and _is_target_item(det, target_box, target_bbox):
            target_i = i
            continue
        colour = _colour(i)
        x, y, w, h = det.bbox
        _draw_box(draw_ctx, x, y, w, h, colour, thickness=2)
        label = "%s %.0f%%" % (det.cls, det.conf * 100)
        _draw_label(draw_ctx, x, y, label, colour, ImageFont)
    if target_i is not None:
        det = result.detections[target_i]
        x, y, w, h = det.bbox
        _draw_box(draw_ctx, x, y, w, h, _TARGET_BOX_COLOUR, thickness=4)
        _draw_label(draw_ctx, x, y, "%s %.0f%%" % (det.cls, det.conf * 100), _TARGET_BOX_COLOUR, ImageFont)
    return img


def _draw_box(
    draw_ctx: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    colour: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw a rectangle outline given top-left (x, y) and size (w, h)."""
    x0, y0 = int(x), int(y)
    x1, y1 = int(x + w), int(y + h)
    for t in range(thickness):
        draw_ctx.rectangle(
            [x0 - t, y0 - t, x1 + t, y1 + t],
            outline=colour + (255,),
        )


def _draw_label(
    draw_ctx: Any,
    x: float,
    y: float,
    text: str,
    colour: Tuple[int, int, int],
    ImageFont: Any,
) -> None:
    """Draw a label string just above (x, y) with a darkened background band."""
    font = _load_font(ImageFont, size=14)
    # Estimate text size (font.getbbox is available in Pillow >= 9.2).
    try:
        bbox = font.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        tw, th = len(text) * 8, 14  # rough fallback

    tx = int(x)
    ty = max(0, int(y) - th - 4)

    # Semi-transparent dark background rectangle.
    draw_ctx.rectangle(
        [tx, ty, tx + tw + 4, ty + th + 4],
        fill=colour + (200,),
    )
    draw_ctx.text((tx + 2, ty + 2), text, fill=(255, 255, 255, 255), font=font)


# ---------------------------------------------------------------------------
# Segmentation masks
# ---------------------------------------------------------------------------

def _draw_masks(
    result: "Result",
    img: Any,
    alpha: float,
    Image: Any,
    ImageDraw: Any,
    ImageFont: Any,
    target_box: Any = None,
) -> Any:
    w_img, h_img = img.size
    target_bbox = _target_bbox(target_box)

    for i, mask in enumerate(result.masks):
        colour = _colour(i)
        is_target = _is_target_item(mask, target_box, target_bbox)
        # Decode the RLE — requires numpy.
        try:
            arr = mask.to_ndarray(w_img, h_img)
        except ImportError:
            # numpy not available: fall back to drawing only the bbox outline.
            arr = None
        except ValueError:
            arr = None

        if arr is not None:
            # Build an RGBA overlay where foreground pixels = colour at given alpha.
            try:
                import numpy as np

                overlay = Image.new("RGBA", (w_img, h_img), (0, 0, 0, 0))
                overlay_arr = np.array(overlay)
                a_val = int(alpha * 255)
                overlay_arr[arr, 0] = colour[0]
                overlay_arr[arr, 1] = colour[1]
                overlay_arr[arr, 2] = colour[2]
                overlay_arr[arr, 3] = a_val
                overlay = Image.fromarray(overlay_arr, "RGBA")
                img = Image.alpha_composite(img, overlay)
            except ImportError:
                pass  # no numpy — bbox-only fallback below

        # Draw bbox outline + label (red + thicker when this is the target).
        draw_ctx = ImageDraw.Draw(img)
        x, y, bw, bh = mask.bbox
        box_colour = _TARGET_BOX_COLOUR if is_target else colour
        _draw_box(draw_ctx, x, y, bw, bh, box_colour, thickness=4 if is_target else 2)
        label = "mask %.0f%%" % (mask.conf * 100)
        _draw_label(draw_ctx, x, y, label, box_colour, ImageFont)

    return img


# ---------------------------------------------------------------------------
# Grasps
# ---------------------------------------------------------------------------

def _quality_colour(q: float) -> Tuple[int, int, int]:
    """Map grasp quality in [0,1] to a red→yellow→green colour."""
    q = 0.0 if q < 0.0 else 1.0 if q > 1.0 else q
    if q < 0.5:
        # red → yellow
        t = q / 0.5
        return (255, int(255 * t), 0)
    # yellow → green
    t = (q - 0.5) / 0.5
    return (int(255 * (1.0 - t)), 255, 0)


def _grasp_object_key(g: Any, objects: List[Tuple[float, float, float, float]]) -> Any:
    """Return the index of the SMALLEST object bbox whose interior contains the grasp
    centre, or ``None`` if no bbox contains it."""
    best = None
    best_area: Optional[float] = None
    for i, (x, y, w, h) in enumerate(objects):
        if x <= g.x <= x + w and y <= g.y <= y + h:
            area = w * h
            if best_area is None or area < best_area:
                best_area = area
                best = i
    return best


def _grasps_per_object(result: "Result", max_per_object: Optional[int]) -> List[Any]:
    """Group grasps by the object (detection bbox, else mask bbox) containing each
    grasp centre and keep the ``max_per_object`` highest-quality grasps per group.

    Falls back to grouping by class label when no object bbox contains a grasp (and
    to a single group when there are no objects at all). ``None``/``<=0`` keeps all.
    """
    grasps = result.grasps
    if max_per_object is None or max_per_object <= 0:
        return list(grasps)

    # Prefer detection bboxes (class-aware); else mask bboxes (class-agnostic automask).
    objects = [d.bbox for d in result.detections] or [m.bbox for m in result.masks]

    groups: dict = {}
    for g in grasps:
        key = _grasp_object_key(g, objects) if objects else None
        if key is None:
            key = ("cls", g.cls)  # ungrouped → bucket by label so we still sample per kind
        groups.setdefault(key, []).append(g)

    out: List[Any] = []
    for gs in groups.values():
        gs.sort(key=lambda g: g.quality, reverse=True)
        out.extend(gs[:max_per_object])
    return out


def _draw_grasps(
    result: "Result",
    img: Any,
    ImageDraw: Any,
    ImageFont: Any,
    max_per_object: Optional[int] = 3,
    target_grasp: Optional[Any] = None,
) -> Any:
    """Draw each grasp as a parallel-jaw gripper glyph.

    The glyph is the standard grasp-rectangle: a CLOSING line through the centre
    along ``theta`` of length ``width`` (the two jaw-contact points at its ends),
    plus a short JAW PLATE drawn perpendicular at each contact.

    The ``target_grasp`` (identified by object identity via ``is``) is drawn in
    solid red (255, 0, 0). All other grasps are coloured by quality
    (red→yellow→green via :func:`_quality_colour`).

    Only the top ``max_per_object`` grasps per object are drawn (see
    :func:`_grasps_per_object`) to keep the overlay legible.
    """
    import math

    _TARGET_COLOUR: Tuple[int, int, int, int] = (255, 0, 0, 255)

    def _draw_one(draw_ctx: Any, g: Any, is_target: bool) -> None:
        colour: Tuple[int, int, int, int] = _TARGET_COLOUR if is_target else _quality_colour(g.quality) + (255,)
        label_colour: Tuple[int, int, int] = (255, 0, 0) if is_target else _quality_colour(g.quality)
        line_width = 3 if is_target else 2

        cos_t, sin_t = math.cos(g.theta), math.sin(g.theta)
        hw = g.width / 2.0
        c0 = (g.x - cos_t * hw, g.y - sin_t * hw)
        c1 = (g.x + cos_t * hw, g.y + sin_t * hw)
        plate = max(6.0, min(g.width * 0.35, 22.0))
        px, py = -sin_t * plate / 2.0, cos_t * plate / 2.0

        draw_ctx.line([c0, c1], fill=colour, width=line_width)
        draw_ctx.line([(c0[0] - px, c0[1] - py), (c0[0] + px, c0[1] + py)], fill=colour, width=line_width + 1)
        draw_ctx.line([(c1[0] - px, c1[1] - py), (c1[0] + px, c1[1] + py)], fill=colour, width=line_width + 1)
        r = 3 if is_target else 2
        draw_ctx.ellipse([g.x - r, g.y - r, g.x + r, g.y + r], fill=colour)

        label = ("%s " % g.cls if g.cls else "") + "q%.2f" % g.quality
        _draw_label(draw_ctx, g.x, g.y - 4, label, label_colour, ImageFont)

    draw_ctx = ImageDraw.Draw(img)
    grasps_to_draw = _grasps_per_object(result, max_per_object)

    # Draw non-target grasps first, then the target on top so it is always visible.
    for g in grasps_to_draw:
        if target_grasp is None or g is not target_grasp:
            _draw_one(draw_ctx, g, False)
    if target_grasp is not None:
        for g in grasps_to_draw:
            if g is target_grasp:
                _draw_one(draw_ctx, g, True)
                break

    return img


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _draw_classifications(
    result: "Result", img: Any, ImageDraw: Any, ImageFont: Any
) -> Any:
    draw_ctx = ImageDraw.Draw(img)
    font = _load_font(ImageFont, size=16)
    margin_x, margin_y, line_height = 20, 20, 30
    for i, clf in enumerate(result.classifications):
        colour = _colour(i)
        label = "%s %.0f%%" % (clf.cls, clf.conf * 100)
        ty = margin_y + i * line_height
        draw_ctx.text((margin_x, ty), label, fill=colour + (255,), font=font)
    return img


# ---------------------------------------------------------------------------
# Depth map
# ---------------------------------------------------------------------------

def _draw_depth(result: "Result", Image: Any) -> Any:
    """Render the depth map as a turbo-style colourmap image."""
    dw = result.depth_width
    dh = result.depth_height
    depth = result.depth_map

    if not depth or dw <= 0 or dh <= 0:
        # Nothing to render — return a small black square.
        return Image.new("RGB", (1, 1), (0, 0, 0))

    # Normalise to [0, 1].
    d_min = min(depth)
    d_max = max(depth)
    d_range = d_max - d_min if d_max > d_min else 1.0
    normalised = [(v - d_min) / d_range for v in depth]

    # Apply turbo-style colormap.
    pixels: List[Tuple[int, int, int]] = [_turbo_colour(t) for t in normalised]

    # Construct the image.
    cm_img = Image.new("RGB", (dw, dh))
    cm_img.putdata(pixels)  # type: ignore[arg-type]

    return cm_img


def _turbo_colour(t: float) -> Tuple[int, int, int]:
    """Map a value in [0, 1] to an RGB colour using a turbo-style ramp.

    Segments:
        0.00–0.25: blue  → cyan
        0.25–0.50: cyan  → green
        0.50–0.75: green → yellow
        0.75–1.00: yellow → red
    """
    t = max(0.0, min(1.0, t))
    if t < 0.25:
        s = t / 0.25
        r = 0
        g = int(s * 255)
        b = 255
    elif t < 0.5:
        s = (t - 0.25) / 0.25
        r = 0
        g = 255
        b = int((1.0 - s) * 255)
    elif t < 0.75:
        s = (t - 0.5) / 0.25
        r = int(s * 255)
        g = 255
        b = 0
    else:
        s = (t - 0.75) / 0.25
        r = 255
        g = int((1.0 - s) * 255)
        b = 0
    return (r, g, b)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_image(image: Any, Image: Any) -> Any:
    """Accept PIL.Image, str path, bytes, or numpy ndarray; always return a PIL.Image."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, str):
        return Image.open(image)
    if isinstance(image, (bytes, bytearray)):
        import io as _io
        return Image.open(_io.BytesIO(bytes(image)))
    # numpy ndarray — HWC uint8 or float [0,1]
    try:
        import numpy as np
        if isinstance(image, np.ndarray):
            a = image
            if a.dtype.kind == "f":
                a = np.clip(a, 0.0, 1.0)
                a = (a * 255.0 + 0.5).astype(np.uint8)
            elif a.dtype != np.uint8:
                a = np.clip(a, 0, 255).astype(np.uint8)
            if a.ndim == 2:
                a = np.stack([a, a, a], axis=-1)
            return Image.fromarray(a)
    except ImportError:
        pass
    raise TypeError(
        "unsupported image type %r; expected PIL.Image, str path, bytes, or numpy.ndarray"
        % type(image)
    )


def _load_font(ImageFont: Any, size: int = 14) -> Any:
    """Try to load a reasonably sized font, falling back to the default bitmap font."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except (IOError, OSError):
        pass
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except (IOError, OSError):
        pass
    # Pillow built-in bitmap font (no size argument).
    return ImageFont.load_default()
