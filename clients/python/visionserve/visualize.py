"""Visualization helpers for VisionServe prediction results.

Requires **Pillow** (``pip install pillow`` or ``pip install 'visionserve[images]'``).
Pillow is imported lazily so that the rest of the SDK can be used without it.

Public API::

    from visionserve.visualize import draw, draw_prompts

    annotated = draw(result, "photo.jpg")
    annotated.save("out.jpg")
    draw_prompts("photo.jpg", boxes=[10, 20, 100, 80], points=[[60, 50, 1]]).save("prompt.jpg")
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from .types import Result

RGB = Tuple[int, int, int]

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
# Index palette (color_by="index") — identical to the Go server's overlay palette.
_PALETTE: List[RGB] = [
    (255, 59, 59),
    (255, 165, 0),
    (50, 205, 50),
    (0, 191, 255),
    (238, 130, 238),
    (255, 215, 0),
    (0, 255, 127),
    (255, 99, 71),
]

# Class palette (color_by="class"): 16 well-separated colours (Trubetskoy's list without the
# near-white, grey, black, navy and maroon entries, which vanish on photos or under white text).
# A class name picks one by a stable hash, so "dog" has the same colour in every picture, frame
# and process.
_CLASS_PALETTE: List[RGB] = [
    (230, 25, 75),    # red
    (60, 180, 75),    # green
    (255, 225, 25),   # yellow
    (0, 130, 200),    # blue
    (245, 130, 48),   # orange
    (145, 30, 180),   # purple
    (70, 240, 240),   # cyan
    (240, 50, 230),   # magenta
    (210, 245, 60),   # lime
    (250, 190, 212),  # pink
    (0, 128, 128),    # teal
    (220, 190, 255),  # lavender
    (170, 110, 40),   # brown
    (170, 255, 195),  # mint
    (128, 128, 0),    # olive
    (255, 215, 180),  # apricot
]

_COLOR_BY = ("class", "index")
_DEPTH_MODES = ("map", "overlay", "side", "none")
_CLASS_MODES = ("text", "bars", "none")


def _colour(idx: int) -> RGB:
    return _PALETTE[idx % len(_PALETTE)]


def _fnv1a(text: str) -> int:
    """32-bit FNV-1a of the UTF-8 bytes: a stable hash (Python's ``hash()`` is salted per process)
    that is a few lines in any language."""
    h = 0x811C9DC5
    for b in text.encode("utf-8"):
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def _class_colour_map(names: Sequence[str]) -> dict:
    """``{class: colour}`` for the classes of one picture. Each class starts at
    ``_CLASS_PALETTE[fnv1a_32(utf8(name)) % 16]``, so it keeps its colour from picture to
    picture; when two classes of one picture land on the same colour, the one later in
    alphabetical order takes the next free colour (up to 16 classes are always distinct)."""
    n = len(_CLASS_PALETTE)
    used: set = set()
    out = {}
    for name in sorted(set(names)):
        idx = _fnv1a(name) % n
        while idx in used and len(used) < n:
            idx = (idx + 1) % n
        used.add(idx)
        out[name] = _CLASS_PALETTE[idx]
    return out


def _text_colour(bg: RGB) -> RGB:
    """Black or white, whichever reads better on *bg*."""
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return (0, 0, 0) if lum > 150 else (255, 255, 255)


# Highlight colour for the selected target box (matches the target-grasp red).
_TARGET_BOX_COLOUR: RGB = (255, 0, 0)


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
# Sizes
# ---------------------------------------------------------------------------

def _auto_sizes(width: int, height: int, font_size: Optional[int] = None,
                line_width: Optional[int] = None) -> Tuple[int, int]:
    """``(font_px, line_px)`` for a picture of this size: proportional to its SHORTER side,
    clamped, so labels read the same on a 320 px thumbnail and a 4000 x 3000 photo. A 640 x 426
    photo gets 14 and 2 (the sizes before scaling existed). Explicit values win."""
    short = max(1, min(int(width), int(height)))
    font = int(font_size) if font_size is not None else max(12, min(160, int(round(short / 30.0))))
    lw = int(line_width) if line_width is not None else max(1, min(40, int(round(short / 210.0))))
    if font < 1 or lw < 1:
        raise ValueError("font_size and line_width must be >= 1, got %r and %r" % (font_size, line_width))
    return font, lw


class _Style:
    """What every drawing step shares: sizes, the colour rule and the font loader."""

    def __init__(self, font_px: int, lw: int, color_by: str, ImageFont: Any, result: Any = None):
        self.font_px = font_px
        self.lw = lw
        self.color_by = color_by
        self._ImageFont = ImageFont
        names = []
        if result is not None and color_by == "class":
            names = [d.cls for d in result.detections if d.cls]
            names += [c.cls for c in result.classifications if c.cls]
        self._class_cols = _class_colour_map(names)

    def font(self, size: Optional[int] = None) -> Any:
        return _load_font(self._ImageFont, int(size or self.font_px))

    def item_colour(self, idx: int, cls: str = "") -> RGB:
        if self.color_by == "class" and cls:
            col = self._class_cols.get(cls)
            return col if col is not None else _CLASS_PALETTE[_fnv1a(cls) % len(_CLASS_PALETTE)]
        return _colour(idx)


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
    mask_boxes: bool = True,
    color_by: str = "class",
    mask_outline: bool = False,
    font_size: Optional[int] = None,
    line_width: Optional[int] = None,
    depth: str = "map",
    depth_colormap: Optional[Callable[[Any], Any]] = None,
    classes: str = "text",
    title: Optional[str] = None,
) -> Any:
    """Draw *result* predictions on *image* and return an annotated PIL image.

    What is drawn: masks as translucent colour fills, detection boxes with ``"class conf%"``
    labels on top of them, then a box and a ``"mask conf%"`` label for each mask that has no
    detection with the same box (Grounded-SAM and the grasp pipelines copy the detection's box
    onto its mask, so such a mask is labelled once, with its class), grasps, and the
    classification labels in the top-left corner. With the default ``depth="map"`` a ``depth``
    result is returned as a colour picture of the depth map alone (blue = far, red = near) at the
    MODEL's resolution (``depth_width x depth_height``); *image* is not drawn on.

    Args:
        result: :class:`~visionserve.Result` from :meth:`~visionserve.Client.predict`.
        image:  Input image — one of:

                * ``PIL.Image.Image`` (used directly, not modified in-place),
                * ``str`` / ``pathlib.Path`` file path,
                * ``bytes`` raw encoded image data,
                * ``numpy.ndarray`` ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)`` or ``(H, W, 4)``.

                A JPEG given as a path or bytes is turned upright by its EXIF orientation tag,
                as the server does, so the boxes land on the right pixels. A PIL image or an
                array is drawn as it is (the SDK uploads those without the tag).
        alpha:  Opacity for mask colour overlays (0.0 = transparent, 1.0 = opaque); also the
                strength of the depth colours with ``depth="overlay"``.
        max_grasps_per_object: For a ``grasp`` result, draw at most this many
                highest-quality grasps PER object (grouped by the detection each grasp was
                planned on, exactly as :meth:`~visionserve.Result.filter_grasps` does) instead
                of all of them — the analytic search can return hundreds per object. ``None``
                or ``<= 0`` draws every grasp. Defaults to 3.
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
        mask_boxes: ``False`` draws masks as colour fills only, without their box and label
                (for the dozens of overlapping masks of an automatic-mask result).
        color_by: ``"class"`` (default): every item of one class gets the same colour, chosen
                by a stable hash of the class name, so a class keeps its colour across pictures
                and frames (two classes of one picture that hash to the same colour are told
                apart: the one later in alphabetical order takes the next free colour); items
                without a class (automatic or box-prompted masks) fall back to their index. ``"index"``: item ``i`` gets
                colour ``i`` of the server's 8-colour overlay palette (the behaviour before
                0.3.1).
        mask_outline: ``True`` also draws each mask's contour in its colour (numpy only, no
                OpenCV), so touching masks of one colour stay apart.
        font_size, line_width: label font size and box line width in pixels. ``None`` scales
                both with the photo's shorter side (14 and 2 on a 640 x 426 photo, about 100
                and 14 on a 4000 x 3000 one, at least 12 and 1).
        depth:  How a depth map is shown. ``"map"`` (default): a ``depth`` result is the depth
                map alone at the model's size, as described above. ``"side"``: the photo with
                the depth map, stretched to the photo's size, to its right, with a
                ``far … near`` colour bar (relative depth). ``"overlay"``: the depth colours
                blended onto the photo at ``alpha``, with the colour bar. ``"none"``: the depth
                map is not drawn (a ``depth`` result gives the photo back).
        depth_colormap: Your own colour map for the depth picture and its colour bar: a function
                from a float array of values in ``[0, 1]`` (0 = far, 1 = near) to RGB, ``uint8``
                0-255 or float 0-1 (an extra alpha channel is dropped), e.g.
                ``matplotlib.colormaps["inferno"]``. Needs numpy. ``None``: blue (far) to red
                (near).
        classes: How classifications are shown. ``"text"`` (default): the labels in the
                top-left corner on a dark band. ``"bars"``: a panel to the right of the photo
                with the top 5 labels as bars. ``"none"``: not drawn.
        title:  Text for a dark band added above the picture (all panels). ``None``: no band.

    Returns:
        Annotated ``PIL.Image.Image`` (RGB). It has the photo's size unless ``depth="side"``,
        ``classes="bars"`` or ``title`` add a panel or a band.

    Raises:
        ImportError: if Pillow is not installed.
        ValueError: for an unknown ``color_by`` / ``depth`` / ``classes`` value, or a
            ``font_size`` / ``line_width`` below 1.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ImportError(
            "visionserve.visualize.draw() requires Pillow. "
            "Install with: pip install pillow  (or pip install 'visionserve[images]')"
        ) from exc
    for name, value, allowed in (("color_by", color_by, _COLOR_BY), ("depth", depth, _DEPTH_MODES),
                                 ("classes", classes, _CLASS_MODES)):
        if value not in allowed:
            raise ValueError("%s must be one of %s, got %r" % (name, ", ".join(map(repr, allowed)), value))

    img = _open_image(image, Image)
    font_px, lw = _auto_sizes(img.size[0], img.size[1], font_size, line_width)
    style = _Style(font_px, lw, color_by, ImageFont, result)

    # A depth result with the default depth="map": the depth map alone, at the model's size.
    task = (result.task or "").lower()
    if task == "depth" and depth == "map":
        out = _draw_depth(result, Image, depth_colormap)
        if title:
            f, _ = _auto_sizes(out.size[0], out.size[1], font_size, line_width)
            out = _add_title(out, title, _Style(f, lw, color_by, ImageFont), Image, ImageDraw)
        return out

    # Start with a working copy so we don't mutate the caller's image.
    img = img.convert("RGBA")
    has_depth = depth in ("overlay", "side") and _has_depth(result)
    depth_pic = _draw_depth(result, Image, depth_colormap).resize(img.size, Image.BILINEAR) if has_depth else None
    if depth == "overlay" and depth_pic is not None:
        img = Image.blend(img, depth_pic.convert("RGBA"), max(0.0, min(1.0, alpha)))

    # Does the target box correspond to one of this result's own detections/masks?
    target_bbox = _target_bbox(target_box)
    target_matched = target_box is not None and any(
        _is_target_item(it, target_box, target_bbox)
        for it in (list(result.detections) + list(result.masks))
    )

    # Mask fills first, so the boxes and labels drawn after them stay sharp.
    mask_cols = _mask_colours(result, style)
    if result.masks:
        outline = max(1, lw // 2) if mask_outline else 0
        img = _fill_masks(result, img, alpha, Image, mask_cols, outline)

    if result.detections:
        img = _draw_detections(result, img, ImageDraw, style, target_box)

    if result.masks and (mask_boxes or target_box is not None):
        # With mask_boxes=False only a target mask still gets its (red) box.
        img = _draw_mask_boxes(result, img, ImageDraw, style, mask_cols, target_box, all_boxes=mask_boxes)

    if result.grasps:
        img = _draw_grasps(result, img, ImageDraw, style, max_grasps_per_object, target_grasp)

    if result.classifications and classes == "text":
        img = _draw_classifications(result, img, ImageDraw, style)

    # A target box that is not one of this result's own items (e.g. selected on a
    # separate detection result) is drawn as a standalone red highlight.
    if target_bbox is not None and not target_matched:
        x, y, w, h = target_bbox
        _draw_box(ImageDraw.Draw(img), x, y, w, h, _TARGET_BOX_COLOUR, thickness=2 * lw)

    out = img.convert("RGB")
    panels = [out]
    if depth_pic is not None:
        if depth == "overlay":
            _draw_depth_legend(out, style, ImageDraw, depth_colormap)
        else:
            _draw_depth_legend(depth_pic, style, ImageDraw, depth_colormap)
            panels.append(depth_pic)
    if classes == "bars" and result.classifications:
        panels.append(_bars_panel(result, out.size[1], style, Image, ImageDraw))
    if len(panels) > 1:
        out = _hstack(panels, 4 * lw, Image)
    if title:
        out = _add_title(out, title, style, Image, ImageDraw)
    return out


def draw_prompts(
    image: Any,
    boxes: Any = None,
    points: Any = None,
    labels: Optional[Sequence[int]] = None,
    *,
    line_width: Optional[int] = None,
) -> Any:
    """Draw the prompts you send with ``predict(box=..., point=...)`` on a copy of *image*.

    Args:
        image:  the photo, as for :func:`draw` (a JPEG path or bytes is turned upright by EXIF).
        boxes:  one box ``[x, y, w, h]`` or a list of boxes (a numpy array works), in the
                photo's pixels, drawn as black-and-white dashed rectangles.
        points: one point ``[x, y]`` / ``[x, y, label]`` or a list of them — the same format as
                ``predict(point=...)``. Label 1 (the default) is drawn as a green dot ("on the
                object"), label 0 as a red cross ("not on the object").
        labels: one label per point, instead of a third value in each point.
        line_width: line width in pixels; ``None`` scales it with the photo, as in :func:`draw`.

    Returns:
        A new ``PIL.Image.Image`` (RGB) of the photo's size; *image* is not changed.

    Raises:
        ImportError: if Pillow is not installed.
        ValueError: a box without 4 values, a point without 2 or 3, or ``labels`` of another
            length than ``points``.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError(
            "visionserve.visualize.draw_prompts() requires Pillow. "
            "Install with: pip install pillow  (or pip install 'visionserve[images]')"
        ) from exc
    from .client import _normalize_list  # the same box/point parsing as predict()

    img = _open_image(image, Image).convert("RGB")  # convert() is a copy: the caller's is kept
    _, lw = _auto_sizes(img.size[0], img.size[1], None, line_width)
    d = ImageDraw.Draw(img)
    box_list = _normalize_list(boxes)
    point_list = _normalize_list(points)
    if labels is not None:
        labels = list(labels)
        if len(labels) != len(point_list):
            raise ValueError("labels has %d values for %d points" % (len(labels), len(point_list)))

    dash = 4 * lw
    for b in box_list:
        if len(b) != 4:
            raise ValueError("box must have 4 values [x,y,w,h], got %r" % (b,))
        x, y, bw, bh = (float(v) for v in b)
        corners = [(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh), (x, y)]
        for (x0, y0), (x1, y1) in zip(corners, corners[1:]):
            d.line([(x0, y0), (x1, y1)], fill=(0, 0, 0), width=lw)  # dark gaps: visible on white
            n = max(1, int(max(abs(x1 - x0), abs(y1 - y0)) // dash))
            for k in range(0, n, 2):
                a, c = k / n, min(1.0, (k + 1) / n)
                d.line([(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a),
                        (x0 + (x1 - x0) * c, y0 + (y1 - y0) * c)], fill=(255, 255, 255), width=lw)

    r = max(5, 3 * lw)
    for i, p in enumerate(point_list):
        if len(p) not in (2, 3):
            raise ValueError("point must have 2 or 3 values [x,y[,label]], got %r" % (p,))
        x, y = float(p[0]), float(p[1])
        label = int(labels[i]) if labels is not None else (int(p[2]) if len(p) > 2 else 1)
        if label == 1:
            d.ellipse([x - r, y - r, x + r, y + r], fill=(0, 220, 0), outline=(255, 255, 255),
                      width=max(1, lw))
        else:
            for (ax, ay), (bx, by) in (((x - r, y - r), (x + r, y + r)), ((x - r, y + r), (x + r, y - r))):
                d.line([(ax, ay), (bx, by)], fill=(255, 0, 0), width=lw + 2)
    return img


# ---------------------------------------------------------------------------
# Detections / open-vocab
# ---------------------------------------------------------------------------

def _draw_detections(result: "Result", img: Any, ImageDraw: Any, style: _Style, target_box: Any = None) -> Any:
    draw_ctx = ImageDraw.Draw(img)
    target_bbox = _target_bbox(target_box)
    target_i: Optional[int] = None
    lw = style.lw
    for i, det in enumerate(result.detections):
        # Defer the target so it is drawn last (on top) in the highlight colour.
        if target_i is None and _is_target_item(det, target_box, target_bbox):
            target_i = i
            continue
        colour = style.item_colour(i, det.cls)
        x, y, w, h = det.bbox
        _draw_box(draw_ctx, x, y, w, h, colour, thickness=lw)
        label = "%s %.0f%%" % (det.cls, det.conf * 100)
        if getattr(det, "track_id", None) is not None:  # Client.watch(track=True)
            label += " #%d" % det.track_id
        _draw_label(draw_ctx, img.size, x, y, label, colour, style)
    if target_i is not None:
        det = result.detections[target_i]
        x, y, w, h = det.bbox
        _draw_box(draw_ctx, x, y, w, h, _TARGET_BOX_COLOUR, thickness=2 * lw)
        _draw_label(draw_ctx, img.size, x, y, "%s %.0f%%" % (det.cls, det.conf * 100), _TARGET_BOX_COLOUR, style)
    return img


def _draw_box(
    draw_ctx: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    colour: RGB,
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


def _text_size(draw_ctx: Any, text: str, font: Any) -> Tuple[int, int, int, int]:
    """``(left, top, width, height)`` of *text*'s ink, relative to the drawing origin."""
    try:
        l, t, r, b = draw_ctx.textbbox((0, 0), text, font=font)
        return l, t, r - l, b - t
    except AttributeError:  # Pillow < 8: rough size
        size = getattr(font, "size", 11)
        return 0, 0, int(len(text) * size * 0.6), size


def _draw_label(
    draw_ctx: Any,
    img_size: Tuple[int, int],
    x: float,
    y: float,
    text: str,
    colour: RGB,
    style: _Style,
) -> None:
    """A label on a band of *colour* just above (x, y), kept inside the picture (at the top
    edge it overlaps the box). Black or white text, whichever reads better."""
    font = style.font()
    pad = max(2, style.font_px // 7)
    l, t, tw, th = _text_size(draw_ctx, text, font)
    bw, bh = tw + 2 * pad, th + 2 * pad
    tx = int(max(0, min(x, img_size[0] - bw)))
    ty = int(max(0, int(y) - bh))
    draw_ctx.rectangle([tx, ty, tx + bw, ty + bh], fill=colour + (255,))
    draw_ctx.text((tx + pad - l, ty + pad - t), text, fill=_text_colour(colour) + (255,), font=font)


# ---------------------------------------------------------------------------
# Segmentation masks
# ---------------------------------------------------------------------------

def _mask_colours(result: "Result", style: _Style) -> List[RGB]:
    """Colour of each mask: with ``color_by="class"`` the class of the detection with the same
    box (Grounded-SAM / grasp pipelines copy it onto the mask), else colour ``i``."""
    box_cls = {}
    for d in result.detections:
        box_cls.setdefault(tuple(float(v) for v in d.bbox), d.cls)
    return [style.item_colour(i, box_cls.get(tuple(float(v) for v in m.bbox), ""))
            for i, m in enumerate(result.masks)]


def _erode(m: Any) -> Any:
    """A one-pixel (4-neighbour) erosion of a boolean array; outside the array counts as set."""
    out = m.copy()
    out[1:, :] &= m[:-1, :]
    out[:-1, :] &= m[1:, :]
    out[:, 1:] &= m[:, :-1]
    out[:, :-1] &= m[:, 1:]
    return out


def _mask_edge(arr: Any, width: int) -> Optional[Tuple[int, int, Any]]:
    """The inner contour band (*width* px) of a bool mask, cropped to the mask:
    ``(y0, x0, band)``, or ``None`` for an empty mask."""
    import numpy as np

    rows = np.flatnonzero(arr.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(arr.any(axis=0))
    y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    m = np.pad(arr[y0:y1, x0:x1], width, constant_values=False)  # the mask's own border is an edge
    er = m
    for _ in range(width):
        er = _erode(er)
    band = (m & ~er)[width:-width, width:-width]
    return y0, x0, band


def _fill_masks(result: "Result", img: Any, alpha: float, Image: Any,
                colours: Optional[List[RGB]] = None, outline: int = 0) -> Any:
    """Composite every mask's colour onto *img* (RGBA), and its contour (*outline* px, 0 = none)
    on top of all fills. ``colours[i]`` colours mask ``i`` (default: palette colour ``i``)."""
    w_img, h_img = img.size
    # All masks go into ONE RGBA overlay composited once (a full-image composite per mask was
    # O(masks x pixels)); where masks overlap, the later one's colour wins.
    try:
        import numpy as np
    except ImportError:  # no numpy: bbox outlines only
        return img
    if colours is None:
        colours = [_colour(i) for i in range(len(result.masks))]
    overlay_arr = np.zeros((h_img, w_img, 4), np.uint8)
    a_val = int(round(max(0.0, min(1.0, alpha)) * 255))
    any_mask = False
    edges = []
    for i, mask in enumerate(result.masks):
        try:
            arr = mask.to_ndarray(w_img, h_img)
        except ValueError:  # RLE for another image size: outline only
            continue
        overlay_arr[arr] = colours[i] + (a_val,)
        any_mask = True
        if outline > 0:
            e = _mask_edge(arr, outline)
            if e is not None:
                edges.append((e, colours[i]))
    for (y0, x0, band), col in edges:  # contours after all fills: a later fill cannot hide one
        overlay_arr[y0:y0 + band.shape[0], x0:x0 + band.shape[1]][band] = col + (255,)
    if any_mask:
        img = Image.alpha_composite(img, Image.fromarray(overlay_arr, "RGBA"))
    return img


def _draw_mask_boxes(
    result: "Result",
    img: Any,
    ImageDraw: Any,
    style: _Style,
    colours: List[RGB],
    target_box: Any = None,
    all_boxes: bool = True,
) -> Any:
    """Box + ``"mask conf%"`` label for each mask that no detection already labels.

    A mask whose bbox equals a detection's bbox (Grounded-SAM / grasp pipelines copy the
    detection box onto its mask) was drawn with its class by :func:`_draw_detections`; drawing
    it again would cover that label. ``all_boxes=False`` draws only the target mask's box.
    """
    target_bbox = _target_bbox(target_box)
    det_boxes = {tuple(float(v) for v in d.bbox) for d in result.detections}
    draw_ctx = ImageDraw.Draw(img)
    for i, mask in enumerate(result.masks):
        if tuple(float(v) for v in mask.bbox) in det_boxes:
            continue
        is_target = _is_target_item(mask, target_box, target_bbox)
        if not (all_boxes or is_target):
            continue
        # Draw bbox outline + label (red + thicker when this is the target).
        x, y, bw, bh = mask.bbox
        box_colour = _TARGET_BOX_COLOUR if is_target else colours[i]
        _draw_box(draw_ctx, x, y, bw, bh, box_colour, thickness=2 * style.lw if is_target else style.lw)
        label = "mask %.0f%%" % (mask.conf * 100)
        _draw_label(draw_ctx, img.size, x, y, label, box_colour, style)

    return img


# ---------------------------------------------------------------------------
# Grasps
# ---------------------------------------------------------------------------

def _quality_colour(q: float) -> RGB:
    """Map grasp quality in [0,1] to a red→yellow→green colour."""
    q = 0.0 if q < 0.0 else 1.0 if q > 1.0 else q
    if q < 0.5:
        # red → yellow
        t = q / 0.5
        return (255, int(255 * t), 0)
    # yellow → green
    t = (q - 0.5) / 0.5
    return (int(255 * (1.0 - t)), 255, 0)


def _grasps_per_object(result: "Result", max_per_object: Optional[int]) -> List[Any]:
    """The grasps to draw: the ``max_per_object`` highest-quality grasps per object, grouped
    exactly as :meth:`~visionserve.Result.filter_grasps` does. ``None``/``<=0`` keeps all."""
    return list(result.filter_grasps(max_per_object).grasps)


def _draw_grasps(
    result: "Result",
    img: Any,
    ImageDraw: Any,
    style: _Style,
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

    lw = style.lw
    extra = max(1, lw // 2)

    def _draw_one(draw_ctx: Any, g: Any, is_target: bool) -> None:
        label_colour: RGB = _TARGET_BOX_COLOUR if is_target else _quality_colour(g.quality)
        colour = label_colour + (255,)
        line_width = lw + extra if is_target else lw

        cos_t, sin_t = math.cos(g.theta), math.sin(g.theta)
        hw = g.width / 2.0
        c0 = (g.x - cos_t * hw, g.y - sin_t * hw)
        c1 = (g.x + cos_t * hw, g.y + sin_t * hw)
        plate = max(3.0 * lw, min(g.width * 0.35, 11.0 * lw))
        px, py = -sin_t * plate / 2.0, cos_t * plate / 2.0

        draw_ctx.line([c0, c1], fill=colour, width=line_width)
        draw_ctx.line([(c0[0] - px, c0[1] - py), (c0[0] + px, c0[1] + py)], fill=colour, width=line_width + 1)
        draw_ctx.line([(c1[0] - px, c1[1] - py), (c1[0] + px, c1[1] + py)], fill=colour, width=line_width + 1)
        r = lw + extra if is_target else lw
        draw_ctx.ellipse([g.x - r, g.y - r, g.x + r, g.y + r], fill=colour)

        label = ("%s " % g.cls if g.cls else "") + "q%.2f" % g.quality
        _draw_label(draw_ctx, img.size, g.x, g.y - 2 * lw, label, label_colour, style)

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

def _draw_classifications(result: "Result", img: Any, ImageDraw: Any, style: _Style) -> Any:
    draw_ctx = ImageDraw.Draw(img)
    fs = style.font_px
    font = style.font(int(round(fs * 16 / 14.0)))
    margin, line_height, pad = int(round(fs * 20 / 14.0)), int(round(fs * 30 / 14.0)), max(2, int(round(fs * 4 / 14.0)))
    for i, clf in enumerate(result.classifications):
        colour = style.item_colour(i, clf.cls)
        label = "%s %.0f%%" % (clf.cls, clf.conf * 100)
        ty = margin + i * line_height
        # A dark band behind the text keeps it readable on a light photo (sky, snow, paper).
        try:
            l, t, r, b = draw_ctx.textbbox((margin, ty), label, font=font)
        except AttributeError:  # Pillow < 8: rough size
            l, t, r, b = margin, ty, margin + 9 * len(label), ty + 16
        draw_ctx.rectangle([l - pad, t - pad + 1, r + pad, b + pad - 1], fill=(20, 20, 20, 255))
        draw_ctx.text((margin, ty), label, fill=colour + (255,), font=font)
    return img


def _fit(draw_ctx: Any, text: str, font: Any, max_w: int) -> str:
    """*text*, cut with an ellipsis to at most *max_w* pixels wide."""
    if _text_size(draw_ctx, text, font)[2] <= max_w:
        return text
    while text and _text_size(draw_ctx, text + "…", font)[2] > max_w:
        text = text[:-1]
    return text + "…" if text else ""


def _bars_panel(result: "Result", height: int, style: _Style, Image: Any, ImageDraw: Any) -> Any:
    """A white panel, *height* tall, with the top 5 classifications as horizontal bars."""
    fs = style.font_px
    width, pad = 18 * fs, fs
    panel = Image.new("RGB", (width, max(1, height)), (255, 255, 255))
    d = ImageDraw.Draw(panel)
    font, small = style.font(), style.font(max(8, int(round(fs * 0.85))))
    top = sorted(result.classifications, key=lambda c: c.conf, reverse=True)[:5]
    l, t, _, th = _text_size(d, "Ag", font)
    y = pad
    d.text((pad - l, y - t), "top %d" % len(top), fill=(110, 110, 110), font=small)
    y += th + pad
    bar_h, inner = max(2, int(round(fs * 0.8))), width - 2 * pad
    for i, c in enumerate(top):
        if y + th + fs // 3 + bar_h > height - pad // 2:
            break  # no room for another row (a very large font_size)
        pct = "%.0f%%" % (c.conf * 100)
        pl, pt, pw, _ = _text_size(d, pct, font)
        name = _fit(d, c.cls, font, inner - pw - fs // 2)
        d.text((pad - l, y - t), name, fill=(30, 30, 30), font=font)
        d.text((width - pad - pw - pl, y - pt), pct, fill=(30, 30, 30), font=font)
        y += th + fs // 3
        d.rectangle([pad, y, pad + inner, y + bar_h], fill=(228, 228, 228))
        conf = max(0.0, min(1.0, float(c.conf)))
        if conf > 0:
            d.rectangle([pad, y, pad + max(1, int(round(inner * conf))), y + bar_h],
                        fill=style.item_colour(i, c.cls))
        y += bar_h + int(round(fs * 0.9))
    return panel


# ---------------------------------------------------------------------------
# Panels and bands
# ---------------------------------------------------------------------------

def _hstack(panels: List[Any], gap: int, Image: Any) -> Any:
    """Panels left to right on white, *gap* px apart, top-aligned."""
    w = sum(p.size[0] for p in panels) + gap * (len(panels) - 1)
    h = max(p.size[1] for p in panels)
    out = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.size[0] + gap
    return out


def _add_title(img: Any, title: str, style: _Style, Image: Any, ImageDraw: Any) -> Any:
    """*img* with a dark band holding *title* added on top."""
    fs = style.font_px
    band = int(round(fs * 1.9))
    out = Image.new("RGB", (img.size[0], img.size[1] + band), (32, 32, 32))
    out.paste(img, (0, band))
    d = ImageDraw.Draw(out)
    font = style.font()
    pad = int(round(fs * 0.6))
    text = _fit(d, str(title), font, img.size[0] - 2 * pad)
    l, t, _, th = _text_size(d, text, font)
    d.text((pad - l, (band - th) // 2 - t), text, fill=(255, 255, 255), font=font)
    return out


# ---------------------------------------------------------------------------
# Depth map
# ---------------------------------------------------------------------------

def _has_depth(result: "Result") -> bool:
    return bool(len(result.depth_map)) and result.depth_width > 0 and result.depth_height > 0


def _draw_depth_legend(img: Any, style: _Style, ImageDraw: Any, cmap: Optional[Callable[[Any], Any]] = None) -> None:
    """A ``relative depth: far [blue→red] near`` colour bar in the bottom-right corner (in place)."""
    fs = style.font_px
    d = ImageDraw.Draw(img)
    font = style.font()
    pad = max(3, fs // 3)
    bar_w, bar_h = 6 * fs, max(4, int(round(fs * 0.8)))
    pieces = ["relative depth: far", "near"]
    sizes = [_text_size(d, s, font) for s in pieces]
    th = max(s[3] for s in sizes)
    box_w = sizes[0][2] + bar_w + sizes[1][2] + 4 * pad
    box_h = max(th, bar_h) + 2 * pad
    margin = max(4, fs // 2)
    x0 = max(0, img.size[0] - box_w - margin)
    y0 = max(0, img.size[1] - box_h - margin)
    d.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(25, 25, 25))
    cy = y0 + box_h // 2
    x = x0 + pad
    for k, (text, (l, t, tw, h)) in enumerate(zip(pieces, sizes)):
        d.text((x - l, cy - h // 2 - t), text, fill=(255, 255, 255), font=font)
        x += tw + pad
        if k == 0:
            ts = [j / max(1.0, bar_w - 1.0) for j in range(bar_w)]
            if cmap is not None:
                import numpy as np

                cols = [tuple(int(v) for v in c) for c in _apply_cmap(cmap, np.asarray([ts]))[0]]
            else:
                cols = [_turbo_colour(t) for t in ts]
            for j, col in enumerate(cols):
                d.line([(x + j, cy - bar_h // 2), (x + j, cy + bar_h // 2)], fill=col)
            x += bar_w + pad


def _apply_cmap(cmap: Callable[[Any], Any], t: Any) -> Any:
    """``cmap(t)`` for values *t* in [0, 1] as an ``(..., 3)`` uint8 array (float 0-1 output is
    scaled; a 4th channel is dropped)."""
    import numpy as np

    rgb = np.asarray(cmap(t))
    if rgb.ndim != t.ndim + 1 or rgb.shape[-1] not in (3, 4):
        raise ValueError("depth_colormap must return RGB or RGBA per value, got shape %r" % (rgb.shape,))
    rgb = rgb[..., :3]
    if rgb.dtype.kind == "f":
        rgb = rgb * 255.0
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def _draw_depth(result: "Result", Image: Any, cmap: Optional[Callable[[Any], Any]] = None) -> Any:
    """Render the depth map as a turbo-style colourmap image (or with *cmap*)."""
    dw = result.depth_width
    dh = result.depth_height
    depth = result.depth_map

    if not depth or dw <= 0 or dh <= 0:
        # Nothing to render — return a small black square.
        return Image.new("RGB", (1, 1), (0, 0, 0))

    try:
        import numpy as np
    except ImportError:
        if cmap is not None:
            raise ImportError("depth_colormap needs numpy: pip install numpy") from None
        np = None
    if np is not None:  # vectorised: same ramp as _turbo_colour, ~100x faster than per pixel
        d = np.asarray(depth, np.float64).reshape(dh, dw)  # list or FloatArray (no Python floats)
        lo, hi = float(d.min()), float(d.max())
        t = np.clip((d - lo) / (hi - lo if hi > lo else 1.0), 0.0, 1.0)
        if cmap is not None:
            return Image.fromarray(_apply_cmap(cmap, t), "RGB")
        s = np.where(t < 0.25, t / 0.25, np.where(t < 0.5, (t - 0.25) / 0.25,
                                                   np.where(t < 0.75, (t - 0.5) / 0.25, (t - 0.75) / 0.25)))
        r = np.where(t < 0.5, 0, np.where(t < 0.75, s * 255, 255))
        g = np.where(t < 0.25, s * 255, np.where(t < 0.75, 255, (1.0 - s) * 255))
        b = np.where(t < 0.25, 255, np.where(t < 0.5, (1.0 - s) * 255, 0))
        rgb = np.stack([r, g, b], -1).astype(np.int32).clip(0, 255).astype(np.uint8)
        return Image.fromarray(rgb, "RGB")

    # Normalise to [0, 1].
    d_min = min(depth)
    d_max = max(depth)
    d_range = d_max - d_min if d_max > d_min else 1.0
    normalised = [(v - d_min) / d_range for v in depth]

    # Apply turbo-style colormap.
    pixels: List[RGB] = [_turbo_colour(t) for t in normalised]

    # Construct the image.
    cm_img = Image.new("RGB", (dw, dh))
    cm_img.putdata(pixels)  # type: ignore[arg-type]

    return cm_img


def _turbo_colour(t: float) -> RGB:
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
    import os

    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (str, os.PathLike)):
        return _upright(Image.open(image))
    if isinstance(image, (bytes, bytearray)):
        import io as _io
        return _upright(Image.open(_io.BytesIO(bytes(image))))
    from .client import _maybe_ndarray, _ndarray_to_pil  # the same ndarray rule as predict()

    if _maybe_ndarray(image) is not None:
        return _ndarray_to_pil(image)
    raise TypeError(
        "unsupported image type %r; expected PIL.Image, str path, bytes, or numpy.ndarray"
        % type(image)
    )


def _upright(img: Any) -> Any:
    """A JPEG turned upright by its EXIF orientation tag, as the server decodes it
    (``imaging.AutoOrientation``, JPEG only), so result coordinates match its pixels."""
    if getattr(img, "format", None) != "JPEG":
        return img
    try:
        from PIL import ImageOps

        return ImageOps.exif_transpose(img)
    except Exception:  # noqa: BLE001 — a broken EXIF block: the server ignores it too
        return img


def _load_font(ImageFont: Any, size: int = 14) -> Any:
    """A TrueType font of *size* px (cached), falling back to Pillow's own scalable font
    (Pillow >= 10.1) and then to the fixed-size bitmap font."""
    return _load_font_cached(ImageFont, int(size))


@functools.lru_cache(maxsize=32)
def _load_font_cached(ImageFont: Any, size: int) -> Any:
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: bitmap font, no size argument
        return ImageFont.load_default()

