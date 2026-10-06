"""Tests for the draw() options added in 0.3.1 (sizes, colour by class, mask outlines, title,
depth panels, classification bars) and for draw_prompts()."""

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from visionserve.types import Classification, Detection, Grasp, Mask, Result  # noqa: E402
from visionserve.visualize import (  # noqa: E402
    _CLASS_PALETTE, _PALETTE, _auto_sizes, _class_colour_map, _fnv1a, draw, draw_prompts)


def _rle(mask):
    flat = mask.flatten(order="F")
    runs, cur, n = [], False, 0
    for v in flat:
        if bool(v) == cur:
            n += 1
        else:
            runs.append(n)
            cur, n = bool(v), 1
    runs.append(n)
    return " ".join(map(str, runs))


def _grey(w=200, h=120):
    return Image.new("RGB", (w, h), (128, 128, 128))


def _px(img, x, y):
    return tuple(int(v) for v in np.asarray(img)[y, x])


# ----------------------------------------------------------------------------- sizes
def test_auto_sizes_scale_with_the_shorter_side_and_are_clamped():
    assert _auto_sizes(640, 426) == (14, 2)       # the sizes every figure had before scaling
    assert _auto_sizes(426, 640) == (14, 2)       # shorter side, either orientation
    assert _auto_sizes(4000, 3000) == (100, 14)   # readable when shown at ~800 px wide
    assert _auto_sizes(320, 213) == (12, 1)       # not huge on a thumbnail
    assert _auto_sizes(20, 20) == (12, 1)         # floors
    assert _auto_sizes(100000, 100000) == (160, 40)  # caps
    assert _auto_sizes(640, 426, font_size=30, line_width=7) == (30, 7)
    with pytest.raises(ValueError):
        _auto_sizes(640, 426, line_width=0)


def _box_line_width(img, x, y_mid, colour):
    """Number of consecutive pixels of *colour* going LEFT from column x (the box's left edge)."""
    a = np.asarray(img).astype(int)
    n = 0
    while x - n >= 0 and np.abs(a[y_mid, x - n] - colour).max() <= 2:
        n += 1
    return n


def test_line_width_scales_with_the_photo_and_can_be_overridden():
    det = Detection(bbox=[1000.0, 1000.0, 800.0, 600.0], cls="dog", conf=0.9)
    r = Result(task="detection", model="m", detections=[det])
    col = np.array(_class_colour_map(["dog"])["dog"])
    big = draw(r, _grey(4000, 3000))
    assert _box_line_width(big, 1000, 1300, col) == 14
    assert _box_line_width(draw(r, _grey(4000, 3000), line_width=3), 1000, 1300, col) == 3
    small = Result(task="detection", model="m",
                   detections=[Detection(bbox=[100.0, 60.0, 80.0, 40.0], cls="dog", conf=0.9)])
    assert _box_line_width(draw(small, _grey(320, 213)), 100, 80, col) == 1


def test_label_band_grows_with_font_size():
    det = Detection(bbox=[50.0, 60.0, 80.0, 40.0], cls="dog", conf=0.9)
    r = Result(task="detection", model="m", detections=[det])
    col = np.array(_class_colour_map(["dog"])["dog"])

    def band_height(img):
        a = np.asarray(img).astype(int)
        rows = [y for y in range(0, 60) if (np.abs(a[y, 52] - col).max() <= 2)]
        return len(rows)

    assert band_height(draw(r, _grey(), font_size=24)) > band_height(draw(r, _grey(), font_size=12)) + 8


# ----------------------------------------------------------------------------- colours
def test_color_by_class_gives_one_class_one_colour_and_index_does_not():
    dets = [Detection(bbox=[10.0, 10.0, 40.0, 30.0], cls="dog", conf=0.9),
            Detection(bbox=[100.0, 10.0, 40.0, 30.0], cls="dog", conf=0.8)]
    r = Result(task="detection", model="m", detections=dets)
    a = draw(r, _grey())
    assert _px(a, 10, 30) == _px(a, 100, 30)  # both left edges: the class colour
    b = draw(r, _grey(), color_by="index")
    assert _px(b, 10, 30) == _PALETTE[0] and _px(b, 100, 30) == _PALETTE[1]


def test_class_colour_is_stable_and_colliding_classes_are_told_apart():
    assert _fnv1a("") == 0x811C9DC5 and _fnv1a("a") == 0xE40C292C  # the published FNV-1a vectors
    assert _class_colour_map(["dog"])["dog"] == _CLASS_PALETTE[_fnv1a("dog") % 16]
    # the same class gets the same colour in another picture with other (non-colliding) classes
    assert _class_colour_map(["dog", "person"])["dog"] == _class_colour_map(["dog"])["dog"]
    # "bench" and "dog" hash to the same colour: in one picture they still differ
    assert _fnv1a("bench") % 16 == _fnv1a("dog") % 16
    both = _class_colour_map(["dog", "bench"])
    assert both["bench"] != both["dog"] and both["bench"] == _class_colour_map(["bench"])["bench"]
    many = _class_colour_map(["c%d" % i for i in range(16)])
    assert len(set(many.values())) == 16


def test_unclassed_masks_keep_index_colours_and_classed_masks_take_their_detection_colour():
    w, h = 200, 120
    m0 = np.zeros((h, w), bool)
    m0[20:60, 20:60] = True
    m1 = np.zeros((h, w), bool)
    m1[20:60, 120:160] = True
    det = Detection(bbox=[120.0, 20.0, 40.0, 40.0], cls="cat", conf=0.9)
    r = Result(task="open_vocab", model="m", detections=[det], masks=[
        Mask(rle=_rle(m0), bbox=[20.0, 20.0, 40.0, 40.0], conf=0.9),
        Mask(rle=_rle(m1), bbox=[120.0, 20.0, 40.0, 40.0], conf=0.9)])
    a = draw(r, Image.new("RGB", (w, h), (0, 0, 0)), alpha=1.0, mask_boxes=False)
    assert _px(a, 40, 40) == _PALETTE[0]                         # no class: index colour 0
    assert _px(a, 140, 40) == _class_colour_map(["cat"])["cat"]  # the cat's colour


# ----------------------------------------------------------------------------- mask outline
def test_mask_outline_draws_the_contour_in_full_colour():
    w, h = 200, 120
    m = np.zeros((h, w), bool)
    m[30:90, 40:120] = True
    r = Result(task="segmentation", model="m", masks=[Mask(rle=_rle(m), bbox=[40.0, 30.0, 80.0, 60.0], conf=0.9)])
    black = Image.new("RGB", (w, h), (0, 0, 0))
    plain = draw(r, black, alpha=0.3, mask_boxes=False)
    lined = draw(r, black, alpha=0.3, mask_boxes=False, mask_outline=True)
    assert _px(lined, 40, 60) == _PALETTE[0]    # on the contour: the full colour
    assert _px(plain, 40, 60) != _PALETTE[0]    # without outline: only tinted
    assert _px(lined, 80, 60) == _px(plain, 80, 60)  # the inside is unchanged
    assert _px(lined, 20, 60) == (0, 0, 0)      # outside: untouched
    thick = draw(r, black, alpha=0.3, mask_boxes=False, mask_outline=True, line_width=6)
    assert _px(thick, 42, 60) == _PALETTE[0] and _px(lined, 42, 60) != _PALETTE[0]


# ----------------------------------------------------------------------------- title
def test_title_adds_a_dark_band_on_top():
    r = Result(task="detection", model="m")
    out = draw(r, _grey(), title="rf-detr: 3 detections")
    assert out.size[0] == 200 and out.size[1] > 120
    band = out.size[1] - 120
    assert _px(out, 199, 1) == (32, 32, 32)
    assert _px(out, 100, band + 5) == (128, 128, 128)  # the photo below the band
    # a title longer than the picture is cut, not overflowing
    assert draw(r, _grey(), title="x" * 500).size == out.size


# ----------------------------------------------------------------------------- depth
def _depth_result(task="depth"):
    vals = [float(x) / 7 for y in range(6) for x in range(8)]  # near on the right
    return Result(task=task, model="midas", depth_map=vals, depth_width=8, depth_height=6)


def test_depth_default_is_still_the_map_alone_at_model_size():
    assert draw(_depth_result(), _grey()).size == (8, 6)


def test_depth_side_puts_the_stretched_map_right_of_the_photo_with_a_colour_bar():
    out = draw(_depth_result(), _grey(), depth="side")
    _, lw = _auto_sizes(200, 120)
    assert out.size == (2 * 200 + 4 * lw, 120)
    assert _px(out, 100, 60) == (128, 128, 128)  # the photo, untouched
    left, right = _px(out, 200 + 4 * lw + 2, 10), _px(out, 399, 10)
    assert left[2] > 200 and left[0] < 50        # far = blue
    assert right[0] > 200 and right[2] < 50      # near = red
    a = np.asarray(out).astype(int)
    assert (a[110, 300:395] == 25).all(axis=1).any()  # the legend's dark box, bottom right
    titled = draw(_depth_result(), _grey(), depth="side", title="midas")
    assert titled.size[0] == out.size[0] and titled.size[1] > 120


def test_depth_overlay_and_none():
    over = draw(_depth_result(), _grey(), depth="overlay")
    assert over.size == (200, 120)
    assert _px(over, 5, 10) != (128, 128, 128)
    assert draw(_depth_result(), _grey(), depth="none").size == (200, 120)
    assert np.array_equal(np.asarray(draw(_depth_result(), _grey(), depth="none")), np.asarray(_grey()))
    # no depth map in the result: "side" adds nothing
    assert draw(Result(task="detection", model="m"), _grey(), depth="side").size == (200, 120)


def test_depth_colormap_is_used_for_the_picture_and_the_colour_bar():
    def grey(t):  # float 0-1 RGBA, like a matplotlib colour map
        return np.stack([t, t, t, np.ones_like(t)], -1)

    out = draw(_depth_result(), _grey(), depth="side", depth_colormap=grey)
    _, lw = _auto_sizes(200, 120)
    assert _px(out, 200 + 4 * lw + 2, 10) == (0, 0, 0)       # far = black
    assert _px(out, 399, 10) == (255, 255, 255)              # near = white
    a = np.asarray(out).astype(int)
    bar_row = a[105:115, 210:400]
    assert ((bar_row.max(axis=2) - bar_row.min(axis=2)) == 0).all()  # the bar is grey too: no hue
    m = draw(_depth_result(), _grey(), depth_colormap=lambda t: (np.stack([t] * 3, -1) * 255).astype(np.uint8))
    assert m.size == (8, 6) and _px(m, 7, 0) == (255, 255, 255)
    with pytest.raises(ValueError):
        draw(_depth_result(), _grey(), depth_colormap=lambda t: t)  # not RGB


# ----------------------------------------------------------------------------- classification
def test_classes_bars_adds_a_panel_and_none_draws_nothing():
    r = Result(task="classification", model="m", classifications=[
        Classification("tusker", 0.46), Classification("African elephant", 0.45),
        Classification("Indian elephant", 0.02)])
    bars = draw(r, _grey(), classes="bars")
    font, lw = _auto_sizes(200, 120)
    assert bars.size == (200 + 4 * lw + 18 * font, 120)
    assert np.array_equal(np.asarray(bars)[:, :200], np.asarray(_grey()))  # no text on the photo
    panel = np.asarray(bars)[:, 200 + 4 * lw:].astype(int)
    assert (panel == 255).all(axis=2).mean() > 0.5  # a white panel ...
    assert (np.abs(panel - np.array(_class_colour_map(["tusker", "African elephant",
                                                       "Indian elephant"])["tusker"])).max(axis=2) == 0).any()
    none = draw(r, _grey(), classes="none")
    assert np.array_equal(np.asarray(none), np.asarray(_grey()))
    assert not np.array_equal(np.asarray(draw(r, _grey())), np.asarray(_grey()))  # "text" default


def test_unknown_option_values_are_refused():
    r = Result(task="detection", model="m")
    for kw in ({"color_by": "rainbow"}, {"depth": "beside"}, {"classes": "pie"}):
        with pytest.raises(ValueError):
            draw(r, _grey(), **kw)


def test_grasps_are_still_drawn_and_scaled():
    g = Grasp(x=100, y=60, theta=0.0, width=40, quality=0.9, cls="cup", conf=0.8)
    r = Result(task="grasp", model="m", detections=[Detection([70.0, 40.0, 60.0, 40.0], "cup", 0.8)], grasps=[g])
    out = np.asarray(draw(r, _grey(), max_grasps_per_object=None)).astype(int)
    assert out[60, 90, 1] > 200  # the closing line, in the quality colour (green-ish)


# ----------------------------------------------------------------------------- draw_prompts
def test_draw_prompts_dots_crosses_and_dashed_boxes():
    img = Image.new("RGB", (64, 48), (0, 0, 0))
    out = np.asarray(draw_prompts(img, boxes=[4, 4, 30, 20], points=[[40, 30], [10, 40, 0]]))
    assert out.shape == (48, 64, 3)
    assert out[30, 40, 1] > 150 and out[30, 40, 0] < 50   # green dot (label 1)
    assert out[40, 10, 0] > 150 and out[40, 10, 1] < 50   # red cross (label 0)
    top = out[4, 4:34]
    assert (top == 255).all(axis=1).any() and (top == 0).all(axis=1).any()  # white dashes, dark gaps
    assert np.asarray(img).max() == 0                     # the caller's image is not changed
    one = np.asarray(draw_prompts(img, points=[40, 30]))  # a single point is fine too
    assert one[30, 40, 1] > 150
    lab = np.asarray(draw_prompts(img, points=np.array([[40, 30], [10, 40]]), labels=[0, 1]))
    assert lab[30, 40, 0] > 150 and lab[40, 10, 1] > 150
    on_white = np.asarray(draw_prompts(Image.new("RGB", (64, 48), (255, 255, 255)), boxes=[4, 4, 30, 20]))
    assert (on_white[4, 4:34] == 0).all(axis=1).any()     # the box shows on white too


def test_draw_prompts_refuses_bad_input():
    img = Image.new("RGB", (64, 48))
    with pytest.raises(ValueError):
        draw_prompts(img, points=[[1, 2], [3, 4]], labels=[1])
    with pytest.raises(ValueError):
        draw_prompts(img, boxes=[[1, 2, 3]])
    with pytest.raises(ValueError):
        draw_prompts(img, points=[[1, 2, 3, 4]])


def test_draw_prompts_is_exported():
    import visionserve

    assert visionserve.draw_prompts is draw_prompts
