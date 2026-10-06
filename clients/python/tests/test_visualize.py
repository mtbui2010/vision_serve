"""Tests for visionserve.visualize.draw: EXIF orientation, mask labels, mask_boxes, classification
labels."""

import io

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from visionserve.types import Classification, Detection, Mask, Result  # noqa: E402
from visionserve.visualize import draw  # noqa: E402

W, H = 64, 40


def _rle(mask):
    """Column-major RLE of a bool (H, W) mask, starting with a background run."""
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


def _mask(x, y, w, h, conf=0.9):
    m = np.zeros((H, W), bool)
    m[y:y + h, x:x + w] = True
    return Mask(rle=_rle(m), bbox=[float(x), float(y), float(w), float(h)], conf=conf)


def _grey():
    return Image.new("RGB", (W, H), (128, 128, 128))


def test_exif_rotated_jpeg_is_drawn_upright(tmp_path):
    # Pixels stored sideways (H x W swapped) plus EXIF "rotate 90": the server decodes it upright,
    # so the result's boxes are in the upright W x H frame and draw() must use that frame too.
    upright = Image.new("RGB", (W, H), (0, 0, 0))
    upright.paste((255, 255, 255), (0, 0, W // 2, H))  # left half white
    ex = Image.Exif()
    ex[0x0112] = 6
    buf = io.BytesIO()
    upright.transpose(Image.Transpose.ROTATE_90).save(buf, "JPEG", exif=ex, quality=95)
    p = tmp_path / "phone.jpg"
    p.write_bytes(buf.getvalue())
    r = Result(task="detection", model="m")
    for image in (p, str(p), buf.getvalue()):
        out = draw(r, image)
        assert out.size == (W, H)
        a = np.asarray(out).astype(int)
        assert a[H // 2, 5].mean() > 200 and a[H // 2, W - 5].mean() < 50  # left white, right black


def test_png_and_pil_inputs_are_not_rotated(tmp_path):
    p = tmp_path / "x.png"
    _grey().save(p)
    r = Result(task="detection", model="m")
    assert draw(r, p).size == (W, H)
    # A PIL image is drawn as given (the SDK uploads it without the tag), even with a tag set.
    ex = Image.Exif()
    ex[0x0112] = 6
    buf = io.BytesIO()
    _grey().save(buf, "JPEG", exif=ex)
    assert draw(r, Image.open(io.BytesIO(buf.getvalue()))).size == (W, H)


def test_mask_with_detection_box_is_labelled_once_by_its_detection():
    det = Detection(bbox=[10.0, 10.0, 20.0, 15.0], cls="dog", conf=0.8)
    r = Result(task="open_vocab", model="m", detections=[det], masks=[_mask(10, 10, 20, 15)])
    # The mask shares the detection's box: its own box + "mask 90%" label are not drawn, so
    # mask_boxes makes no difference.
    a = np.asarray(draw(r, _grey()))
    b = np.asarray(draw(r, _grey(), mask_boxes=False))
    assert np.array_equal(a, b)
    # The mask fill is still there.
    assert not np.array_equal(b, np.asarray(draw(Result(task="open_vocab", model="m", detections=[det]), _grey())))


def test_mask_boxes_false_hides_boxes_of_plain_masks_but_keeps_fills():
    r = Result(task="segmentation", model="m", masks=[_mask(10, 10, 20, 15)])
    with_boxes = np.asarray(draw(r, _grey())).astype(int)
    fill_only = np.asarray(draw(r, _grey(), mask_boxes=False)).astype(int)
    assert not np.array_equal(with_boxes, fill_only)
    assert (fill_only[20, 20] != 128).any()          # inside the mask: tinted
    assert (fill_only[5, 5] == 128).all()            # outside: untouched (no box, no label)


def test_target_mask_keeps_its_red_box_with_mask_boxes_false():
    m = _mask(10, 10, 20, 15)
    r = Result(task="segmentation", model="m", masks=[m])
    a = np.asarray(draw(r, _grey(), mask_boxes=False, target_box=m)).astype(int)
    assert tuple(a[10 + 7, 10 - 1]) == (255, 0, 0)  # the left edge of the thick red outline


def test_classification_labels_have_a_dark_band():
    r = Result(task="classification", model="m",
               classifications=[Classification(cls="tusker", conf=0.46)])
    img = Image.new("RGB", (200, 80), (240, 240, 240))  # a light photo
    a = np.asarray(draw(r, img)).astype(int)
    assert a[30, 17].max() < 60  # the band just left of the text start is dark
