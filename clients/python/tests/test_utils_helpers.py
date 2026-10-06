"""visionserve.utils helpers with the inputs the SDK gives you (bool masks, PIL images, mask
stacks), and the documented default of select_target_object's distance_sigma."""

import pytest

np = pytest.importorskip("numpy")

from visionserve import utils as U  # noqa: E402
from visionserve.postprocess import select_target_object  # noqa: E402
from visionserve.types import Detection, Result  # noqa: E402


def _rgb():
    return np.full((20, 30, 3), 100, np.uint8)


def test_show_masks_on_rgb_accepts_a_mask_stack():
    masks = np.zeros((2, 20, 30), bool)
    masks[0, 2:5, 2:5] = True
    masks[1, 10:12, 10:12] = True
    out = U.show_masks_on_rgb(_rgb(), masks)          # an (N, H, W) array, not only a list
    assert (out[3, 3] != 100).any() and (out[11, 11] != 100).any() and (out[0, 0] == 100).all()
    assert U.show_masks_on_rgb(_rgb(), np.zeros((0, 20, 30), bool)).shape == (20, 30, 3)


def test_get_valid_depth_locs_accepts_a_bool_mask_and_a_float_box():
    pytest.importorskip("cv2")
    depth = np.full((20, 30), 0.5, np.float32)
    mask = np.zeros((20, 30), bool)                   # what Mask.to_ndarray returns
    mask[5:15, 5:20] = True
    ys, xs = U.get_valid_depth_locs(depth, mask=mask)
    ys8, xs8 = U.get_valid_depth_locs(depth, mask=mask.astype(np.uint8))
    assert len(ys) > 0 and np.array_equal(ys, ys8) and np.array_equal(xs, xs8)
    ys, xs = U.get_valid_depth_locs(depth, box=[2.0, 3.0, 6.7, 8.2])  # corners, floats allowed
    assert len(ys) == (8 - 3) * (6 - 2)


def test_opencv_drawing_helpers_accept_pil_images():
    pytest.importorskip("cv2")
    Image = pytest.importorskip("PIL.Image")
    pil = Image.fromarray(_rgb())
    out = U.show_box_on_rgb(pil, [2, 2, 10, 10], color=(255, 0, 0))
    assert isinstance(out, np.ndarray) and tuple(out[2, 5]) == (255, 0, 0)
    assert U.show_boxes_on_rgb(pil, [[2, 2, 10, 10]]).shape == (20, 30, 3)
    assert U.show_line_on_rgb(pil, [0, 0, 29, 19]).shape == (20, 30, 3)
    assert U.show_text_on_rgb(pil, "a", (2, 15)).shape == (20, 30, 3)
    assert U.show_texts_on_rgb(pil, ["a"], [(2, 15)]).shape == (20, 30, 3)
    assert np.asarray(pil)[2, 5, 0] == 100            # the input is not drawn on


def test_xyz2Ixy_rounds_and_inverts_Ixy2xyz():
    cam = [600, 600, 320, 213]
    X, Y, Z = U.Ixy2xyz(420, 313, 1.5, cam)
    got = U.xyz2Ixy(X, Y, Z, cam)
    assert got == (420, 313) and all(type(v) is int for v in got)  # was (419, 312): truncated
    xs, ys = U.xyz2Ixy(np.array([X, 0.0]), np.array([Y, 0.0]), np.array([Z, 1.0]), cam)
    assert xs.tolist() == [420, 320] and ys.tolist() == [313, 213]


def test_select_target_object_distance_sigma_default_is_half_the_target():
    # Two objects at 1.3 m (conf 0.9) and 1.0 m (conf 0.5), target 1.0 m, conf + distance
    # weighted equally. With the default sigma 0.5 * 1.0 the confident object still wins
    # (0.9 + 0.835) / 2 > (0.5 + 1) / 2; with sigma 0.15 the 1.0 m one would.
    K = [100.0, 100.0, 5.0, 5.0]
    depth = np.zeros((10, 20), np.float32)
    depth[:, :10] = 1.3
    depth[:, 10:] = 1.0
    res = Result(task="detection", model="m", detections=[
        Detection(bbox=[4.0, 4.0, 2.0, 2.0], cls="a", conf=0.9),
        Detection(bbox=[14.0, 4.0, 2.0, 2.0], cls="b", conf=0.5),
    ])
    kw = dict(depth_result=depth, intrinsics=K, target_distance=1.0, weights={"conf": 1, "distance": 1})
    assert select_target_object(res, **kw).cls == "a"
    assert select_target_object(res, distance_sigma=0.15, **kw).cls == "b"
