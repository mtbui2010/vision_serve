import cv2, numpy as np


def get_valid_depth_locs(depth, mask=None, box=None, bound_pixels=False):
    """
    IQR-filtered pixel locations of valid depth within a region.

    mask or box defines the region; bound_pixels=True selects the
    boundary ring of the mask instead of its eroded interior.
    """
    if bound_pixels and mask is None:
        raise ValueError("bound_pixels=True requires a mask")
    if mask is not None:
        k_big   = np.ones((21, 21), "uint8")
        k_small = np.ones((5, 5), "uint8")
        region = (
            cv2.dilate(mask, k_big) - cv2.dilate(mask, k_small)
            if bound_pixels
            else cv2.erode(mask, k_small)
        )
        valid_depth = region.astype("float32") * depth
    elif box is not None:
        x0, y0, x1, y1 = box
        valid_depth = np.zeros_like(depth)
        valid_depth[y0:y1, x0:x1] = depth[y0:y1, x0:x1]
    else:
        valid_depth = depth

    values = valid_depth[valid_depth > 0]
    if len(values) == 0:
        return [(), ()]
    q1, q3 = np.percentile(values, 25), np.percentile(values, 50)
    iqr = q3 - q1
    return np.where(
        (valid_depth >= q1 - 1.5 * iqr) & (valid_depth <= q3 + 1.5 * iqr)
    )


def Ixy2xyz(Ix, Iy, Z, cam_params):
    """Pixel coordinates + depth → 3D camera-frame coordinates."""
    fx, fy, cx, cy = cam_params[:4]
    scalar = not isinstance(Ix, (np.ndarray, list, tuple))
    if scalar:
        return (Ix - cx) * Z / fx, (Iy - cy) * Z / fy, Z
    Ix = np.asarray(Ix, dtype="float64") - cx
    Iy = np.asarray(Iy, dtype="float64") - cy
    Z  = np.asarray(Z,  dtype="float64")
    return Ix * Z / fx, Iy * Z / fy, Z


def xyz2Ixy(x, y, z, cam_params, eps=1e-10):
    """3D camera-frame coordinates → pixel coordinates."""
    fx, fy, cx, cy = cam_params[:4]
    Ix = np.divide(x, z + eps) * fx + cx
    Iy = np.divide(y, z + eps) * fy + cy
    try:
        return Ix.astype("int"), Iy.astype("int")
    except AttributeError:
        return int(Ix), int(Iy)

def show_mask_on_rgb(rgb, mask):
    rgb, mask = _as_numpy(rgb), _as_numpy(mask)
    locs = np.where(mask > 0)
    heatmap = 255 - cv2.applyColorMap(
        (255 * np.repeat(mask[..., np.newaxis], 3, axis=-1)).astype("uint8"),
        cv2.COLORMAP_JET,
    )
    out = rgb.copy()
    out[locs] = 0.6 * out[locs] + 0.4 * heatmap[locs]
    return out


def show_masks_on_rgb(rgb, masks, colors=None):
    if not masks:
        return rgb
    rgb = _as_numpy(rgb)
    colors = n2colormap(len(masks)) if colors is None else colors
    out = rgb.copy()
    for m, color in zip(masks, colors):
        m = _as_numpy(m)
        loc = np.where(m > 0)
        out[loc] = (0.6 * out[loc] + 0.4 * np.asarray(color)).astype(out.dtype)
    return out


def show_box_on_rgb(rgb, box, color=(0, 255, 0), thick=1, label=None):
    x0, y0, x1, y1 = (int(v) for v in box)
    out = cv2.rectangle(rgb.copy(), (x0, y0), (x1, y1), color, thick)
    # out = cv2.drawMarker(out, ((x0 + x1) // 2, (y0 + y1) // 2),
    #                      color, cv2.MARKER_TILTED_CROSS, 10, 2)
    if label is not None:
        out = cv2.putText(out, label, (x0, y0), cv2.FONT_HERSHEY_COMPLEX, 1., (255,0,0),2)
    return out


def show_boxes_on_rgb(rgb, boxes, color=(0, 255, 0), thick=1):
    out = rgb.copy()
    for box in boxes:
        out = show_box_on_rgb(out, box, color=color, thick=thick)
    return out


def show_line_on_rgb(rgb, line, color=(0, 255, 0), thick=1):
    x0, y0, x1, y1 = [int(el) for el in line]
    return cv2.line(rgb.copy(), (x0, y0), (x1, y1), color, thick)


def show_text_on_rgb(rgb, text, org, size=0.4, color=(0, 255, 0), thick=1):
    return cv2.putText(rgb.copy(), text, (int(org[0]), int(org[1])),
                       cv2.FONT_HERSHEY_COMPLEX, size, color, thick)


def show_texts_on_rgb(rgb, texts, orgs, size=0.4, color=(0, 255, 0), thick=1):
    out = rgb.copy()
    for text, org in zip(texts, orgs):
        out = show_text_on_rgb(out, text, org, size, color, thick)
    return out


# ── Mask / geometry helpers (copied from pyinterfaces.utils) ──────────────────
def get_mask_locs_with_stride(mask, stride=5):
    """Subsample nonzero mask locations on a regular grid."""
    H, W = mask.shape
    ys, xs = np.arange(0, H, stride), np.arange(0, W, stride)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    valid = mask[yy, xx] > 0
    return yy[valid], xx[valid]


def calc_normalvector(points, weights=None):
    """Weighted least-squares plane fit → unit normal vector (shape 3,)."""
    if weights is None:
        weights = np.ones(len(points), dtype="float32")
    w = weights / (np.sum(weights) + 1e-10)
    centroid = np.sum(points * w[:, None], axis=0)
    centered = points - centroid
    cov = (centered * w[:, None]).T @ centered
    _, _, vh = np.linalg.svd(cov)
    return -vh[-1]