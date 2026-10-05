package preprocess

import "math"

// The size arithmetic of each mode, exactly as the upstream recipe (and the hand-written code
// this package replaced) computes it. Every function takes the SOURCE size first (w, h) and the
// target second (W, H).

// LetterboxSize: scale = min(W/w, H/h), new sides rounded half up (int(x+0.5), at least 1),
// centred: pad = (W-nw)/2, (H-nh)/2 floored.
func LetterboxSize(w, h, W, H int) (nw, nh int, scale float64, padX, padY int) {
	scale = float64(W) / float64(w)
	if s := float64(H) / float64(h); s < scale {
		scale = s
	}
	nw = int(float64(w)*scale + 0.5)
	nh = int(float64(h)*scale + 0.5)
	if nw < 1 {
		nw = 1
	}
	if nh < 1 {
		nh = 1
	}
	return nw, nh, scale, (W - nw) / 2, (H - nh) / 2
}

// CoverSize is HuggingFace CLIPImageProcessor's get_resize_output_image_size + center_crop:
//
//	short side -> its target, long side -> int(target * long / short)   (truncated)
//	offset     -> (resized - crop) // 2                                  (floored)
//
// It returns the resized size and the crop offset in RESIZED pixels (input = orig*scale - off).
func CoverSize(w, h, W, H int) (rw, rh, offX, offY int) {
	if w <= h { // width is the short side
		rw = W
		rh = int(float64(W) * float64(h) / float64(w))
	} else {
		rh = H
		rw = int(float64(H) * float64(w) / float64(h))
	}
	if rw < W {
		rw = W
	}
	if rh < H {
		rh = H
	}
	return rw, rh, (rw - W) / 2, (rh - H) / 2
}

// CenterCropSize is the center_crop geometry with timm's crop_pct: the image is resized to cover
// floor(W/pct)×floor(H/pct) (CoverSize: short side to its target, long side truncated) and the
// centred W×H window is kept, offset floored. pct = 0.875 at 224 is timm's default eval transform
// and torchvision's Resize(256) + CenterCrop(224); pct <= 0 or >= 1 is CoverSize itself (CLIP).
// It returns the resized size and the crop offset in RESIZED pixels (input = orig*scale - off).
func CenterCropSize(w, h, W, H int, pct float32) (rw, rh, offX, offY int) {
	sw, sh := CropScaleSize(W, H, pct)
	rw, rh, _, _ = CoverSize(w, h, sw, sh)
	return rw, rh, (rw - W) / 2, (rh - H) / 2
}

// CropScaleSize is the size center_crop covers before cutting W×H out: timm's
// math.floor(size / crop_pct) per side (W×H itself for pct <= 0 or >= 1). The 1e-4 keeps a ratio
// such as 224/232 written as a float (0.9655172) from flooring 232 to 231.
func CropScaleSize(W, H int, pct float32) (int, int) {
	if pct <= 0 || pct >= 1 {
		return W, H
	}
	return int(math.Floor(float64(W)/float64(pct) + 1e-4)), int(math.Floor(float64(H)/float64(pct) + 1e-4))
}

// DPTKeepAspectSize returns the size HuggingFace's DPTImageProcessor resizes a w×h image to when
// keep_aspect_ratio is set (get_resize_output_image_size in transformers' image_processing_dpt.py),
// for a target tw×th and ensure_multiple_of = multiple (<= 0 means 1):
//
//	scale_w = tw / w, scale_h = th / h
//	"scale as little as possible": if |1-scale_w| < |1-scale_h| both axes use scale_w,
//	                               otherwise (ties included) both use scale_h
//	side = round(scale*side / multiple) * multiple   (Python round: half to EVEN)
//
// The result is NOT bounded by tw×th: one side matches its target, the other follows the aspect
// ratio and may be larger or smaller. Depth Anything V2 (518, multiple 14) maps 848×480 -> 910×518.
//
// One deliberate difference: HF returns 0 for a side that rounds to nothing (a 3000×20 image gives
// height 0, which then fails inside the resize). Here every side is at least `multiple`, so an
// extreme panorama yields a thin valid input instead of an error.
func DPTKeepAspectSize(w, h, tw, th, multiple int) (nw, nh int) {
	if multiple <= 0 {
		multiple = 1
	}
	if w <= 0 || h <= 0 {
		return tw, th
	}
	scaleW := float64(tw) / float64(w)
	scaleH := float64(th) / float64(h)
	if math.Abs(1-scaleW) < math.Abs(1-scaleH) {
		scaleH = scaleW
	} else {
		scaleW = scaleH
	}
	return constrainToMultiple(scaleW*float64(w), multiple), constrainToMultiple(scaleH*float64(h), multiple)
}

// constrainToMultiple is DPT's constrain_to_multiple_of with no max_val and min_val = 0, except
// that the result is never below m (see DPTKeepAspectSize).
func constrainToMultiple(v float64, m int) int {
	x := int(math.RoundToEven(v/float64(m))) * m
	if x < m {
		x = m
	}
	return x
}

// TopLeftSize reproduces InsightFace scrfd.py detect()'s resize arithmetic (each side at least 1):
//
//	if h/w > H/W: new_h = H; new_w = int(new_h / (h/w))
//	else:         new_w = W; new_h = int(new_w * (h/w))
//
// It deliberately does NOT return upstream's det_scale = new_h / h. Upstream maps boxes back by
// dividing BOTH axes by it, but new_w and new_h are truncated separately, so the content's x
// scale is new_w / w, not new_h / h. On ordinary images x is off by less than W/new_h input
// pixels at the far edge of a landscape image (under 1.8 for 16:9 into 640×640) and less than
// one in a portrait one; on extreme aspect ratios it is wildly off (10000×10 into 640×640:
// new_h = int(0.64) -> 1, det_scale = 0.1, while x was scaled by 640/10000). Apply records
// each axis's own scale.
func TopLeftSize(w, h, W, H int) (nw, nh int) {
	imRatio := float64(h) / float64(w)
	modelRatio := float64(H) / float64(W)
	if imRatio > modelRatio {
		nh = H
		nw = int(float64(nh) / imRatio)
	} else {
		nw = W
		nh = int(float64(nw) * imRatio)
	}
	if nw < 1 {
		nw = 1
	}
	if nh < 1 {
		nh = 1
	}
	return nw, nh
}

// LongSideSize: scale = min(W/w, H/h) — for a square target, the long side goes to its target
// (SAM's ResizeLongestSide) — capped at 1 when noUpscale; sides rounded half away from zero
// (math.Round), at least 1.
func LongSideSize(w, h, W, H int, noUpscale bool) (nw, nh int, scale float64) {
	scale = math.Min(float64(W)/float64(w), float64(H)/float64(h))
	if noUpscale && scale > 1.0 {
		scale = 1.0
	}
	nw = int(math.Round(float64(w) * scale))
	nh = int(math.Round(float64(h) * scale))
	if nw < 1 {
		nw = 1
	}
	if nh < 1 {
		nh = 1
	}
	return nw, nh, scale
}

// ceilMultiple rounds v up to a multiple of m (m > 0).
func ceilMultiple(v, m int) int { return ((v + m - 1) / m) * m }
