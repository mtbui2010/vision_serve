package imageproc

import "math"

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
