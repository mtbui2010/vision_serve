package imageproc

import "visionserve/internal/vision/preprocess"

// DPTKeepAspectSize returns the size HuggingFace's DPTImageProcessor resizes a w×h image to when
// keep_aspect_ratio is set, for a target tw×th and ensure_multiple_of = multiple. It is
// preprocess.DPTKeepAspectSize (the keep_aspect resize mode); see there for the exact rule.
func DPTKeepAspectSize(w, h, tw, th, multiple int) (nw, nh int) {
	return preprocess.DPTKeepAspectSize(w, h, tw, th, multiple)
}
