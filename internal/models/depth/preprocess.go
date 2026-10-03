package depth

import (
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	prep "visionserve/internal/vision/preprocess"
)

// arch: the two geometries a depth manifest may declare (default squash):
//   - squash (MiDaS, and any manifest without keep_aspect): bilinear to width×height. No
//     letterbox — the full field of view matters for depth.
//   - keep_aspect (Depth Anything V2): HuggingFace DPTImageProcessor's rule — scale both axes by
//     whichever of width/W, height/H is closer to 1, round each side to multiple_of
//     (prep.DPTKeepAspectSize), bicubic, no crop, no pad. The tensor's H×W then varies per image,
//     so the ONNX graph must have dynamic height/width axes.
//
// The depth map is returned as the whole frame at the model's output resolution, so a mode that
// pads (letterbox borders in the map) or crops (lost margins) is refused — and ignored, as it
// always was, when only the legacy input.letterbox / input.crop flags ask for it.
var arch = prep.Arch{Name: "depth", Modes: []prep.Mode{prep.Squash, prep.KeepAspect}}

// preprocess: original image -> NCHW [1,3,H,W] tensor + ImageNet normalization.
// PreprocessMeta records the per-axis scale (input = orig * scale; no padding).
func preprocess(img image.Image, cfg models.Config) (engine.Tensor, models.PreprocessMeta, error) {
	s, err := arch.Resolve(cfg.PreprocessSpec())
	if err != nil {
		return engine.Tensor{}, models.PreprocessMeta{}, err
	}
	return s.Apply(img)
}
