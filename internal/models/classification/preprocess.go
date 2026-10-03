package classification

import (
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	prep "visionserve/internal/vision/preprocess"
)

// arch: classification models (EfficientNet, MobileNet) squash the full frame to the target
// resolution (224×224) — the legacy input.letterbox/crop flags were never honoured here. They
// are trained with center-crop preprocessing in practice, but at inference with an
// already-cropped or full-frame image a squash is standard and sufficient; no letterbox padding
// (black borders hurt classification accuracy).
var arch = prep.Arch{Name: "classification", Modes: []prep.Mode{prep.Squash}}

// preprocess: original image -> NCHW [1,3,H,W] tensor, squash-resized + ImageNet-normalized.
// PreprocessMeta records per-axis scale (not used in postprocess for classification, but kept
// for interface consistency).
func preprocess(img image.Image, cfg models.Config) (engine.Tensor, models.PreprocessMeta, error) {
	s, err := arch.Resolve(cfg.PreprocessSpec())
	if err != nil {
		return engine.Tensor{}, models.PreprocessMeta{}, err
	}
	return s.Apply(img)
}
