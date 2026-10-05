package classification

import (
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	prep "visionserve/internal/vision/preprocess"
)

// arch: the geometries a classification manifest may declare (default squash):
//   - center_crop (with crop_pct): how ImageNet classifiers are evaluated — timm's eval transform
//     and torchvision's weights.transforms() resize the short side to size/crop_pct (256 for 224)
//     and keep the centred size×size window. The shipped efficientnet-b0 (timm, bicubic) and
//     mobilenet-v3 (torchvision, bilinear) declare it: served on 5000 ImageNet val images it
//     scored top-1 79.04 / 71.82 against 77.44 / 68.08 squashed (timm 78.84, torchvision 71.86
//     with their own transforms; BUGS_TO_FIX.md #1, audit). The legacy input.crop: center now
//     selects it too (it was ignored before).
//   - squash: the whole frame stretched to width×height (manifests that declare no crop).
//
// No letterbox: black borders are not what these models were trained on, and the legacy
// input.letterbox flag was never honoured here (it falls back to squash).
var arch = prep.Arch{Name: "classification", Modes: []prep.Mode{prep.Squash, prep.CenterCrop}}

// preprocess: original image -> NCHW [1,3,H,W] tensor, resized as the manifest declares (see arch)
// + ImageNet-normalized.
// PreprocessMeta records per-axis scale (not used in postprocess for classification, but kept
// for interface consistency).
func preprocess(img image.Image, cfg models.Config) (engine.Tensor, models.PreprocessMeta, error) {
	s, err := arch.Resolve(cfg.PreprocessSpec())
	if err != nil {
		return engine.Tensor{}, models.PreprocessMeta{}, err
	}
	return s.Apply(img)
}
