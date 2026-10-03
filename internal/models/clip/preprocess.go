package clip

import (
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	prep "visionserve/internal/vision/preprocess"
)

// defaultMean and defaultStd are CLIP's normalization constants.
// They differ from standard ImageNet! (CLIP was trained with slightly different values.)
var (
	defaultMean = []float32{0.48145466, 0.4578275, 0.40821073}
	defaultStd  = []float32{0.26862954, 0.26130258, 0.27577711}
)

// arch: center_crop is how CLIP was trained and how its reference processor feeds it — resize
// the short side (bicubic) and keep the centre; the shipped manifest declares it. Squashing the
// whole frame (the default, for manifests that declare no crop) measured cosine 0.86-0.89
// against the reference embedding on non-square photos.
var arch = prep.Arch{Name: "clip", Modes: []prep.Mode{prep.Squash, prep.CenterCrop}}

// spec resolves the manifest's preprocessing, with CLIP's defaults: 224×224 when no size is
// given (a Config built in code), and CLIP's mean/std for whichever of the two is not declared.
func spec(cfg models.Config) (prep.Spec, error) {
	s := cfg.PreprocessSpec()
	if s.Width <= 0 {
		s.Width = 224
	}
	if s.Height <= 0 {
		s.Height = 224
	}
	if len(s.Mean) == 0 {
		s.Mean = defaultMean
	}
	if len(s.Std) == 0 {
		s.Std = defaultStd
	}
	return arch.Resolve(s)
}

// preprocess turns the image into an NCHW float32 tensor [1, 3, H, W] with CLIP normalization.
//
// PreprocessMeta is populated with scale factors so that downstream callers follow the
// same interface; for embedding tasks the meta is not used in postprocess (no bbox mapping).
func preprocess(img image.Image, cfg models.Config) (engine.Tensor, models.PreprocessMeta, error) {
	s, err := spec(cfg)
	if err != nil {
		return engine.Tensor{}, models.PreprocessMeta{}, err
	}
	return s.Apply(img)
}
