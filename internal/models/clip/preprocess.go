package clip

import (
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/imageproc"
	"visionserve/internal/models"
)

// defaultMean and defaultStd are CLIP's normalization constants.
// They differ from standard ImageNet! (CLIP was trained with slightly different values.)
var (
	defaultMean = []float32{0.48145466, 0.4578275, 0.40821073}
	defaultStd  = []float32{0.26862954, 0.26130258, 0.27577711}
)

// preprocess resizes the image to cfg.Width × cfg.Height (squash, no letterbox —
// this is correct for CLIP which was trained on square-resized images), then converts
// to an NCHW float32 tensor [1, 3, H, W] with CLIP normalization.
//
// PreprocessMeta is populated with scale factors so that downstream callers follow the
// same interface; for embedding tasks the meta is not used in postprocess (no bbox mapping).
func preprocess(img image.Image, cfg models.Config) (engine.Tensor, models.PreprocessMeta, error) {
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()

	w := cfg.Width
	h := cfg.Height
	if w <= 0 {
		w = 224
	}
	if h <= 0 {
		h = 224
	}

	// Squash resize (no letterbox) — standard for CLIP.
	resized := imageproc.Resize(img, w, h)

	// Use manifest normalization if provided, otherwise fall back to CLIP defaults.
	mean := cfg.Mean
	std := cfg.Std
	if len(mean) == 0 {
		mean = defaultMean
	}
	if len(std) == 0 {
		std = defaultStd
	}

	scaleX, scaleY := imageproc.ResizeScale(origW, origH, w, h)
	meta := models.PreprocessMeta{
		OrigWidth:  origW,
		OrigHeight: origH,
		ScaleX:     scaleX,
		ScaleY:     scaleY,
		PadX:       0,
		PadY:       0,
	}

	return imageproc.ImageToCHWFloat(resized, mean, std), meta, nil
}
