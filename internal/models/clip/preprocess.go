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

// preprocess turns the image into an NCHW float32 tensor [1, 3, H, W] with CLIP normalization.
// With input.crop: center (the shipped manifest) it resizes the short side and keeps the centre,
// as CLIP was trained; otherwise it squashes the whole frame to cfg.Width × cfg.Height.
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

	// input.crop: center is how CLIP was trained and how its reference processor feeds it:
	// resize the short side (bicubic) and keep the centre. Squashing the whole frame instead
	// measured cosine 0.86-0.89 against the reference embedding on non-square photos.
	if cfg.Crop == "center" {
		cropped, sx, sy, offX, offY := imageproc.ResizeShortCenterCrop(img, w, h)
		meta := models.PreprocessMeta{OrigWidth: origW, OrigHeight: origH,
			ScaleX: sx, ScaleY: sy, PadX: -offX, PadY: -offY}
		return imageproc.ImageToCHWFloat(cropped, pick(cfg.Mean, defaultMean), pick(cfg.Std, defaultStd)), meta, nil
	}

	// Squash resize (no letterbox): the default, kept for manifests that do not declare a crop.
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

func pick(v, def []float32) []float32 {
	if len(v) == 0 {
		return def
	}
	return v
}
