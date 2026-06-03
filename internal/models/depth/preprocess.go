package depth

import (
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/imageproc"
	"visionserve/internal/models"
)

// preprocess: original image -> NCHW [1,3,H,W] tensor, squash-resized + ImageNet-normalized.
//
// Depth models use squash resize (no letterbox) because the full field of view matters;
// slight aspect distortion is acceptable for relative depth estimation. Both Depth Anything V2
// and MiDaS were trained with squash-resized inputs.
//
// PreprocessMeta records per-axis scale so downstream callers can map pixel coordinates if needed.
func preprocess(img image.Image, cfg models.Config) (engine.Tensor, models.PreprocessMeta, error) {
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()

	resized := imageproc.Resize(img, cfg.Width, cfg.Height)
	scaleX, scaleY := imageproc.ResizeScale(origW, origH, cfg.Width, cfg.Height)

	meta := models.PreprocessMeta{
		OrigWidth:  origW,
		OrigHeight: origH,
		ScaleX:     scaleX,
		ScaleY:     scaleY,
		PadX:       0,
		PadY:       0,
	}
	return imageproc.ImageToCHWFloat(resized, cfg.Mean, cfg.Std), meta, nil
}
