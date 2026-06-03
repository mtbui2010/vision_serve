package classification

import (
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/imageproc"
	"visionserve/internal/models"
)

// preprocess: original image -> NCHW [1,3,H,W] tensor, squash-resized + ImageNet-normalized.
//
// Classification models (EfficientNet, MobileNet) are trained with center-crop preprocessing
// in practice, but at inference with an already-cropped or full-frame image a simple squash
// resize to the target resolution (224×224) is standard and sufficient. No letterbox padding
// is used — padding introduces spurious black border content that hurts classification accuracy.
//
// PreprocessMeta records per-axis scale (not used in postprocess for classification, but kept
// for interface consistency).
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
