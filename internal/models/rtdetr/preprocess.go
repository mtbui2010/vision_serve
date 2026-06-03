package rtdetr

import (
	"image"
	"image/color"

	"visionserve/internal/engine"
	"visionserve/internal/imageproc"
	"visionserve/internal/models"
)

// preprocess: original image -> NCHW [1,3,H,W] tensor, letterboxed + normalized.
// PreprocessMeta stores scale/pad so postprocess can map boxes back to ORIGINAL image coords.
// Input dimensions are read from cfg (default 640×640 per manifest, not hardcoded).
func (m *rtDETR) preprocess(img image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()

	var lb imageproc.LetterboxResult
	var processed image.Image

	if m.cfg.Letterbox {
		lb = imageproc.Letterbox(img, m.cfg.Width, m.cfg.Height, color.NRGBA{0, 0, 0, 255})
		processed = lb.Img
	} else {
		processed = imageproc.Resize(img, m.cfg.Width, m.cfg.Height)
		sx, sy := imageproc.ResizeScale(origW, origH, m.cfg.Width, m.cfg.Height)
		lb = imageproc.LetterboxResult{Scale: sx, PadX: 0, PadY: 0}
		// without letterbox, scale can differ between the two axes
		meta := models.PreprocessMeta{
			OrigWidth: origW, OrigHeight: origH,
			ScaleX: sx, ScaleY: sy, PadX: 0, PadY: 0,
		}
		_ = lb // silence unused warning; lb is used in the letterbox branch above
		return imageproc.ImageToCHWFloat(processed, m.cfg.Mean, m.cfg.Std), meta, nil
	}

	meta := models.PreprocessMeta{
		OrigWidth:  origW,
		OrigHeight: origH,
		ScaleX:     lb.Scale,
		ScaleY:     lb.Scale, // letterbox preserves aspect ratio -> both axes share the same scale
		PadX:       lb.PadX,
		PadY:       lb.PadY,
	}
	return imageproc.ImageToCHWFloat(processed, m.cfg.Mean, m.cfg.Std), meta, nil
}
