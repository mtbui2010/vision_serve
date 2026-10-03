package detr

import (
	"image"
	"image/color"

	"visionserve/internal/engine"
	"visionserve/internal/imageproc"
	"visionserve/internal/models"
)

// preprocess: original image -> NCHW [1,3,H,W] tensor (cfg.Width×cfg.Height), resized +
// normalized. PreprocessMeta stores scale/pad so postprocess can map boxes back to
// ORIGINAL image coords.
func (m *detr) preprocess(img image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()

	if m.cfg.Letterbox {
		// NOT how RF-DETR is trained. rfdetr's recipe squashes (square_resize_div_64=True, and
		// its own predict() is F.resize(img, [res, res]) with no padding), so RF-DETR manifests
		// declare letterbox: false. Serving a squash-trained checkpoint letterboxed measured
		// -7.35 mAP on the fine-tuned 512 detectors and -1.91 on COCO base (BUGS_TO_FIX.md #1).
		// Kept for exports that really were trained aspect-preserving (the RT-DETR manifest
		// letterboxes); pad is black.
		lb := imageproc.Letterbox(img, m.cfg.Width, m.cfg.Height, color.NRGBA{0, 0, 0, 255})
		meta := models.PreprocessMeta{
			OrigWidth:  origW,
			OrigHeight: origH,
			ScaleX:     lb.Scale,
			ScaleY:     lb.Scale, // letterbox preserves aspect ratio -> both axes share the same scale
			PadX:       lb.PadX,
			PadY:       lb.PadY,
		}
		return imageproc.ImageToCHWFloat(lb.Img, m.cfg.Mean, m.cfg.Std), meta, nil
	}

	// Squash: the scale can differ between the two axes.
	processed := imageproc.Resize(img, m.cfg.Width, m.cfg.Height)
	sx, sy := imageproc.ResizeScale(origW, origH, m.cfg.Width, m.cfg.Height)
	meta := models.PreprocessMeta{
		OrigWidth: origW, OrigHeight: origH,
		ScaleX: sx, ScaleY: sy, PadX: 0, PadY: 0,
	}
	return imageproc.ImageToCHWFloat(processed, m.cfg.Mean, m.cfg.Std), meta, nil
}
