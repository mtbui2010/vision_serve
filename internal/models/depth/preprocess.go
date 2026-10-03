package depth

import (
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/imageproc"
	"visionserve/internal/models"
)

// preprocess: original image -> NCHW [1,3,H,W] tensor + ImageNet normalization.
//
// Two geometries, chosen by the manifest:
//   - default (MiDaS, and any manifest without input.keep_aspect): squash to width×height,
//     bilinear. No letterbox — the full field of view matters for depth.
//   - input.keep_aspect (Depth Anything V2): HuggingFace DPTImageProcessor's rule — scale both
//     axes by whichever of width/W, height/H is closer to 1, round each side to
//     input.multiple_of (imageproc.DPTKeepAspectSize), bicubic, no crop, no pad. The tensor's
//     H×W then varies per image, so the ONNX graph must have dynamic height/width axes.
//
// PreprocessMeta records the per-axis scale (input = orig * scale; no padding).
func preprocess(img image.Image, cfg models.Config) (engine.Tensor, models.PreprocessMeta, error) {
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()

	w, h := cfg.Width, cfg.Height
	var resized *image.NRGBA
	if cfg.KeepAspect {
		w, h = imageproc.DPTKeepAspectSize(origW, origH, cfg.Width, cfg.Height, cfg.MultipleOf)
		resized = imageproc.ResizeBicubic(img, w, h)
	} else {
		resized = imageproc.Resize(img, w, h)
	}
	scaleX, scaleY := imageproc.ResizeScale(origW, origH, w, h)

	meta := models.PreprocessMeta{
		OrigWidth:  origW,
		OrigHeight: origH,
		ScaleX:     scaleX,
		ScaleY:     scaleY,
	}
	return imageproc.ImageToCHWFloat(resized, cfg.Mean, cfg.Std), meta, nil
}
