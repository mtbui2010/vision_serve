package scrfd

import (
	"image"
	"image/color"

	"visionserve/internal/engine"
	"visionserve/internal/imageproc"
	"visionserve/internal/models"
)

// preprocess resizes the image to cfg.Width × cfg.Height using letterbox (preserve
// aspect ratio, pad with black), then normalizes with SCRFD's non-ImageNet formula:
//
//	pixel_normalized = (pixel/255 - 127.5/255) / (128.0/255)
//	                 = (pixel - 127.5) / 128.0
//
// The Mean/Std in the manifest are [127.5, 127.5, 127.5] / [128.0, 128.0, 128.0].
// imageproc.ImageToCHWFloat applies: v' = (v/255 - mean[c]) / std[c], so we pass
// mean/255 and std/255 — but the manifest already stores the values in [0,255] scale,
// while ImageToCHWFloat expects [0,1] scale for mean and std is applied to the [0,1]
// pixel.  We therefore convert: mean_01 = mean[c]/255, std_01 = std[c]/255.
//
// After conversion: v' = (v/255 - mean[c]/255) / (std[c]/255)
//
//	= (v - mean[c]) / std[c]   ✓  (matches SCRFD paper normalization)
func preprocess(img image.Image, cfg models.Config) (engine.Tensor, models.PreprocessMeta, error) {
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()

	lb := imageproc.Letterbox(img, cfg.Width, cfg.Height, color.NRGBA{0, 0, 0, 255})

	// Convert manifest Mean/Std ([0,255] scale) to [0,1] scale for ImageToCHWFloat.
	mean01 := normalizeTo01(cfg.Mean)
	std01 := normalizeTo01(cfg.Std)

	tensor := imageproc.ImageToCHWFloat(lb.Img, mean01, std01)

	meta := models.PreprocessMeta{
		OrigWidth:  origW,
		OrigHeight: origH,
		ScaleX:     lb.Scale,
		ScaleY:     lb.Scale, // letterbox: same scale on both axes
		PadX:       lb.PadX,
		PadY:       lb.PadY,
	}
	return tensor, meta, nil
}

// normalizeTo01 divides each value by 255 so that ImageToCHWFloat (which works in [0,1])
// produces the same result as the SCRFD formula: (pixel - mean) / std.
func normalizeTo01(vals []float32) []float32 {
	if len(vals) == 0 {
		return vals
	}
	out := make([]float32, len(vals))
	for i, v := range vals {
		out[i] = v / 255.0
	}
	return out
}
