package scrfd

import (
	"fmt"
	"image"
	"image/color"

	"github.com/disintegration/imaging"

	"visionserve/internal/engine"
	"visionserve/internal/imageproc"
	"visionserve/internal/models"
)

// preprocess mirrors InsightFace scrfd.py detect(): resize preserving aspect ratio so the
// image fits cfg.Width × cfg.Height (new size truncated with int(), as upstream), paste it
// at the TOP-LEFT corner of a black canvas (InsightFace does det_img[:nh, :nw] = resized —
// NOT a centred letterbox), then normalise with SCRFD's non-ImageNet formula:
//
//	pixel_normalized = (pixel - 127.5) / 128.0   (RGB order: blobFromImage swapRB=True)
//
// The Mean/Std in the manifest are [127.5, 127.5, 127.5] / [128.0, 128.0, 128.0] in [0,255]
// scale, while imageproc.ImageToCHWFloat computes v' = (v/255 - mean[c]) / std[c], so we
// pass mean/255 and std/255: (v/255 - mean/255) / (std/255) = (v - mean) / std.
//
// Placement matters: the anchors are a fixed grid over the input, so a centred pad shifts
// every face relative to the grid and changed the boxes (IoU vs the reference 0.90–0.95
// with a centred pad; see postprocess_test.go).
func preprocess(img image.Image, cfg models.Config) (engine.Tensor, models.PreprocessMeta, error) {
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()
	if origW <= 0 || origH <= 0 {
		return engine.Tensor{}, models.PreprocessMeta{}, fmt.Errorf("scrfd: empty image %dx%d", origW, origH)
	}

	newW, newH, scale := fitTopLeft(origW, origH, cfg.Width, cfg.Height)
	resized := imaging.Resize(img, newW, newH, imaging.Linear)
	canvas := imaging.New(cfg.Width, cfg.Height, color.NRGBA{0, 0, 0, 255})
	canvas = imaging.Paste(canvas, resized, image.Pt(0, 0))

	// Convert manifest Mean/Std ([0,255] scale) to [0,1] scale for ImageToCHWFloat.
	mean01 := normalizeTo01(cfg.Mean)
	std01 := normalizeTo01(cfg.Std)

	tensor := imageproc.ImageToCHWFloat(canvas, mean01, std01)

	meta := models.PreprocessMeta{
		OrigWidth:  origW,
		OrigHeight: origH,
		ScaleX:     scale,
		ScaleY:     scale, // one det_scale for both axes, as upstream
		PadX:       0,
		PadY:       0,
	}
	return tensor, meta, nil
}

// fitTopLeft reproduces InsightFace's resize arithmetic:
//
//	if h/w > H/W: new_h = H; new_w = int(new_h / (h/w))
//	else:         new_w = W; new_h = int(new_w * (h/w))
//	det_scale = new_h / h
func fitTopLeft(w, h, W, H int) (newW, newH int, scale float64) {
	imRatio := float64(h) / float64(w)
	modelRatio := float64(H) / float64(W)
	if imRatio > modelRatio {
		newH = H
		newW = int(float64(newH) / imRatio)
	} else {
		newW = W
		newH = int(float64(newW) * imRatio)
	}
	if newW < 1 {
		newW = 1
	}
	if newH < 1 {
		newH = 1
	}
	return newW, newH, float64(newH) / float64(h)
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
