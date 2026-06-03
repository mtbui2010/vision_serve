package efficientsam

import (
	"image"
	"math"

	"github.com/disintegration/imaging"

	"visionserve/internal/engine"
)

// ImageNet mean and std (per-channel, RGB order).
// EfficientSAM's ViT encoder expects ImageNet-normalized NCHW input.
//
// TODO: confirm normalization expectations against the actual ONNX export.
// If the graph bakes normalization in (like MobileSAM's encoder) pass raw 0..255 instead.
var (
	imagenetMean = [3]float32{0.485, 0.456, 0.406}
	imagenetStd  = [3]float32{0.229, 0.224, 0.225}
)

// encoderInput builds the EfficientSAM encoder input tensor and returns the coordinate
// scale factor.
//
// EfficientSAM's ViT encoder expects NCHW float32 with ImageNet normalization:
//
//	pixel_normalized = (pixel/255 - mean) / std
//
// The long side is resized to 1024; aspect ratio is preserved. No padding is applied —
// the encoder processes a rectangular image of arbitrary width/height up to 1024.
// (MobileSAM pads to exactly 1024×1024 inside the graph; EfficientSAM does not.)
//
// scale = 1024 / max(origW, origH) maps original pixel coordinates to the resized space
// the decoder's batched_point_coords must use.
//
// Returns: tensor [1, 3, newH, newW], scale, error.
func encoderInput(img image.Image) (engine.Tensor, float64, error) {
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()

	scale := float64(encoderSize) / float64(max(origW, origH))
	newW := int(math.Round(float64(origW) * scale))
	newH := int(math.Round(float64(origH) * scale))
	if newW < 1 {
		newW = 1
	}
	if newH < 1 {
		newH = 1
	}

	resized := imaging.Resize(img, newW, newH, imaging.Linear) // *image.NRGBA

	// Build NCHW float32: [1, 3, newH, newW], ImageNet-normalized.
	data := make([]float32, 3*newH*newW)
	planeSize := newH * newW
	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			c := resized.NRGBAAt(resized.Bounds().Min.X+x, resized.Bounds().Min.Y+y)
			idx := y*newW + x
			data[0*planeSize+idx] = (float32(c.R)/255.0 - imagenetMean[0]) / imagenetStd[0] // R
			data[1*planeSize+idx] = (float32(c.G)/255.0 - imagenetMean[1]) / imagenetStd[1] // G
			data[2*planeSize+idx] = (float32(c.B)/255.0 - imagenetMean[2]) / imagenetStd[2] // B
		}
	}

	// Shape [1, 3, newH, newW] (NCHW, batch=1).
	t := engine.F32(data, 1, 3, int64(newH), int64(newW))
	return t, scale, nil
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
