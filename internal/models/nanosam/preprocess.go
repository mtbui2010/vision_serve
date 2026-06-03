package nanosam

import (
	"image"
	"math"

	"github.com/disintegration/imaging"

	"visionserve/internal/engine"
)

// ImageNet normalization constants.
// NanoSAM's ResNet-18 encoder expects pre-normalized input — unlike MobileSAM's
// TinyViT which bakes normalization into the ONNX graph.
var (
	imagenetMean = [3]float32{0.485, 0.456, 0.406}
	imagenetStd  = [3]float32{0.229, 0.224, 0.225}
)

// encoderInput resizes img to 1024×1024 (preserving aspect ratio, zero-padding
// the shorter side on the bottom/right), then produces an NCHW float32 tensor
// [1, 3, 1024, 1024] with ImageNet normalization applied in Go.
//
// Why different from MobileSAM:
//   - MobileSAM encoder bakes SAM pixel normalize + pad into its ONNX graph and
//     accepts raw HWC float32 in [0,255]. Go only resizes; the graph does the rest.
//   - NanoSAM encoder (ResNet-18) expects the standard ImageNet-normalized NCHW
//     input that torchvision transforms produce. Go must:
//     1. Resize long side to 1024 (aspect-ratio preserving),
//     2. Zero-pad to square 1024×1024 (bottom/right),
//     3. Normalize: pixel = (raw/255 - mean) / std per channel.
//
// Returns:
//   - t: NCHW tensor [1, 3, 1024, 1024].
//   - scale: factor such that resized_coord = original_coord * scale. Used by the
//     caller to map prompt coordinates into the 1024 space the decoder expects.
func encoderInput(img image.Image) (engine.Tensor, float32) {
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()

	const encoderSize = 1024
	scale := float32(encoderSize) / float32(maxInt(origW, origH))
	newW := int(math.Round(float64(origW) * float64(scale)))
	newH := int(math.Round(float64(origH) * float64(scale)))
	if newW < 1 {
		newW = 1
	}
	if newH < 1 {
		newH = 1
	}

	resized := imaging.Resize(img, newW, newH, imaging.Linear) // *image.NRGBA

	// NCHW layout with zero-padding to [1024, 1024].
	// Channel plane size is always 1024*1024 regardless of content region.
	data := make([]float32, 3*encoderSize*encoderSize) // zero-initialized = zero-pad

	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			c := resized.NRGBAAt(resized.Bounds().Min.X+x, resized.Bounds().Min.Y+y)
			r := (float32(c.R)/255.0 - imagenetMean[0]) / imagenetStd[0]
			g := (float32(c.G)/255.0 - imagenetMean[1]) / imagenetStd[1]
			bv := (float32(c.B)/255.0 - imagenetMean[2]) / imagenetStd[2]

			// NCHW: channel 0 = all R pixels, channel 1 = all G, channel 2 = all B.
			// pixel (y,x) in plane c is at offset: c*H*W + y*W + x
			data[0*encoderSize*encoderSize+y*encoderSize+x] = r
			data[1*encoderSize*encoderSize+y*encoderSize+x] = g
			data[2*encoderSize*encoderSize+y*encoderSize+x] = bv
		}
	}

	return engine.F32(data, 1, 3, encoderSize, encoderSize), scale
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
