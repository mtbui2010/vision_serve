package mobilesam

import (
	"image"
	"math"

	"github.com/disintegration/imaging"

	"visionserve/internal/engine"
)

// encoderInput builds the SAM encoder input and returns the coordinate scale.
//
// The exported encoder bakes normalization (SAM pixel mean/std) AND padding-to-1024
// INTO the graph and takes a raw HWC uint8-range image. So Go only:
//  1. resizes the LONG side to 1024 (keeping aspect ratio),
//  2. feeds an HWC float32 tensor [newH, newW, 3] with values still in 0..255.
//
// scale = 1024 / max(origW, origH) maps original-image coordinates into the resized
// space the decoder's point prompts must use (padding is bottom/right only, so no pad
// offset is needed on coordinates).
func encoderInput(img image.Image) (engine.Tensor, float64) {
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

	// HWC float32, channel order R,G,B, values 0..255 (no normalization in Go).
	data := make([]float32, newH*newW*3)
	i := 0
	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			c := resized.NRGBAAt(resized.Bounds().Min.X+x, resized.Bounds().Min.Y+y)
			data[i] = float32(c.R)
			data[i+1] = float32(c.G)
			data[i+2] = float32(c.B)
			i += 3
		}
	}
	return engine.F32(data, int64(newH), int64(newW), 3), scale
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
