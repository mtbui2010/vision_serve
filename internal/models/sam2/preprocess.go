package sam2

import (
	"image"
	"math"

	"github.com/disintegration/imaging"

	"visionserve/internal/engine"
)

// SAM2 pixel normalization constants (ImageNet mean/std used by SAM2).
// These are applied in Go because the SAM2 ONNX encoder expects a pre-normalized
// NCHW tensor (unlike MobileSAM which bakes normalization into the graph and
// takes raw 0..255 HWC input).
//
// TODO: verify these constants match your ONNX export. Some community exports bake
// normalization into the graph (like MobileSAM) and expect raw 0..255 input instead.
// If you see wildly wrong masks, try commenting out normalization and passing 0..255.
var (
	sam2Mean = [3]float32{0.485, 0.456, 0.406} // R, G, B
	sam2Std  = [3]float32{0.229, 0.224, 0.225}
)

// encoderInput builds the SAM2 encoder input tensor and returns coordinate metadata.
//
// SAM2 encoder expects NCHW float32 [1, 3, 1024, 1024] with SAM2 normalization
// (ImageNet mean/std applied, values in roughly [-2.1..2.6] range).
//
// The image is resized so the long side equals 1024 (aspect ratio preserved),
// then zero-padded to exactly 1024×1024 (bottom/right padding only).
//
// Returns:
//   - tensor: [1, 3, 1024, 1024] float32 NCHW
//   - scale:  factor to map original coords → 1024-space (same on both axes since
//     aspect ratio is preserved; padding is at the edges)
//   - padX, padY: right/bottom padding in pixels (for coordinate mapping if needed)
//   - err: non-nil on failure
func encoderInput(img image.Image) (tensor engine.Tensor, scale float32, padX, padY int, err error) {
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()

	sc := float64(encoderSize) / float64(maxInt(origW, origH))
	newW := int(math.Round(float64(origW) * sc))
	newH := int(math.Round(float64(origH) * sc))
	if newW < 1 {
		newW = 1
	}
	if newH < 1 {
		newH = 1
	}

	resized := imaging.Resize(img, newW, newH, imaging.Linear) // *image.NRGBA

	padX = encoderSize - newW
	padY = encoderSize - newH

	// NCHW layout: [1, 3, 1024, 1024] — channels first, padded canvas.
	// Pixel values: normalize per channel with ImageNet mean/std.
	// Padded region stays at 0.0 (zero-mean after normalization would be ≈-mean/std,
	// but we leave it at 0 — padding is masked by the aspect-ratio scale anyway).
	data := make([]float32, 3*encoderSize*encoderSize)
	minX := resized.Bounds().Min.X
	minY := resized.Bounds().Min.Y
	for c := 0; c < 3; c++ {
		chOff := c * encoderSize * encoderSize
		for y := 0; y < newH; y++ {
			for x := 0; x < newW; x++ {
				px := resized.NRGBAAt(minX+x, minY+y)
				var raw float32
				switch c {
				case 0:
					raw = float32(px.R) / 255.0
				case 1:
					raw = float32(px.G) / 255.0
				case 2:
					raw = float32(px.B) / 255.0
				}
				data[chOff+y*encoderSize+x] = (raw - sam2Mean[c]) / sam2Std[c]
			}
		}
		// Padded rows/cols remain 0.0 (already zero-initialised).
	}

	tensor = engine.F32(data, 1, 3, int64(encoderSize), int64(encoderSize))
	scale = float32(sc)
	return tensor, scale, padX, padY, nil
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
