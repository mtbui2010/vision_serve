package groundingdino

import (
	"image"

	"github.com/disintegration/imaging"

	"visionserve/internal/engine"
)

// Input geometry + ImageNet normalization (verified against the exported ONNX graph).
const (
	inputSize = 800 // GroundingDINO pixel_values is [1,3,800,800]
)

var (
	imagenetMean = [3]float32{0.485, 0.456, 0.406}
	imagenetStd  = [3]float32{0.229, 0.224, 0.225}
)

// preprocessImage builds pixel_values and pixel_mask.
//
// The image is force-resized to EXACTLY 800x800 (a plain squash, NO aspect ratio / NO
// letterbox), then per channel: v/255, then (v-mean)/std, laid out NCHW float32.
// pixel_mask is all ones [1,800,800] int64.
func preprocessImage(img image.Image) (pixelValues, pixelMask engine.Tensor) {
	resized := imaging.Resize(img, inputSize, inputSize, imaging.Linear) // *image.NRGBA, bilinear

	plane := inputSize * inputSize
	data := make([]float32, 3*plane)
	for y := 0; y < inputSize; y++ {
		for x := 0; x < inputSize; x++ {
			c := resized.NRGBAAt(resized.Bounds().Min.X+x, resized.Bounds().Min.Y+y)
			idx := y*inputSize + x
			data[idx] = (float32(c.R)/255 - imagenetMean[0]) / imagenetStd[0]
			data[plane+idx] = (float32(c.G)/255 - imagenetMean[1]) / imagenetStd[1]
			data[2*plane+idx] = (float32(c.B)/255 - imagenetMean[2]) / imagenetStd[2]
		}
	}

	mask := make([]int64, plane)
	for i := range mask {
		mask[i] = 1
	}

	return engine.F32(data, 1, 3, inputSize, inputSize), engine.I64(mask, 1, inputSize, inputSize)
}
