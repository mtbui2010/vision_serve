package sam2

import (
	"fmt"
	"image"

	"github.com/disintegration/imaging"

	"visionserve/internal/engine"
)

// SAM2 pixel normalization constants (ImageNet mean/std, upstream SAM2Transforms).
// The SharpAI/sam2-hiera-tiny-onnx encoder does NOT bake normalization into the graph
// (verified: feeding the upstream-normalized tensor reproduces the upstream masks), so
// Go applies it before ORT.
var (
	sam2Mean = [3]float32{0.485, 0.456, 0.406} // R, G, B
	sam2Std  = [3]float32{0.229, 0.224, 0.225}
)

// encoderInput builds the SAM2 encoder input tensor, mirroring upstream
// sam2/utils/transforms.py SAM2Transforms exactly:
//
//	ToTensor (/255) → Resize((1024, 1024)) → Normalize(ImageNet mean/std)
//
// i.e. the image is SQUASHED to 1024×1024 (aspect ratio NOT preserved, no padding).
// Prompt coordinates must therefore be scaled PER AXIS (x·1024/W, y·1024/H — upstream
// transform_coords with normalize_coords=True), and the 256×256 low-res mask covers the
// whole image, so it is upsampled straight to (H, W) (upstream postprocess_masks).
//
// Returns the NCHW [1,3,1024,1024] float32 tensor and the per-axis scale factors that
// map ORIGINAL pixel coords into the 1024-space.
func encoderInput(img image.Image) (tensor engine.Tensor, scaleX, scaleY float32, err error) {
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()
	if origW <= 0 || origH <= 0 {
		return engine.Tensor{}, 0, 0, fmt.Errorf("sam2: empty image %dx%d", origW, origH)
	}

	// Bilinear (imaging.Linear widens its support when downscaling, i.e. antialiased —
	// the same as torchvision Resize's default antialias=True).
	resized := imaging.Resize(img, encoderSize, encoderSize, imaging.Linear) // *image.NRGBA

	const plane = encoderSize * encoderSize
	data := make([]float32, 3*plane)
	for y := 0; y < encoderSize; y++ {
		row := resized.Pix[y*resized.Stride : y*resized.Stride+encoderSize*4]
		for x := 0; x < encoderSize; x++ {
			p := row[x*4 : x*4+3]
			i := y*encoderSize + x
			data[i] = (float32(p[0])/255.0 - sam2Mean[0]) / sam2Std[0]
			data[plane+i] = (float32(p[1])/255.0 - sam2Mean[1]) / sam2Std[1]
			data[2*plane+i] = (float32(p[2])/255.0 - sam2Mean[2]) / sam2Std[2]
		}
	}

	tensor = engine.F32(data, 1, 3, int64(encoderSize), int64(encoderSize))
	return tensor, float32(encoderSize) / float32(origW), float32(encoderSize) / float32(origH), nil
}
