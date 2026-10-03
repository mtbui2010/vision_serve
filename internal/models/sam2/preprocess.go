package sam2

import (
	"fmt"
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/vision/preprocess"
)

// SAM2 pixel normalization constants (ImageNet mean/std, upstream SAM2Transforms).
// The SharpAI/sam2-hiera-tiny-onnx encoder does NOT bake normalization into the graph
// (verified: feeding the upstream-normalized tensor reproduces the upstream masks), so
// Go applies it before ORT.
var (
	sam2Mean = [3]float32{0.485, 0.456, 0.406} // R, G, B
	sam2Std  = [3]float32{0.229, 0.224, 0.225}
)

// encoderSpec mirrors upstream sam2/utils/transforms.py SAM2Transforms exactly:
//
//	ToTensor (/255) → Resize((1024, 1024)) → Normalize(ImageNet mean/std)
//
// i.e. the image is SQUASHED to 1024×1024 (aspect ratio NOT preserved, no padding), bilinear
// (imaging.Linear widens its support when downscaling, i.e. antialiased — the same as
// torchvision Resize's default antialias=True). NCHW [1,3,1024,1024]. Fixed by the export, not
// read from the manifest.
var encoderSpec = preprocess.Spec{
	Resize: preprocess.Squash,
	Width:  encoderSize,
	Height: encoderSize,
	Mean:   sam2Mean[:],
	Std:    sam2Std[:],
}

// encoderInput builds the SAM2 encoder input tensor (encoderSpec).
//
// Prompt coordinates must be scaled PER AXIS (x·1024/W, y·1024/H — upstream transform_coords
// with normalize_coords=True), and the 256×256 low-res mask covers the whole image, so it is
// upsampled straight to (H, W) (upstream postprocess_masks). It returns the per-axis float32
// scale factors that map ORIGINAL pixel coords into the 1024-space.
func encoderInput(img image.Image) (tensor engine.Tensor, scaleX, scaleY float32, err error) {
	tensor, _, err = encoderSpec.Apply(img)
	if err != nil {
		return engine.Tensor{}, 0, 0, fmt.Errorf("sam2: %w", err)
	}
	b := img.Bounds()
	return tensor, float32(encoderSize) / float32(b.Dx()), float32(encoderSize) / float32(b.Dy()), nil
}
