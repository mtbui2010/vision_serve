package efficientsam

import (
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/vision/preprocess"
)

// encoderSpec is the EfficientSAM encoder input.
//
// VERIFIED against models/efficient-sam/efficient_sam_encoder.onnx (graph inspection):
// the graph's first ops are
//
//	Resize(batched_images → [1024, 1024], mode=linear, half_pixel)   // squash, no aspect keep
//	Sub([0.485, 0.456, 0.406]) → Div([0.229, 0.224, 0.225])         // ImageNet norm
//
// so the encoder expects the ORIGINAL-resolution image as NCHW float32 in [0, 1]
// (pixel/255) — exactly what the official yformer/EfficientSAM ONNX example feeds.
// Resizing or ImageNet-normalizing in Go as well would apply both steps twice (the
// previous implementation did, which gave ~0.01 mask IoU vs the reference).
// Fixed by the export, not read from the manifest.
var encoderSpec = preprocess.Spec{Resize: preprocess.None}

// encoderInput returns the [1, 3, H, W] encoder tensor at the original image size.
func encoderInput(img image.Image) (engine.Tensor, error) {
	t, _, err := encoderSpec.Apply(img)
	return t, err
}
