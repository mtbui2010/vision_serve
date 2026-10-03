package efficientsam

import (
	"fmt"
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/imageproc"
)

// encoderInput builds the EfficientSAM encoder input tensor.
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
//
// Returns a [1, 3, H, W] tensor at the original image size.
func encoderInput(img image.Image) (engine.Tensor, error) {
	b := img.Bounds()
	if b.Dx() <= 0 || b.Dy() <= 0 {
		return engine.Tensor{}, fmt.Errorf("empty image (%dx%d)", b.Dx(), b.Dy())
	}
	// Empty mean/std → plain pixel/255 in [0,1], RGB planes.
	return imageproc.ImageToCHWFloat(img, nil, nil), nil
}
