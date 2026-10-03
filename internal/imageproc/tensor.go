package imageproc

import (
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/vision/preprocess"
)

// ImageToCHWFloat converts an image into a float32 tensor with NCHW layout [1,3,H,W],
// normalized by mean/std (ImageNet-style): v' = (v/255 - mean[c]) / std[c].
// If mean/std are empty -> only divide by 255 (mapping to [0,1]).
//
// Channel order: RGB (channel 0=R, 1=G, 2=B).
//
// It is vision/preprocess's shared normalise step (preprocess.Spec.Tensor); new code should
// declare a preprocess.Spec instead of resizing and calling this by hand.
func ImageToCHWFloat(img image.Image, mean, std []float32) engine.Tensor {
	return preprocess.Spec{Mean: mean, Std: std}.Tensor(img)
}
