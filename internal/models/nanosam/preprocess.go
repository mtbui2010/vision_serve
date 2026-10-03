package nanosam

import (
	"fmt"
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/vision/preprocess"
)

// encoderSpec is the NanoSAM encoder input: long side → 1024 (aspect kept), ImageNet-normalised
// in Go, then zero-padded at the bottom/right to 1024×1024 — SAM's own order (normalise, then
// F.pad with zeros), so the padding is 0 in the NORMALISED tensor. NCHW [1, 3, 1024, 1024].
//
// Why different from MobileSAM: the MobileSAM encoder bakes SAM pixel normalize + pad into its
// ONNX graph and accepts raw HWC float32 in [0,255]; NanoSAM's ResNet-18 encoder expects the
// standard ImageNet-normalized NCHW input that torchvision transforms produce. Fixed by the
// export, not read from the manifest (whose input block is reference only).
var encoderSpec = preprocess.Spec{
	Resize: preprocess.LongSidePad,
	Width:  1024,
	Height: 1024,
	Mean:   imagenetMean[:],
	Std:    imagenetStd[:],
}

// ImageNet normalization constants (NanoSAM's encoder expects pre-normalized input).
var (
	imagenetMean = [3]float32{0.485, 0.456, 0.406}
	imagenetStd  = [3]float32{0.229, 0.224, 0.225}
)

// encoderInput returns the encoder tensor and scale, the factor such that resized_coord =
// original_coord * scale, used to map prompt coordinates into the 1024 space the decoder
// expects. scale is computed in float32 as it always was (it feeds float32 decoder inputs).
func encoderInput(img image.Image) (engine.Tensor, float32, error) {
	t, _, err := encoderSpec.Apply(img)
	if err != nil {
		return engine.Tensor{}, 0, fmt.Errorf("nanosam: %w", err)
	}
	b := img.Bounds()
	return t, float32(1024) / float32(max(b.Dx(), b.Dy())), nil
}
