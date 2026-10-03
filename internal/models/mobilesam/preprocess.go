package mobilesam

import (
	"fmt"
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/vision/preprocess"
)

// encoderSpec is the SAM encoder input. The exported encoder bakes normalization (SAM pixel
// mean/std) AND padding-to-1024 INTO the graph and takes a raw HWC uint8-range image, so the
// spec only resizes the LONG side to 1024 (keeping aspect ratio) and feeds an HWC float32
// tensor [newH, newW, 3] with values still in 0..255.
//
// It is fixed by the export, not read from the manifest: mobile-sam's input block (and that of
// grasp / background / grounded-sam, which build MobileSAM from their own manifest) is
// reference only.
var encoderSpec = preprocess.Spec{
	Resize:    preprocess.LongSide,
	Width:     encoderSize,
	Height:    encoderSize,
	NoRescale: true,
	Layout:    preprocess.HWC,
}

// encoderInput builds the SAM encoder input and returns the coordinate scale:
// scale = 1024 / max(origW, origH) maps original-image coordinates into the resized space the
// decoder's point prompts must use (padding is bottom/right only, so no pad offset is needed on
// coordinates).
func encoderInput(img image.Image) (engine.Tensor, float64, error) {
	t, meta, err := encoderSpec.Apply(img)
	if err != nil {
		return engine.Tensor{}, 0, fmt.Errorf("mobilesam: %w", err)
	}
	return t, meta.ScaleX, nil
}
