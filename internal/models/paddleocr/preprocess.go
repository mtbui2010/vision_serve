package paddleocr

import (
	"fmt"
	"image"
	"math"

	"github.com/disintegration/imaging"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/vision/preprocess"
)

// detNormMean and detNormStd are the ImageNet-style normalization constants for DBNet++.
var (
	detNormMean = [3]float32{0.485, 0.456, 0.406}
	detNormStd  = [3]float32{0.229, 0.224, 0.225}
)

// recNormMean and recNormStd are the normalization constants for the SVTR-tiny recognizer.
var (
	recNormMean = [3]float32{0.5, 0.5, 0.5}
	recNormStd  = [3]float32{0.5, 0.5, 0.5}
)

const (
	detMaxSide  = 960 // default max side for det model
	detGridSize = 32  // pad dimensions to multiples of 32
	recHeight   = 48  // fixed height for rec model
	recMaxWidth = 320 // typical max width cap for rec model
)

// detPreprocessMeta holds the info needed to map det-model coordinates back to original.
type detPreprocessMeta struct {
	models.PreprocessMeta
	// detW and detH are the padded detection model input dimensions.
	detW, detH int
}

// detSpec is the DBNet++ det input: the longer side scaled to <= maxSide (never upscaled),
// ImageNet-normalised, then zero-padded at the bottom/right (in the NORMALISED tensor) up to the
// next multiples of 32. maxSide is the manifest's input.width (detMaxSide when unset); the rest
// is fixed by the export.
func detSpec(maxSide int) preprocess.Spec {
	if maxSide <= 0 {
		maxSide = detMaxSide
	}
	return preprocess.Spec{
		Resize:     preprocess.LongSidePad,
		Width:      maxSide,
		Height:     maxSide,
		MultipleOf: detGridSize,
		NoUpscale:  true,
		Mean:       detNormMean[:],
		Std:        detNormStd[:],
	}
}

// detPreprocess resizes img so the longer side <= maxSide (cfg.Width or detMaxSide),
// pads width and height up to multiples of 32, normalizes NCHW float32 (detSpec).
// Returns the tensor, and a meta struct for mapping boxes back to original coordinates.
func detPreprocess(img image.Image, maxSide int) (engine.Tensor, detPreprocessMeta, error) {
	t, meta, err := detSpec(maxSide).Apply(img)
	if err != nil {
		return engine.Tensor{}, detPreprocessMeta{}, fmt.Errorf("paddleocr: %w", err)
	}
	return t, detPreprocessMeta{PreprocessMeta: meta, detW: int(t.Shape[3]), detH: int(t.Shape[2])}, nil
}

// recPreprocess crops the text region [bbox = x,y,w,h in original image coords] from img,
// resizes to h=48 preserving aspect ratio, normalizes NCHW for the SVTR-tiny rec model.
// Returns tensor [1, 3, 48, W] and the actual width W.
func recPreprocess(img image.Image, bbox [4]float64) (engine.Tensor, int) {
	b := img.Bounds()
	origW := b.Dx()
	origH := b.Dy()

	// Clamp bbox to image bounds.
	x0 := int(math.Max(0, math.Round(bbox[0])))
	y0 := int(math.Max(0, math.Round(bbox[1])))
	x1 := int(math.Min(float64(origW), math.Round(bbox[0]+bbox[2])))
	y1 := int(math.Min(float64(origH), math.Round(bbox[1]+bbox[3])))

	if x1 <= x0 {
		x1 = x0 + 1
	}
	if y1 <= y0 {
		y1 = y0 + 1
	}
	if x1 > origW {
		x1 = origW
	}
	if y1 > origH {
		y1 = origH
	}

	cropW := x1 - x0
	cropH := y1 - y0

	// Compute target width: resize height to 48, preserve aspect ratio, cap at recMaxWidth.
	targetW := int(math.Round(float64(cropW) * float64(recHeight) / float64(cropH)))
	if targetW < 1 {
		targetW = 1
	}
	if targetW > recMaxWidth {
		targetW = recMaxWidth
	}

	// Crop and resize.
	cropped := imaging.Crop(img, image.Rect(x0, y0, x1, y1))
	resized := imaging.Resize(cropped, targetW, recHeight, imaging.Linear)
	if resized.Rect.Dx() != targetW || resized.Rect.Dy() != recHeight {
		// A box entirely past the image edge crops to nothing and imaging returns an empty
		// image; the rec input is then all zero pixels (it always was), never a 0-wide tensor.
		resized = image.NewNRGBA(image.Rect(0, 0, targetW, recHeight))
	}

	return recNorm.Tensor(resized), targetW
}

// recNorm is the SVTR-tiny normalisation ((p/255 - 0.5) / 0.5, NCHW) of a text-line crop. The
// crop + height-48 resize stays model-specific; the pixels → tensor step is the shared one.
var recNorm = preprocess.Spec{Mean: recNormMean[:], Std: recNormStd[:]}
