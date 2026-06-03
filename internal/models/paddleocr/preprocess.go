package paddleocr

import (
	"image"
	"math"

	"github.com/disintegration/imaging"

	"visionserve/internal/engine"
	"visionserve/internal/models"
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

// detPreprocess resizes img so the longer side <= maxSide (cfg.Width or detMaxSide),
// pads width and height up to multiples of 32, normalizes NCHW float32.
// Returns the tensor, and a meta struct for mapping boxes back to original coordinates.
func detPreprocess(img image.Image, maxSide int) (engine.Tensor, detPreprocessMeta) {
	if maxSide <= 0 {
		maxSide = detMaxSide
	}

	b := img.Bounds()
	origW := b.Dx()
	origH := b.Dy()

	// Scale so the longer side <= maxSide.
	scaleX := float64(maxSide) / float64(origW)
	scaleY := float64(maxSide) / float64(origH)
	scale := math.Min(scaleX, scaleY)
	if scale > 1.0 {
		scale = 1.0 // never upscale
	}

	newW := int(math.Round(float64(origW) * scale))
	newH := int(math.Round(float64(origH) * scale))
	if newW < 1 {
		newW = 1
	}
	if newH < 1 {
		newH = 1
	}

	// Pad to multiples of detGridSize.
	padW := ((newW + detGridSize - 1) / detGridSize) * detGridSize
	padH := ((newH + detGridSize - 1) / detGridSize) * detGridSize

	resized := imaging.Resize(img, newW, newH, imaging.Linear) // *image.NRGBA

	plane := padW * padH
	data := make([]float32, 3*plane) // zero-padded by default

	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			c := resized.NRGBAAt(resized.Bounds().Min.X+x, resized.Bounds().Min.Y+y)
			idx := y*padW + x
			data[idx] = (float32(c.R)/255.0 - detNormMean[0]) / detNormStd[0]
			data[plane+idx] = (float32(c.G)/255.0 - detNormMean[1]) / detNormStd[1]
			data[2*plane+idx] = (float32(c.B)/255.0 - detNormMean[2]) / detNormStd[2]
		}
	}

	meta := detPreprocessMeta{
		PreprocessMeta: models.PreprocessMeta{
			OrigWidth:  origW,
			OrigHeight: origH,
			ScaleX:     scale,
			ScaleY:     scale,
			PadX:       0,
			PadY:       0,
		},
		detW: padW,
		detH: padH,
	}

	return engine.F32(data, 1, 3, int64(padH), int64(padW)), meta
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

	plane := recHeight * targetW
	data := make([]float32, 3*plane)

	for y := 0; y < recHeight; y++ {
		for x := 0; x < targetW; x++ {
			c := resized.NRGBAAt(resized.Bounds().Min.X+x, resized.Bounds().Min.Y+y)
			idx := y*targetW + x
			data[idx] = (float32(c.R)/255.0 - recNormMean[0]) / recNormStd[0]
			data[plane+idx] = (float32(c.G)/255.0 - recNormMean[1]) / recNormStd[1]
			data[2*plane+idx] = (float32(c.B)/255.0 - recNormMean[2]) / recNormStd[2]
		}
	}

	return engine.F32(data, 1, 3, recHeight, int64(targetW)), targetW
}
