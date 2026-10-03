// Package imageproc holds shared image-processing utilities, written in pure Go
// (github.com/disintegration/imaging + the image package) to keep the binary lightweight.
// Do NOT use OpenCV/cgo here (CLAUDE.md).
//
// Model input preprocessing lives in internal/vision/preprocess (one Spec-driven
// implementation); the resize / letterbox / tensor helpers here are thin wrappers over it, kept
// for drawing code and the packages that still compose their input by hand.
package imageproc

import (
	"image"
	"image/color"

	"github.com/disintegration/imaging"

	"visionserve/internal/vision/preprocess"
)

// LetterboxResult holds the letterboxed image + the info needed to map coordinates back.
// Relation: input_coord = orig_coord * Scale + Pad.
type LetterboxResult struct {
	Img   *image.NRGBA
	Scale float64
	PadX  int
	PadY  int
}

// Letterbox resizes the image preserving aspect ratio into a WxH frame, then pads it to full size.
// padColor is the background color of the pad region. The geometry is
// preprocess.LetterboxSize; a model that needs the letterboxed TENSOR declares a preprocess.Spec
// with Resize: preprocess.Letterbox instead (no intermediate canvas).
func Letterbox(src image.Image, w, h int, padColor color.NRGBA) LetterboxResult {
	b := src.Bounds()
	newW, newH, scale, padX, padY := preprocess.LetterboxSize(b.Dx(), b.Dy(), w, h)
	resized := imaging.Resize(src, newW, newH, imaging.Linear)
	canvas := imaging.New(w, h, padColor)
	canvas = imaging.Paste(canvas, resized, image.Pt(padX, padY))
	return LetterboxResult{Img: canvas, Scale: scale, PadX: padX, PadY: padY}
}

// MapBoxToOriginal maps a bbox [x,y,w,h] from letterboxed-image coordinates back to the ORIGINAL image.
// This is the most error-prone step (CLAUDE.md) — kept separate to test it thoroughly.
func (lb LetterboxResult) MapBoxToOriginal(x, y, w, h float64) (ox, oy, ow, oh float64) {
	ox = (x - float64(lb.PadX)) / lb.Scale
	oy = (y - float64(lb.PadY)) / lb.Scale
	ow = w / lb.Scale
	oh = h / lb.Scale
	return
}
