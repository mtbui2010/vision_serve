// Package imageproc holds shared image-processing utilities, written in pure Go
// (github.com/disintegration/imaging + the image package) to keep the binary lightweight.
// Do NOT use OpenCV/cgo here (CLAUDE.md).
package imageproc

import (
	"image"
	"image/color"

	"github.com/disintegration/imaging"
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
// padColor is the background color of the pad region.
//
// TODO(verify): the correct pad color for RF-DETR needs to be confirmed against how the model was
// trained/exported (many DETRs pad with 0, some pipelines use gray 114). Defaults to (0,0,0).
func Letterbox(src image.Image, w, h int, padColor color.NRGBA) LetterboxResult {
	b := src.Bounds()
	sw, sh := b.Dx(), b.Dy()

	scale := float64(w) / float64(sw)
	if s := float64(h) / float64(sh); s < scale {
		scale = s
	}
	newW := int(float64(sw)*scale + 0.5)
	newH := int(float64(sh)*scale + 0.5)
	if newW < 1 {
		newW = 1
	}
	if newH < 1 {
		newH = 1
	}

	resized := imaging.Resize(src, newW, newH, imaging.Linear)

	canvas := imaging.New(w, h, padColor)
	padX := (w - newW) / 2
	padY := (h - newH) / 2
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
