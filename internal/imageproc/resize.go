package imageproc

import (
	"image"

	"github.com/disintegration/imaging"
)

// Resize resizes the image to exactly WxH (does NOT preserve aspect ratio). For models without letterboxing.
// When aspect ratio must be preserved, use Letterbox.
func Resize(src image.Image, w, h int) *image.NRGBA {
	return imaging.Resize(src, w, h, imaging.Linear)
}

// ResizeBicubic is Resize with a CUBIC filter instead of a linear one, for checkpoints whose own
// preprocessor specifies bicubic — SigLIP's `preprocessor_config.json` does.
//
// It is not interchangeable with Resize. The filter is part of a vision tower's input contract: a
// head fitted against bicubic crops and served linear ones sees a slightly different distribution
// than it was trained on, and nothing downstream reports an error. CatmullRom is the cubic PIL's
// BICUBIC implements (a = -0.5), which is what the reference preprocessing eventually calls.
func ResizeBicubic(src image.Image, w, h int) *image.NRGBA {
	return imaging.Resize(src, w, h, imaging.CatmullRom)
}

// ResizeScale returns the per-axis scale factors (to map coordinates back when not letterboxing).
func ResizeScale(origW, origH, dstW, dstH int) (scaleX, scaleY float64) {
	return float64(dstW) / float64(origW), float64(dstH) / float64(origH)
}
