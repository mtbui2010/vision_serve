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

// ResizeScale returns the per-axis scale factors (to map coordinates back when not letterboxing).
func ResizeScale(origW, origH, dstW, dstH int) (scaleX, scaleY float64) {
	return float64(dstW) / float64(origW), float64(dstH) / float64(origH)
}
