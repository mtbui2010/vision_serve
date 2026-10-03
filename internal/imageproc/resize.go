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

// ResizeShortCenterCrop resizes src so that it COVERS w×h while keeping its aspect ratio, bicubic,
// then cuts the centred w×h window: the CLIP recipe, written to match HuggingFace's
// CLIPImageProcessor pixel for pixel (get_resize_output_image_size + center_crop):
//
//	short side -> its target, long side -> int(target * long / short)   (truncated)
//	offset     -> (resized - crop) // 2                                  (floored)
//
// It returns the crop, the per-axis resize scale and the crop offset in RESIZED pixels, so
// input = orig * scale - offset on each axis.
func ResizeShortCenterCrop(src image.Image, w, h int) (*image.NRGBA, float64, float64, int, int) {
	b := src.Bounds()
	ow, oh := b.Dx(), b.Dy()
	var rw, rh int
	if ow <= oh { // width is the short side
		rw = w
		rh = int(float64(w) * float64(oh) / float64(ow))
	} else {
		rh = h
		rw = int(float64(h) * float64(ow) / float64(oh))
	}
	if rw < w {
		rw = w
	}
	if rh < h {
		rh = h
	}
	resized := imaging.Resize(src, rw, rh, imaging.CatmullRom)
	offX, offY := (rw-w)/2, (rh-h)/2
	crop := imaging.Crop(resized, image.Rect(offX, offY, offX+w, offY+h))
	return crop, float64(rw) / float64(ow), float64(rh) / float64(oh), offX, offY
}
