package preprocess

import (
	"image"
	"image/color"
)

// toNRGBA returns img as an *image.NRGBA whose Pix starts at the image's top-left. An NRGBA is
// returned as is; every other type is converted pixel by pixel EXACTLY as image.NRGBA.Set does
// (color.NRGBAModel: the 16-bit premultiplied RGBA() of the pixel, un-premultiplied, >> 8), but
// through the concrete XxxAt accessors, so no color.Color is boxed per pixel. A JPEG decodes to
// *image.YCbCr; converting a 12 MP photo through At() allocated twice per pixel.
func toNRGBA(img image.Image) *image.NRGBA {
	if n, ok := img.(*image.NRGBA); ok {
		return n
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	dst := image.NewNRGBA(image.Rect(0, 0, w, h))
	put := func(x, y int, r, g, bl, a uint32) {
		i := y*dst.Stride + x*4
		dst.Pix[i], dst.Pix[i+1], dst.Pix[i+2], dst.Pix[i+3] = nrgba8(r, g, bl, a)
	}
	switch src := img.(type) {
	case *image.YCbCr:
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				r, g, bl, a := src.YCbCrAt(b.Min.X+x, b.Min.Y+y).RGBA()
				put(x, y, r, g, bl, a)
			}
		}
	case *image.RGBA:
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				r, g, bl, a := src.RGBAAt(b.Min.X+x, b.Min.Y+y).RGBA()
				put(x, y, r, g, bl, a)
			}
		}
	case *image.Gray:
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				r, g, bl, a := src.GrayAt(b.Min.X+x, b.Min.Y+y).RGBA()
				put(x, y, r, g, bl, a)
			}
		}
	case *image.Gray16:
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				r, g, bl, a := src.Gray16At(b.Min.X+x, b.Min.Y+y).RGBA()
				put(x, y, r, g, bl, a)
			}
		}
	case *image.RGBA64:
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				r, g, bl, a := src.RGBA64At(b.Min.X+x, b.Min.Y+y).RGBA()
				put(x, y, r, g, bl, a)
			}
		}
	case *image.NRGBA64:
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				r, g, bl, a := src.NRGBA64At(b.Min.X+x, b.Min.Y+y).RGBA()
				put(x, y, r, g, bl, a)
			}
		}
	case *image.NYCbCrA:
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				r, g, bl, a := src.NYCbCrAAt(b.Min.X+x, b.Min.Y+y).RGBA()
				put(x, y, r, g, bl, a)
			}
		}
	case *image.CMYK:
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				r, g, bl, a := src.CMYKAt(b.Min.X+x, b.Min.Y+y).RGBA()
				put(x, y, r, g, bl, a)
			}
		}
	case *image.Paletted:
		if !palettedToNRGBA(dst, src) {
			genericToNRGBA(dst, img)
		}
	default:
		genericToNRGBA(dst, img)
	}
	return dst
}

// nrgba8 is color.NRGBAModel's conversion of a premultiplied 16-bit RGBA to 8-bit NRGBA.
func nrgba8(r, g, b, a uint32) (uint8, uint8, uint8, uint8) {
	if a == 0xffff {
		return uint8(r >> 8), uint8(g >> 8), uint8(b >> 8), 0xff
	}
	if a == 0 {
		return 0, 0, 0, 0
	}
	r = (r * 0xffff) / a
	g = (g * 0xffff) / a
	b = (b * 0xffff) / a
	return uint8(r >> 8), uint8(g >> 8), uint8(b >> 8), uint8(a >> 8)
}

// palettedToNRGBA converts through a per-palette-entry table (color.NRGBAModel.Convert of each
// entry, which returns a color.NRGBA entry untouched). It reports false — leaving the work to
// the generic path, which behaves as At() always did — when a pixel indexes past the palette.
func palettedToNRGBA(dst *image.NRGBA, src *image.Paletted) bool {
	lut := make([]color.NRGBA, len(src.Palette))
	for i, c := range src.Palette {
		if c == nil {
			return false
		}
		lut[i] = color.NRGBAModel.Convert(c).(color.NRGBA)
	}
	b := src.Rect
	w, h := b.Dx(), b.Dy()
	for y := 0; y < h; y++ {
		row := src.Pix[y*src.Stride : y*src.Stride+w]
		for _, idx := range row {
			if int(idx) >= len(lut) {
				return false
			}
		}
		o := y * dst.Stride
		for x, idx := range row {
			c := lut[idx]
			dst.Pix[o+4*x], dst.Pix[o+4*x+1], dst.Pix[o+4*x+2], dst.Pix[o+4*x+3] = c.R, c.G, c.B, c.A
		}
	}
	return true
}

// genericToNRGBA is the reference conversion (image.NRGBA.Set of each At()) for image types
// without a fast path.
func genericToNRGBA(dst *image.NRGBA, img image.Image) {
	b := img.Bounds()
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			dst.Set(x, y, img.At(b.Min.X+x, b.Min.Y+y))
		}
	}
}
