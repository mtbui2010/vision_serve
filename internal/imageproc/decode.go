package imageproc

import (
	"bytes"
	"image"

	"github.com/disintegration/imaging"
	"golang.org/x/image/webp"
)

// Decode decodes an encoded image (JPEG, PNG, WebP, BMP, GIF, TIFF) for inference:
//
//   - EXIF orientation is applied (imaging.AutoOrientation), so a phone photo is processed the
//     way every viewer shows it (and the way transformers' load_image feeds models);
//   - a lossy WebP is converted to RGB the way libwebp does it (DecodeWebP).
//
// It does not bound the input; callers that take untrusted bytes check the header first
// (image.DecodeConfig) as the server does.
func Decode(raw []byte) (image.Image, error) {
	if IsWebP(raw) {
		return DecodeWebP(raw)
	}
	return imaging.Decode(bytes.NewReader(raw), imaging.AutoOrientation(true))
}

// IsWebP reports whether raw starts with a WebP container header ("RIFF" size "WEBP").
func IsWebP(raw []byte) bool {
	return len(raw) >= 12 && string(raw[0:4]) == "RIFF" && string(raw[8:12]) == "WEBP"
}

// DecodeWebP decodes a WebP image. A lossless (VP8L) image comes back as decoded. A lossy (VP8)
// one is Y'CbCr in the limited ("studio") range of BT.601, and golang.org/x/image/webp hands it
// back as an image.YCbCr, whose colour model is JPEG's FULL-range transform: converting it that
// way washes colours out (pure blue decoded as (15,15,239) instead of libwebp's (0,0,255)). It
// is converted here with libwebp's own fixed-point transform (src/dsp/yuv.h, VP8YUVToR/G/B),
// nearest chroma sample per pixel (libwebp's default "fancy" upsampling blends neighbouring
// chroma samples, so pixels on a colour edge can differ by a few levels).
func DecodeWebP(raw []byte) (image.Image, error) {
	img, err := webp.Decode(bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	switch m := img.(type) {
	case *image.YCbCr:
		return ycbcrToNRGBA(m, nil), nil
	case *image.NYCbCrA:
		return ycbcrToNRGBA(&m.YCbCr, m), nil
	}
	return img, nil
}

func ycbcrToNRGBA(m *image.YCbCr, alpha *image.NYCbCrA) *image.NRGBA {
	b := m.Bounds()
	out := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := b.Min.Y; y < b.Max.Y; y++ {
		row := out.Pix[(y-b.Min.Y)*out.Stride:]
		for x := b.Min.X; x < b.Max.X; x++ {
			yy := int(m.Y[m.YOffset(x, y)])
			ci := m.COffset(x, y)
			u, v := int(m.Cb[ci]), int(m.Cr[ci])
			p := row[(x-b.Min.X)*4:]
			p[0] = vp8Clip8(mulHi(yy, 19077) + mulHi(v, 26149) - 14234)
			p[1] = vp8Clip8(mulHi(yy, 19077) - mulHi(u, 6419) - mulHi(v, 13320) + 8708)
			p[2] = vp8Clip8(mulHi(yy, 19077) + mulHi(u, 33050) - 17685)
			p[3] = 255
			if alpha != nil {
				p[3] = alpha.A[alpha.AOffset(x, y)]
			}
		}
	}
	return out
}

// mulHi and vp8Clip8 are libwebp's MultHi and VP8Clip8 (YUV_FIX2 = 6).
func mulHi(v, coeff int) int { return (v * coeff) >> 8 }

func vp8Clip8(v int) uint8 {
	const fix2, mask2 = 6, (256 << 6) - 1
	switch {
	case v&^mask2 == 0:
		return uint8(v >> fix2)
	case v < 0:
		return 0
	}
	return 255
}
