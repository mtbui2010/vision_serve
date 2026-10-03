package preprocess_test

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"math/rand"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/vision/preprocess"
)

// testImage is one named input of the equivalence tests.
type testImage struct {
	name string
	img  image.Image
}

// sizes covers square, landscape, portrait, odd, 1-pixel and extreme aspect ratios.
var sizes = [][2]int{
	{1, 1}, {1, 2}, {2, 1}, {3, 5}, {5, 3}, {7, 1}, {1, 7}, {17, 31}, {31, 17}, {64, 64},
	{97, 173}, {173, 97}, {224, 224}, {225, 223}, {300, 500}, {500, 300}, {518, 518},
	{560, 420}, {640, 480}, {480, 640}, {799, 601}, {810, 1080}, {1023, 767}, {1024, 1024},
	{1280, 720}, {1500, 200}, {200, 1500}, {2001, 999},
}

// bigOutSizes is the subset used by the 1024×1024 encoders (each call writes a 1024² tensor).
var bigOutSizes = [][2]int{
	{1, 1}, {2, 1}, {3, 5}, {17, 31}, {97, 173}, {173, 97}, {640, 480}, {480, 640},
	{810, 1080}, {1023, 767}, {1024, 1024}, {1500, 200}, {2001, 999},
}

// typeSizes get every image type, not only NRGBA.
var typeSizes = [][2]int{{1, 1}, {3, 5}, {17, 31}, {97, 173}, {640, 480}}

// nrgbaImage: gradients, solid blocks and noise, opaque.
func nrgbaImage(w, h int, seed int64) *image.NRGBA {
	r := rand.New(rand.NewSource(seed))
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := img.PixOffset(x, y)
			img.Pix[i] = uint8((x*255)/max(w, 1) ^ (y & 31))
			img.Pix[i+1] = uint8((y * 255) / max(h, 1))
			img.Pix[i+2] = uint8(128 + 100*math.Sin(float64(x+y)/7))
			img.Pix[i+3] = 255
		}
	}
	for k := 0; k < 5; k++ {
		x0, y0 := r.Intn(w), r.Intn(h)
		c := color.NRGBA{uint8(r.Intn(256)), uint8(r.Intn(256)), uint8(r.Intn(256)), 255}
		for y := y0; y < min(h, y0+1+r.Intn(max(h/3, 1))); y++ {
			for x := x0; x < min(w, x0+1+r.Intn(max(w/3, 1))); x++ {
				img.SetNRGBA(x, y, c)
			}
		}
	}
	for i := 0; i < len(img.Pix); i += 4 {
		for c := 0; c < 3; c++ {
			img.Pix[i+c] = uint8(min(255, max(0, int(img.Pix[i+c])+r.Intn(17)-8)))
		}
	}
	return img
}

// allTypes renders the same picture as every image type a decoder (or a caller) can hand in.
func allTypes(w, h int, seed int64) []testImage {
	base := nrgbaImage(w, h, seed)
	r := rand.New(rand.NewSource(seed + 1000))
	rect := base.Rect
	out := []testImage{{"nrgba", base}}

	// Semi-transparent NRGBA (the alpha channel is ignored by the tensor, but not by resizers).
	alpha := image.NewNRGBA(rect)
	copy(alpha.Pix, base.Pix)
	for i := 3; i < len(alpha.Pix); i += 4 {
		alpha.Pix[i] = uint8(r.Intn(256))
	}
	out = append(out, testImage{"nrgba-alpha", alpha})

	// A sub-image: non-zero Bounds().Min, Pix starting mid-buffer.
	big := nrgbaImage(w+5, h+3, seed+7)
	out = append(out, testImage{"nrgba-subimage", big.SubImage(image.Rect(3, 2, 3+w, 2+h))})

	rgba := image.NewRGBA(rect)
	gray := image.NewGray(rect)
	gray16 := image.NewGray16(rect)
	rgba64 := image.NewRGBA64(rect)
	nrgba64 := image.NewNRGBA64(rect)
	cmyk := image.NewCMYK(rect)
	pal := color.Palette{color.Black, color.White, color.NRGBA{200, 10, 30, 128},
		color.RGBA{40, 80, 120, 200}, color.Gray{77}, color.NRGBA64{0x1234, 0xfedc, 0x8000, 0xff00}}
	for len(pal) < 200 {
		pal = append(pal, color.NRGBA{uint8(r.Intn(256)), uint8(r.Intn(256)), uint8(r.Intn(256)), uint8(r.Intn(256))})
	}
	paletted := image.NewPaletted(rect, pal)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := base.NRGBAAt(x, y)
			a := uint8(r.Intn(256))
			rgba.SetRGBA(x, y, color.RGBA{uint8(uint16(c.R) * uint16(a) / 255), uint8(uint16(c.G) * uint16(a) / 255),
				uint8(uint16(c.B) * uint16(a) / 255), a})
			gray.SetGray(x, y, color.Gray{c.G})
			gray16.SetGray16(x, y, color.Gray16{uint16(r.Intn(65536))})
			a16 := uint16(r.Intn(65536))
			rgba64.SetRGBA64(x, y, color.RGBA64{uint16(uint32(c.R) * 257 * uint32(a16) / 65535),
				uint16(uint32(c.G) * 257 * uint32(a16) / 65535), uint16(uint32(c.B) * 257 * uint32(a16) / 65535), a16})
			nrgba64.SetNRGBA64(x, y, color.NRGBA64{uint16(r.Intn(65536)), uint16(r.Intn(65536)), uint16(r.Intn(65536)), uint16(r.Intn(65536))})
			cmyk.SetCMYK(x, y, color.CMYK{uint8(r.Intn(256)), uint8(r.Intn(256)), uint8(r.Intn(256)), uint8(r.Intn(256))})
			paletted.SetColorIndex(x, y, uint8(r.Intn(len(pal))))
		}
	}
	out = append(out, testImage{"rgba", rgba}, testImage{"gray", gray}, testImage{"gray16", gray16},
		testImage{"rgba64", rgba64}, testImage{"nrgba64", nrgba64}, testImage{"cmyk", cmyk},
		testImage{"paletted", paletted})

	for _, ratio := range []image.YCbCrSubsampleRatio{image.YCbCrSubsampleRatio444, image.YCbCrSubsampleRatio422,
		image.YCbCrSubsampleRatio420, image.YCbCrSubsampleRatio440} {
		yc := image.NewYCbCr(rect, ratio)
		for i := range yc.Y {
			yc.Y[i] = uint8(r.Intn(256))
		}
		for i := range yc.Cb {
			yc.Cb[i], yc.Cr[i] = uint8(r.Intn(256)), uint8(r.Intn(256))
		}
		out = append(out, testImage{"ycbcr-" + ratio.String(), yc})
	}
	nya := image.NewNYCbCrA(rect, image.YCbCrSubsampleRatio420)
	for i := range nya.Y {
		nya.Y[i], nya.A[i] = uint8(r.Intn(256)), uint8(r.Intn(256))
	}
	for i := range nya.Cb {
		nya.Cb[i], nya.Cr[i] = uint8(r.Intn(256)), uint8(r.Intn(256))
	}
	out = append(out, testImage{"nycbcra", nya})
	return out
}

// inputs returns NRGBA images at `sz` plus every image type at the typeSizes among them.
func inputs(sz [][2]int) []testImage {
	var out []testImage
	typed := map[[2]int]bool{}
	for _, s := range typeSizes {
		typed[s] = true
	}
	for i, s := range sz {
		if typed[s] {
			for _, ti := range allTypes(s[0], s[1], int64(i+1)) {
				out = append(out, testImage{fmt.Sprintf("%dx%d/%s", s[0], s[1], ti.name), ti.img})
			}
			continue
		}
		out = append(out, testImage{fmt.Sprintf("%dx%d/nrgba", s[0], s[1]), nrgbaImage(s[0], s[1], int64(i+1))})
	}
	return out
}

// sweep returns the inputs of the equivalence sweeps: small for ordinary outputs, big for the
// 1024×1024 encoders. The full sweep runs normally; -short and -race (which slows imaging ~25x,
// and has nothing to find in this single-goroutine-per-call code beyond imaging's own
// parallelism) run a few sizes, every image type still included.
func sweep() (small, big []testImage) {
	switch {
	case raceEnabled:
		return inputs([][2]int{{3, 5}, {17, 31}}), inputs([][2]int{{3, 5}})
	case testing.Short():
		return inputs(typeSizes), inputs([][2]int{{3, 5}, {640, 480}})
	}
	return inputs(sizes), inputs(bigOutSizes)
}

// sameTensor fails unless got and want have the same shape, dtype and BITS.
func sameTensor(t *testing.T, what string, got, want engine.Tensor) {
	t.Helper()
	if fmt.Sprint(got.Shape) != fmt.Sprint(want.Shape) || got.Dtype != want.Dtype {
		t.Fatalf("%s: shape %v dtype %q, want %v %q", what, got.Shape, got.Dtype, want.Shape, want.Dtype)
	}
	if len(got.Data) != len(want.Data) {
		t.Fatalf("%s: %d values, want %d", what, len(got.Data), len(want.Data))
	}
	for i := range got.Data {
		if math.Float32bits(got.Data[i]) != math.Float32bits(want.Data[i]) {
			t.Fatalf("%s: value %d is %v (%#x), want %v (%#x)", what, i, got.Data[i],
				math.Float32bits(got.Data[i]), want.Data[i], math.Float32bits(want.Data[i]))
		}
	}
}

func sameMeta(t *testing.T, what string, got, want preprocess.Meta) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: meta %+v, want %+v", what, got, want)
	}
}
