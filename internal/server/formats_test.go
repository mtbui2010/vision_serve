package server

import (
	"bytes"
	"image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"
	"os"
	"testing"

	"golang.org/x/image/bmp"
	"golang.org/x/image/tiff"
)

// quadrants is the 24x16 test picture of testdata/make_webp.py: red, green, blue and white
// 12x8 quadrants.
func quadrants() *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, 24, 16))
	colors := []color.NRGBA{{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255}, {255, 255, 255, 255}}
	for y := 0; y < 16; y++ {
		for x := 0; x < 24; x++ {
			img.SetNRGBA(x, y, colors[(y/8)*2+x/12])
		}
	}
	return img
}

// quadrantCentres are the pixels checkPixels reads: the centre of each quadrant.
var quadrantCentres = []image.Point{{6, 4}, {18, 4}, {6, 12}, {18, 12}}

// checkPixels compares img's quadrant centres with want (8-bit RGB), within tol per channel.
func checkPixels(t *testing.T, name string, img image.Image, want [4][3]int, tol int) {
	t.Helper()
	if b := img.Bounds(); b.Dx() != 24 || b.Dy() != 16 {
		t.Fatalf("%s: decoded %dx%d, want 24x16", name, b.Dx(), b.Dy())
	}
	for i, p := range quadrantCentres {
		r, g, b, _ := img.At(p.X, p.Y).RGBA()
		got := [3]int{int(r >> 8), int(g >> 8), int(b >> 8)}
		for c := range got {
			if d := got[c] - want[i][c]; d > tol || d < -tol {
				t.Errorf("%s: pixel %v = %v, want %v ±%d", name, p, got, want[i], tol)
				break
			}
		}
	}
}

// exact is quadrants() at the quadrant centres.
var exact = [4][3]int{{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 255}}

// Every format the docs promise decodes through decodeImage (the path every upload takes). WebP
// was refused with 400 ("image: unknown format"). Its fixtures were written by PIL
// (testdata/make_webp.py) and the expected pixels are PIL's (libwebp's) decode of them: the lossy
// one must match libwebp's colours, not golang.org/x/image/webp's full-range YCbCr conversion,
// which gives (15,15,239) for the blue quadrant.
func TestDecodeImageFormats(t *testing.T) {
	for _, c := range []struct {
		file string
		want [4][3]int // PIL: Image.open(file).convert("RGB").getpixel(centre)
		tol  int
	}{
		{"testdata/quadrants.webp", exact, 0}, // lossless (VP8L)
		{"testdata/quadrants-lossy.webp", [4][3]int{{255, 1, 0}, {0, 255, 1}, {0, 0, 255}, {255, 255, 255}}, 2}, // lossy (VP8)
	} {
		raw, err := os.ReadFile(c.file)
		if err != nil {
			t.Fatal(err)
		}
		img, err := decodeImage(bytes.NewReader(raw))
		if err != nil {
			t.Fatalf("%s: %v", c.file, err)
		}
		checkPixels(t, c.file, img, c.want, c.tol)
	}

	encoders := map[string]func(*bytes.Buffer, image.Image) error{
		"png":  func(b *bytes.Buffer, m image.Image) error { return png.Encode(b, m) },
		"jpeg": func(b *bytes.Buffer, m image.Image) error { return jpeg.Encode(b, m, &jpeg.Options{Quality: 95}) },
		"gif":  func(b *bytes.Buffer, m image.Image) error { return gif.Encode(b, m, nil) },
		"bmp":  func(b *bytes.Buffer, m image.Image) error { return bmp.Encode(b, m) },
		"tiff": func(b *bytes.Buffer, m image.Image) error { return tiff.Encode(b, m, nil) },
	}
	for name, enc := range encoders {
		var b bytes.Buffer
		if err := enc(&b, quadrants()); err != nil {
			t.Fatal(err)
		}
		img, err := decodeImage(&b)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		tol := 0
		if name == "jpeg" {
			tol = 8
		}
		checkPixels(t, name, img, exact, tol)
	}
}
