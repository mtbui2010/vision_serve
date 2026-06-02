package imageproc

import (
	"image"
	"image/color"
	"math"
	"testing"
)

func solidImage(w, h int) image.Image {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetNRGBA(x, y, color.NRGBA{255, 0, 0, 255})
		}
	}
	return img
}

func TestLetterboxScaleAndPad(t *testing.T) {
	// 200x100 -> 100x100: scale=0.5, newW=100,newH=50, padX=0, padY=25
	lb := Letterbox(solidImage(200, 100), 100, 100, color.NRGBA{0, 0, 0, 255})
	if math.Abs(lb.Scale-0.5) > 1e-9 {
		t.Fatalf("scale = %v, want 0.5", lb.Scale)
	}
	if lb.PadX != 0 || lb.PadY != 25 {
		t.Fatalf("pad = (%d,%d), want (0,25)", lb.PadX, lb.PadY)
	}
	if b := lb.Img.Bounds(); b.Dx() != 100 || b.Dy() != 100 {
		t.Fatalf("canvas size = %v, want 100x100", b)
	}
}

func TestMapBoxToOriginalInverse(t *testing.T) {
	lb := LetterboxResult{Scale: 0.5, PadX: 0, PadY: 25}
	// box covering the real image region in the canvas (x=0,y=25,w=100,h=50) -> the whole 200x100 original
	ox, oy, ow, oh := lb.MapBoxToOriginal(0, 25, 100, 50)
	if ox != 0 || oy != 0 || ow != 200 || oh != 100 {
		t.Fatalf("inverse map = (%v,%v,%v,%v), want (0,0,200,100)", ox, oy, ow, oh)
	}
}
