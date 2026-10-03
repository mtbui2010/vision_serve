package scrfd

import (
	"image"
	"image/color"
	"math"
	"testing"

	"visionserve/internal/models"
	prep "visionserve/internal/vision/preprocess"
)

// TestFitTopLeft mirrors InsightFace's resize arithmetic (int() truncation, one scale).
func TestFitTopLeft(t *testing.T) {
	cases := []struct {
		w, h, nw, nh int
		scale        float64
	}{
		{640, 566, 640, 566, 1},              // COCO 000000489842: no resize, pad bottom
		{1280, 720, 640, 360, 0.5},           // landscape
		{486, 640, 486, 640, 1},              // portrait, already fits
		{1000, 3000, 213, 640, 640.0 / 3000}, // int(640/3) = 213
	}
	for _, c := range cases {
		nw, nh, s := prep.TopLeftSize(c.w, c.h, 640, 640)
		if nw != c.nw || nh != c.nh || math.Abs(s-c.scale) > 1e-12 {
			t.Errorf("%dx%d: got %dx%d scale %v, want %dx%d scale %v", c.w, c.h, nw, nh, s, c.nw, c.nh, c.scale)
		}
	}
}

// TestPreprocess_TopLeftPadAndNormalization: the image is placed at (0,0) (InsightFace
// det_img[:nh,:nw] = resized), padding is black → (0-127.5)/128, and meta has no pad.
func TestPreprocess_TopLeftPadAndNormalization(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 64, 32))
	for y := 0; y < 32; y++ {
		for x := 0; x < 64; x++ {
			img.Set(x, y, color.NRGBA{255, 128, 0, 255})
		}
	}
	cfg := models.Config{Width: 64, Height: 64, Mean: []float32{127.5, 127.5, 127.5}, Std: []float32{128, 128, 128}}
	ten, meta, err := preprocess(img, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(ten.Shape) != 4 || ten.Shape[1] != 3 || ten.Shape[2] != 64 || ten.Shape[3] != 64 {
		t.Fatalf("shape %v, want [1 3 64 64]", ten.Shape)
	}
	if meta.PadX != 0 || meta.PadY != 0 || meta.ScaleX != 1 || meta.ScaleY != 1 {
		t.Fatalf("meta %+v, want no pad and scale 1", meta)
	}
	at := func(c, y, x int) float32 { return ten.Data[c*64*64+y*64+x] }
	const eps = 1e-4
	// top rows = image (R=255 → 0.99609, G=128 → 0.00391, B=0 → -0.99609), RGB order
	if math.Abs(float64(at(0, 0, 0))-(255-127.5)/128) > eps || math.Abs(float64(at(1, 10, 5))-(128-127.5)/128) > eps || math.Abs(float64(at(2, 31, 63))-(0-127.5)/128) > eps {
		t.Errorf("image region values wrong: %v %v %v", at(0, 0, 0), at(1, 10, 5), at(2, 31, 63))
	}
	// bottom rows = black padding
	if math.Abs(float64(at(0, 40, 10))-(0-127.5)/128) > eps {
		t.Errorf("padding value %v, want %v", at(0, 40, 10), (0-127.5)/128)
	}
}
