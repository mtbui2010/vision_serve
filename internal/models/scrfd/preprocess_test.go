package scrfd

import (
	"image"
	"image/color"
	"math"
	"testing"

	"visionserve/internal/models"
	prep "visionserve/internal/vision/preprocess"
)

// TestFitTopLeft mirrors InsightFace's resize arithmetic (int() truncation, sides >= 1), and the
// Meta records each axis's own scale nw/w, nh/h — not upstream's single det_scale = nh/h.
func TestFitTopLeft(t *testing.T) {
	cases := []struct {
		w, h, nw, nh   int
		scaleX, scaleY float64
	}{
		{640, 566, 640, 566, 1, 1},                         // COCO 000000489842: no resize, pad bottom
		{1280, 720, 640, 360, 0.5, 0.5},                    // landscape, exact
		{486, 640, 486, 640, 1, 1},                         // portrait, already fits
		{1080, 1920, 360, 640, 360.0 / 1080, 640.0 / 1920}, // portrait, exact
		{1000, 3000, 213, 640, 213.0 / 1000, 640.0 / 3000}, // int(640/3) = 213; det_scale 0.21333 on x
		{1919, 1080, 640, 360, 640.0 / 1919, 360.0 / 1080}, // int(360.19) = 360; det_scale 0.33333 on x
		{1366, 768, 640, 359, 640.0 / 1366, 359.0 / 768},   // int(359.8) = 359
		{2001, 999, 640, 319, 640.0 / 2001, 319.0 / 999},   // geometry_sync sizes
		{999, 2001, 319, 640, 319.0 / 999, 640.0 / 2001},   //
		{17, 31, 350, 640, 350.0 / 17, 640.0 / 31},         // upscaled
		{1, 1, 640, 640, 640, 640},                         //
		{10000, 10, 640, 1, 640.0 / 10000, 1.0 / 10},       // new_h = int(0.64) -> 1: det_scale 0.1 on x
		{10, 10000, 1, 640, 1.0 / 10, 640.0 / 10000},       // new_w = int(0.64) -> 1: det_scale 0.064 on x
		{5000, 3, 640, 1, 640.0 / 5000, 1.0 / 3},           // a side never truncates to 0
	}
	for _, c := range cases {
		nw, nh := prep.TopLeftSize(c.w, c.h, 640, 640)
		if nw != c.nw || nh != c.nh {
			t.Errorf("%dx%d: got %dx%d, want %dx%d", c.w, c.h, nw, nh, c.nw, c.nh)
		}
		_, meta, err := preprocess(image.NewNRGBA(image.Rect(0, 0, c.w, c.h)), models.Config{Width: 640, Height: 640})
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(meta.ScaleX-c.scaleX) > 1e-12 || math.Abs(meta.ScaleY-c.scaleY) > 1e-12 {
			t.Errorf("%dx%d: meta scale %v x %v, want %v x %v", c.w, c.h, meta.ScaleX, meta.ScaleY, c.scaleX, c.scaleY)
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
