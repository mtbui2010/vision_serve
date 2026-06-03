package clip

import (
	"image"
	"image/color"
	"math"
	"testing"

	"visionserve/internal/models"
)

// fillRGBA creates a w×h NRGBA test image filled with a single colour (r,g,b,255).
func fillRGBA(w, h int, r, g, b uint8) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	c := color.NRGBA{R: r, G: g, B: b, A: 255}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetNRGBA(x, y, c)
		}
	}
	return img
}

// TestPreprocessOutputShape verifies that a non-square input is squashed to 224×224
// and the tensor has the correct NCHW shape [1,3,224,224].
func TestPreprocessOutputShape(t *testing.T) {
	cfg := models.Config{Name: "clip", Width: 224, Height: 224, Letterbox: false}
	img := fillRGBA(640, 480, 100, 150, 200)

	ten, meta, err := preprocess(img, cfg)
	if err != nil {
		t.Fatalf("preprocess error: %v", err)
	}

	wantShape := []int64{1, 3, 224, 224}
	if len(ten.Shape) != 4 {
		t.Fatalf("tensor rank = %d, want 4 (NCHW)", len(ten.Shape))
	}
	for i, want := range wantShape {
		if ten.Shape[i] != want {
			t.Fatalf("shape[%d] = %d, want %d (full shape %v)", i, ten.Shape[i], want, ten.Shape)
		}
	}
	if int64(len(ten.Data)) != ten.NumElements() {
		t.Fatalf("len(Data)=%d != NumElements=%d", len(ten.Data), ten.NumElements())
	}

	// Meta: original dimensions preserved, no padding.
	if meta.OrigWidth != 640 || meta.OrigHeight != 480 {
		t.Fatalf("orig = %dx%d, want 640x480", meta.OrigWidth, meta.OrigHeight)
	}
	if meta.PadX != 0 || meta.PadY != 0 {
		t.Fatalf("pad = (%d,%d), want (0,0) — CLIP does not letterbox", meta.PadX, meta.PadY)
	}
}

// TestPreprocessCLIPNormalization checks that pixel values are normalised with CLIP's
// specific mean/std (different from standard ImageNet).
func TestPreprocessCLIPNormalization(t *testing.T) {
	mean := []float32{0.48145466, 0.4578275, 0.40821073}
	std := []float32{0.26862954, 0.26130258, 0.27577711}
	cfg := models.Config{
		Name:      "clip",
		Width:     224,
		Height:    224,
		Letterbox: false,
		Mean:      mean,
		Std:       std,
	}

	r, g, b := uint8(127), uint8(64), uint8(200)
	img := fillRGBA(224, 224, r, g, b)

	ten, _, err := preprocess(img, cfg)
	if err != nil {
		t.Fatalf("preprocess error: %v", err)
	}

	W, H := 224, 224
	plane := H * W

	rGot := ten.Data[0]
	gGot := ten.Data[plane]
	bGot := ten.Data[2*plane]

	rWant := (float32(r)/255.0 - mean[0]) / std[0]
	gWant := (float32(g)/255.0 - mean[1]) / std[1]
	bWant := (float32(b)/255.0 - mean[2]) / std[2]

	const tol = 1e-4
	if math.Abs(float64(rGot-rWant)) > tol {
		t.Fatalf("R = %v, want %v", rGot, rWant)
	}
	if math.Abs(float64(gGot-gWant)) > tol {
		t.Fatalf("G = %v, want %v", gGot, gWant)
	}
	if math.Abs(float64(bGot-bWant)) > tol {
		t.Fatalf("B = %v, want %v", bGot, bWant)
	}
}

// TestPreprocessDefaultNormalization confirms CLIP defaults are used when manifest
// Mean/Std fields are left empty.
func TestPreprocessDefaultNormalization(t *testing.T) {
	cfg := models.Config{Name: "clip", Width: 224, Height: 224} // no Mean/Std

	r, g, b := uint8(255), uint8(255), uint8(255)
	img := fillRGBA(224, 224, r, g, b)

	ten, _, err := preprocess(img, cfg)
	if err != nil {
		t.Fatalf("preprocess error: %v", err)
	}

	// With all-white pixels (1.0 per channel) and CLIP defaults:
	//   R: (1.0 - 0.48145466) / 0.26862954 ≈ 1.929
	//   G: (1.0 - 0.4578275)  / 0.26130258 ≈ 2.075
	//   B: (1.0 - 0.40821073) / 0.27577711 ≈ 2.146
	W, H := 224, 224
	plane := H * W

	rWant := (1.0 - defaultMean[0]) / defaultStd[0]
	gWant := (1.0 - defaultMean[1]) / defaultStd[1]
	bWant := (1.0 - defaultMean[2]) / defaultStd[2]

	const tol = 1e-4
	if math.Abs(float64(ten.Data[0]-rWant)) > tol {
		t.Fatalf("R default mean/std: got %v, want %v", ten.Data[0], rWant)
	}
	if math.Abs(float64(ten.Data[plane]-gWant)) > tol {
		t.Fatalf("G default mean/std: got %v, want %v", ten.Data[plane], gWant)
	}
	if math.Abs(float64(ten.Data[2*plane]-bWant)) > tol {
		t.Fatalf("B default mean/std: got %v, want %v", ten.Data[2*plane], bWant)
	}
}

// TestPreprocessScaleMetaNonSquare verifies that ScaleX and ScaleY are set correctly
// when the input has a non-1:1 aspect ratio (squash resize).
func TestPreprocessScaleMetaNonSquare(t *testing.T) {
	cfg := models.Config{Name: "clip", Width: 224, Height: 224, Letterbox: false}
	// 448×336 image: ScaleX = 224/448 = 0.5, ScaleY = 224/336 ≈ 0.6667
	img := fillRGBA(448, 336, 0, 0, 0)

	_, meta, err := preprocess(img, cfg)
	if err != nil {
		t.Fatalf("preprocess error: %v", err)
	}

	wantScaleX := 224.0 / 448.0
	wantScaleY := 224.0 / 336.0

	if math.Abs(meta.ScaleX-wantScaleX) > 1e-9 {
		t.Fatalf("ScaleX = %v, want %v", meta.ScaleX, wantScaleX)
	}
	if math.Abs(meta.ScaleY-wantScaleY) > 1e-9 {
		t.Fatalf("ScaleY = %v, want %v", meta.ScaleY, wantScaleY)
	}
}
