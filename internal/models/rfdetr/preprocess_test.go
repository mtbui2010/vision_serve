package rfdetr

import (
	"image"
	"image/color"
	"math"
	"testing"

	"visionserve/internal/models"
)

// fillRGBA creates a wxh NRGBA test image, filled with a single color (r,g,b).
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

// RF-DETR preprocess with letterbox: the output tensor has the correct NCHW [1,3,H,W] shape,
// meta keeps the original dimensions, and a NON-square image must produce padding on exactly one axis.
func TestPreprocessLetterboxNonSquare(t *testing.T) {
	m := &rfDETR{cfg: models.Config{
		Name:      "rf-detr",
		Width:     64,
		Height:    64,
		Letterbox: true,
	}}

	// 80x40 image (wider than tall) -> letterbox into 64x64.
	// scale = min(64/80, 64/40) = min(0.8, 1.6) = 0.8
	// newW = 80*0.8 = 64, newH = 40*0.8 = 32
	// padX = (64-64)/2 = 0, padY = (64-32)/2 = 16
	img := fillRGBA(80, 40, 10, 20, 30)

	ten, meta, err := m.preprocess(img)
	if err != nil {
		t.Fatalf("preprocess error: %v", err)
	}

	wantShape := []int64{1, 3, 64, 64}
	if len(ten.Shape) != 4 {
		t.Fatalf("shape len = %d, want 4 (NCHW)", len(ten.Shape))
	}
	for i := range wantShape {
		if ten.Shape[i] != wantShape[i] {
			t.Fatalf("shape = %v, want %v", ten.Shape, wantShape)
		}
	}
	if got := int64(len(ten.Data)); got != ten.NumElements() {
		t.Fatalf("len(Data)=%d differs from NumElements=%d", got, ten.NumElements())
	}

	if meta.OrigWidth != 80 || meta.OrigHeight != 40 {
		t.Fatalf("orig = %dx%d, want 80x40", meta.OrigWidth, meta.OrigHeight)
	}
	if math.Abs(meta.ScaleX-0.8) > 1e-9 || math.Abs(meta.ScaleY-0.8) > 1e-9 {
		t.Fatalf("scale = (%v,%v), want (0.8,0.8)", meta.ScaleX, meta.ScaleY)
	}
	if meta.PadX != 0 || meta.PadY != 16 {
		t.Fatalf("pad = (%d,%d), want (0,16)", meta.PadX, meta.PadY)
	}
	if meta.ScaleX != meta.ScaleY {
		t.Fatalf("letterbox must preserve aspect ratio: ScaleX(%v) != ScaleY(%v)", meta.ScaleX, meta.ScaleY)
	}
}

// Image TALLER than wide -> padding must be on the X axis.
func TestPreprocessLetterboxTallImage(t *testing.T) {
	m := &rfDETR{cfg: models.Config{Width: 64, Height: 64, Letterbox: true}}

	// 40x80 image (taller than wide) -> scale = min(64/40,64/80)=min(1.6,0.8)=0.8
	// newW = 40*0.8 = 32, newH = 80*0.8 = 64
	// padX = (64-32)/2 = 16, padY = (64-64)/2 = 0
	img := fillRGBA(40, 80, 0, 0, 0)
	_, meta, err := m.preprocess(img)
	if err != nil {
		t.Fatalf("preprocess error: %v", err)
	}
	if meta.PadX != 16 || meta.PadY != 0 {
		t.Fatalf("pad = (%d,%d), want (16,0)", meta.PadX, meta.PadY)
	}
	if math.Abs(meta.ScaleX-0.8) > 1e-9 {
		t.Fatalf("scale = %v, want 0.8", meta.ScaleX)
	}
}

// Verify NCHW normalization against the formula v' = (v/255 - mean[c]) / std[c].
// Pick a pixel in the REAL IMAGE region (not the pad region) to compare values.
func TestPreprocessNormalizeNCHW(t *testing.T) {
	mean := []float32{0.485, 0.456, 0.406}
	std := []float32{0.229, 0.224, 0.225}
	m := &rfDETR{cfg: models.Config{
		Width:     64,
		Height:    64,
		Letterbox: true,
		Mean:      mean,
		Std:       std,
	}}

	// Single-color image (100,150,200). The real image region is at padY..padY+newH.
	// 80x40 -> newH=32, padY=16, padX=0 -> row y=16, col x=0 is definitely real image.
	r, g, b := uint8(100), uint8(150), uint8(200)
	img := fillRGBA(80, 40, r, g, b)

	ten, _, err := m.preprocess(img)
	if err != nil {
		t.Fatalf("preprocess error: %v", err)
	}

	W, H := 64, 64
	plane := H * W
	// pixel (x=0, y=16) — inside the real image region (padX=0, padY=16).
	px, py := 0, 16
	idx := py*W + px

	rGot := ten.Data[idx]
	gGot := ten.Data[plane+idx]
	bGot := ten.Data[2*plane+idx]

	rWant := (float32(r)/255.0 - mean[0]) / std[0]
	gWant := (float32(g)/255.0 - mean[1]) / std[1]
	bWant := (float32(b)/255.0 - mean[2]) / std[2]

	// linear resize on a single-color image preserves the color -> small error.
	const tol = 1e-3
	if math.Abs(float64(rGot-rWant)) > tol {
		t.Fatalf("R = %v, want %v", rGot, rWant)
	}
	if math.Abs(float64(gGot-gWant)) > tol {
		t.Fatalf("G = %v, want %v", gGot, gWant)
	}
	if math.Abs(float64(bGot-bWant)) > tol {
		t.Fatalf("B = %v, want %v", bGot, bWant)
	}

	// Pixel in the PAD region (y=0, black pad) -> normalization of 0.
	padIdx := 0*W + 0 // y=0 is in the pad region (padY=16)
	rPadWant := (0.0/255.0 - mean[0]) / std[0]
	if math.Abs(float64(ten.Data[padIdx]-rPadWant)) > tol {
		t.Fatalf("R pad region = %v, want %v (normalization of 0)", ten.Data[padIdx], rPadWant)
	}
}

// When Letterbox=false: hard resize to WxH, ScaleX/ScaleY may differ, pad=0.
func TestPreprocessNoLetterboxScalePerAxis(t *testing.T) {
	m := &rfDETR{cfg: models.Config{Width: 64, Height: 32, Letterbox: false}}

	// 80x40 -> ScaleX = 64/80 = 0.8, ScaleY = 32/40 = 0.8 (equal in this case),
	// use a different ratio to demonstrate per-axis: 100x40 -> ScaleX=64/100=0.64, ScaleY=32/40=0.8
	img := fillRGBA(100, 40, 5, 5, 5)
	ten, meta, err := m.preprocess(img)
	if err != nil {
		t.Fatalf("preprocess error: %v", err)
	}
	if ten.Shape[2] != 32 || ten.Shape[3] != 64 {
		t.Fatalf("shape HxW = %dx%d, want 32x64", ten.Shape[2], ten.Shape[3])
	}
	if math.Abs(meta.ScaleX-0.64) > 1e-9 || math.Abs(meta.ScaleY-0.8) > 1e-9 {
		t.Fatalf("scale = (%v,%v), want (0.64,0.8)", meta.ScaleX, meta.ScaleY)
	}
	if meta.PadX != 0 || meta.PadY != 0 {
		t.Fatalf("pad = (%d,%d), want (0,0) when not letterboxing", meta.PadX, meta.PadY)
	}
	if meta.OrigWidth != 100 || meta.OrigHeight != 40 {
		t.Fatalf("orig = %dx%d, want 100x40", meta.OrigWidth, meta.OrigHeight)
	}
}
