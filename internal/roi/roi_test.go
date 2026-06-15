package roi

import "testing"

func TestClampPixelsAndNormalized(t *testing.T) {
	w, h := 1280, 720
	// pixel ROI
	px, ok := Clamp([4]float64{300, 380, 700, 330}, w, h)
	if !ok || px.Min.X != 300 || px.Min.Y != 380 || px.Max.X != 1000 || px.Max.Y != 710 {
		t.Fatalf("pixel ROI = %v ok=%v", px, ok)
	}
	// normalized ROI (same region as fractions) → must scale to the same rectangle
	nm, ok := Clamp([4]float64{300.0 / 1280, 380.0 / 720, 700.0 / 1280, 330.0 / 720}, w, h)
	if !ok {
		t.Fatal("normalized ROI not ok")
	}
	// allow ±1px rounding
	if abs(nm.Min.X-300) > 1 || abs(nm.Min.Y-380) > 1 || abs(nm.Max.X-1000) > 1 || abs(nm.Max.Y-710) > 1 {
		t.Fatalf("normalized ROI = %v, want ~[300,380]-[1000,710]", nm)
	}
}

func TestClampDegenerateAndBeyond(t *testing.T) {
	w, h := 100, 100
	if _, ok := Clamp([4]float64{0, 0, 0, 0}, w, h); ok {
		t.Error("zero-size ROI should be ok=false")
	}
	// beyond bounds clamps to the image
	r, ok := Clamp([4]float64{50, 50, 200, 200}, w, h)
	if !ok || r.Max.X != 100 || r.Max.Y != 100 {
		t.Fatalf("beyond-bounds ROI = %v ok=%v", r, ok)
	}
}

func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}
