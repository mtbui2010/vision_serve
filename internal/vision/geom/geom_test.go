package geom

import (
	"math"
	"testing"
)

func near(a, b float64) bool { return math.Abs(a-b) < 1e-12 }

func eqBox(a, b [4]float64) bool {
	for i := range a {
		if !near(a[i], b[i]) {
			return false
		}
	}
	return true
}

func TestSigmoid(t *testing.T) {
	if Sigmoid(0) != 0.5 {
		t.Fatalf("Sigmoid(0) = %v", Sigmoid(0))
	}
	if s := Sigmoid(5); math.Abs(s-0.9933071490757153) > 1e-15 {
		t.Fatalf("Sigmoid(5) = %v", s)
	}
	if s := Sigmoid(-800); s != 0 {
		t.Fatalf("Sigmoid(-800) = %v, want 0 (exp overflow to +Inf)", s)
	}
}

func TestNormToInput(t *testing.T) {
	// cxcywh (0.5,0.5,0.5,0.5) on 100×50 -> centre (50,25), size (50,25) -> x=25,y=12.5.
	if got := NormToInput(0.5, 0.5, 0.5, 0.5, FormatCXCYWH, 100, 50); !eqBox(got, [4]float64{25, 12.5, 50, 25}) {
		t.Fatalf("cxcywh: %v", got)
	}
	if got := NormToInput(0.5, 0.5, 0.5, 0.5, "", 100, 50); !eqBox(got, [4]float64{25, 12.5, 50, 25}) {
		t.Fatalf("default format must be cxcywh: %v", got)
	}
	if got := NormToInput(0.1, 0.2, 0.6, 1, FormatXYXY, 100, 50); !eqBox(got, [4]float64{10, 10, 50, 40}) {
		t.Fatalf("xyxy: %v", got)
	}
}

func TestAffineBoxToOrig(t *testing.T) {
	// letterbox scale 0.5, pad (0,25): input (25,25,50,50) -> orig (50,0,100,100).
	a := Affine{ScaleX: 0.5, ScaleY: 0.5, PadY: 25}
	if got := a.BoxToOrig([4]float64{25, 25, 50, 50}); !eqBox(got, [4]float64{50, 0, 100, 100}) {
		t.Fatalf("letterbox: %v", got)
	}
	// squash: per-axis scale.
	a = Affine{ScaleX: 0.16, ScaleY: 0.64}
	if got := a.BoxToOrig([4]float64{0, 0, 64, 64}); !eqBox(got, [4]float64{0, 0, 400, 100}) {
		t.Fatalf("squash: %v", got)
	}
}

func TestClamp(t *testing.T) {
	cases := []struct{ in, want [4]float64 }{
		{[4]float64{10, 10, 20, 20}, [4]float64{10, 10, 20, 20}},         // inside
		{[4]float64{-5, -10, 20, 20}, [4]float64{0, 0, 15, 10}},          // top-left overflow shrinks
		{[4]float64{90, 40, 20, 20}, [4]float64{90, 40, 10, 10}},         // bottom-right cut
		{[4]float64{-50, 10, 20, 5}, [4]float64{0, 10, 0, 5}},            // fully left -> zero width
		{[4]float64{120, 70, 5, 5}, [4]float64{120, 70, 0, 0}},           // fully outside -> zero size
		{[4]float64{-10, -10, 200, 200}, [4]float64{0, 0, 100, 50}},      // covers the image
		{[4]float64{0, 0, 0, 0}, [4]float64{0, 0, 0, 0}},                 // degenerate
		{[4]float64{99.5, 49.5, 1, 1}, [4]float64{99.5, 49.5, 0.5, 0.5}}, // sub-pixel
	}
	for _, c := range cases {
		if got := Clamp(c.in, 100, 50); !eqBox(got, c.want) {
			t.Errorf("Clamp(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestIoU(t *testing.T) {
	if v := IoU([4]float64{0, 0, 10, 10}, [4]float64{20, 20, 10, 10}); v != 0 {
		t.Fatalf("disjoint: %v", v)
	}
	if v := IoU([4]float64{0, 0, 10, 10}, [4]float64{10, 0, 10, 10}); v != 0 {
		t.Fatalf("touching: %v", v)
	}
	if v := IoU([4]float64{0, 0, 10, 10}, [4]float64{0, 0, 10, 10}); v != 1 {
		t.Fatalf("identical: %v", v)
	}
	if v := IoU([4]float64{0, 0, 10, 10}, [4]float64{5, 0, 10, 10}); !near(v, 50.0/150) {
		t.Fatalf("half overlap: %v", v)
	}
}
