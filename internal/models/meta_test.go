package models

import (
	"math"
	"testing"
)

// The letterbox inverse: orig = (input - pad) / scale for the corner, size / scale.
func TestPreprocessMetaAffine(t *testing.T) {
	meta := PreprocessMeta{OrigWidth: 800, OrigHeight: 600, ScaleX: 0.5, ScaleY: 0.25, PadX: 80, PadY: 20}
	got := meta.Affine().BoxToOrig([4]float64{100, 40, 50, 60})
	want := [4]float64{40, 80, 100, 240}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Fatalf("BoxToOrig = %v, want %v", got, want)
		}
	}
}
