package util

import (
	"math"
	"reflect"
	"testing"

	"visionserve/internal/engine"
)

func TestFirstName(t *testing.T) {
	if got := FirstName([]string{"input", "other"}, "fb"); got != "input" {
		t.Errorf("FirstName = %q, want input", got)
	}
	if got := FirstName(nil, "fb"); got != "fb" {
		t.Errorf("FirstName(nil) = %q, want the fallback", got)
	}
}

func TestShapesOf(t *testing.T) {
	ts := []engine.Tensor{engine.F32(nil, 1, 300, 4), engine.F32(nil, 2)}
	if got, want := ShapesOf(ts), [][]int64{{1, 300, 4}, {2}}; !reflect.DeepEqual(got, want) {
		t.Errorf("ShapesOf = %v, want %v", got, want)
	}
}

// The normaliser every tower was measured with: norm accumulated in float64, ONE float32
// factor 1/‖v‖ applied to every element. A float32 accumulation or a per-element division
// moves the last bits of every cosine.
func TestL2NormalizeArithmetic(t *testing.T) {
	v := []float32{0.1, -0.7, 2.5, 3e-4, 1}
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	inv := float32(1 / math.Sqrt(s))
	want := make([]float32, len(v))
	for i, x := range v {
		want[i] = x * inv
	}

	cp := L2Normalized(v)
	if !reflect.DeepEqual(cp, want) {
		t.Errorf("L2Normalized = %v, want %v", cp, want)
	}
	if v[0] != 0.1 {
		t.Error("L2Normalized modified its input")
	}
	in := append([]float32(nil), v...)
	if got := L2NormalizeInPlace(in); !reflect.DeepEqual(got, want) || &got[0] != &in[0] {
		t.Errorf("L2NormalizeInPlace = %v, want %v in place", got, want)
	}
}

func TestL2NormalizeZeroVector(t *testing.T) {
	z := []float32{0, 0, 0}
	if got := L2NormalizeInPlace(z); !reflect.DeepEqual(got, []float32{0, 0, 0}) {
		t.Errorf("zero vector became %v", got)
	}
	if got := L2Normalized(z); !reflect.DeepEqual(got, []float32{0, 0, 0}) || &got[0] == &z[0] {
		t.Errorf("L2Normalized(zero) = %v, want a zero COPY", got)
	}
}
