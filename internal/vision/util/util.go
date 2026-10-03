// Package util holds the few tiny helpers every model package used to copy: picking a session's
// input name, describing tensor shapes in an error, and L2-normalising an embedding row.
//
// They are deliberately trivial. They live in one place because each copy was free to drift
// (a normaliser that divides in float32 instead of float64 changes every cosine downstream), not
// because any of them is hard.
package util

import (
	"math"

	"visionserve/internal/engine"
)

// FirstName returns names[0], or fallback when the session declares no name (an export probed
// without I/O names). It is how a model binds its single input without hard-coding the export's
// naming.
func FirstName(names []string, fallback string) string {
	if len(names) > 0 {
		return names[0]
	}
	return fallback
}

// ShapesOf lists the shapes of ts, for error messages that must say what a session returned.
func ShapesOf(ts []engine.Tensor) [][]int64 {
	out := make([][]int64, len(ts))
	for i, t := range ts {
		out[i] = t.Shape
	}
	return out
}

// L2NormalizeInPlace scales v to unit length in place and returns it. A zero vector is returned
// unchanged. The norm is accumulated and inverted in float64 and applied as one float32 factor —
// the arithmetic every text/image tower in this repository was measured with.
func L2NormalizeInPlace(v []float32) []float32 {
	inv, ok := invNorm(v)
	if !ok {
		return v
	}
	for i := range v {
		v[i] *= inv
	}
	return v
}

// L2Normalized returns a unit-length COPY of v (a copy of v itself when it is a zero vector),
// leaving v untouched — for rows that alias a session's output buffer. Bit-identical to
// L2NormalizeInPlace on a copy.
func L2Normalized(v []float32) []float32 {
	out := make([]float32, len(v))
	copy(out, v)
	return L2NormalizeInPlace(out)
}

func invNorm(v []float32) (float32, bool) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return 0, false
	}
	return float32(1 / math.Sqrt(sum)), true
}
