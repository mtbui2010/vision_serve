package clip

import (
	"math"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// TestPostprocessL2Normalization verifies that a non-unit-length vector is L2-normalised.
func TestPostprocessL2Normalization(t *testing.T) {
	raw := []float32{3.0, 0.0, 4.0} // ||v|| = 5
	t1 := engine.F32(raw, 1, 3)

	result, err := postprocess([]engine.Tensor{t1})
	if err != nil {
		t.Fatalf("postprocess error: %v", err)
	}

	if result.Task != models.TaskEmbed {
		t.Fatalf("task = %q, want %q", result.Task, models.TaskEmbed)
	}
	if len(result.Embeddings) != 1 {
		t.Fatalf("len(Embeddings) = %d, want 1", len(result.Embeddings))
	}
	vec := result.Embeddings[0]
	if len(vec) != 3 {
		t.Fatalf("embedding length = %d, want 3", len(vec))
	}

	// Expected: [3/5, 0/5, 4/5] = [0.6, 0, 0.8]
	const tol = 1e-6
	if math.Abs(float64(vec[0])-0.6) > tol {
		t.Fatalf("vec[0] = %v, want 0.6", vec[0])
	}
	if math.Abs(float64(vec[1])-0.0) > tol {
		t.Fatalf("vec[1] = %v, want 0.0", vec[1])
	}
	if math.Abs(float64(vec[2])-0.8) > tol {
		t.Fatalf("vec[2] = %v, want 0.8", vec[2])
	}
}

// TestPostprocessAlreadyUnit verifies that an already unit-norm vector is preserved.
func TestPostprocessAlreadyUnit(t *testing.T) {
	// [1/sqrt(2), 1/sqrt(2)] is already unit-norm.
	v := float32(1.0 / math.Sqrt2)
	raw := []float32{v, v}
	t1 := engine.F32(raw, 1, 2)

	result, err := postprocess([]engine.Tensor{t1})
	if err != nil {
		t.Fatalf("postprocess error: %v", err)
	}

	vec := result.Embeddings[0]
	if len(vec) != 2 {
		t.Fatalf("embedding length = %d, want 2", len(vec))
	}

	// After normalization the result should still be approximately [v, v].
	const tol = 1e-5
	if math.Abs(float64(vec[0])-float64(v)) > tol {
		t.Fatalf("vec[0] = %v, want %v", vec[0], v)
	}
	if math.Abs(float64(vec[1])-float64(v)) > tol {
		t.Fatalf("vec[1] = %v, want %v", vec[1], v)
	}
}

// TestPostprocessUnitNormResult verifies that the output vector always has ||v|| = 1.
func TestPostprocessUnitNormResult(t *testing.T) {
	// Simulate a realistic 512-d embedding with arbitrary values.
	dim := 512
	raw := make([]float32, dim)
	for i := range raw {
		raw[i] = float32(i+1) * 0.001
	}
	t1 := engine.F32(raw, 1, int64(dim))

	result, err := postprocess([]engine.Tensor{t1})
	if err != nil {
		t.Fatalf("postprocess error: %v", err)
	}

	vec := result.Embeddings[0]
	var sumSq float64
	for _, v := range vec {
		sumSq += float64(v) * float64(v)
	}
	norm := math.Sqrt(sumSq)
	if math.Abs(norm-1.0) > 1e-5 {
		t.Fatalf("||v|| = %v, want 1.0", norm)
	}
}

// TestPostprocess1DShape verifies that a [D] (rank-1) tensor is accepted in addition
// to the standard [1, D] (rank-2) shape.
func TestPostprocess1DShape(t *testing.T) {
	raw := []float32{0.0, 1.0}
	// rank-1: shape [2]
	t1 := engine.F32(raw, 2)

	result, err := postprocess([]engine.Tensor{t1})
	if err != nil {
		t.Fatalf("postprocess error: %v", err)
	}
	if len(result.Embeddings[0]) != 2 {
		t.Fatalf("embedding length = %d, want 2", len(result.Embeddings[0]))
	}
}

// TestPostprocessZeroVector verifies that a zero vector does not produce NaN.
func TestPostprocessZeroVector(t *testing.T) {
	raw := []float32{0.0, 0.0, 0.0}
	t1 := engine.F32(raw, 1, 3)

	result, err := postprocess([]engine.Tensor{t1})
	if err != nil {
		t.Fatalf("postprocess error: %v", err)
	}
	for i, v := range result.Embeddings[0] {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("vec[%d] = %v (NaN/Inf) for zero vector", i, v)
		}
	}
}

// TestPostprocessNoOutputsError verifies that an empty output slice returns an error.
func TestPostprocessNoOutputsError(t *testing.T) {
	_, err := postprocess([]engine.Tensor{})
	if err == nil {
		t.Fatal("expected error for empty output slice, got nil")
	}
}

// TestPostprocessBatchSizeError verifies that batch size != 1 returns an error.
func TestPostprocessBatchSizeError(t *testing.T) {
	raw := make([]float32, 2*512)
	t1 := engine.F32(raw, 2, 512) // batch=2
	_, err := postprocess([]engine.Tensor{t1})
	if err == nil {
		t.Fatal("expected error for batch size 2, got nil")
	}
}
