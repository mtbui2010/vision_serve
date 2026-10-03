package textalign

import (
	"math"
	"reflect"
	"testing"
)

// reference computes the head exactly as the design note specifies:
// logit = a·⟨t̂_c, P f/‖P f‖⟩ + b, in float64, without any of the folding tricks.
func reference(pr *Projection, tRows [][]float32, f []float32) []float64 {
	z := make([]float64, pr.DText)
	var n float64
	for k := 0; k < pr.DText; k++ {
		var s float64
		for j, fv := range f {
			s += float64(pr.P[k*pr.DFeat+j]) * float64(fv)
		}
		z[k] = s
		n += s * s
	}
	n = math.Sqrt(n)
	out := make([]float64, len(tRows))
	for c, tr := range tRows {
		var dot float64
		for k, tv := range tr {
			dot += float64(tv) * z[k] / n
		}
		out[c] = float64(pr.Scale)*dot + float64(pr.Bias)
	}
	return out
}

func TestLogitsMatchCosineReference(t *testing.T) {
	pr := testProjection(t)
	tRows := [][]float32{{1, 0, 0}, {0.6, 0.8, 0}, {0, -0.6, 0.8}}
	w, err := pr.Fold(tRows)
	if err != nil {
		t.Fatal(err)
	}
	h := &head{classes: []string{"a", "b", "c"}, w: w}

	feats := []float32{1, 1, 2, 2, 0.5, -1, 3, 0.25} // 2 queries × dFeat 4
	got, err := h.logits(pr, feats, 2, true)
	if err != nil {
		t.Fatalf("logits: %v", err)
	}
	for i := 0; i < 2; i++ {
		want := reference(pr, tRows, feats[i*4:(i+1)*4])
		for c := range want {
			if math.Abs(float64(got[i*3+c])-want[c]) > 1e-5 {
				t.Errorf("logit[q%d][c%d] = %v, want %v", i, c, got[i*3+c], want[c])
			}
		}
	}
}

// TestFoldedIsExactUpToTheNormScalar is the numerical claim the deployment rests on:
// dropping ‖P f‖ multiplies (logit − b) by that per-query positive scalar, so the class
// ORDER is preserved and the two agree exactly when ‖P f‖ = 1.
func TestFoldedIsExactUpToTheNormScalar(t *testing.T) {
	pr := testProjection(t)
	tRows := [][]float32{{1, 0, 0}, {0.6, 0.8, 0}, {0, -0.6, 0.8}}
	w, _ := pr.Fold(tRows)
	h := &head{classes: []string{"a", "b", "c"}, w: w}

	f := []float32{1, 1, 2, 2}
	exact, err := h.logits(pr, f, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	folded, err := h.logits(pr, f, 1, false)
	if err != nil {
		t.Fatal(err)
	}
	n := float64(pr.ProjNorm(f))
	for c := range exact {
		want := (float64(exact[c]) - float64(pr.Bias)) * n
		if math.Abs(float64(folded[c])-float64(pr.Bias)-want) > 1e-5 {
			t.Errorf("folded[%d] = %v, want %v", c, folded[c], want+float64(pr.Bias))
		}
	}
	if argmax(exact) != argmax(folded) {
		t.Errorf("argmax changed between exact (%d) and folded (%d)", argmax(exact), argmax(folded))
	}
}

func argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

func TestLogitsRejectsShortFeatures(t *testing.T) {
	pr := testProjection(t)
	w, _ := pr.Fold([][]float32{{1, 0, 0}})
	h := &head{classes: []string{"a"}, w: w}
	if _, err := h.logits(pr, []float32{1, 2, 3}, 1, true); err == nil {
		t.Errorf("expected an error when query_feats is shorter than queries×d_feat")
	}
	if _, err := h.logits(pr, []float32{1, 2, 3, 4}, 0, true); err == nil {
		t.Errorf("expected an error for zero queries")
	}
}

// A zero feature makes ‖P f‖ = 0 (cosine undefined). It must fall back to the bias
// instead of dividing by zero (no NaN reaching the sigmoid, no panic).
func TestLogitsHandlesZeroNorm(t *testing.T) {
	pr := testProjection(t)
	w, _ := pr.Fold([][]float32{{1, 0, 0}, {0, 1, 0}})
	h := &head{classes: []string{"a", "b"}, w: w}
	got, err := h.logits(pr, []float32{0, 0, 0, 0}, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	for c, v := range got {
		if v != pr.Bias {
			t.Errorf("logit[%d] = %v, want the bias %v", c, v, pr.Bias)
		}
	}
}

func TestNormalizeVocab(t *testing.T) {
	got := normalizeVocab([]string{" cup ", "", "Cup", "cola can", "  "})
	want := []string{"cup", "cola can"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("normalizeVocab = %v, want %v", got, want)
	}
}
