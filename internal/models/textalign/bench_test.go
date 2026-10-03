package textalign

import (
	"os"
	"path/filepath"
	"testing"
	"visionserve/internal/vision/util"
)

// benchSetup loads the SHIPPED projection and builds a C-class head, so the numbers below
// are the real per-image cost of the head at 300 queries.
func benchSetup(b *testing.B, classes int) (*Projection, *head, []float32) {
	b.Helper()
	path := filepath.Join("..", "..", "..", "models", "rfdetr-textalign-etri", projFile)
	if _, err := os.Stat(path); err != nil {
		b.Skip("no proj.bin in models/rfdetr-textalign-etri")
	}
	pr, err := LoadProjection(path)
	if err != nil {
		b.Fatal(err)
	}
	rows := make([][]float32, classes)
	for c := range rows {
		r := make([]float32, pr.DText)
		for i := range r {
			r[i] = float32((i*7+c)%13) / 13
		}
		rows[c] = util.L2NormalizeInPlace(r)
	}
	w, err := pr.Fold(rows)
	if err != nil {
		b.Fatal(err)
	}
	names := make([]string, classes)
	feats := make([]float32, 300*pr.DFeat)
	for i := range feats {
		feats[i] = float32((i%37)-18) / 100
	}
	return pr, &head{classes: names, w: w}, feats
}

// BenchmarkLogitsExact is the cosine head: the ‖P f‖ term dominates.
func BenchmarkLogitsExact(b *testing.B) {
	pr, h, feats := benchSetup(b, 22)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := h.logits(pr, feats, 300, true); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkLogitsFolded is the plain-linear head: one [300,256]×[256,C] matmul.
func BenchmarkLogitsFolded(b *testing.B) {
	pr, h, feats := benchSetup(b, 22)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := h.logits(pr, feats, 300, false); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkFold is the per-VOCABULARY cost (cache miss only): W = a·T̂P.
func BenchmarkFold(b *testing.B) {
	pr, _, _ := benchSetup(b, 22)
	rows := make([][]float32, 22)
	for c := range rows {
		rows[c] = make([]float32, pr.DText)
		rows[c][c] = 1
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := pr.Fold(rows); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkComputeGram is the one-off cost paid when the model is LOADED.
func BenchmarkComputeGram(b *testing.B) {
	pr, _, _ := benchSetup(b, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pr.computeGram()
	}
}
