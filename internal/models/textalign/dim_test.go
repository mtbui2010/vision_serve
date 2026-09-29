package textalign

import (
	"math"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"

	_ "visionserve/internal/models/rfdetr"
)

// TestProjectionIsDimensionAgnostic pins the property the SigLIP work depends on: the text
// dimension is a FILE FIELD, not a constant. A 768-d teacher (SigLIP) must load and fold exactly
// like the 512-d one (CLIP ViT-B/32) with no code change.
//
// Everything outside this package still says "512" in prose because that is what ships today;
// this test is what makes those comments descriptive rather than load-bearing.
func TestProjectionIsDimensionAgnostic(t *testing.T) {
	for _, dText := range []int{512, 768, 1152} { // ViT-B/32, SigLIP base, SigLIP so400m
		flat := make([]float32, dText*256)
		for k := 0; k < dText; k++ {
			for j := 0; j < 256; j++ {
				flat[k*256+j] = float32((k*7+j*3)%11) / 11.0
			}
		}
		raw := buildProj(dText, 256, 2.0, -0.5, 0, flat)
		p, err := ParseProjection(raw)
		if err != nil {
			t.Fatalf("d_text %d: %v", dText, err)
		}
		if p.DText != dText || p.DFeat != 256 {
			t.Fatalf("d_text %d: parsed %dx%d", dText, p.DText, p.DFeat)
		}

		// Fold must accept text rows of exactly that width, and reject any other width.
		rows := [][]float32{make([]float32, dText), make([]float32, dText)}
		for i := range rows[0] {
			rows[0][i] = 1 / float32(math.Sqrt(float64(dText)))
			rows[1][i] = -1 / float32(math.Sqrt(float64(dText)))
		}
		w, err := p.Fold(rows)
		if err != nil {
			t.Fatalf("d_text %d: Fold: %v", dText, err)
		}
		if len(w) != 2*p.DFeat {
			t.Fatalf("d_text %d: folded W has %d values, want %d", dText, len(w), 2*p.DFeat)
		}
		if _, err := p.Fold([][]float32{make([]float32, dText+1)}); err == nil {
			t.Errorf("d_text %d: Fold accepted a %d-wide text row", dText, dText+1)
		}

		// ProjNorm uses the precomputed Gram matrix; it must agree with ‖P f‖ computed directly.
		f := make([]float32, p.DFeat)
		for i := range f {
			f[i] = float32(i%5) - 2
		}
		var want float64
		for k := 0; k < p.DText; k++ {
			var s float64
			for j, fv := range f {
				s += float64(p.P[k*p.DFeat+j]) * float64(fv)
			}
			want += s * s
		}
		want = math.Sqrt(want)
		if got := float64(p.ProjNorm(f)); math.Abs(got-want) > 1e-3*math.Max(1, want) {
			t.Errorf("d_text %d: ProjNorm = %v, direct ‖Pf‖ = %v", dText, got, want)
		}
	}
}

// TestGatedIsDimensionAgnostic drives the shipped decode path with a 768-d projection, so the
// "score from the detector, name from head B" split is exercised at SigLIP's width and not only
// at CLIP's.
func TestGatedIsDimensionAgnostic(t *testing.T) {
	const dText, dFeat = 768, 4
	// P maps f to a vector whose first two coordinates carry the signal; the rest are zero, so
	// the arithmetic stays checkable by hand at any width.
	flat := make([]float32, dText*dFeat)
	flat[0*dFeat+0] = 1
	flat[1*dFeat+1] = 1
	raw := buildProj(dText, dFeat, 3.0, -0.25, 0, flat)
	p, err := ParseProjection(raw)
	if err != nil {
		t.Fatal(err)
	}
	classes := []string{"cup", "hat"}
	rows := make([][]float32, 2)
	for i := range rows {
		rows[i] = make([]float32, dText)
	}
	rows[0][0], rows[1][1] = 1, 1 // "cup" reads coordinate 0, "hat" reads coordinate 1
	w, err := p.Fold(rows)
	if err != nil {
		t.Fatal(err)
	}

	cfg := models.Config{
		Name: "test", Width: 512, Height: 512, Layout: "NCHW",
		PostType: "detr", BoxFormat: "cxcywh", ConfThresh: 0.5, MaxDet: 300,
		Labels: []string{"remote", "N/A"},
	}
	rf, err := newRFDETR(cfg, classes)
	if err != nil {
		t.Fatal(err)
	}
	m := &textAlign{cfg: cfg, proj: p}
	h := &head{classes: classes, w: w, rf: rf}

	// One query whose feature points along coordinate 1 → head B must name it "hat"; the
	// detector's own logit (+3) supplies the score.
	feats := engine.F32([]float32{0, 5, 0, 0}, 1, 1, dFeat)
	boxes := engine.F32([]float32{0.5, 0.5, 0.25, 0.25}, 1, 1, 4)
	cls := engine.F32([]float32{3.0, 9.0}, 1, 1, 2) // second column is "N/A" and must be ignored
	meta := models.PreprocessMeta{OrigWidth: 640, OrigHeight: 480, ScaleX: 0.8, ScaleY: 0.8}

	res, err := m.decodeGated(h, boxes, cls, feats, meta)
	if err != nil {
		t.Fatalf("decodeGated at d_text=%d: %v", dText, err)
	}
	if len(res.Detections) != 1 {
		t.Fatalf("got %d detections, want 1: %+v", len(res.Detections), res.Detections)
	}
	if res.Detections[0].Class != "hat" {
		t.Errorf("class = %q, want \"hat\"", res.Detections[0].Class)
	}
	if want := 1 / (1 + math.Exp(-3.0)); math.Abs(res.Detections[0].Conf-want) > 1e-4 {
		t.Errorf("conf = %v, want sigmoid(detector logit) = %v", res.Detections[0].Conf, want)
	}
}
