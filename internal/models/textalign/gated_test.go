package textalign

import (
	"math"
	"math/rand"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"

	_ "visionserve/internal/models/rfdetr"
)

// TestFoldPreservesArgmax is the algebraic claim gated.go rests on, exercised on random
// data: s_fold − b = ρ·(s_exact − b) with ρ = ‖P f‖ > 0, so the folded head can never
// reorder the CLASSES of a query no matter how far ρ is from 1.
//
// This is what makes it safe to name with the cheap path. It is deliberately tested with a
// projection whose ρ is nowhere near 1 — on the real dec1 head ρ ranges over [0.77, 1.47],
// which the old ‖P f‖≈1 fold-safety criterion would reject outright.
func TestFoldPreservesArgmax(t *testing.T) {
	pr := testProjection(t) // dFeat 4, dText 3
	rng := rand.New(rand.NewSource(7))

	rows := make([][]float32, 6)
	for c := range rows {
		v := make([]float32, pr.DText)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		rows[c] = l2Normalize(v)
	}
	w, err := pr.Fold(rows)
	if err != nil {
		t.Fatal(err)
	}
	h := &head{classes: make([]string, len(rows)), w: w}

	const q = 256
	feats := make([]float32, q*pr.DFeat)
	for i := range feats {
		feats[i] = float32(rng.NormFloat64() * 3) // large spread → ρ far from 1
	}

	exact, err := h.logits(pr, feats, q, true)
	if err != nil {
		t.Fatal(err)
	}
	folded, err := h.logits(pr, feats, q, false)
	if err != nil {
		t.Fatal(err)
	}

	c := len(rows)
	var checked, rhoMin, rhoMax = 0, math.Inf(1), math.Inf(-1)
	for i := 0; i < q; i++ {
		rho := float64(pr.ProjNorm(feats[i*pr.DFeat : (i+1)*pr.DFeat]))
		if rho <= 0 {
			continue
		}
		rhoMin, rhoMax = math.Min(rhoMin, rho), math.Max(rhoMax, rho)
		be, bf := 0, 0
		for k := 1; k < c; k++ {
			if exact[i*c+k] > exact[i*c+be] {
				be = k
			}
			if folded[i*c+k] > folded[i*c+bf] {
				bf = k
			}
		}
		if be != bf {
			t.Fatalf("query %d: folded argmax %d != exact argmax %d (ρ=%.4f) — the fold identity is broken",
				i, bf, be, rho)
		}
		// and the identity itself, not just its consequence
		for k := 0; k < c; k++ {
			want := rho * float64(exact[i*c+k]-pr.Bias)
			got := float64(folded[i*c+k] - pr.Bias)
			if math.Abs(got-want) > 1e-4*math.Max(1, math.Abs(want)) {
				t.Fatalf("query %d class %d: folded−b = %.6f, want ρ·(exact−b) = %.6f", i, k, got, want)
			}
		}
		checked++
	}
	if checked < q/2 {
		t.Fatalf("only %d/%d queries exercised", checked, q)
	}
	if rhoMax/rhoMin < 2 {
		t.Fatalf("ρ spread too small (%.3f..%.3f) to be a meaningful test", rhoMin, rhoMax)
	}
}

// TestRealClassColsFindsBackgroundByName pins the trap the COCO-91 and 22-class exports set:
// the "N/A" column is at index 0 in one and at the END in the other, so it must be located by
// name. Reading it as a real class silently corrupts every objectness score.
func TestRealClassColsFindsBackgroundByName(t *testing.T) {
	coco91 := realClassCols([]string{"N/A", "person", "bicycle"})
	if coco91[0] || !coco91[1] || !coco91[2] {
		t.Errorf("COCO-91 layout: got %v, want [false true true]", coco91)
	}
	etri := realClassCols([]string{"remote", "coffee", "n/a"})
	if !etri[0] || !etri[1] || etri[2] {
		t.Errorf("ETRI layout: got %v, want [true true false] (match must be case-insensitive)", etri)
	}
	if got := realClassCols([]string{"cup", "hat"}); !got[0] || !got[1] {
		t.Errorf("an export with no background column must yield all-true, got %v", got)
	}
}

func TestObjectnessSkipsBackground(t *testing.T) {
	// 2 queries × 3 classes, background last and deliberately the largest value.
	cls := engine.F32([]float32{
		1.0, -2.0, 9.0,
		-1.0, 0.5, 9.0,
	}, 1, 2, 3)
	real := realClassCols([]string{"remote", "coffee", "N/A"})
	obj, err := objectness(cls, real, 2)
	if err != nil {
		t.Fatal(err)
	}
	if obj[0] != 1.0 || obj[1] != 0.5 {
		t.Errorf("objectness = %v, want [1 0.5] — the 9.0 background column must be ignored", obj)
	}
	// A labels file that does not match the export must be an error, not a wrong answer.
	if _, err := objectness(cls, realClassCols([]string{"a", "b"}), 2); err == nil {
		t.Error("expected an error when the label count does not match the class head width")
	}
}

func TestGatedLogitsNamesAndScores(t *testing.T) {
	// 2 queries, 3 classes. Head B prefers class 2 then class 0; objectness is unrelated to
	// head B's magnitudes, which is the whole point of the split.
	names := []float32{
		0.1, 0.2, 5.0,
		7.0, 1.0, 2.0,
	}
	obj := []float32{-1.5, 3.0}
	out, err := gatedLogits(names, obj, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{
		gatedLogitFloor, gatedLogitFloor, -1.5,
		3.0, gatedLogitFloor, gatedLogitFloor,
	}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("gatedLogits = %v, want %v", out, want)
		}
	}
	if _, err := gatedLogits(names, obj, 2, 0); err == nil {
		t.Error("expected an error for an empty vocabulary")
	}
}

// TestDecodeGatedSelectsByObjectness is the end-to-end statement: two queries whose head B
// scores rank them one way and whose detector objectness ranks them the other. The gated
// path must take the NAME from head B and the SCORE (hence the ordering and the threshold)
// from the detector.
func TestDecodeGatedSelectsByObjectness(t *testing.T) {
	pr := testProjection(t)
	classes := []string{"cup", "hat"}
	n := float32(math.Sqrt(14))
	zhat := []float32{1 / n, 2 / n, 3 / n}
	w, err := pr.Fold([][]float32{zhat, {-zhat[0], -zhat[1], -zhat[2]}})
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
	m := &textAlign{cfg: cfg, proj: pr}
	h := &head{classes: classes, w: w, rf: rf}

	// Both queries have the same feature, so head B names both "cup" with the same score.
	f := []float32{1, 1, 2, 2}
	feats := engine.F32(append(append([]float32{}, f...), f...), 1, 2, 4)
	boxes := engine.F32([]float32{
		0.5, 0.5, 0.25, 0.25,
		0.25, 0.25, 0.125, 0.125,
	}, 1, 2, 4)
	// Detector objectness: query 1 is confident (+3), query 0 is not (−3 → sigmoid 0.047,
	// below conf 0.5). The background column is large in both and must be ignored.
	cls := engine.F32([]float32{
		-3.0, 8.0,
		3.0, 8.0,
	}, 1, 2, 2)
	meta := models.PreprocessMeta{OrigWidth: 640, OrigHeight: 480, ScaleX: 0.8, ScaleY: 0.8, PadY: 64}

	res, err := m.decodeGated(h, boxes, cls, feats, meta)
	if err != nil {
		t.Fatalf("decodeGated: %v", err)
	}
	if len(res.Detections) != 1 {
		t.Fatalf("got %d detections, want 1 — selection must follow the detector, not head B: %+v",
			len(res.Detections), res.Detections)
	}
	d := res.Detections[0]
	if d.Class != "cup" {
		t.Errorf("class = %q, want \"cup\" (the name still comes from head B)", d.Class)
	}
	if want := 1 / (1 + math.Exp(-3.0)); math.Abs(d.Conf-want) > 1e-4 {
		t.Errorf("conf = %v, want sigmoid(objectness) = %v", d.Conf, want)
	}
	// The surviving query is the SECOND one: its box, not the first's.
	// cxcywh (0.25,0.25,0.125,0.125) of 512² → centre (128,128), 64×64 → corner (96,96);
	// then un-letterbox: (96−0)/0.8, (96−64)/0.8, 64/0.8, 64/0.8.
	wantBox := [4]float64{120, 40, 80, 80}
	for i := range wantBox {
		if math.Abs(d.BBox[i]-wantBox[i]) > 1e-6 {
			t.Fatalf("BBox = %v, want %v (the high-objectness query's box)", d.BBox, wantBox)
		}
	}
}

func TestClassLogitsRejectsMismatchedLabels(t *testing.T) {
	const q, dFeat = 4, 8
	dets := engine.F32(make([]float32, q*4), 1, q, 4)
	labels := engine.F32(make([]float32, q*23), 1, q, 23)
	qf := engine.F32(make([]float32, q*dFeat), 1, q, dFeat)
	outs := []engine.Tensor{dets, labels, qf}

	got, err := classLogits(outs, dets, dFeat, 23)
	if err != nil {
		t.Fatalf("classLogits: %v", err)
	}
	if got.Dim(-1) != 23 {
		t.Errorf("picked shape %v, want the [1,Q,23] class head", got.Shape)
	}
	// 91 labels against a 23-wide head is the manifest/export mismatch that must be caught.
	if _, err := classLogits(outs, dets, dFeat, 91); err == nil {
		t.Error("expected an error when the label count does not match any output")
	}
	// A stripped export (no class head) must say so rather than fall back silently.
	if _, err := classLogits([]engine.Tensor{dets, qf}, dets, dFeat, 23); err == nil {
		t.Error("expected an error when the export has no class head")
	}
}
