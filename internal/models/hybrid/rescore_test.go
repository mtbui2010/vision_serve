package hybrid

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"path/filepath"
	"sync/atomic"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/models/siglip"
	"visionserve/internal/pipeline"
)

// siglipDir is where the text tower's tokenizer.json lives. Model directories are not part of
// the source tree everywhere, so tests that need the real tokenizer skip rather than fail.
func siglipDir() string { return filepath.Join("..", "..", "..", "models", "siglip-text") }

// stubTowers stands in for lifecycle's Runner on the two SigLIP roles. It returns embeddings
// chosen by the test, so the whole rescoring path — crop → embed → ensemble → cosine → softmax →
// rename/reject — is exercised without loading a 400 MB ONNX graph.
//
// The text tower answers with basis vectors in CLASS-MAJOR order: every template row of class c
// is e_c, so promptens.Average collapses them to exactly e_c and a cosine against crop row i is
// simply that row's c-th component. That makes the expected confidences arithmetic, not a fit.
type stubTowers struct {
	dim       int
	cropRows  [][]float32 // one row per crop the tower is asked to embed, in order
	textCalls int64
	cropCalls int64
	lastTexts int64 // rows the text tower was asked for on its last call
}

func (s *stubTowers) Run(role string, in map[string]engine.Tensor) ([]engine.Tensor, error) {
	switch role {
	case roleText:
		t, ok := in["input_ids"]
		if !ok {
			return nil, fmt.Errorf("stub: text inputs %v carry no input_ids", in)
		}
		if len(t.Shape) != 2 {
			return nil, fmt.Errorf("stub: input_ids shape %v, want [N,L]", t.Shape)
		}
		n := int(t.Shape[0])
		atomic.AddInt64(&s.textCalls, 1)
		atomic.StoreInt64(&s.lastTexts, int64(n))
		data := make([]float32, n*s.dim)
		for i := 0; i < n; i++ {
			// Class-major: rows [c*K, (c+1)*K) all belong to class c.
			data[i*s.dim+(i/len(measuredTemplates))%s.dim] = 2 // unnormalised on purpose
		}
		return []engine.Tensor{engine.F32(data, int64(n), int64(s.dim))}, nil

	case roleCrop:
		t, ok := in["pixel_values"]
		if !ok {
			return nil, fmt.Errorf("stub: crop inputs %v carry no pixel_values", in)
		}
		n := int(t.Shape[0])
		atomic.AddInt64(&s.cropCalls, 1)
		if n > len(s.cropRows) {
			return nil, fmt.Errorf("stub: asked to embed %d crops, only %d rows prepared", n, len(s.cropRows))
		}
		data := make([]float32, 0, n*s.dim)
		for i := 0; i < n; i++ {
			data = append(data, s.cropRows[i]...)
		}
		return []engine.Tensor{engine.F32(data, int64(n), int64(s.dim))}, nil
	}
	return nil, fmt.Errorf("stub: unexpected role %q", role)
}

func (s *stubTowers) InputNames(role string) []string {
	if role == roleText {
		return []string{"input_ids"}
	}
	return []string{"pixel_values"}
}
func (s *stubTowers) OutputNames(role string) []string { return []string{"embeds"} }

// rescore runs the router's rescoring stage on one detection list, as Infer does for the open
// branch (the request's crop_temp is temp).
func (m *hybrid) rescore(img image.Image, dets []models.Detection, words []string,
	temp float64, r models.Runner) ([]models.Detection, error) {
	if m.rs == nil {
		return dets, nil
	}
	return pipeline.CropRescorer{Namer: m.rs, Prefix: "hybrid"}.Rescore(
		pipeline.Call{Img: img, Prompt: models.Prompt{CropTemp: temp}, Runner: r}, dets, words)
}

// newTestHybrid builds a router carrying only the rescorer — the detectors are never called,
// since rescore() operates on detections that already exist.
func newTestHybrid(t *testing.T) *hybrid {
	t.Helper()
	tok, err := siglip.LoadTokenizer(siglipDir())
	if err != nil {
		t.Skipf("no SigLIP tokenizer assets in %s: %v", siglipDir(), err)
	}
	return &hybrid{rs: newNamer(pipeline.SigLIPTokenizer{T: tok}, measuredTemplates)}
}

// canvas is an image big enough for the test boxes; content is irrelevant because the stub
// tower ignores pixels, but it must be real so CropTensor's clamping behaves as in production.
func canvas() image.Image {
	im := image.NewRGBA(image.Rect(0, 0, 100, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 100; x++ {
			im.Set(x, y, color.RGBA{uint8(x), uint8(y), 128, 255})
		}
	}
	return im
}

// expectedConf is the confidence rule under test, written out independently of the
// implementation: P(box) × softmax(cosines / T)[argmax].
func expectedConf(conf float64, cos []float32, k int, temp float64) float64 {
	var sum float64
	for _, c := range cos {
		sum += math.Exp(float64(c) / temp)
	}
	return conf * math.Exp(float64(cos[k])/temp) / sum
}

// The headline behaviour: the winning word renames the detection and its probability multiplies
// the confidence. Both halves matter and they were measured separately — renaming alone is worth
// +1.4 mAP, rescoring alone +8.8 (ovd-edge/docs/FINDINGS.md §2).
func TestRescoreRenamesAndReweights(t *testing.T) {
	m := newTestHybrid(t)
	words := []string{"towel", "hat"}
	// crop 0 leans towards "hat", crop 1 towards "towel".
	rows := [][]float32{{0.1, 0.4}, {0.5, 0.2}}
	st := &stubTowers{dim: 2, cropRows: rows}

	dets := []models.Detection{
		{Class: "towel", Conf: 0.9, BBox: [4]float64{10, 10, 20, 20}},
		{Class: "hat", Conf: 0.8, BBox: [4]float64{40, 40, 20, 20}},
	}
	got, err := m.rescore(canvas(), dets, words, cropTemp, st)
	if err != nil {
		t.Fatalf("rescore: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("kept %d detections, want 2: %+v", len(got), got)
	}
	if got[0].Class != "hat" || got[1].Class != "towel" {
		t.Errorf("classes = %q, %q; want hat, towel", got[0].Class, got[1].Class)
	}
	// The stub's rows are L2-normalised by DecodeImageEmbeddings before scoring, so the cosines
	// the softmax sees are the normalised components.
	for i, want := range []float64{
		expectedConf(0.9, norm2(rows[0]), 1, cropTemp),
		expectedConf(0.8, norm2(rows[1]), 0, cropTemp),
	} {
		if math.Abs(got[i].Conf-want) > 1e-6 {
			t.Errorf("detection %d conf = %.9f, want %.9f", i, got[i].Conf, want)
		}
	}
}

// The multiplier is a REJECTOR, not a general penalty, and the distinction is the finding: at
// the default T a crop with a clear winner keeps essentially all of its confidence, while a crop
// SigLIP cannot place among the requested words loses almost all of it. That asymmetry is what
// drains the 538 boxes that land on nothing, without demoting the true positives alongside them.
func TestRescoreSparesConfidentCropsAndSinksAmbiguousOnes(t *testing.T) {
	m := newTestHybrid(t)
	words := []string{"towel", "hat"}
	confident := []float32{0.05, 0.95} // one word fits
	ambiguous := []float32{0.5, 0.5}   // neither does
	st := &stubTowers{dim: 2, cropRows: [][]float32{confident, ambiguous}}

	dets := []models.Detection{
		{Class: "towel", Conf: 0.9, BBox: [4]float64{10, 10, 20, 20}},
		{Class: "towel", Conf: 0.9, BBox: [4]float64{40, 40, 20, 20}},
	}
	got, err := m.rescore(canvas(), dets, words, cropTemp, st)
	if err != nil {
		t.Fatalf("rescore: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("kept %d detections, want 2", len(got))
	}
	if got[0].Conf < 0.89 {
		t.Errorf("confident crop kept only %.4f of its 0.9 — the rescorer is penalising true positives", got[0].Conf)
	}
	if got[1].Conf > 0.5 {
		t.Errorf("ambiguous crop kept %.4f of its 0.9, want it at most halved — "+
			"rejection is where the +8.8 mAP comes from", got[1].Conf)
	}
	if got[1].Conf >= got[0].Conf {
		t.Errorf("ambiguous crop (%.4f) did not rank below the confident one (%.4f)", got[1].Conf, got[0].Conf)
	}
}

// Rejection is what the +8.8 actually buys: 538 of GroundingDINO's 849 held-out boxes land on
// nothing (FINDINGS §1). A crop that matches no requested word must not come back at all.
func TestRescoreRejectsBelowFloor(t *testing.T) {
	m := newTestHybrid(t)
	st := &stubTowers{dim: 2, cropRows: [][]float32{{-0.3, -0.4}, {0.2, 0.6}}}
	dets := []models.Detection{
		{Class: "towel", Conf: 0.9, BBox: [4]float64{10, 10, 20, 20}},
		{Class: "towel", Conf: 0.5, BBox: [4]float64{40, 40, 20, 20}},
	}
	got, err := m.rescore(canvas(), dets, []string{"towel", "hat"}, cropTemp, st)
	if err != nil {
		t.Fatalf("rescore: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("kept %d detections, want 1 (the negative-cosine crop must be dropped): %+v", len(got), got)
	}
	if got[0].Class != "hat" {
		t.Errorf("surviving detection = %q, want hat", got[0].Class)
	}
}

// A degenerate box carries no embedding, so it must be dropped — AND the boxes after it must
// keep their own rows. Losing that mapping renames a detection with another box's embedding,
// which is the failure siglip.EmbedCrops returns `kept` to prevent.
func TestRescoreDropsDegenerateBoxAndKeepsAlignment(t *testing.T) {
	m := newTestHybrid(t)
	// Only two crops are ever embedded: the middle box is zero-width.
	st := &stubTowers{dim: 2, cropRows: [][]float32{{0.1, 0.4}, {0.5, 0.2}}}
	dets := []models.Detection{
		{Class: "towel", Conf: 0.9, BBox: [4]float64{10, 10, 20, 20}},
		{Class: "towel", Conf: 0.7, BBox: [4]float64{50, 50, 0, 0}}, // degenerate
		{Class: "towel", Conf: 0.8, BBox: [4]float64{40, 40, 20, 20}},
	}
	got, err := m.rescore(canvas(), dets, []string{"towel", "hat"}, cropTemp, st)
	if err != nil {
		t.Fatalf("rescore: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("kept %d detections, want 2: %+v", len(got), got)
	}
	// Row 0 belongs to the FIRST box (conf 0.9) and row 1 to the THIRD (conf 0.8), not the second.
	if got[0].Class != "hat" || got[1].Class != "towel" {
		t.Errorf("classes = %q, %q; want hat, towel — the crop rows drifted off the detections",
			got[0].Class, got[1].Class)
	}
	if got[0].Conf > 0.9 || got[1].Conf > 0.8 {
		t.Errorf("confidences %v came from the wrong detections", []float64{got[0].Conf, got[1].Conf})
	}
}

// crop_temp is a per-request field, and this project has already published a number that was
// silently a different condition's because a per-request field never arrived (ovd-adapt
// PITFALLS §1; ovd-edge/HANDOFF.md §5.2). Prove it reaches the softmax.
func TestRescoreHonoursRequestTemperature(t *testing.T) {
	m := newTestHybrid(t)
	rows := [][]float32{{0.1, 0.4}}
	det := []models.Detection{{Class: "towel", Conf: 1.0, BBox: [4]float64{10, 10, 20, 20}}}
	words := []string{"towel", "hat"}

	confAt := func(temp float64) float64 {
		got, err := m.rescore(canvas(), det, words, temp, &stubTowers{dim: 2, cropRows: rows})
		if err != nil {
			t.Fatalf("rescore(T=%v): %v", temp, err)
		}
		if len(got) != 1 {
			t.Fatalf("rescore(T=%v) kept %d detections, want 1", temp, len(got))
		}
		return got[0].Conf
	}

	sharp, flat := confAt(0.01), confAt(1.0)
	if math.Abs(sharp-flat) < 1e-3 {
		t.Fatalf("T=0.01 and T=1.0 both gave conf %.6f — the request temperature never arrived", sharp)
	}
	// Lower temperature is more decisive, so the winner's probability must be higher.
	if sharp <= flat {
		t.Errorf("conf at T=0.01 (%.6f) should exceed conf at T=1.0 (%.6f)", sharp, flat)
	}
	// A non-positive temperature must fall back to the package default, not divide by zero.
	if zero, def := confAt(0), confAt(cropTemp); math.Abs(zero-def) > 1e-9 {
		t.Errorf("T=0 gave %.9f, want the default T=%v's %.9f", zero, cropTemp, def)
	}
}

// Without the cache the text tower runs on every request and latency scales with the number of
// unknown WORDS rather than the number of crops (~4.3 ms per word in the sibling head).
func TestVocabCachedPerWordList(t *testing.T) {
	m := newTestHybrid(t)
	st := &stubTowers{dim: 3, cropRows: [][]float32{{0.1, 0.4, 0.2}, {0.1, 0.4, 0.2}, {0.1, 0.4, 0.2}}}
	det := []models.Detection{{Class: "towel", Conf: 1.0, BBox: [4]float64{10, 10, 20, 20}}}

	if _, err := m.rescore(canvas(), det, []string{"towel", "hat"}, cropTemp, st); err != nil {
		t.Fatalf("rescore: %v", err)
	}
	if got, want := atomic.LoadInt64(&st.lastTexts), int64(2*len(measuredTemplates)); got != want {
		t.Errorf("text tower embedded %d rows, want %d (2 words × %d templates)",
			got, want, len(measuredTemplates))
	}
	if _, err := m.rescore(canvas(), det, []string{"towel", "hat"}, cropTemp, st); err != nil {
		t.Fatalf("rescore (repeat): %v", err)
	}
	if n := atomic.LoadInt64(&st.textCalls); n != 1 {
		t.Errorf("text tower ran %d times for the same word list, want 1 (cache miss)", n)
	}
	if _, err := m.rescore(canvas(), det, []string{"bread"}, cropTemp, st); err != nil {
		t.Fatalf("rescore (new words): %v", err)
	}
	if n := atomic.LoadInt64(&st.textCalls); n != 2 {
		t.Errorf("text tower ran %d times across two distinct word lists, want 2", n)
	}
}

// The feature is opt-in: a manifest declaring neither tower must produce a router that behaves
// exactly as it does today, and one declaring only half of it must fail loudly rather than
// serve a silent half-feature.
func TestNewRescorerOptIn(t *testing.T) {
	rs, err := newRescorer(models.Config{Files: map[string]string{"rfdetr": "a.onnx", "gdino": "b.onnx"}})
	if err != nil || rs != nil {
		t.Errorf("no crop/text files → rescorer %v, err %v; want nil, nil", rs, err)
	}
	for _, only := range []string{roleCrop, roleText} {
		if _, err := newRescorer(models.Config{Files: map[string]string{only: "x.onnx"}}); err == nil {
			t.Errorf("files.%s alone was accepted; want an error", only)
		}
	}
}

// Roles drives what lifecycle loads. Declaring the towers must claim their sessions; not
// declaring them must not.
func TestRolesIncludeTowersOnlyWhenPresent(t *testing.T) {
	plain := (&hybrid{rf: fakeRF{}}).Roles()
	if len(plain) != 2 {
		t.Errorf("plain router roles = %v, want just the two detectors", plain)
	}
	with := (&hybrid{rf: fakeRF{}, rs: &pipeline.CropNamer{}}).Roles()
	if !contains(with, roleCrop) || !contains(with, roleText) {
		t.Errorf("router with a rescorer has roles %v, want %q and %q", with, roleCrop, roleText)
	}
}

// A router with no rescorer, no detections, or no unknown words returns its input untouched —
// the opt-in guarantee, checked at the call site rather than assumed.
func TestRescoreNoOp(t *testing.T) {
	dets := []models.Detection{{Class: "towel", Conf: 0.9, BBox: [4]float64{10, 10, 20, 20}}}
	for _, c := range []struct {
		name  string
		m     *hybrid
		dets  []models.Detection
		words []string
	}{
		{"no rescorer", &hybrid{}, dets, []string{"towel"}},
		{"no detections", &hybrid{rs: &pipeline.CropNamer{}}, nil, []string{"towel"}},
		{"no unknown words", &hybrid{rs: &pipeline.CropNamer{}}, dets, nil},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, err := c.m.rescore(canvas(), c.dets, c.words, cropTemp, nil)
			if err != nil {
				t.Fatalf("rescore: %v", err)
			}
			if len(got) != len(c.dets) {
				t.Errorf("returned %d detections, want the input's %d", len(got), len(c.dets))
			}
		})
	}
}

func TestSoftmaxAt(t *testing.T) {
	row := []float32{0.1, 0.4, 0.2}
	var sum float64
	for _, v := range row {
		sum += math.Exp(float64(v) / cropTemp)
	}
	if got, want := float64(pipeline.SoftmaxAt(row, 1, cropTemp)), math.Exp(0.4/cropTemp)/sum; math.Abs(got-want) > 1e-6 {
		t.Errorf("softmaxAt = %.9f, want %.9f", got, want)
	}
	if got := pipeline.SoftmaxAt(row, 5, cropTemp); got != 0 {
		t.Errorf("out-of-range index gave %v, want 0", got)
	}
	if got := pipeline.SoftmaxAt([]float32{0.3}, 0, cropTemp); got != 1 {
		t.Errorf("single-word softmax gave %v, want 1", got)
	}
	// Large cosines under a small temperature must not overflow to NaN.
	if got := pipeline.SoftmaxAt([]float32{40, 1}, 0, 0.02); math.IsNaN(float64(got)) || got != 1 {
		t.Errorf("softmaxAt on a saturating row gave %v, want 1", got)
	}
}

func norm2(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	out := make([]float32, len(v))
	inv := float32(1 / math.Sqrt(sum))
	for i, x := range v {
		out[i] = x * inv
	}
	return out
}

func contains(ss []string, want string) bool {
	for _, s := range ss {
		if s == want {
			return true
		}
	}
	return false
}
