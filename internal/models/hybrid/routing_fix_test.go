package hybrid

import (
	"fmt"
	"image"
	"sync"
	"testing"
	"time"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/models/groundingdino"
	"visionserve/internal/pipeline"
)

// wordTowers is stubTowers with a text tower that, like the real one, embeds identical text
// identically: every distinct input_ids row gets its own basis vector, in first-seen order.
type wordTowers struct {
	stubTowers
	seen map[string]int
}

func (w *wordTowers) Run(role string, in map[string]engine.Tensor) ([]engine.Tensor, error) {
	if role != roleText {
		return w.stubTowers.Run(role, in)
	}
	t := in["input_ids"]
	n, l := int(t.Shape[0]), int(t.Shape[1])
	if w.seen == nil {
		w.seen = map[string]int{}
	}
	data := make([]float32, n*w.dim)
	for i := 0; i < n; i++ {
		key := fmt.Sprint(t.DataI64[i*l : (i+1)*l])
		idx, ok := w.seen[key]
		if !ok {
			idx = len(w.seen)
			w.seen[key] = idx
		}
		data[i*w.dim+idx%w.dim] = 1
	}
	return []engine.Tensor{engine.F32(data, int64(n), int64(w.dim))}, nil
}

// The consequence B2 caused, on the rescorer: with the word list a duplicated prompt used to
// produce, a crop that matches "zebra" perfectly came back at HALF its confidence.
func TestRescoreConfidenceNotDividedByDuplicateWords(t *testing.T) {
	m := newTestHybrid(t)
	// Templates are the same for every word, so the tower sees len(templates) rows per word;
	// a 64-wide space keeps each distinct (word, template) row on its own axis.
	st := &wordTowers{stubTowers: stubTowers{dim: 64}}
	crop := make([]float32, 64)
	for k := 0; k < len(measuredTemplates); k++ {
		crop[k] = 1 // the template-average of "zebra"'s rows
	}
	st.cropRows = [][]float32{crop}
	dets := []models.Detection{{Class: "zebra", Conf: 0.8, BBox: [4]float64{10, 10, 20, 20}}}

	dup, err := m.rescore(canvas(), dets, []string{"zebra", "zebra"}, cropTemp, st)
	if err != nil {
		t.Fatal(err)
	}
	if len(dup) != 1 || dup[0].Conf > 0.41 {
		t.Fatalf("setup: duplicated words gave %+v, want the halved 0.4 this test guards against", dup)
	}
	got, err := m.rescore(canvas(), dets, pipeline.ParseClasses("zebra. Zebra."), cropTemp, st)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Conf < 0.8-1e-6 {
		t.Fatalf("rescored %+v, want one detection keeping conf 0.8 (a single word is certain)", got)
	}
}

// fakeRF is a plain RF-DETR stand-in: one fixed "cup" detection, no session of its own.
type fakeRF struct{}

func (fakeRF) Name() string          { return "fake-rf" }
func (fakeRF) Task() models.Task     { return models.TaskDetection }
func (fakeRF) InputName() string     { return "images" }
func (fakeRF) OutputNames() []string { return nil }
func (fakeRF) Preprocess(image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	return engine.F32([]float32{0}, 1), models.PreprocessMeta{}, nil
}
func (fakeRF) Postprocess([]engine.Tensor, models.PreprocessMeta) (models.Result, error) {
	return models.Result{Detections: []models.Detection{{Class: "cup", Conf: 0.9, BBox: [4]float64{1, 1, 4, 4}}}}, nil
}

// lockProbeRunner answers the rfdetr role, and records whether lock (the model's GroundingDINO
// lock) was held while the gdino role ran.
type lockProbeRunner struct {
	lock        *sync.Mutex
	mu          sync.Mutex
	gdinoLocked []bool
}

func (r *lockProbeRunner) Run(role string, in map[string]engine.Tensor) ([]engine.Tensor, error) {
	switch role {
	case roleRFDETR:
		return []engine.Tensor{engine.F32(make([]float32, 4), 1, 1, 4), engine.F32(make([]float32, 1), 1, 1, 1)}, nil
	case roleGDINO:
		held := !r.lock.TryLock()
		if !held {
			r.lock.Unlock()
		}
		r.mu.Lock()
		r.gdinoLocked = append(r.gdinoLocked, held)
		r.mu.Unlock()
		const dim = 256
		lg := make([]float32, dim)
		for i := range lg {
			lg[i] = -6
		}
		return []engine.Tensor{engine.F32(lg, 1, 1, dim), engine.F32([]float32{0.5, 0.5, 0.2, 0.2}, 1, 1, 4)}, nil
	}
	return nil, fmt.Errorf("unexpected role %q", role)
}
func (r *lockProbeRunner) InputNames(string) []string { return []string{"x"} }
func (r *lockProbeRunner) OutputNames(role string) []string {
	if role == roleGDINO {
		return []string{"logits", "pred_boxes"}
	}
	return []string{"dets"}
}

func newLockTestHybrid(t *testing.T) *hybrid {
	t.Helper()
	tok := gdinoTokenizer(t)
	m := &hybrid{cfg: models.Config{Name: "rfdetr-gdino", Labels: []string{"cup"}}, rf: fakeRF{}}
	m.wire(&pipeline.GDINO{Role: roleGDINO, Tok: tok})
	return m
}

func gdinoTokenizer(t *testing.T) *groundingdino.Tokenizer {
	t.Helper()
	tok, err := groundingdino.LoadTokenizer("../../../models/grounding-dino/vocab.txt")
	if err != nil {
		t.Skipf("no GroundingDINO vocab: %v", err)
	}
	return tok
}

// B5: a request GroundingDINO will not touch (every word in RF-DETR's vocabulary, or no prompt)
// must not queue behind the GroundingDINO lock — the router is the recommended default, and
// serialising it serialised every closed-set request on the server.
func TestInferClosedSetDoesNotTakeTheGroundingDINOLock(t *testing.T) {
	m := newLockTestHybrid(t)
	m.open.Lock() // a GroundingDINO request is running on this very model
	defer m.open.Unlock()

	for _, prompt := range []string{"cup.", ""} {
		done := make(chan error, 1)
		go func() {
			_, err := m.Infer(image.NewRGBA(image.Rect(0, 0, 8, 8)), models.Prompt{Text: prompt}, &lockProbeRunner{lock: &m.open})
			done <- err
		}()
		select {
		case err := <-done:
			if err != nil {
				t.Fatalf("prompt %q: %v", prompt, err)
			}
		case <-time.After(5 * time.Second):
			t.Fatalf("prompt %q: Infer blocked on the GroundingDINO lock although GroundingDINO never runs", prompt)
		}
	}
}

// Whenever the gdino role runs, this model's lock is held — and released when Infer returns.
func TestInferHoldsTheGroundingDINOLockWhileItRuns(t *testing.T) {
	m := newLockTestHybrid(t)
	r := &lockProbeRunner{lock: &m.open}
	if _, err := m.Infer(image.NewRGBA(image.Rect(0, 0, 8, 8)), models.Prompt{Text: "cup. zebra."}, r); err != nil {
		t.Fatal(err)
	}
	if len(r.gdinoLocked) == 0 {
		t.Fatal("GroundingDINO never ran for an out-of-vocabulary word")
	}
	for i, held := range r.gdinoLocked {
		if !held {
			t.Errorf("gdino pass %d ran without the lock", i)
		}
	}
	if !m.open.TryLock() {
		t.Fatal("the lock is still held after Infer returned")
	}
	m.open.Unlock()
}

// The lock is per loaded MODEL: a GroundingDINO request on one router does not wait for another
// router's (the process-wide groundingdino.PipelineMu made every GroundingDINO pipeline on the
// server queue behind every other).
func TestGroundingDINOLockIsPerModel(t *testing.T) {
	a, b := newLockTestHybrid(t), newLockTestHybrid(t)
	a.open.Lock() // model a is busy in its GroundingDINO section
	defer a.open.Unlock()
	done := make(chan error, 1)
	go func() {
		_, err := b.Infer(image.NewRGBA(image.Rect(0, 0, 8, 8)), models.Prompt{Text: "zebra."}, &lockProbeRunner{lock: &b.open})
		done <- err
	}()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("model b's GroundingDINO request waited for model a")
	}
}

// The router locks its GroundingDINO section itself, so it must NOT ask the runtime to serialise
// whole requests: that would put closed-set requests back behind GroundingDINO ones.
func TestRouterIsNotExclusive(t *testing.T) {
	var m models.Base = &hybrid{}
	if ex, ok := m.(models.Exclusive); ok && ex.Exclusive() {
		t.Fatal("the router asks lifecycle for whole-request exclusivity")
	}
}

// B8: when EVERY detection handed to the rescorer is degenerate (e.g. one sub-pixel box on the
// image edge), the request must come back empty, not fail.
func TestRescoreAllDegenerateReturnsEmptyNotError(t *testing.T) {
	m := newTestHybrid(t)
	st := &stubTowers{dim: 2, cropRows: [][]float32{{1, 0}}}
	dets := []models.Detection{
		{Class: "zebra", Conf: 0.5, BBox: [4]float64{99.8, 10, 0.2, 5}}, // sub-pixel sliver on the edge
		{Class: "zebra", Conf: 0.4, BBox: [4]float64{100, 100, 0, 0}},
	}
	got, err := m.rescore(canvas(), dets, []string{"zebra", "cup"}, cropTemp, st)
	if err != nil {
		t.Fatalf("rescore failed the request: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("got %+v, want no detections (none had pixels to score)", got)
	}
	if st.cropCalls != 0 {
		t.Errorf("crop tower ran %d times with nothing to embed", st.cropCalls)
	}
}
