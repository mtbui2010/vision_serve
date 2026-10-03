package hybrid

import (
	"fmt"
	"image"
	"reflect"
	"sync"
	"testing"
	"time"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/models/groundingdino"
)

// B2: a word repeated in the prompt ("zebra. Zebra.") must reach the rescorer ONCE. Twice, the
// softmax runs over two identical rows and every rescored confidence is divided by the repeat
// count; it also costs a duplicate GroundingDINO phrase.
func TestParseClassesDedupesCaseInsensitively(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"zebra. zebra.", []string{"zebra"}},
		{"Zebra. cup. ZEBRA. cup.", []string{"zebra", "cup"}},
		{"cup. zebra. cup.", []string{"cup", "zebra"}}, // first-seen order
	}
	for _, c := range cases {
		if got := parseClasses(c.in); !reflect.DeepEqual(got, c.want) {
			t.Errorf("parseClasses(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

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
	got, err := m.rescore(canvas(), dets, parseClasses("zebra. Zebra."), cropTemp, st)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Conf < 0.8-1e-6 {
		t.Fatalf("rescored %+v, want one detection keeping conf 0.8 (a single word is certain)", got)
	}
}

// B7: internal whitespace must not decide the route — "dining  table" is RF-DETR's
// "dining table".
func TestParseClassesCollapsesInternalWhitespace(t *testing.T) {
	if got, want := parseClasses("dining  table.\tteddy \t bear."), []string{"dining table", "teddy bear"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("parseClasses = %q, want %q", got, want)
	}
	m := &hybrid{vocab: map[string]bool{"dining table": true}}
	if known, unknown := m.partition(parseClasses("dining  table.")); len(known) != 1 || len(unknown) != 0 {
		t.Fatalf("partition = %v / %v, want dining table routed to RF-DETR", known, unknown)
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

// lockProbeRunner answers the rfdetr role, and records whether groundingdino.PipelineMu was
// held while the gdino role ran.
type lockProbeRunner struct {
	mu          sync.Mutex
	gdinoLocked []bool
}

func (r *lockProbeRunner) Run(role string, in map[string]engine.Tensor) ([]engine.Tensor, error) {
	switch role {
	case roleRFDETR:
		return []engine.Tensor{engine.F32(make([]float32, 4), 1, 1, 4)}, nil
	case roleGDINO:
		held := !groundingdino.PipelineMu.TryLock()
		if !held {
			groundingdino.PipelineMu.Unlock()
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
	return &hybrid{cfg: models.Config{Name: "rfdetr-gdino"}, rf: fakeRF{}, tok: tok,
		vocab: map[string]bool{"cup": true}}
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
// must not queue behind PipelineMu — the router is the recommended default, and serialising it
// globally serialised every closed-set request on the server.
func TestInferClosedSetDoesNotTakePipelineMu(t *testing.T) {
	m := newLockTestHybrid(t)
	groundingdino.PipelineMu.Lock() // another GroundingDINO pipeline is running
	defer groundingdino.PipelineMu.Unlock()

	for _, prompt := range []string{"cup.", ""} {
		done := make(chan error, 1)
		go func() {
			_, err := m.Infer(image.NewRGBA(image.Rect(0, 0, 8, 8)), models.Prompt{Text: prompt}, &lockProbeRunner{})
			done <- err
		}()
		select {
		case err := <-done:
			if err != nil {
				t.Fatalf("prompt %q: %v", prompt, err)
			}
		case <-time.After(5 * time.Second):
			t.Fatalf("prompt %q: Infer blocked on PipelineMu although GroundingDINO never runs", prompt)
		}
	}
}

// The guarantee PipelineMu exists for is kept: whenever the gdino role runs, the lock is held.
func TestInferHoldsPipelineMuWhileGroundingDINORuns(t *testing.T) {
	m := newLockTestHybrid(t)
	r := &lockProbeRunner{}
	if _, err := m.Infer(image.NewRGBA(image.Rect(0, 0, 8, 8)), models.Prompt{Text: "cup. zebra."}, r); err != nil {
		t.Fatal(err)
	}
	if len(r.gdinoLocked) == 0 {
		t.Fatal("GroundingDINO never ran for an out-of-vocabulary word")
	}
	for i, held := range r.gdinoLocked {
		if !held {
			t.Errorf("gdino pass %d ran without PipelineMu", i)
		}
	}
	if !groundingdino.PipelineMu.TryLock() {
		t.Fatal("PipelineMu still held after Infer returned")
	}
	groundingdino.PipelineMu.Unlock()
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
