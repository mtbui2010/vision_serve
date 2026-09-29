package textalign

import (
	"fmt"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/models/clip"

	_ "visionserve/internal/models/rfdetr"
)

// stubRunner stands in for lifecycle's Runner: it answers the "text" role with a
// deterministic fake CLIP output, so the vocabulary-compilation path (tokenize → embed →
// ensemble → fold → cache) can be tested without loading a 255 MB ONNX graph.
type stubRunner struct {
	dText int
	calls int64 // number of clip-text invocations
	rows  int64 // number of prompt rows embedded
}

func (s *stubRunner) Run(role string, inputs map[string]engine.Tensor) ([]engine.Tensor, error) {
	if role != roleText {
		return nil, fmt.Errorf("stub: unexpected role %q", role)
	}
	ids, ok := inputs["input_ids"]
	if !ok {
		return nil, fmt.Errorf("stub: inputs %v have no input_ids", inputs)
	}
	if len(ids.Shape) != 2 || ids.Shape[1] != int64(clip.ContextLength) {
		return nil, fmt.Errorf("stub: input_ids shape %v, want [N,%d]", ids.Shape, clip.ContextLength)
	}
	n := int(ids.Shape[0])
	atomic.AddInt64(&s.calls, 1)
	atomic.AddInt64(&s.rows, int64(n))
	// Row i = the (i mod dText)-th basis vector, scaled — the model must L2-normalise it.
	data := make([]float32, n*s.dText)
	for i := 0; i < n; i++ {
		data[i*s.dText+i%s.dText] = float32(3 + i)
	}
	return []engine.Tensor{engine.F32(data, int64(n), int64(s.dText))}, nil
}

func (s *stubRunner) InputNames(role string) []string  { return []string{"input_ids"} }
func (s *stubRunner) OutputNames(role string) []string { return []string{"text_embeds"} }

// newTestModel wires a textAlign around the real CLIP tokenizer (skipping when the
// tokenizer assets are not present, since model directories are not committed).
func newTestModel(t *testing.T, pr *Projection, templates []string) *textAlign {
	t.Helper()
	dir := filepath.Join("..", "..", "..", "models", "clip-text")
	tok, err := clip.LoadTokenizer(dir)
	if err != nil {
		t.Skipf("no CLIP tokenizer assets in %s: %v", dir, err)
	}
	cfg := models.Config{Name: "ta-test", Width: 512, Height: 512, BoxFormat: "cxcywh", ConfThresh: 0.5, MaxDet: 300}
	return &textAlign{cfg: cfg, proj: pr, tok: clipTokenizer{tok}, tmpl: templates, cache: map[string]*head{}}
}

// TestHeadForCachesVocabularies is the "one clip-text call per NEW vocabulary, never per
// request" claim the latency budget rests on.
func TestHeadForCachesVocabularies(t *testing.T) {
	pr := testProjection(t) // dText 3
	m := newTestModel(t, pr, []string{"a photo of a {}.", "itap of a {}."})
	r := &stubRunner{dText: pr.DText}

	h1, err := m.headFor([]string{"cup", "hat"}, r)
	if err != nil {
		t.Fatalf("headFor: %v", err)
	}
	if got := atomic.LoadInt64(&r.rows); got != 4 {
		t.Errorf("embedded %d prompt rows, want 4 (2 classes × 2 templates)", got)
	}
	if len(h1.w) != 2*pr.DFeat {
		t.Errorf("W has %d values, want %d", len(h1.w), 2*pr.DFeat)
	}

	h2, err := m.headFor([]string{"cup", "hat"}, r)
	if err != nil {
		t.Fatal(err)
	}
	if h2 != h1 {
		t.Errorf("the same vocabulary rebuilt a head instead of hitting the cache")
	}
	if got := atomic.LoadInt64(&r.calls); got != 1 {
		t.Errorf("clip-text was called %d times for one vocabulary, want 1", got)
	}

	if _, err := m.headFor([]string{"cup", "banana"}, r); err != nil {
		t.Fatal(err)
	}
	if got := atomic.LoadInt64(&r.calls); got != 2 {
		t.Errorf("a NEW vocabulary must call clip-text (calls=%d, want 2)", got)
	}
}

// TestHeadForConcurrent exercises the cache from many goroutines (the server serves
// requests in parallel — CLAUDE.md). Run with -race.
func TestHeadForConcurrent(t *testing.T) {
	pr := testProjection(t)
	m := newTestModel(t, pr, defaultTemplates)
	r := &stubRunner{dText: pr.DText}

	var wg sync.WaitGroup
	errs := make(chan error, 32)
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			vocab := []string{"cup", "hat"}
			if i%2 == 1 {
				vocab = []string{fmt.Sprintf("class%d", i%5)}
			}
			h, err := m.headFor(vocab, r)
			if err != nil {
				errs <- err
				return
			}
			if len(h.w) != len(vocab)*pr.DFeat {
				errs <- fmt.Errorf("W has %d values for %d classes", len(h.w), len(vocab))
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	if n := len(m.cache); n != 6 { // {cup,hat} + class0..class4
		t.Errorf("cache holds %d vocabularies, want 6", n)
	}
}

// TestEmbedVocabRejectsWrongTextDim: a text tower whose width does not match the
// projection must fail loudly rather than fold a mis-shaped matrix.
func TestEmbedVocabRejectsWrongTextDim(t *testing.T) {
	pr := testProjection(t)
	m := newTestModel(t, pr, defaultTemplates)
	if _, err := m.embedVocab([]string{"cup"}, &stubRunner{dText: pr.DText + 1}); err == nil {
		t.Errorf("expected an error when clip-text's width != projection d_text")
	}
}
