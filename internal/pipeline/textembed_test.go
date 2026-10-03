package pipeline

import (
	"fmt"
	"hash/fnv"
	"math/rand"
	"reflect"
	"strings"
	"sync"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models/promptens"
	"visionserve/internal/vision/util"
)

// fakeTok encodes a text as its first ctx bytes (zero-padded): deterministic, distinct per text.
type fakeTok struct{ ctx int }

func (f fakeTok) EncodeBatch(texts []string) ([]int64, error) {
	out := make([]int64, 0, len(texts)*f.ctx)
	for _, t := range texts {
		for i := 0; i < f.ctx; i++ {
			v := int64(0)
			if i < len(t) {
				v = int64(t[i])
			}
			out = append(out, v)
		}
	}
	return out, nil
}
func (f fakeTok) ContextLength() int { return f.ctx }

// towerRunner is a text tower whose output row is a pseudo-random, UNNORMALISED function of that
// row's ids alone — like a real tower, identical text embeds identically wherever it sits in the
// batch. It records every batch it is handed.
type towerRunner struct {
	dim     int
	mu      sync.Mutex
	batches [][]int64
	rows    int
}

func (tr *towerRunner) Run(role string, in map[string]engine.Tensor) ([]engine.Tensor, error) {
	ids, ok := in["input_ids"]
	if !ok {
		return nil, fmt.Errorf("no input_ids in %v", in)
	}
	n, l := int(ids.Shape[0]), int(ids.Shape[1])
	tr.mu.Lock()
	tr.batches = append(tr.batches, append([]int64(nil), ids.DataI64...))
	tr.rows += n
	tr.mu.Unlock()
	out := make([]float32, n*tr.dim)
	for i := 0; i < n; i++ {
		h := fnv.New64a()
		fmt.Fprint(h, ids.DataI64[i*l:(i+1)*l])
		r := rand.New(rand.NewSource(int64(h.Sum64())))
		for j := 0; j < tr.dim; j++ {
			out[i*tr.dim+j] = float32(r.NormFloat64()) * 3
		}
	}
	return []engine.Tensor{engine.F32(out, int64(n), int64(tr.dim))}, nil
}
func (tr *towerRunner) InputNames(string) []string  { return []string{"input_ids"} }
func (tr *towerRunner) OutputNames(string) []string { return []string{"text_embeds"} }

var testTemplates = []string{"a photo of a {}.", "itap of a {}.", "a {} on a table."}

// refEmbed is the whole-list computation the three caches it replaced performed on a miss:
// expand class-major, ONE tower call over every row, L2-normalise each row, average per word.
func refEmbed(t *testing.T, words []string, tr *towerRunner) ([][]float32, []int64) {
	t.Helper()
	tok := fakeTok{ctx: 12}
	texts := promptens.Apply(testTemplates, words)
	ids, _ := tok.EncodeBatch(texts)
	outs, err := tr.Run("text", map[string]engine.Tensor{"input_ids": engine.I64(ids, int64(len(texts)), 12)})
	if err != nil {
		t.Fatal(err)
	}
	embs := make([][]float32, len(texts))
	for i := range embs {
		embs[i] = util.L2Normalized(outs[0].Data[i*tr.dim : (i+1)*tr.dim])
	}
	rows, err := promptens.Average(embs, len(words), len(testTemplates))
	if err != nil {
		t.Fatal(err)
	}
	return rows, ids
}

// The acceptance criterion of the per-word cache: the embeddings of a word list are IDENTICAL to
// what the whole-list caches computed — on a cold cache (same single batch, bit for bit) and when
// part of the list is already cached from other requests.
func TestEmbedMatchesWholeListComputation(t *testing.T) {
	words := []string{"cup", "towel", "zebra", "snack bag"}
	want, wantIDs := refEmbed(t, words, &towerRunner{dim: 16})

	tr := &towerRunner{dim: 16}
	e := NewTextEmbedder("text", fakeTok{ctx: 12}, testTemplates, 0)
	got, err := e.Embed(words, tr)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatal("cold cache: embeddings differ from the whole-list computation")
	}
	if len(tr.batches) != 1 || !reflect.DeepEqual(tr.batches[0], wantIDs) {
		t.Fatalf("cold cache: the tower saw %d batches; want exactly the whole-list batch", len(tr.batches))
	}

	// Warm, partially: a fresh embedder that has already embedded two of the words elsewhere.
	tr2 := &towerRunner{dim: 16}
	e2 := NewTextEmbedder("text", fakeTok{ctx: 12}, testTemplates, 0)
	if _, err := e2.Embed([]string{"zebra", "lamp", "cup"}, tr2); err != nil {
		t.Fatal(err)
	}
	got2, err := e2.Embed(words, tr2)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got2, want) {
		t.Fatal("warm cache: embeddings differ from the whole-list computation")
	}
	if tr2.rows != (3+2)*len(testTemplates) {
		t.Errorf("tower embedded %d rows, want %d: only towel and snack bag were new", tr2.rows, 5*len(testTemplates))
	}
}

func TestEmbedCachesPerWord(t *testing.T) {
	tr := &towerRunner{dim: 8}
	e := NewTextEmbedder("text", fakeTok{ctx: 12}, testTemplates, 0)
	for i := 0; i < 3; i++ {
		if _, err := e.Embed([]string{"cup", "hat"}, tr); err != nil {
			t.Fatal(err)
		}
	}
	if len(tr.batches) != 1 {
		t.Errorf("the tower ran %d times for one repeated word list, want 1", len(tr.batches))
	}
	// Duplicates in one request are embedded once and share one row.
	rows, err := e.Embed([]string{"bread", "bread", "cup"}, tr)
	if err != nil {
		t.Fatal(err)
	}
	if len(tr.batches) != 2 || tr.rows != 3*len(testTemplates) {
		t.Errorf("batches=%d rows=%d; want one new batch holding only bread", len(tr.batches), tr.rows)
	}
	if &rows[0][0] != &rows[1][0] {
		t.Error("a repeated word did not share its row")
	}
	if e.Cached() != 3 {
		t.Errorf("Cached = %d, want 3", e.Cached())
	}
}

// Templates are part of every key: the same word under another ensemble is another row.
func TestEmbedKeyIncludesTemplates(t *testing.T) {
	a := NewTextEmbedder("text", nil, []string{"a photo of a {}."}, 0)
	b := NewTextEmbedder("text", nil, []string{"a photo of the {}."}, 0)
	if a.Key("cup") == b.Key("cup") || a.Key("cup") == a.Key("hat") {
		t.Error("keys must differ by word and by template set")
	}
	if a2 := NewTextEmbedder("text", nil, []string{"a photo of a {}."}, 0); a.Key("cup") != a2.Key("cup") ||
		!strings.HasSuffix(a.Key("cup"), "\x00cup") {
		t.Errorf("key %q is not <template hash>\\x00<word>", a.Key("cup"))
	}
}

// A tower of another checkpoint (another width) must fail before anything is cached.
func TestEmbedRejectsWrongWidth(t *testing.T) {
	e := NewTextEmbedder("text", fakeTok{ctx: 12}, testTemplates, 32)
	if _, err := e.Embed([]string{"cup"}, &towerRunner{dim: 16}); err == nil ||
		!strings.Contains(err.Error(), "expected [3,32]") {
		t.Fatalf("got %v, want a width mismatch error", err)
	}
	if e.Cached() != 0 {
		t.Error("a mis-shaped row was cached")
	}
}

func TestEmbedConcurrent(t *testing.T) {
	tr := &towerRunner{dim: 8}
	e := NewTextEmbedder("text", fakeTok{ctx: 12}, testTemplates, 0)
	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			words := []string{"cup", fmt.Sprintf("w%d", g%4)}
			rows, err := e.Embed(words, tr)
			if err != nil || len(rows) != 2 || len(rows[1]) != 8 {
				t.Errorf("Embed(%v) = %d rows, %v", words, len(rows), err)
			}
		}(g)
	}
	wg.Wait()
	if e.Cached() != 5 {
		t.Errorf("Cached = %d, want 5", e.Cached())
	}
}

func TestLoadTextTokenizerPicksByAssets(t *testing.T) {
	if _, err := LoadTextTokenizer(t.TempDir()); err == nil {
		t.Error("a directory with no tokenizer assets was accepted")
	}
	for dir, want := range map[string]string{"siglip-text": "SigLIPTokenizer", "clip-text": "CLIPTokenizer"} {
		tok, err := LoadTextTokenizer("../../models/" + dir)
		if err != nil {
			t.Logf("%s: %v (assets not present)", dir, err)
			continue
		}
		if got := reflect.TypeOf(tok).Name(); got != want {
			t.Errorf("%s: picked %s, want %s", dir, got, want)
		}
	}
}
