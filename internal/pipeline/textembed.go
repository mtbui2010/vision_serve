package pipeline

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/models/clip"
	"visionserve/internal/models/promptens"
	"visionserve/internal/models/siglip"
	"visionserve/internal/vision/util"
)

// MaxCachedWords bounds each TextEmbedder's cache. A row is one text-tower width of float32
// (768 for SigLIP base: 3 KB), so the bound is ~3 MB per loaded model; it exists to stop
// unbounded growth under adversarial per-request prompts, not to save memory. A request whose
// own word list is longer than the bound still gets every row (they are returned before
// eviction can matter); it just does not keep them all.
const MaxCachedWords = 1024

// TextTokenizer is what an embedder needs from a text tower's tokenizer: ids for a batch of
// prompts, padded to the tower's fixed context width.
//
// The tokenizer is a property of the TEACHER a head was distilled against, not of the head: CLIP
// is byte-level BPE at 77 tokens, SigLIP SentencePiece Unigram at 64 (padded with </s>,
// BUGS_TO_FIX.md #5). Serving a head with the other tokenizer fails loudly at load or, worse,
// silently.
type TextTokenizer interface {
	EncodeBatch(texts []string) ([]int64, error)
	ContextLength() int
}

// CLIPTokenizer adapts internal/models/clip's tokenizer, whose EncodeBatch cannot fail.
type CLIPTokenizer struct{ T *clip.Tokenizer }

func (c CLIPTokenizer) EncodeBatch(texts []string) ([]int64, error) {
	return c.T.EncodeBatch(texts), nil
}
func (c CLIPTokenizer) ContextLength() int { return clip.ContextLength }

// SigLIPTokenizer adapts internal/models/siglip's, which refuses non-ASCII rather than guessing.
type SigLIPTokenizer struct{ T *siglip.Tokenizer }

func (s SigLIPTokenizer) EncodeBatch(texts []string) ([]int64, error) { return s.T.EncodeBatch(texts) }
func (s SigLIPTokenizer) ContextLength() int                          { return s.T.MaxLen() }

// LoadTextTokenizer picks the tokenizer by what the text tower's directory actually contains:
// tokenizer.json → SigLIP (SentencePiece Unigram), vocab.json + merges.txt → CLIP (BPE). The
// assets and the ONNX ship together, so the directory already carries the answer, and a manifest
// field that disagreed with its own weights would be a new way to be silently wrong.
func LoadTextTokenizer(dir string) (TextTokenizer, error) {
	if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); err == nil {
		t, err := siglip.LoadTokenizer(dir)
		if err != nil {
			return nil, fmt.Errorf("load SigLIP tokenizer from %s: %w", dir, err)
		}
		return SigLIPTokenizer{t}, nil
	}
	if _, err := os.Stat(filepath.Join(dir, clip.VocabFile)); err == nil {
		t, err := clip.LoadTokenizer(dir)
		if err != nil {
			return nil, fmt.Errorf("load CLIP tokenizer from %s: %w", dir, err)
		}
		return CLIPTokenizer{t}, nil
	}
	return nil, fmt.Errorf(
		"%s has neither tokenizer.json (SigLIP) nor %s (CLIP) — files.text must point "+
			"at a text tower directory that carries its tokenizer assets", dir, clip.VocabFile)
}

// TextEmbedder turns words into one L2-normalised text embedding each, through a prompt ensemble
// and the text tower session under Role, with a bounded per-WORD cache.
//
// The embedding of a word is exactly what the whole-list caches it replaced computed for it:
// every word is expanded through the templates in class-major order (promptens.Apply), all rows
// of the words being embedded go through the tower in ONE batched call, each output row is
// L2-normalised, and the per-word rows are averaged and re-normalised (promptens.Average). Only
// the words missing from the cache are embedded, so a new word list costs its new words — about
// 4.3 ms per word per request for SigLIP — instead of re-embedding every word of the list.
type TextEmbedder struct {
	Role      string        // the text tower's session role
	Tok       TextTokenizer // the tower's own tokenizer
	Templates []string      // the prompt ensemble; part of the measured contract
	// Dim, when > 0, is the width the tower must emit (a head's projection d_text): a tower of
	// another checkpoint is refused before anything is cached.
	Dim int

	tkey  string // hash of Templates, the first half of every cache key
	cache *LRU[[]float32]
}

// NewTextEmbedder builds an embedder with an empty cache of MaxCachedWords words.
func NewTextEmbedder(role string, tok TextTokenizer, templates []string, dim int) *TextEmbedder {
	return &TextEmbedder{
		Role: role, Tok: tok, Templates: templates, Dim: dim,
		tkey:  promptens.TemplateKey(templates),
		cache: NewLRU[[]float32](MaxCachedWords),
	}
}

// Key is the cache key of word under this embedder's templates. A word is keyed exactly as it is
// embedded (no case folding): callers normalise their vocabularies before asking.
func (e *TextEmbedder) Key(word string) string { return e.tkey + "\x00" + word }

// Cached reports how many words the cache holds.
func (e *TextEmbedder) Cached() int { return e.cache.Len() }

// Embed returns one row per word, in order. Rows are shared with the cache: callers must not
// modify them.
func (e *TextEmbedder) Embed(words []string, r models.Runner) ([][]float32, error) {
	rows := make([][]float32, len(words))
	var miss []string
	seen := map[string]bool{}
	for i, w := range words {
		if v, ok := e.cache.Get(e.Key(w)); ok {
			rows[i] = v
		} else if !seen[w] {
			seen[w] = true
			miss = append(miss, w)
		}
	}
	if len(miss) == 0 {
		return rows, nil
	}
	fresh, err := e.embed(miss, r)
	if err != nil {
		return nil, err
	}
	got := make(map[string][]float32, len(miss))
	for j, w := range miss {
		got[w] = e.cache.Add(e.Key(w), fresh[j])
	}
	for i, w := range words {
		if rows[i] == nil {
			rows[i] = got[w]
		}
	}
	return rows, nil
}

// embed runs the tower once over words × templates and ensembles the rows per word.
func (e *TextEmbedder) embed(words []string, r models.Runner) ([][]float32, error) {
	if e.Tok == nil {
		return nil, fmt.Errorf("text tower %q: no tokenizer", e.Role)
	}
	texts := promptens.Apply(e.Templates, words)
	if len(texts) == 0 {
		return nil, fmt.Errorf("text tower %q: nothing to embed", e.Role)
	}
	ids, err := e.Tok.EncodeBatch(texts)
	if err != nil {
		return nil, err
	}
	outs, err := r.Run(e.Role, map[string]engine.Tensor{
		pickInput(r.InputNames(e.Role), "input_ids"): engine.I64(ids, int64(len(texts)), int64(e.Tok.ContextLength())),
	})
	if err != nil {
		return nil, fmt.Errorf("text tower inference: %w", err)
	}
	if len(outs) == 0 || len(outs[0].Shape) != 2 {
		return nil, fmt.Errorf("text tower returned an unexpected output %v (want [N,D])", util.ShapesOf(outs))
	}
	t := outs[0]
	n, dim := int(t.Shape[0]), int(t.Shape[1])
	if n != len(texts) || dim <= 0 || len(t.Data) < n*dim {
		return nil, fmt.Errorf("text tower returned [%d,%d] with %d values for %d prompts", n, dim, len(t.Data), len(texts))
	}
	if e.Dim > 0 && dim != e.Dim {
		return nil, fmt.Errorf("text tower returned [%d,%d], expected [%d,%d] (the head's projection d_text) — "+
			"head and tower are different checkpoints", n, dim, len(texts), e.Dim)
	}
	embs := make([][]float32, n)
	for i := range embs {
		embs[i] = util.L2Normalized(t.Data[i*dim : (i+1)*dim])
	}
	return promptens.Average(embs, len(words), len(e.Templates))
}

// pickInput returns want when the session declares it, otherwise the session's first input (an
// export with other names), and want when the session declares none.
func pickInput(names []string, want string) string {
	for _, n := range names {
		if n == want {
			return want
		}
	}
	return util.FirstName(names, want)
}

// joinPhrases rebuilds a GroundingDINO-style prompt from words ([cat, remote] -> "cat. remote.").
func joinPhrases(words []string) string {
	var b strings.Builder
	for _, w := range words {
		b.WriteString(w)
		b.WriteString(". ")
	}
	return strings.TrimSpace(b.String())
}
