package textalign

import (
	"fmt"
	"os"
	"path/filepath"

	"visionserve/internal/models/clip"
	"visionserve/internal/models/siglip"
)

// textTokenizer is the only thing head B needs from a text tower's tokenizer: ids for a batch of
// prompts, and the fixed context width those ids are padded to.
//
// It exists because the tokenizer is NOT a property of head B — it is a property of the teacher
// the head was distilled against. A head distilled from CLIP must be served with CLIP's
// byte-level BPE at 77 tokens; one distilled from SigLIP with SentencePiece Unigram at 64. Before
// this, New() called clip.LoadTokenizer unconditionally, so pointing files.text at the SigLIP
// tower failed at load with a missing vocab.json — which is how it was caught.
type textTokenizer interface {
	EncodeBatch(texts []string) ([]int64, error)
	ContextLength() int
}

// clipTokenizer adapts internal/models/clip's tokenizer, whose EncodeBatch cannot fail.
type clipTokenizer struct{ t *clip.Tokenizer }

func (c clipTokenizer) EncodeBatch(texts []string) ([]int64, error) {
	return c.t.EncodeBatch(texts), nil
}
func (c clipTokenizer) ContextLength() int { return clip.ContextLength }

// siglipTokenizer adapts internal/models/siglip's, which refuses non-ASCII rather than guessing.
type siglipTokenizer struct{ t *siglip.Tokenizer }

func (s siglipTokenizer) EncodeBatch(texts []string) ([]int64, error) {
	return s.t.EncodeBatch(texts)
}
func (s siglipTokenizer) ContextLength() int { return s.t.MaxLen() }

// loadTextTokenizer picks the tokenizer by what the text tower's directory actually contains:
// tokenizer.json → SigLIP (SentencePiece Unigram), vocab.json + merges.txt → CLIP (BPE).
//
// Probing beats a manifest field here: the assets and the ONNX ship together, so the directory
// already carries the answer, and a manifest that disagreed with its own weights would be a new
// way to be silently wrong.
func loadTextTokenizer(dir string) (textTokenizer, error) {
	if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); err == nil {
		t, err := siglip.LoadTokenizer(dir)
		if err != nil {
			return nil, fmt.Errorf("textalign: load SigLIP tokenizer from %s: %w", dir, err)
		}
		return siglipTokenizer{t}, nil
	}
	if _, err := os.Stat(filepath.Join(dir, clip.VocabFile)); err == nil {
		t, err := clip.LoadTokenizer(dir)
		if err != nil {
			return nil, fmt.Errorf("textalign: load CLIP tokenizer from %s: %w", dir, err)
		}
		return clipTokenizer{t}, nil
	}
	return nil, fmt.Errorf(
		"textalign: %s has neither tokenizer.json (SigLIP) nor %s (CLIP) — files.text must point "+
			"at a text tower directory that carries its tokenizer assets", dir, clip.VocabFile)
}
