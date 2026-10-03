// Package promptens holds the prompt-ensemble mechanics shared by every head that turns a
// vocabulary into one text embedding per word: expand each class through a set of templates,
// embed all the rows in one batched tower call, average the per-template rows and re-normalise.
//
// It exists because two packages now do this against two DIFFERENT text towers — textalign
// against CLIP (head B's training contract) and hybrid against SigLIP (the crop rescorer) — and
// the part that must not drift between them is not the tower call but the ENSEMBLE: template
// expansion order, the normalise-average-normalise order, and the cache key. Getting the average
// order wrong (averaging raw embeddings instead of normalised ones) lets long templates dominate
// and silently changes every cosine downstream.
//
// It owns no session and knows nothing about ONNX: callers embed the strings Apply returns and
// hand the rows back to Average.
package promptens

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"strings"

	"visionserve/internal/vision/util"
)

// Placeholder is what Apply substitutes the class name for.
const Placeholder = "{}"

// Load reads a templates file (one template per line, "#" comments allowed). A missing file
// yields fallback, so a manifest that ships no templates.txt still works; every template present
// must contain Placeholder, because one that does not would embed the same string for every
// class and produce a vocabulary whose rows are all identical.
func Load(path string, fallback []string) ([]string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return fallback, nil
		}
		return nil, fmt.Errorf("promptens: cannot read %s: %w", path, err)
	}
	var out []string
	for _, line := range strings.Split(string(raw), "\n") {
		s := strings.TrimSpace(line)
		if s == "" || strings.HasPrefix(s, "#") {
			continue
		}
		if !strings.Contains(s, Placeholder) {
			return nil, fmt.Errorf("promptens: template %q in %s has no %s placeholder", s, path, Placeholder)
		}
		out = append(out, s)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("promptens: %s contains no templates", path)
	}
	return out, nil
}

// Apply expands the vocabulary into the strings to embed: classes × templates, in CLASS-MAJOR
// order (class 0's templates first). Average below relies on exactly this order.
func Apply(templates, classes []string) []string {
	out := make([]string, 0, len(classes)*len(templates))
	for _, c := range classes {
		for _, t := range templates {
			out = append(out, strings.ReplaceAll(t, Placeholder, c))
		}
	}
	return out
}

// Average collapses the [nClasses*nTemplates] embeddings Apply's strings produced into one
// L2-normalised row per class. The rows handed in must ALREADY be L2-normalised — averaging raw
// embeddings weights each template by its magnitude, which is not the ensemble anyone measured.
func Average(embs [][]float32, nClasses, nTemplates int) ([][]float32, error) {
	if nTemplates <= 0 || nClasses <= 0 {
		return nil, fmt.Errorf("promptens: bad ensemble shape (%d classes × %d templates)", nClasses, nTemplates)
	}
	if len(embs) != nClasses*nTemplates {
		return nil, fmt.Errorf("promptens: text tower returned %d embeddings, expected %d (%d classes × %d templates)",
			len(embs), nClasses*nTemplates, nClasses, nTemplates)
	}
	dim := len(embs[0])
	out := make([][]float32, nClasses)
	for c := 0; c < nClasses; c++ {
		acc := make([]float32, dim)
		for k := 0; k < nTemplates; k++ {
			e := embs[c*nTemplates+k]
			if len(e) != dim {
				return nil, fmt.Errorf("promptens: text embedding %d has dim %d, expected %d",
					c*nTemplates+k, len(e), dim)
			}
			for j, v := range e {
				acc[j] += v
			}
		}
		out[c] = util.L2NormalizeInPlace(acc)
	}
	return out, nil
}

// Key hashes (templates, classes) into a cache key. The templates are part of the key because
// they change the embeddings: the same words under a different ensemble are a different head.
// Class names are lowercased (the routing sets already are) but NOT sorted — order is meaningful,
// since the caller indexes score columns by it.
func Key(templates, classes []string) string {
	h := sha256.New()
	for _, t := range templates {
		_, _ = h.Write([]byte(t))
		_, _ = h.Write([]byte{0})
	}
	_, _ = h.Write([]byte{1})
	for _, c := range classes {
		_, _ = h.Write([]byte(strings.ToLower(c)))
		_, _ = h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}
