package textalign

import (
	"fmt"
	"strings"

	"visionserve/internal/models"
)

// head is one vocabulary, compiled for deployment: the class names, the folded class
// matrix W = a·T̂P, and an rf-detr sub-model carrying those names as its labels (so the
// EXISTING RF-DETR postprocess maps boxes back to ORIGINAL image coordinates and names
// the classes — this package never re-implements letterbox/scale mapping).
type head struct {
	classes []string
	w       []float32    // [C * DFeat], row-major — the deploy-time class matrix
	rf      models.Model // rf-detr sub-model whose cfg.Labels == classes
}

// logits computes per-query class logits from the detector's query_feats.
//
//	feats    flat [q * DFeat] (the query_feats tensor, row-major)
//	normalize  true  → EXACT head:   logit_ic = a·⟨t̂_c, P f_i⟩/‖P f_i‖ + b   (= ⟨W_c,f_i⟩/‖P f_i‖ + b)
//	           false → FOLDED head:  logit_ic = ⟨W_c, f_i⟩ + b                (a plain nn.Linear)
//
// Both branches share one matmul against W; the only difference is the per-query scalar
// 1/‖P f_i‖, which costs the extra P·f product. The folded form has the SAME argmax over
// classes (the scalar is positive and class-independent) but a different sigmoid
// calibration — see README, and the FlagUnitNorm bit on the projection.
//
// The returned slice is row-major [q, C], pre-sigmoid — exactly the layout RF-DETR's own
// postprocess expects for a class-logits tensor.
func (h *head) logits(p *Projection, feats []float32, q int, normalize bool) ([]float32, error) {
	c := len(h.classes)
	if q <= 0 || c == 0 {
		return nil, fmt.Errorf("textalign: nothing to score (queries=%d classes=%d)", q, c)
	}
	if len(feats) < q*p.DFeat {
		return nil, fmt.Errorf("textalign: query_feats has %d values, need %d (%d queries × %d)",
			len(feats), q*p.DFeat, q, p.DFeat)
	}
	out := make([]float32, q*c)
	for i := 0; i < q; i++ {
		f := feats[i*p.DFeat : (i+1)*p.DFeat]
		inv := float32(1)
		if normalize {
			n := p.ProjNorm(f)
			if n <= 0 {
				// Degenerate query (‖P f‖ = 0): cosine is undefined, so score it as
				// background (all logits = bias) instead of dividing by zero.
				for k := 0; k < c; k++ {
					out[i*c+k] = p.Bias
				}
				continue
			}
			inv = 1 / n
		}
		for k := 0; k < c; k++ {
			row := h.w[k*p.DFeat : (k+1)*p.DFeat]
			var s float32
			for j, fv := range f {
				s += row[j] * fv
			}
			out[i*c+k] = s*inv + p.Bias
		}
	}
	return out, nil
}

// normalizeVocab lowercases/trims the requested phrases and drops empties and duplicates
// (a duplicate class would produce two identical columns and duplicate detections).
//
// The phrase itself is lowercased, not only its dedup key: the text cache is keyed per word, and
// keeping the caller's case let the first request's spelling ("Cup") become the label later "cup"
// requests received (the cache this replaced was keyed on the lowercased vocabulary). Both text
// towers lowercase before tokenizing, so the embeddings are unchanged.
func normalizeVocab(in []string) []string {
	seen := make(map[string]bool, len(in))
	out := make([]string, 0, len(in))
	for _, s := range in {
		s = strings.ToLower(strings.TrimSpace(s))
		if s == "" || seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	return out
}
