package textalign

import (
	"fmt"
	"math"
	"strings"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// roleHead is the projection exported as ONNX (`files.head: head.onnx`, built from proj.bin by
// models/rfdetr-textalign-dec1-siglip/export_head_onnx.py). OPTIONAL: when the manifest declares
// it, method "exact" runs as a session owned by lifecycle.Manager on the manifest's provider chain
// (the detector's), through the Runner like every other role. When it does not, the exact head is
// scored in Go exactly as before.
//
// WHY ONLY "exact". The exact head divides by ‖P f‖ per query, which in Go is a 256×256 quadratic
// form per query — ~22 ms per request at 300 queries, more than the rest of the head together.
// "folded", "gated" and "dual" skip the norm and are one [300,256]×[256,C] matmul against the
// cached W, so they stay in Go and keep reading proj.bin. proj.bin therefore remains the source of
// truth, and checkHeadAgainstProj holds head.onnx to it on every request.
//
// It is the SAME graph as the fast-path head in internal/models/hybrid (same VSTXALN1 object,
// same function), so both packages run it through RunHeadONNX.
const roleHead = "head"

// The head graph's contract (see the export scripts' docstrings for the derivation):
//
//	query_feats  float32 [1, Q, d_feat]    the detector's own query features
//	text_embeds  float32 [C, d_text]       L2-normalised text rows, UNSCALED — per request
//	logits       float32 [1, Q, C]         Scale·cos(P f, t) + Bias, RAW (no sigmoid)
//
// Scale and Bias are initializers of the graph and applied there exactly once; callers pass the
// text rows as the text embedder returns them and add nothing.
const (
	HeadInFeats   = "query_feats"
	HeadInText    = "text_embeds"
	HeadOutLogits = "logits"
)

// RunHeadONNX runs the head graph declared under `role` on the first q query features against the
// text rows, and returns its RAW logits row-major [q][len(text)]. Errors carry no package prefix;
// callers wrap them.
func RunHeadONNX(r models.Runner, role string, feats engine.Tensor, text [][]float32, q int) ([]float32, error) {
	if r == nil {
		return nil, fmt.Errorf("the ONNX head needs a Runner")
	}
	dFeat := int(feats.Dim(-1))
	if q <= 0 || dFeat <= 0 || len(feats.Data) < q*dFeat {
		return nil, fmt.Errorf("query features %v do not cover %d queries", feats.Shape, q)
	}
	if len(text) == 0 || len(text[0]) == 0 {
		return nil, fmt.Errorf("no text embeddings to score the head against")
	}
	names := r.InputNames(role)
	if !hasName(names, HeadInFeats) || !hasName(names, HeadInText) {
		return nil, fmt.Errorf("files.%s has inputs %v, want %q and %q — export it with "+
			"export_head_onnx.py (models/rfdetr-textalign-dec1-siglip/ for a textalign proj.bin, "+
			"models/rfdetr-gdino-fastpath/ for a fast-path head.bin)", role, names, HeadInFeats, HeadInText)
	}
	n, dText := len(text), len(text[0])
	flat := make([]float32, 0, n*dText)
	for c, row := range text {
		if len(row) != dText {
			return nil, fmt.Errorf("text embedding %d has dim %d, embedding 0 has %d", c, len(row), dText)
		}
		flat = append(flat, row...)
	}
	outs, err := r.Run(role, map[string]engine.Tensor{
		HeadInFeats: engine.F32(feats.Data[:q*dFeat], 1, int64(q), int64(dFeat)),
		HeadInText:  engine.F32(flat, int64(n), int64(dText)),
	})
	if err != nil {
		return nil, fmt.Errorf("head session (files.%s) rejected query features [1 %d %d] "+
			"and text [%d %d] — the head must be exported from the projection trained against this "+
			"detector export and this text tower: %w", role, q, dFeat, n, dText, err)
	}
	idx := -1
	for i, name := range r.OutputNames(role) {
		if name == HeadOutLogits {
			idx = i
		}
	}
	if idx < 0 && len(outs) == 1 {
		idx = 0
	}
	if idx < 0 || idx >= len(outs) {
		return nil, fmt.Errorf("head session has no %q output (outputs %v)", HeadOutLogits, r.OutputNames(role))
	}
	out := outs[idx]
	if len(out.Shape) != 3 || int(out.Dim(1)) != q || int(out.Dim(2)) != n || len(out.Data) != q*n {
		return nil, fmt.Errorf("head logits have shape %v, want [1 %d %d]", out.Shape, q, n)
	}
	return out.Data, nil
}

func hasName(names []string, want string) bool {
	for _, n := range names {
		if n == want {
			return true
		}
	}
	return false
}

// hasHeadONNX reports whether the manifest wires the exact head as an ONNX session.
func (m *textAlign) hasHeadONNX() bool { return strings.TrimSpace(m.cfg.Files[roleHead]) != "" }

// exactLogitsONNX is h.logits(…, normalize=true) on the roleHead session: the same [q, C] raw
// logits, held to proj.bin by checkHeadAgainstProj.
func (m *textAlign) exactLogitsONNX(h *head, feats engine.Tensor, q int, r models.Runner) ([]float32, error) {
	if q <= 0 || len(h.classes) == 0 {
		return nil, fmt.Errorf("textalign: nothing to score (queries=%d classes=%d)", q, len(h.classes))
	}
	if d := int(feats.Dim(-1)); d != m.proj.DFeat || len(feats.Data) < q*d {
		return nil, fmt.Errorf("textalign: query_feats %v, need [1 %d %d]", feats.Shape, q, m.proj.DFeat)
	}
	if len(h.text) != len(h.classes) {
		return nil, fmt.Errorf("textalign: %d text rows for %d classes", len(h.text), len(h.classes))
	}
	logits, err := RunHeadONNX(r, roleHead, feats, h.text, q)
	if err != nil {
		return nil, fmt.Errorf("textalign: %w", err)
	}
	if err := m.checkHeadAgainstProj(h, feats.Data, logits, q); err != nil {
		return nil, err
	}
	return logits, nil
}

// headProjTol is how far, in COSINE units (logit / Scale), head.onnx may stray from proj.bin on
// the query checkHeadAgainstProj samples. ORT on CPU agrees with Go to ~1e-6 logits; a GPU build
// that runs fp32 MatMul as TF32 is ~1e-3 in cosine. A head exported from a different proj.bin is
// off by tenths: the cosines of two unrelated projections are not close.
const headProjTol = 0.02

// checkHeadAgainstProj recomputes ONE query's exact logits in Go from proj.bin (the cached W row
// per word plus one ‖P f‖ — about 0.1 ms) and compares them with what head.onnx returned.
//
// proj.bin is still read for the other three methods, so a head.onnx left behind after proj.bin
// was retrained would make "exact" and "gated" two different heads with nothing to say so. That
// is the failure this catches, at the cost of one query's worth of the Go path.
func (m *textAlign) checkHeadAgainstProj(h *head, feats []float32, got []float32, q int) error {
	p, c := m.proj, len(h.classes)
	// The query with the largest ‖P f‖: well away from the degenerate row, where a stale head would
	// still agree (both give Bias).
	best, bestN := -1, float32(0)
	for i := 0; i < q && i < 8; i++ {
		if n := p.ProjNorm(feats[i*p.DFeat : (i+1)*p.DFeat]); n > bestN {
			best, bestN = i, n
		}
	}
	if best < 0 {
		return nil // nothing non-degenerate among the sampled queries: nothing to compare
	}
	f := feats[best*p.DFeat : (best+1)*p.DFeat]
	scale := math.Abs(float64(p.Scale))
	for k := 0; k < c; k++ {
		row := h.w[k*p.DFeat : (k+1)*p.DFeat]
		var s float32
		for j, fv := range f {
			s += row[j] * fv
		}
		want := s/bestN + p.Bias
		if d := math.Abs(float64(got[best*c+k]-want)) / scale; d > headProjTol || math.IsNaN(d) {
			return fmt.Errorf("textalign: files.%s disagrees with %s (query %d, %q: logit %.4f, "+
				"proj.bin gives %.4f) — head.onnx was exported from a different projection; re-run "+
				"export_head_onnx.py on this directory's %s",
				roleHead, projFile, best, h.classes[k], got[best*c+k], want, projFile)
		}
	}
	return nil
}
