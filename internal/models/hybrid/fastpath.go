package hybrid

import (
	"fmt"
	"path/filepath"
	"strings"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/models/textalign"
	"visionserve/internal/pipeline"
)

// headFile is an OPTIONAL distilled score head, next to the manifest. Declaring it replaces the
// GroundingDINO pass for out-of-vocabulary words with a matmul on query features RF-DETR has
// already produced — no second detector, no second image pass.
//
// The head is a projection P and a logit calibration (a, b) in the SAME "VSTXALN1" file the
// textalign package already reads, because it is the same object: score = a·cos(P f, ψ(w)) + b.
// Reusing the format means reusing its loader and its validation, and it means a head trained
// for either system can be served by the other.
const headFile = "head.bin"

// roleHead is the SAME head exported as ONNX (`files.head: head.onnx`, built from head.bin by
// models/rfdetr-gdino-fastpath/export_head_onnx.py). When the manifest declares it, the head runs
// as a session owned by lifecycle.Manager on the detector's provider, through the Runner like
// every other role, and head.bin is not read. When it does not, head.bin is scored in Go.
//
// WHY BOTH. The Go path is the reference the ONNX path is differenced against
// (TestHeadONNXMatchesGo*), and it keeps older model directories that ship only head.bin
// working. It is the fallback, not the recommended path: scoring 300 queries against a
// 768x256 projection is ~19 ms of scalar Go per request even folded (ovd-edge FINDINGS §15,
// BUGS_TO_FIX #4), and CLAUDE.md rule 2 puts inference in ONNX Runtime.
//
// The graph's contract (see the export script's docstring for the derivation):
//
//	query_feats  float32 [1, Q, d_feat]    the detector's own query features
//	text_embeds  float32 [C, d_text]       L2-normalised text rows, UNSCALED — per request
//	logits       float32 [1, Q, C]         Scale·cos(P f, ψ) + Bias, RAW (no sigmoid)
//
// Scale lives INSIDE the graph and is applied there exactly once; the Go side passes the text
// rows as the rescorer returns them and adds nothing (bug #7 was Scale applied twice).
//
// textalign's exact head is the same graph over the same VSTXALN1 object, so the contract and the
// call (textalign.RunHeadONNX) live there and are shared.
const (
	roleHead      = "head"
	headInFeats   = textalign.HeadInFeats
	headInText    = textalign.HeadInText
	headOutLogits = textalign.HeadOutLogits
)

// fastPath is the distilled open branch: RF-DETR's own 300 queries, scored against the requested
// unknown words by a head trained to predict what GroundingDINO would have said.
//
// WHY IT EXISTS, in numbers (ovd-edge/docs/FINDINGS.md §10–§13). GroundingDINO answers the open
// branch at 49.54 mAP on the held-out-names protocol for a ~96 ms forward pass. This head plus
// SigLIP rescoring answers it at **46.41** — 3.13 lower — for a matmul (0.06 ms) plus the crops
// the rescorer was already going to take. The trade is ~3 mAP for the second detector.
//
// WHY IT WAS NOT OBVIOUS THAT IT WOULD WORK. FINDINGS §4 measured the rescorer's `conf × P` rule
// at 7.29 on RF-DETR's raw queries against 62.16 on GroundingDINO's — the worst scorer there,
// because "multiplying by confidence only helps when the confidence knows the question". These
// are RF-DETR's boxes but the head's score, and the head was distilled to know the question. The
// product composes: SigLIP is worth **+24.87** on top of this head, against +11.40 on
// GroundingDINO's boxes.
//
// PROPOSAL BUDGET. §11 measures the accuracy-per-crop curve and it is steep at the bottom: 9
// boxes reach 42.86, 22 reach 46.33, and 44 reach 46.41 for twice the crop cost of 22. The
// budget is a manifest field because the right point depends on the crop tower's speed, which
// §13 changed by 1.9× and §14 says is 1.9× from being changed again.
type fastPath struct {
	proj   *textalign.Projection // Go fallback (head.bin); nil when onnx is set
	onnx   bool                  // the head runs as the roleHead session through the Runner
	budget int

	// Set when the router is wired: the manifest (its RF-DETR decode for the head's logits) and
	// the rescorer's text embedder, whose cached SigLIP rows the head scores against.
	cfg  models.Config
	text *pipeline.TextEmbedder
}

// defaultBudget is the knee of §11's curve: 22 boxes reach 99.8 % of what 44 reach, for 57 % of
// the crop cost. 44 is measured to be strictly dominated — it costs a GroundingDINO pass and
// scores 3.13 below one — so it is deliberately not the default.
const defaultBudget = 22

// newFastPath loads the head, or returns nil when the manifest does not declare one.
//
// files.head (ONNX) wins over head.bin: the session itself is created by lifecycle.Manager when
// Roles() lists roleHead, so there is nothing to load here, and a stale head.bin next to it is
// deliberately NOT read — serving one head while validating against another is worse than either.
func newFastPath(cfg models.Config) (*fastPath, error) {
	budget := defaultBudget
	if cfg.MaxDet > 0 && cfg.MaxDet < budget {
		budget = cfg.MaxDet
	}
	if strings.TrimSpace(cfg.Files[roleHead]) != "" {
		return &fastPath{onnx: true, budget: budget}, nil
	}
	path := filepath.Join(cfg.Dir, headFile)
	proj, err := textalign.LoadProjection(path)
	if err != nil {
		if strings.Contains(err.Error(), "no such file") {
			return nil, nil // absent is the normal case: the router keeps using GroundingDINO
		}
		return nil, fmt.Errorf("hybrid: distilled head %s: %w", path, err)
	}
	return &fastPath{proj: proj, budget: budget}, nil
}

// DetectQueries implements pipeline.QueryHead: the unknown words answered from the router's
// RF-DETR pass — its boxes and query features — with no second detector and no second image pass.
// The router rescores the result.
func (fp *fastPath) DetectQueries(c pipeline.Call, pass *pipeline.ClosedPass, words []string) ([]models.Detection, error) {
	if pass.Feats.Data == nil {
		return nil, fmt.Errorf("hybrid: the detector export emitted no query "+
			"features, so the distilled head has no input — %s needs the -qf export", headFile)
	}
	return fp.detect(pass.Boxes, pass.Feats, words, pass.Meta, c.Runner)
}

// detect scores every query against `words` and returns the top-`budget` detections per image.
//
// `boxes` and `feats` are the detector's own outputs, so no second image pass happens. The score
// is sigmoid(a·cos(P f, ψ(w)) + b) — the head's own calibration, NOT the detector's objectness,
// which FINDINGS §4 measures as actively misleading on unseen words (median 0.033 against 0.891
// on trained ones).
func (fp *fastPath) detect(boxes, feats engine.Tensor, words []string,
	meta models.PreprocessMeta, r models.Runner) ([]models.Detection, error) {
	q := int(boxes.Dim(1))
	if !fp.onnx {
		// Checked before the text tower runs: the Go head knows its width from head.bin. The ONNX
		// head's width is in its graph, and ORT's own shape error is wrapped with the same advice.
		if dFeat := int(feats.Dim(-1)); dFeat != fp.proj.DFeat {
			return nil, fmt.Errorf("hybrid: head expects %d-wide query features, detector emits %d — "+
				"the head was trained against a different detector export", fp.proj.DFeat, dFeat)
		}
	}

	text, err := fp.text.Embed(words, r) // the same cached SigLIP text embeddings the rescorer uses
	if err != nil {
		return nil, fmt.Errorf("hybrid: %w", err)
	}
	logits, err := fp.score(feats, text, q, r)
	if err != nil {
		return nil, err
	}

	n := len(words)
	rf, err := newRFDETRWithLabels(fp.cfg, words)
	if err != nil {
		return nil, err
	}
	res, err := rf.Postprocess([]engine.Tensor{boxes, engine.F32(logits, 1, int64(q), int64(n))}, meta)
	if err != nil {
		return nil, err
	}
	if len(res.Detections) > fp.budget {
		res.Detections = res.Detections[:fp.budget]
	}
	return res.Detections, nil
}

// score returns the head's RAW logits, row-major [q][len(text)], for the first q query features.
// Both implementations compute Scale·cos(P f, ψ) + Bias with Scale applied exactly once;
// TestHeadONNXMatchesGo* holds them to each other.
func (fp *fastPath) score(feats engine.Tensor, text [][]float32, q int, r models.Runner) ([]float32, error) {
	dFeat := int(feats.Dim(-1))
	if q <= 0 || dFeat <= 0 || len(feats.Data) < q*dFeat {
		return nil, fmt.Errorf("hybrid: query features %v do not cover %d queries", feats.Shape, q)
	}
	if len(text) == 0 {
		return nil, fmt.Errorf("hybrid: no text embeddings to score the head against")
	}
	if fp.onnx {
		return fp.scoreONNX(feats, text, q, r)
	}
	return fp.scoreGo(feats, text, q)
}

// scoreONNX runs the head as the roleHead session. Nothing is folded or scaled here: the text rows
// go in exactly as the text tower returned them, and Scale and Bias are initializers of the graph.
func (fp *fastPath) scoreONNX(feats engine.Tensor, text [][]float32, q int, r models.Runner) ([]float32, error) {
	logits, err := textalign.RunHeadONNX(r, roleHead, feats, text, q)
	if err != nil {
		return nil, fmt.Errorf("hybrid: %w", err)
	}
	return logits, nil
}

// scoreGo is the head.bin fallback, in scalar Go.
func (fp *fastPath) scoreGo(feats engine.Tensor, text [][]float32, q int) ([]float32, error) {
	dFeat := int(feats.Dim(-1))
	if dFeat != fp.proj.DFeat {
		return nil, fmt.Errorf("hybrid: head expects %d-wide query features, detector emits %d — "+
			"the head was trained against a different detector export", fp.proj.DFeat, dFeat)
	}
	if len(text[0]) != fp.proj.DText {
		return nil, fmt.Errorf("hybrid: head projects to %d dimensions, the text tower emits %d — "+
			"head and tower are different checkpoints", fp.proj.DText, len(text[0]))
	}

	// FOLD the words through P once, instead of projecting every query into text space.
	//
	//     <P f, psi>  =  <f, P^T psi>
	//
	// so a word costs one 768x256 fold and then a 256-long dot product per query, rather than a
	// 768x256 product per query. Combined with ProjNorm below — which gets ||P f|| out of the
	// precomputed Gram P^T P as a 256x256 quadratic form — this is measured at **19.1 ms against
	// 55.9 ms** for the direct form at the real sizes (300 queries, 5 words). Both numbers are
	// scalar Go, which is why files.head (the same head on ONNX Runtime) is preferred.
	//
	// textalign.Fold and textalign.ProjNorm already implement exactly this pair, for exactly this
	// reason, and are reused rather than rewritten.
	folded, err := fp.proj.Fold(text)
	if err != nil {
		return nil, fmt.Errorf("hybrid: folding the vocabulary through the head: %w", err)
	}

	n := len(text)
	logits := make([]float32, q*n)
	for i := 0; i < q; i++ {
		f := feats.Data[i*dFeat : (i+1)*dFeat]
		norm := fp.proj.ProjNorm(f)
		inv := float32(1)
		if norm > 0 {
			inv = 1 / norm
		}
		for c := 0; c < n; c++ {
			w := folded[c*dFeat : (c+1)*dFeat]
			var dot float32
			for j, v := range f {
				dot += w[j] * v
			}
			// RAW logit, not a probability: internal/models/rfdetr/postprocess.go takes
			// "[1, Q, C] BEFORE sigmoid" and applies the sigmoid itself. Passing a probability
			// here would square it and quietly change what every threshold in the manifest
			// means relative to the closed branch.
			// NO Scale here: textalign.Fold already folded it in (`tk := t[k] * p.Scale`).
			// Applying it again multiplied every logit by 10.28 and cost 9.2 mAP in a served
			// run before TestFoldedScoringMatchesDirect was written to catch it.
			logits[i*n+c] = dot*inv + fp.proj.Bias
		}
	}
	return logits, nil
}

func hasName(names []string, want string) bool {
	for _, n := range names {
		if n == want {
			return true
		}
	}
	return false
}

// newRFDETRWithLabels builds a label-only rf-detr sub-model: it owns no session and exists so the
// detector's own box decode, threshold and top-k run over the head's logits. Same construction
// textalign uses for its dual head.
func newRFDETRWithLabels(cfg models.Config, labels []string) (models.Model, error) {
	c := cfg
	c.Labels = append([]string{}, labels...)
	// The head has its OWN calibration and must not inherit the closed branch's threshold. The
	// first served run of this path set conf_threshold to 0.001 in the manifest so the head's
	// scores would survive, and that leaked into RF-DETR's supervised head: the 17-name control
	// moved 82.62 -> 84.79, which is a control moving and therefore a bug, not a result.
	// Selection for the open branch happens at the crop budget instead.
	c.ConfThresh = 0
	c.MaxDet = 300
	base, err := models.New("rf-detr", c)
	if err != nil {
		return nil, fmt.Errorf("hybrid: rf-detr sub-model for the head: %w", err)
	}
	rf, ok := base.(models.Model)
	if !ok {
		return nil, fmt.Errorf("hybrid: rf-detr did not yield a plain Model")
	}
	return rf, nil
}

// hasFastPath reports whether a distilled head is wired AND the rescorer it depends on is too.
// The head produces a ranking; FINDINGS §10 measures it at 21.53 mAP alone and 46.41 with SigLIP
// rescoring on top, so shipping the head without the rescorer would serve less than half of what
// the configuration is worth. Requiring both is not a convenience — it is the measured
// difference between a usable branch and an unusable one.
func (m *hybrid) hasFastPath() bool { return m.fp != nil && m.rs != nil }
