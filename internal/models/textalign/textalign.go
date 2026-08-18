// Package textalign implements a TEXT-ALIGNED ("head B") open-vocabulary detection head
// on top of a FROZEN RF-DETR detector (Apache-2.0 + MIT CLIP text tower — both permissive,
// no AGPL, no Python at runtime; see CLAUDE.md).
//
// # The idea
//
// The frozen detector emits, per object query i, a 256-d decoder feature f_i (the tensor
// its own class head reads) — exposed as the extra ONNX output `query_feats` by
// models/rfdetr-small-etri-qf. Head B is ONE trained matrix P (d_text×256) that maps f_i into
// CLIP text space, so a class score is a cosine:
//
//	logit_ic = a · ⟨ t̂_c , P f_i / ‖P f_i‖ ⟩ + b
//
// with t̂_c the L2-normalised CLIP text embedding of class c. Because ⟨t̂_c, P f_i⟩ is
// bilinear, at deploy time — when the vocabulary is known — the text tower and P FOLD into
// a single small matrix
//
//	W = a · T̂ P        (shape [C, 256], exactly the role of RF-DETR's class_embed)
//	logit_ic = ⟨W_c, f_i⟩ / ‖P f_i‖ + b
//
// so open-vocabulary detection costs one extra [300,256]×[256,C] matmul at runtime, and
// changing the vocabulary swaps a small matrix — NO ONNX re-export.
//
// # How it fits VisionServe (zero core changes)
//
// This is a PipelineModel (it is prompt-driven and chains two sessions), composing existing
// models instead of duplicating them — the same pattern as internal/models/hybrid:
//
//   - role "rfdetr" → models/rfdetr-small-etri-qf/model.onnx. Boxes, letterbox mapping and
//     the NMS-free DETR decode are reused from internal/models/rfdetr via a sub-Model, by
//     handing its Postprocess the detector's OWN box tensor plus OUR class logits. So
//     Detection.BBox is produced by the identical code path as the bare detector and is in
//     ORIGINAL image coordinates, [x,y,w,h].
//   - role "text" → models/clip-text/model.onnx (CLIP ViT-B/32 text tower) + the pure-Go BPE
//     tokenizer from internal/models/clip. Used ONLY on a vocabulary cache miss.
//   - P itself is a 512 KB side-car (proj.bin) next to the manifest — see proj.go. Weights
//     that are not an ONNX graph do not belong in the `files:` map (lifecycle would try to
//     open them as a session), and the manifest parser needs no new field for them.
//
// The heavy sessions stay owned by lifecycle.Manager; this model only calls them by role
// through the Runner.
//
// # Requests
//
//	--prompt "cup. water bottle. cola can."   the vocabulary (GroundingDINO/CLIP convention).
//	                                          Empty prompt → the manifest's labels file.
//	--method exact | folded | gated           exact (default) keeps the ‖P f‖ normalisation;
//	                                          folded drops it (a literal linear head);
//	                                          gated names with the folded head but selects
//	                                          boxes with the detector's own objectness.
//
// Pick `gated` when the detector export still carries its own class head: folding is
// provably exact for naming and provably wrong for ranking queries against each other, and
// `gated` is the split that keeps only the exact half. See gated.go for the algebra.
package textalign

import (
	"fmt"
	"image"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/models/clip"
)

func init() {
	models.Register("rfdetr-textalign", New)
}

const (
	// roleDetector is the frozen detector exposing `query_feats` (rfdetr-small-etri-qf).
	roleDetector = "rfdetr"
	// roleText is the CLIP text tower, consulted only when a vocabulary is not cached.
	roleText = "text"

	// projFile is the trained projection side-car, read from the model directory.
	projFile = "proj.bin"
	// templatesFile optionally overrides the prompt templates (one per line, "{}" =
	// the class name). It is part of the head-B CONTRACT: T̂ must be built with the same
	// templates the head was trained against, so it ships next to proj.bin.
	templatesFile = "templates.txt"
	// templatePlaceholder is what applyTemplates substitutes the class name for.
	templatePlaceholder = "{}"

	// maxVocabCache bounds the number of compiled vocabularies kept in memory
	// (C×256 floats each — kilobytes; the bound exists to stop unbounded growth under
	// adversarial per-request prompts, not to save memory).
	maxVocabCache = 32
)

// defaultTemplates is the fallback prompt ensemble when templates.txt is absent.
var defaultTemplates = []string{"a photo of a {}."}

type textAlign struct {
	cfg  models.Config
	proj *Projection
	tok  textTokenizer
	tmpl []string

	// base is the rf-detr sub-model built from the manifest labels; it also serves
	// ExplainPreprocess and the no-prompt (fixed vocabulary) path.
	base      models.Model
	baseVocab []string

	mu    sync.RWMutex
	cache map[string]*head
	order []string // insertion order, for FIFO eviction of the vocabulary cache
}

// New builds the head. The manifest's input/postprocess/labels block MUST carry the
// DETECTOR's values (512×512, letterbox, ImageNet normalize, its labels file) because the
// rf-detr sub-model is created from the same Config.
func New(cfg models.Config) (models.Base, error) {
	if cfg.Files[roleDetector] == "" || cfg.Files[roleText] == "" {
		return nil, fmt.Errorf("textalign: manifest must declare files.%s (a detector exporting query_feats) and files.%s (the CLIP text tower)",
			roleDetector, roleText)
	}
	if cfg.Dir == "" {
		return nil, fmt.Errorf("textalign: model directory unknown, cannot load %s", projFile)
	}

	proj, err := LoadProjection(filepath.Join(cfg.Dir, projFile))
	if err != nil {
		return nil, err
	}

	// The tokenizer assets live next to the text weights (files.text is "../clip-text/model.onnx"
	// or "../siglip-text/model.onnx"), exactly like hybrid resolves GroundingDINO's vocab.txt.
	// WHICH tokenizer is decided by what that directory contains — see tokenizer.go.
	tok, err := loadTextTokenizer(filepath.Dir(cfg.Files[roleText]))
	if err != nil {
		return nil, err
	}

	tmpl, err := loadTemplates(filepath.Join(cfg.Dir, templatesFile))
	if err != nil {
		return nil, err
	}

	base, err := newRFDETR(cfg, cfg.Labels)
	if err != nil {
		return nil, err
	}

	return &textAlign{
		cfg:       cfg,
		proj:      proj,
		tok:       tok,
		tmpl:      tmpl,
		base:      base,
		baseVocab: baseVocab(cfg.Labels),
		cache:     map[string]*head{},
	}, nil
}

// newRFDETR builds an rf-detr sub-model carrying `labels` as its class names. It owns no
// session — it is pure pre/postprocess, so building one per vocabulary is free.
func newRFDETR(cfg models.Config, labels []string) (models.Model, error) {
	sub := cfg
	sub.Labels = labels
	b, err := models.New("rf-detr", sub)
	if err != nil {
		return nil, fmt.Errorf("textalign: rf-detr sub-model: %w", err)
	}
	m, ok := b.(models.Model)
	if !ok {
		return nil, fmt.Errorf("textalign: rf-detr did not yield a plain Model")
	}
	return m, nil
}

// baseVocab is the manifest labels file minus RF-DETR's "N/A" padding class — the
// vocabulary used when a request carries no prompt (the "fixed vocabulary at deploy" case).
func baseVocab(labels []string) []string {
	out := make([]string, 0, len(labels))
	for _, l := range labels {
		if s := strings.TrimSpace(l); s != "" && !strings.EqualFold(s, "n/a") {
			out = append(out, s)
		}
	}
	return normalizeVocab(out)
}

// loadTemplates reads templates.txt (one template per line, "#" comments allowed); a
// missing file yields defaultTemplates. Every template must contain "{}".
func loadTemplates(path string) ([]string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return defaultTemplates, nil
		}
		return nil, fmt.Errorf("textalign: cannot read %s: %w", path, err)
	}
	var out []string
	for _, line := range strings.Split(string(raw), "\n") {
		s := strings.TrimSpace(line)
		if s == "" || strings.HasPrefix(s, "#") {
			continue
		}
		if !strings.Contains(s, templatePlaceholder) {
			return nil, fmt.Errorf("textalign: template %q in %s has no %s placeholder", s, path, templatePlaceholder)
		}
		out = append(out, s)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("textalign: %s contains no templates", path)
	}
	return out, nil
}

func (m *textAlign) Name() string      { return m.cfg.Name }
func (m *textAlign) Task() models.Task { return models.TaskOpenVocab }

// Roles: the frozen detector + the CLIP text tower.
func (m *textAlign) Roles() []string { return []string{roleDetector, roleText} }

// ExplainPreprocess implements models.ExplainPreprocessor so /api/explain works on the
// detector role (the qf export still carries cross_attn_weights).
func (m *textAlign) ExplainPreprocess(img image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	return m.base.Preprocess(img)
}

// Infer runs the full head: compile the requested vocabulary into W (cached), run the
// frozen detector once, score every query against W, and hand the detector's own boxes +
// our logits to RF-DETR's postprocess.
func (m *textAlign) Infer(img image.Image, prompt models.Prompt, r models.Runner) (models.Result, error) {
	mode, err := parseMethod(prompt.Method)
	if err != nil {
		return models.Result{}, err
	}

	classes := normalizeVocab(clip.SplitPhrases(prompt.Text))
	if len(classes) == 0 {
		classes = m.baseVocab
	}
	if len(classes) == 0 {
		return models.Result{}, fmt.Errorf("textalign: no vocabulary — pass one as the prompt, e.g. --prompt \"cup. cola can.\"")
	}

	h, err := m.headFor(classes, r)
	if err != nil {
		return models.Result{}, err
	}

	in, meta, err := m.base.Preprocess(img)
	if err != nil {
		return models.Result{}, err
	}
	inName := firstName(r.InputNames(roleDetector), m.base.InputName())
	if inName == "" {
		return models.Result{}, fmt.Errorf("textalign: detector session %q has no input name", roleDetector)
	}
	outs, err := r.Run(roleDetector, map[string]engine.Tensor{inName: in})
	if err != nil {
		return models.Result{}, fmt.Errorf("textalign: detector inference: %w", err)
	}

	boxes, feats, err := splitOutputs(outs, m.proj.DFeat)
	if err != nil {
		return models.Result{}, err
	}
	if mode == modeGated || mode == modeDual {
		cls, err := classLogits(outs, boxes, m.proj.DFeat, len(m.cfg.Labels))
		if err != nil {
			return models.Result{}, err
		}
		if mode == modeDual {
			return m.decodeDual(h, boxes, cls, feats, meta, claimThreshold(prompt.ClaimThresh))
		}
		return m.decodeGated(h, boxes, cls, feats, meta)
	}
	return m.decode(h, boxes, feats, meta, mode.normalize())
}

// decode scores every object query against the compiled vocabulary and turns the result
// into detections.
//
// The scoring is ours; the geometry is NOT. The detector's own box tensor and OUR class
// logits are handed to RF-DETR's postprocess, which maps boxes back to ORIGINAL image
// coordinates ([x,y,w,h] via PreprocessMeta), thresholds sigmoid(logit), sorts and cuts to
// max_detections — NMS-free. That is why boxes are bit-identical to the bare detector's.
func (m *textAlign) decode(h *head, boxes, feats engine.Tensor, meta models.PreprocessMeta, normalize bool) (models.Result, error) {
	q := int(boxes.Dim(1))
	logits, err := h.logits(m.proj, feats.Data, q, normalize)
	if err != nil {
		return models.Result{}, err
	}
	return h.rf.Postprocess(
		[]engine.Tensor{boxes, engine.F32(logits, 1, int64(q), int64(len(h.classes)))}, meta)
}

// parseMethod maps the per-request `method` option to a scoring mode. See gated.go for why
// "folded" and "gated" differ: they compute the same names and select different boxes.
func parseMethod(method string) (scoreMode, error) {
	switch strings.ToLower(strings.TrimSpace(method)) {
	case "", "exact", "cosine":
		return modeExact, nil
	case "folded", "linear":
		return modeFolded, nil
	case "gated", "split":
		return modeGated, nil
	case "dual", "twohead":
		return modeDual, nil
	default:
		return modeExact, fmt.Errorf("textalign: unknown method %q (want \"exact\", \"folded\", \"gated\" or \"dual\")", method)
	}
}

// splitOutputs picks the box tensor and the query-feature tensor out of the detector's
// outputs BY SHAPE (names differ between exports): boxes = last dim 4, query_feats = last
// dim DFeat with the same query count. `labels` (the frozen class head) and
// `cross_attn_weights` are ignored — head B replaces the former for naming. modeGated wants
// `labels` back for SELECTION; it asks for it separately via classLogits.
func splitOutputs(outs []engine.Tensor, dFeat int) (boxes, feats engine.Tensor, err error) {
	for _, t := range outs {
		if t.Dim(-1) == 4 && boxes.Data == nil {
			boxes = t
		}
	}
	if boxes.Data == nil {
		return boxes, feats, fmt.Errorf("textalign: no detector output has a last dimension of 4 (boxes)")
	}
	q := boxes.Dim(1)
	for _, t := range outs {
		if len(t.Shape) == 3 && t.Dim(-1) == int64(dFeat) && t.Dim(1) == q && feats.Data == nil {
			feats = t
		}
	}
	if feats.Data == nil {
		return boxes, feats, fmt.Errorf(
			"textalign: no detector output of shape [1,%d,%d] (query_feats) — the manifest's files.%s must point at an export that exposes it, e.g. models/rfdetr-small-etri-qf/model.onnx",
			q, dFeat, roleDetector)
	}
	return boxes, feats, nil
}

// classLogits finds the detector's OWN class head among its outputs: the [1,Q,C] tensor that
// is neither the boxes (C=4) nor query_feats (C=DFeat). Only modeGated needs it.
//
// It is matched against the manifest's label count rather than "whatever is left", so a
// mismatch between the export and labels.txt is reported here instead of silently scoring
// the wrong columns.
func classLogits(outs []engine.Tensor, boxes engine.Tensor, dFeat, nLabels int) (engine.Tensor, error) {
	q := boxes.Dim(1)
	for _, t := range outs {
		if len(t.Shape) == 3 && t.Dim(1) == q && t.Dim(-1) == int64(nLabels) && int(t.Dim(-1)) != dFeat {
			return t, nil
		}
	}
	return engine.Tensor{}, fmt.Errorf(
		"textalign: method \"gated\" needs the detector's own class head, but no output has shape [1,%d,%d] (the manifest lists %d labels) — use method \"exact\" or \"folded\" with this export",
		q, nLabels, nLabels)
}

// headFor returns the compiled head for a vocabulary, embedding the class names through
// the CLIP text tower on a cache miss (see README: one clip-text call per NEW vocabulary,
// never per request).
func (m *textAlign) headFor(classes []string, r models.Runner) (*head, error) {
	key := vocabKey(m.tmpl, classes)
	m.mu.RLock()
	h := m.cache[key]
	m.mu.RUnlock()
	if h != nil {
		return h, nil
	}

	rows, err := m.embedVocab(classes, r)
	if err != nil {
		return nil, err
	}
	w, err := m.proj.Fold(rows)
	if err != nil {
		return nil, err
	}
	rf, err := newRFDETR(m.cfg, classes)
	if err != nil {
		return nil, err
	}
	h = &head{classes: classes, w: w, rf: rf}

	m.mu.Lock()
	defer m.mu.Unlock()
	if existing := m.cache[key]; existing != nil {
		return existing, nil // lost a race; both are identical, keep the published one
	}
	m.put(key, h)
	return h, nil
}

// put inserts a compiled head, evicting the oldest entry when the cache is full.
// The caller must hold m.mu for writing.
func (m *textAlign) put(key string, h *head) {
	if len(m.order) >= maxVocabCache {
		delete(m.cache, m.order[0])
		m.order = m.order[1:]
	}
	m.cache[key] = h
	m.order = append(m.order, key)
}

// embedVocab returns one L2-normalised CLIP text embedding per class: every class is
// expanded through the prompt templates, all rows go through the text tower in ONE batched
// call, and the per-class rows are averaged and re-normalised (prompt ensembling).
//
// It tokenizes with internal/models/clip's tokenizer rather than calling the clip-text
// model's Infer, because that entrypoint splits its prompt on "." — which would strip the
// trailing period of a template like "a photo of a {}.".
func (m *textAlign) embedVocab(classes []string, r models.Runner) ([][]float32, error) {
	texts := applyTemplates(m.tmpl, classes)
	ids, err := m.tok.EncodeBatch(texts)
	if err != nil {
		return nil, err
	}

	name := "input_ids"
	if in := r.InputNames(roleText); len(in) > 0 {
		found := false
		for _, n := range in {
			if n == name {
				found = true
				break
			}
		}
		if !found {
			name = in[0]
		}
	}
	outs, err := r.Run(roleText, map[string]engine.Tensor{
		name: engine.I64(ids, int64(len(texts)), int64(m.tok.ContextLength())),
	})
	if err != nil {
		return nil, fmt.Errorf("textalign: text tower inference: %w", err)
	}
	if len(outs) == 0 || len(outs[0].Shape) != 2 {
		return nil, fmt.Errorf("textalign: text tower returned an unexpected output (want [N,D])")
	}
	t := outs[0]
	n, dim := int(t.Shape[0]), int(t.Shape[1])
	if n != len(texts) || dim != m.proj.DText || len(t.Data) < n*dim {
		return nil, fmt.Errorf("textalign: clip-text returned [%d,%d], expected [%d,%d] (projection d_text)",
			n, dim, len(texts), m.proj.DText)
	}
	embs := make([][]float32, n)
	for i := 0; i < n; i++ {
		row := make([]float32, dim)
		copy(row, t.Data[i*dim:(i+1)*dim])
		embs[i] = l2Normalize(row)
	}
	return averageTemplates(embs, len(classes), len(m.tmpl))
}

func firstName(names []string, fallback string) string {
	if len(names) > 0 {
		return names[0]
	}
	return fallback
}
