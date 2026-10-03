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
//     tokenizer from internal/models/clip. Used ONLY for words not embedded yet (per-word cache).
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
	"math"
	"os"
	"path/filepath"
	"strings"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/models/clip"
	"visionserve/internal/models/detr"
	"visionserve/internal/models/promptens"
	"visionserve/internal/pipeline"
	"visionserve/internal/vision/util"
)

func init() {
	models.Register("rfdetr-textalign", New)
}

const (
	// roleDetector is the frozen detector exposing `query_feats` (rfdetr-small-etri-qf).
	roleDetector = "rfdetr"
	// roleText is the text tower (CLIP or SigLIP), consulted only for words not cached yet.
	roleText = "text"

	// projFile is the trained projection side-car, read from the model directory.
	projFile = "proj.bin"
	// templatesFile optionally overrides the prompt templates (one per line, "{}" =
	// the class name). It is part of the head-B CONTRACT: T̂ must be built with the same
	// templates the head was trained against, so it ships next to proj.bin.
	templatesFile = "templates.txt"
	// templatePlaceholder is what the prompt ensemble substitutes the class name for.
	templatePlaceholder = promptens.Placeholder
)

// defaultTemplates is the fallback prompt ensemble when templates.txt is absent.
var defaultTemplates = []string{"a photo of a {}."}

type textAlign struct {
	cfg  models.Config
	proj *Projection

	// text embeds the requested words through the prompt ensemble and the text tower, with a
	// bounded per-WORD cache: one text-tower call per NEW word, never per request. Head B and
	// the crop head read the same rows.
	text *pipeline.TextEmbedder
	// folds caches each word's row of the deploy-time matrix W = a·T̂P, keyed like text's cache.
	// Fold is row-separable, so a W assembled from cached rows is bit-identical to folding the
	// whole vocabulary at once — and a request pays for the fold of its new words only (a dense
	// row is ~0.4 ms of Go: 9 ms for 22 words, 34 ms for 78, more than the exact head itself).
	folds *pipeline.LRU[foldedRow]
	// namer is the crop head (files.crop): nil when the manifest wires none.
	namer *pipeline.CropNamer

	// base is the rf-detr sub-model built from the manifest labels; it also serves
	// ExplainPreprocess and the no-prompt (fixed vocabulary) path.
	base      models.Model
	baseVocab []string
}

// foldedRow is one word's row of W, with the embedding row it was folded from: a hit is only
// used while the text cache still serves that very row, so W can never disagree with T̂.
type foldedRow struct {
	src *float32
	w   []float32
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
	// WHICH tokenizer is decided by what that directory contains (pipeline.LoadTextTokenizer):
	// the head was distilled against one teacher and must be served with that teacher's.
	tok, err := pipeline.LoadTextTokenizer(filepath.Dir(cfg.Files[roleText]))
	if err != nil {
		return nil, fmt.Errorf("textalign: %w", err)
	}

	tmpl, err := loadTemplates(filepath.Join(cfg.Dir, templatesFile))
	if err != nil {
		return nil, err
	}

	base, err := newRFDETR(cfg, cfg.Labels)
	if err != nil {
		return nil, err
	}

	m := newTextAlign(cfg, proj, tok, tmpl)
	m.base, m.baseVocab = base, baseVocab(cfg.Labels)
	warnDeadOpenHead(cfg)
	return m, nil
}

// newTextAlign wires the towers' stages around a loaded projection: the text embedder (checked
// against the projection's d_text), the per-word fold cache, and the crop namer when files.crop
// is declared.
func newTextAlign(cfg models.Config, proj *Projection, tok pipeline.TextTokenizer, tmpl []string) *textAlign {
	m := &textAlign{
		cfg:   cfg,
		proj:  proj,
		text:  pipeline.NewTextEmbedder(roleText, tok, tmpl, proj.DText),
		folds: pipeline.NewLRU[foldedRow](pipeline.MaxCachedWords),
	}
	if m.hasCropHead() {
		m.namer = &pipeline.CropNamer{CropRole: roleCrop, Text: m.text, Temp: cropTemp, Floor: cropNameFloor}
	}
	return m
}

// warnDeadOpenHead catches a configuration whose open head can never fire.
//
// For a prompt containing the whole base vocabulary — the ordinary open-vocabulary case — the
// closed head's claim score IS the objectness, because both are the max over the same class
// logits. So every detection that survives `conf_threshold` automatically clears any
// `claim_threshold` at or below it, the closed head takes all of them, and the crop head is
// unreachable BY CONSTRUCTION rather than by tuning.
//
// Measured: at conf_threshold 0.35 with the default claim 0.15, the crop model returned a
// detection stream byte-identical to the plain closed detector. The feature was wired, loaded,
// paid for in VRAM, and dead. Nothing in the manifest hints at the coupling, because the two
// knobs are set in different places and neither mentions the other.
func warnDeadOpenHead(cfg models.Config) {
	if strings.TrimSpace(cfg.Files[roleCrop]) == "" || cfg.ConfThresh <= 0 {
		return
	}
	claim := 1 / (1 + math.Exp(-float64(dualClaimThresh))) // the default, as a probability
	if claim > cfg.ConfThresh {
		return
	}
	fmt.Fprintf(os.Stderr,
		"textalign: WARNING %s has a crop head, but claim_threshold (%.3g) <= conf_threshold "+
			"(%.3g). For a prompt covering the base vocabulary the claim score IS the objectness, "+
			"so everything that survives conf_threshold is claimed by the closed head and the open "+
			"head never runs. Lower conf_threshold below %.3g, or raise claim_threshold per "+
			"request.\n",
		cfg.Name, claim, cfg.ConfThresh, claim)
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
	return promptens.Load(path, defaultTemplates)
}

func (m *textAlign) Name() string      { return m.cfg.Name }
func (m *textAlign) Task() models.Task { return models.TaskOpenVocab }

// Roles: the frozen detector + the CLIP text tower.
func (m *textAlign) Roles() []string {
	roles := []string{roleDetector, roleText}
	if m.hasCropHead() {
		roles = append(roles, roleCrop)
	}
	return roles
}

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
		return models.Result{}, models.BadPrompt(fmt.Errorf("textalign: no vocabulary — pass one as the prompt, e.g. --prompt \"cup. cola can.\""))
	}

	h, err := m.headFor(classes, r)
	if err != nil {
		return models.Result{}, err
	}

	in, meta, err := m.base.Preprocess(img)
	if err != nil {
		return models.Result{}, err
	}
	inName := util.FirstName(r.InputNames(roleDetector), m.base.InputName())
	if inName == "" {
		return models.Result{}, fmt.Errorf("textalign: detector session %q has no input name", roleDetector)
	}
	outs, err := r.Run(roleDetector, map[string]engine.Tensor{inName: in})
	if err != nil {
		return models.Result{}, fmt.Errorf("textalign: detector inference: %w", err)
	}

	o, err := detectorOutputs(outs, m.proj.DFeat, len(m.cfg.Labels))
	if err != nil {
		return models.Result{}, err
	}
	boxes, feats := o.Boxes, o.Feats
	if mode == modeGated || mode == modeDual {
		cls, err := classHead(o, len(m.cfg.Labels))
		if err != nil {
			return models.Result{}, err
		}
		if mode == modeDual {
			claim := claimThreshold(prompt.ClaimThresh)
			if m.hasCropHead() {
				return m.decodeDualCrop(h, img, boxes, cls, meta, claim, prompt.CropTemp, r)
			}
			return m.decodeDual(h, boxes, cls, feats, meta, claim)
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

// detectorOutputs identifies the detector's boxes, class head and query features with
// detr.SplitOutputs — the rule the router shares — and insists on query_feats, which every mode
// scores. `labels` (the frozen class head) is only read back by modeGated/modeDual, for
// SELECTION, through classHead; `cross_attn_weights` is ignored.
func detectorOutputs(outs []engine.Tensor, dFeat, nLabels int) (detr.Outputs, error) {
	o, err := detr.SplitOutputs(outs, nLabels, dFeat)
	if err != nil {
		return o, fmt.Errorf("textalign: %w", err)
	}
	if o.Feats.Data == nil {
		return o, fmt.Errorf(
			"textalign: no detector output of shape [1,%d,%d] (query_feats) — the manifest's files.%s must point at an export that exposes it, e.g. models/rfdetr-small-etri-qf/model.onnx",
			o.Boxes.Dim(1), dFeat, roleDetector)
	}
	return o, nil
}

// classHead returns the detector's OWN class head — the [1,Q,C] output with C = the manifest's
// label count — which modeGated and modeDual select with.
//
// It is matched against the label count rather than taken as "whatever is left", so a mismatch
// between the export and labels.txt is reported here instead of silently scoring the wrong
// columns.
func classHead(o detr.Outputs, nLabels int) (engine.Tensor, error) {
	q, cls := o.Boxes.Dim(1), o.Logits
	if cls.Data == nil || len(cls.Shape) != 3 || cls.Dim(1) != q || cls.Dim(-1) != int64(nLabels) {
		return engine.Tensor{}, fmt.Errorf(
			"textalign: method \"gated\" needs the detector's own class head, but no output has shape [1,%d,%d] (the manifest lists %d labels) — use method \"exact\" or \"folded\" with this export",
			q, nLabels, nLabels)
	}
	return cls, nil
}

// headFor compiles the head for a vocabulary: the words' text rows (embedded through the text
// tower only for words not cached yet — see README: one text-tower call per NEW word, never per
// request), W = a·T̂P assembled from per-word folded rows, and an rf-detr sub-model carrying the
// words as its labels (pure pre/postprocess, no session: building one is free).
func (m *textAlign) headFor(classes []string, r models.Runner) (*head, error) {
	rows, err := m.text.Embed(classes, r)
	if err != nil {
		return nil, fmt.Errorf("textalign: %w", err)
	}
	w, err := m.fold(classes, rows)
	if err != nil {
		return nil, err
	}
	rf, err := newRFDETR(m.cfg, classes)
	if err != nil {
		return nil, err
	}
	return &head{classes: classes, w: w, rf: rf}, nil
}

// fold assembles W [C, DFeat] row by row: Projection.Fold computes each row from its own text
// row only, so folding the words one at a time (and caching the rows) is bit-identical to
// folding the vocabulary at once.
func (m *textAlign) fold(classes []string, rows [][]float32) ([]float32, error) {
	d := m.proj.DFeat
	w := make([]float32, len(rows)*d)
	for c, row := range rows {
		if len(row) == 0 {
			return nil, fmt.Errorf("textalign: no text embedding for %q", classes[c])
		}
		key := m.text.Key(classes[c])
		f, ok := m.folds.Get(key)
		if !ok || f.src != &row[0] {
			fr, err := m.proj.Fold([][]float32{row})
			if err != nil {
				return nil, err
			}
			f = foldedRow{src: &row[0], w: fr}
			m.folds.Set(key, f) // new word, or re-embedded since this row was folded
		}
		copy(w[c*d:(c+1)*d], f.w)
	}
	return w, nil
}
