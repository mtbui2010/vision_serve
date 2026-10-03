// Package groundingdino implements GroundingDINO (Apache-2.0) for text-prompt-driven
// open-vocabulary detection — a fully free community feature (no AGPL, no Python at
// runtime; see CLAUDE.md).
//
// GroundingDINO is a single ONNX graph but it is PROMPTED (it needs the text), so it is a
// PipelineModel (role "model") that drives its own inference via the Runner.
//
// Verified I/O contract (real runs):
//
//	inputs:  pixel_values   [1,3,800,800] float32 (ImageNet-normalized, plain 800x800 squash)
//	         pixel_mask     [1,800,800]   int64   (all ones)
//	         input_ids      [1,L]         int64   (BERT WordPiece, [CLS]..[SEP])
//	         attention_mask [1,L]         int64   (all ones)
//	         token_type_ids [1,L]         int64   (all zeros)
//	outputs: logits     [1,900,256] float32
//	         pred_boxes [1,900,4]   float32 (cxcywh, normalized 0..1)
//
// Postprocess follows HF post_process_grounded_object_detection — sigmoid(logits),
// box_threshold filters queries — with ONE deliberate difference: the prompt is split into
// "."-separated phrases and each query is assigned to its single best-scoring phrase.
// HF instead concatenates every token above text_threshold across the whole prompt, which
// labels a query "chair tv vase bear" once the prompt holds several classes. Here
// text_threshold gates whether the winning phrase is assignable at all.
//
// # KNOWN DEFECT in the COMMUNITY ONNX export (why a per-phrase path still exists)
//
// FIXED by re-export: models/grounding-dino/model-fixedmask.onnx has a correct, dynamic text
// mask, and Detect scores the whole prompt in ONE pass on it. The per-phrase path stays for
// users who still have the old model.onnx on disk; New probes the weights with
// SupportsJointTextPass and picks the regime. The analysis below is kept because it is what
// the probe keys on.
//
// GroundingDINO's BERT text branch needs a BLOCK-DIAGONAL text self-attention mask so the
// tokens of one class phrase cannot attend to another phrase's tokens. HF builds it inside
// the model from input_ids (generate_masks_with_special_tokens_and_transfer_map), so the
// ONNX graph has to rebuild it too — it is NOT a graph input.
//
// onnx-community/grounding-dino-tiny-ONNX (traced with torch.onnx.export from transformers
// 4.48.0.dev0, whose implementation of that function is a PYTHON for-loop over
// torch.nonzero(special_tokens_mask)) BAKED THE LOOP TRIP COUNT INTO THE GRAPH. Verified by
// reading the graph: input_ids feeds Equal(101)/Equal(102)/Equal(1012)/Equal(1029) -> Or
// chain -> NonZero -> Transpose -> Gather(axis 0, index 0), Gather(index 1), Gather(index 2)
// — exactly THREE hard-coded iterations, feeding 6 ScatterND nodes (mask + position_ids per
// iteration). The export prompt held one class ([CLS] . [SEP] = 3 special tokens).
//
// Consequence at runtime: only the FIRST phrase gets its attention block. Every later phrase
// keeps just the EyeLike identity (each token attends to itself alone) and position_id 0.
// Confirmed numerically by extracting the mask subgraph and comparing to HF: for "cat." the
// masks are identical, for "cat. remote." the ONNX loses the second block. End to end on
// demo/images/000000000139.jpg, HF PyTorch returns the same 15 detections for every
// reordering of a 12-class prompt while the raw joint ONNX run returns 4 / 0 / 0 / 3.
//
// Nothing on the Go side can repair that graph: the only text inputs are input_ids /
// attention_mask / token_type_ids, all [1,L], and attention_mask is used solely as the
// padding mask (Cast -> Not -> Tile into the fusion/decoder attention) — it can never
// reconstruct a block-diagonal [L,L] mask. On such weights Detect therefore runs the session
// ONCE PER "."-SEPARATED PHRASE, the one regime the export handles exactly right, and
// concatenates the detections: N passes for N classes, but order-stable and correct.
//
// The re-export (models/grounding-dino/README.md, scripts in the export notes) replaces
// generate_masks_with_special_tokens_and_transfer_map with a vectorized, data-independent
// equivalent — isin -> Equal/Or chain, cummax/cummin -> LxL compare + ReduceMax/ReduceMin,
// torch.eye -> (i==j) — so the mask follows the dynamic sequence_length. Its logits match HF
// PyTorch to ~1e-4 in sigmoid space and it is order-stable across prompt permutations.
package groundingdino

import (
	"fmt"
	"image"
	"path/filepath"
	"strings"
	"sync"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/vision/geom"
	"visionserve/internal/vision/util"
	"visionserve/pkg/api"
)

func init() {
	models.Register("grounding-dino", New)
}

// PipelineMu serializes whole GroundingDINO-based pipelines end-to-end (grounding-dino,
// grounded-sam, grasp-gd). The GroundingDINO graph plus the chained MobileSAM decoder pool
// issue many concurrent cgo ONNX Runtime calls; under concurrent request load this manifests
// as sporadic request errors and (without async-preemption disabled) the Go-runtime fatal
// "non-Go code set up signal handler without SA_ONSTACK" from the ORT/CUDA native layer.
// These pipelines are heavy and low-QPS, so we trade cross-request concurrency for correctness
// by running one whole pipeline at a time. Pipelines that do NOT use GroundingDINO
// (e.g. grasp-rfdetr) are unaffected and stay fully concurrent — and so are the requests of a
// mixed pipeline that never reach GroundingDINO: the rfdetr-gdino router takes the lock just
// before its GroundingDINO pass and holds it to the end, so closed-set requests stay concurrent.
var PipelineMu sync.Mutex

const roleModel = "model"

type groundingDINO struct {
	cfg models.Config
	tok *Tokenizer
	// joint is true when the weights on disk rebuild the text mask for any number of class
	// phrases, so the whole prompt can be scored in one pass. Probed once, at load.
	joint bool
}

// New loads the tokenizer (vocab.txt in the model directory) once, probes which text-mask
// regime the weights implement, and returns the model.
func New(cfg models.Config) (models.Base, error) {
	vocabPath := filepath.Join(cfg.Dir, "vocab.txt")
	tok, err := LoadTokenizer(vocabPath)
	if err != nil {
		return nil, err
	}
	// A probe failure must NOT fail the load: JointTextPassOrSafe answers false in that case,
	// which selects the safe per-phrase path.
	return &groundingDINO{cfg: cfg, tok: tok, joint: JointTextPassOrSafe(cfg.Files[roleModel])}, nil
}

func (m *groundingDINO) Name() string      { return m.cfg.Name }
func (m *groundingDINO) Task() models.Task { return models.TaskOpenVocab }

// Roles: a single session keyed "model".
func (m *groundingDINO) Roles() []string { return []string{roleModel} }

// Infer runs the full open-vocab detection pipeline for the text prompt.
func (m *groundingDINO) Infer(img image.Image, prompt models.Prompt, r models.Runner) (models.Result, error) {
	PipelineMu.Lock()
	defer PipelineMu.Unlock()
	if strings.TrimSpace(prompt.Text) == "" {
		return models.Result{}, fmt.Errorf("grounding-dino requires a text prompt, e.g. --prompt \"cat. remote.\"")
	}
	boxThresh, textThresh := Thresholds(m.cfg.ConfThresh, m.cfg.TextThresh, prompt)

	run := func(inputs map[string]engine.Tensor) ([]engine.Tensor, error) {
		return r.Run(roleModel, inputs)
	}
	dets, err := Detect(img, prompt.Text, m.tok, run, r.OutputNames(roleModel), boxThresh, textThresh,
		WithJointTextPass(m.joint))
	if err != nil {
		return models.Result{}, err
	}
	return models.Result{Detections: dets}, nil
}

// Option tunes Detect. It is variadic so the existing call sites (grounded-sam, grasp,
// hybrid) keep compiling unchanged.
type Option func(*detectOpts)

type detectOpts struct{ joint bool }

// WithJointTextPass scores the WHOLE prompt in a single session pass instead of one pass per
// "."-separated phrase. Only pass true for a graph that rebuilds the block-diagonal text
// self-attention mask for any number of phrases — ask SupportsJointTextPass(onnxPath), never
// assume. On the defective community export a joint pass silently drops classes and depends
// on their order (see the package doc).
func WithJointTextPass(joint bool) Option {
	return func(o *detectOpts) { o.joint = joint }
}

// Detect is the REUSABLE core (also called by grounded-sam). It builds the 5 inputs from
// the image + text, runs the session via run, and postprocesses logits/pred_boxes into
// detections in ORIGINAL image coordinates.
//
//   - tok:      a tokenizer loaded from the matching vocab.txt.
//   - run:      executes the GroundingDINO session, binding inputs by name.
//   - outNames: the session's output names (used to map logits vs pred_boxes; falls back
//     to shape matching when names are unhelpful).
//
// By default the prompt is split on "." and the session runs ONCE PER PHRASE, because the
// community export only builds the text self-attention block for the first phrase. Pass
// WithJointTextPass(true) — gated on SupportsJointTextPass — to score the whole prompt in one
// pass on a correctly re-exported graph. The DEFAULT IS THE SAFE ONE: a caller that knows
// nothing gets correct, order-stable results, just slower.
//
// Either way no pass exceeds MaxTextLen text ids: the joint regime packs phrases greedily into
// as many passes as needed, and a single phrase too long to fit on its own is an error.
// Detection labels are the caller's phrase text (one per ","-separated piece).
func Detect(
	img image.Image,
	text string,
	tok *Tokenizer,
	run func(map[string]engine.Tensor) ([]engine.Tensor, error),
	outNames []string,
	boxThresh, textThresh float64,
	opts ...Option,
) ([]api.Detection, error) {
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("grounding-dino requires a non-empty text prompt")
	}
	if tok == nil {
		return nil, fmt.Errorf("grounding-dino: nil tokenizer")
	}
	phrases := SplitPhrases(text)
	if len(phrases) == 0 {
		return nil, fmt.Errorf("grounding-dino: prompt %q holds no class phrase", text)
	}
	var o detectOpts
	for _, opt := range opts {
		opt(&o)
	}

	// Size every phrase BEFORE any pass runs: a phrase that cannot fit even alone is the
	// caller's error and must not surface as an ONNX Runtime shape error halfway through.
	pp := make([]promptPhrase, len(phrases))
	for i, p := range phrases {
		n := len(tok.tokenize(p))
		if n+3 > MaxTextLen { // [CLS] phrase . [SEP]
			return nil, fmt.Errorf("grounding-dino: class phrase %q is %d tokens; the model reads at most %d "+
				"text tokens per pass, so one phrase may hold at most %d — shorten it",
				truncateForError(p), n, MaxTextLen, MaxTextLen-3)
		}
		pp[i] = promptPhrase{text: p, ntok: n, labels: phraseLabels(p, tok)}
	}

	origW := img.Bounds().Dx()
	origH := img.Bounds().Dy()
	pixelValues, pixelMask := preprocessImage(img)

	// Joint regime: as few passes as the text limit allows. phraseSpans still splits the
	// logits per class, so the labels stay one-class-per-detection. A prompt that fits is ONE
	// pass with exactly the ids it always had; a longer one is packed greedily into passes of
	// at most MaxTextLen ids, each reusing the same pixel tensors, and the detections are
	// concatenated — the same merge the per-phrase regime has always done.
	if o.joint {
		var dets []api.Detection
		for _, chunk := range packPhrases(pp) {
			got, err := detectPass(pixelValues, pixelMask, chunk, tok, run, outNames, origW, origH, boxThresh, textThresh)
			if err != nil {
				return nil, err
			}
			dets = append(dets, got...)
		}
		return dets, nil
	}

	var dets []api.Detection
	for i := range pp {
		got, err := detectPass(pixelValues, pixelMask, pp[i:i+1], tok, run, outNames, origW, origH, boxThresh, textThresh)
		if err != nil {
			return nil, err
		}
		dets = append(dets, got...)
	}
	return dets, nil
}

// MaxTextLen is the number of text positions the exported graph accepts, [CLS] and [SEP]
// included: HF's max_text_len (config.json) and the width of the logits' last axis. Verified on
// the shipped export: L=256 runs, L=257 fails inside ONNX Runtime with "invalid expand shape".
const MaxTextLen = 256

// promptPhrase is one "."-separated class phrase as the caller wrote it.
type promptPhrase struct {
	text   string   // trimmed phrase, exactly as it goes into the pass text
	labels []string // the caller's label text for each ","-separated piece that has tokens
	ntok   int      // WordPiece tokens, no specials
}

// packPhrases groups phrases, in order, into passes whose encoding — [CLS] p1 . p2 . … [SEP]
// — stays within MaxTextLen ids. Every phrase is assumed to fit on its own (Detect checks).
func packPhrases(pp []promptPhrase) [][]promptPhrase {
	var chunks [][]promptPhrase
	start, used := 0, 2 // [CLS] + [SEP]
	for i, p := range pp {
		cost := p.ntok + 1 // the phrase and its "."
		if i > start && used+cost > MaxTextLen {
			chunks = append(chunks, pp[start:i])
			start, used = i, 2
		}
		used += cost
	}
	return append(chunks, pp[start:])
}

// phraseLabels returns the label for each ","-separated piece of a phrase, in order, keeping
// only pieces that produce at least one token — exactly the pieces phraseSpans turns into
// spans, since "," is the only character that encodes to idComma. Internal whitespace is
// collapsed; case and punctuation are the caller's.
func phraseLabels(phrase string, tok *Tokenizer) []string {
	var out []string
	for _, piece := range strings.Split(phrase, ",") {
		if len(tok.tokenize(piece)) == 0 {
			continue
		}
		out = append(out, strings.Join(strings.Fields(piece), " "))
	}
	return out
}

func truncateForError(s string) string {
	const max = 60
	if r := []rune(s); len(r) > max {
		return string(r[:max]) + "…"
	}
	return s
}

// detectPass runs ONE session pass over the given phrases (a single phrase on the defective
// export, as many as fit on a fixed one) and postprocesses it.
func detectPass(
	pixelValues, pixelMask engine.Tensor,
	phrases []promptPhrase,
	tok *Tokenizer,
	run func(map[string]engine.Tensor) ([]engine.Tensor, error),
	outNames []string,
	origW, origH int,
	boxThresh, textThresh float64,
) ([]api.Detection, error) {
	texts := make([]string, len(phrases))
	var labels []string
	for i, p := range phrases {
		texts[i] = p.text
		labels = append(labels, p.labels...)
	}
	text := strings.Join(texts, ". ")
	enc := tok.Encode(text + ".")
	L := int64(len(enc.InputIDs))
	if L > MaxTextLen {
		return nil, fmt.Errorf("grounding-dino: internal error: pass for %q is %d ids, limit %d", truncateForError(text), L, MaxTextLen)
	}

	inputs := map[string]engine.Tensor{
		"pixel_values":   pixelValues,
		"pixel_mask":     pixelMask,
		"input_ids":      engine.I64(enc.InputIDs, 1, L),
		"attention_mask": engine.I64(enc.AttentionMask, 1, L),
		"token_type_ids": engine.I64(enc.TokenTypeIDs, 1, L),
	}

	outs, err := run(inputs)
	if err != nil {
		return nil, fmt.Errorf("grounding-dino: inference failed for %q: %w", text, err)
	}

	logits, boxes := pickLogitsAndBoxes(outNames, outs)
	if logits == nil || boxes == nil {
		return nil, fmt.Errorf("grounding-dino: could not identify logits/pred_boxes among outputs (shapes %v)", util.ShapesOf(outs))
	}

	return postprocess(logits, boxes, enc.InputIDs, tok, labels, origW, origH, boxThresh, textThresh)
}

// SplitPhrases splits a GroundingDINO prompt into its "."-separated class phrases, trimmed
// and with empty pieces dropped ("cup.. remote." -> ["cup", "remote"]). A prompt without any
// "." is a single phrase. This is the string-level counterpart of phraseSpans, which does
// the same split on the TOKEN ids.
func SplitPhrases(text string) []string {
	var out []string
	for _, p := range strings.Split(text, ".") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// pickLogitsAndBoxes maps outputs to logits ([.,.,256]) and pred_boxes ([.,.,4]) by name
// first, then by last-dim shape.
func pickLogitsAndBoxes(names []string, outs []engine.Tensor) (logits, boxes *engine.Tensor) {
	for i := range outs {
		name := ""
		if i < len(names) {
			name = strings.ToLower(names[i])
		}
		switch {
		case strings.Contains(name, "box"):
			boxes = &outs[i]
		case strings.Contains(name, "logit"):
			logits = &outs[i]
		}
	}
	if logits == nil || boxes == nil {
		for i := range outs {
			switch outs[i].Dim(-1) {
			case 4:
				if boxes == nil {
					boxes = &outs[i]
				}
			case 256:
				if logits == nil {
					logits = &outs[i]
				}
			}
		}
	}
	return logits, boxes
}

// postprocess turns logits/pred_boxes into detections (original-image xywh, clamped to the
// image). labels, when it holds one entry per phrase span, names the spans with the caller's
// own text; otherwise the spans are named by decoding their tokens.
func postprocess(
	logits, boxes *engine.Tensor,
	inputIDs []int64,
	tok *Tokenizer,
	labels []string,
	origW, origH int,
	boxThresh, textThresh float64,
) ([]api.Detection, error) {
	nq := int(logits.Dim(1))   // 900 queries
	dim := int(logits.Dim(2))  // 256 text dims
	bDim := int(boxes.Dim(-1)) // 4
	if nq <= 0 || dim <= 0 || bDim != 4 {
		return nil, fmt.Errorf("grounding-dino: unexpected output shapes logits=%v boxes=%v", logits.Shape, boxes.Shape)
	}

	if len(inputIDs) > dim {
		// Positions past the logits' width have no score at all; dropping their phrases would
		// silently answer a different question. Detect packs passes to MaxTextLen, so this only
		// fires for an export whose text width is not the one the package was verified on.
		return nil, fmt.Errorf("grounding-dino: prompt pass has %d text tokens but the model scores %d positions", len(inputIDs), dim)
	}
	spans := phraseSpans(inputIDs, tok, labels)
	if len(spans) == 0 {
		return nil, nil // prompt held no real tokens
	}

	var dets []api.Detection
	for q := 0; q < nq; q++ {
		base := q * dim
		// Score each PHRASE separately (max sigmoid over its token positions) and assign
		// the query to the single best-scoring phrase. Scoring the whole prompt at once
		// and concatenating every token above text_threshold — what HF's
		// post_process_grounded_object_detection does — yields labels like
		// "chair tv vase bear" as soon as the prompt holds several classes.
		score, best := 0.0, -1
		for si, sp := range spans {
			s := 0.0
			for i := sp.start; i < sp.end; i++ {
				if p := geom.Sigmoid(float64(logits.Data[base+i])); p > s {
					s = p
				}
			}
			if s > score {
				score, best = s, si
			}
		}
		if best < 0 || score <= boxThresh || score <= textThresh {
			continue // below threshold, or no assignable phrase
		}
		phrase := spans[best].text

		dets = append(dets, api.Detection{
			BBox:  pixelBox(boxes.Data[q*4:q*4+4], origW, origH),
			Class: phrase,
			Conf:  score,
		})
	}
	return dets, nil
}

// pixelBox maps one normalized cxcywh pred_box to [x, y, w, h] in ORIGINAL pixels (plain squash,
// so multiply by W/H), clamped to the image like RF-DETR's boxes; a box already inside is
// bit-identical. The corner is computed as (c - size/2)·W — that rounding is part of every
// measured number — and min() moves a corner past the far edge onto it, which geom.Clamp alone
// would leave in place.
func pixelBox(bb []float32, origW, origH int) [4]float64 {
	cx, cy, w, h := float64(bb[0]), float64(bb[1]), float64(bb[2]), float64(bb[3])
	fw, fh := float64(origW), float64(origH)
	return geom.Clamp([4]float64{min((cx-w/2)*fw, fw), min((cy-h/2)*fh, fh), w * fw, h * fh}, origW, origH)
}

// phraseSpan is the half-open token-index range [start,end) of one prompt phrase, with
// the label a query assigned to it receives.
type phraseSpan struct {
	start, end int
	text       string
}

// phraseSpans splits an encoded prompt into one span per "."-separated class phrase.
//
//	"cup. water bottle."  ->  [CLS] cup . water bottle . [SEP]
//	                      ->  {1,2,"cup"}, {3,5,"water bottle"}
//
// [CLS], [SEP] and the separators themselves are excluded, as are empty spans (".."),
// so a query can never be labelled with punctuation.
//
// Span i is labelled labels[i] — the caller's own text, so "t-shirt" stays "t-shirt" rather
// than the WordPiece round trip "t - shirt", and "café" is not "cafe" — when labels has one
// entry per span. Otherwise (no labels supplied) each span is named by decoding its tokens.
// The caller (postprocess) guarantees every id has a logit column, so nothing is truncated.
func phraseSpans(inputIDs []int64, tok *Tokenizer, labels []string) []phraseSpan {
	var spans []phraseSpan
	start := 1 // skip [CLS]
	flush := func(end int) {
		if end > start {
			spans = append(spans, phraseSpan{start: start, end: end})
		}
	}
	limit := len(inputIDs) - 1 // skip [SEP]
	for i := start; i < limit; i++ {
		if inputIDs[i] == idPeriod || inputIDs[i] == idComma {
			flush(i)
			start = i + 1
		}
	}
	flush(limit) // trailing phrase when the prompt has no final "."
	useLabels := len(labels) == len(spans)
	for i := range spans {
		if useLabels {
			spans[i].text = labels[i]
		} else {
			spans[i].text = tok.Decode(inputIDs[spans[i].start:spans[i].end])
		}
	}
	return spans
}
