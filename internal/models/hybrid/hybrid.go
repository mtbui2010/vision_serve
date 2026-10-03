// Package hybrid implements a router model (Apache-2.0) that picks the right detector
// per request: RF-DETR (fast, closed-set COCO) when every requested class is in RF-DETR's
// vocabulary, otherwise GroundingDINO (open-vocab, text-prompted). With the optional
// MobileSAM encoder+decoder roles it also returns one mask per detected box (Grounded-SAM
// style). It is a fully free community pipeline — no AGPL, no Python at runtime (CLAUDE.md).
//
// It composes the three existing models WITHOUT duplicating their logic:
//   - RF-DETR is reused via the plain Model interface (Preprocess → Run → Postprocess),
//     exactly like grasp's modelDetector.
//   - GroundingDINO via groundingdino.Detect; MobileSAM via mobilesam.Segment.
//
// The heavy ONNX sessions (rfdetr, gdino, encoder, decoder) are owned by lifecycle.Manager;
// this model only orchestrates them via the Runner (it never creates or frees sessions).
package hybrid

import (
	"fmt"
	"image"
	"os"
	"path/filepath"
	"strings"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/models/detr"
	"visionserve/internal/models/groundingdino"
	"visionserve/internal/models/mobilesam"
	"visionserve/internal/pipeline"
	"visionserve/internal/vision/util"
)

func init() {
	models.Register("rfdetr-gdino", New)
	models.Register("gdino-siglip", NewGDINOSigLIP)
}

const (
	roleRFDETR  = "rfdetr"
	roleGDINO   = "gdino"
	roleEncoder = "encoder"
	roleDecoder = "decoder"
)

type hybrid struct {
	cfg     models.Config
	rf      models.Model             // RF-DETR sub-model (plain Model), driven via the Runner
	noRF    bool                     // gdino-siglip: no closed-set detector, every word → GroundingDINO
	tok     *groundingdino.Tokenizer // GroundingDINO tokenizer (vocab next to files.gdino)
	vocab   map[string]bool          // RF-DETR class names, lowercased (the routing set)
	withSAM bool                     // encoder+decoder present → also emit one mask per box
	joint   bool                     // gdino weights take the whole prompt in ONE pass
	rs      *pipeline.CropNamer      // optional SigLIP crop+text towers; nil = no rescoring
	fp      *fastPath                // optional distilled head; nil = unknown words go to GroundingDINO
}

// New builds the router. The RF-DETR sub-model is created from the SAME cfg, so the
// manifest's input/postprocess/labels block MUST carry RF-DETR's values (e.g. 560×560,
// letterbox, ImageNet normalize, coco91.txt). GroundingDINO ignores that block — it
// preprocesses to its own 800×800 internally.
func New(cfg models.Config) (models.Base, error) {
	if cfg.Files[roleRFDETR] == "" || cfg.Files[roleGDINO] == "" {
		return nil, fmt.Errorf("hybrid: manifest must declare files.%s and files.%s", roleRFDETR, roleGDINO)
	}

	base, err := models.New("rf-detr", cfg)
	if err != nil {
		return nil, fmt.Errorf("hybrid: rf-detr sub-model: %w", err)
	}
	rf, ok := base.(models.Model)
	if !ok {
		return nil, fmt.Errorf("hybrid: rf-detr did not yield a plain Model")
	}
	b, err := build(cfg, rf)
	if err != nil {
		return nil, err // not `return build(...)`: a typed nil *hybrid would be a non-nil Base
	}
	return b, nil
}

// NewGDINOSigLIP builds the router with NO closed-set detector (architecture "gdino-siglip"):
// every requested word goes to GroundingDINO, and every GroundingDINO detection is rescored by
// SigLIP — Grounded-SAM-style open vocabulary with a rejector, for domains no RF-DETR was trained
// on. The SigLIP pair is REQUIRED: without it this would be plain grounding-dino / grounded-sam
// under another name, and a manifest that forgot one role would silently serve the unrescored
// model. The input/postprocess/labels block is not read (GroundingDINO preprocesses itself).
func NewGDINOSigLIP(cfg models.Config) (models.Base, error) {
	if cfg.Files[roleGDINO] == "" {
		return nil, fmt.Errorf("gdino-siglip: manifest must declare files.%s", roleGDINO)
	}
	if cfg.Files[roleRFDETR] != "" {
		return nil, fmt.Errorf("gdino-siglip: files.%s is declared — use architecture rfdetr-gdino for the router", roleRFDETR)
	}
	if cfg.Files[roleCrop] == "" || cfg.Files[roleText] == "" {
		return nil, fmt.Errorf("gdino-siglip: manifest must declare files.%s and files.%s (without SigLIP, "+
			"use grounding-dino or grounded-sam)", roleCrop, roleText)
	}
	if hasHead(cfg) {
		return nil, fmt.Errorf("gdino-siglip: the distilled head reads RF-DETR's query features; it needs architecture rfdetr-gdino")
	}
	cfg.Labels = nil // no closed vocabulary: nothing is routed away from GroundingDINO
	b, err := build(cfg, nil)
	if err != nil {
		return nil, err
	}
	b.noRF = true
	return b, nil
}

// build is the part both architectures share once the closed-set detector (or its absence) is
// decided.
func build(cfg models.Config, rf models.Model) (*hybrid, error) {
	tok, err := groundingdino.LoadTokenizer(groundingdino.VocabPath(cfg.Files[roleGDINO], cfg.Dir))
	if err != nil {
		return nil, fmt.Errorf("hybrid: load tokenizer: %w", err)
	}

	vocab := make(map[string]bool, len(cfg.Labels))
	for _, l := range cfg.Labels {
		l = normClass(l)
		if l != "" && l != "n/a" {
			vocab[l] = true
		}
	}

	rs, err := newRescorer(cfg)
	if err != nil {
		return nil, err
	}

	fp, err := newFastPath(cfg)
	if err != nil {
		return nil, err
	}
	if fp != nil && rs == nil {
		return nil, fmt.Errorf("hybrid: the distilled head (files.head or %s) needs the SigLIP rescorer (files.%s and files.%s) — "+
			"the head scores 21.53 mAP alone and 46.41 with rescoring on top, so serving it "+
			"without one gives less than half of what it is worth (ovd-edge FINDINGS §10)",
			headFile, roleCrop, roleText)
	}

	withSAM := cfg.Files[roleEncoder] != "" && cfg.Files[roleDecoder] != ""
	return &hybrid{
		cfg: cfg, rf: rf, tok: tok, vocab: vocab, withSAM: withSAM, rs: rs, fp: fp,
		joint: groundingdino.JointTextPassOrSafe(cfg.Files[roleGDINO]),
	}, nil
}

func (m *hybrid) Name() string      { return m.cfg.Name }
func (m *hybrid) Task() models.Task { return models.TaskOpenVocab }

// Roles: both detectors, plus the two MobileSAM sessions when masks are enabled.
func (m *hybrid) Roles() []string {
	roles := []string{roleRFDETR, roleGDINO}
	if m.noRF {
		roles = []string{roleGDINO}
	}
	if m.withSAM {
		roles = append(roles, roleEncoder, roleDecoder)
	}
	if m.rs != nil {
		roles = append(roles, roleCrop, roleText)
	}
	if m.fp != nil && m.fp.onnx {
		// The distilled head as its own session: lifecycle creates it with the manifest's
		// provider chain, the same one the detector gets (BUGS_TO_FIX #4).
		roles = append(roles, roleHead)
	}
	return roles
}

// PoolSizes requests 4 decoder copies so multiple boxes segment concurrently (mask mode).
func (m *hybrid) PoolSizes() map[string]int {
	if m.withSAM {
		return map[string]int{roleDecoder: 4}
	}
	return nil
}

// Infer routes to RF-DETR or GroundingDINO, then optionally segments each detected box.
func (m *hybrid) Infer(img image.Image, prompt models.Prompt, r models.Runner) (models.Result, error) {
	// groundingdino.PipelineMu serialises GroundingDINO pipelines end to end — the GroundingDINO
	// pass AND the MobileSAM decoder pool chained after it — because that combination under
	// concurrent load produced sporadic ORT errors (see its comment). It is taken only when this
	// request actually reaches GroundingDINO, just before the pass, and held to the end so the
	// chained SAM stage stays covered. A request answered by RF-DETR alone (every word in its
	// vocabulary, no prompt, or the distilled head) never runs GroundingDINO and is exactly the
	// grasp-rfdetr shape the lock has always left concurrent; taking the lock for it made the
	// recommended default model serialise every request on the server.
	locked := false
	defer func() {
		if locked {
			groundingdino.PipelineMu.Unlock()
		}
	}()

	classes := parseClasses(prompt.Text)
	if m.noRF && len(classes) == 0 {
		return models.Result{}, fmt.Errorf("%s: a text prompt is required (e.g. \"cup. towel.\") — "+
			"there is no closed-set detector to answer an empty one", m.cfg.Name)
	}
	known, unknown := m.partition(classes)

	// Each detector is asked ONLY about the words it is the right tool for, and the two answers
	// are concatenated. The alternative — the whole request going to GroundingDINO as soon as one
	// word is out of vocabulary — threw away the in-domain specialist for the words it was trained
	// on: "cup. zebra." lost RF-DETR's cup entirely.
	var dets []models.Detection
	// The detector runs at most ONCE and serves both branches: the closed head names the known
	// words and, when a distilled head is wired, the SAME query features answer the unknown ones.
	// That sharing is the whole latency argument — the open branch costs a matmul rather than a
	// second detector (ovd-edge FINDINGS §10-§11).
	var qf engine.Tensor
	var qBoxes engine.Tensor
	var qMeta models.PreprocessMeta
	if !m.noRF && (len(classes) == 0 || len(known) > 0 || m.hasFastPath()) {
		d, boxes, feats, meta, err := m.detectRFDETRWithFeats(img, r)
		if err != nil {
			return models.Result{}, err
		}
		qf, qBoxes, qMeta = feats, boxes, meta
		if len(classes) > 0 && len(known) == 0 {
			d = nil // the caller asked only about unknown words; the closed head has no answer
		} else if len(known) > 0 {
			d = filterByClass(d, known)
		}
		dets = append(dets, d...)
	}

	if len(unknown) > 0 && m.hasFastPath() {
		if qf.Data == nil {
			return models.Result{}, fmt.Errorf("hybrid: the detector export emitted no query "+
				"features, so the distilled head has no input — %s needs the -qf export",
				headFile)
		}
		d, err := m.fastDetect(qBoxes, qf, unknown, qMeta, r)
		if err != nil {
			return models.Result{}, err
		}
		if d, err = m.rescore(img, d, unknown, prompt.CropTemp, r); err != nil {
			return models.Result{}, err
		}
		dets = append(dets, d...)
		unknown = nil // answered; GroundingDINO is not consulted
	}

	if len(unknown) > 0 {
		// GroundingDINO now sees only the words RF-DETR could not serve. That is the point, but
		// it is not a pure subset of the old behaviour: the fusion and decoder layers attend over
		// the whole prompt, so dropping the in-vocabulary words removes them as distractors and
		// can move the remaining scores slightly. Prompt splitting is a modelling decision, not
		// just a dispatch optimisation.
		groundingdino.PipelineMu.Lock()
		locked = true
		sub := prompt
		sub.Text = joinClasses(unknown)
		d, err := m.detectGDINO(img, sub, r)
		if err != nil {
			return models.Result{}, err
		}
		// Rescore ONLY GroundingDINO's detections, and only against the words it was asked
		// about. RF-DETR's are not touched: its supervised head already answered for those
		// words on one calibrated scale, and the 17-name control column must not move.
		if m.rs != nil {
			if d, err = m.rescore(img, d, unknown, prompt.CropTemp, r); err != nil {
				return models.Result{}, err
			}
		}
		dets = append(dets, d...)
	}

	res := models.Result{Detections: dets}
	if !m.withSAM || len(dets) == 0 {
		return res, nil
	}

	// Segment one mask per detected box (MobileSAM), index-aligned with detections.
	boxes := make([][4]float64, len(dets))
	for i, d := range dets {
		boxes[i] = d.BBox
	}
	encRun := func(in map[string]engine.Tensor) ([]engine.Tensor, error) { return r.Run(roleEncoder, in) }
	decRun := func(in map[string]engine.Tensor) ([]engine.Tensor, error) { return r.Run(roleDecoder, in) }
	masks, err := mobilesam.Segment(img, boxes, encRun, decRun,
		util.FirstName(r.InputNames(roleEncoder), "input_image"), r.OutputNames(roleDecoder))
	if err != nil {
		return models.Result{}, err
	}
	for i := range masks {
		if i < len(dets) {
			masks[i].BBox = dets[i].BBox
			masks[i].Conf = dets[i].Conf
		}
	}
	res.Masks = masks
	return res, nil
}

// parseClasses splits a GroundingDINO-style prompt ("cat. remote.") into lowercase class
// names ([cat, remote]), whitespace-collapsed and de-duplicated in first-seen order. An empty
// prompt yields nil.
//
// De-duplication is not cosmetic: the rescorer softmaxes over the word list, so "zebra. zebra."
// put two identical rows in the softmax and halved every rescored confidence (and asked
// GroundingDINO the same phrase twice). Whitespace collapsing keeps "dining  table" on
// RF-DETR's "dining table" route instead of sending it to GroundingDINO.
func parseClasses(text string) []string {
	var out []string
	seen := make(map[string]bool)
	for _, p := range strings.Split(text, ".") {
		p = normClass(p)
		if p == "" || seen[p] {
			continue
		}
		seen[p] = true
		out = append(out, p)
	}
	return out
}

// normClass is the one normal form class names are compared in: lowercase, single spaces.
func normClass(s string) string {
	return strings.Join(strings.Fields(strings.ToLower(s)), " ")
}

// partition splits the requested classes into the ones RF-DETR was trained on and the ones only
// GroundingDINO can serve. The router reads text and dispatches; it never detects anything itself.
// An empty prompt yields two empty slices and is handled by the caller as "RF-DETR, everything it
// knows" — its own vocabulary IS the prompt in that case.
func (m *hybrid) partition(classes []string) (known, unknown []string) {
	for _, c := range classes {
		if m.vocab[c] {
			known = append(known, c)
		} else {
			unknown = append(unknown, c)
		}
	}
	return known, unknown
}

// joinClasses rebuilds a GroundingDINO prompt from class names ([cat, remote] -> "cat. remote.").
func joinClasses(classes []string) string {
	var b strings.Builder
	for _, c := range classes {
		b.WriteString(c)
		b.WriteString(". ")
	}
	return strings.TrimSpace(b.String())
}

// filterByClass keeps only detections whose class is among the requested names.
func filterByClass(dets []models.Detection, classes []string) []models.Detection {
	want := make(map[string]bool, len(classes))
	for _, c := range classes {
		want[c] = true
	}
	out := make([]models.Detection, 0, len(dets))
	for _, d := range dets {
		if want[normClass(d.Class)] {
			out = append(out, d)
		}
	}
	return out
}

// detectRFDETRWithFeats drives the plain RF-DETR Model via the Runner (role "rfdetr") and hands
// back the raw tensors as well as the decoded detections: the model's own Preprocess builds the
// input tensor + meta, the Runner executes the session, and the model's Postprocess decodes
// detections (BBox already in ORIGINAL image coords).
//
// It returns `boxes`, `feats` and `meta` on top of the detections so the distilled head can score
// the SAME forward pass instead of paying for another one. `feats` is the `query_feats` output of
// the `-qf` exports and is a ZERO tensor when the export does not carry one — callers check
// `feats.Data == nil` rather than assuming, because the plain rf-detr export is a perfectly valid
// thing to wire here and simply cannot drive a head.
func (m *hybrid) detectRFDETRWithFeats(img image.Image, r models.Runner) (
	[]models.Detection, engine.Tensor, engine.Tensor, models.PreprocessMeta, error) {
	var boxes, feats engine.Tensor
	in, meta, err := m.rf.Preprocess(img)
	if err != nil {
		return nil, boxes, feats, meta, err
	}
	inName := util.FirstName(r.InputNames(roleRFDETR), m.rf.InputName())
	if inName == "" {
		return nil, boxes, feats, meta, fmt.Errorf("hybrid: rf-detr session %q has no input name", roleRFDETR)
	}
	outs, err := r.Run(roleRFDETR, map[string]engine.Tensor{inName: in})
	if err != nil {
		return nil, boxes, feats, meta, err
	}
	// Identify the tensors by SHAPE (detr.SplitOutputs, the rule textalign shares): boxes are the
	// output whose last dimension is 4, the class logits the one matching the label count, and
	// query features the [1, Q, D] output that is neither. Indexing by position would break the
	// moment an export adds an output, which is how a 17-class cache once served a 22-class model.
	o, err := detr.SplitOutputs(outs, len(m.cfg.Labels), 0)
	if err != nil {
		return nil, boxes, feats, meta, err
	}
	if o.Logits.Data == nil {
		return nil, boxes, feats, meta, fmt.Errorf("hybrid: rf-detr session %q emitted no class logits (outputs %v)", roleRFDETR, util.ShapesOf(outs))
	}
	res, err := m.rf.Postprocess([]engine.Tensor{o.Boxes, o.Logits}, meta)
	if err != nil {
		return nil, boxes, feats, meta, err
	}
	return res.Detections, o.Boxes, o.Feats, meta, nil
}

// detectGDINO runs GroundingDINO (text-prompted) for boxes + labels. It uses GroundingDINO's
// own thresholds (request override → built-in default), never the manifest's: conf_threshold
// there is RF-DETR's, on a different scale.
func (m *hybrid) detectGDINO(img image.Image, prompt models.Prompt, r models.Runner) ([]models.Detection, error) {
	box, text := groundingdino.Thresholds(0, 0, prompt)
	run := func(in map[string]engine.Tensor) ([]engine.Tensor, error) { return r.Run(roleGDINO, in) }
	return groundingdino.Detect(img, prompt.Text, m.tok, run, r.OutputNames(roleGDINO), box, text,
		groundingdino.WithJointTextPass(m.joint))
}

// ExplainPreprocess implements models.ExplainPreprocessor.
// Delegates to the rfdetr sub-model so the lifecycle can preprocess images for
// the rfdetr role's explain session without knowing the hybrid internals.
func (m *hybrid) ExplainPreprocess(img image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	if m.noRF {
		return engine.Tensor{}, models.PreprocessMeta{}, fmt.Errorf("%s: no RF-DETR session to explain", m.cfg.Name)
	}
	return m.rf.Preprocess(img)
}

// hasHead reports whether a distilled head is configured, as an ONNX role or as head.bin.
func hasHead(cfg models.Config) bool {
	if strings.TrimSpace(cfg.Files[roleHead]) != "" {
		return true
	}
	_, err := os.Stat(filepath.Join(cfg.Dir, headFile))
	return err == nil
}
