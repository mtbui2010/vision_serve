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
	"path/filepath"
	"strings"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/models/groundingdino"
	"visionserve/internal/models/mobilesam"
)

func init() {
	models.Register("rfdetr-gdino", New)
}

const (
	roleRFDETR  = "rfdetr"
	roleGDINO   = "gdino"
	roleEncoder = "encoder"
	roleDecoder = "decoder"
)

// GroundingDINO box/text thresholds when neither the request nor a sensible default applies.
// (RF-DETR uses its own conf_threshold from the manifest via its Postprocess; we deliberately
// do NOT reuse it for the GroundingDINO path — the two scores are on different scales.)
const (
	defaultBoxThresh  = 0.3
	defaultTextThresh = 0.25
)

type hybrid struct {
	cfg     models.Config
	rf      models.Model             // RF-DETR sub-model (plain Model), driven via the Runner
	tok     *groundingdino.Tokenizer // GroundingDINO tokenizer (vocab next to files.gdino)
	vocab   map[string]bool          // RF-DETR class names, lowercased (the routing set)
	withSAM bool                     // encoder+decoder present → also emit one mask per box
	joint   bool                     // gdino weights take the whole prompt in ONE pass
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

	tok, err := groundingdino.LoadTokenizer(resolveVocab(cfg))
	if err != nil {
		return nil, fmt.Errorf("hybrid: load tokenizer: %w", err)
	}

	vocab := make(map[string]bool, len(cfg.Labels))
	for _, l := range cfg.Labels {
		l = strings.ToLower(strings.TrimSpace(l))
		if l != "" && l != "n/a" {
			vocab[l] = true
		}
	}

	withSAM := cfg.Files[roleEncoder] != "" && cfg.Files[roleDecoder] != ""
	return &hybrid{
		cfg: cfg, rf: rf, tok: tok, vocab: vocab, withSAM: withSAM,
		joint: groundingdino.JointTextPassOrSafe(cfg.Files[roleGDINO]),
	}, nil
}

// resolveVocab finds vocab.txt next to the GroundingDINO weights (files.gdino is typically
// "../grounding-dino/model.onnx"); falls back to <cfg.Dir>/vocab.txt.
func resolveVocab(cfg models.Config) string {
	if g := cfg.Files[roleGDINO]; g != "" {
		return filepath.Join(filepath.Dir(g), "vocab.txt")
	}
	return filepath.Join(cfg.Dir, "vocab.txt")
}

func (m *hybrid) Name() string      { return m.cfg.Name }
func (m *hybrid) Task() models.Task { return models.TaskOpenVocab }

// Roles: both detectors, plus the two MobileSAM sessions when masks are enabled.
func (m *hybrid) Roles() []string {
	roles := []string{roleRFDETR, roleGDINO}
	if m.withSAM {
		roles = append(roles, roleEncoder, roleDecoder)
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
	// GroundingDINO may run on this path, so serialize the whole pipeline (see PipelineMu).
	groundingdino.PipelineMu.Lock()
	defer groundingdino.PipelineMu.Unlock()

	classes := parseClasses(prompt.Text)
	known, unknown := m.partition(classes)

	// Each detector is asked ONLY about the words it is the right tool for, and the two answers
	// are concatenated. The alternative — the whole request going to GroundingDINO as soon as one
	// word is out of vocabulary — threw away the in-domain specialist for the words it was trained
	// on: "cup. zebra." lost RF-DETR's cup entirely.
	var dets []models.Detection
	if len(classes) == 0 || len(known) > 0 {
		d, err := m.detectRFDETR(img, r)
		if err != nil {
			return models.Result{}, err
		}
		if len(known) > 0 {
			d = filterByClass(d, known)
		}
		dets = append(dets, d...)
	}
	if len(unknown) > 0 {
		// GroundingDINO now sees only the words RF-DETR could not serve. That is the point, but
		// it is not a pure subset of the old behaviour: the fusion and decoder layers attend over
		// the whole prompt, so dropping the in-vocabulary words removes them as distractors and
		// can move the remaining scores slightly. Prompt splitting is a modelling decision, not
		// just a dispatch optimisation.
		sub := prompt
		sub.Text = joinClasses(unknown)
		d, err := m.detectGDINO(img, sub, r)
		if err != nil {
			return models.Result{}, err
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
		firstName(r.InputNames(roleEncoder), "input_image"), r.OutputNames(roleDecoder))
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
// names ([cat, remote]). An empty prompt yields nil.
func parseClasses(text string) []string {
	var out []string
	for _, p := range strings.Split(text, ".") {
		if p = strings.ToLower(strings.TrimSpace(p)); p != "" {
			out = append(out, p)
		}
	}
	return out
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
		if want[strings.ToLower(d.Class)] {
			out = append(out, d)
		}
	}
	return out
}

// detectRFDETR drives the plain RF-DETR Model via the Runner (role "rfdetr"): the model's
// own Preprocess builds the input tensor + meta, the Runner executes the session, and the
// model's Postprocess decodes detections (BBox already in ORIGINAL image coords).
func (m *hybrid) detectRFDETR(img image.Image, r models.Runner) ([]models.Detection, error) {
	in, meta, err := m.rf.Preprocess(img)
	if err != nil {
		return nil, err
	}
	inName := firstName(r.InputNames(roleRFDETR), m.rf.InputName())
	if inName == "" {
		return nil, fmt.Errorf("hybrid: rf-detr session %q has no input name", roleRFDETR)
	}
	outs, err := r.Run(roleRFDETR, map[string]engine.Tensor{inName: in})
	if err != nil {
		return nil, err
	}
	res, err := m.rf.Postprocess(outs, meta)
	if err != nil {
		return nil, err
	}
	return res.Detections, nil
}

// detectGDINO runs GroundingDINO (text-prompted) for boxes + labels. It uses GroundingDINO's
// own thresholds (request override → built-in default), never RF-DETR's conf_threshold.
func (m *hybrid) detectGDINO(img image.Image, prompt models.Prompt, r models.Runner) ([]models.Detection, error) {
	box, text := defaultBoxThresh, defaultTextThresh
	if prompt.BoxThresh > 0 {
		box = prompt.BoxThresh
	}
	if prompt.TextThresh > 0 {
		text = prompt.TextThresh
	}
	run := func(in map[string]engine.Tensor) ([]engine.Tensor, error) { return r.Run(roleGDINO, in) }
	return groundingdino.Detect(img, prompt.Text, m.tok, run, r.OutputNames(roleGDINO), box, text,
		groundingdino.WithJointTextPass(m.joint))
}

// ExplainPreprocess implements models.ExplainPreprocessor.
// Delegates to the rfdetr sub-model so the lifecycle can preprocess images for
// the rfdetr role's explain session without knowing the hybrid internals.
func (m *hybrid) ExplainPreprocess(img image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	return m.rf.Preprocess(img)
}

func firstName(names []string, fallback string) string {
	if len(names) > 0 {
		return names[0]
	}
	return fallback
}
