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
	return &hybrid{cfg: cfg, rf: rf, tok: tok, vocab: vocab, withSAM: withSAM}, nil
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

	var dets []models.Detection
	var err error
	if m.routeRFDETR(classes) {
		if dets, err = m.detectRFDETR(img, r); err != nil {
			return models.Result{}, err
		}
		if len(classes) > 0 {
			dets = filterByClass(dets, classes) // keep only the requested COCO classes
		}
	} else {
		if dets, err = m.detectGDINO(img, prompt, r); err != nil {
			return models.Result{}, err
		}
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

// routeRFDETR reports whether RF-DETR can serve the request: no prompt (detect everything it
// knows), or EVERY requested class is in RF-DETR's vocabulary. A single out-of-vocab class
// routes the whole request to GroundingDINO (open-vocab).
func (m *hybrid) routeRFDETR(classes []string) bool {
	if len(classes) == 0 {
		return true
	}
	for _, c := range classes {
		if !m.vocab[c] {
			return false
		}
	}
	return true
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
	return groundingdino.Detect(img, prompt.Text, m.tok, run, r.OutputNames(roleGDINO), box, text)
}

func firstName(names []string, fallback string) string {
	if len(names) > 0 {
		return names[0]
	}
	return fallback
}
