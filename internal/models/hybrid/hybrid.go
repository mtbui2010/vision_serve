// Package hybrid implements a router model (Apache-2.0) that picks the right detector
// per request: RF-DETR (fast, closed-set COCO) when every requested class is in RF-DETR's
// vocabulary, otherwise GroundingDINO (open-vocab, text-prompted). With the optional
// MobileSAM encoder+decoder roles it also returns one mask per detected box (Grounded-SAM
// style). It is a fully free community pipeline — no AGPL, no Python at runtime (CLAUDE.md).
//
// Both architectures registered here are CONFIGURATIONS of one pipeline.Router:
//
//	rfdetr-gdino   Closed (RF-DETR) + Open (GroundingDINO) [+ SigLIP Rescore] [+ distilled Head]
//	               [+ MobileSAM Segment]
//	gdino-siglip   Open (GroundingDINO) + SigLIP Rescore [+ MobileSAM Segment]
//
// Every stage reuses an existing model package instead of duplicating it: RF-DETR through its
// plain Model interface (pipeline.Closed), GroundingDINO through groundingdino.Detect
// (pipeline.GDINO), MobileSAM through mobilesam.SegmentDetections (pipeline.SAM), the SigLIP
// towers through the shared crop namer (pipeline.CropNamer). This package keeps what is its own:
// which roles exist, the rescorer's measured constants (rescore.go) and the distilled head
// (fastpath.go).
//
// The heavy ONNX sessions (rfdetr, gdino, encoder, decoder, crop, text, head) are owned by
// lifecycle.Manager; this model only orchestrates them via the Runner (it never creates or frees
// sessions).
package hybrid

import (
	"fmt"
	"image"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/pipeline"
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
	rf      models.Model        // RF-DETR sub-model (plain Model); nil for gdino-siglip
	withSAM bool                // encoder+decoder present → also emit one mask per box
	rs      *pipeline.CropNamer // optional SigLIP crop+text towers; nil = no rescoring
	fp      *fastPath           // optional distilled head; nil = unknown words go to GroundingDINO
	rt      *pipeline.Router    // the stages above, composed

	// open serialises THIS model's GroundingDINO section (the router's OpenLock).
	open sync.Mutex
}

// New builds the router. The RF-DETR sub-model is created from the SAME cfg, so the
// manifest's input/postprocess/labels block MUST carry RF-DETR's values (e.g. 560×560,
// squash, ImageNet normalize, coco91.txt). GroundingDINO ignores that block — it
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
	return b, nil
}

// build is the part both architectures share once the closed-set detector (or its absence) is
// decided: load each optional stage the manifest declares, then compose them.
func build(cfg models.Config, rf models.Model) (*hybrid, error) {
	// GroundingDINO's thresholds are request → built-in default only: this manifest's
	// conf_threshold is RF-DETR's, on a different scale, and it declares no text_threshold.
	gdino, err := pipeline.NewGDINO(roleGDINO, cfg.Files[roleGDINO], cfg.Dir, 0, 0)
	if err != nil {
		return nil, fmt.Errorf("hybrid: load tokenizer: %w", err)
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

	m := &hybrid{
		cfg: cfg, rf: rf, rs: rs, fp: fp,
		withSAM: cfg.Files[roleEncoder] != "" && cfg.Files[roleDecoder] != "",
	}
	m.wire(gdino)
	return m, nil
}

// wire composes the router from the model's stages.
func (m *hybrid) wire(gdino pipeline.Detector) {
	rt := &pipeline.Router{
		Name:  m.cfg.Name,
		Vocab: pipeline.ClosedVocab(m.cfg.Labels),
		Open:  gdino,
		// One GroundingDINO section at a time on THIS model — the GroundingDINO pass and the
		// stages chained after it (rescoring, MobileSAM) — like every GroundingDINO pipeline (see
		// the groundingdino package doc). It is per instance: other models, including other
		// GroundingDINO pipelines, are not held up by it (the process-wide PipelineMu it
		// replaced made them wait on each other). It is taken only when a request reaches
		// GroundingDINO, just before the pass, and held to the end. A request answered by RF-DETR
		// alone (every word in its vocabulary, no prompt, or the distilled head) never takes it:
		// taking a lock for those made the recommended default model serialise every request.
		// Not models.Exclusive for that reason.
		OpenLock: &m.open,
	}
	if m.rf != nil {
		rt.Closed = &pipeline.Closed{Role: roleRFDETR, Model: m.rf, DETR: true, Labels: len(m.cfg.Labels), Prefix: "hybrid"}
	}
	if m.rs != nil {
		rt.Rescore = pipeline.CropRescorer{Namer: m.rs, Prefix: "hybrid"}
	}
	if m.hasFastPath() {
		m.fp.cfg, m.fp.text = m.cfg, m.rs.Text // the head scores against the rescorer's text rows
		rt.Head = m.fp
	}
	if m.withSAM {
		rt.Segment = pipeline.SAM{Encoder: roleEncoder, Decoder: roleDecoder}
	}
	m.rt = rt
}

func (m *hybrid) Name() string      { return m.cfg.Name }
func (m *hybrid) Task() models.Task { return models.TaskOpenVocab }

// Roles: both detectors (GroundingDINO alone for gdino-siglip), plus the two MobileSAM sessions
// when masks are enabled, the SigLIP pair when rescoring, and the ONNX head.
func (m *hybrid) Roles() []string {
	roles := []string{roleRFDETR, roleGDINO}
	if m.rf == nil {
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

// Infer routes each requested word to RF-DETR, the distilled head or GroundingDINO, rescores the
// open answers, then optionally segments each detected box (pipeline.Router).
func (m *hybrid) Infer(img image.Image, prompt models.Prompt, r models.Runner) (models.Result, error) {
	return m.rt.Infer(pipeline.Call{Img: img, Prompt: prompt, Runner: r})
}

// ExplainPreprocess implements models.ExplainPreprocessor.
// Delegates to the rfdetr sub-model so the lifecycle can preprocess images for
// the rfdetr role's explain session without knowing the hybrid internals.
func (m *hybrid) ExplainPreprocess(img image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	if m.rf == nil {
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
