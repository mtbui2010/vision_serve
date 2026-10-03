// Package grasp implements the "grasp" architecture: a configurable planar
// parallel-jaw grasp pipeline.
//
//		(optional detector) → segmenter (mask) → mask2grasp (analytic)
//
//	  - Segmenter (mandatory, default "mobile-sam") turns the image into object
//	    masks: box-prompted when a detector supplies boxes, or whole-image automask
//	    when there is no detector.
//	  - Detector (OPTIONAL) supplies boxes + class labels. Set it ("rf-detr",
//	    "grounding-dino", …) for CLASS-AWARE grasps; omit it for CLASS-AGNOSTIC
//	    grasps (automask over the whole image).
//	  - The final stage is the pure-Go analytic mask2grasp search (internal/grasp);
//	    it adds no ONNX session and no weights.
//
// The model is a configuration of pipeline stages composed by pipeline.Grasp: a
// pipeline.Closed (any registered plain detector) or pipeline.GDINO detector, a
// pipeline.SAMBitmaps segmenter over MobileSAM, and the pipeline.AnalyticGrasp planner.
// Like Grounded-SAM, it only ORCHESTRATES sessions owned by lifecycle.Manager (VRAM-safe):
// the detector session under role "det" and the MobileSAM encoder/decoder under
// "encoder"/"decoder".
package grasp

import (
	"fmt"
	"image"
	"strings"

	"visionserve/internal/models"
	"visionserve/internal/models/mobilesam"
	"visionserve/internal/pipeline"
)

func init() {
	models.Register("grasp", New)
}

const (
	roleDet     = "det"
	roleEncoder = "encoder"
	roleDecoder = "decoder"
)

const defaultSegmenter = "mobile-sam"

// defaultMaxGraspsPerMask caps the grasps returned for EACH mask (the best-quality ones). It is
// deliberately separate from the manifest's max_detections, which is the DETECTOR's cap (300 on
// grasp-rfdetr, read by RF-DETR's own postprocess): reused as the grasp cap it let one
// star-shaped mask return thousands of grasps and the response grow to megabytes. A grasp
// consumer executes the best few; the Python client's own post-filter keeps 3 per object.
const defaultMaxGraspsPerMask = 20

// segmenter is what the grasp pipeline needs from its segmentation model: the sessions it uses
// (Roles, PoolSizes via the PipelineModel) and its masks' bitmaps (InferMasks), so the grasp
// search reads pixels without decoding the RLE back. MobileSAM — the only segmenter New
// accepts — implements it, and New refuses one that does not.
type segmenter interface {
	models.PipelineModel
	pipeline.MaskInferer
}

// graspModel is the composed pipeline.
type graspModel struct {
	cfg     models.Config
	sam     segmenter              // MobileSAM: its roles, and the segmentation stage
	planner pipeline.AnalyticGrasp // manifest gripper defaults; the request may override them
	p       pipeline.Grasp
	// serialize is true only for the GroundingDINO detector variant (grasp-gd), which runs one
	// whole pipeline at a time per loaded model (Exclusive). grasp-rfdetr and the class-agnostic
	// automask path stay fully concurrent.
	serialize bool
}

// New builds the grasp model. The segmenter (MobileSAM) is constructed from the
// same manifest (files.encoder/decoder). A detector is constructed only when the
// manifest sets `detector:` and provides files.det.
func New(cfg models.Config) (models.Base, error) {
	seg := strings.TrimSpace(cfg.Segmenter)
	if seg == "" {
		seg = defaultSegmenter
	}
	if seg != "mobile-sam" {
		return nil, fmt.Errorf("grasp: segmenter %q not supported yet (only mobile-sam)", seg)
	}
	if cfg.Files[roleEncoder] == "" || cfg.Files[roleDecoder] == "" {
		return nil, fmt.Errorf("grasp: manifest must declare files.%s and files.%s", roleEncoder, roleDecoder)
	}
	samBase, err := mobilesam.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("grasp: segmenter: %w", err)
	}
	sam, ok := samBase.(segmenter)
	if !ok {
		return nil, fmt.Errorf("grasp: segmenter does not expose its mask bitmaps")
	}

	g := &graspModel{cfg: cfg, sam: sam, planner: plannerFor(cfg)}
	g.p = pipeline.Grasp{Segmenter: pipeline.SAMBitmaps{Model: sam}, Planner: g.planner}

	if name := strings.TrimSpace(cfg.Detector); name != "" {
		if cfg.Files[roleDet] == "" {
			return nil, fmt.Errorf("grasp: detector %q set but files.%s is missing", name, roleDet)
		}
		if err := g.setDetector(name, cfg); err != nil {
			return nil, err
		}
	}
	return g, nil
}

// plannerFor is the manifest's planner: its gripper bounds (a request may override them) and the
// per-mask cap, which is defaultMaxGraspsPerMask whatever the detector's max_detections says.
func plannerFor(cfg models.Config) pipeline.AnalyticGrasp {
	return pipeline.AnalyticGrasp{GripMin: cfg.GripperMin, GripMax: cfg.GripperMax, MaxPerMask: defaultMaxGraspsPerMask}
}

// setDetector picks the detector stage by name. GroundingDINO is the text-prompted special case
// (the request's phrases are its words); everything else must be a registered plain Model, which
// answers with every class it knows (the request's text is not read).
func (g *graspModel) setDetector(name string, cfg models.Config) error {
	if name == "grounding-dino" {
		det, err := pipeline.NewGDINO(roleDet, cfg.Files[roleDet], cfg.Dir, cfg.ConfThresh, cfg.TextThresh)
		if err != nil {
			return fmt.Errorf("grasp: detector grounding-dino: %w", err)
		}
		g.p.Detector, g.p.Words = det, gdinoWords
		g.serialize = true // one GroundingDINO pipeline at a time on this model (Exclusive)
		return nil
	}
	base, err := models.New(name, cfg)
	if err != nil {
		return fmt.Errorf("grasp: detector %q: %w", name, err)
	}
	m, ok := base.(models.Model)
	if !ok {
		return fmt.Errorf("grasp: detector %q is not a plain Model (use grounding-dino for text-prompted detectors)", name)
	}
	g.p.Detector = &pipeline.Closed{Role: roleDet, Model: m, Prefix: "grasp"}
	return nil
}

// gdinoWords: the request's "."-separated phrases; a text prompt is required.
func gdinoWords(p models.Prompt) ([]string, error) {
	if strings.TrimSpace(p.Text) == "" {
		return nil, models.BadPrompt(fmt.Errorf("grasp: the grounding-dino detector requires a text prompt, e.g. --prompt \"cup. bottle.\""))
	}
	return pipeline.TextPhrases(p.Text)
}

func (g *graspModel) Name() string      { return g.cfg.Name }
func (g *graspModel) Task() models.Task { return models.TaskGrasp }

// Roles: encoder+decoder (segmenter) plus the detector session when present.
func (g *graspModel) Roles() []string {
	roles := g.sam.Roles()
	if g.p.Detector != nil {
		roles = append([]string{roleDet}, roles...)
	}
	return roles
}

// PoolSizes delegates to the segmenter (MobileSAM pools the decoder).
func (g *graspModel) PoolSizes() map[string]int {
	if ps, ok := g.sam.(models.PoolSizer); ok {
		return ps.PoolSizes()
	}
	return nil
}

// Exclusive implements models.Exclusive for grasp-gd: lifecycle runs one Infer at a time on it,
// as on every GroundingDINO pipeline (see the groundingdino package doc); other models are not
// held up by it.
func (g *graspModel) Exclusive() bool { return g.serialize }

// Infer runs the pipeline. With a detector: detect → size-filter → segment per box
// → grasp per mask (class/conf inherited from the detection). Without a detector:
// the request's boxes, or automask → size-filter → grasp per mask (class-agnostic).
func (g *graspModel) Infer(img image.Image, prompt models.Prompt, r models.Runner) (models.Result, error) {
	return g.p.Infer(pipeline.Call{Img: img, Prompt: prompt, Runner: r})
}
