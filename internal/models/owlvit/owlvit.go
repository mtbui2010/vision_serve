// Package owlvit implements OWL-ViT / OWLv2 (Apache-2.0) for image-conditioned one-shot
// (instance) detection — a fully free community feature (no AGPL, no Python at runtime;
// see CLAUDE.md).
//
// OWLv2 is a PipelineModel with a single ONNX session (role "model"). It performs
// image-guided detection: given a scene image and one or more template images, it returns
// bounding boxes of scene regions that look like the templates.
//
// Verified I/O shapes (owlv2-base-patch16-ensemble, 960×960):
//
//	inputs:  query_pixel_values   [1, 3, H, W]  float32  — scene image (CLIP-normalized)
//	         query_image_features [N, 3, H, W]  float32  — N template images (same norm)
//	outputs: logits               [N, P, 1]     float32  — similarity per template per patch
//	         pred_boxes           [1, P, 4]     float32  — [cx, cy, w, h] normalized [0,1]
//	                                                     relative to the PADDED square input
//
// where P = (H/patch_size)^2.  For owlv2-base-patch16 at 960×960: P=3600.
//
// See docs/owlvit-export.md for the PyTorch ONNX export recipe.
package owlvit

import (
	"fmt"
	"image"
	"math"
	"strings"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/vision/geom"
	"visionserve/internal/vision/nms"
	"visionserve/pkg/api"
)

// Compile-time checks of the interfaces lifecycle type-asserts at load: a signature drift
// fails the build instead of silently changing how the model is run.
var _ models.PipelineModel = (*owlVIT)(nil)

func init() {
	models.Register("owlvit", New)
}

// CLIP normalization constants — the default for OWL-ViT / OWLv2 (from the paper + HF config).
// If the manifest declares normalize.mean/std, those take precedence (see preprocessImage).
var (
	clipMean = [3]float32{0.48145466, 0.4578275, 0.40821073}
	clipStd  = [3]float32{0.26862954, 0.26130258, 0.27577711}
)

// Default inference constants when the manifest leaves them zero.
const (
	defaultSimThreshold = 0.1
	defaultPatchSize    = 16
)

const roleModel = "model"

type owlVIT struct {
	cfg models.Config

	simThreshold float64 // minimum sigmoid(logit) to emit a detection
	maxTemplates int     // cap on template count per call (0 = no limit)
	patchSize    int     // ViT patch size (determines num_patches from input H/W)

	// resolved normalization (from manifest or CLIP defaults)
	mean [3]float32
	std  [3]float32
}

// New builds an owlVIT model from the manifest config.
// Called by models.Register("owlvit", New) — the lifecycle manager creates the ONNX
// session separately; this factory only sets up the pre/postprocess state.
func New(cfg models.Config) (models.Base, error) {
	m := &owlVIT{
		cfg:          cfg,
		simThreshold: cfg.InstanceSimThreshold,
		maxTemplates: cfg.InstanceMaxTemplates,
		patchSize:    cfg.InstancePatchSize,
	}
	if m.simThreshold <= 0 {
		m.simThreshold = defaultSimThreshold
	}
	if m.patchSize <= 0 {
		m.patchSize = defaultPatchSize
	}
	// Conf threshold from postprocess block takes precedence over instance block.
	if cfg.ConfThresh > 0 {
		m.simThreshold = cfg.ConfThresh
	}

	// Resolve normalization: manifest → CLIP defaults.
	if len(cfg.Mean) == 3 && len(cfg.Std) == 3 {
		copy(m.mean[:], cfg.Mean)
		copy(m.std[:], cfg.Std)
	} else {
		m.mean = clipMean
		m.std = clipStd
	}
	return m, nil
}

func (m *owlVIT) Name() string      { return m.cfg.Name }
func (m *owlVIT) Task() models.Task { return api.TaskInstanceDetection }

// Roles declares the single ONNX session this model needs.
func (m *owlVIT) Roles() []string { return []string{roleModel} }

// Infer runs the full one-shot detection pipeline.
//
// OWLv2 unrolls its template loop at ONNX trace time, so the exported model
// is fixed at N=1 template per forward pass. We run one inference per template
// and keep the maximum similarity score per spatial patch, then threshold to
// produce detections. pred_boxes is taken from the first run (they depend only
// on the scene image, not the templates).
func (m *owlVIT) Infer(img image.Image, prompt models.Prompt, r models.Runner) (models.Result, error) {
	if len(prompt.TemplateImages) == 0 {
		return models.Result{}, models.BadPrompt(fmt.Errorf(
			"owlvit: no template images in prompt — register templates via /api/templates first",
		))
	}

	queryTensor, meta, err := m.preprocessImage(img)
	if err != nil {
		return models.Result{}, fmt.Errorf("owlvit: query preprocess: %w", err)
	}

	templates := prompt.TemplateImages
	if m.maxTemplates > 0 && len(templates) > m.maxTemplates {
		templates = templates[:m.maxTemplates]
	}

	outNames := r.OutputNames(roleModel)

	var (
		patchScores []float64 // max similarity per patch across all templates
		boxesData   []float32 // pred_boxes from first run (scene-only, stable)
		numPatches  int
	)

	for i, tmpl := range templates {
		tmplTensor, _, err := m.preprocessImage(tmpl)
		if err != nil {
			return models.Result{}, fmt.Errorf("owlvit: template[%d] preprocess: %w", i, err)
		}

		outs, err := r.Run(roleModel, map[string]engine.Tensor{
			"query_pixel_values":   queryTensor,
			"query_image_features": tmplTensor, // [1, 3, H, W] — N=1 fixed export
		})
		if err != nil {
			return models.Result{}, fmt.Errorf("owlvit: inference template[%d]: %w", i, err)
		}

		logits, boxes := pickLogitsAndBoxes(outNames, outs)
		if logits == nil || boxes == nil {
			return models.Result{}, fmt.Errorf(
				"owlvit: cannot identify logits/pred_boxes (shapes %v)", shapesOf(outs),
			)
		}

		// logits shape [1, P, 1] for N=1 export: flat index p = score for patch p.
		P := int(logits.Dim(1))
		if i == 0 {
			numPatches = P
			patchScores = make([]float64, P)
			boxesData = boxes.Data
		}

		for p := 0; p < P && p < numPatches; p++ {
			s := geom.Sigmoid(float64(logits.Data[p]))
			if s > patchScores[p] {
				patchScores[p] = s
			}
		}
	}

	return m.postprocess(patchScores, boxesData, numPatches, meta, m.threshold(prompt))
}

// threshold is the score a patch must beat for this request: the request's box_threshold when it
// sets one, else the manifest's (conf_threshold / sim_threshold, default 0.1). Image-guided
// scores are a raw sigmoid and not calibrated: HF's image-guided example keeps boxes above 0.9
// (Owlv2ForObjectDetection.image_guided_detection docstring: threshold=0.9, nms_threshold=0.3),
// against 0.1 for text queries, so a caller that gets background boxes raises this.
func (m *owlVIT) threshold(prompt models.Prompt) float64 {
	if prompt.BoxThresh > 0 {
		return prompt.BoxThresh
	}
	return m.simThreshold
}

// preprocessImage reproduces HF Owlv2ImageProcessor: rescale to [0,1], pad to a square
// (bottom/right, black) WITHOUT changing the aspect ratio, anti-aliased bilinear resize to
// cfg.Width×cfg.Width, CLIP-normalize → NCHW float32 [1, 3, H, W] (see padResizeOWLv2).
//
// OWLv2 pred_boxes are normalized to the PADDED square, so the returned meta carries the
// single scale input/S (S = max(origW, origH)) on both axes and no offset (pad is on the
// bottom/right): orig = norm * S. Squashing to 960×960 instead (the previous behaviour)
// distorts the aspect ratio the model was trained on.
func (m *owlVIT) preprocessImage(img image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	W, H := m.cfg.Width, m.cfg.Height
	if W <= 0 || H <= 0 {
		return engine.Tensor{}, models.PreprocessMeta{}, fmt.Errorf(
			"owlvit: manifest input.width/height must be > 0 (got %d×%d)", W, H,
		)
	}
	if W != H {
		return engine.Tensor{}, models.PreprocessMeta{}, fmt.Errorf(
			"owlvit: input must be square (OWLv2 pads to a square), got %d×%d", W, H,
		)
	}

	origBounds := img.Bounds()
	origW := origBounds.Dx()
	origH := origBounds.Dy()
	if origW <= 0 || origH <= 0 {
		return engine.Tensor{}, models.PreprocessMeta{}, fmt.Errorf("owlvit: empty image %d×%d", origW, origH)
	}

	data, side := padResizeOWLv2(img, W)
	plane := W * H
	for c := 0; c < 3; c++ {
		ch := data[c*plane : (c+1)*plane]
		for i, v := range ch {
			ch[i] = (v - m.mean[c]) / m.std[c]
		}
	}

	s := float64(W) / float64(side)
	meta := models.PreprocessMeta{
		OrigWidth:  origW,
		OrigHeight: origH,
		ScaleX:     s,
		ScaleY:     s,
		PadX:       0,
		PadY:       0,
	}
	return engine.F32(data, 1, 3, int64(H), int64(W)), meta, nil
}

// postprocess converts aggregated per-patch scores and pred_boxes into api.Detections.
// patchScores[p] = max sigmoid(logit) over all templates for patch p.
// boxesData is the flat [1, P, 4] tensor (cxcywh normalized [0,1]).
func (m *owlVIT) postprocess(
	patchScores []float64,
	boxesData []float32,
	numPatches int,
	meta models.PreprocessMeta,
	thresh float64,
) (models.Result, error) {
	maxDet := m.cfg.MaxDet
	if maxDet <= 0 {
		maxDet = 100
	}

	if len(boxesData) < numPatches*4 || len(patchScores) < numPatches {
		return models.Result{}, fmt.Errorf("owlvit: pred_boxes has %d values, scores %d, want %d patches",
			len(boxesData), len(patchScores), numPatches)
	}
	if meta.ScaleX <= 0 {
		return models.Result{}, fmt.Errorf("owlvit: invalid preprocess scale %v", meta.ScaleX)
	}

	origW := float64(meta.OrigWidth)
	origH := float64(meta.OrigHeight)
	// Boxes are normalized to the padded square of side S = input/scale = max(origW, origH).
	side := float64(m.cfg.Width) / meta.ScaleX

	// Collect ALL patches above threshold (boxes mapped to original pixels and clamped to the
	// image), sort by confidence, NMS, THEN cut to maxDet. Cutting before NMS (the previous
	// order) spent the maxDet budget on near-duplicates of the strongest instance and
	// dropped the other instances.
	dets := make([]api.Detection, 0, 64)
	for p := 0; p < numPatches; p++ {
		if !(patchScores[p] > thresh) {
			continue
		}
		bb := boxesData[p*4 : p*4+4]
		cx, cy, bw, bh := float64(bb[0]), float64(bb[1]), float64(bb[2]), float64(bb[3])
		x1 := math.Max(0, (cx-bw/2)*side)
		y1 := math.Max(0, (cy-bh/2)*side)
		x2 := math.Min(origW, (cx+bw/2)*side)
		y2 := math.Min(origH, (cy+bh/2)*side)
		if !(x2 > x1 && y2 > y1) { // also rejects NaN
			continue
		}
		dets = append(dets, api.Detection{
			BBox:  [4]float64{x1, y1, x2 - x1, y2 - y1},
			Class: "object",
			Conf:  patchScores[p],
		})
	}
	// Greedy NMS on max(IoU, containment) — a large background box that encloses a tight
	// object box is suppressed even when their plain IoU is low — then cut to maxDet.
	dets = nms.Detections(dets, nms.Options{IoU: 0.5, ClassAgnostic: true, Containment: true, TopK: maxDet})
	return models.Result{Detections: dets}, nil
}

// pickLogitsAndBoxes maps outputs to logits ([.,.,N]) and pred_boxes ([.,.,4]) by name
// first, then by last-dim shape (4 → boxes; anything else → logits).
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
			default:
				if logits == nil {
					logits = &outs[i]
				}
			}
		}
	}
	return logits, boxes
}

func shapesOf(ts []engine.Tensor) [][]int64 {
	out := make([][]int64, len(ts))
	for i, t := range ts {
		out[i] = t.Shape
	}
	return out
}
