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

	"github.com/disintegration/imaging"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/pkg/api"
)

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
		return models.Result{}, fmt.Errorf(
			"owlvit: no template images in prompt — register templates via /api/templates first",
		)
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
			s := sigmoid(logits.Data[p])
			if s > patchScores[p] {
				patchScores[p] = s
			}
		}
	}

	return m.postprocess(patchScores, boxesData, numPatches, meta)
}

// preprocessImage resizes img to cfg.Width×cfg.Height (plain squash, no letterbox),
// normalizes with CLIP mean/std, and returns an NCHW float32 tensor [1, 3, H, W].
// The returned PreprocessMeta carries the mapping back to original image coordinates.
func (m *owlVIT) preprocessImage(img image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	W, H := m.cfg.Width, m.cfg.Height
	if W <= 0 || H <= 0 {
		return engine.Tensor{}, models.PreprocessMeta{}, fmt.Errorf(
			"owlvit: manifest input.width/height must be > 0 (got %d×%d)", W, H,
		)
	}

	origBounds := img.Bounds()
	origW := origBounds.Dx()
	origH := origBounds.Dy()

	resized := imaging.Resize(img, W, H, imaging.Linear)

	plane := W * H
	data := make([]float32, 3*plane)
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			c := resized.NRGBAAt(resized.Bounds().Min.X+x, resized.Bounds().Min.Y+y)
			idx := y*W + x
			data[idx] = (float32(c.R)/255.0 - m.mean[0]) / m.std[0]
			data[plane+idx] = (float32(c.G)/255.0 - m.mean[1]) / m.std[1]
			data[2*plane+idx] = (float32(c.B)/255.0 - m.mean[2]) / m.std[2]
		}
	}

	meta := models.PreprocessMeta{
		OrigWidth:  origW,
		OrigHeight: origH,
		ScaleX:     float64(W) / float64(origW),
		ScaleY:     float64(H) / float64(origH),
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
) (models.Result, error) {
	maxDet := m.cfg.MaxDet
	if maxDet <= 0 {
		maxDet = 100
	}

	origW := float64(meta.OrigWidth)
	origH := float64(meta.OrigHeight)

	// Collect ALL patches above threshold, then sort by confidence and take top maxDet.
	type candidate struct {
		score float64
		patch int
	}
	var cands []candidate
	for p := 0; p < numPatches; p++ {
		if patchScores[p] > m.simThreshold {
			cands = append(cands, candidate{patchScores[p], p})
		}
	}
	// Sort descending by score.
	for i := 0; i < len(cands)-1; i++ {
		for j := i + 1; j < len(cands); j++ {
			if cands[j].score > cands[i].score {
				cands[i], cands[j] = cands[j], cands[i]
			}
		}
	}
	if len(cands) > maxDet {
		cands = cands[:maxDet]
	}

	dets := make([]api.Detection, 0, len(cands))
	for _, c := range cands {
		bb := boxesData[c.patch*4 : c.patch*4+4]
		cx, cy, bw, bh := float64(bb[0]), float64(bb[1]), float64(bb[2]), float64(bb[3])
		dets = append(dets, api.Detection{
			BBox:  [4]float64{(cx - bw/2) * origW, (cy - bh/2) * origH, bw * origW, bh * origH},
			Class: "object",
			Conf:  c.score,
		})
	}
	return models.Result{Detections: nmsDetections(dets, 0.5)}, nil
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


// nmsDetections applies greedy NMS on boxes sorted by confidence (desc).
// Uses max(IoU, containment) as the suppression score so that a large background
// box that encloses a tight object box is suppressed even when standard IoU is low.
func nmsDetections(dets []api.Detection, iouThresh float64) []api.Detection {
	keep := make([]api.Detection, 0, len(dets))
	suppressed := make([]bool, len(dets))
	for i := range dets {
		if suppressed[i] {
			continue
		}
		keep = append(keep, dets[i])
		ax, ay, aw, ah := dets[i].BBox[0], dets[i].BBox[1], dets[i].BBox[2], dets[i].BBox[3]
		aArea := aw * ah
		for j := i + 1; j < len(dets); j++ {
			if suppressed[j] {
				continue
			}
			bx, by, bw, bh := dets[j].BBox[0], dets[j].BBox[1], dets[j].BBox[2], dets[j].BBox[3]
			ix := math.Max(ax, bx)
			iy := math.Max(ay, by)
			iw := math.Min(ax+aw, bx+bw) - ix
			ih := math.Min(ay+ah, by+bh) - iy
			if iw <= 0 || ih <= 0 {
				continue
			}
			inter := iw * ih
			bArea := bw * bh
			unionIoU := inter / (aArea + bArea - inter)
			// containment: fraction of the smaller box covered by intersection
			minArea := math.Min(aArea, bArea)
			containIoU := 0.0
			if minArea > 0 {
				containIoU = inter / minArea
			}
			if math.Max(unionIoU, containIoU) > iouThresh {
				suppressed[j] = true
			}
		}
	}
	return keep
}

func sigmoid(x float32) float64 {
	return 1.0 / (1.0 + math.Exp(-float64(x)))
}

func shapesOf(ts []engine.Tensor) [][]int64 {
	out := make([][]int64, len(ts))
	for i, t := range ts {
		out[i] = t.Shape
	}
	return out
}
