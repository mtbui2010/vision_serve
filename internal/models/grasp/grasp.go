// Package grasp implements the "grasp" architecture: a configurable planar
// parallel-jaw grasp pipeline.
//
//	(optional detector) → segmenter (mask) → mask2grasp (analytic)
//
//   - Segmenter (mandatory, default "mobile-sam") turns the image into object
//     masks: box-prompted when a detector supplies boxes, or whole-image automask
//     when there is no detector.
//   - Detector (OPTIONAL) supplies boxes + class labels. Set it ("rf-detr",
//     "grounding-dino", …) for CLASS-AWARE grasps; omit it for CLASS-AGNOSTIC
//     grasps (automask over the whole image).
//   - The final stage is the pure-Go analytic mask2grasp search (internal/grasp);
//     it adds no ONNX session and no weights.
//
// Like Grounded-SAM, this model only ORCHESTRATES sessions owned by
// lifecycle.Manager (VRAM-safe): the detector session under role "det" and the
// MobileSAM encoder/decoder under "encoder"/"decoder". It reuses the existing
// model packages (groundingdino.Detect, MobileSAM's Infer, any plain detector's
// Preprocess/Postprocess) rather than duplicating their logic.
package grasp

import (
	"fmt"
	"image"
	"path/filepath"
	"strconv"
	"strings"

	"visionserve/internal/engine"
	graspcore "visionserve/internal/grasp"
	"visionserve/internal/models"
	"visionserve/internal/models/groundingdino"
	"visionserve/internal/models/mobilesam"
	"visionserve/pkg/api"
)

func init() {
	models.Register("grasp", New)
}

const (
	roleDet     = "det"
	roleEncoder = "encoder"
	roleDecoder = "decoder"
)

const (
	defaultSegmenter  = "mobile-sam"
	defaultBoxThresh  = 0.3
	defaultTextThresh = 0.25
)

// detector abstracts the optional box stage so the plain-Model detectors (rf-detr,
// rt-detr) and the text-prompted GroundingDINO can share one seam. Both drive a
// single ONNX session under role "det" via the Runner.
type detector interface {
	detect(img image.Image, prompt models.Prompt, r models.Runner) ([]models.Detection, error)
}

// graspModel is the composed pipeline.
type graspModel struct {
	cfg     models.Config
	sam     models.PipelineModel // segmenter (MobileSAM)
	det     detector             // optional; nil => class-agnostic
	gripMin float64              // manifest gripper defaults (px); 0 => core default
	gripMax float64
	// serialize is true only for the GroundingDINO detector variant (grasp-gd), which must
	// run one whole pipeline at a time (see groundingdino.PipelineMu). grasp-rfdetr and the
	// class-agnostic automask path stay fully concurrent.
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
	sam, ok := samBase.(models.PipelineModel)
	if !ok {
		return nil, fmt.Errorf("grasp: segmenter is not a PipelineModel")
	}

	g := &graspModel{cfg: cfg, sam: sam, gripMin: cfg.GripperMin, gripMax: cfg.GripperMax}

	if name := strings.TrimSpace(cfg.Detector); name != "" {
		if cfg.Files[roleDet] == "" {
			return nil, fmt.Errorf("grasp: detector %q set but files.%s is missing", name, roleDet)
		}
		det, err := newDetector(name, cfg)
		if err != nil {
			return nil, err
		}
		g.det = det
		if name == "grounding-dino" {
			g.serialize = true // GroundingDINO pipeline must be serialized (see PipelineMu)
		}
	}
	return g, nil
}

// newDetector picks the detector implementation by name. GroundingDINO is a
// text-prompted special case (needs a tokenizer + the Detect helper); everything
// else must be a registered plain Model driven via its Preprocess/Postprocess.
func newDetector(name string, cfg models.Config) (detector, error) {
	if name == "grounding-dino" {
		tok, err := groundingdino.LoadTokenizer(resolveVocab(cfg))
		if err != nil {
			return nil, fmt.Errorf("grasp: detector grounding-dino: %w", err)
		}
		box, text := cfg.ConfThresh, cfg.TextThresh
		if box <= 0 {
			box = defaultBoxThresh
		}
		if text <= 0 {
			text = defaultTextThresh
		}
		return &gdinoDetector{tok: tok, box: box, text: text}, nil
	}
	base, err := models.New(name, cfg)
	if err != nil {
		return nil, fmt.Errorf("grasp: detector %q: %w", name, err)
	}
	m, ok := base.(models.Model)
	if !ok {
		return nil, fmt.Errorf("grasp: detector %q is not a plain Model (use grounding-dino for text-prompted detectors)", name)
	}
	return &modelDetector{m: m}, nil
}

func (g *graspModel) Name() string      { return g.cfg.Name }
func (g *graspModel) Task() models.Task { return models.TaskGrasp }

// Roles: encoder+decoder (segmenter) plus the detector session when present.
func (g *graspModel) Roles() []string {
	roles := g.sam.Roles()
	if g.det != nil {
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

// bitmapSegmenter is the fast seam: a segmenter that hands back the raw mask
// bitmaps (original-image resolution) alongside the RLE-encoded masks, so the grasp
// search runs on the bitmap directly instead of decoding the RLE straight back.
// MobileSAM implements it; any segmenter that does not falls back to RLE decode.
type bitmapSegmenter interface {
	InferMasks(img image.Image, prompt models.Prompt, r models.Runner) ([]models.Mask, []mobilesam.MaskBitmap, error)
}

// Infer runs the pipeline. With a detector: detect → size-filter → segment per box
// → grasp per mask (class/conf inherited from the detection). Without a detector:
// automask → size-filter → grasp per mask (class-agnostic).
func (g *graspModel) Infer(img image.Image, prompt models.Prompt, r models.Runner) (models.Result, error) {
	if g.serialize {
		groundingdino.PipelineMu.Lock()
		defer groundingdino.PipelineMu.Unlock()
	}
	params := g.graspParams(prompt)
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	var res models.Result
	res.Task = models.TaskGrasp

	if g.det != nil {
		dets, err := g.det.detect(img, prompt, r)
		if err != nil {
			return models.Result{}, err
		}
		dets = filterDetections(dets, prompt, w, h)
		if len(dets) == 0 {
			return res, nil
		}
		boxes := make([][4]float64, len(dets))
		for i, d := range dets {
			boxes[i] = d.BBox
		}
		masks, bitmaps, err := g.segment(img, models.Prompt{Boxes: boxes}, r)
		if err != nil {
			return models.Result{}, err
		}
		for i := range bitmaps {
			gs := graspcore.FromMask(bitmaps[i], params)
			if i < len(dets) {
				for j := range gs {
					gs[j].Class = dets[i].Class
					gs[j].Conf = dets[i].Conf
				}
				if i < len(masks) {
					masks[i].BBox = dets[i].BBox
					masks[i].Conf = dets[i].Conf
				}
			}
			res.Grasps = append(res.Grasps, gs...)
		}
		res.Detections = dets
		res.Masks = masks
		return res, nil
	}

	// Box-prompted fast path (no detector): segment ONLY the requested boxes and grasp
	// each. This is the "select the target client-side (e.g. grounding-dino boxes →
	// select_target_object), then grasp just it" flow — one segmentation + one FromMask
	// per box, instead of automask + FromMask over EVERY object in the scene.
	if len(prompt.Boxes) > 0 {
		masks, bitmaps, err := g.segment(img, models.Prompt{Boxes: prompt.Boxes}, r)
		if err != nil {
			return models.Result{}, err
		}
		for i := range bitmaps {
			res.Grasps = append(res.Grasps, graspcore.FromMask(bitmaps[i], params)...)
		}
		res.Masks = masks
		return res, nil
	}

	// Class-agnostic: whole-image automask, then grasp each mask.
	masks, bitmaps, err := g.segment(img, models.Prompt{}, r)
	if err != nil {
		return models.Result{}, err
	}
	masks, bitmaps = filterMasksBitmaps(masks, bitmaps, prompt, w, h)
	for i := range bitmaps {
		res.Grasps = append(res.Grasps, graspcore.FromMask(bitmaps[i], params)...)
	}
	res.Masks = masks
	return res, nil
}

// segment runs the segmenter and returns the RLE masks (for the API response) plus
// the index-aligned raw bitmaps (for the grasp search). It prefers the bitmapSegmenter
// seam — one RLE encode, ZERO decode — and falls back to Infer + RLE decode for any
// segmenter that does not expose bitmaps.
func (g *graspModel) segment(img image.Image, prompt models.Prompt, r models.Runner) ([]models.Mask, []graspcore.Bitmap, error) {
	if bs, ok := g.sam.(bitmapSegmenter); ok {
		masks, bms, err := bs.InferMasks(img, prompt, r)
		if err != nil {
			return nil, nil, err
		}
		bitmaps := make([]graspcore.Bitmap, len(bms))
		for i := range bms {
			bitmaps[i] = graspcore.Bitmap{W: bms[i].W, H: bms[i].H, Data: bms[i].Data}
		}
		return masks, bitmaps, nil
	}

	// Fallback: encode→decode round-trip (defensive; mobile-sam implements the seam).
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	segRes, err := g.sam.Infer(img, prompt, r)
	if err != nil {
		return nil, nil, err
	}
	bitmaps := make([]graspcore.Bitmap, len(segRes.Masks))
	for i := range segRes.Masks {
		bm, err := decodeRLEColumnMajor(segRes.Masks[i].RLE, w, h)
		if err != nil {
			return nil, nil, fmt.Errorf("grasp: decode mask RLE: %w", err)
		}
		bitmaps[i] = bm
	}
	return segRes.Masks, bitmaps, nil
}

// graspParams resolves the gripper opening bounds: core defaults, overridden by
// the manifest, overridden again by the request (Prompt). MaxGrasps comes from the
// manifest's max_detections, if any.
func (g *graspModel) graspParams(prompt models.Prompt) graspcore.Params {
	p := graspcore.DefaultParams()
	if g.gripMin > 0 {
		p.Dmin = g.gripMin
	}
	if g.gripMax > 0 {
		p.Dmax = g.gripMax
	}
	if prompt.GripperMin > 0 {
		p.Dmin = prompt.GripperMin
	}
	if prompt.GripperMax > 0 {
		p.Dmax = prompt.GripperMax
	}
	if g.cfg.MaxDet > 0 {
		p.MaxGrasps = g.cfg.MaxDet
	}
	return p
}

func filterDetections(dets []models.Detection, prompt models.Prompt, w, h int) []models.Detection {
	if prompt.MinSize <= 0 && prompt.MaxSize <= 0 {
		return dets
	}
	return api.FilterBySizePct(api.Result{Detections: dets}, prompt.MinSize, prompt.MaxSize, w, h).Detections
}

// filterMasksBitmaps applies the same bbox-area size filter as api.FilterBySizePct,
// keeping the masks and their index-aligned bitmaps in lockstep (mask[i].BBox ==
// bitmap[i] bbox, both from the same MaskBitmap).
func filterMasksBitmaps(masks []models.Mask, bitmaps []graspcore.Bitmap, prompt models.Prompt, w, h int) ([]models.Mask, []graspcore.Bitmap) {
	if prompt.MinSize <= 0 && prompt.MaxSize <= 0 {
		return masks, bitmaps
	}
	area := float64(w * h)
	var minAbs, maxAbs float64
	if prompt.MinSize > 0 {
		minAbs = prompt.MinSize / 100.0 * area
	}
	if prompt.MaxSize > 0 {
		maxAbs = prompt.MaxSize / 100.0 * area
	}
	outM := make([]models.Mask, 0, len(masks))
	outB := make([]graspcore.Bitmap, 0, len(bitmaps))
	for i := range masks {
		a := masks[i].BBox[2] * masks[i].BBox[3]
		if minAbs > 0 && a < minAbs {
			continue
		}
		if maxAbs > 0 && a > maxAbs {
			continue
		}
		outM = append(outM, masks[i])
		if i < len(bitmaps) {
			outB = append(outB, bitmaps[i])
		}
	}
	return outM, outB
}

// --- detector implementations ---

// modelDetector drives a registered plain Model (rf-detr, rt-detr) via the Runner:
// the model's own Preprocess produces the input tensor and PreprocessMeta, the
// Runner executes role "det", and the model's Postprocess decodes the detections.
type modelDetector struct{ m models.Model }

func (d *modelDetector) detect(img image.Image, _ models.Prompt, r models.Runner) ([]models.Detection, error) {
	in, meta, err := d.m.Preprocess(img)
	if err != nil {
		return nil, fmt.Errorf("grasp: detector preprocess: %w", err)
	}
	inName := firstName(r.InputNames(roleDet), "")
	if inName == "" {
		return nil, fmt.Errorf("grasp: detector session %q has no input name", roleDet)
	}
	outs, err := r.Run(roleDet, map[string]engine.Tensor{inName: in})
	if err != nil {
		return nil, fmt.Errorf("grasp: detector inference: %w", err)
	}
	res, err := d.m.Postprocess(outs, meta)
	if err != nil {
		return nil, fmt.Errorf("grasp: detector postprocess: %w", err)
	}
	return res.Detections, nil
}

// gdinoDetector runs GroundingDINO (text-prompted) for boxes + labels.
type gdinoDetector struct {
	tok       *groundingdino.Tokenizer
	box, text float64
}

func (d *gdinoDetector) detect(img image.Image, prompt models.Prompt, r models.Runner) ([]models.Detection, error) {
	if strings.TrimSpace(prompt.Text) == "" {
		return nil, fmt.Errorf("grasp: the grounding-dino detector requires a text prompt, e.g. --prompt \"cup. bottle.\"")
	}
	run := func(inputs map[string]engine.Tensor) ([]engine.Tensor, error) {
		return r.Run(roleDet, inputs)
	}
	// Per-request threshold overrides (>0) take precedence over the manifest-derived defaults.
	box, text := d.box, d.text
	if prompt.BoxThresh > 0 {
		box = prompt.BoxThresh
	}
	if prompt.TextThresh > 0 {
		text = prompt.TextThresh
	}
	return groundingdino.Detect(img, prompt.Text, d.tok, run, r.OutputNames(roleDet), box, text)
}

// --- helpers ---

// resolveVocab finds vocab.txt for GroundingDINO: prefer the directory of the
// detector weights (files.det, typically "../grounding-dino/model.onnx"), fall
// back to the model directory.
func resolveVocab(cfg models.Config) string {
	if det := cfg.Files[roleDet]; det != "" {
		return filepath.Join(filepath.Dir(det), "vocab.txt")
	}
	return filepath.Join(cfg.Dir, "vocab.txt")
}

func firstName(names []string, fallback string) string {
	if len(names) > 0 {
		return names[0]
	}
	return fallback
}

// decodeRLEColumnMajor inverts the column-major RLE produced by the SAM models'
// encodeRLEColumnMajor: space-separated run lengths over a column-major (x outer,
// y inner) traversal of a w×h grid, alternating runs that START with background
// (false). The reconstructed Bitmap is row-major (Data[y*w+x]). Empty => all-false.
func decodeRLEColumnMajor(rle string, w, h int) (graspcore.Bitmap, error) {
	data := make([]bool, w*h)
	bm := graspcore.Bitmap{W: w, H: h, Data: data}
	if rle == "" || w <= 0 || h <= 0 {
		return bm, nil
	}
	val := false
	idx := 0
	total := w * h
	for _, f := range strings.Fields(rle) {
		n, err := strconv.Atoi(f)
		if err != nil {
			return graspcore.Bitmap{}, fmt.Errorf("bad run length %q: %w", f, err)
		}
		if n < 0 {
			return graspcore.Bitmap{}, fmt.Errorf("negative run length %d", n)
		}
		for k := 0; k < n && idx < total; k++ {
			x := idx / h
			y := idx % h
			data[y*w+x] = val
			idx++
		}
		val = !val
	}
	return bm, nil
}
