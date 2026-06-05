// Package maskgrasp turns segmentation masks into planar parallel-jaw grasps.
//
// It is a PipelineModel that REUSES MobileSAM's encoder+decoder sessions (owned
// and kept alive by lifecycle.Manager, like Grounded-SAM) and appends a final,
// pure-Go ANALYTIC stage — no extra ONNX session, no depth, no weights. The
// grasp stage is an antipodal force-closure search on each mask's boundary
// normals (see internal/grasp), so a box/point/text prompt that yields a mask
// also yields grasps for that object.
//
// Pipeline: prompt → MobileSAM → masks (unified column-major RLE) → decode each
// mask → grasp.FromMask → Result.Grasps. Task is reported as TaskGrasp.
package maskgrasp

import (
	"fmt"
	"image"
	"strconv"
	"strings"

	"visionserve/internal/grasp"
	"visionserve/internal/models"
	"visionserve/internal/models/mobilesam"
)

func init() {
	models.Register("mask-grasp", New)
}

// maskGrasp wraps a MobileSAM model and post-processes its masks into grasps.
type maskGrasp struct {
	cfg    models.Config
	sam    models.PipelineModel // the underlying MobileSAM (encoder+decoder)
	params grasp.Params
}

// New builds a mask-grasp model. It constructs a MobileSAM from the SAME manifest
// (the encoder/decoder files in cfg.Files) and reuses it; lifecycle still owns the
// ONNX sessions via the roles MobileSAM declares.
func New(cfg models.Config) (models.Base, error) {
	base, err := mobilesam.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("mask-grasp: %w", err)
	}
	sam, ok := base.(models.PipelineModel)
	if !ok {
		return nil, fmt.Errorf("mask-grasp: underlying mobilesam is not a PipelineModel")
	}
	return &maskGrasp{cfg: cfg, sam: sam, params: grasp.DefaultParams()}, nil
}

func (m *maskGrasp) Name() string      { return m.cfg.Name }
func (m *maskGrasp) Task() models.Task { return models.TaskGrasp }

// Roles / PoolSizes delegate to MobileSAM so lifecycle loads exactly the sessions
// the segmentation stage needs (encoder + decoder, decoder pooled).
func (m *maskGrasp) Roles() []string { return m.sam.Roles() }

func (m *maskGrasp) PoolSizes() map[string]int {
	if ps, ok := m.sam.(models.PoolSizer); ok {
		return ps.PoolSizes()
	}
	return nil
}

// Infer runs MobileSAM to obtain masks, then derives grasps from each mask. The
// returned Result keeps the masks (still useful) and adds Grasps; Task=TaskGrasp.
func (m *maskGrasp) Infer(img image.Image, prompt models.Prompt, r models.Runner) (models.Result, error) {
	res, err := m.sam.Infer(img, prompt, r)
	if err != nil {
		return res, err
	}
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	for _, mk := range res.Masks {
		bm, err := decodeRLEColumnMajor(mk.RLE, w, h)
		if err != nil {
			return models.Result{}, fmt.Errorf("mask-grasp: decode mask RLE: %w", err)
		}
		res.Grasps = append(res.Grasps, grasp.FromMask(bm, m.params)...)
	}
	res.Task = models.TaskGrasp
	return res, nil
}

// decodeRLEColumnMajor inverts the column-major RLE produced by the SAM models'
// encodeRLEColumnMajor: space-separated run lengths over a column-major (x outer,
// y inner) traversal of a w×h grid, alternating runs that START with background
// (false). The reconstructed Bitmap is row-major (Data[y*w+x]), matching
// grasp.Bitmap. An empty RLE yields an all-false mask.
func decodeRLEColumnMajor(rle string, w, h int) (grasp.Bitmap, error) {
	data := make([]bool, w*h)
	bm := grasp.Bitmap{W: w, H: h, Data: data}
	if rle == "" || w <= 0 || h <= 0 {
		return bm, nil
	}
	val := false // runs start with background, per the encoder
	idx := 0      // linear index along the column-major traversal
	total := w * h
	for _, f := range strings.Fields(rle) {
		n, err := strconv.Atoi(f)
		if err != nil {
			return grasp.Bitmap{}, fmt.Errorf("bad run length %q: %w", f, err)
		}
		if n < 0 {
			return grasp.Bitmap{}, fmt.Errorf("negative run length %d", n)
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
