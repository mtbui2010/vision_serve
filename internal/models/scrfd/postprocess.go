package scrfd

import (
	"fmt"
	"math"

	"visionserve/internal/engine"
	"visionserve/internal/imageproc"
	"visionserve/internal/models"
)

// scrfdStride describes one scale level of SCRFD's feature pyramid.
type scrfdStride struct {
	stride      int
	numAnchors  int // anchors per location (always 2 for SCRFD-10GF)
	numProposals int // total proposals = (inputH/stride) * (inputW/stride) * numAnchors
}

// knownStrides lists the three SCRFD-10GF strides at 640×640 input.
// Each stride s has grid (640/s) × (640/s) and 2 anchors per cell.
var knownStrides = []scrfdStride{
	{stride: 8, numAnchors: 2, numProposals: 12800}, // 80*80*2
	{stride: 16, numAnchors: 2, numProposals: 3200},  // 40*40*2
	{stride: 32, numAnchors: 2, numProposals: 800},   // 20*20*2
}

// postprocess decodes raw SCRFD outputs into face detections.
//
// ===================== OUTPUT FORMAT =====================
// The InsightFace official det_10g.onnx has 6 or 9 outputs (without/with kps):
//
//	score_8  [1, 12800, 1]  — face probability (raw logit, apply sigmoid)
//	score_16 [1,  3200, 1]
//	score_32 [1,   800, 1]
//	bbox_8   [1, 12800, 4]  — distances (l,r,t,b) in stride units
//	bbox_16  [1,  3200, 4]
//	bbox_32  [1,   800, 4]
//	kps_8    [1, 12800, 10] — 5 keypoints × (dx,dy), optional
//	kps_16   [1,  3200, 10] — optional
//	kps_32   [1,   800, 10] — optional
//
// Tensors are identified by (dim1, dim2) — NOT by name — so different export names
// won't break decoding.
// =========================================================
func postprocess(outs []engine.Tensor, meta models.PreprocessMeta, cfg models.Config) (models.Result, error) {
	// Step 1: bucket output tensors by stride using (numProposals, lastDim).
	type perStride struct {
		score *engine.Tensor // [1, N, 1]
		bbox  *engine.Tensor // [1, N, 4]
		// kps ignored — not reflected in the Result schema
	}
	strideBuckets := make(map[int]*perStride, len(knownStrides))
	for i := range knownStrides {
		strideBuckets[knownStrides[i].numProposals] = &perStride{}
	}

	for i := range outs {
		t := &outs[i]
		if len(t.Shape) < 2 {
			continue
		}
		// For output tensors the shape is [1, N, C] with batch=1.
		// Dim(1) = N (proposals), Dim(-1) = last dimension.
		n := int(t.Dim(1))
		last := int(t.Dim(-1))

		bucket, known := strideBuckets[n]
		if !known {
			continue // unknown proposal count — skip (e.g. dynamic batch shape variation)
		}
		switch last {
		case 1:
			bucket.score = t
		case 4:
			bucket.bbox = t
		// case 10: kps — intentionally ignored
		}
	}

	// Step 2: decode all strides.
	confThresh := cfg.ConfThresh
	if confThresh <= 0 {
		confThresh = 0.5
	}
	maxDet := cfg.MaxDet
	if maxDet <= 0 {
		maxDet = 1000
	}

	// inputW / inputH — the letterboxed space (e.g. 640×640).
	inputW, inputH := cfg.Width, cfg.Height

	var candidates []models.Detection

	for _, sd := range knownStrides {
		bucket := strideBuckets[sd.numProposals]
		if bucket.score == nil || bucket.bbox == nil {
			// Stride tensors missing — skip rather than error; some lightweight SCRFD
			// variants omit certain strides (e.g. the stride-32 head at very small
			// resolutions). A complete det_10g.onnx will always have all three.
			continue
		}

		dets, err := decodeStride(bucket.score, bucket.bbox, sd, inputW, inputH, confThresh)
		if err != nil {
			return models.Result{}, fmt.Errorf("scrfd: decode stride %d: %w", sd.stride, err)
		}
		candidates = append(candidates, dets...)
	}

	if len(candidates) == 0 {
		return models.Result{Detections: []models.Detection{}}, nil
	}

	// Step 3: NMS (all faces share the same class "face").
	kept := imageproc.NMS(candidates, 0.4)

	// Step 4: map from 640-space (letterboxed) back to ORIGINAL image coordinates.
	for i := range kept {
		ox, oy, ow, oh := mapToOrig(kept[i].BBox, meta)
		ox, oy, ow, oh = clampBox(ox, oy, ow, oh, meta.OrigWidth, meta.OrigHeight)
		kept[i].BBox = [4]float64{ox, oy, ow, oh}
	}

	// Step 5: cut to max_detections (NMS already sorted by conf).
	if maxDet > 0 && len(kept) > maxDet {
		kept = kept[:maxDet]
	}

	return models.Result{Detections: kept}, nil
}

// decodeStride decodes proposals for one pyramid level.
//
// Anchor layout: for a grid of H×W cells with A anchors per cell, proposal index k is:
//
//	row   i = k / (W * A)
//	colMod  = k % (W * A)
//	col   j = colMod / A
//	anch  a = colMod % A
//
// Both anchors at (i,j) share the same center: cx=(j+0.5)*stride, cy=(i+0.5)*stride.
//
// dist2bbox (distances l,r,t,b in stride units):
//
//	x1 = cx - l*stride   y1 = cy - t*stride
//	x2 = cx + r*stride   y2 = cy + b*stride
//	→ xywh in pixels of the 640-space input image
func decodeStride(
	scoreT, bboxT *engine.Tensor,
	sd scrfdStride,
	inputW, inputH int,
	confThresh float64,
) ([]models.Detection, error) {
	n := sd.numProposals
	if int(scoreT.Dim(1)) != n || int(bboxT.Dim(1)) != n {
		return nil, fmt.Errorf("expected %d proposals, got score=%d bbox=%d",
			n, scoreT.Dim(1), bboxT.Dim(1))
	}

	stride := sd.stride
	numAnchors := sd.numAnchors

	// grid width in cells
	gridW := inputW / stride
	// gridH := inputH / stride  // implicit from numProposals / (gridW * numAnchors)

	scores := scoreT.Data // length n*1 = n
	bboxes := bboxT.Data  // length n*4

	dets := make([]models.Detection, 0, n/4) // rough pre-alloc
	for k := 0; k < n; k++ {
		rawScore := float64(scores[k]) // shape [1,N,1] → flat index = k
		prob := sigmoid(rawScore)
		if prob < confThresh {
			continue
		}

		// Anchor center
		anchorInRow := k % (gridW * numAnchors)
		row := k / (gridW * numAnchors)
		col := anchorInRow / numAnchors
		// anchor index within cell (0 or 1) — same center for both anchors in SCRFD
		cx := (float64(col) + 0.5) * float64(stride)
		cy := (float64(row) + 0.5) * float64(stride)

		// Distances (already in stride units per SCRFD convention)
		base := k * 4
		l := float64(bboxes[base])
		t := float64(bboxes[base+1])
		r := float64(bboxes[base+2])
		b := float64(bboxes[base+3])

		x1 := cx - l*float64(stride)
		y1 := cy - t*float64(stride)
		x2 := cx + r*float64(stride)
		y2 := cy + b*float64(stride)

		// Clamp to the letterboxed input image bounds before mapping back.
		x1 = math.Max(0, math.Min(x1, float64(inputW)))
		y1 = math.Max(0, math.Min(y1, float64(inputH)))
		x2 = math.Max(0, math.Min(x2, float64(inputW)))
		y2 = math.Max(0, math.Min(y2, float64(inputH)))

		w := x2 - x1
		h := y2 - y1
		if w <= 0 || h <= 0 {
			continue
		}

		dets = append(dets, models.Detection{
			BBox:  [4]float64{x1, y1, w, h},
			Class: "face",
			Conf:  prob,
		})
	}
	return dets, nil
}

// mapToOrig maps a box [x,y,w,h] from letterboxed-640 space back to original image coords.
// Relation (from letterbox.go): input_coord = orig_coord * Scale + Pad
// Inverse:                       orig_coord  = (input_coord - Pad) / Scale
func mapToOrig(box [4]float64, meta models.PreprocessMeta) (ox, oy, ow, oh float64) {
	ox = (box[0] - float64(meta.PadX)) / meta.ScaleX
	oy = (box[1] - float64(meta.PadY)) / meta.ScaleY
	ow = box[2] / meta.ScaleX
	oh = box[3] / meta.ScaleY
	return
}

// clampBox ensures the box stays within the original image boundaries.
func clampBox(x, y, w, h float64, maxW, maxH int) (float64, float64, float64, float64) {
	if x < 0 {
		w += x
		x = 0
	}
	if y < 0 {
		h += y
		y = 0
	}
	if x+w > float64(maxW) {
		w = float64(maxW) - x
	}
	if y+h > float64(maxH) {
		h = float64(maxH) - y
	}
	if w < 0 {
		w = 0
	}
	if h < 0 {
		h = 0
	}
	return x, y, w, h
}

func sigmoid(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }
