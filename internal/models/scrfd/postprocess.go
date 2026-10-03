package scrfd

import (
	"fmt"
	"math"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/vision/geom"
	"visionserve/internal/vision/nms"
)

// scrfdStride describes one scale level of SCRFD's feature pyramid.
type scrfdStride struct {
	stride       int
	numAnchors   int // anchors per location (always 2 for SCRFD-10GF)
	numProposals int // total proposals = (inputH/stride) * (inputW/stride) * numAnchors
}

// scrfdStrides / scrfdNumAnchors are the SCRFD-10GF pyramid (InsightFace
// _feat_stride_fpn = [8, 16, 32], _num_anchors = 2 for the 3-level variant).
var scrfdStrides = []int{8, 16, 32}

const scrfdNumAnchors = 2

// stridesFor derives each level's proposal count from the ACTUAL network input size
// (the det_10g.onnx input is dynamic [1,3,?,?]), exactly like InsightFace:
// height = input_height // stride, width = input_width // stride.
func stridesFor(inputW, inputH int) []scrfdStride {
	out := make([]scrfdStride, 0, len(scrfdStrides))
	for _, s := range scrfdStrides {
		out = append(out, scrfdStride{
			stride:       s,
			numAnchors:   scrfdNumAnchors,
			numProposals: (inputH / s) * (inputW / s) * scrfdNumAnchors,
		})
	}
	return out
}

// postprocess decodes raw SCRFD outputs into face detections.
//
// ===================== OUTPUT FORMAT (verified on det_10g.onnx) =====================
// The InsightFace det_10g.onnx (buffalo_l pack) has 9 outputs, all 2-D (NO batch dim):
//
//	score_8  [12800, 1]  — face probability (the graph ends in a Sigmoid op: do NOT
//	score_16 [ 3200, 1]    sigmoid again)
//	score_32 [  800, 1]
//	bbox_8   [12800, 4]  — distances (l,t,r,b) in stride units
//	bbox_16  [ 3200, 4]
//	bbox_32  [  800, 4]
//	kps_8    [12800, 10] — 5 keypoints × (dx,dy) (ignored: not in the Result schema)
//	kps_16   [ 3200, 10]
//	kps_32   [  800, 10]
//
// (counts shown for a 640×640 input; they scale with the input size.) Some exports keep a
// leading batch dim ([1,N,C]); both are accepted — N is read as Dim(-2), C as Dim(-1).
// Tensors are identified by (N, C) — NOT by name — so different export names won't break
// decoding.
// =====================================================================================
func postprocess(outs []engine.Tensor, meta models.PreprocessMeta, cfg models.Config) (models.Result, error) {
	// inputW / inputH — the letterboxed network input space (e.g. 640×640).
	inputW, inputH := cfg.Width, cfg.Height
	levels := stridesFor(inputW, inputH)

	// Step 1: bucket output tensors by stride using (numProposals, lastDim).
	type perStride struct {
		score *engine.Tensor // [N, 1] or [1, N, 1]
		bbox  *engine.Tensor // [N, 4] or [1, N, 4]
		// kps ignored — not reflected in the Result schema
	}
	strideBuckets := make(map[int]*perStride, len(levels))
	for _, sd := range levels {
		strideBuckets[sd.numProposals] = &perStride{}
	}

	for i := range outs {
		t := &outs[i]
		if len(t.Shape) < 2 || len(t.Shape) > 3 {
			continue
		}
		if len(t.Shape) == 3 && t.Shape[0] != 1 {
			return models.Result{}, fmt.Errorf("scrfd: batch size %d not supported (output shape %v)", t.Shape[0], t.Shape)
		}
		n := int(t.Dim(-2))
		last := int(t.Dim(-1))

		bucket, known := strideBuckets[n]
		if !known {
			continue // proposal count does not match any level for this input size
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

	var candidates []models.Detection
	decoded := 0
	for _, sd := range levels {
		bucket := strideBuckets[sd.numProposals]
		if bucket.score == nil || bucket.bbox == nil {
			continue
		}
		dets, err := decodeStride(bucket.score, bucket.bbox, sd, inputW, inputH, confThresh)
		if err != nil {
			return models.Result{}, fmt.Errorf("scrfd: decode stride %d: %w", sd.stride, err)
		}
		decoded++
		candidates = append(candidates, dets...)
	}
	if decoded == 0 {
		shapes := make([][]int64, len(outs))
		for i := range outs {
			shapes[i] = outs[i].Shape
		}
		return models.Result{}, fmt.Errorf("scrfd: no output matched a %dx%d input's stride levels (output shapes %v)", inputW, inputH, shapes)
	}

	if len(candidates) == 0 {
		return models.Result{Detections: []models.Detection{}}, nil
	}

	// Step 3: NMS (all faces share the same class "face"), IoU 0.4 as InsightFace.
	kept := nms.Detections(candidates, nms.Options{IoU: 0.4})

	// Step 4: map from network-input space (letterboxed) back to ORIGINAL image coordinates
	// and clamp to the image.
	toOrig := meta.Affine()
	out := kept[:0]
	for i := range kept {
		b := geom.Clamp(toOrig.BoxToOrig(kept[i].BBox), meta.OrigWidth, meta.OrigHeight)
		if b[2] <= 0 || b[3] <= 0 {
			continue // entirely inside the letterbox padding
		}
		kept[i].BBox = b
		out = append(out, kept[i])
	}

	// Step 5: cut to max_detections (NMS already sorted by conf).
	if len(out) > maxDet {
		out = out[:maxDet]
	}

	return models.Result{Detections: out}, nil
}

// decodeStride decodes proposals for one pyramid level.
//
// Anchor layout (InsightFace scrfd.py): for a grid of H×W cells with A anchors per cell,
// proposal index k is:
//
//	row   i = k / (W * A)
//	col   j = (k % (W * A)) / A
//
// Both anchors at (i,j) share the same centre, at the cell's TOP-LEFT corner:
// cx = j*stride, cy = i*stride (np.mgrid[:H,:W][::-1] * stride — no +0.5).
//
// distance2bbox (distances l,t,r,b in stride units):
//
//	x1 = cx - l*stride   y1 = cy - t*stride
//	x2 = cx + r*stride   y2 = cy + b*stride
//	→ xywh in pixels of the network input image (not clamped: the caller clamps
//	  in ORIGINAL coordinates, like InsightFace which never clips here).
//
// Scores are already probabilities (in-graph Sigmoid) and are compared to confThresh
// directly (InsightFace: scores >= det_thresh).
func decodeStride(
	scoreT, bboxT *engine.Tensor,
	sd scrfdStride,
	inputW, inputH int,
	confThresh float64,
) ([]models.Detection, error) {
	n := sd.numProposals
	if int(scoreT.Dim(-2)) != n || int(bboxT.Dim(-2)) != n {
		return nil, fmt.Errorf("expected %d proposals, got score=%d bbox=%d",
			n, scoreT.Dim(-2), bboxT.Dim(-2))
	}
	if len(scoreT.Data) < n || len(bboxT.Data) < n*4 {
		return nil, fmt.Errorf("short output data: score=%d (want %d) bbox=%d (want %d)",
			len(scoreT.Data), n, len(bboxT.Data), n*4)
	}

	stride := sd.stride
	numAnchors := sd.numAnchors
	gridW := inputW / stride
	if gridW <= 0 {
		return nil, fmt.Errorf("input width %d smaller than stride %d", inputW, stride)
	}

	scores := scoreT.Data
	bboxes := bboxT.Data

	var dets []models.Detection
	for k := 0; k < n; k++ {
		prob := float64(scores[k])
		if prob < -1e-6 || prob > 1+1e-6 || math.IsNaN(prob) {
			return nil, fmt.Errorf("score %v at proposal %d is not a probability: this decoder expects the InsightFace export whose score outputs end in a Sigmoid", prob, k)
		}
		if prob < confThresh {
			continue
		}

		row := k / (gridW * numAnchors)
		col := (k % (gridW * numAnchors)) / numAnchors
		cx := float64(col * stride)
		cy := float64(row * stride)

		base := k * 4
		s := float64(stride)
		x1 := cx - float64(bboxes[base])*s
		y1 := cy - float64(bboxes[base+1])*s
		x2 := cx + float64(bboxes[base+2])*s
		y2 := cy + float64(bboxes[base+3])*s

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
