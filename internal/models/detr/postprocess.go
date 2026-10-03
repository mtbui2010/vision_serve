package detr

import (
	"fmt"
	"sort"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/vision/geom"
)

// postprocess decodes DETR output (NMS-free) -> normalized Result.
//
// ===================== OUTPUT FORMAT =====================
// Two tensors matter:
//   - logits: shape [1, Q, C]  (Q = number of queries, C = number of classes) — BEFORE sigmoid
//   - boxes : shape [1, Q, 4]  — cxcywh (or xyxy per manifest box_format), normalized [0,1]
//     relative to the INPUT image
//
// Which tensor is which is decided by SHAPE (last dim == 4 → boxes), not by name — export
// names differ across releases. Both models use sigmoid (focal loss, no "no-object"
// softmax class): each query takes the class with the highest probability after sigmoid,
// filtered by conf_threshold.
//
// VERIFIED on REAL WEIGHTS:
//
//	rf-detr-base (COCO; models/rf-detr/README.md):
//	  inputs:  input      [1,3,560,560] f32
//	  outputs: pred_logits[1,Q,91] f32  |  pred_boxes[1,Q,4] f32   (Q=300 queries)
//	  1. boxes detected by last dim == 4 (name-independent). OK.
//	  2. logits through SIGMOID — against real data gives a reasonable score (~0.95);
//	     softmax gives wrong results. OK.
//	  3. box is CXCYWH normalized [0,1] — decoding cxcywh lands on the right object; xyxy
//	     is off. OK.
//	  C=91 is the COCO "paper" space (index 0 = N/A), NOT a contiguous 80 — use a 91-line
//	  labels file (models/rf-detr/coco91.txt). If another export returns C=80, change the
//	  manifest labels to match — do not guess.
//	rfdetr-small exports a third output, cross_attn_weights [3,1,16,300,1024] (explain);
//	RF-DETR ignores extra outputs.
//
//	RT-DETR (onnx-community/RT-DETR-l-hf):
//	  inputs:  pixel_values [1,3,640,640] f32
//	  outputs: pred_logits  [1,Q,80] f32  |  pred_boxes [1,Q,4] f32  (Q=300 queries)
//	  COCO-80 (indices 0-79, no N/A gap) — use coco80.txt. RT-DETR requires EXACTLY two
//	  outputs.
//
// =========================================================
func (m *detr) postprocess(outs []engine.Tensor, meta models.PreprocessMeta) (models.Result, error) {
	boxes, logits, err := m.v.split(outs)
	if err != nil {
		return models.Result{}, err
	}

	q := int(logits.Dim(1))
	c := int(logits.Dim(2))
	if q == 0 || c == 0 || int(boxes.Dim(1)) != q {
		return models.Result{}, fmt.Errorf("%s: mismatched shapes logits=%v boxes=%v", m.v.prefix, logits.Shape, boxes.Shape)
	}

	conf := m.cfg.ConfThresh
	boxFormat := m.cfg.BoxFormat
	if boxFormat == "" {
		boxFormat = geom.FormatCXCYWH
	}
	toOrig := meta.Affine()

	dets := make([]models.Detection, 0, q)
	for i := 0; i < q; i++ {
		// class with the highest score after sigmoid
		bestCls, bestScore := -1, 0.0
		base := i * c
		for k := 0; k < c; k++ {
			s := geom.Sigmoid(float64(logits.Data[base+k]))
			if s > bestScore {
				bestScore = s
				bestCls = k
			}
		}
		if bestCls < 0 || bestScore < conf {
			continue
		}

		bo := i * 4
		// normalized box -> pixels on the INPUT image -> ORIGINAL image (orig = (input -
		// pad) / scale), clamped to the original image bounds.
		in := geom.NormToInput(float64(boxes.Data[bo]), float64(boxes.Data[bo+1]),
			float64(boxes.Data[bo+2]), float64(boxes.Data[bo+3]), boxFormat, m.cfg.Width, m.cfg.Height)
		bbox := geom.Clamp(toOrig.BoxToOrig(in), meta.OrigWidth, meta.OrigHeight)

		dets = append(dets, models.Detection{
			BBox:  bbox,
			Class: m.classLabel(bestCls),
			Conf:  bestScore,
		})
	}

	// NMS-free: only sort by conf + cut to max_detections, do NOT run NMS.
	sort.Slice(dets, func(a, b int) bool { return dets[a].Conf > dets[b].Conf })
	if m.cfg.MaxDet > 0 && len(dets) > m.cfg.MaxDet {
		dets = dets[:m.cfg.MaxDet]
	}

	return models.Result{Detections: dets}, nil
}

// splitRF (RF-DETR): at least two outputs; the first output whose last dim is 4 is boxes,
// the first other one is logits, and extra outputs (e.g. cross_attn_weights for explain)
// are ignored. That is SplitOutputs with no label count and no feature width, so it is
// SplitOutputs: the decoder, the router and the text-aligned head share one rule
// (TestSplitRFMatchesReference pins it to the rule this function used to spell out itself).
func splitRF(outs []engine.Tensor) (boxes, logits engine.Tensor, err error) {
	if len(outs) < 2 {
		return boxes, logits, fmt.Errorf("rfdetr: expected at least 2 outputs (logits + boxes), got %d — verify the ONNX export", len(outs))
	}
	o, err := SplitOutputs(outs, 0, 0)
	if err != nil {
		// SplitOutputs fails only when no output has a last dim of 4. RF-DETR keeps its own
		// wording, which /api/predict has always returned (pinned by internal/models/golden).
		return boxes, logits, fmt.Errorf("rfdetr: could not identify the boxes tensor (no output has a last dimension == 4)")
	}
	// With two or more outputs and nLabels = dFeat = 0 SplitOutputs always finds logits; an
	// output without data (never produced by the engine) is still refused, not decoded.
	if o.Boxes.Data == nil || o.Logits.Data == nil {
		return boxes, logits, fmt.Errorf("rfdetr: boxes or logits output carries no data (shapes %v)", shapes(outs))
	}
	return o.Boxes, o.Logits, nil
}

// splitRT (RT-DETR): exactly two outputs, one of which has a last dim of 4.
func splitRT(outs []engine.Tensor) (boxes, logits engine.Tensor, err error) {
	if len(outs) != 2 {
		return boxes, logits, fmt.Errorf("rtdetr: expected exactly 2 outputs (logits + boxes), got %d — verify the ONNX export", len(outs))
	}
	switch {
	case outs[0].Dim(-1) == 4:
		return outs[0], outs[1], nil
	case outs[1].Dim(-1) == 4:
		return outs[1], outs[0], nil
	}
	return boxes, logits, fmt.Errorf("rtdetr: could not identify the boxes tensor (no output has a last dimension == 4)")
}

// classLabel returns the label for a class index, or class_<id> when the manifest has none.
func (m *detr) classLabel(idx int) string {
	if idx >= 0 && idx < len(m.cfg.Labels) {
		return m.cfg.Labels[idx]
	}
	return fmt.Sprintf("class_%d", idx)
}
