package rfdetr

import (
	"fmt"
	"math"
	"sort"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// postprocess decodes RF-DETR output (DETR-style, NMS-free) -> normalized Result.
//
// ===================== OUTPUT FORMAT =====================
// The model emits EXACTLY 2 tensors:
//   - logits: shape [1, Q, C]  (Q = number of queries, C = number of classes)  — BEFORE sigmoid
//   - boxes : shape [1, Q, 4]  — box format cxcywh, normalized [0,1] relative to the INPUT image
//
// We identify which tensor is boxes by SHAPE (last dim == 4); the other one is logits
// — more robust than relying on tensor names (which differ across RF-DETR exports).
//
// RF-DETR uses sigmoid (no "no-object" softmax class) -> each query takes the class with
// the highest probability after sigmoid, filtered by conf_threshold.
//
// VERIFIED on REAL WEIGHTS (rf-detr-base, COCO; see models/rf-detr/README.md):
//
//		inputs:  input      [1,3,560,560] f32
//		outputs: pred_logits[1,Q,91] f32  |  pred_boxes[1,Q,4] f32   (Q=300 queries)
//
//	 1. exactly 2 outputs, boxes detected by last dim == 4 (name-independent). OK.
//	 2. logits through SIGMOID (RF-DETR uses focal loss, NOT softmax+no-obj) — against
//	    real data gives a reasonable score (~0.95); softmax gives wrong results. OK.
//	 3. box is CXCYWH normalized [0,1] — decoding cxcywh lands on the right object; xyxy is off. OK.
//
// NOTE C=91 (the COCO "paper" space, index 0 = N/A), NOT a contiguous 80 — you must use a
// 91-line labels file (models/rf-detr/coco91.txt). The bundled dummy ONNX file also emulates
// [1,Q,91] to match this contract (see models/rf-detr/gen_dummy_onnx.py).
// If another export returns C=80, change the manifest labels to match — do not guess.
// =========================================================
func (m *rfDETR) postprocess(outs []engine.Tensor, meta models.PreprocessMeta) (models.Result, error) {
	if len(outs) != 2 {
		return models.Result{}, fmt.Errorf("rfdetr: expected exactly 2 outputs (logits + boxes), got %d — verify the ONNX export", len(outs))
	}

	var boxes, logits engine.Tensor
	if outs[0].Dim(-1) == 4 {
		boxes, logits = outs[0], outs[1]
	} else if outs[1].Dim(-1) == 4 {
		boxes, logits = outs[1], outs[0]
	} else {
		return models.Result{}, fmt.Errorf("rfdetr: could not identify the boxes tensor (no output has a last dimension == 4)")
	}

	q := int(logits.Dim(1))
	c := int(logits.Dim(2))
	if q == 0 || c == 0 || int(boxes.Dim(1)) != q {
		return models.Result{}, fmt.Errorf("rfdetr: mismatched shapes logits=%v boxes=%v", logits.Shape, boxes.Shape)
	}

	conf := m.cfg.ConfThresh
	boxFormat := m.cfg.BoxFormat
	if boxFormat == "" {
		boxFormat = "cxcywh"
	}

	dets := make([]models.Detection, 0, q)
	for i := 0; i < q; i++ {
		// class with the highest score after sigmoid
		bestCls, bestScore := -1, 0.0
		base := i * c
		for k := 0; k < c; k++ {
			s := sigmoid(float64(logits.Data[base+k]))
			if s > bestScore {
				bestScore = s
				bestCls = k
			}
		}
		if bestCls < 0 || bestScore < conf {
			continue
		}

		bo := i * 4
		b0 := float64(boxes.Data[bo])
		b1 := float64(boxes.Data[bo+1])
		b2 := float64(boxes.Data[bo+2])
		b3 := float64(boxes.Data[bo+3])

		// -> pixel coordinates on the INPUT image (letterboxed)
		x, y, w, h := toInputXYWH(b0, b1, b2, b3, boxFormat, m.cfg.Width, m.cfg.Height)

		// map back to the ORIGINAL image: orig = (input - pad) / scale
		ox := (x - float64(meta.PadX)) / meta.ScaleX
		oy := (y - float64(meta.PadY)) / meta.ScaleY
		ow := w / meta.ScaleX
		oh := h / meta.ScaleY

		// clamp to the original image bounds
		ox, oy, ow, oh = clampBox(ox, oy, ow, oh, meta.OrigWidth, meta.OrigHeight)

		dets = append(dets, models.Detection{
			BBox:  [4]float64{ox, oy, ow, oh},
			Class: m.classLabel(bestCls),
			Conf:  bestScore,
		})
	}

	// RF-DETR is NMS-free: only sort by conf + cut to max_detections, do NOT run NMS.
	sort.Slice(dets, func(a, b int) bool { return dets[a].Conf > dets[b].Conf })
	if m.cfg.MaxDet > 0 && len(dets) > m.cfg.MaxDet {
		dets = dets[:m.cfg.MaxDet]
	}

	return models.Result{Detections: dets}, nil
}

// toInputXYWH converts a box (normalized [0,1]) to [x,y,w,h] pixels on the WxH input image.
func toInputXYWH(a, b, cc, d float64, format string, w, h int) (x, y, bw, bh float64) {
	fw, fh := float64(w), float64(h)
	switch format {
	case "xyxy":
		x1, y1, x2, y2 := a*fw, b*fh, cc*fw, d*fh
		return x1, y1, x2 - x1, y2 - y1
	default: // cxcywh
		cx, cy, ww, hh := a*fw, b*fh, cc*fw, d*fh
		return cx - ww/2, cy - hh/2, ww, hh
	}
}

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

func (m *rfDETR) classLabel(idx int) string {
	if idx >= 0 && idx < len(m.cfg.Labels) {
		return m.cfg.Labels[idx]
	}
	return fmt.Sprintf("class_%d", idx)
}

func sigmoid(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }
