package rfdetr

import (
	"fmt"
	"math"
	"sort"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// postprocess decode output RF-DETR (DETR-style, NMS-free) -> Result chuẩn hóa.
//
// ===================== FORMAT OUTPUT =====================
// Model xuất ĐÚNG 2 tensor:
//   - logits: shape [1, Q, C]  (Q = số queries, C = số class)  — TRƯỚC sigmoid
//   - boxes : shape [1, Q, 4]  — box format cxcywh, chuẩn hóa [0,1] theo ảnh ĐẦU VÀO
//
// Ta nhận diện tensor nào là boxes bằng SHAPE (chiều cuối == 4), phần còn lại là logits
// — bền vững hơn so với dựa vào tên tensor (khác nhau giữa các bản export RF-DETR).
//
// RF-DETR dùng sigmoid (không có lớp "no-object" softmax) -> mỗi query lấy class có
// xác suất cao nhất sau sigmoid, lọc theo conf_threshold.
//
// ĐÃ XÁC MINH trên WEIGHTS THẬT (rf-detr-base, COCO; xem models/rf-detr/README.md):
//
//		inputs:  input      [1,3,560,560] f32
//		outputs: pred_logits[1,Q,91] f32  |  pred_boxes[1,Q,4] f32   (Q=300 queries)
//
//	 1. đúng 2 output, phát hiện boxes theo chiều cuối == 4 (không phụ thuộc tên). OK.
//	 2. logits qua SIGMOID (RF-DETR dùng focal loss, KHÔNG softmax+no-obj) — đối chiếu
//	    thật cho score hợp lý (~0.95); softmax cho kết quả sai. OK.
//	 3. box là CXCYWH chuẩn hóa [0,1] — giải mã cxcywh land đúng vật thể; xyxy thì lệch. OK.
//
// LƯU Ý C=91 (không gian COCO "paper", index 0 = N/A), KHÔNG phải 80 liền mạch — phải
// dùng file labels 91 dòng (models/rf-detr/coco91.txt). File ONNX dummy bundled cũng
// mô phỏng đúng [1,Q,91] để khớp hợp đồng này (xem models/rf-detr/gen_dummy_onnx.py).
// Nếu một bản export khác trả C=80, đổi manifest labels cho khớp — đừng đoán.
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
		// class có score lớn nhất sau sigmoid
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

		// -> toạ độ pixel trên ảnh ĐẦU VÀO (đã letterbox)
		x, y, w, h := toInputXYWH(b0, b1, b2, b3, boxFormat, m.cfg.Width, m.cfg.Height)

		// map ngược về ảnh GỐC: orig = (input - pad) / scale
		ox := (x - float64(meta.PadX)) / meta.ScaleX
		oy := (y - float64(meta.PadY)) / meta.ScaleY
		ow := w / meta.ScaleX
		oh := h / meta.ScaleY

		// clamp vào biên ảnh gốc
		ox, oy, ow, oh = clampBox(ox, oy, ow, oh, meta.OrigWidth, meta.OrigHeight)

		dets = append(dets, models.Detection{
			BBox:  [4]float64{ox, oy, ow, oh},
			Class: m.classLabel(bestCls),
			Conf:  bestScore,
		})
	}

	// RF-DETR là NMS-free: chỉ sắp theo conf + cắt max_detections, KHÔNG chạy NMS.
	sort.Slice(dets, func(a, b int) bool { return dets[a].Conf > dets[b].Conf })
	if m.cfg.MaxDet > 0 && len(dets) > m.cfg.MaxDet {
		dets = dets[:m.cfg.MaxDet]
	}

	return models.Result{Detections: dets}, nil
}

// toInputXYWH chuyển box (chuẩn hóa [0,1]) về [x,y,w,h] pixel trên ảnh đầu vào WxH.
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
