// Package rfdetr implement model RF-DETR (detection, Apache 2.0) — MVP lõi tier free.
//
// RF-DETR là kiến trúc DETR-style, NMS-free: output là một tập "queries" cố định,
// mỗi query gồm logits theo class + box. KHÔNG áp logic NMS của YOLO vào đây (CLAUDE.md).
//
// Đăng ký với tên kiến trúc "rf-detr" qua init() — thêm model không phải sửa core.
package rfdetr

import (
	"fmt"
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

func init() {
	models.Register("rf-detr", New)
}

type rfDETR struct {
	cfg models.Config
}

// New là factory được lifecycle gọi sau khi parse manifest.
func New(cfg models.Config) (models.Base, error) {
	if cfg.Width <= 0 || cfg.Height <= 0 {
		return nil, fmt.Errorf("rfdetr: invalid input width/height (%dx%d)", cfg.Width, cfg.Height)
	}
	if len(cfg.Labels) == 0 {
		// Không fail cứng: vẫn detect được, chỉ là class hiển thị dạng id.
		// (manifest nên khai báo labels: coco.txt)
	}
	return &rfDETR{cfg: cfg}, nil
}

func (m *rfDETR) Name() string      { return m.cfg.Name }
func (m *rfDETR) Task() models.Task { return models.TaskDetection }

// InputName/OutputNames để rỗng -> engine tự dò tên I/O từ file ONNX.
// (Postprocess nhận diện đâu là boxes/logits theo SHAPE, không theo tên — bền vững
// hơn khi tên export khác nhau giữa các bản RF-DETR.)
func (m *rfDETR) InputName() string     { return "" }
func (m *rfDETR) OutputNames() []string { return nil }

func (m *rfDETR) Preprocess(img image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	return m.preprocess(img)
}

func (m *rfDETR) Postprocess(outs []engine.Tensor, meta models.PreprocessMeta) (models.Result, error) {
	return m.postprocess(outs, meta)
}
