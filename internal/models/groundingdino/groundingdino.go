// Package groundingdino — EXTENSION POINT cho tier PAID (open-vocab / auto-label).
//
// // PAID TIER — KHÔNG triển khai ở đây.
//
// Grounding DINO (bản Apache 2.0) phục vụ auto-label dựa trên text prompt. Tính năng
// này thuộc SẢN PHẨM ĐÓNG KÍN (labeling/auto-label/fine-tune), KHÔNG nằm trong repo
// mã nguồn mở này (xem CLAUDE.md mục 5). File này chỉ giữ điểm cắm + stub để tier paid
// đăng ký implementation thật sau, mà không phải sửa core.
package groundingdino

import (
	"fmt"
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

func init() {
	// Đăng ký stub để registry/list nhận biết loại model, nhưng mọi thao tác đều
	// trả lỗi "chưa triển khai (paid tier)".
	models.Register("grounding-dino", New)
}

type stub struct {
	cfg models.Config
}

func New(cfg models.Config) (models.Base, error) {
	return &stub{cfg: cfg}, nil
}

func (s *stub) Name() string          { return s.cfg.Name }
func (s *stub) Task() models.Task     { return models.TaskOpenVocab }
func (s *stub) InputName() string     { return "" }
func (s *stub) OutputNames() []string { return nil }

// PAID TIER — không triển khai ở đây.
func (s *stub) Preprocess(img image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	return engine.Tensor{}, models.PreprocessMeta{}, errPaid()
}

// PAID TIER — không triển khai ở đây.
func (s *stub) Postprocess(outs []engine.Tensor, meta models.PreprocessMeta) (models.Result, error) {
	return models.Result{}, errPaid()
}

func errPaid() error {
	return fmt.Errorf("grounding-dino thuộc TIER PAID (auto-label) — không triển khai trong repo mã nguồn mở này")
}
