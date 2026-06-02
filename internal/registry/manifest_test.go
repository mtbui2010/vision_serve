package registry

import (
	"strings"
	"testing"
)

// License AGPL PHẢI bị từ chối (nguyên tắc tối thượng #1 — CLAUDE.md).
func TestValidateRejectsAGPL(t *testing.T) {
	m := &Manifest{Name: "yolo", License: "AGPL-3.0", Task: "detection", ModelFile: "x.onnx"}
	m.Input.Width, m.Input.Height = 640, 640
	err := m.validate()
	if err == nil || !strings.Contains(err.Error(), "license") {
		t.Fatalf("muốn lỗi license khi AGPL, nhận: %v", err)
	}
}

// Task không hợp lệ -> từ chối.
func TestValidateRejectsBadTask(t *testing.T) {
	m := &Manifest{Name: "x", License: "Apache-2.0", Task: "classification", ModelFile: "x.onnx"}
	m.Input.Width, m.Input.Height = 1, 1
	if err := m.validate(); err == nil || !strings.Contains(err.Error(), "task") {
		t.Fatalf("muốn lỗi task không hợp lệ, nhận: %v", err)
	}
}

// Manifest hợp lệ cấu trúc (license permissive) PHẢI qua validate dù chưa có weights —
// model vẫn được liệt kê, chỉ là chưa sẵn sàng load.
func TestValidatePassesWithoutWeights(t *testing.T) {
	m := &Manifest{Name: "ok", License: "Apache-2.0", Task: "detection", ModelFile: "nope.onnx"}
	m.Input.Width, m.Input.Height = 100, 100
	if err := m.validate(); err != nil {
		t.Fatalf("validate cấu trúc phải pass khi license/task/dims hợp lệ, nhận: %v", err)
	}
	if m.WeightsExist() {
		t.Fatal("WeightsExist phải false cho file không tồn tại")
	}
}
