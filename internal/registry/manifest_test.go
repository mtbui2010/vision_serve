package registry

import (
	"strings"
	"testing"
)

// An AGPL license MUST be rejected (top principle #1 — CLAUDE.md).
func TestValidateRejectsAGPL(t *testing.T) {
	m := &Manifest{Name: "yolo", License: "AGPL-3.0", Task: "detection", ModelFile: "x.onnx"}
	m.Input.Width, m.Input.Height = 640, 640
	err := m.validate()
	if err == nil || !strings.Contains(err.Error(), "license") {
		t.Fatalf("expected a license error for AGPL, got: %v", err)
	}
}

// An invalid task -> rejected.
func TestValidateRejectsBadTask(t *testing.T) {
	m := &Manifest{Name: "x", License: "Apache-2.0", Task: "not_a_real_task", ModelFile: "x.onnx"}
	m.Input.Width, m.Input.Height = 1, 1
	if err := m.validate(); err == nil || !strings.Contains(err.Error(), "task") {
		t.Fatalf("expected an invalid-task error, got: %v", err)
	}
}

// A structurally valid manifest (permissive license) MUST pass validate even without weights —
// the model is still listed, just not ready to load.
func TestValidatePassesWithoutWeights(t *testing.T) {
	m := &Manifest{Name: "ok", License: "Apache-2.0", Task: "detection", ModelFile: "nope.onnx"}
	m.Input.Width, m.Input.Height = 100, 100
	if err := m.validate(); err != nil {
		t.Fatalf("structural validate must pass when license/task/dims are valid, got: %v", err)
	}
	if m.WeightsExist() {
		t.Fatal("WeightsExist must be false for a nonexistent file")
	}
}
