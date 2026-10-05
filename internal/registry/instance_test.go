package registry

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The instance block (template mode) is parsed and range-checked: sim_threshold is the template
// requests' default score threshold, in [0, 1) (0 = the model's default), and it does not trip
// the unknown-key warning.
func TestInstanceBlockParsedAndValidated(t *testing.T) {
	load := func(t *testing.T, instance string) (*Manifest, error) {
		t.Helper()
		y := "name: owl\ntask: instance_detection\nlicense: Apache-2.0\narchitecture: owlvit\n" +
			"files:\n  model: m.onnx\ninput:\n  width: 960\n  height: 960\n" +
			"postprocess:\n  max_detections: 10\ninstance:\n" + instance
		path := filepath.Join(t.TempDir(), "manifest.yaml")
		if err := os.WriteFile(path, []byte(y), 0o644); err != nil {
			t.Fatal(err)
		}
		return LoadManifest(path)
	}
	m, err := load(t, "  sim_threshold: 0.9\n  max_templates: 5\n  patch_size: 16\n")
	if err != nil {
		t.Fatal(err)
	}
	if m.Instance == nil || m.Instance.SimThreshold != 0.9 || m.Instance.MaxTemplates != 5 || m.Instance.PatchSize != 16 {
		t.Fatalf("instance = %+v, want sim_threshold 0.9, max_templates 5, patch_size 16", m.Instance)
	}
	if u := m.UnknownKeys(); len(u) != 0 {
		t.Fatalf("unknown keys %v, want none", u)
	}
	if _, err := load(t, "  patch_size: 16\n"); err != nil {
		t.Fatalf("an absent sim_threshold (the model's default) must load: %v", err)
	}
	for _, bad := range []string{
		"  sim_threshold: 1\n", "  sim_threshold: 1.5\n", "  sim_threshold: -0.1\n", "  sim_threshold: .nan\n",
		"  max_templates: -1\n", "  patch_size: -16\n",
	} {
		if _, err := load(t, bad); err == nil || !strings.Contains(err.Error(), "instance.") {
			t.Errorf("instance %q: err = %v, want an instance.* range error", strings.TrimSpace(bad), err)
		}
	}
}
