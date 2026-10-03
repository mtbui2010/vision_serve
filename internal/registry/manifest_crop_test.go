package registry

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestManifestInputCropValidation(t *testing.T) {
	base := func() *Manifest {
		m := &Manifest{Name: "m", License: "MIT", Task: "embed", ModelFile: "model.onnx"}
		m.Input.Width, m.Input.Height = 224, 224
		return m
	}
	m := base()
	m.Input.Crop = "center"
	if err := m.validate(); err != nil {
		t.Fatalf("crop: center refused: %v", err)
	}
	m = base()
	m.Input.Crop = "top"
	if err := m.validate(); err == nil || !strings.Contains(err.Error(), "input.crop") {
		t.Fatalf("unknown crop mode accepted: %v", err)
	}
	m = base()
	m.Input.Crop, m.Input.Letterbox = "center", true
	if err := m.validate(); err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("crop + letterbox accepted: %v", err)
	}
}

func TestManifestNameMustBeOnePathSegment(t *testing.T) {
	for _, bad := range []string{"../x", "/abs", "a/b", ".hidden", "-x", "a b", ""} {
		m := &Manifest{Name: bad, License: "MIT", Task: "embed", ModelFile: "m.onnx"}
		m.Input.Width, m.Input.Height = 8, 8
		if err := m.validate(); err == nil {
			t.Errorf("name %q accepted", bad)
		}
	}
	for _, good := range []string{"rf-detr", "rfdetr-gdino-siglip-etri", "clip_v2.1", "M2"} {
		m := &Manifest{Name: good, License: "MIT", Task: "embed", ModelFile: "m.onnx"}
		m.Input.Width, m.Input.Height = 8, 8
		if err := m.validate(); err != nil {
			t.Errorf("name %q refused: %v", good, err)
		}
	}
}

// A model set aside during a --force swap keeps its manifest (same name); it must not be listed
// next to — or instead of — the real install.
func TestScanSkipsDotDirectories(t *testing.T) {
	root := t.TempDir()
	write := func(dir string) {
		_ = os.MkdirAll(filepath.Join(root, dir), 0o755)
		man := "name: m\ntask: embed\nlicense: MIT\nmodel_file: m.onnx\ninput:\n  width: 8\n  height: 8\n"
		_ = os.WriteFile(filepath.Join(root, dir, "manifest.yaml"), []byte(man), 0o644)
	}
	write("m")
	write(".tmp-m-123-old")
	r := New(root)
	if _, err := r.Scan(); err != nil {
		t.Fatal(err)
	}
	e, ok := r.Get("m")
	if !ok || filepath.Base(e.Manifest.Dir()) != "m" {
		t.Fatalf("m resolved to %+v, want the real install, not the dot-directory", e)
	}
}
