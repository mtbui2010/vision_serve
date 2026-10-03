package registry

import (
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
