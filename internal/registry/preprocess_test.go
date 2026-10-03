package registry

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"visionserve/internal/vision/preprocess"
)

// specRec is the canonical form of a resolved spec in testdata/preprocess_sync.json (the
// Python converter's sync test reads the same file).
type specRec struct {
	Resize     string    `json:"resize"`
	Width      int       `json:"width"`
	Height     int       `json:"height"`
	MultipleOf int       `json:"multiple_of"`
	NoUpscale  bool      `json:"no_upscale"`
	Resample   string    `json:"resample"`
	Mean       []float32 `json:"mean"`
	Std        []float32 `json:"std"`
	Rescale    *bool     `json:"rescale"`
	Layout     string    `json:"layout"`
	Pad        float32   `json:"pad"`
	Legacy     bool      `json:"legacy"`
}

type syncCase struct {
	Name       string         `json:"name"`
	Input      map[string]any `json:"input"`
	Preprocess map[string]any `json:"preprocess"`
	Spec       *specRec       `json:"spec"`
	Error      []string       `json:"error"`
}

// writeManifest writes a manifest with the given input / preprocess blocks (nil = omitted).
func writeManifest(t *testing.T, input, pre map[string]any) string {
	t.Helper()
	doc := map[string]any{"name": "m", "task": "detection", "license": "Apache-2.0", "model_file": "m.onnx"}
	if input != nil {
		doc["input"] = input
	}
	if pre != nil {
		doc["preprocess"] = pre
	}
	raw, err := yaml.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "manifest.yaml")
	if err := os.WriteFile(p, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func sameF32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func checkSpec(t *testing.T, name string, got preprocess.Spec, want *specRec) {
	t.Helper()
	rescale := want.Rescale == nil || *want.Rescale
	if string(got.Resize) != want.Resize || got.Width != want.Width || got.Height != want.Height ||
		got.MultipleOf != want.MultipleOf || got.NoUpscale != want.NoUpscale || string(got.Resample) != want.Resample ||
		!sameF32(got.Mean, want.Mean) || !sameF32(got.Std, want.Std) || got.NoRescale == rescale ||
		string(got.Layout) != want.Layout || got.PadValue != want.Pad || got.Legacy != want.Legacy {
		t.Errorf("%s:\n got  %+v\n want %+v (rescale %v)", name, got, *want, rescale)
	}
}

// TestPreprocessSyncCorpus: legacy-only, block-only, both-consistent and both-conflicting
// manifests resolve as testdata/preprocess_sync.json says, through LoadManifest (so an explicit
// `letterbox: false` is told from an absent one).
func TestPreprocessSyncCorpus(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "preprocess_sync.json"))
	if err != nil {
		t.Fatal(err)
	}
	var corpus struct {
		Cases []syncCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &corpus); err != nil {
		t.Fatal(err)
	}
	if len(corpus.Cases) < 30 {
		t.Fatalf("corpus has %d cases", len(corpus.Cases))
	}
	for _, c := range corpus.Cases {
		m, err := LoadManifest(writeManifest(t, c.Input, c.Preprocess))
		if c.Error != nil {
			if err == nil {
				t.Errorf("%s: loaded, want an error containing %q", c.Name, c.Error)
				continue
			}
			for _, sub := range c.Error {
				if !strings.Contains(err.Error(), sub) {
					t.Errorf("%s: error %q does not name %q", c.Name, err, sub)
				}
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		got, err := m.PreprocessSpec()
		if err != nil {
			t.Errorf("%s: PreprocessSpec after load: %v", c.Name, err)
			continue
		}
		checkSpec(t, c.Name, got, c.Spec)
	}
}

// A block-only manifest fills the legacy input.* view, so code reading those fields (lifecycle's
// models.Config, `visionserve list`, the keep_aspect graph check) sees the declared values.
func TestPreprocessBlockFillsLegacyView(t *testing.T) {
	m, err := LoadManifest(writeManifest(t, nil, map[string]any{
		"resize": "letterbox", "size": 640, "layout": "NCHW",
		"mean": []float64{0.485, 0.456, 0.406}, "std": []float64{0.229, 0.224, 0.225},
	}))
	if err != nil {
		t.Fatal(err)
	}
	in := m.Input
	if in.Width != 640 || in.Height != 640 || !in.Letterbox || in.Crop != "" || in.KeepAspect || in.Layout != "NCHW" ||
		len(in.Normalize.Mean) != 3 || len(in.Normalize.Std) != 3 {
		t.Fatalf("legacy view %+v", in)
	}

	m, err = LoadManifest(writeManifest(t, map[string]any{"width": 518, "height": 518},
		map[string]any{"resize": "keep_aspect", "multiple_of": 14}))
	if err != nil {
		t.Fatal(err)
	}
	if !m.Input.KeepAspect || m.Input.MultipleOf != 14 || m.Input.Letterbox {
		t.Fatalf("keep_aspect legacy view %+v", m.Input)
	}

	// 0..255-unit mean/std (rescale: false) and modes without a legacy flag stay out of input.*.
	m, err = LoadManifest(writeManifest(t, nil, map[string]any{
		"resize": "top_left_pad", "size": 640, "rescale": false,
		"mean": []float64{127.5, 127.5, 127.5}, "std": []float64{128, 128, 128},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if m.Input.Letterbox || len(m.Input.Normalize.Mean) != 0 || m.Input.Width != 640 {
		t.Fatalf("top_left_pad legacy view %+v", m.Input)
	}
	// Re-validating the filled manifest is idempotent.
	if err := m.validate(); err != nil {
		t.Fatalf("re-validate: %v", err)
	}
}

// A Manifest built in code (no YAML key information) counts its non-zero legacy fields as
// declared: a letterbox flag still conflicts with a squash block.
func TestPreprocessSpecManifestInCode(t *testing.T) {
	m := &Manifest{Name: "m", License: "MIT", Task: "detection", ModelFile: "m.onnx"}
	m.Input.Width, m.Input.Height, m.Input.Letterbox = 640, 640, true
	m.Preprocess = &PreprocessBlock{Resize: "squash"}
	if err := m.validate(); err == nil || !strings.Contains(err.Error(), "input.letterbox: true") {
		t.Fatalf("err = %v, want a letterbox conflict", err)
	}
	m.Preprocess = &PreprocessBlock{Resize: "letterbox"}
	if err := m.validate(); err != nil {
		t.Fatal(err)
	}
	m.Preprocess = nil
	s, err := m.PreprocessSpec()
	if err != nil || s.Resize != preprocess.Letterbox || !s.Legacy {
		t.Fatalf("legacy spec %+v, %v", s, err)
	}
}

// Every shipped manifest that loads resolves its preprocessing; without a block, to a legacy
// spec. (A manifest the registry refuses for other reasons is not this test's business.)
func TestShippedManifestsResolveLegacy(t *testing.T) {
	paths, err := filepath.Glob(filepath.Join("..", "..", "models", "*", "manifest.yaml"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("no shipped manifests (%v)", err)
	}
	loaded := 0
	for _, p := range paths {
		m, err := LoadManifest(p)
		if err != nil {
			if strings.Contains(err.Error(), "preprocess") {
				t.Errorf("%s: %v", p, err)
			}
			continue
		}
		loaded++
		s, err := m.PreprocessSpec()
		if err != nil {
			t.Errorf("%s: %v", p, err)
			continue
		}
		if m.Preprocess == nil && !s.Legacy {
			t.Errorf("%s: no block but spec %+v is not legacy", p, s)
		}
	}
	if loaded < len(paths)/2 {
		t.Fatalf("only %d of %d shipped manifests load", loaded, len(paths))
	}
}
