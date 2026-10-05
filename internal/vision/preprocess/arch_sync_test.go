package preprocess_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"visionserve/internal/models"
	"visionserve/internal/registry"
	"visionserve/internal/vision/preprocess"
)

// archCase is one row of testdata/arch_resolve_sync.json: a manifest's preprocessing as one
// architecture resolves it. clients/python/tests/test_go_python_sync.py runs the same rows through
// convert/spec.py (spec_from_manifest + resolve_arch), which `visionserve check` uses for its
// manifest-only reference: a Python side that read SCRFD's legacy `letterbox: true` as a centred
// letterbox in 0..1 units reported a false 22137-gray-level FAIL.
type archCase struct {
	Name          string         `json:"name"`
	Architecture  string         `json:"architecture"`
	Input         map[string]any `json:"input"`
	Preprocess    map[string]any `json:"preprocess"`
	Spec          *archSpec      `json:"spec"`
	Error         []string       `json:"error"`
	FixedByExport bool           `json:"fixed_by_export"`
}

type archSpec struct {
	Resize     string    `json:"resize"`
	Width      int       `json:"width"`
	Height     int       `json:"height"`
	MultipleOf int       `json:"multiple_of"`
	NoUpscale  bool      `json:"no_upscale"`
	CropPct    float32   `json:"crop_pct"`
	Resample   string    `json:"resample"`
	Mean       []float32 `json:"mean"`
	Std        []float32 `json:"std"`
	Rescale    *bool     `json:"rescale"`
	Layout     string    `json:"layout"`
	Pad        float32   `json:"pad"`
	Legacy     bool      `json:"legacy"`
}

func (w archSpec) equal(s preprocess.Spec) bool {
	same := func(a, b []float32) bool {
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
	rescale := w.Rescale == nil || *w.Rescale
	return string(s.Resize) == w.Resize && s.Width == w.Width && s.Height == w.Height &&
		s.MultipleOf == w.MultipleOf && s.NoUpscale == w.NoUpscale && s.CropPct == w.CropPct &&
		string(s.Resample) == w.Resample &&
		same(s.Mean, w.Mean) && same(s.Std, w.Std) && s.NoRescale != rescale && string(s.Layout) == w.Layout &&
		s.PadValue == w.Pad && s.Legacy == w.Legacy
}

// archConfig builds the models.Config lifecycle builds from a loaded manifest (load.go).
func archConfig(man *registry.Manifest, spec preprocess.Spec) models.Config {
	return models.Config{
		Name: man.Name, Preprocess: &spec,
		Width: man.Input.Width, Height: man.Input.Height, Layout: man.Input.Layout,
		Mean: man.Input.Normalize.Mean, Std: man.Input.Normalize.Std,
		Letterbox: man.Input.Letterbox, Crop: man.Input.Crop, KeepAspect: man.Input.KeepAspect,
		MultipleOf: man.Input.MultipleOf,
		Files:      map[string]string{"encoder": "e.onnx", "decoder": "d.onnx", "det": "det.onnx", "rec": "rec.onnx", "model": "m.onnx"},
	}
}

// TestArchResolveSyncCorpus: every row of the shared corpus resolves as it says through the path
// the server takes — registry.LoadManifest, Manifest.PreprocessSpec, models.New, and the
// model's ResolvedPreprocess (which TestResolvedPreprocessIsWhatPreprocessApplies ties to what its
// Preprocess feeds).
func TestArchResolveSyncCorpus(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "arch_resolve_sync.json"))
	if err != nil {
		t.Fatal(err)
	}
	var corpus struct {
		Cases []archCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &corpus); err != nil {
		t.Fatal(err)
	}
	if len(corpus.Cases) < 30 {
		t.Fatalf("corpus has %d cases", len(corpus.Cases))
	}
	seen := map[string]bool{}
	for _, c := range corpus.Cases {
		seen[c.Architecture] = true
		doc := map[string]any{"name": "m", "task": "detection", "license": "Apache-2.0", "model_file": "m.onnx",
			"architecture": c.Architecture}
		if c.Input != nil {
			doc["input"] = c.Input
		}
		if c.Preprocess != nil {
			doc["preprocess"] = c.Preprocess
		}
		y, err := yaml.Marshal(doc)
		if err != nil {
			t.Fatal(err)
		}
		p := filepath.Join(t.TempDir(), "manifest.yaml")
		if err := os.WriteFile(p, y, 0o644); err != nil {
			t.Fatal(err)
		}
		man, err := registry.LoadManifest(p)
		if err != nil {
			t.Errorf("%s: the manifest does not load: %v", c.Name, err)
			continue
		}
		spec, err := man.PreprocessSpec()
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		cfg := archConfig(man, spec)
		cfg.Dir = filepath.Dir(p)
		if err := os.WriteFile(filepath.Join(cfg.Dir, "ppocr_keys_v1.txt"), []byte("a\nb\n"), 0o644); err != nil {
			t.Fatal(err) // paddle-ocr reads its charset at New
		}
		base, err := models.New(c.Architecture, cfg)
		switch {
		case c.Error != nil:
			if err == nil {
				if rep, ok := base.(models.PreprocessReporter); ok {
					_, err = rep.ResolvedPreprocess()
				}
			}
			if err == nil {
				t.Errorf("%s: resolved, want an error containing %q", c.Name, c.Error)
				continue
			}
			for _, sub := range c.Error {
				if !strings.Contains(err.Error(), sub) {
					t.Errorf("%s: error %q does not name %q", c.Name, err, sub)
				}
			}
		case err != nil:
			t.Errorf("%s: models.New: %v", c.Name, err)
		case c.FixedByExport:
			if _, ok := base.(models.PreprocessReporter); ok {
				t.Errorf("%s: %s reports a manifest-driven preprocessing; the corpus says its export fixes it", c.Name, c.Architecture)
			}
		default:
			rep, ok := base.(models.PreprocessReporter)
			if !ok {
				t.Errorf("%s: %s does not report its preprocessing", c.Name, c.Architecture)
				continue
			}
			got, err := rep.ResolvedPreprocess()
			if err != nil {
				t.Errorf("%s: %v", c.Name, err)
				continue
			}
			if c.Spec == nil || !c.Spec.equal(got) {
				t.Errorf("%s:\n got  %+v\n want %+v", c.Name, got, c.Spec)
			}
		}
	}
	// Every architecture that resolves a manifest's preprocessing its own way is covered.
	for _, a := range []string{"rf-detr", "rt-detr", "efficientnet", "mobilenet-v3", "clip", "midas",
		"depth-anything-v2", "scrfd", "mobile-sam", "efficient-sam", "sam2", "nano-sam", "paddle-ocr"} {
		if !seen[a] {
			t.Errorf("the corpus has no case for %s", a)
		}
	}
}
