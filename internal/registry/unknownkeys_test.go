package registry

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

const unknownKeysBase = `name: m
task: detection
license: MIT
model_file: m.onnx
input: {width: 64, height: 64}
`

// A typo of a real key is reported with its path and line, and the manifest still loads (with
// the misspelt setting at its default — which is exactly why it must be reported).
func TestUnknownKeysReportsATypo(t *testing.T) {
	raw := unknownKeysBase + `runtime:
  prefer: [cpu]
  idle_unload_second: 30
`
	m := writeAndLoad(t, raw)
	want := []string{"runtime.idle_unload_second (line 8)"}
	if got := m.UnknownKeys(); !reflect.DeepEqual(got, want) {
		t.Fatalf("UnknownKeys = %q, want %q", got, want)
	}
	if m.Runtime.IdleUnloadSeconds != 0 {
		t.Fatalf("the typo must not set idle_unload_seconds, got %d", m.Runtime.IdleUnloadSeconds)
	}
}

// Every key a Manifest field reads is known, at every depth — including map keys (files roles,
// explain outputs, runtime.threads roles, sha256 roles, sha256_files paths), which are data, not
// schema. Unknown keys are found at the top level and inside nested structs.
func TestUnknownKeysWalksTheSchema(t *testing.T) {
	raw := `name: m
task: open_vocab
license: Apache-2.0
architecture: grounded-sam
source_url: https://example.com/m
files: {det: d.onnx, encoder: e.onnx, decoder: x.onnx}
sha256: {det: aa, encoder: bb, decoder: cc}
sha256_files: {vocab.txt: dd}
input:
  width: 800
  height: 800
  layout: NCHW
  letterbox: false
  crop: ""
  keep_aspect: false
  multiple_of: 0
  normalize: {mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5]}
  colour: rgb
preprocess: {resize: squash, size: 800, mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5], rescale: true, pad: 0, padding: 3}
postprocess: {type: gdino, box_format: cxcywh, conf_threshold: 0.3, text_threshold: 0.25, max_detections: 100}
labels: labels.txt
detector: grounding-dino
segmenter: mobile-sam
grasp: {gripper_min: 1, gripper_max: 2}
explain: {type: attention, role: det, outputs: {attention: attn, anything: x}, spatial_stride: 32, top_channels: 8}
instance: {max_templates: 1, sim_threshold: 0.5, patch_size: 16}
runtime: {prefer: [cpu], idle_unload_seconds: 5, threads: {encoder: 1, decoder: 2}}
lisence: MIT
`
	got, err := unknownManifestKeys([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"input.colour (line 18)", "preprocess.padding (line 19)", "lisence (line 28)"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unknownManifestKeys = %q, want %q", got, want)
	}
}

// YAML anchors / merge keys are followed, not reported as the literal key "<<".
func TestUnknownKeysFollowsMergeKeys(t *testing.T) {
	raw := unknownKeysBase + `defaults: &d {prefer: [cpu], idle_unload_secs: 1}
runtime:
  <<: *d
  idle_unload_seconds: 5
`
	got, err := unknownManifestKeys([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"defaults (line 6)", "runtime.idle_unload_secs (line 6)"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unknownManifestKeys = %q, want %q", got, want)
	}
}

// Scan loads a manifest with unknown keys and returns a warning naming them; a clean manifest
// gets none.
func TestScanWarnsAboutUnknownKeys(t *testing.T) {
	root := t.TempDir()
	write := func(dir, body string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Join(root, dir), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, dir, "manifest.yaml"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("typo", strings.Replace(unknownKeysBase, "name: m", "name: typo", 1)+"runtime: {idle_unload_second: 30}\n")
	write("clean", strings.Replace(unknownKeysBase, "name: m", "name: clean", 1))

	reg := New(root)
	warns, err := reg.Scan()
	if err != nil {
		t.Fatal(err)
	}
	if len(warns) != 1 || !strings.Contains(warns[0].Error(), "runtime.idle_unload_second") ||
		!strings.Contains(warns[0].Error(), filepath.Join(root, "typo", "manifest.yaml")) {
		t.Fatalf("want one warning naming typo/manifest.yaml and runtime.idle_unload_second, got %v", warns)
	}
	for _, name := range []string{"typo", "clean"} {
		if _, ok := reg.Get(name); !ok {
			t.Errorf("%s must still be registered", name)
		}
	}
}

// No shipped manifest carries a key the parser ignores (a typo there would be silent).
func TestShippedManifestsHaveNoUnknownKeys(t *testing.T) {
	paths, err := filepath.Glob(filepath.Join("..", "..", "models", "*", "manifest.yaml"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("no shipped manifests (%v)", err)
	}
	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		unknown, err := unknownManifestKeys(raw)
		if err != nil {
			t.Errorf("%s: %v", p, err)
			continue
		}
		if len(unknown) > 0 {
			t.Errorf("%s: unknown keys %q", p, unknown)
		}
	}
}

func writeAndLoad(t *testing.T, body string) *Manifest {
	t.Helper()
	p := filepath.Join(t.TempDir(), "manifest.yaml")
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := LoadManifest(p)
	if err != nil {
		t.Fatal(err)
	}
	return m
}
