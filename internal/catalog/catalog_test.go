package catalog

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A composed entry points INTO its dependencies' directories ("../grounding-dino/<file>"). Those
// filenames are written by a different entry, so they can drift apart — and did: grounding-dino
// switched to model-fixedmask.onnx while grounded-sam, rfdetr-gdino(-sam) and grasp-gd still
// referenced model.onnx, so every one of them pulled "successfully" and failed at load.
func TestComposedEntriesReferenceFilesTheirDependenciesWrite(t *testing.T) {
	for _, e := range builtin {
		refs := map[string]string{}
		for role, rel := range e.VirtualFiles {
			refs["files."+role] = rel
		}
		if strings.HasPrefix(e.LabelsFile, "../") {
			refs["labels"] = e.LabelsFile
		}
		for what, rel := range refs {
			parts := strings.SplitN(strings.TrimPrefix(rel, "../"), "/", 2)
			if len(parts) != 2 || !strings.HasPrefix(rel, "../") {
				t.Errorf("%s: %s = %q is not of the form ../<dependency>/<file>", e.Name, what, rel)
				continue
			}
			dep, file := parts[0], parts[1]
			if !contains(e.Dependencies, dep) {
				t.Errorf("%s: %s points into %q, which is not in Dependencies %v", e.Name, what, dep, e.Dependencies)
			}
			de, ok := Lookup(dep)
			if !ok {
				t.Errorf("%s: dependency %q is not in the catalog", e.Name, dep)
				continue
			}
			if !writes(de, file) {
				t.Errorf("%s: %s = %q, but `pull %s` never writes %q", e.Name, what, rel, dep, file)
			}
		}
	}
}

func TestLookupResolvesAliasesToTheCanonicalEntry(t *testing.T) {
	for _, alias := range []string{"groundingdino", "gdino"} {
		e, ok := Lookup(alias)
		if !ok || e.Name != "grounding-dino" {
			t.Fatalf("Lookup(%q) = %q, %v; want grounding-dino", alias, e.Name, ok)
		}
		// The alias must serve the CORRECTED joint-text-pass export, never onnx-community's.
		if e.HFRepo != "mtbui2010/grounding-dino-tiny-fixedmask-ONNX" {
			t.Fatalf("%s pulls from %s, want the fixed-mask re-export", alias, e.HFRepo)
		}
	}
}

func TestAliasesAreUnique(t *testing.T) {
	seen := map[string]string{}
	for _, e := range builtin {
		for _, n := range append([]string{e.Name}, e.Aliases...) {
			if prev, dup := seen[n]; dup {
				t.Errorf("name %q used by both %s and %s", n, prev, e.Name)
			}
			seen[n] = e.Name
		}
	}
}

// Dependencies already installed are not re-downloaded, and a composed model whose dependency
// install lacks a referenced file is refused at pull time with the dependency to re-pull, not
// written out to fail at load. Offline: every dependency is pre-installed as a stub.
func TestPullComposedChecksDependencyFiles(t *testing.T) {
	dir := t.TempDir()
	e, _ := Lookup("rfdetr-gdino-siglip-etri")
	for _, dep := range e.Dependencies {
		mustWrite(t, filepath.Join(dir, dep, "manifest.yaml"), "stub")
	}
	// An old-style grounding-dino install: model.onnx, not model-fixedmask.onnx.
	for _, rel := range []string{"rfdetr-small-etri/model.onnx", "rfdetr-small-etri/labels.txt",
		"grounding-dino/model.onnx", "siglip-image/model.onnx", "siglip-text/model.onnx"} {
		mustWrite(t, filepath.Join(dir, rel), "x")
	}

	var log bytes.Buffer
	err := Pull("rfdetr-gdino-siglip-etri", PullOptions{ModelsDir: dir, Out: &log})
	if err == nil || !strings.Contains(err.Error(), "visionserve pull grounding-dino --force") {
		t.Fatalf("want a re-pull hint for grounding-dino, got %v", err)
	}
	if strings.Contains(log.String(), "pulling it first") {
		t.Fatalf("installed dependencies were pulled again:\n%s", log.String())
	}

	mustWrite(t, filepath.Join(dir, "grounding-dino/model-fixedmask.onnx"), "x")
	if err := Pull("rfdetr-gdino-siglip-etri", PullOptions{ModelsDir: dir, Out: &log}); err != nil {
		t.Fatalf("pull with every dependency file present: %v", err)
	}
	man, err := os.ReadFile(filepath.Join(dir, "rfdetr-gdino-siglip-etri", "manifest.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"gdino: ../grounding-dino/model-fixedmask.onnx", "letterbox: false",
		"crop: ../siglip-image/model.onnx", "labels: ../rfdetr-small-etri/labels.txt"} {
		if !strings.Contains(string(man), want) {
			t.Errorf("generated manifest lacks %q:\n%s", want, man)
		}
	}
}

func TestSizeFloorOnlyAppliesToWeights(t *testing.T) {
	if got := sizeFloor(File{LocalFilename: "labels.txt", SHA256: "ab"}); got != 1 {
		t.Errorf("pinned labels file floor = %d, want 1", got)
	}
	for _, f := range []File{{LocalFilename: "model.onnx", SHA256: "ab"}, {LocalFilename: "model.onnx.data", SHA256: "ab"}, {LocalFilename: "keys.txt"}} {
		if got := sizeFloor(f); got != minSaneSize {
			t.Errorf("%s floor = %d, want %d", f.LocalFilename, got, minSaneSize)
		}
	}
}

func writes(e Entry, file string) bool {
	if file == e.LabelsFile && e.EmbeddedLabels != "" {
		return true
	}
	for _, f := range e.Files {
		if f.LocalFilename == file {
			return true
		}
	}
	return false
}

func contains(xs []string, x string) bool {
	for _, v := range xs {
		if v == x {
			return true
		}
	}
	return false
}

func mustWrite(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// A re-pull must replace a manifest an older catalog generated (that is how `pull grounding-dino`
// moves an old install onto the fixed-mask graph), but never one a user edited.
func TestPullRegeneratesOnlyGeneratedManifests(t *testing.T) {
	dir := t.TempDir()
	e, _ := Lookup("rfdetr-gdino")
	for _, dep := range e.Dependencies {
		mustWrite(t, filepath.Join(dir, dep, "manifest.yaml"), "stub")
	}
	for _, rel := range []string{"rf-detr/rf-detr-base.onnx", "rf-detr/coco91.txt", "grounding-dino/model-fixedmask.onnx"} {
		mustWrite(t, filepath.Join(dir, rel), "x")
	}
	path := filepath.Join(dir, "rfdetr-gdino", "manifest.yaml")
	old := generatedHeader + " rfdetr-gdino` — old catalog\nfiles:\n  gdino: ../grounding-dino/model.onnx\n"
	mustWrite(t, path, old)
	var log bytes.Buffer
	if err := Pull("rfdetr-gdino", PullOptions{ModelsDir: dir, Out: &log}); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(path); string(got) != e.RenderManifest() {
		t.Fatalf("stale generated manifest was kept:\n%s", got)
	}

	custom := "name: rfdetr-gdino\n# my edits\n"
	mustWrite(t, path, custom)
	if err := Pull("rfdetr-gdino", PullOptions{ModelsDir: dir, Out: &log}); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(path); string(got) != custom {
		t.Fatalf("hand-edited manifest was overwritten:\n%s", got)
	}
}
