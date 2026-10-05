package registry

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// A model folder added after the first scan appears after Refresh, and Refresh rescans at most
// once per interval.
func TestRefreshPicksUpNewModelsRateLimited(t *testing.T) {
	root := t.TempDir()
	r := New(root)
	if _, err := r.Scan(); err != nil {
		t.Fatal(err)
	}
	write := func(name string) {
		dir := filepath.Join(root, name)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		m := "name: " + name + "\ntask: detection\nlicense: Apache-2.0\nmodel_file: m.onnx\n" +
			"input:\n  width: 32\n  height: 32\n"
		if err := os.WriteFile(filepath.Join(dir, "manifest.yaml"), []byte(m), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	write("first")
	if !r.Refresh(time.Hour) {
		t.Fatal("the first Refresh did not rescan")
	}
	if _, ok := r.Get("first"); !ok {
		t.Fatal("a model added after Scan is missing after Refresh")
	}

	write("second")
	if r.Refresh(time.Hour) {
		t.Fatal("Refresh rescanned again within its interval")
	}
	if _, ok := r.Get("second"); ok {
		t.Fatal("rate-limited Refresh still picked up a new model")
	}
	if !r.Refresh(0) {
		t.Fatal("Refresh(0) did not rescan")
	}
	if _, ok := r.Get("second"); !ok {
		t.Fatal("a model added later is missing after the next Refresh")
	}
}
