package registry

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func writeModel(t *testing.T, root, dir, name string) {
	t.Helper()
	d := filepath.Join(root, dir)
	if err := os.MkdirAll(d, 0o755); err != nil {
		t.Fatal(err)
	}
	m := "name: " + name + "\ntask: detection\nlicense: Apache-2.0\nmodel_file: m.onnx\n" +
		"input:\n  width: 32\n  height: 32\n"
	if err := os.WriteFile(filepath.Join(d, "manifest.yaml"), []byte(m), 0o644); err != nil {
		t.Fatal(err)
	}
}

// pinRootMtime sets the root's mtime to a fixed instant, as a coarse filesystem clock would leave
// it when two changes land within one tick.
func pinRootMtime(t *testing.T, root string, at time.Time) {
	t.Helper()
	if err := os.Chtimes(root, at, at); err != nil {
		t.Fatal(err)
	}
}

// A model folder added after the first scan appears after Refresh; with the root directory
// unchanged, Refresh rescans at most once per interval.
func TestRefreshPicksUpNewModelsRateLimited(t *testing.T) {
	root := t.TempDir()
	r := New(root)
	if _, err := r.Scan(); err != nil {
		t.Fatal(err)
	}

	writeModel(t, root, "first", "first")
	if !r.Refresh(time.Hour) {
		t.Fatal("the first Refresh did not rescan")
	}
	if _, ok := r.Get("first"); !ok {
		t.Fatal("a model added after Scan is missing after Refresh")
	}

	// A manifest edited inside an existing model directory leaves the root as it was: rate-limited.
	writeModel(t, root, "first", "renamed")
	if r.Refresh(time.Hour) {
		t.Fatal("Refresh rescanned an unchanged root within its interval")
	}
	if _, ok := r.Get("renamed"); ok {
		t.Fatal("rate-limited Refresh still re-read an edited manifest")
	}
	if !r.Refresh(0) {
		t.Fatal("Refresh(0) did not rescan")
	}
	if _, ok := r.Get("renamed"); !ok {
		t.Fatal("an edited manifest is not picked up after the interval")
	}
}

// An install within the interval (a staging dir renamed into the root, as catalog.Install does)
// is listed by the very next Refresh: the converter lists /api/models right after installing.
func TestRefreshSeesAnInstallWithinTheInterval(t *testing.T) {
	root := t.TempDir()
	r := New(root)
	writeModel(t, root, "first", "first")
	if !r.Refresh(time.Hour) {
		t.Fatal("the first Refresh did not rescan")
	}

	writeModel(t, root, ".tmp-second-123", "second")
	if err := os.Rename(filepath.Join(root, ".tmp-second-123"), filepath.Join(root, "second")); err != nil {
		t.Fatal(err)
	}
	if !r.Refresh(time.Hour) {
		t.Fatal("Refresh did not rescan a root that gained a model directory")
	}
	if _, ok := r.Get("second"); !ok {
		t.Fatal("a model installed within the interval is not listed")
	}
	if r.Refresh(time.Hour) {
		t.Fatal("Refresh rescanned again although nothing changed")
	}

	// Same mtime (a coarse clock), new name: still seen.
	st, err := os.Stat(root)
	if err != nil {
		t.Fatal(err)
	}
	writeModel(t, root, "third", "third")
	pinRootMtime(t, root, st.ModTime())
	if !r.Refresh(time.Hour) {
		t.Fatal("Refresh missed a new directory whose install left the root mtime unchanged")
	}
	if _, ok := r.Get("third"); !ok {
		t.Fatal("third is not listed")
	}

	// A --force swap keeps the names but changes the root's mtime.
	if err := os.Rename(filepath.Join(root, "third"), filepath.Join(root, ".tmp-third-1-old")); err != nil {
		t.Fatal(err)
	}
	writeModel(t, root, "third", "third-v2")
	pinRootMtime(t, root, st.ModTime().Add(time.Minute))
	if !r.Refresh(time.Hour) {
		t.Fatal("Refresh missed a swapped model directory")
	}
	if _, ok := r.Get("third-v2"); !ok {
		t.Fatal("the swapped-in model is not listed")
	}
}
