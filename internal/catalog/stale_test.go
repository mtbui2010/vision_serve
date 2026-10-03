package catalog

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// age sets path's mtime d in the past (path itself, not a symlink target: callers only age
// directories and regular files).
func age(t *testing.T, path string, d time.Duration) {
	t.Helper()
	old := time.Now().Add(-d)
	if err := os.Chtimes(path, old, old); err != nil {
		t.Fatal(err)
	}
}

// An interrupted `pull <folder>` leaves <models>/.tmp-<name>-<n> (the staging copy) and
// <models>/.tmp-<name>-<n>-old (the previous install moved aside). The next install removes those
// once they are old — and nothing else: not fresh ones (a concurrent install's), not other names,
// not files, not symlinks or what they point to, not anything below the registry's direct children.
func TestInstallRemovesStaleStagingOnly(t *testing.T) {
	models := t.TempDir()
	outside := t.TempDir()
	mk := func(rel string, ago time.Duration) string {
		p := filepath.Join(models, rel)
		mustWrite(t, filepath.Join(p, "manifest.yaml"), "name: x\n")
		age(t, p, ago)
		return p
	}
	staleStage := mk(".tmp-my-detector-123456", 2*time.Hour)
	staleAside := mk(".tmp-other-42-old", 3*time.Hour)
	mk("other", 3*time.Hour)                         // "other" is installed again, so its aside can go
	onlyCopy := mk(".tmp-lonely-7-old", 3*time.Hour) // no "lonely" install: the aside is the only copy
	fresh := mk(".tmp-busy-777", time.Minute)        // a concurrent install's staging
	userModel := mk("legacy-old", 5*time.Hour)       // a model that is merely named *-old
	notOurs := mk(".tmp-notdigits", 5*time.Hour)     // not a MkdirTemp name
	nested := mk("legacy-old/.tmp-x-1", 5*time.Hour) // below a model directory
	age(t, userModel, 5*time.Hour)
	staleFile := filepath.Join(models, ".tmp-file-9")
	mustWrite(t, staleFile, "x")
	age(t, staleFile, 5*time.Hour)
	// A symlink with a staging name, pointing outside the registry at an old directory.
	target := filepath.Join(outside, "precious")
	mustWrite(t, filepath.Join(target, "keep.txt"), "keep")
	age(t, target, 5*time.Hour)
	link := filepath.Join(models, ".tmp-link-5")
	if err := os.Symlink(target, link); err != nil {
		t.Skip("symlinks unsupported:", err)
	}

	var out bytes.Buffer
	if err := InstallLocal(writeModelFolder(t, validManifest), PullOptions{ModelsDir: models, Out: &out}); err != nil {
		t.Fatalf("InstallLocal: %v\n%s", err, out.String())
	}

	for _, gone := range []string{staleStage, staleAside} {
		if _, err := os.Lstat(gone); !os.IsNotExist(err) {
			t.Errorf("stale staging %s was not removed", filepath.Base(gone))
		}
	}
	for _, kept := range []string{onlyCopy, fresh, userModel, notOurs, nested, staleFile, link, filepath.Join(target, "keep.txt")} {
		if _, err := os.Lstat(kept); err != nil {
			t.Errorf("%s must be kept: %v", kept, err)
		}
	}
	if !strings.Contains(out.String(), "removed .tmp-my-detector-123456") || !strings.Contains(out.String(), "removed .tmp-other-42-old") {
		t.Errorf("cleanup not reported:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "WARNING: .tmp-lonely-7-old holds the previous install") {
		t.Errorf("the kept only copy is not reported:\n%s", out.String())
	}
	if _, err := os.Stat(filepath.Join(models, "my-detector", "manifest.yaml")); err != nil {
		t.Errorf("the install itself did not happen: %v", err)
	}
}

// Catalog pulls clean up too (any install is a chance to), and a missing registry is not an error.
func TestPullRemovesStaleStaging(t *testing.T) {
	models := t.TempDir()
	stale := filepath.Join(models, ".tmp-x-1")
	mustWrite(t, filepath.Join(stale, "model.onnx"), "x")
	age(t, stale, 2*time.Hour)
	e, _ := Lookup("rfdetr-gdino")
	for _, dep := range e.Dependencies {
		mustWrite(t, filepath.Join(models, dep, "manifest.yaml"), "stub")
	}
	for _, rel := range e.VirtualFiles {
		mustWrite(t, filepath.Join(models, e.Name, rel), "x")
	}
	mustWrite(t, filepath.Join(models, "rf-detr", "coco91.txt"), "x")
	if err := Pull(e.Name, PullOptions{ModelsDir: models, Out: &bytes.Buffer{}}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Lstat(stale); !os.IsNotExist(err) {
		t.Error("stale staging survived a catalog pull")
	}

	// A mistyped name is an error before anything in the registry is touched.
	other := filepath.Join(models, ".tmp-y-2")
	mustWrite(t, filepath.Join(other, "model.onnx"), "x")
	age(t, other, 2*time.Hour)
	if err := Pull("no-such-model", PullOptions{ModelsDir: models, Out: &bytes.Buffer{}}); err == nil {
		t.Fatal("pull of an unknown model succeeded")
	}
	if _, err := os.Lstat(other); err != nil {
		t.Errorf("a pull of an unknown model cleaned the registry: %v", err)
	}

	cleanStaleStaging(filepath.Join(t.TempDir(), "does-not-exist"), &bytes.Buffer{}) // must not panic or create it
}

// A live install keeps its directories fresh: the staging root is touched after every copied
// file (a file in a sub-directory does not update it), and the install being replaced is touched
// right before it is moved aside (a rename keeps its old mtime).
func TestInstallKeepsItsOwnStagingFresh(t *testing.T) {
	src := writeModelFolder(t, validManifest)
	mustWrite(t, filepath.Join(src, "sub", "deep", "extra.bin"), "x")
	stage := filepath.Join(t.TempDir(), "stage")
	if err := os.MkdirAll(stage, 0o755); err != nil {
		t.Fatal(err)
	}
	age(t, stage, 2*time.Hour)
	if _, err := copyTree(src, stage); err != nil {
		t.Fatal(err)
	}
	if st, _ := os.Stat(stage); time.Since(st.ModTime()) > time.Minute {
		t.Errorf("staging root mtime not refreshed by the copy: %v", st.ModTime())
	}

	// --force over an old install: the moved-aside copy must not look stale while it exists, or a
	// concurrent install would delete the copy this one restores from on failure.
	models := t.TempDir()
	installed := filepath.Join(models, "my-detector")
	mustWrite(t, filepath.Join(installed, "manifest.yaml"), "x")
	age(t, installed, 10*time.Hour)
	aside := filepath.Join(models, ".tmp-my-detector-1-old")
	if err := moveAside(installed, aside); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	cleanStaleStaging(models, &out)
	if _, err := os.Stat(filepath.Join(aside, "manifest.yaml")); err != nil {
		t.Fatalf("a just-moved-aside install was removed as stale: %v\n%s", err, out.String())
	}
}
