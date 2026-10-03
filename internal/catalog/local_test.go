package catalog

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"visionserve/internal/models"
)

// register a throwaway architecture so InstallLocal's "is the architecture a
// registered factory?" check passes for the happy-path tests. Real architectures
// are registered by blank-importing the model packages (see cmd/visionserve).
func init() {
	models.Register("test-arch", func(models.Config) (models.Base, error) { return nil, nil })
}

const validManifest = `name: my-detector
task: detection
license: Apache-2.0
architecture: test-arch
model_file: model.onnx
input:
  width: 560
  height: 560
  layout: NCHW
postprocess:
  type: detr
runtime:
  prefer: [cpu]
`

// writeModelFolder builds a src model folder with a manifest + a dummy >1KiB
// .onnx (so WeightsExist passes) and returns its path.
func writeModelFolder(t *testing.T, manifest string) string {
	t.Helper()
	dir := t.TempDir()
	src := filepath.Join(dir, "src")
	if err := os.MkdirAll(src, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(src, "manifest.yaml"), []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(src, "model.onnx"), bytes.Repeat([]byte{0}, 2048), 0o644); err != nil {
		t.Fatal(err)
	}
	return src
}

func TestInstallLocal_CopiesValidFolder(t *testing.T) {
	src := writeModelFolder(t, validManifest)
	modelsDir := t.TempDir()

	var out bytes.Buffer
	if err := InstallLocal(src, PullOptions{ModelsDir: modelsDir, Out: &out}); err != nil {
		t.Fatalf("InstallLocal: %v", err)
	}

	dst := filepath.Join(modelsDir, "my-detector")
	for _, f := range []string{"manifest.yaml", "model.onnx"} {
		if _, err := os.Stat(filepath.Join(dst, f)); err != nil {
			t.Errorf("expected %s in registry: %v", f, err)
		}
	}
	if !strings.Contains(out.String(), "installed my-detector") {
		t.Errorf("missing success message, got: %q", out.String())
	}
}

func TestInstallLocal_RejectsUnknownArchitecture(t *testing.T) {
	m := strings.Replace(validManifest, "architecture: test-arch", "architecture: not-a-real-arch", 1)
	src := writeModelFolder(t, m)

	err := InstallLocal(src, PullOptions{ModelsDir: t.TempDir(), Out: &bytes.Buffer{}})
	if err == nil || !strings.Contains(err.Error(), "architecture") {
		t.Fatalf("expected architecture error, got: %v", err)
	}
}

func TestInstallLocal_RejectsAGPLLicense(t *testing.T) {
	m := strings.Replace(validManifest, "license: Apache-2.0", "license: AGPL-3.0", 1)
	src := writeModelFolder(t, m)

	err := InstallLocal(src, PullOptions{ModelsDir: t.TempDir(), Out: &bytes.Buffer{}})
	if err == nil || !strings.Contains(err.Error(), "license") {
		t.Fatalf("expected license rejection, got: %v", err)
	}
}

func TestInstallLocal_RejectsMissingWeights(t *testing.T) {
	src := writeModelFolder(t, validManifest)
	if err := os.Remove(filepath.Join(src, "model.onnx")); err != nil {
		t.Fatal(err)
	}
	err := InstallLocal(src, PullOptions{ModelsDir: t.TempDir(), Out: &bytes.Buffer{}})
	if err == nil || !strings.Contains(err.Error(), "weights") {
		t.Fatalf("expected missing-weights error, got: %v", err)
	}
}

func TestInstallLocal_RejectsEscapingPath(t *testing.T) {
	m := strings.Replace(validManifest, "model_file: model.onnx", "model_file: ../escape.onnx", 1)
	src := writeModelFolder(t, m)
	err := InstallLocal(src, PullOptions{ModelsDir: t.TempDir(), Out: &bytes.Buffer{}})
	if err == nil || !strings.Contains(err.Error(), "outside") {
		t.Fatalf("expected escape rejection, got: %v", err)
	}
}

func TestInstallLocal_RefusesOverwriteWithoutForce(t *testing.T) {
	src := writeModelFolder(t, validManifest)
	modelsDir := t.TempDir()
	opts := PullOptions{ModelsDir: modelsDir, Out: &bytes.Buffer{}}

	if err := InstallLocal(src, opts); err != nil {
		t.Fatalf("first install: %v", err)
	}
	// Second install without force must fail.
	if err := InstallLocal(src, opts); err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("expected already-exists error, got: %v", err)
	}
	// With force it should succeed.
	opts.Force = true
	if err := InstallLocal(src, opts); err != nil {
		t.Fatalf("force install: %v", err)
	}
}

func TestInstallLocal_MissingManifest(t *testing.T) {
	dir := t.TempDir() // empty, no manifest.yaml
	err := InstallLocal(dir, PullOptions{ModelsDir: t.TempDir(), Out: &bytes.Buffer{}})
	if err == nil || !strings.Contains(err.Error(), "manifest.yaml") {
		t.Fatalf("expected missing-manifest error, got: %v", err)
	}
}

// Review #1: the registry reached through a symlink is the SAME directory as the source. A
// string compare of the two absolute paths missed that, and --force then truncated the source
// weights to 0 bytes (os.Create on the file it was about to read).
func TestInstallLocal_ForceSameDirViaSymlinkKeepsWeights(t *testing.T) {
	root := t.TempDir()
	real := filepath.Join(root, "real")
	src := filepath.Join(real, "my-detector")
	mustWrite(t, filepath.Join(src, "manifest.yaml"), validManifest)
	mustWrite(t, filepath.Join(src, "model.onnx"), strings.Repeat("w", 2048))
	link := filepath.Join(root, "link")
	if err := os.Symlink(real, link); err != nil {
		t.Skipf("symlinks unsupported: %v", err)
	}
	if err := InstallLocal(src, PullOptions{ModelsDir: link, Force: true, Out: &bytes.Buffer{}}); err != nil {
		t.Fatalf("InstallLocal: %v", err)
	}
	if b, _ := os.ReadFile(filepath.Join(src, "model.onnx")); len(b) != 2048 {
		t.Fatalf("source weights destroyed: %d bytes left", len(b))
	}
}

// Review #2: a registry INSIDE the source folder made copyTree copy into its own walk.
func TestInstallLocal_RefusesRegistryInsideSource(t *testing.T) {
	src := writeModelFolder(t, validManifest)
	err := InstallLocal(src, PullOptions{ModelsDir: filepath.Join(src, "models"), Out: &bytes.Buffer{}})
	if err == nil || !strings.Contains(err.Error(), "inside the source folder") {
		t.Fatalf("expected a registry-inside-source refusal, got: %v", err)
	}
	// Same through "." (the reviewer's `pull . --models ./models`), via a symlinked source.
	link := filepath.Join(t.TempDir(), "srclink")
	if err := os.Symlink(src, link); err != nil {
		t.Skipf("symlinks unsupported: %v", err)
	}
	err = InstallLocal(link, PullOptions{ModelsDir: filepath.Join(src, "reg"), Out: &bytes.Buffer{}})
	if err == nil || !strings.Contains(err.Error(), "inside the source folder") {
		t.Fatalf("expected a registry-inside-source refusal through a symlink, got: %v", err)
	}
}

// Review #3: HF-cache style folders are symlinks into a blob store. They passed the existence
// check, then copyTree skipped every non-regular file and reported "installed" without weights.
func TestInstallLocal_CopiesSymlinkedWeightsAndFolders(t *testing.T) {
	root := t.TempDir()
	blob := filepath.Join(root, "blobs", "abc123")
	mustWrite(t, blob, strings.Repeat("b", 4096))
	snap := filepath.Join(root, "snapshot")
	mustWrite(t, filepath.Join(snap, "manifest.yaml"), validManifest)
	if err := os.Symlink(blob, filepath.Join(snap, "model.onnx")); err != nil {
		t.Skipf("symlinks unsupported: %v", err)
	}
	modelsDir := t.TempDir()
	if err := InstallLocal(snap, PullOptions{ModelsDir: modelsDir, Out: &bytes.Buffer{}}); err != nil {
		t.Fatalf("InstallLocal: %v", err)
	}
	dst := filepath.Join(modelsDir, "my-detector", "model.onnx")
	st, err := os.Lstat(dst)
	if err != nil || !st.Mode().IsRegular() || st.Size() != 4096 {
		t.Fatalf("symlinked weights not copied as content: %v %v", st, err)
	}

	// A symlink to the whole folder installs its files too (it used to install 0).
	link := filepath.Join(root, "folderlink")
	if err := os.Symlink(writeModelFolder(t, validManifest), link); err != nil {
		t.Fatal(err)
	}
	modelsDir2 := t.TempDir()
	if err := InstallLocal(link, PullOptions{ModelsDir: modelsDir2, Out: &bytes.Buffer{}}); err != nil {
		t.Fatalf("InstallLocal via folder symlink: %v", err)
	}
	for _, f := range []string{"manifest.yaml", "model.onnx"} {
		if _, err := os.Stat(filepath.Join(modelsDir2, "my-detector", f)); err != nil {
			t.Errorf("%s missing after installing a symlinked folder: %v", f, err)
		}
	}
}

// Review #3 (refusal half): a referenced file that is a symlink to a directory cannot be copied
// as content; the install must fail instead of reporting success without it.
func TestInstallLocal_FailsWhenReferencedFileNotCopied(t *testing.T) {
	root := t.TempDir()
	src := filepath.Join(root, "src")
	mustWrite(t, filepath.Join(src, "manifest.yaml"), validManifest)
	weightsDir := filepath.Join(root, "weights-dir")
	mustWrite(t, filepath.Join(weightsDir, "x"), "y")
	if err := os.Symlink(weightsDir, filepath.Join(src, "model.onnx")); err != nil {
		t.Skipf("symlinks unsupported: %v", err)
	}
	modelsDir := t.TempDir()
	if err := InstallLocal(src, PullOptions{ModelsDir: modelsDir, Out: &bytes.Buffer{}}); err == nil {
		t.Fatal("a model.onnx that is a directory symlink must not install")
	}
	if _, err := os.Stat(filepath.Join(modelsDir, "my-detector")); err == nil {
		t.Fatal("a failed install left a model directory behind")
	}
}

// Review #4: --force must replace the install as a whole (no stale files from the old one), and
// a failed install must leave the previous install untouched and no staging debris.
func TestInstallLocal_ForceReplacesAtomically(t *testing.T) {
	modelsDir := t.TempDir()
	src := writeModelFolder(t, validManifest)
	mustWrite(t, filepath.Join(src, "stale.txt"), "old")
	opts := PullOptions{ModelsDir: modelsDir, Out: &bytes.Buffer{}}
	if err := InstallLocal(src, opts); err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(modelsDir, "my-detector")

	// v2 drops stale.txt and changes the weights.
	if err := os.Remove(filepath.Join(src, "stale.txt")); err != nil {
		t.Fatal(err)
	}
	mustWrite(t, filepath.Join(src, "model.onnx"), strings.Repeat("2", 3000))
	opts.Force = true
	if err := InstallLocal(src, opts); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dst, "stale.txt")); err == nil {
		t.Error("--force left a stale file from the previous install")
	}
	if st, _ := os.Stat(filepath.Join(dst, "model.onnx")); st == nil || st.Size() != 3000 {
		t.Errorf("--force did not install the new weights: %v", st)
	}

	// v3 fails mid-copy (an unreadable file): v2 must survive intact.
	if os.Geteuid() == 0 {
		t.Skip("root can read a 0000 file")
	}
	bad := filepath.Join(src, "zz-unreadable.bin")
	mustWrite(t, bad, "secret")
	if err := os.Chmod(bad, 0); err != nil {
		t.Fatal(err)
	}
	defer os.Chmod(bad, 0o644)
	if err := InstallLocal(src, opts); err == nil {
		t.Fatal("expected the copy to fail on an unreadable file")
	}
	if st, _ := os.Stat(filepath.Join(dst, "model.onnx")); st == nil || st.Size() != 3000 {
		t.Errorf("a failed --force install damaged the previous one: %v", st)
	}
	if _, err := os.Stat(filepath.Join(dst, "manifest.yaml")); err != nil {
		t.Errorf("a failed --force install removed the previous manifest: %v", err)
	}
	entries, _ := os.ReadDir(modelsDir)
	for _, e := range entries {
		if e.Name() != "my-detector" {
			t.Errorf("staging debris left in the registry: %s", e.Name())
		}
	}
}
