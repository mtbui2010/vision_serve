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
