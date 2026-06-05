package registry

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// writeWeight writes some bytes to <dir>/<name> and returns the path + its sha256.
func writeWeight(t *testing.T, dir, name string, content []byte) (path, digest string) {
	t.Helper()
	path = filepath.Join(dir, name)
	if err := os.WriteFile(path, content, 0o644); err != nil {
		t.Fatalf("write weight: %v", err)
	}
	sum := sha256.Sum256(content)
	return path, hex.EncodeToString(sum[:])
}

// A correct sha256 for a single-file model passes.
func TestVerifyWeightsSingleFileMatch(t *testing.T) {
	dir := t.TempDir()
	_, digest := writeWeight(t, dir, "model.onnx", []byte("fake onnx bytes"))

	m := &Manifest{Name: "x", License: "Apache-2.0", Task: "detection", ModelFile: "model.onnx", dir: dir}
	m.SHA256 = SHA256Field{single: digest}
	if err := m.VerifyWeights(); err != nil {
		t.Fatalf("expected match to pass, got: %v", err)
	}
}

// A wrong sha256 is rejected with a clear mismatch error (the relabeling hole).
func TestVerifyWeightsSingleFileMismatch(t *testing.T) {
	dir := t.TempDir()
	writeWeight(t, dir, "model.onnx", []byte("fake onnx bytes"))

	m := &Manifest{Name: "x", License: "Apache-2.0", Task: "detection", ModelFile: "model.onnx", dir: dir}
	m.SHA256 = SHA256Field{single: strings.Repeat("0", 64)}
	err := m.VerifyWeights()
	if err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("expected a sha256 mismatch rejection, got: %v", err)
	}
}

// No sha256 declared → still loads (backward compatibility — the common case).
func TestVerifyWeightsAbsentIsBackwardCompatible(t *testing.T) {
	dir := t.TempDir()
	writeWeight(t, dir, "model.onnx", []byte("fake onnx bytes"))

	m := &Manifest{Name: "x", License: "Apache-2.0", Task: "detection", ModelFile: "model.onnx", dir: dir}
	if !m.SHA256.IsEmpty() {
		t.Fatal("SHA256 should be empty when not declared")
	}
	if err := m.VerifyWeights(); err != nil {
		t.Fatalf("absent sha256 must not block load, got: %v", err)
	}
}

// Multi-file model: each pinned role is verified; correct digests pass.
func TestVerifyWeightsMultiFileMatch(t *testing.T) {
	dir := t.TempDir()
	_, encDigest := writeWeight(t, dir, "enc.onnx", []byte("encoder graph"))
	_, decDigest := writeWeight(t, dir, "dec.onnx", []byte("decoder graph"))

	m := &Manifest{
		Name: "sam", License: "Apache-2.0", Task: "segmentation", dir: dir,
		Files: map[string]string{"encoder": "enc.onnx", "decoder": "dec.onnx"},
	}
	m.SHA256 = SHA256Field{byRole: map[string]string{"encoder": encDigest, "decoder": decDigest}}
	if err := m.VerifyWeights(); err != nil {
		t.Fatalf("expected multi-file match to pass, got: %v", err)
	}
}

// Multi-file model: a single bad role digest is rejected.
func TestVerifyWeightsMultiFileMismatch(t *testing.T) {
	dir := t.TempDir()
	_, encDigest := writeWeight(t, dir, "enc.onnx", []byte("encoder graph"))
	writeWeight(t, dir, "dec.onnx", []byte("decoder graph"))

	m := &Manifest{
		Name: "sam", License: "Apache-2.0", Task: "segmentation", dir: dir,
		Files: map[string]string{"encoder": "enc.onnx", "decoder": "dec.onnx"},
	}
	m.SHA256 = SHA256Field{byRole: map[string]string{"encoder": encDigest, "decoder": strings.Repeat("a", 64)}}
	err := m.VerifyWeights()
	if err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("expected a mismatch rejection for the decoder, got: %v", err)
	}
}

// Multi-file model: pinning only ONE role still verifies that role and skips the
// unpinned one (optional per-role pinning).
func TestVerifyWeightsMultiFilePartialPin(t *testing.T) {
	dir := t.TempDir()
	_, encDigest := writeWeight(t, dir, "enc.onnx", []byte("encoder graph"))
	// decoder file exists but is intentionally not pinned and not written-to-match.
	writeWeight(t, dir, "dec.onnx", []byte("decoder graph"))

	m := &Manifest{
		Name: "sam", License: "Apache-2.0", Task: "segmentation", dir: dir,
		Files: map[string]string{"encoder": "enc.onnx", "decoder": "dec.onnx"},
	}
	m.SHA256 = SHA256Field{byRole: map[string]string{"encoder": encDigest}}
	if err := m.VerifyWeights(); err != nil {
		t.Fatalf("partial pin should verify encoder only, got: %v", err)
	}
}

// The sha256 field unmarshals from BOTH a scalar string and a role→hash map.
func TestSHA256FieldUnmarshalShapes(t *testing.T) {
	t.Run("scalar", func(t *testing.T) {
		var m struct {
			SHA256 SHA256Field `yaml:"sha256"`
		}
		// Mixed case + a "sha256:" algo prefix must be normalized away.
		if err := yaml.Unmarshal([]byte("sha256: \"SHA256:AB12CD\"\n"), &m); err != nil {
			t.Fatalf("unmarshal scalar: %v", err)
		}
		got, ok := m.SHA256.expectedFor("")
		if !ok || got != "ab12cd" {
			t.Fatalf("scalar digest normalize failed: got %q ok=%v", got, ok)
		}
	})
	t.Run("map", func(t *testing.T) {
		var m struct {
			SHA256 SHA256Field `yaml:"sha256"`
		}
		y := "sha256:\n  encoder: \"AA\"\n  decoder: \"bb\"\n"
		if err := yaml.Unmarshal([]byte(y), &m); err != nil {
			t.Fatalf("unmarshal map: %v", err)
		}
		if d, ok := m.SHA256.expectedFor("encoder"); !ok || d != "aa" {
			t.Fatalf("encoder digest: got %q ok=%v", d, ok)
		}
		if d, ok := m.SHA256.expectedFor("decoder"); !ok || d != "bb" {
			t.Fatalf("decoder digest: got %q ok=%v", d, ok)
		}
	})
	t.Run("absent", func(t *testing.T) {
		var m struct {
			SHA256 SHA256Field `yaml:"sha256"`
		}
		if err := yaml.Unmarshal([]byte("name: x\n"), &m); err != nil {
			t.Fatalf("unmarshal absent: %v", err)
		}
		if !m.SHA256.IsEmpty() {
			t.Fatal("absent sha256 must be empty")
		}
	})
}

// The verified-source allowlist hook is opt-in: empty by default it changes
// nothing; populated, it rejects a source_url outside the audited prefixes and
// accepts one inside.
func TestVerifiedSourceAllowlist(t *testing.T) {
	dir := t.TempDir()
	_, digest := writeWeight(t, dir, "model.onnx", []byte("fake onnx bytes"))

	base := func() *Manifest {
		m := &Manifest{Name: "x", License: "Apache-2.0", Task: "detection", ModelFile: "model.onnx", dir: dir}
		m.SHA256 = SHA256Field{single: digest}
		return m
	}

	// Default (empty allowlist): source_url is ignored, load proceeds.
	if err := base().VerifyWeights(); err != nil {
		t.Fatalf("empty allowlist must not reject, got: %v", err)
	}

	// Configure an allowlist for the rest of this test only.
	orig := VerifiedSourcePrefixes
	VerifiedSourcePrefixes = []string{"https://huggingface.co/official/"}
	defer func() { VerifiedSourcePrefixes = orig }()

	// Missing source_url under an enforced allowlist → rejected.
	if err := base().VerifyWeights(); err == nil || !strings.Contains(err.Error(), "source_url") {
		t.Fatalf("expected missing source_url rejection, got: %v", err)
	}

	// Out-of-allowlist source_url → rejected.
	bad := base()
	bad.SourceURL = "https://huggingface.co/some-random-user/model"
	if err := bad.VerifyWeights(); err == nil || !strings.Contains(err.Error(), "allowlist") {
		t.Fatalf("expected out-of-allowlist rejection, got: %v", err)
	}

	// Audited source_url + matching hash → passes.
	good := base()
	good.SourceURL = "https://huggingface.co/official/rf-detr"
	if err := good.VerifyWeights(); err != nil {
		t.Fatalf("audited source + matching hash must pass, got: %v", err)
	}
}

// End-to-end: a manifest YAML with sha256 round-trips through LoadManifest and
// verifies (proves the field is wired into the real parse path, not just the struct).
func TestLoadManifestWithSHA256Verifies(t *testing.T) {
	dir := t.TempDir()
	_, digest := writeWeight(t, dir, "model.onnx", []byte("real-ish onnx"))
	y := "name: x\n" +
		"task: detection\n" +
		"license: Apache-2.0\n" +
		"model_file: model.onnx\n" +
		"source_url: https://huggingface.co/official/x\n" +
		"sha256: \"" + digest + "\"\n" +
		"input:\n  width: 10\n  height: 10\n"
	mpath := filepath.Join(dir, "manifest.yaml")
	if err := os.WriteFile(mpath, []byte(y), 0o644); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	m, err := LoadManifest(mpath)
	if err != nil {
		t.Fatalf("LoadManifest: %v", err)
	}
	if m.SourceURL != "https://huggingface.co/official/x" {
		t.Fatalf("source_url not parsed: %q", m.SourceURL)
	}
	if err := m.VerifyWeights(); err != nil {
		t.Fatalf("verify after load: %v", err)
	}
}
