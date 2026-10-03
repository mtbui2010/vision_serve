package registry

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeSideFileManifest writes a single-file model whose graph keeps its tensors in external
// data (model.onnx.data, the ONNX >2 GB layout siglip uses) plus a tokenizer side file, and
// returns the manifest path. pinData/pinTok control which side files sha256_files pins.
func writeSideFileManifest(t *testing.T, dir, source string, pinData, pinTok bool) string {
	t.Helper()
	_, graph := writeWeight(t, dir, "model.onnx", []byte("graph that references model.onnx.data"))
	_, data := writeWeight(t, dir, "model.onnx.data", []byte("authentic tensor bytes"))
	_, tok := writeWeight(t, dir, "tokenizer.json", []byte(`{"vocab": {}}`))
	y := "name: side\ntask: classification\nlicense: Apache-2.0\nmodel_file: model.onnx\n" +
		"source_url: " + source + "\n" +
		"sha256: " + graph + "\n"
	if pinData || pinTok {
		y += "sha256_files:\n"
		if pinData {
			y += "  model.onnx.data: " + data + "\n"
		}
		if pinTok {
			y += "  tokenizer.json: \"SHA256:" + strings.ToUpper(tok) + "\"\n"
		}
	}
	y += "input:\n  width: 8\n  height: 8\nruntime:\n  prefer: [cpu]\n"
	p := filepath.Join(dir, "manifest.yaml")
	if err := os.WriteFile(p, []byte(y), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// Review #8: the sha256 pin covered model.onnx but never its external data or side files, so
// the tensors (the actual weights) could be swapped under a pinned graph and still load.
func TestVerifyWeightsSideFilesPinned(t *testing.T) {
	dir := t.TempDir()
	m, err := LoadManifest(writeSideFileManifest(t, dir, "https://example.com/x/", true, true))
	if err != nil {
		t.Fatal(err)
	}
	if err := m.VerifyWeights(); err != nil {
		t.Fatalf("intact side files must verify: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.onnx.data"), []byte("SWAPPED tensors"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := m.VerifyWeights(); err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("swapped external data under a pinned graph must be refused, got: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.onnx.data"), []byte("authentic tensor bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{"vocab": {"evil": 1}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := m.VerifyWeights(); err == nil || !strings.Contains(err.Error(), "tokenizer.json") {
		t.Fatalf("a modified pinned side file must be refused, got: %v", err)
	}
}

// Old manifests (no sha256_files) stay valid and load in the default gate, external data or not.
func TestVerifyWeightsSideFilesBackwardCompatible(t *testing.T) {
	dir := t.TempDir()
	m, err := LoadManifest(writeSideFileManifest(t, dir, "https://example.com/x/", false, false))
	if err != nil {
		t.Fatal(err)
	}
	if err := m.VerifyWeights(); err != nil {
		t.Fatalf("default gate must not require side-file pins: %v", err)
	}
}

// Verified mode: external weight data next to a pinned graph must itself be pinned, otherwise
// omitting the sha256_files line would opt the tensors out of the check.
func TestVerifiedModeRequiresExternalDataPin(t *testing.T) {
	src := "https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX/" // audited, no recorded digests
	EnableVerifiedMode()
	defer DisableVerifiedMode()

	dir := t.TempDir()
	m, err := LoadManifest(writeSideFileManifest(t, dir, src, false, false))
	if err != nil {
		t.Fatal(err)
	}
	if err := m.VerifyWeights(); err == nil || !strings.Contains(err.Error(), "model.onnx.data") {
		t.Fatalf("verified mode must refuse unpinned external data, got: %v", err)
	}

	dir2 := t.TempDir()
	m2, err := LoadManifest(writeSideFileManifest(t, dir2, src, true, false))
	if err != nil {
		t.Fatal(err)
	}
	if err := m2.VerifyWeights(); err != nil {
		t.Fatalf("pinned external data must be admitted in verified mode: %v", err)
	}
}

// The ledger anchor covers side-file digests too: a self-consistent sha256_files digest that the
// maintainer never audited is refused, exactly like a forged main pin (gate case G).
func TestAnchoredPinsCoverSideFiles(t *testing.T) {
	src := "https://huggingface.co/mtbui2010/siglip-base-patch16-224-ONNX/"
	led, ok := lookupLedger(src)
	if !ok || len(led.WeightSHA256) < 2 {
		t.Fatal("test needs the siglip ledger entry")
	}
	m := &Manifest{Name: "siglip", SourceURL: src}
	m.SHA256 = SHA256Field{single: led.WeightSHA256[0]}
	m.SHA256Files = map[string]string{"model.onnx.data": led.WeightSHA256[1]}
	if err := m.checkAnchoredPins(); err != nil {
		t.Fatalf("audited side-file digest must pass: %v", err)
	}
	m.SHA256Files["model.onnx.data"] = strings.Repeat("e", 64)
	if err := m.checkAnchoredPins(); err == nil || !strings.Contains(err.Error(), "not among the digests") {
		t.Fatalf("an unaudited side-file digest must be refused, got: %v", err)
	}
}

func TestValidateRejectsEscapingSideFile(t *testing.T) {
	dir := t.TempDir()
	y := "name: x\ntask: classification\nlicense: MIT\nmodel_file: m.onnx\n" +
		"sha256_files:\n  ../other/model.onnx.data: " + strings.Repeat("a", 64) + "\n" +
		"input:\n  width: 8\n  height: 8\n"
	p := filepath.Join(dir, "manifest.yaml")
	if err := os.WriteFile(p, []byte(y), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadManifest(p); err == nil || !strings.Contains(err.Error(), "sha256_files") {
		t.Fatalf("a side-file pin outside the model dir must be rejected, got: %v", err)
	}
}
