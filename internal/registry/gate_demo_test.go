package registry

import (
	"strings"
	"testing"
)

// TestGate_AdversarialDemo is the reproducible "license-enforcing loader" demo for the paper
// (contribution C1). It shows, end-to-end, that the load-time gate REFUSES a model under each
// of the three adversarial conditions the threat model names, and ADMITS only a model whose
// declared-permissive license is bound to audited bytes from an audited origin:
//
//	(0) baseline      — permissive license + correct sha256 + audited source  -> ADMITTED
//	(A) relabel       — a non-permissive (AGPL) model declared at validate()   -> REFUSED (license)
//	(B) byte-swap     — declared sha256 ≠ actual bytes (weights tampered/swapped) -> REFUSED (hash)
//	(C) wrong-origin  — source_url outside the audited allowlist               -> REFUSED (source)
//
// Run:  go test ./internal/registry -run TestGate_AdversarialDemo -v
// The -v transcript is the figure/listing reproduced in the paper (docs/threat-model.md).
func TestGate_AdversarialDemo(t *testing.T) {
	// A valid permissive manifest helper (passes structural validate()).
	newPermissive := func(dir string) *Manifest {
		m := &Manifest{
			Name:      "demo-model",
			License:   "Apache-2.0",
			Task:      "classification",
			ModelFile: "model.onnx",
			SourceURL: "https://huggingface.co/onnxmodelzoo/mobilenet_v3_small_Opset17/",
			dir:       dir,
		}
		m.Input.Width, m.Input.Height = 224, 224
		m.Input.Layout = "NCHW"
		m.Runtime.Prefer = []string{"cpu"}
		return m
	}

	// (0) BASELINE — permissive + correct hash + audited source -> ADMITTED ------------------
	t.Run("0_baseline_admitted", func(t *testing.T) {
		dir := t.TempDir()
		_, digest := writeWeight(t, dir, "model.onnx", []byte("authentic permissive weights"))
		m := newPermissive(dir)
		m.SHA256 = SHA256Field{single: digest}

		// audited-source allowlist ON, source_url under an audited prefix
		defer setAllowlist([]string{"https://huggingface.co/onnxmodelzoo/"})()

		if err := m.validate(); err != nil {
			t.Fatalf("baseline must pass structural validate(), got: %v", err)
		}
		if err := m.VerifyWeights(); err != nil {
			t.Fatalf("baseline must pass the load-time gate, got: %v", err)
		}
		t.Logf("ADMITTED: license=%s, sha256 bound to bytes, source under audited prefix", m.License)
	})

	// (A) RELABEL — a copyleft (AGPL) model declared -> REFUSED at validate() -----------------
	t.Run("A_agpl_license_refused", func(t *testing.T) {
		dir := t.TempDir()
		writeWeight(t, dir, "model.onnx", []byte("agpl model bytes"))
		m := newPermissive(dir)
		m.License = "AGPL-3.0" // e.g. Ultralytics YOLO / FastSAM / YOLO-World

		err := m.validate()
		if err == nil || !strings.Contains(err.Error(), "not allowed") {
			t.Fatalf("AGPL model must be refused at validate(), got: %v", err)
		}
		t.Logf("REFUSED (license): %v", err)
	})

	// (B) BYTE-SWAP — declared sha256 ≠ actual bytes -> REFUSED at load ----------------------
	t.Run("B_hash_mismatch_refused", func(t *testing.T) {
		dir := t.TempDir()
		// weights on disk are NOT what the manifest's digest pins (swapped/tampered bytes)
		writeWeight(t, dir, "model.onnx", []byte("SWAPPED malicious weights"))
		m := newPermissive(dir)
		m.SHA256 = SHA256Field{single: strings.Repeat("a", 64)} // digest of the *expected* bytes

		err := m.VerifyWeights()
		if err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
			t.Fatalf("tampered weights must be refused on hash mismatch, got: %v", err)
		}
		t.Logf("REFUSED (hash): %v", err)
	})

	// (C) WRONG-ORIGIN — source_url outside the audited allowlist -> REFUSED -----------------
	t.Run("C_unaudited_source_refused", func(t *testing.T) {
		dir := t.TempDir()
		_, digest := writeWeight(t, dir, "model.onnx", []byte("weights from an unvetted mirror"))
		m := newPermissive(dir)
		m.SHA256 = SHA256Field{single: digest}                 // hash is fine...
		m.SourceURL = "https://random-mirror.example.com/x.onnx" // ...but origin is not audited

		defer setAllowlist([]string{"https://huggingface.co/onnxmodelzoo/"})()

		err := m.VerifyWeights()
		if err == nil || !strings.Contains(err.Error(), "not under any audited prefix") {
			t.Fatalf("unaudited source must be refused, got: %v", err)
		}
		t.Logf("REFUSED (source): %v", err)
	})
}

// setAllowlist sets the package-global VerifiedSourcePrefixes for a test and returns a
// restore func (call via defer) so other tests see the default empty allowlist.
func setAllowlist(prefixes []string) func() {
	prev := VerifiedSourcePrefixes
	VerifiedSourcePrefixes = prefixes
	return func() { VerifiedSourcePrefixes = prev }
}
