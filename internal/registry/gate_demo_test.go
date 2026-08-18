package registry

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestGate_AdversarialDemo is the reproducible "license-enforcing loader" demo for the paper
// (contribution C1). It shows, end-to-end, that the load-time gate REFUSES a model under each
// adversarial condition the threat model names, and ADMITS only a model whose declared-permissive
// license is bound to audited bytes from an audited origin:
//
//	(0) baseline      — permissive license + correct sha256 + audited source  -> ADMITTED
//	(A) relabel       — a non-permissive (AGPL) model declared at validate()   -> REFUSED (license)
//	(B) byte-swap     — declared sha256 ≠ actual bytes (weights tampered/swapped) -> REFUSED (hash)
//	(C) wrong-origin  — source_url outside the audited allowlist               -> REFUSED (source)
//	(D) relicense     — permissive, but ≠ the maintainer-audited upstream licence -> REFUSED (ledger)
//	(E) unledgered    — upstream nobody audited                                -> REFUSED (no entry)
//	(F) no pin        — declares no sha256 at all                              -> REFUSED (unpinned)
//	(G) forged pin    — self-consistent sha256 over substituted bytes          -> REFUSED (unaudited pin)
//
// Cases (F) and (G) were added after (0)-(E) were found to share a blind spot: every one of them
// sets the digest itself, so between them they only ever demonstrated that a manifest agrees with
// itself. (F) is the manifest that opts out of the check by omission; (G) is the manifest that
// opts in with a digest of its own choosing. Both used to be admitted.
//
// TestGate_ComposedModels covers the pipelines that own no weights at all.
//
// Run:  go test ./internal/registry -run TestGate -v
// The -v transcript is the figure/listing reproduced in the paper (docs/threat-model.md).
func TestGate_AdversarialDemo(t *testing.T) {
	// A valid permissive manifest helper (passes structural validate()).
	newPermissive := func(dir string) *Manifest {
		m := &Manifest{
			Name:      "demo-model",
			License:   "Apache-2.0",
			Task:      "classification",
			ModelFile: "model.onnx",
			// An audited upstream that records no weight digests, so the manifest's own pin is
			// the only content check — the pre-anchor regime, still supported for upstreams the
			// maintainer has not yet enumerated. Case (G) covers an upstream that DOES record
			// digests, where a self-consistent manifest pin is no longer enough.
			SourceURL: "https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX/",
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
		defer setAllowlist([]string{"https://huggingface.co/onnx-community/"})()

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
		m.SHA256 = SHA256Field{single: digest}                   // hash is fine...
		m.SourceURL = "https://random-mirror.example.com/x.onnx" // ...but origin is not audited

		defer setAllowlist([]string{"https://huggingface.co/onnx-community/"})()

		err := m.VerifyWeights()
		if err == nil || !strings.Contains(err.Error(), "not under any audited prefix") {
			t.Fatalf("unaudited source must be refused, got: %v", err)
		}
		t.Logf("REFUSED (source): %v", err)
	})

	// (D) RELICENSE-vs-LEDGER — declared license is permissive AND on the allowlist, but it does
	// NOT match the maintainer-audited upstream license recorded in the provenance ledger. The
	// allowlist alone CANNOT catch this (both are permissive); only the ledger cross-check does.
	// This is the control that closes the "trust the string the PR author typed" hole. -> REFUSED
	t.Run("D_ledger_license_mismatch_refused", func(t *testing.T) {
		dir := t.TempDir()
		_, digest := writeWeight(t, dir, "model.onnx", []byte("authentic permissive weights"))
		m := newPermissive(dir)
		m.SHA256 = SHA256Field{single: digest}
		// onnxmodelzoo/mobilenet... is audited as Apache-2.0 in the ledger; declare MIT instead.
		m.License = "MIT" // still on the permissive allowlist — validate() passes!
		m.SourceURL = "https://huggingface.co/onnxmodelzoo/mobilenet_v3_small_Opset17/resolve/main/x.onnx"

		EnableVerifiedMode()
		defer DisableVerifiedMode()

		if err := m.validate(); err != nil {
			t.Fatalf("MIT is permissive, validate() must pass (allowlist cannot catch this): %v", err)
		}
		err := m.VerifyWeights()
		if err == nil || !strings.Contains(err.Error(), "does not match the maintainer-audited") {
			t.Fatalf("license mismatching the audited ledger must be refused, got: %v", err)
		}
		t.Logf("REFUSED (ledger): %v", err)
	})

	// (E) UNLEDGERED-SOURCE — a source with no maintainer-audited ledger entry is refused in
	// verified mode (we only serve models whose upstream a human has audited). -> REFUSED
	t.Run("E_unledgered_source_refused", func(t *testing.T) {
		dir := t.TempDir()
		_, digest := writeWeight(t, dir, "model.onnx", []byte("weights from an un-audited upstream"))
		m := newPermissive(dir)
		m.SHA256 = SHA256Field{single: digest}
		m.SourceURL = "https://huggingface.co/some-random-user/unaudited-model/resolve/main/x.onnx"

		EnableVerifiedMode()
		defer DisableVerifiedMode()

		err := m.VerifyWeights()
		if err == nil || !strings.Contains(err.Error(), "no maintainer-audited license-ledger entry") {
			t.Fatalf("un-audited upstream must be refused in verified mode, got: %v", err)
		}
		t.Logf("REFUSED (no ledger entry): %v", err)
	})

	// (F) NO CONTENT PIN — the license is permissive, the upstream is audited and the declared
	// license matches the ledger, but the manifest declares NO sha256 at all. Cases (0)-(E) every
	// one of them set m.SHA256, so none of them exercises this: the hash check is only reached by
	// a manifest that opted into it. Without a pin, "audited origin" says where the bytes were
	// SUPPOSED to come from and nothing about the bytes actually on disk — which is precisely the
	// binding verified mode exists to provide. -> REFUSED
	t.Run("F_missing_content_pin_refused", func(t *testing.T) {
		dir := t.TempDir()
		writeWeight(t, dir, "model.onnx", []byte("could be anything at all"))
		m := newPermissive(dir) // audited source_url, Apache-2.0, matches the ledger...
		// ...and deliberately NO m.SHA256.

		EnableVerifiedMode()
		defer DisableVerifiedMode()

		if err := m.validate(); err != nil {
			t.Fatalf("sha256 is optional by design, validate() must still pass: %v", err)
		}
		err := m.VerifyWeights()
		if err == nil || !strings.Contains(err.Error(), "no sha256 content pin") {
			t.Fatalf("verified mode must refuse an unpinned model, got: %v", err)
		}
		t.Logf("REFUSED (no content pin): %v", err)
	})

	// (G) FORGED PIN — the sharpest case, and the one cases (0)-(F) all miss. Every one of them
	// lets the manifest choose its own digest, so all any of them proves is that the manifest is
	// SELF-CONSISTENT. An attacker who can write the model directory writes the manifest too: put
	// arbitrary weights in place, hash them, declare that hash, and copy an audited source_url and
	// its licence. Nothing above objects.
	//
	// The fix is that the digest must also appear in the maintainer's in-binary record for that
	// upstream (ledger.go, WeightSHA256), which is not in the directory being attacked. -> REFUSED
	t.Run("G_forged_pin_refused", func(t *testing.T) {
		dir := t.TempDir()
		_, digest := writeWeight(t, dir, "model.onnx", []byte("substituted weights, honestly hashed"))
		m := newPermissive(dir)
		m.SHA256 = SHA256Field{single: digest} // self-consistent: the bytes DO hash to this
		// An upstream the maintainer has enumerated digests for.
		m.SourceURL = "https://huggingface.co/mtbui2010/grounding-dino-tiny-fixedmask-ONNX/"

		EnableVerifiedMode()
		defer DisableVerifiedMode()

		err := m.VerifyWeights()
		if err == nil || !strings.Contains(err.Error(), "not among the digests the maintainer audited") {
			t.Fatalf("a self-consistent but unaudited pin must be refused, got: %v", err)
		}
		t.Logf("REFUSED (unaudited pin): %v", err)
	})
}

// TestGate_ComposedModels covers the pipelines VisionServe actually ships — grounded-sam,
// rfdetr-gdino, grasp-gd — which own no weights at all and only wire together sessions their
// dependencies own. They have no source_url and nothing to pin, so before this they were refused
// outright: turning verified mode on disabled the project's flagship features. A composition is
// now as audited as what it composes, with one guard, exercised below: it may only reference
// weights the owning model actually declares.
func TestGate_ComposedModels(t *testing.T) {
	// A dependency that stands on its own: audited upstream, pinned to its own bytes. Its upstream
	// records no digests, so the manifest pin is the content check (see case (0)).
	newDep := func(t *testing.T, root, name string) (dir, weight, digest string) {
		t.Helper()
		dir = filepath.Join(root, name)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		weight, digest = writeWeight(t, dir, "model.onnx", []byte("weights of "+name))
		yaml := "name: " + name + "\ntask: classification\nlicense: Apache-2.0\n" +
			"source_url: https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX/\n" +
			"sha256:\n  model: " + digest + "\nfiles:\n  model: model.onnx\n" +
			"input:\n  width: 224\n  height: 224\n  layout: NCHW\nruntime:\n  prefer: [cpu]\n"
		if err := os.WriteFile(filepath.Join(dir, "manifest.yaml"), []byte(yaml), 0o644); err != nil {
			t.Fatal(err)
		}
		return dir, weight, digest
	}

	composed := func(t *testing.T, root string, files map[string]string) *Manifest {
		t.Helper()
		dir := filepath.Join(root, "composed")
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		m := &Manifest{Name: "composed-model", License: "Apache-2.0", Task: "open_vocab",
			Files: files, dir: dir}
		m.Input.Width, m.Input.Height = 224, 224
		m.Input.Layout = "NCHW"
		m.Runtime.Prefer = []string{"cpu"}
		return m
	}

	t.Run("admitted_via_its_dependencies", func(t *testing.T) {
		root := t.TempDir()
		newDep(t, root, "dep-a")
		newDep(t, root, "dep-b")
		m := composed(t, root, map[string]string{
			"a": "../dep-a/model.onnx",
			"b": "../dep-b/model.onnx",
		})

		EnableVerifiedMode()
		defer DisableVerifiedMode()

		if err := m.VerifyWeights(); err != nil {
			t.Fatalf("a composition of admitted models must be admitted, got: %v", err)
		}
		t.Logf("ADMITTED: every weight is owned and declared by an admitted model")
	})

	t.Run("undeclared_file_refused", func(t *testing.T) {
		root := t.TempDir()
		depDir, _, _ := newDep(t, root, "dep-a")
		// A file sitting INSIDE an admitted model's directory that its manifest never declares.
		// Without the declaresFile guard, a composed manifest could smuggle in anything this way.
		if err := os.WriteFile(filepath.Join(depDir, "stray.onnx"), []byte("unvetted"), 0o644); err != nil {
			t.Fatal(err)
		}
		m := composed(t, root, map[string]string{"a": "../dep-a/stray.onnx"})

		EnableVerifiedMode()
		defer DisableVerifiedMode()

		err := m.VerifyWeights()
		if err == nil || !strings.Contains(err.Error(), "does not declare") {
			t.Fatalf("referencing an undeclared file must be refused, got: %v", err)
		}
		t.Logf("REFUSED (undeclared): %v", err)
	})

	t.Run("refused_dependency_propagates", func(t *testing.T) {
		root := t.TempDir()
		depDir, weight, _ := newDep(t, root, "dep-a")
		_ = depDir
		// Tamper with the dependency's bytes AFTER its manifest pinned them.
		if err := os.WriteFile(weight, []byte("tampered after pinning"), 0o644); err != nil {
			t.Fatal(err)
		}
		m := composed(t, root, map[string]string{"a": "../dep-a/model.onnx"})

		EnableVerifiedMode()
		defer DisableVerifiedMode()

		err := m.VerifyWeights()
		if err == nil || !strings.Contains(err.Error(), "is not admitted") {
			t.Fatalf("a refused dependency must refuse the composition, got: %v", err)
		}
		t.Logf("REFUSED (dependency): %v", err)
	})
}

// setAllowlist sets the package-global VerifiedSourcePrefixes for a test and returns a
// restore func (call via defer) so other tests see the default empty allowlist.
func setAllowlist(prefixes []string) func() {
	prev := VerifiedSourcePrefixes
	VerifiedSourcePrefixes = prefixes
	return func() { VerifiedSourcePrefixes = prev }
}
