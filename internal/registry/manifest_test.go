package registry

import (
	"strings"
	"testing"
)

// An AGPL license MUST be rejected (top principle #1 — CLAUDE.md).
func TestValidateRejectsAGPL(t *testing.T) {
	m := &Manifest{Name: "yolo", License: "AGPL-3.0", Task: "detection", ModelFile: "x.onnx"}
	m.Input.Width, m.Input.Height = 640, 640
	err := m.validate()
	if err == nil || !strings.Contains(err.Error(), "license") {
		t.Fatalf("expected a license error for AGPL, got: %v", err)
	}
}

// SPDX ids are case-insensitive by spec, and HuggingFace model cards spell them lowercase
// ("license: apache-2.0"), so a manifest copied from an HF card must load. Accepting any casing
// must NOT widen WHICH licenses pass — AGPL stays refused in every casing — and an accepted
// license must be normalized to its canonical SPDX spelling (see
// TestValidateNormalizedLicenseMatchesLedger for why the normalization is load-bearing).
func TestValidateLicenseCaseInsensitive(t *testing.T) {
	cases := []struct {
		name      string
		declared  string
		wantOK    bool
		wantCanon string // expected m.License after validate() (only when wantOK)
	}{
		// Canonical SPDX spellings — the baseline, must keep working.
		{"canonical apache", "Apache-2.0", true, "Apache-2.0"},
		{"canonical mit", "MIT", true, "MIT"},
		{"canonical bsd3", "BSD-3-Clause", true, "BSD-3-Clause"},
		{"canonical bsd2", "BSD-2-Clause", true, "BSD-2-Clause"},
		// Lowercase — exactly what an HF model card frontmatter carries.
		{"hf lowercase apache", "apache-2.0", true, "Apache-2.0"},
		{"hf lowercase mit", "mit", true, "MIT"},
		{"hf lowercase bsd3", "bsd-3-clause", true, "BSD-3-Clause"},
		// Uppercase / mixed case / stray whitespace.
		{"uppercase apache", "APACHE-2.0", true, "Apache-2.0"},
		{"uppercase bsd2", "BSD-2-CLAUSE", true, "BSD-2-Clause"},
		{"mixed case apache", "ApAcHe-2.0", true, "Apache-2.0"},
		{"mixed case mit", "Mit", true, "MIT"},
		{"padded apache", "  apache-2.0\t", true, "Apache-2.0"},
		// AGPL — strictly forbidden (CLAUDE.md principle #1) in EVERY casing/variant.
		{"agpl canonical", "AGPL-3.0", false, ""},
		{"agpl lowercase", "agpl-3.0", false, ""},
		{"agpl mixed case", "AgPl-3.0", false, ""},
		{"agpl only variant", "AGPL-3.0-only", false, ""},
		{"agpl only lowercase", "agpl-3.0-only", false, ""},
		{"agpl or-later", "AGPL-3.0-or-later", false, ""},
		// Other non-permissive / unknown ids stay refused.
		{"gpl", "GPL-3.0", false, ""},
		{"cc non-commercial", "cc-by-nc-4.0", false, ""},
		{"unknown", "totally-made-up", false, ""},
		{"empty", "", false, ""},
		// Near-miss on a permissive id must NOT be accepted (no fuzzy matching).
		{"apache without version", "apache", false, ""},
		{"apache spaced", "Apache 2.0", false, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := &Manifest{Name: "m", License: tc.declared, Task: "detection", ModelFile: "x.onnx"}
			m.Input.Width, m.Input.Height = 640, 640
			err := m.validate()
			if !tc.wantOK {
				if err == nil || !strings.Contains(err.Error(), "license") {
					t.Fatalf("license %q must be refused, got err=%v", tc.declared, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("license %q must be accepted, got: %v", tc.declared, err)
			}
			if m.License != tc.wantCanon {
				t.Fatalf("license %q must normalize to %q, got %q", tc.declared, tc.wantCanon, m.License)
			}
		})
	}
}

// The normalization is not cosmetic: VerifyLicenseProvenance compares the manifest license
// against the maintainer-audited ledger with ==. If validate() accepted "apache-2.0" but stored
// it verbatim, verified mode would refuse the very same model for a bogus license MISMATCH —
// the bug moved, not fixed. Guard that exact interaction.
func TestValidateNormalizedLicenseMatchesLedger(t *testing.T) {
	// mobilenet_v3_small is audited as "Apache-2.0" in LicenseLedger.
	const src = "https://huggingface.co/onnxmodelzoo/mobilenet_v3_small_Opset17/resolve/main/x.onnx"
	m := &Manifest{Name: "hf-style", License: "apache-2.0", Task: "classification", ModelFile: "x.onnx", SourceURL: src}
	m.Input.Width, m.Input.Height = 224, 224
	if err := m.validate(); err != nil {
		t.Fatalf("an HF-style lowercase license must pass validate(): %v", err)
	}

	EnableVerifiedMode()
	defer DisableVerifiedMode()

	if err := m.VerifyLicenseProvenance(); err != nil {
		t.Fatalf("normalized license must match the audited ledger entry, got: %v", err)
	}
}

// An invalid task -> rejected.
func TestValidateRejectsBadTask(t *testing.T) {
	m := &Manifest{Name: "x", License: "Apache-2.0", Task: "not_a_real_task", ModelFile: "x.onnx"}
	m.Input.Width, m.Input.Height = 1, 1
	if err := m.validate(); err == nil || !strings.Contains(err.Error(), "task") {
		t.Fatalf("expected an invalid-task error, got: %v", err)
	}
}

// A structurally valid manifest (permissive license) MUST pass validate even without weights —
// the model is still listed, just not ready to load.
func TestValidatePassesWithoutWeights(t *testing.T) {
	m := &Manifest{Name: "ok", License: "Apache-2.0", Task: "detection", ModelFile: "nope.onnx"}
	m.Input.Width, m.Input.Height = 100, 100
	if err := m.validate(); err != nil {
		t.Fatalf("structural validate must pass when license/task/dims are valid, got: %v", err)
	}
	if m.WeightsExist() {
		t.Fatal("WeightsExist must be false for a nonexistent file")
	}
}
