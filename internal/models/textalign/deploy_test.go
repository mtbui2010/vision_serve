package textalign

import (
	"os"
	"path/filepath"
	"testing"

	"visionserve/internal/registry"
)

// modelDir is the shipped model directory these tests validate. They are deliberately about
// the DEPLOYED artifacts (proj.bin, templates.txt, manifest.yaml) rather than the algebra —
// every failure mode below was hit for real while landing the trained head, and each one is
// silent at load time and only shows up as "the server returns nothing" or "/api/explain
// 500s" in production.
func modelDir(t *testing.T) string {
	t.Helper()
	d := filepath.Join("..", "..", "..", "models", "rfdetr-textalign-etri")
	if _, err := os.Stat(filepath.Join(d, "manifest.yaml")); err != nil {
		t.Skip("models/rfdetr-textalign-etri not present (weights are not committed)")
	}
	return d
}

// TestShippedTemplates: T̂ must be built from the SAME prompt templates the head was fitted
// against, so templates.txt is part of the P contract, not a nicety. A template without the
// "{}" placeholder would silently embed a constant string for every class.
func TestShippedTemplates(t *testing.T) {
	dir := modelDir(t)
	tmpl, err := loadTemplates(filepath.Join(dir, templatesFile))
	if err != nil {
		t.Fatalf("loadTemplates: %v", err)
	}
	if len(tmpl) == 0 {
		t.Fatal("no templates")
	}
	for _, s := range tmpl {
		if !contains(s, templatePlaceholder) {
			t.Errorf("template %q has no %s placeholder", s, templatePlaceholder)
		}
	}
	t.Logf("%d prompt templates, first %q", len(tmpl), tmpl[0])
}

// TestShippedProjectionIsTrained guards the exact mistake this deployment is one file copy
// away from: shipping the RANDOM matrix that make_dummy_proj.py writes. Its header is
// recognisable — the note's INITIAL (a, b) = (1/0.07, −4.6), never a trained pair.
func TestShippedProjectionIsTrained(t *testing.T) {
	dir := modelDir(t)
	pr, err := LoadProjection(filepath.Join(dir, projFile))
	if err != nil {
		t.Fatalf("LoadProjection: %v", err)
	}
	const dummyScale, dummyBias = float32(14.285714), float32(-4.6)
	if pr.Scale == dummyScale && pr.Bias == dummyBias {
		t.Errorf("proj.bin carries the dummy header a=%v b=%v — a TRAINED P must ship its own "+
			"calibration (headb_deploy/install_proj.py copies a and b out of headB.npz)",
			pr.Scale, pr.Bias)
	}
	if !pr.UnitNorm() {
		t.Errorf("flags bit0 is clear: this P was not trained with the (‖Pf‖−1)² penalty, so " +
			"method=folded is not calibrated (see headb_deploy/README.md §2.3)")
	}
}

// TestShippedManifestRoles: the manifest declares an `explain` role that is deliberately NOT
// in Roles() — it exists so /api/explain can lazily open the UN-stripped export while the
// detect path uses the stripped one. lifecycle.Manager resolves explain.role against the
// files map only on the first /api/explain call, so a typo there is invisible until a user
// hits it. Check the wiring statically, plus that every declared file exists.
func TestShippedManifestRoles(t *testing.T) {
	dir := modelDir(t)
	man, err := registry.LoadManifest(filepath.Join(dir, "manifest.yaml"))
	if err != nil {
		t.Fatalf("LoadManifest: %v", err)
	}
	files := man.FilesAbs()
	for _, role := range (&textAlign{}).Roles() {
		if _, ok := files[role]; !ok {
			t.Errorf("files map has no role %q (Roles() requires it)", role)
		}
	}
	if man.Explain == nil {
		t.Fatal("no explain block: /api/explain would be unavailable")
	}
	if man.Explain.Role == "" {
		t.Fatal("explain.role is empty; a PipelineModel needs it")
	}
	if _, ok := files[man.Explain.Role]; !ok {
		t.Errorf("explain.role %q is not a key of files: %v", man.Explain.Role, keys(files))
	}
	if !man.WeightsExist() {
		t.Errorf("a file declared in files: does not exist on disk")
	}
	// The detect role must not be the same graph as the explain role — that is the whole
	// point of the 114 MB stripped copy (it saves a 59 MB device→host copy per request).
	if files[roleDetector] == files[man.Explain.Role] {
		t.Errorf("files.%s and files.%s are the same graph; the detect path is paying for "+
			"cross_attn_weights on every request", roleDetector, man.Explain.Role)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func keys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
