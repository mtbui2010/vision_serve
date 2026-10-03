package hybrid

import (
	"image"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"visionserve/internal/models"
)

// gdinoSigLIPConfig points at the repo's own tokenizer assets; only vocab.txt and tokenizer.json
// are read at construction, never the ONNX graphs.
func gdinoSigLIPConfig(t *testing.T) models.Config {
	t.Helper()
	gdVocab := filepath.Join("..", "..", "..", "models", "grounding-dino", "vocab.txt")
	if _, err := os.Stat(gdVocab); err != nil {
		t.Skipf("no GroundingDINO vocab at %s", gdVocab)
	}
	if _, err := os.Stat(filepath.Join(siglipDir(), "tokenizer.json")); err != nil {
		t.Skipf("no SigLIP tokenizer in %s", siglipDir())
	}
	return models.Config{
		Name: "gdino-siglip",
		Dir:  t.TempDir(),
		Files: map[string]string{
			roleGDINO: filepath.Join(filepath.Dir(gdVocab), "model-fixedmask.onnx"),
			roleCrop:  filepath.Join(siglipDir(), "..", "siglip-image", "model.onnx"),
			roleText:  filepath.Join(siglipDir(), "model.onnx"),
		},
		Labels: []string{"cup", "towel"}, // must be IGNORED: nothing is routed away from GroundingDINO
	}
}

func TestGDINOSigLIPRolesAndRouting(t *testing.T) {
	b, err := NewGDINOSigLIP(gdinoSigLIPConfig(t))
	if err != nil {
		t.Fatalf("NewGDINOSigLIP: %v", err)
	}
	m := b.(*hybrid)
	if got := strings.Join(m.Roles(), ","); got != "gdino,crop,text" {
		t.Fatalf("Roles() = %s, want gdino,crop,text (no rfdetr session may be requested)", got)
	}
	known, unknown := m.rt.Partition([]string{"cup", "towel", "zebra"})
	if len(known) != 0 || len(unknown) != 3 {
		t.Fatalf("partition = %v / %v, want every word routed to GroundingDINO", known, unknown)
	}
	if _, err := m.Infer(image.NewRGBA(image.Rect(0, 0, 8, 8)), models.Prompt{}, nil); err == nil ||
		!strings.Contains(err.Error(), "text prompt is required") {
		t.Fatalf("empty prompt: got %v, want a clear 'prompt required' error", err)
	}
}

func TestGDINOSigLIPWithSAMRoles(t *testing.T) {
	cfg := gdinoSigLIPConfig(t)
	cfg.Files[roleEncoder], cfg.Files[roleDecoder] = "enc.onnx", "dec.onnx"
	b, err := NewGDINOSigLIP(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(b.(*hybrid).Roles(), ","); got != "gdino,encoder,decoder,crop,text" {
		t.Fatalf("Roles() = %s", got)
	}
}

// Each misconfiguration must fail at load, not degrade into a different model silently.
func TestGDINOSigLIPRejectsMisconfiguration(t *testing.T) {
	for name, mutate := range map[string]func(*models.Config){
		"no gdino":     func(c *models.Config) { delete(c.Files, roleGDINO) },
		"no crop":      func(c *models.Config) { delete(c.Files, roleCrop) },
		"no text":      func(c *models.Config) { delete(c.Files, roleText) },
		"has rfdetr":   func(c *models.Config) { c.Files[roleRFDETR] = "rf.onnx" },
		"onnx head":    func(c *models.Config) { c.Files[roleHead] = "head.onnx" },
		"head.bin dir": func(c *models.Config) { _ = os.WriteFile(filepath.Join(c.Dir, headFile), []byte("x"), 0o644) },
	} {
		t.Run(name, func(t *testing.T) {
			cfg := gdinoSigLIPConfig(t)
			mutate(&cfg)
			if _, err := NewGDINOSigLIP(cfg); err == nil {
				t.Fatalf("NewGDINOSigLIP accepted a manifest with %s", name)
			}
		})
	}
}
