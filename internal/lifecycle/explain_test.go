package lifecycle

import (
	"context"
	"image"
	"strings"
	"testing"
)

// Explain must describe what is LOADED. It used to re-read the registry, so a manifest edited on
// disk after the load (and picked up by a rescan) made explain build sessions and read outputs the
// loaded model was never configured for. Here the edit adds an explain block the loaded model
// does not have: the answer must still be "does not support explain".
func TestExplainUsesLoadTimeManifest(t *testing.T) {
	root := t.TempDir()
	writeTestModel(t, root, "snap", "test-pipe", "")
	reg := scanRegistry(t, root)
	m, _ := newFakeManager(t, reg)
	if err := m.Load(context.Background(), "snap"); err != nil {
		t.Fatal(err)
	}

	writeTestModel(t, root, "snap", "test-pipe",
		"explain:\n  type: attention\n  role: x\n  outputs:\n    attention: attn\n")
	if _, err := reg.Scan(); err != nil {
		t.Fatal(err)
	}
	if e, _ := reg.Get("snap"); e == nil || e.Manifest.Explain == nil {
		t.Fatal("fixture: the rescan did not pick up the edited manifest")
	}

	_, err := m.Explain(context.Background(), "snap", image.NewRGBA(image.Rect(0, 0, 4, 4)), ExplainRequest{})
	if err == nil || !strings.Contains(err.Error(), "does not support explain") {
		t.Fatalf("Explain = %v, want the load-time answer (no explain block)", err)
	}
}
