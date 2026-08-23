package textalign

import (
	"os"
	"strings"
	"testing"

	"visionserve/internal/models"
)

// capture runs f with stderr redirected and returns what it wrote.
func capture(t *testing.T, f func()) string {
	t.Helper()
	old := os.Stderr
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stderr = w
	f()
	w.Close()
	os.Stderr = old
	buf := make([]byte, 4096)
	n, _ := r.Read(buf)
	return string(buf[:n])
}

// The configuration that shipped with the crop head dead. For a prompt covering the base
// vocabulary the claim score IS the objectness, so everything surviving conf_threshold clears
// claim_threshold and the closed head takes it all. Nothing in the manifest hints at the coupling:
// the two knobs live in different places and neither mentions the other.
func TestWarnDeadOpenHead(t *testing.T) {
	crop := map[string]string{roleCrop: "../siglip-image/model.onnx"}

	cases := []struct {
		name string
		cfg  models.Config
		warn bool
	}{
		{
			"conf above the default claim — the open head can never run",
			models.Config{Name: "dead", Files: crop, ConfThresh: 0.35},
			true,
		},
		{
			// Just above the claim is unambiguously dead. Exact equality is deliberately NOT
			// tested: dualClaimThresh is a rounded logit, so sigmoid of it is 0.1500007 rather
			// than 0.15, and a test pinned to that knife-edge would assert a floating-point
			// artefact rather than a behaviour anyone can configure.
			"just above the claim is dead",
			models.Config{Name: "edge", Files: crop, ConfThresh: 0.16},
			true,
		},
		{
			"conf below the claim leaves room for the closed head to decline",
			models.Config{Name: "ok", Files: crop, ConfThresh: 0.05},
			false,
		},
		{
			// Without a crop head the coupling is irrelevant — head B is named inside the decode
			// and never gated on the claim in this way.
			"no crop head, no warning",
			models.Config{Name: "headb", Files: map[string]string{}, ConfThresh: 0.35},
			false,
		},
		{
			"no threshold set at all",
			models.Config{Name: "nothr", Files: crop, ConfThresh: 0},
			false,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			out := capture(t, func() { warnDeadOpenHead(c.cfg) })
			got := strings.Contains(out, "WARNING")
			if got != c.warn {
				t.Errorf("warned=%v want %v; output was %q", got, c.warn, out)
			}
			if c.warn && !strings.Contains(out, c.cfg.Name) {
				t.Errorf("the warning must name the model, got %q", out)
			}
		})
	}
}
