package background

import (
	"errors"
	"image"
	"testing"

	"visionserve/internal/models"
)

// TestResolveMethodRefusalsAreBadPrompt: every method the caller can get wrong — an unknown name,
// or one this manifest has no session for — is a models.BadPrompt (HTTP 400), never a 500, and it
// is refused before any work starts.
func TestResolveMethodRefusalsAreBadPrompt(t *testing.T) {
	m := &backgroundModel{} // no depth, no SAM: auto and cv only
	for _, method := range []string{"nonsense", "depth", "sam", "automask", "auto-cv"} {
		if _, err := m.resolveMethod(models.Prompt{Method: method}); !errors.Is(err, models.ErrBadPrompt) {
			t.Errorf("resolveMethod(%q) = %v, want an error matching models.ErrBadPrompt", method, err)
		}
		img := image.NewNRGBA(image.Rect(0, 0, 8, 8))
		if _, err := m.Infer(img, models.Prompt{Method: method}, nil); !errors.Is(err, models.ErrBadPrompt) {
			t.Errorf("Infer(method=%q) = %v, want an error matching models.ErrBadPrompt", method, err)
		}
	}
	for _, method := range []string{"", "auto", "cv", " CV "} {
		if _, err := m.resolveMethod(models.Prompt{Method: method}); err != nil {
			t.Errorf("resolveMethod(%q): %v", method, err)
		}
	}
}
