package lifecycle

import (
	"visionserve/internal/models"
	"visionserve/internal/registry"
)

// UsefulSide is the client-resize hint GET /api/models publishes for a manifest
// (api.ModelInfo.MaxUsefulSide / MaxUsefulShortSide): the manifest's runtime.max_useful_side
// when it sets one (N > 0 bounds the longer side, 0 = never resize), else what the
// architecture derives from the manifest's preprocessing (models.RegisterUsefulSide). An
// architecture that registers nothing — masks, OCR, depth-aligned grasping, crop namers,
// templates, anything not reasoned about — gets the zero value: send full resolution.
//
// It reads only the manifest: no weights, no model is built, so listing stays cheap and a model
// that is not downloaded yet already shows its hint.
func UsefulSide(man *registry.Manifest) models.UsefulSide {
	if man == nil {
		return models.UsefulSide{}
	}
	if n, ok := man.UsefulSideOverride(); ok {
		return models.UsefulSide{Long: n}
	}
	spec, err := man.PreprocessSpec()
	if err != nil {
		return models.UsefulSide{}
	}
	return models.UsefulSideOf(man.ArchOrName(), spec)
}
