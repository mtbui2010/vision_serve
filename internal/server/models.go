package server

import (
	"visionserve/internal/lifecycle"
	"visionserve/internal/registry"
	"visionserve/pkg/api"
)

// withUsefulSide fills info's client-resize hint (GET /api/models: max_useful_side /
// max_useful_short_side) from the manifest: lifecycle.UsefulSide, at most one side set, both
// null when clients must send the full photo.
func withUsefulSide(info api.ModelInfo, man *registry.Manifest) api.ModelInfo {
	switch u := lifecycle.UsefulSide(man); {
	case u.Long > 0:
		info.MaxUsefulSide = &u.Long
	case u.Short > 0:
		info.MaxUsefulShortSide = &u.Short
	}
	return info
}
