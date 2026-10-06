package server

import (
	"visionserve/internal/lifecycle"
	"visionserve/internal/registry"
	"visionserve/pkg/api"
)

// withClientHints fills info's client hints from the manifest (GET /api/models): the
// client-resize hint (max_useful_side / max_useful_short_side: lifecycle.UsefulSide, at most one
// side set, both null when clients must send the full photo) and accepts_depth
// (lifecycle.AcceptsDepth).
func withClientHints(info api.ModelInfo, man *registry.Manifest) api.ModelInfo {
	info.AcceptsDepth = lifecycle.AcceptsDepth(man)
	switch u := lifecycle.UsefulSide(man); {
	case u.Long > 0:
		info.MaxUsefulSide = &u.Long
	case u.Short > 0:
		info.MaxUsefulShortSide = &u.Short
	}
	return info
}
