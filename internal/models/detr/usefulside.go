package detr

import "visionserve/internal/models"

// Both DETRs read only their fixed-size input (squash or letterbox), so a client may shrink an
// image to models.UsefulSide of it (GET /api/models: max_useful_side / max_useful_short_side).
func init() {
	models.RegisterUsefulSide("rf-detr", models.ResolvedUsefulSide(arch(rfVariant.prefix)))
	models.RegisterUsefulSide("rt-detr", models.ResolvedUsefulSide(arch(rtVariant.prefix)))
}
