package classification

import "visionserve/internal/models"

// A classifier reads only its fixed squash input (see models.UsefulSide).
func init() {
	models.RegisterUsefulSide("efficientnet", models.ResolvedUsefulSide(arch))
	models.RegisterUsefulSide("mobilenet-v3", models.ResolvedUsefulSide(arch))
}
