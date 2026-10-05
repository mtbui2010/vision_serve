package depth

import "visionserve/internal/models"

// The depth map comes back at the MODEL's resolution, so a squash model (midas) reads nothing
// past its input (see models.UsefulSide). keep_aspect (depth-anything-v2) sizes the tensor from
// the image's own aspect ratio and gets no hint (models.FixedTargetUsefulSide).
func init() {
	models.RegisterUsefulSide("midas", models.ResolvedUsefulSide(arch))
	models.RegisterUsefulSide("depth-anything-v2", models.ResolvedUsefulSide(arch))
}
