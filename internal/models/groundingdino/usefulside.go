package groundingdino

import (
	"visionserve/internal/models"
	prep "visionserve/internal/vision/preprocess"
)

// GroundingDINO squashes every image to exactly inputSize² (preprocessImage) whatever the
// manifest says, and reads nothing else from it, so a client may shrink an image's shorter side
// to models.UsefulSideFactor × inputSize (see models.UsefulSide).
func init() {
	models.RegisterUsefulSide("grounding-dino", func(prep.Spec) (models.UsefulSide, error) {
		return models.FixedTargetUsefulSide(prep.Spec{Resize: prep.Squash, Width: inputSize, Height: inputSize}), nil
	})
}
