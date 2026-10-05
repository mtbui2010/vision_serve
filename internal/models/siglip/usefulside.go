package siglip

import (
	"visionserve/internal/models"
	prep "visionserve/internal/vision/preprocess"
)

// siglip-image embeds the whole image as one crop squashed to ImageSize² (EmbedCrops), so a
// client may shrink an image's shorter side to models.UsefulSideFactor × ImageSize (see
// models.UsefulSide). Pipelines that call CropTensor on detections crop the ORIGINAL image and
// do not inherit this.
func init() {
	models.RegisterUsefulSide("siglip-image", func(prep.Spec) (models.UsefulSide, error) {
		return models.FixedTargetUsefulSide(prep.Spec{Resize: prep.Squash, Width: ImageSize, Height: ImageSize}), nil
	})
}
