package scrfd

import (
	"visionserve/internal/models"
	prep "visionserve/internal/vision/preprocess"
)

// SCRFD fits the image inside its fixed input (top_left_pad), so the longer side bounds what it
// can read (see models.UsefulSide).
func init() {
	models.RegisterUsefulSide("scrfd", func(s prep.Spec) (models.UsefulSide, error) {
		r, err := spec(models.Config{Preprocess: &s})
		if err != nil {
			return models.UsefulSide{}, err
		}
		return models.FixedTargetUsefulSide(r), nil
	})
}
