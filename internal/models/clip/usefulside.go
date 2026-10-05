package clip

import (
	"visionserve/internal/models"
	prep "visionserve/internal/vision/preprocess"
)

// The image tower reads only its fixed center_crop (or squash) input (see models.UsefulSide).
func init() {
	models.RegisterUsefulSide("clip", func(s prep.Spec) (models.UsefulSide, error) {
		r, err := spec(models.Config{Preprocess: &s})
		if err != nil {
			return models.UsefulSide{}, err
		}
		return models.FixedTargetUsefulSide(r), nil
	})
}
