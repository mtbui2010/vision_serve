package classification

import (
	"visionserve/internal/models"
	prep "visionserve/internal/vision/preprocess"
)

var _ models.PreprocessReporter = (*classificationModel)(nil)

// ResolvedPreprocess is the spec preprocess applies (models.PreprocessReporter).
func (m *classificationModel) ResolvedPreprocess() (prep.Spec, error) {
	return arch.Resolve(m.cfg.PreprocessSpec())
}
