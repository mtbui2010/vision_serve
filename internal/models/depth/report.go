package depth

import (
	"visionserve/internal/models"
	prep "visionserve/internal/vision/preprocess"
)

var _ models.PreprocessReporter = (*depthModel)(nil)

// ResolvedPreprocess is the spec preprocess applies (models.PreprocessReporter).
func (m *depthModel) ResolvedPreprocess() (prep.Spec, error) {
	return arch.Resolve(m.cfg.PreprocessSpec())
}
