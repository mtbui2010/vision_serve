package clip

import (
	"visionserve/internal/models"
	prep "visionserve/internal/vision/preprocess"
)

var _ models.PreprocessReporter = (*clipModel)(nil)

// ResolvedPreprocess is the spec preprocess applies (models.PreprocessReporter).
func (m *clipModel) ResolvedPreprocess() (prep.Spec, error) { return spec(m.cfg) }
