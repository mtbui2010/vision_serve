package scrfd

import (
	"visionserve/internal/models"
	prep "visionserve/internal/vision/preprocess"
)

var _ models.PreprocessReporter = (*scrfdModel)(nil)

// ResolvedPreprocess is the spec preprocess applies (models.PreprocessReporter).
func (m *scrfdModel) ResolvedPreprocess() (prep.Spec, error) { return spec(m.cfg) }
