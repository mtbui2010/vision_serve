package detr

import (
	"visionserve/internal/models"
	"visionserve/internal/vision/preprocess"
)

var _ models.PreprocessReporter = (*detr)(nil)

// ResolvedPreprocess is the spec preprocess applies (models.PreprocessReporter).
func (m *detr) ResolvedPreprocess() (preprocess.Spec, error) { return m.spec() }
