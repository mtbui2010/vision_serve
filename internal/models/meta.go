package models

import "visionserve/internal/vision/preprocess"

// PreprocessMeta holds the info needed to map results back to original image coordinates:
// input = orig * Scale + Pad on each axis (Affine() gives the geom mapping). With letterbox,
// the scale on both axes is equal (aspect ratio preserved). It is vision/preprocess's Meta, so
// preprocess.Spec.Apply returns it directly.
type PreprocessMeta = preprocess.Meta

// PreprocessSpec returns the preprocessing the manifest declares: Config.Preprocess when the
// registry resolved it, else the legacy fields of a Config built in code mapped the same way
// (preprocess.FromLegacy). A model passes it through its preprocess.Arch.Resolve.
func (c Config) PreprocessSpec() preprocess.Spec {
	if c.Preprocess != nil {
		return *c.Preprocess
	}
	return preprocess.FromLegacy(preprocess.LegacyFields{
		Width: c.Width, Height: c.Height, Layout: c.Layout,
		Letterbox: c.Letterbox, Crop: c.Crop, KeepAspect: c.KeepAspect, MultipleOf: c.MultipleOf,
		Mean: c.Mean, Std: c.Std,
	})
}
