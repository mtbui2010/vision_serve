package models

import "visionserve/internal/vision/geom"

// Affine returns the input = orig*Scale + Pad mapping recorded by Preprocess, for mapping
// boxes back to ORIGINAL image coordinates with geom.Affine.BoxToOrig.
func (m PreprocessMeta) Affine() geom.Affine {
	return geom.Affine{ScaleX: m.ScaleX, ScaleY: m.ScaleY, PadX: float64(m.PadX), PadY: float64(m.PadY)}
}
