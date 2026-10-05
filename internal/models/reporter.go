package models

import "visionserve/internal/vision/preprocess"

// PreprocessReporter is an OPTIONAL interface of a plain Model whose preprocessing is one
// preprocess.Spec applied to the image. ResolvedPreprocess returns the spec its Preprocess really
// applies: the manifest's, after the architecture's Arch.Resolve (defaults filled in, legacy
// fields read the architecture's way — SCRFD's legacy `letterbox` is a top-left pad in 0..255
// units, for instance). Tools that describe a model (`visionserve inspect`) show this instead of
// re-deriving it from the manifest; serving never calls it. A model without it is described by
// its manifest alone.
type PreprocessReporter interface {
	ResolvedPreprocess() (preprocess.Spec, error)
}
