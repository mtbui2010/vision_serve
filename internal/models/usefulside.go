package models

import (
	"sync"

	"visionserve/internal/vision/preprocess"
)

// UsefulSideFactor is the safety margin of the client-side resize hint: a client may shrink an
// image until the model's own resize still DOWN-scales it by at least this factor on every axis,
// so the model keeps resampling from more pixels than it keeps. 2 leaves the antialiased resize
// a full octave of real detail; measured on RF-DETR it costs no mAP (docs/clients, "Client-side
// resizing").
const UsefulSideFactor = 2

// UsefulSide says how far a client may shrink an image before uploading it without changing what
// the model can see (GET /api/models: max_useful_side / max_useful_short_side). At most one of
// the two is set; the zero value means "send the image at full resolution".
//
// Which side bounds the useful size depends on the preprocessing:
//
//   - Long: the model fits the image INSIDE its input (letterbox, top_left_pad, long_side,
//     long_side_pad). Its scale is min(W/w, H/h), so it is set by the image's LONGER side: a
//     longer side of UsefulSideFactor × max(W, H) still leaves every axis down-scaled by
//     at least the factor.
//   - Short: the model fills its input on BOTH axes (squash, center_crop). The shorter side is
//     the one the model resamples least, so it is the one bounded: a shorter side of
//     UsefulSideFactor × max(W, H) keeps every axis of any aspect ratio down-scaled by at least the
//     factor. (Bounding the longer side instead would squash a wide panorama's few rows into
//     fewer still before the model stretched them back up.)
type UsefulSide struct {
	Long  int // max useful longer side, pixels; 0 = not this rule
	Short int // max useful shorter side, pixels; 0 = not this rule
}

// IsZero reports whether the hint is "never resize".
func (u UsefulSide) IsZero() bool { return u.Long <= 0 && u.Short <= 0 }

// FixedTargetUsefulSide is the hint for a RESOLVED preprocess.Spec (Arch.Resolve applied): the
// fixed-size modes get UsefulSideFactor × the larger target side, on the side the mode is bound by
// (see UsefulSide); keep_aspect (the tensor follows the image's aspect ratio, so shrinking
// changes it), none (the graph takes the image as it is) and an unknown or sizeless spec get the
// zero value.
func FixedTargetUsefulSide(s preprocess.Spec) UsefulSide {
	if s.Width <= 0 || s.Height <= 0 {
		return UsefulSide{}
	}
	side := UsefulSideFactor * max(s.Width, s.Height)
	switch s.Resize {
	case preprocess.Letterbox, preprocess.TopLeftPad, preprocess.LongSide, preprocess.LongSidePad:
		return UsefulSide{Long: side}
	case preprocess.Squash, preprocess.CenterCrop:
		return UsefulSide{Short: side}
	}
	return UsefulSide{}
}

// UsefulSideFunc returns an architecture's client-resize hint for the manifest's preprocessing
// (the registry's spec, NOT yet resolved by the architecture: the function resolves it the way
// the architecture's Preprocess does). It must be cheap and pure — GET /api/models calls it
// for every listed model without building the model.
type UsefulSideFunc func(spec preprocess.Spec) (UsefulSide, error)

var (
	usefulMu   sync.RWMutex
	usefulSide = map[string]UsefulSideFunc{}
)

// RegisterUsefulSide declares that architecture arch can take an image a client has shrunk
// (see UsefulSide). It is OPTIONAL and opt-in: an architecture that does not register is sent
// every image at full resolution, which is always correct. Register only when the result really
// cannot depend on pixels past the hint — never for models that return full-resolution output
// (masks, OCR text lines), align the image with another input (depth), crop the ORIGINAL image
// (crop namers) or compare it with templates. Call it in the model package's init(); a
// duplicate panics like Register.
func RegisterUsefulSide(arch string, f UsefulSideFunc) {
	usefulMu.Lock()
	defer usefulMu.Unlock()
	if _, dup := usefulSide[arch]; dup {
		panic("models: useful side of " + arch + " already registered")
	}
	usefulSide[arch] = f
}

// UsefulSideOf returns architecture arch's hint for spec; the zero value when the architecture
// did not register one or its function fails (a manifest the architecture refuses also fails to
// load, and an unloadable model gains nothing from a hint).
func UsefulSideOf(arch string, spec preprocess.Spec) UsefulSide {
	usefulMu.RLock()
	f, ok := usefulSide[arch]
	usefulMu.RUnlock()
	if !ok {
		return UsefulSide{}
	}
	u, err := f(spec)
	if err != nil || u.Long < 0 || u.Short < 0 {
		return UsefulSide{}
	}
	return u
}

// ResolvedUsefulSide is the UsefulSideFunc of an architecture whose preprocessing is spec run
// through its preprocess.Arch.Resolve (the plain models: detr, classification, clip, depth,
// scrfd).
func ResolvedUsefulSide(a preprocess.Arch) UsefulSideFunc {
	return func(spec preprocess.Spec) (UsefulSide, error) {
		r, err := a.Resolve(spec)
		if err != nil {
			return UsefulSide{}, err
		}
		return FixedTargetUsefulSide(r), nil
	}
}
