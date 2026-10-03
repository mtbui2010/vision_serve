// Package preprocess is the ONE implementation of image preprocessing: a Spec (data, declared in
// the manifest's `preprocess:` block or mapped from the legacy `input.*` fields) turns an image
// into the tensor a model is fed plus the Meta that maps model-input coordinates back to the
// original image.
//
// Model packages do not hand-write resize / pad / HWC→CHW / normalise loops: they declare a Spec
// (or take the manifest's) and call Apply. The Python converter's reference preprocessing
// (clients/python/visionserve/convert/reference.py) implements the same Spec semantics, so tier B1
// compares the server against the declared spec directly. Pure Go (disintegration/imaging), no cgo.
package preprocess

import (
	"fmt"
	"math"
	"strings"

	"github.com/disintegration/imaging"

	"visionserve/internal/vision/geom"
)

// Mode is how an image is brought to the model's input size. Only modes that a served
// architecture really uses exist; each one reproduces its upstream recipe exactly.
type Mode string

const (
	// Squash resizes to exactly Width×Height; the aspect ratio is not kept (RF-DETR, MiDaS,
	// EfficientNet, SAM2, …). Default resample: bilinear.
	Squash Mode = "squash"
	// Letterbox fits the image inside Width×Height keeping its aspect ratio (scale =
	// min(W/w, H/h), sides rounded half up), centres it and pads with the pixel PadValue
	// before normalisation. Default resample: bilinear.
	Letterbox Mode = "letterbox"
	// CenterCrop resizes the SHORT side to its target (long side truncated) and keeps the
	// centred Width×Height window, offset floored: HuggingFace CLIPImageProcessor. Default
	// resample: bicubic.
	CenterCrop Mode = "center_crop"
	// KeepAspect is HuggingFace DPTImageProcessor's keep_aspect_ratio rule (DPTKeepAspectSize,
	// sides rounded to MultipleOf): no crop, no pad, the tensor size varies per image. Default
	// resample: bicubic.
	KeepAspect Mode = "keep_aspect"
	// LongSide scales by min(W/w, H/h) (the long side → its target for a square target), sides
	// rounded half away from zero, no pad: the tensor size varies per image (MobileSAM, whose
	// graph pads itself). Default resample: bilinear.
	LongSide Mode = "long_side"
	// LongSidePad is LongSide, then the tensor is padded at the bottom/right — to Width×Height,
	// or with MultipleOf > 0 up to the next multiple of it — AFTER normalisation with PadValue
	// (SAM: normalise, then F.pad with zeros; PaddleOCR DBNet). Default resample: bilinear.
	LongSidePad Mode = "long_side_pad"
	// TopLeftPad is InsightFace's SCRFD detect(): fit inside Width×Height by the image/model
	// aspect ratios (new size truncated), paste at the TOP-LEFT of a PadValue canvas, then
	// normalise. Default resample: bilinear. Meta carries the content's per-axis scales
	// (new_w/w, new_h/h), not upstream's single det_scale = new_h/h (see TopLeftSize).
	TopLeftPad Mode = "top_left_pad"
	// None feeds the image at its original size (EfficientSAM: the graph resizes itself).
	None Mode = "none"
)

// Modes lists every resize mode, in documentation order.
var Modes = []Mode{Squash, Letterbox, CenterCrop, KeepAspect, LongSide, LongSidePad, TopLeftPad, None}

// Pads reports whether the mode keeps the aspect ratio and fills an area outside the resized
// image — what the legacy `input.letterbox: true` ("keep aspect ratio + pad") describes.
func (m Mode) Pads() bool { return m == Letterbox || m == TopLeftPad || m == LongSidePad }

// Known reports whether m is one of Modes.
func (m Mode) Known() bool {
	for _, k := range Modes {
		if m == k {
			return true
		}
	}
	return false
}

// Resample is the interpolation filter of the resize.
type Resample string

const (
	// Bilinear is imaging.Linear: an antialiased triangle filter (PIL BILINEAR, torchvision
	// Resize with antialias=True).
	Bilinear Resample = "bilinear"
	// Bicubic is imaging.CatmullRom: the a = -0.5 cubic PIL's BICUBIC implements.
	Bicubic Resample = "bicubic"
)

// Layout is the memory order of the produced tensor.
type Layout string

const (
	NCHW Layout = "NCHW" // [1,3,H,W]
	NHWC Layout = "NHWC" // [1,H,W,3]
	HWC  Layout = "HWC"  // [H,W,3] — no batch axis (MobileSAM's encoder)
)

// Spec declares one model's preprocessing. The zero value of each field is the common case.
//
// Pixel values become tensor values as follows (c = channel, p = 0..255):
//
//	default:                  v = (p/255 - Mean[c]) / Std[c]   (no Mean/Std: v = p/255)
//	NoRescale, no Mean/Std:   v = p                              (raw 0..255, MobileSAM)
//	NoRescale with Mean/Std:  v = (p - Mean[c]) / Std[c]         (Mean/Std in 0..255 units, SCRFD),
//	                          computed as (p/255 - Mean[c]/255) / (Std[c]/255)
//
// A missing Mean entry counts as 0 and a missing or zero Std entry as 1 (in the units the
// formula divides by), which is what the hand-written loops this package replaced did.
type Spec struct {
	// Resize: the geometry; "" = the architecture's default (Arch.Resolve fills it in).
	Resize Mode
	// Width, Height: the target size (unused by None). LongSide/LongSidePad scale by
	// min(Width/w, Height/h).
	Width, Height int
	// MultipleOf: KeepAspect rounds each side to a multiple of it; LongSidePad pads each side up
	// to a multiple of it (instead of to Width×Height). 0 = unused.
	MultipleOf int
	// NoUpscale: LongSide/LongSidePad never enlarge (the scale is capped at 1) — PaddleOCR's det.
	NoUpscale bool
	// Resample: "" = the mode's default (see Mode).
	Resample Resample
	// Mean, Std: per-channel normalisation, RGB order (see the formula above).
	Mean, Std []float32
	// NoRescale: keep pixels in 0..255 instead of dividing by 255 (HuggingFace do_rescale=False).
	NoRescale bool
	// Layout: "" = NCHW.
	Layout Layout
	// PadValue fills the padding. Letterbox / TopLeftPad: a pixel gray level (0..255, default 0 =
	// black) normalised like the image. LongSidePad: the value written into the NORMALISED tensor
	// (SAM pads the normalised image with zeros).
	PadValue float32

	// Legacy is true when the spec was mapped from the legacy manifest fields (input.letterbox,
	// input.crop, input.keep_aspect, input.normalize, input.layout) rather than declared in a
	// `preprocess:` block. Architectures predate the block and read those fields their own way
	// (Arch.Resolve): a legacy flag an architecture never honoured keeps being ignored, while
	// the same mode declared in a block is an error.
	Legacy bool
}

// LegacyFields holds the manifest's legacy `input.*` preprocessing fields (and their models.Config
// mirrors).
type LegacyFields struct {
	Width, Height int
	Layout        string
	Letterbox     bool
	Crop          string
	KeepAspect    bool
	MultipleOf    int
	Mean, Std     []float32
}

// FromLegacy maps the legacy fields to a Spec (Legacy = true). input.keep_aspect →
// KeepAspect (with input.multiple_of), input.crop: center → CenterCrop, input.letterbox →
// Letterbox, none of them → Squash; the registry rejects combining them, and this order is
// the precedence for configs built in code. input.normalize → Mean/Std, input.layout → Layout.
func FromLegacy(l LegacyFields) Spec {
	s := Spec{Width: l.Width, Height: l.Height, Mean: l.Mean, Std: l.Std,
		Layout: Layout(strings.ToUpper(strings.TrimSpace(l.Layout))), Legacy: true}
	switch {
	case l.KeepAspect:
		s.Resize, s.MultipleOf = KeepAspect, l.MultipleOf
	case l.Crop == "center":
		s.Resize = CenterCrop
	case l.Letterbox:
		s.Resize = Letterbox
	default:
		s.Resize = Squash
	}
	return s
}

// Filter returns the imaging filter of the spec's resample (or of its mode's default).
func (s Spec) Filter() imaging.ResampleFilter {
	r := s.Resample
	if r == "" {
		r = Bilinear
		if s.Resize == CenterCrop || s.Resize == KeepAspect {
			r = Bicubic
		}
	}
	if r == Bicubic {
		return imaging.CatmullRom
	}
	return imaging.Linear
}

// Validate checks the spec. A declared spec (Legacy = false) is checked strictly; a legacy one
// only for what the code always needed (the old fields were never validated further, and a
// manifest that loaded before must keep loading).
//
// Resize "" is valid — "the architecture's default", which Arch.Resolve fills in — but Apply
// needs a concrete mode.
func (s Spec) Validate() error {
	if s.Resize != "" && !s.Resize.Known() {
		return fmt.Errorf("preprocess: resize %q is invalid (%s)", s.Resize, modeList())
	}
	if s.Resize != None && (s.Width <= 0 || s.Height <= 0) {
		return fmt.Errorf("preprocess: resize %s needs width/height > 0 (got %dx%d)", s.Resize, s.Width, s.Height)
	}
	if s.MultipleOf < 0 {
		return fmt.Errorf("preprocess: multiple_of must be >= 0")
	}
	switch s.Layout {
	case "", NCHW, NHWC, HWC:
	default:
		return fmt.Errorf("preprocess: layout %q is invalid (NCHW, NHWC or HWC)", s.Layout)
	}
	switch s.Resample {
	case "", Bilinear, Bicubic:
	default:
		return fmt.Errorf("preprocess: resample %q is invalid (bilinear or bicubic)", s.Resample)
	}
	// A NaN or Inf anywhere in the normalisation only ever yields NaN (or zeroed) tensor values,
	// legacy spec or not.
	for _, vals := range [][]float32{s.Mean, s.Std, {s.PadValue}} {
		for _, v := range vals {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("preprocess: mean, std and pad must be finite numbers (got mean %v, std %v, pad %v)",
					s.Mean, s.Std, s.PadValue)
			}
		}
	}
	if s.Legacy {
		return nil
	}
	if s.MultipleOf > 0 && s.Resize != KeepAspect && s.Resize != LongSidePad {
		return fmt.Errorf("preprocess: multiple_of applies to keep_aspect and long_side_pad, not %s", s.Resize)
	}
	if s.NoUpscale && s.Resize != LongSide && s.Resize != LongSidePad {
		return fmt.Errorf("preprocess: no_upscale applies to long_side and long_side_pad, not %s", s.Resize)
	}
	if s.Resize == None && s.Resample != "" {
		return fmt.Errorf("preprocess: resize none does not resample")
	}
	if s.PadValue != 0 && !s.Resize.Pads() {
		return fmt.Errorf("preprocess: pad applies to letterbox, top_left_pad and long_side_pad, not %s", s.Resize)
	}
	if (s.Resize == Letterbox || s.Resize == TopLeftPad) && !(s.PadValue >= 0 && s.PadValue <= 255) {
		return fmt.Errorf("preprocess: pad %g is a pixel value for %s and must be in [0,255]", s.PadValue, s.Resize)
	}
	if (len(s.Mean) == 0) != (len(s.Std) == 0) {
		return fmt.Errorf("preprocess: mean and std go together (got %d mean, %d std values)", len(s.Mean), len(s.Std))
	}
	if len(s.Mean) > 0 && (len(s.Mean) != 3 || len(s.Std) != 3) {
		return fmt.Errorf("preprocess: mean and std need 3 values each (RGB), got %d and %d", len(s.Mean), len(s.Std))
	}
	for _, v := range s.Std {
		if v == 0 || math.IsNaN(float64(v)) {
			return fmt.Errorf("preprocess: std values must be non-zero numbers, got %v", s.Std)
		}
	}
	return nil
}

func modeList() string {
	names := make([]string, len(Modes))
	for i, m := range Modes {
		names[i] = string(m)
	}
	return strings.Join(names, ", ")
}

// Arch is what one architecture accepts from a manifest.
type Arch struct {
	Name string
	// Modes are the resize modes the architecture serves correctly (its postprocess maps the
	// resulting Meta back); Modes[0] is its default.
	Modes []Mode
}

// Resolve returns the spec the architecture runs for s.
//
// A declared spec must use one of a.Modes (Resize "" picks the default). A legacy spec keeps
// the architecture's historical reading of the legacy fields: a mode it never honoured (rf-detr
// ignored input.crop, SCRFD's input.letterbox always meant InsightFace's top-left pad) falls
// back to the default, and input.layout — never read by any plain model — is NCHW.
func (a Arch) Resolve(s Spec) (Spec, error) {
	if len(a.Modes) == 0 {
		return Spec{}, fmt.Errorf("preprocess: architecture %q declares no resize modes", a.Name)
	}
	if s.Legacy {
		if !a.supports(s.Resize) {
			s.Resize = a.Modes[0]
		}
		s.Layout = NCHW
	} else {
		if s.Resize == "" {
			s.Resize = a.Modes[0]
		}
		if !a.supports(s.Resize) {
			names := make([]string, len(a.Modes))
			for i, m := range a.Modes {
				names[i] = string(m)
			}
			return Spec{}, fmt.Errorf("preprocess: %s does not support resize %q (supported: %s)",
				a.Name, s.Resize, strings.Join(names, ", "))
		}
	}
	if err := s.Validate(); err != nil {
		return Spec{}, fmt.Errorf("%s: %w", a.Name, err)
	}
	return s, nil
}

func (a Arch) supports(m Mode) bool {
	for _, k := range a.Modes {
		if k == m {
			return true
		}
	}
	return false
}

// FixedByExport is for architectures whose preprocessing the export fixes (the SAM family,
// PaddleOCR's detector): it refuses a declared `preprocess:` block, which such a model would
// otherwise silently ignore while `visionserve list` showed the block's size. The legacy input.*
// fields stay accepted — they are reference only for these architectures, as they always were.
func FixedByExport(arch string, s Spec) error {
	if s.Legacy {
		return nil
	}
	return fmt.Errorf("preprocess: %s's preprocessing is fixed by its export and does not read a preprocess: block — remove the block (see docs/manifest-spec.md)", arch)
}

// Meta holds what maps model-input coordinates back to the ORIGINAL image:
// input = orig * Scale + Pad on each axis (Pad is minus the crop offset for CenterCrop).
type Meta struct {
	OrigWidth  int
	OrigHeight int
	ScaleX     float64 // input_x = orig_x * ScaleX + PadX
	ScaleY     float64
	PadX       int
	PadY       int
}

// Affine returns the input = orig*Scale + Pad mapping recorded by Apply, for mapping boxes back
// to ORIGINAL image coordinates with geom.Affine.BoxToOrig.
func (m Meta) Affine() geom.Affine {
	return geom.Affine{ScaleX: m.ScaleX, ScaleY: m.ScaleY, PadX: float64(m.PadX), PadY: float64(m.PadY)}
}
