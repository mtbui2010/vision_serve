package registry

import (
	"fmt"
	"strings"

	"gopkg.in/yaml.v3"

	"visionserve/internal/vision/preprocess"
)

// PreprocessBlock is the manifest's optional `preprocess:` block — a preprocess.Spec written as
// YAML (docs/manifest-spec.md, "Preprocessing"):
//
//	preprocess:
//	  resize: letterbox      # squash | letterbox | center_crop | keep_aspect | long_side |
//	                         # long_side_pad | top_left_pad | none
//	  size: 640              # or width: / height:
//	  multiple_of: 14        # keep_aspect: round sides; long_side_pad: pad sides up to it
//	  no_upscale: true       # long_side / long_side_pad: never enlarge
//	  resample: bilinear     # bilinear | bicubic (default: the mode's)
//	  mean: [0.485, 0.456, 0.406]
//	  std:  [0.229, 0.224, 0.225]
//	  rescale: false         # keep 0..255 (no /255); mean/std are then in 0..255 units
//	  layout: NCHW           # NCHW | NHWC | HWC
//	  pad: 0                 # letterbox / top_left_pad: pixel gray level; long_side_pad: tensor value
type PreprocessBlock struct {
	Resize     string    `yaml:"resize,omitempty"`
	Size       int       `yaml:"size,omitempty"`
	Width      int       `yaml:"width,omitempty"`
	Height     int       `yaml:"height,omitempty"`
	MultipleOf int       `yaml:"multiple_of,omitempty"`
	NoUpscale  bool      `yaml:"no_upscale,omitempty"`
	Resample   string    `yaml:"resample,omitempty"`
	Mean       []float32 `yaml:"mean,omitempty"`
	Std        []float32 `yaml:"std,omitempty"`
	Rescale    *bool     `yaml:"rescale,omitempty"`
	Layout     string    `yaml:"layout,omitempty"`
	Pad        *float32  `yaml:"pad,omitempty"`
}

// legacyKeys says which legacy input.* preprocessing keys a manifest declares.
type legacyKeys struct {
	width, height, layout, letterbox, crop, keepAspect, multipleOf, mean, std bool
}

// declaredInputKeys reads which input.* keys the YAML spells out (an explicit `letterbox: false`
// is a declaration; an absent key is not). An empty list or string declares nothing.
func declaredInputKeys(raw []byte) (legacyKeys, error) {
	var doc struct {
		Input struct {
			Width      *int    `yaml:"width"`
			Height     *int    `yaml:"height"`
			Layout     *string `yaml:"layout"`
			Letterbox  *bool   `yaml:"letterbox"`
			Crop       *string `yaml:"crop"`
			KeepAspect *bool   `yaml:"keep_aspect"`
			MultipleOf *int    `yaml:"multiple_of"`
			Normalize  struct {
				Mean []float32 `yaml:"mean"`
				Std  []float32 `yaml:"std"`
			} `yaml:"normalize"`
		} `yaml:"input"`
	}
	if err := yaml.Unmarshal(raw, &doc); err != nil {
		return legacyKeys{}, err
	}
	in := doc.Input
	return legacyKeys{
		width:      in.Width != nil && *in.Width > 0,
		height:     in.Height != nil && *in.Height > 0,
		layout:     in.Layout != nil && strings.TrimSpace(*in.Layout) != "",
		letterbox:  in.Letterbox != nil,
		crop:       in.Crop != nil && *in.Crop != "",
		keepAspect: in.KeepAspect != nil,
		multipleOf: in.MultipleOf != nil,
		mean:       len(in.Normalize.Mean) > 0,
		std:        len(in.Normalize.Std) > 0,
	}, nil
}

// keys returns the declared legacy keys; for a Manifest built in code, its non-zero fields.
func (m *Manifest) keys() legacyKeys {
	if m.inputKeys != nil {
		return *m.inputKeys
	}
	in := m.Input
	return legacyKeys{
		width: in.Width > 0, height: in.Height > 0, layout: strings.TrimSpace(in.Layout) != "",
		letterbox: in.Letterbox, crop: in.Crop != "", keepAspect: in.KeepAspect, multipleOf: in.MultipleOf != 0,
		mean: len(in.Normalize.Mean) > 0, std: len(in.Normalize.Std) > 0,
	}
}

// legacySpec maps the legacy input.* fields (preprocess.FromLegacy; Spec.Legacy = true).
func (m *Manifest) legacySpec() preprocess.Spec {
	in := m.Input
	return preprocess.FromLegacy(preprocess.LegacyFields{
		Width: in.Width, Height: in.Height, Layout: in.Layout,
		Letterbox: in.Letterbox, Crop: in.Crop, KeepAspect: in.KeepAspect, MultipleOf: in.MultipleOf,
		Mean: in.Normalize.Mean, Std: in.Normalize.Std,
	})
}

// PreprocessSpec resolves the model's preprocessing.
//
// Without a `preprocess:` block it is the legacy input.* fields mapped by preprocess.FromLegacy
// (Legacy = true: each architecture keeps its historical reading of them). With a block, every
// field the block sets wins, every field it leaves out is taken from its legacy alias, and a
// field set on BOTH sides must agree — otherwise the manifest is invalid and the error names
// both fields. input.letterbox is a yes/no alias ("keep the aspect ratio and pad"): true agrees
// with letterbox, top_left_pad and long_side_pad, false with every other mode.
func (m *Manifest) PreprocessSpec() (preprocess.Spec, error) {
	if m.Preprocess == nil {
		return m.legacySpec(), nil
	}
	b, in, k := m.Preprocess, m.Input, m.keys()
	conflict := func(legacy, block string) error {
		return fmt.Errorf("%s conflicts with %s: the preprocess: block and its legacy input.* alias must agree (or drop one of them)", legacy, block)
	}
	// Without a resize in the block, a legacy geometry flag that is set decides; with none of
	// them the mode stays "" — the architecture's default (preprocess.Arch.Resolve).
	s := preprocess.Spec{}
	switch {
	case in.KeepAspect:
		s.Resize = preprocess.KeepAspect
	case in.Crop == "center":
		s.Resize = preprocess.CenterCrop
	case in.Letterbox:
		s.Resize = preprocess.Letterbox
	}

	if r := strings.ToLower(strings.TrimSpace(b.Resize)); r != "" {
		mode := preprocess.Mode(r)
		if !mode.Known() {
			return preprocess.Spec{}, fmt.Errorf("preprocess.resize %q is invalid (%s)", b.Resize, modeNames())
		}
		block := "preprocess.resize: " + r
		if k.letterbox && in.Letterbox != mode.Pads() {
			return preprocess.Spec{}, conflict(fmt.Sprintf("input.letterbox: %t", in.Letterbox), block)
		}
		if k.crop && (in.Crop == "center") != (mode == preprocess.CenterCrop) {
			return preprocess.Spec{}, conflict(fmt.Sprintf("input.crop: %s", in.Crop), block)
		}
		if k.keepAspect && in.KeepAspect != (mode == preprocess.KeepAspect) {
			return preprocess.Spec{}, conflict(fmt.Sprintf("input.keep_aspect: %t", in.KeepAspect), block)
		}
		s.Resize = mode
	}

	w, h := b.Width, b.Height
	if b.Size < 0 || w < 0 || h < 0 {
		return preprocess.Spec{}, fmt.Errorf("preprocess.size/width/height must be > 0")
	}
	if b.Size > 0 {
		if (w != 0 && w != b.Size) || (h != 0 && h != b.Size) {
			return preprocess.Spec{}, fmt.Errorf("preprocess.size: %d disagrees with preprocess.width/height %dx%d", b.Size, w, h)
		}
		w, h = b.Size, b.Size
	}
	if w > 0 && k.width && in.Width != w {
		return preprocess.Spec{}, conflict(fmt.Sprintf("input.width: %d", in.Width), fmt.Sprintf("preprocess width %d", w))
	}
	if h > 0 && k.height && in.Height != h {
		return preprocess.Spec{}, conflict(fmt.Sprintf("input.height: %d", in.Height), fmt.Sprintf("preprocess height %d", h))
	}
	if w == 0 {
		w = in.Width
	}
	if h == 0 {
		h = in.Height
	}
	s.Width, s.Height = w, h

	switch {
	case b.MultipleOf != 0:
		if k.multipleOf && in.MultipleOf != b.MultipleOf {
			return preprocess.Spec{}, conflict(fmt.Sprintf("input.multiple_of: %d", in.MultipleOf), fmt.Sprintf("preprocess.multiple_of: %d", b.MultipleOf))
		}
		s.MultipleOf = b.MultipleOf
	case s.Resize == preprocess.KeepAspect || s.Resize == preprocess.LongSidePad:
		s.MultipleOf = in.MultipleOf
	}

	s.Mean, s.Std = b.Mean, b.Std
	if len(b.Mean) > 0 && k.mean && !sameFloats(in.Normalize.Mean, b.Mean) {
		return preprocess.Spec{}, conflict(fmt.Sprintf("input.normalize.mean: %v", in.Normalize.Mean), fmt.Sprintf("preprocess.mean: %v", b.Mean))
	}
	if len(b.Std) > 0 && k.std && !sameFloats(in.Normalize.Std, b.Std) {
		return preprocess.Spec{}, conflict(fmt.Sprintf("input.normalize.std: %v", in.Normalize.Std), fmt.Sprintf("preprocess.std: %v", b.Std))
	}
	if len(s.Mean) == 0 {
		s.Mean = in.Normalize.Mean
	}
	if len(s.Std) == 0 {
		s.Std = in.Normalize.Std
	}

	legacyLayout := strings.ToUpper(strings.TrimSpace(in.Layout))
	if l := strings.ToUpper(strings.TrimSpace(b.Layout)); l != "" {
		if k.layout && legacyLayout != l {
			return preprocess.Spec{}, conflict("input.layout: "+in.Layout, "preprocess.layout: "+b.Layout)
		}
		s.Layout = preprocess.Layout(l)
	} else {
		s.Layout = preprocess.Layout(legacyLayout)
	}

	s.NoUpscale = b.NoUpscale
	s.Resample = preprocess.Resample(strings.ToLower(strings.TrimSpace(b.Resample)))
	if b.Rescale != nil {
		s.NoRescale = !*b.Rescale
	}
	if b.Pad != nil {
		s.PadValue = *b.Pad
	}
	if err := s.Validate(); err != nil {
		return preprocess.Spec{}, err
	}
	return s, nil
}

// fillLegacyView writes what a resolved block declares into the legacy input.* fields that are
// still empty, so code reading those fields (lifecycle's models.Config, `visionserve list`,
// keep_aspect graph checks, models that predate the block) sees the declared preprocessing.
// Modes without a legacy spelling (top_left_pad, long_side, …) leave the flags alone, and a
// mean/std in 0..255 units (rescale: false) is not copied into input.normalize, which other
// architectures read in [0,1] units.
func (m *Manifest) fillLegacyView(s preprocess.Spec) {
	in := &m.Input
	if in.Width == 0 {
		in.Width = s.Width
	}
	if in.Height == 0 {
		in.Height = s.Height
	}
	switch s.Resize {
	case preprocess.Letterbox:
		in.Letterbox = true
	case preprocess.CenterCrop:
		in.Crop = "center"
	case preprocess.KeepAspect:
		in.KeepAspect = true
		if in.MultipleOf == 0 {
			in.MultipleOf = s.MultipleOf
		}
	}
	if !s.NoRescale {
		if len(in.Normalize.Mean) == 0 {
			in.Normalize.Mean = s.Mean
		}
		if len(in.Normalize.Std) == 0 {
			in.Normalize.Std = s.Std
		}
	}
	if in.Layout == "" && (s.Layout == preprocess.NCHW || s.Layout == preprocess.NHWC) {
		in.Layout = string(s.Layout)
	}
}

func sameFloats(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func modeNames() string {
	names := make([]string, len(preprocess.Modes))
	for i, m := range preprocess.Modes {
		names[i] = string(m)
	}
	return strings.Join(names, ", ")
}
