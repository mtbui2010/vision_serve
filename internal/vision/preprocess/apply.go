package preprocess

import (
	"fmt"
	"image"

	"github.com/disintegration/imaging"

	"visionserve/internal/engine"
)

// padKind says how render fills the frame outside the content.
type padKind int

const (
	noPad     padKind = iota // the content covers the whole frame
	pixelPad                 // PadValue is a pixel gray level, normalised like the image
	tensorPad                // PadValue is written into the normalised tensor as is
)

// Apply turns img into the model input the spec declares, and the Meta that maps model-input
// coordinates back to img (input = orig * Scale + Pad). An empty image is an error.
func (s Spec) Apply(img image.Image) (engine.Tensor, Meta, error) {
	if img == nil {
		return engine.Tensor{}, Meta{}, fmt.Errorf("preprocess: nil image")
	}
	b := img.Bounds()
	ow, oh := b.Dx(), b.Dy()
	if ow <= 0 || oh <= 0 {
		return engine.Tensor{}, Meta{}, fmt.Errorf("preprocess: empty image %dx%d", ow, oh)
	}
	if err := s.Validate(); err != nil {
		return engine.Tensor{}, Meta{}, err
	}
	meta := Meta{OrigWidth: ow, OrigHeight: oh}
	W, H := s.Width, s.Height
	switch s.Resize {
	case Squash:
		r := imaging.Resize(img, W, H, s.Filter())
		meta.ScaleX, meta.ScaleY = float64(W)/float64(ow), float64(H)/float64(oh)
		return s.render(r, 0, 0, W, H, W, H, 0, 0, noPad), meta, nil

	case Letterbox:
		nw, nh, scale, px, py := LetterboxSize(ow, oh, W, H)
		r := imaging.Resize(img, nw, nh, s.Filter())
		meta.ScaleX, meta.ScaleY, meta.PadX, meta.PadY = scale, scale, px, py
		return s.render(r, 0, 0, nw, nh, W, H, px, py, pixelPad), meta, nil

	case CenterCrop:
		rw, rh, offX, offY := CoverSize(ow, oh, W, H)
		r := imaging.Resize(img, rw, rh, s.Filter())
		meta.ScaleX, meta.ScaleY = float64(rw)/float64(ow), float64(rh)/float64(oh)
		meta.PadX, meta.PadY = -offX, -offY
		return s.render(r, offX, offY, W, H, W, H, 0, 0, noPad), meta, nil

	case KeepAspect:
		nw, nh := DPTKeepAspectSize(ow, oh, W, H, s.MultipleOf)
		r := imaging.Resize(img, nw, nh, s.Filter())
		meta.ScaleX, meta.ScaleY = float64(nw)/float64(ow), float64(nh)/float64(oh)
		return s.render(r, 0, 0, nw, nh, nw, nh, 0, 0, noPad), meta, nil

	case TopLeftPad:
		nw, nh, scale := TopLeftSize(ow, oh, W, H)
		r := imaging.Resize(img, nw, nh, s.Filter())
		meta.ScaleX, meta.ScaleY = scale, scale
		return s.render(r, 0, 0, nw, nh, W, H, 0, 0, pixelPad), meta, nil

	case LongSide, LongSidePad:
		nw, nh, scale := LongSideSize(ow, oh, W, H, s.NoUpscale)
		r := imaging.Resize(img, nw, nh, s.Filter())
		meta.ScaleX, meta.ScaleY = scale, scale
		if s.Resize == LongSide {
			return s.render(r, 0, 0, nw, nh, nw, nh, 0, 0, noPad), meta, nil
		}
		outW, outH := W, H
		if s.MultipleOf > 0 {
			outW, outH = ceilMultiple(nw, s.MultipleOf), ceilMultiple(nh, s.MultipleOf)
		}
		return s.render(r, 0, 0, nw, nh, outW, outH, 0, 0, tensorPad), meta, nil

	case None:
		meta.ScaleX, meta.ScaleY = 1, 1
		return s.Tensor(img), meta, nil
	}
	return engine.Tensor{}, Meta{}, fmt.Errorf("preprocess: resize %q is invalid (%s)", s.Resize, modeList())
}

// Tensor normalises img as it is — no resize, no pad — into the spec's layout. It is the shared
// pixels → tensor step for the preprocessing that stays model-specific (crops of detections,
// PaddleOCR's text lines, …); geometry fields of s are ignored. Any image type is accepted: an
// *image.NRGBA is read in place, other types are converted exactly as image.NRGBA.Set would.
func (s Spec) Tensor(img image.Image) engine.Tensor {
	n := toNRGBA(img)
	w, h := n.Rect.Dx(), n.Rect.Dy()
	return s.render(n, 0, 0, w, h, w, h, 0, 0, noPad)
}

// normalizer holds the per-channel affine map of the spec (see the Spec doc comment).
type normalizer struct {
	raw       bool // NoRescale without Mean/Std: v = p
	mean, std [3]float32
}

func (s Spec) normalizer() normalizer {
	mean, std := s.Mean, s.Std
	if s.NoRescale {
		if len(mean) == 0 && len(std) == 0 {
			return normalizer{raw: true}
		}
		// Mean/Std in 0..255 units: (p/255 - mean/255) / (std/255), the arithmetic SCRFD used.
		mean, std = div255(mean), div255(std)
	}
	var n normalizer
	for c := 0; c < 3; c++ {
		n.mean[c], n.std[c] = 0, 1
		if c < len(mean) {
			n.mean[c] = mean[c]
		}
		if c < len(std) && std[c] != 0 {
			n.std[c] = std[c]
		}
	}
	return n
}

func div255(vals []float32) []float32 {
	if len(vals) == 0 {
		return vals
	}
	out := make([]float32, len(vals))
	for i, v := range vals {
		out[i] = v / 255.0
	}
	return out
}

// value is the tensor value of pixel level p (0..255) on channel c. The expression is kept
// exactly as the per-model loops wrote it, so tensors stay bit-identical.
func (n normalizer) value(c int, p float32) float32 {
	if n.raw {
		return p
	}
	return (p/255.0 - n.mean[c]) / n.std[c]
}

// render writes an outW×outH tensor in the spec's layout: the cw×ch window of src starting at
// (sx, sy) lands with its top-left at (dx, dy); the rest of the frame is padding (pad). Content
// past the frame is cut, like imaging.Paste does.
func (s Spec) render(src *image.NRGBA, sx, sy, cw, ch, outW, outH, dx, dy int, pad padKind) engine.Tensor {
	n := s.normalizer()
	var lut [3][256]float32
	for c := 0; c < 3; c++ {
		for p := 0; p < 256; p++ {
			lut[c][p] = n.value(c, float32(p))
		}
	}
	data := make([]float32, 3*outW*outH)

	if pad != noPad {
		var pv [3]float32
		for c := 0; c < 3; c++ {
			if pad == pixelPad {
				pv[c] = n.value(c, s.PadValue)
			} else {
				pv[c] = s.PadValue
			}
		}
		s.fill(data, outW, outH, pv)
	}

	// Clip the content window to the frame.
	if dx+cw > outW {
		cw = outW - dx
	}
	if dy+ch > outH {
		ch = outH - dy
	}
	if cw <= 0 || ch <= 0 {
		return s.shape(data, outW, outH)
	}

	plane := outW * outH
	stride := src.Stride
	pix := src.Pix
	switch s.Layout {
	case NHWC, HWC:
		for y := 0; y < ch; y++ {
			row := pix[(sy+y)*stride+sx*4 : (sy+y)*stride+(sx+cw)*4]
			o := ((dy+y)*outW + dx) * 3
			for x := 0; x < cw; x++ {
				p := row[x*4 : x*4+3 : x*4+3]
				data[o] = lut[0][p[0]]
				data[o+1] = lut[1][p[1]]
				data[o+2] = lut[2][p[2]]
				o += 3
			}
		}
	default: // NCHW
		r, g, bl := data[:plane], data[plane:2*plane], data[2*plane:]
		for y := 0; y < ch; y++ {
			row := pix[(sy+y)*stride+sx*4 : (sy+y)*stride+(sx+cw)*4]
			o := (dy+y)*outW + dx
			for x := 0; x < cw; x++ {
				p := row[x*4 : x*4+3 : x*4+3]
				r[o] = lut[0][p[0]]
				g[o] = lut[1][p[1]]
				bl[o] = lut[2][p[2]]
				o++
			}
		}
	}
	return s.shape(data, outW, outH)
}

// fill writes the per-channel pad value into every element (render then overwrites the
// content). A +0 pad needs nothing: make() already zeroed the slice.
func (s Spec) fill(data []float32, outW, outH int, pv [3]float32) {
	if pv[0] == 0 && pv[1] == 0 && pv[2] == 0 && !negZero(pv) {
		return
	}
	plane := outW * outH
	switch s.Layout {
	case NHWC, HWC:
		for i := 0; i < plane; i++ {
			data[3*i], data[3*i+1], data[3*i+2] = pv[0], pv[1], pv[2]
		}
	default:
		for c := 0; c < 3; c++ {
			p := data[c*plane : (c+1)*plane]
			for i := range p {
				p[i] = pv[c]
			}
		}
	}
}

// negZero reports a -0 among the pad values (it compares equal to 0 but has different bits).
func negZero(pv [3]float32) bool {
	for _, v := range pv {
		if v == 0 && 1/v < 0 {
			return true
		}
	}
	return false
}

func (s Spec) shape(data []float32, w, h int) engine.Tensor {
	switch s.Layout {
	case NHWC:
		return engine.F32(data, 1, int64(h), int64(w), 3)
	case HWC:
		return engine.F32(data, int64(h), int64(w), 3)
	default:
		return engine.F32(data, 1, 3, int64(h), int64(w))
	}
}
