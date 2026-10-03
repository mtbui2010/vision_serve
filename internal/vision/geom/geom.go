// Package geom holds the box geometry every detector decoder needs: normalized-box →
// input-pixel conversion, the input → original-image mapping a preprocess implies,
// clamping to the image, IoU and the logistic sigmoid.
//
// Boxes are [x, y, w, h] (top-left corner + size) unless a name says otherwise — the
// convention of api.Detection.BBox, which is ALWAYS in original-image pixels (CLAUDE.md).
package geom

import "math"

// Sigmoid is the logistic function 1 / (1 + e^-x).
func Sigmoid(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }

// Box formats accepted by NormToInput.
const (
	FormatCXCYWH = "cxcywh"
	FormatXYXY   = "xyxy"
)

// FromCXCYWH converts a centre-format box to [x, y, w, h].
func FromCXCYWH(cx, cy, w, h float64) [4]float64 {
	return [4]float64{cx - w/2, cy - h/2, w, h}
}

// FromXYXY converts a corner-format box to [x, y, w, h].
func FromXYXY(x1, y1, x2, y2 float64) [4]float64 {
	return [4]float64{x1, y1, x2 - x1, y2 - y1}
}

// NormToInput scales a box normalized to [0,1] by the w×h input size and converts it to
// [x, y, w, h] input pixels. format is FormatXYXY or (anything else) FormatCXCYWH.
func NormToInput(a, b, c, d float64, format string, w, h int) [4]float64 {
	fw, fh := float64(w), float64(h)
	if format == FormatXYXY {
		return FromXYXY(a*fw, b*fh, c*fw, d*fh)
	}
	return FromCXCYWH(a*fw, b*fh, c*fw, d*fh)
}

// Affine is the per-axis mapping a preprocess applied: input = orig*Scale + Pad. Pad is
// the letterbox offset (or minus a crop offset).
type Affine struct {
	ScaleX, ScaleY float64
	PadX, PadY     float64
}

// BoxToOrig maps an [x, y, w, h] input-pixel box back to original-image pixels:
// orig = (input - pad) / scale for the corner, size / scale for the extent. No clamping.
func (a Affine) BoxToOrig(b [4]float64) [4]float64 {
	return [4]float64{
		(b[0] - a.PadX) / a.ScaleX,
		(b[1] - a.PadY) / a.ScaleY,
		b[2] / a.ScaleX,
		b[3] / a.ScaleY,
	}
}

// Clamp clips an [x, y, w, h] box to the W×H image: a corner left of / above the image
// moves onto the border and shrinks the box by the same amount, an extent past the
// right/bottom edge is cut, and a box left with a negative size gets size 0.
func Clamp(b [4]float64, w, h int) [4]float64 {
	x, y, bw, bh := b[0], b[1], b[2], b[3]
	if x < 0 {
		bw += x
		x = 0
	}
	if y < 0 {
		bh += y
		y = 0
	}
	if x+bw > float64(w) {
		bw = float64(w) - x
	}
	if y+bh > float64(h) {
		bh = float64(h) - y
	}
	if bw < 0 {
		bw = 0
	}
	if bh < 0 {
		bh = 0
	}
	return [4]float64{x, y, bw, bh}
}

// IoU is the intersection-over-union of two [x, y, w, h] boxes (0 when they do not
// overlap with a positive area, or the union is not positive).
func IoU(a, b [4]float64) float64 {
	ax1, ay1, ax2, ay2 := a[0], a[1], a[0]+a[2], a[1]+a[3]
	bx1, by1, bx2, by2 := b[0], b[1], b[0]+b[2], b[1]+b[3]
	iw := math.Min(ax2, bx2) - math.Max(ax1, bx1)
	ih := math.Min(ay2, by2) - math.Max(ay1, by1)
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	union := a[2]*a[3] + b[2]*b[3] - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}
