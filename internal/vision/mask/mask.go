// Package mask holds the binary-mask primitives every segmentation decoder needs:
// thresholding logits/probabilities into a Bitmap (with its tight bbox), the ONE
// column-major RLE codec of the public Mask schema, bilinear/nearest upsampling and
// min-max normalisation of score maps, and pixel IoU between bitmaps.
//
// Argument order: dimensions are ALWAYS passed rows-first, (h, w), like the [.., H, W]
// tensors they come from. A Bitmap carries its own W and H, so the RLE encoder takes a
// Bitmap and cannot be called with swapped dimensions.
package mask

// Bitmap is a binary mask: row-major Data of length W*H, true = foreground.
type Bitmap struct {
	Data []bool
	W, H int
}

// New returns an all-background h×w bitmap.
func New(h, w int) Bitmap {
	if h < 0 || w < 0 {
		h, w = 0, 0
	}
	return Bitmap{Data: make([]bool, h*w), W: w, H: h}
}

// Extent is the inclusive pixel extent [MinX..MaxX]×[MinY..MaxY] of a bitmap's set pixels
// and their count. An empty extent has Area 0, MaxX = MaxY = -1 and MinX/MinY = W/H.
type Extent struct {
	MinX, MinY, MaxX, MaxY int
	Area                   int
}

func emptyExtent(h, w int) Extent { return Extent{MinX: w, MinY: h, MaxX: -1, MaxY: -1} }

// Empty reports whether no pixel is set.
func (e Extent) Empty() bool { return e.MaxX < 0 }

// XYWH is the tight bbox [x, y, w, h] in pixels (zero for an empty extent).
func (e Extent) XYWH() [4]float64 {
	if e.MaxX < 0 {
		return [4]float64{}
	}
	return [4]float64{float64(e.MinX), float64(e.MinY), float64(e.MaxX - e.MinX + 1), float64(e.MaxY - e.MinY + 1)}
}

// Extent scans the bitmap for its set pixels.
func (b Bitmap) Extent() Extent {
	e := emptyExtent(b.H, b.W)
	for y := 0; y < b.H; y++ {
		row := b.Data[y*b.W : (y+1)*b.W]
		for x, v := range row {
			if v {
				e.add(x, y)
			}
		}
	}
	return e
}

// BBox is the tight bbox [x, y, w, h] of the set pixels (zero when the bitmap is empty).
func (b Bitmap) BBox() [4]float64 { return b.Extent().XYWH() }

func (e *Extent) add(x, y int) {
	e.Area++
	if x < e.MinX {
		e.MinX = x
	}
	if x > e.MaxX {
		e.MaxX = x
	}
	if y < e.MinY {
		e.MinY = y
	}
	if y > e.MaxY {
		e.MaxY = y
	}
}

// ThresholdExtent binarizes the h×w row-major plane starting at data[off] — pixel set
// when float64(v) > thr — and returns the bitmap with its extent.
func ThresholdExtent(data []float32, off, h, w int, thr float64) (Bitmap, Extent) {
	return threshold(data, off, h, w, thr, false)
}

// Threshold is ThresholdExtent returning the tight bbox [x, y, w, h] (SAM: logit > 0;
// probability maps: p > thr).
func Threshold(data []float32, off, h, w int, thr float64) (Bitmap, [4]float64) {
	b, e := threshold(data, off, h, w, thr, false)
	return b, e.XYWH()
}

// ThresholdGE is Threshold with an inclusive test, float64(v) >= thr (EfficientSAM's
// torch.ge(logits, 0)).
func ThresholdGE(data []float32, off, h, w int, thr float64) (Bitmap, [4]float64) {
	b, e := threshold(data, off, h, w, thr, true)
	return b, e.XYWH()
}

func threshold(data []float32, off, h, w int, thr float64, inclusive bool) (Bitmap, Extent) {
	b := New(h, w)
	e := emptyExtent(h, w)
	for y := 0; y < h; y++ {
		row := data[off+y*w : off+(y+1)*w]
		out := b.Data[y*w : (y+1)*w]
		for x, v := range row {
			set := float64(v) > thr
			if inclusive {
				set = float64(v) >= thr
			}
			if set {
				out[x] = true
				e.add(x, y)
			}
		}
	}
	return b, e
}

// PixelIoU is the exact pixel IoU of two same-size bitmaps given their extents; the
// intersection is only counted inside the extents' overlap. 0 for different sizes or an
// empty mask.
func PixelIoU(a Bitmap, ea Extent, b Bitmap, eb Extent) float64 {
	if a.W != b.W || a.H != b.H || ea.Area == 0 || eb.Area == 0 {
		return 0
	}
	x0, y0 := max(ea.MinX, eb.MinX), max(ea.MinY, eb.MinY)
	x1, y1 := min(ea.MaxX, eb.MaxX), min(ea.MaxY, eb.MaxY)
	if x0 > x1 || y0 > y1 {
		return 0
	}
	inter := 0
	for y := y0; y <= y1; y++ {
		ra := a.Data[y*a.W+x0 : y*a.W+x1+1]
		rb := b.Data[y*b.W+x0 : y*b.W+x1+1]
		for x := range ra {
			if ra[x] && rb[x] {
				inter++
			}
		}
	}
	union := ea.Area + eb.Area - inter
	if union <= 0 {
		return 0
	}
	return float64(inter) / float64(union)
}

// IoUMayExceed is a cheap upper bound test for PixelIoU(a, b) > thr: the intersection is at
// most min(areaA, areaB, extent-overlap area). False means the pair can be skipped.
func IoUMayExceed(ea, eb Extent, thr float64) bool {
	ow := min(ea.MaxX, eb.MaxX) - max(ea.MinX, eb.MinX) + 1
	oh := min(ea.MaxY, eb.MaxY) - max(ea.MinY, eb.MinY) + 1
	if ow <= 0 || oh <= 0 {
		return false
	}
	ub := min(ea.Area, eb.Area, ow*oh)
	return float64(ub)/float64(ea.Area+eb.Area-ub) > thr
}
