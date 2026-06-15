// Package morph applies a fast morphological dilate/erode to result masks. A SQUARE
// (Chebyshev) structuring element of radius r is separable into a 1-D pass along rows then
// columns, so the whole operation is O(W*H) regardless of r (running window count). Used
// generically by the predict handler and the run CLI to enlarge (dilate) or shrink (erode)
// segmentation masks via a single signed `dilate` parameter.
package morph

import "visionserve/pkg/api"

// Dilate grows (radius>0) or erodes (radius<0) a ROW-MAJOR bool mask by |radius| pixels with
// a square structuring element. radius==0 returns the input unchanged. Out-of-image is
// treated as "don't care" for erosion (so masks touching the image border are not eroded
// away just by being at the edge) and as background for dilation.
func Dilate(data []bool, w, h, radius int) []bool {
	if radius == 0 || w <= 0 || h <= 0 || len(data) != w*h {
		return data
	}
	erode := radius < 0
	r := radius
	if erode {
		r = -r
	}
	tmp := pass1D(data, w, h, r, erode, true)  // horizontal
	return pass1D(tmp, w, h, r, erode, false)  // vertical
}

// pass1D runs one separable 1-D dilation (OR) or erosion (AND) over a sliding window of
// width 2r+1, along rows (horizontal) or columns (vertical), with a running count of set
// pixels. For erosion, a pixel survives only when EVERY in-bounds pixel of its (clamped)
// window is set.
func pass1D(src []bool, w, h, r int, erode, horizontal bool) []bool {
	out := make([]bool, w*h)
	var n, stride, lineLen int
	if horizontal {
		n, lineLen, stride = h, w, 1 // h lines of length w, step 1 within a row
	} else {
		n, lineLen, stride = w, h, w // w lines of length h, step w within a column
	}
	for line := 0; line < n; line++ {
		var base int
		if horizontal {
			base = line * w // row start
		} else {
			base = line // column start
		}
		at := func(i int) bool { return src[base+i*stride] }

		// Prime the window for i=0: indices [0, min(r, lineLen-1)].
		cnt := 0
		hi0 := r
		if hi0 > lineLen-1 {
			hi0 = lineLen - 1
		}
		for k := 0; k <= hi0; k++ {
			if at(k) {
				cnt++
			}
		}
		for i := 0; i < lineLen; i++ {
			lo := i - r
			if lo < 0 {
				lo = 0
			}
			hi := i + r
			if hi > lineLen-1 {
				hi = lineLen - 1
			}
			win := hi - lo + 1
			var v bool
			if erode {
				v = cnt == win
			} else {
				v = cnt > 0
			}
			out[base+i*stride] = v
			// Slide to i+1: add index i+1+r, drop index i-r.
			if add := i + 1 + r; add <= lineLen-1 && at(add) {
				cnt++
			}
			if rem := i - r; rem >= 0 && at(rem) {
				cnt--
			}
		}
	}
	return out
}

// ApplyToMasks dilates/erodes each mask's bitmap in place — re-encoding the RLE and
// recomputing the tight bbox — at the given w×h resolution. radius==0 is a no-op.
func ApplyToMasks(masks []api.Mask, w, h, radius int) {
	if radius == 0 || len(masks) == 0 || w <= 0 || h <= 0 {
		return
	}
	for i := range masks {
		bin := api.DecodeMaskRLE(masks[i].RLE, w, h)
		bin = Dilate(bin, w, h, radius)
		masks[i].RLE = api.EncodeMaskRLE(bin, w, h)
		masks[i].BBox = tightBBox(bin, w, h)
	}
}

func tightBBox(bin []bool, w, h int) [4]float64 {
	minX, minY, maxX, maxY := w, h, -1, -1
	for y := 0; y < h; y++ {
		row := y * w
		for x := 0; x < w; x++ {
			if bin[row+x] {
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}
	if maxX < 0 {
		return [4]float64{}
	}
	return [4]float64{float64(minX), float64(minY), float64(maxX - minX + 1), float64(maxY - minY + 1)}
}
