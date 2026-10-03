package owlvit

import (
	"image"
	"image/draw"
	"math"
)

// padResizeOWLv2 reproduces HF transformers' Owlv2ImageProcessor (torchvision backend,
// transformers 5.x image_processing_owlv2.py) for one image, returning a CHW float32
// buffer of size 3*out*out in [0,1] (rescaled, NOT yet normalized), plus the side S of the
// padded square:
//
//  1. rescale to [0,1];
//  2. pad to a square S = max(W,H) on the BOTTOM and RIGHT with 0.0 (black) — the image is
//     NOT squashed, so box coordinates are relative to the padded square;
//  3. if S > out: Gaussian anti-alias blur, sigma = (S/out - 1)/2, kernel 2*ceil(3σ)+1,
//     reflect border (torchvision gaussian_blur);
//  4. bilinear resize to out×out, align_corners=False, no further antialias.
//
// The blur is separable and only evaluated at the source rows/columns the bilinear sampler
// touches, so memory stays O(S) instead of O(S²) for large photos.
func padResizeOWLv2(img image.Image, out int) (chw []float32, side int) {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	side = w
	if h > side {
		side = h
	}

	nrgba, ok := img.(*image.NRGBA)
	if !ok || nrgba.Rect.Min != (image.Point{}) {
		nrgba = image.NewNRGBA(image.Rect(0, 0, w, h))
		draw.Draw(nrgba, nrgba.Rect, img, b.Min, draw.Src)
	}

	// Gaussian kernel (torchvision _get_gaussian_kernel1d).
	var kern []float64
	if factor := float64(side) / float64(out); factor > 1 {
		sigma := (factor - 1) / 2
		half := int(math.Ceil(3 * sigma))
		kern = make([]float64, 2*half+1)
		var sum float64
		for i := range kern {
			x := float64(i - half)
			kern[i] = math.Exp(-0.5 * (x / sigma) * (x / sigma))
			sum += kern[i]
		}
		for i := range kern {
			kern[i] /= sum
		}
	}
	half := len(kern) / 2

	// reflect maps an index into [0,side) like torch's "reflect" padding (edge not repeated).
	reflect := func(i int) int {
		if side == 1 {
			return 0
		}
		for i < 0 || i >= side {
			if i < 0 {
				i = -i
			}
			if i >= side {
				i = 2*(side-1) - i
			}
		}
		return i
	}
	// pixel of the PADDED square in [0,1]; the pad region is 0.
	pix := func(y, x, c int) float64 {
		if x >= w || y >= h {
			return 0
		}
		return float64(nrgba.Pix[y*nrgba.Stride+x*4+c]) / 255.0
	}

	// Bilinear source taps (align_corners=False), shared by both axes since the input is square.
	scale := float64(side) / float64(out)
	i0 := make([]int, out)
	i1 := make([]int, out)
	lam := make([]float64, out)
	for d := 0; d < out; d++ {
		s := (float64(d)+0.5)*scale - 0.5
		if s < 0 {
			s = 0
		}
		f := int(math.Floor(s))
		if f > side-1 {
			f = side - 1
		}
		i0[d] = f
		i1[d] = f + 1
		if i1[d] > side-1 {
			i1[d] = side - 1
		}
		lam[d] = s - float64(f)
	}

	// Columns that the sampler reads; the horizontal blur is evaluated only there.
	needCol := make([]int, 0, 2*out)
	colIdx := make(map[int]int, 2*out)
	for d := 0; d < out; d++ {
		for _, c := range [2]int{i0[d], i1[d]} {
			if _, ok := colIdx[c]; !ok {
				colIdx[c] = len(needCol)
				needCol = append(needCol, c)
			}
		}
	}

	// blurredRow returns the fully blurred (vertical then horizontal) values of source row y
	// at needCol, as [3][len(needCol)]. The last two rows are cached (the sampler visits rows
	// in non-decreasing order) and their buffers are recycled on eviction.
	vrow := [3][]float64{make([]float64, side), make([]float64, side), make([]float64, side)}
	type cached struct {
		y   int
		val [3][]float64
	}
	var cache [2]*cached
	newBuf := func() [3][]float64 {
		n := len(needCol)
		return [3][]float64{make([]float64, n), make([]float64, n), make([]float64, n)}
	}
	blurredRow := func(y int) [3][]float64 {
		for _, e := range cache {
			if e != nil && e.y == y {
				return e.val
			}
		}
		// Pick the slot to (re)fill: an empty one, else the older (smaller-y) one.
		slot := 0
		switch {
		case cache[0] == nil:
		case cache[1] == nil:
			slot = 1
		case cache[1].y < cache[0].y:
			slot = 1
		}
		if cache[slot] == nil {
			cache[slot] = &cached{val: newBuf()}
		}
		e := cache[slot]
		e.y = y
		res := e.val
		if kern == nil {
			for k, x := range needCol {
				for c := 0; c < 3; c++ {
					res[c][k] = pix(y, x, c)
				}
			}
			return res
		}
		// Vertical blur over the content part of the row (columns >= w are padding = 0).
		for c := 0; c < 3; c++ {
			clear(vrow[c])
		}
		r, g, bl := vrow[0][:w], vrow[1][:w], vrow[2][:w]
		for t, wt := range kern {
			yy := reflect(y + t - half)
			if yy >= h {
				continue // padding row
			}
			src := nrgba.Pix[yy*nrgba.Stride : yy*nrgba.Stride+4*w]
			for x := 0; x < w; x++ {
				r[x] += wt * float64(src[4*x])
				g[x] += wt * float64(src[4*x+1])
				bl[x] += wt * float64(src[4*x+2])
			}
		}
		// Horizontal blur at the needed columns only.
		for k, x := range needCol {
			for c := 0; c < 3; c++ {
				var acc float64
				for t, wt := range kern {
					acc += wt * vrow[c][reflect(x+t-half)]
				}
				res[c][k] = acc / 255.0
			}
		}
		return res
	}

	k0 := make([]int, out)
	k1 := make([]int, out)
	for d := 0; d < out; d++ {
		k0[d], k1[d] = colIdx[i0[d]], colIdx[i1[d]]
	}

	plane := out * out
	chw = make([]float32, 3*plane)
	for dy := 0; dy < out; dy++ {
		r0 := blurredRow(i0[dy])
		r1 := blurredRow(i1[dy])
		ly := lam[dy]
		for dx := 0; dx < out; dx++ {
			c0, c1 := k0[dx], k1[dx]
			lx := lam[dx]
			for c := 0; c < 3; c++ {
				top := r0[c][c0]*(1-lx) + r0[c][c1]*lx
				bot := r1[c][c0]*(1-lx) + r1[c][c1]*lx
				chw[c*plane+dy*out+dx] = float32(top*(1-ly) + bot*ly)
			}
		}
	}
	return chw, side
}
