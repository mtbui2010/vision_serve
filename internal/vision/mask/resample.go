package mask

// BilinearTaps returns, for each of out output positions, the two source indices and the
// weight of the second one, matching PyTorch F.interpolate(mode="bilinear",
// align_corners=False): src = max((dst+0.5)·in/out − 0.5, 0), i0 = floor(src),
// i1 = min(i0+1, in−1), λ = src − i0.
func BilinearTaps(in, out int) (i0, i1 []int, lambda []float32) {
	i0 = make([]int, out)
	i1 = make([]int, out)
	lambda = make([]float32, out)
	scale := float64(in) / float64(out)
	for d := 0; d < out; d++ {
		src := (float64(d)+0.5)*scale - 0.5
		if src < 0 {
			src = 0
		}
		a := int(src)
		if a > in-1 {
			a = in - 1
		}
		b := a + 1
		if b > in-1 {
			b = in - 1
		}
		i0[d], i1[d], lambda[d] = a, b, float32(src-float64(a))
	}
	return i0, i1, lambda
}

// UpsampleBilinear resizes the top-left sh×sw window of a row-major map whose rows are
// `stride` values apart to dh×dw, with PyTorch bilinear align_corners=False semantics
// (SAM's postprocess_masks / upscale_mask). stride == sw resizes the whole map.
func UpsampleBilinear(src []float32, stride, sh, sw, dh, dw int) []float32 {
	y0, y1, ly := BilinearTaps(sh, dh)
	x0, x1, lx := BilinearTaps(sw, dw)
	dst := make([]float32, dh*dw)
	for y := 0; y < dh; y++ {
		r0 := src[y0[y]*stride : y0[y]*stride+sw]
		r1 := src[y1[y]*stride : y1[y]*stride+sw]
		wy := ly[y]
		out := dst[y*dw : (y+1)*dw]
		for x := 0; x < dw; x++ {
			a, c, wx := x0[x], x1[x], lx[x]
			top := r0[a] + (r0[c]-r0[a])*wx
			bot := r1[a] + (r1[c]-r1[a])*wx
			out[x] = top + (bot-top)*wy
		}
	}
	return dst
}

// UpsampleBilinearThreshold is UpsampleBilinear fused with Threshold: it returns the
// dh×dw bitmap of upsampled values > thr without materialising the float map (a
// full-resolution float plane per mask is the dominant allocation otherwise).
func UpsampleBilinearThreshold(src []float32, stride, sh, sw, dh, dw int, thr float64) Bitmap {
	y0, y1, ly := BilinearTaps(sh, dh)
	x0, x1, lx := BilinearTaps(sw, dw)
	b := New(dh, dw)
	for y := 0; y < dh; y++ {
		r0 := src[y0[y]*stride : y0[y]*stride+sw]
		r1 := src[y1[y]*stride : y1[y]*stride+sw]
		wy := ly[y]
		out := b.Data[y*dw : (y+1)*dw]
		for x := 0; x < dw; x++ {
			a, c, wx := x0[x], x1[x], lx[x]
			top := r0[a] + (r0[c]-r0[a])*wx
			bot := r1[a] + (r1[c]-r1[a])*wx
			out[x] = float64(top+(bot-top)*wy) > thr
		}
	}
	return b
}

// UpsampleNearest nearest-neighbour resizes a row-major sh×sw map to dh×dw
// (source index = dst·src/dst, integer division).
func UpsampleNearest(src []float32, sh, sw, dh, dw int) []float32 {
	dst := make([]float32, dw*dh)
	for dy := 0; dy < dh; dy++ {
		sy := dy * sh / dh
		if sy >= sh {
			sy = sh - 1
		}
		for dx := 0; dx < dw; dx++ {
			sx := dx * sw / dw
			if sx >= sw {
				sx = sw - 1
			}
			dst[dy*dw+dx] = src[sy*sw+sx]
		}
	}
	return dst
}

// MinMaxNormalize rescales xs to [0, 1] as (v − min)/(max − min) into a new slice. When the
// range is 0 or below minRange (pass 0 for "only a perfectly flat input") the result is
// all zeros — a flat map carries no information. An empty input is returned as is.
func MinMaxNormalize(xs []float32, minRange float32) []float32 {
	if len(xs) == 0 {
		return xs
	}
	mn, mx := xs[0], xs[0]
	for _, v := range xs[1:] {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	out := make([]float32, len(xs))
	rng := mx - mn
	if rng == 0 || rng < minRange {
		return out
	}
	for i, v := range xs {
		out[i] = (v - mn) / rng
	}
	return out
}
