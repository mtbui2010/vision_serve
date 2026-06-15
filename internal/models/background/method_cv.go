package background

import (
	"image"
	"math"
	"sort"

	"visionserve/internal/imageproc"
	"visionserve/internal/models"
)

// backgroundCV (method=cv) — classical CV only (no ONNX). The support surface is a large,
// smooth (low-gradient), uniform region whose color matches a seed sampled from the bottom-
// center strip and which touches the image border. Strategy:
//
//   1. Downscale to ~256px long side for speed; work there, upsample the result back.
//   2. Seed color = robust (median) RGB of the bottom-center strip (the most likely surface).
//   3. surface-like = color within an adaptive threshold (derived from seed-region spread) of
//      the seed color, AND in a low-gradient (smooth, untextured) area (Sobel magnitude).
//   4. Connected components (4-connectivity); keep large border-touching component(s).
//   5. Validate with isBackgroundMask (area + border touch); else return nil.
//
// Returns a row-major []bool of length w*h at ORIGINAL resolution (true = support surface),
// or (nil,nil) when no support surface is found. r is unused (no session).
func (m *backgroundModel) backgroundCV(img image.Image, prompt models.Prompt, r models.Runner) ([]bool, error) {
	_ = prompt
	_ = r

	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()
	if origW <= 0 || origH <= 0 {
		return nil, nil
	}

	// 1. Downscale (long side ~256) for speed; keep aspect ratio.
	const target = 256
	w, h := origW, origH
	if origW > target || origH > target {
		if origW >= origH {
			w = target
			h = cvMax1(origH * target / origW)
		} else {
			h = target
			w = cvMax1(origW * target / origH)
		}
	}
	small := imageproc.Resize(img, w, h)

	// Extract RGB planes (0..255) row-major for cheap repeated access.
	rp := make([]float64, w*h)
	gp := make([]float64, w*h)
	bp := make([]float64, w*h)
	stride := small.Stride
	pix := small.Pix
	for y := 0; y < h; y++ {
		row := y * stride
		for x := 0; x < w; x++ {
			i := row + x*4
			idx := y*w + x
			rp[idx] = float64(pix[i])
			gp[idx] = float64(pix[i+1])
			bp[idx] = float64(pix[i+2])
		}
	}

	// 2. Seed color + spread from the bottom-center strip (bottom 15% rows, central 60% cols).
	seedR, seedG, seedB, spread := cvSeedColor(rp, gp, bp, w, h)
	if seedR < 0 {
		return nil, nil // no usable seed region
	}

	// Adaptive color threshold from the seed-region spread, clamped to a sane range.
	thr := spread * 2.5
	if thr < 18.0 {
		thr = 18.0
	}
	if thr > 70.0 {
		thr = 70.0
	}

	// 5 (compute first). Low-gradient map: Sobel magnitude; smooth pixels only.
	grad := cvSobel(rp, gp, bp, w, h)
	// Gradient threshold: a small fraction of the dynamic range. Surface is smooth.
	const gradThr = 50.0

	// 3. surface-like map: color near seed AND low gradient.
	like := make([]bool, w*h)
	for idx := 0; idx < w*h; idx++ {
		dr := rp[idx] - seedR
		dg := gp[idx] - seedG
		db := bp[idx] - seedB
		dist := math.Sqrt(dr*dr + dg*dg + db*db)
		if dist <= thr && grad[idx] <= gradThr {
			like[idx] = true
		}
	}

	// 4. Connected components (4-conn); keep large border-touching component(s).
	mask := cvSelectSurface(like, w, h)
	if mask == nil || !anySet(mask) {
		return nil, nil
	}

	// 6. Validate (area + border) at the working resolution.
	area := bitmapArea(mask)
	if !isBackgroundMask(area, float64(w*h), touchesBorder(mask, w, h)) {
		return nil, nil
	}

	// 7. Upsample back to original resolution.
	return upsampleMask(mask, w, h, origW, origH), nil
}

func cvMax1(v int) int {
	if v < 1 {
		return 1
	}
	return v
}

// cvSeedColor returns the median RGB and a robust color spread (mean per-channel std) of the
// bottom-center strip — the most likely support-surface region. Returns seedR<0 if no samples.
func cvSeedColor(rp, gp, bp []float64, w, h int) (mr, mg, mb, spread float64) {
	y0 := h - h*15/100
	if y0 < 0 {
		y0 = 0
	}
	if y0 >= h {
		y0 = h - 1
	}
	x0 := w * 20 / 100
	x1 := w - x0
	if x0 >= x1 {
		x0, x1 = 0, w
	}

	n := (h - y0) * (x1 - x0)
	if n <= 0 {
		return -1, 0, 0, 0
	}
	rs := make([]float64, 0, n)
	gs := make([]float64, 0, n)
	bs := make([]float64, 0, n)
	for y := y0; y < h; y++ {
		row := y * w
		for x := x0; x < x1; x++ {
			idx := row + x
			rs = append(rs, rp[idx])
			gs = append(gs, gp[idx])
			bs = append(bs, bp[idx])
		}
	}
	if len(rs) == 0 {
		return -1, 0, 0, 0
	}
	mr = cvMedian(rs)
	mg = cvMedian(gs)
	mb = cvMedian(bs)

	// Robust spread: mean per-channel standard deviation about the median.
	sr := cvStd(rs, mr)
	sg := cvStd(gs, mg)
	sb := cvStd(bs, mb)
	spread = (sr + sg + sb) / 3.0
	return mr, mg, mb, spread
}

// cvMedian returns the median of v (modifies a copy via sort on the slice; caller-owned).
func cvMedian(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	c := make([]float64, len(v))
	copy(c, v)
	sort.Float64s(c)
	n := len(c)
	if n%2 == 1 {
		return c[n/2]
	}
	return (c[n/2-1] + c[n/2]) / 2.0
}

// cvStd returns the standard deviation of v about center.
func cvStd(v []float64, center float64) float64 {
	if len(v) == 0 {
		return 0
	}
	var sum float64
	for _, x := range v {
		d := x - center
		sum += d * d
	}
	return math.Sqrt(sum / float64(len(v)))
}

// cvSobel computes a per-pixel gradient magnitude on the grayscale luma of the RGB planes
// (3x3 Sobel). Borders are 0. Row-major, length w*h.
func cvSobel(rp, gp, bp []float64, w, h int) []float64 {
	// Luma (BT.601-ish) for gradient.
	lum := make([]float64, w*h)
	for i := 0; i < w*h; i++ {
		lum[i] = 0.299*rp[i] + 0.587*gp[i] + 0.114*bp[i]
	}
	grad := make([]float64, w*h)
	if w < 3 || h < 3 {
		return grad
	}
	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			tl := lum[(y-1)*w+(x-1)]
			t := lum[(y-1)*w+x]
			tr := lum[(y-1)*w+(x+1)]
			l := lum[y*w+(x-1)]
			rr := lum[y*w+(x+1)]
			bl := lum[(y+1)*w+(x-1)]
			bb := lum[(y+1)*w+x]
			br := lum[(y+1)*w+(x+1)]
			gx := (tr + 2*rr + br) - (tl + 2*l + bl)
			gy := (bl + 2*bb + br) - (tl + 2*t + tr)
			grad[y*w+x] = math.Sqrt(gx*gx + gy*gy)
		}
	}
	return grad
}

// cvSelectSurface runs 4-connectivity connected components over the surface-like map and
// returns a mask of the union of large, border-touching components (the support surface).
// Small specks and interior-only blobs are dropped. Returns nil if nothing qualifies.
func cvSelectSurface(like []bool, w, h int) []bool {
	n := w * h
	labels := make([]int32, n)
	for i := range labels {
		labels[i] = -1
	}
	type comp struct {
		size   int
		border bool
	}
	var comps []comp
	stack := make([]int, 0, 256)

	for start := 0; start < n; start++ {
		if !like[start] || labels[start] >= 0 {
			continue
		}
		id := int32(len(comps))
		c := comp{}
		stack = stack[:0]
		stack = append(stack, start)
		labels[start] = id
		for len(stack) > 0 {
			p := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			c.size++
			px := p % w
			py := p / w
			if px == 0 || py == 0 || px == w-1 || py == h-1 {
				c.border = true
			}
			// 4-connected neighbors.
			if px > 0 {
				q := p - 1
				if like[q] && labels[q] < 0 {
					labels[q] = id
					stack = append(stack, q)
				}
			}
			if px < w-1 {
				q := p + 1
				if like[q] && labels[q] < 0 {
					labels[q] = id
					stack = append(stack, q)
				}
			}
			if py > 0 {
				q := p - w
				if like[q] && labels[q] < 0 {
					labels[q] = id
					stack = append(stack, q)
				}
			}
			if py < h-1 {
				q := p + w
				if like[q] && labels[q] < 0 {
					labels[q] = id
					stack = append(stack, q)
				}
			}
		}
		comps = append(comps, c)
	}

	if len(comps) == 0 {
		return nil
	}

	// Keep border-touching components that are reasonably large (>= 1% of the image), to
	// union fragmented surface regions while rejecting small border specks. If none touch the
	// border, fall back to the single largest component.
	minSize := n / 100
	if minSize < 1 {
		minSize = 1
	}
	keep := make([]bool, len(comps))
	anyBorderKept := false
	for i, c := range comps {
		if c.border && c.size >= minSize {
			keep[i] = true
			anyBorderKept = true
		}
	}
	if !anyBorderKept {
		// Fall back to the single largest component (it may not touch the border).
		best, bestSize := -1, 0
		for i, c := range comps {
			if c.size > bestSize {
				best, bestSize = i, c.size
			}
		}
		if best < 0 {
			return nil
		}
		keep[best] = true
	}

	out := make([]bool, n)
	for i := 0; i < n; i++ {
		if l := labels[i]; l >= 0 && keep[l] {
			out[i] = true
		}
	}
	return out
}
