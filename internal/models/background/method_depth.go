package background

import (
	"image"
	"math"
	"math/rand"

	"visionserve/internal/models"
)

// backgroundDepth (method=depth) — fits the dominant support surface (table / floor) as a
// 3D plane to the MiDaS depth/disparity map and returns its pixels as the background mask.
//
// MiDaS emits RELATIVE inverse depth (disparity): LARGER = NEARER to the camera. A 3D plane
// is affine in inverse depth, so the support surface's disparity is well modeled by
//
//	d(u,v) ≈ a*u + b*v + c
//
// We RANSAC-fit (a,b,c) over the disparity map, treat the plane inliers as the support
// surface, and exclude raised objects (which sit NEARER → disparity ABOVE the plane →
// positive residual → outliers). The result is a row-major []bool of length w*h at the
// ORIGINAL image resolution. Returns (nil, nil) when no clear surface is found.
func (m *backgroundModel) backgroundDepth(img image.Image, prompt models.Prompt, r models.Runner) ([]bool, error) {
	d, dw, dh, err := m.runDepth(img, prompt, r)
	if err != nil {
		return nil, err
	}
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	if w <= 0 || h <= 0 || dw <= 0 || dh <= 0 || len(d) < dw*dh {
		return nil, nil
	}

	// An EXTERNAL depth map (real RGB-D) has true geometry, so a large clean plane is valid
	// — skip the monocular degenerate guard. It may also carry NaN (invalid) pixels.
	external := prompt.Depth != nil
	valid := 0
	for i := 0; i < dw*dh; i++ {
		if !math.IsNaN(float64(d[i])) {
			valid++
		}
	}
	if valid < 3 {
		return nil, nil
	}

	// Adaptive inlier tolerance from the robust scale (MAD) of the (valid) disparity map.
	const (
		ransacIters  = 200
		minTol       = 1e-4
		minInlierPct = 12.0 // below this fraction of the image → "no clear surface"
		maxInlierPct = 70.0 // above this → the global plane is degenerate (no distinct surface)
	)
	mad := depthMAD(d)
	tol := 1.5 * mad
	// Guard against a degenerate (near-constant) disparity map: floor off the spread.
	if spread := depthSpread(d); tol < spread*1e-3 {
		tol = spread * 1e-3
	}
	if tol < minTol {
		tol = minTol
	}

	// RANSAC for the dominant plane, then refit by least squares on its inliers.
	a, b, c, ok := depthRANSAC(d, dw, dh, tol, ransacIters)
	if !ok {
		return nil, nil
	}
	a, b, c = depthRefit(d, dw, dh, a, b, c, tol)

	// Build the inlier (support-surface) mask at depth resolution.
	mask := make([]bool, dw*dh)
	inliers := 0
	for v := 0; v < dh; v++ {
		row := v * dw
		for u := 0; u < dw; u++ {
			pred := a*float64(u) + b*float64(v) + c
			if math.Abs(float64(d[row+u])-pred) <= tol {
				mask[row+u] = true
				inliers++
			}
		}
	}

	// Coverage is measured against VALID pixels (so sensor holes don't deflate it).
	if float64(inliers)/float64(valid)*100.0 < minInlierPct {
		return nil, nil // no clear surface
	}

	// Keep only the largest 4-connected component so the surface is one clean region,
	// then re-check coverage.
	mask = depthLargestComponent(mask, dw, dh)
	cov := float64(depthCount(mask)) / float64(valid) * 100.0
	if cov < minInlierPct {
		return nil, nil // fragmented fit → not a real surface
	}
	if !external && cov > maxInlierPct {
		// The dominant plane swallows most of the frame — typical of monocular RELATIVE
		// depth on a cluttered far-background scene, where the whole image is ~one affine
		// gradient and no distinct support surface stands out. Don't return a garbage
		// near-full mask; signal "no clear surface" (use method=cv/automask, or feed a real
		// RGB-D depth where the plane is well-defined).
		return nil, nil
	}

	return upsampleMask(mask, dw, dh, w, h), nil
}

// depthMAD returns the median absolute deviation of the VALID (non-NaN) values about their
// median — a robust scale.
func depthMAD(d []float32) float64 {
	med := depthMedian(d)
	dev := make([]float64, 0, len(d))
	for _, v := range d {
		f := float64(v)
		if !math.IsNaN(f) {
			dev = append(dev, math.Abs(f-med))
		}
	}
	if len(dev) == 0 {
		return 0
	}
	return depthMedianF(dev)
}

// depthSpread returns max-min over the VALID (non-NaN) values.
func depthSpread(d []float32) float64 {
	first := true
	var mn, mx float64
	for _, v := range d {
		f := float64(v)
		if math.IsNaN(f) {
			continue
		}
		if first {
			mn, mx, first = f, f, false
			continue
		}
		if f < mn {
			mn = f
		}
		if f > mx {
			mx = f
		}
	}
	if first {
		return 0
	}
	return mx - mn
}

func depthMedian(d []float32) float64 {
	cp := make([]float64, 0, len(d))
	for _, v := range d {
		f := float64(v)
		if !math.IsNaN(f) {
			cp = append(cp, f)
		}
	}
	if len(cp) == 0 {
		return 0
	}
	return depthMedianF(cp)
}

// depthMedianF returns the median of vals. It copies before sorting so the caller's slice
// order is preserved.
func depthMedianF(vals []float64) float64 {
	n := len(vals)
	if n == 0 {
		return 0
	}
	cp := make([]float64, n)
	copy(cp, vals)
	depthSort(cp)
	if n%2 == 1 {
		return cp[n/2]
	}
	return 0.5 * (cp[n/2-1] + cp[n/2])
}

// depthSort sorts a float64 slice ascending in place (dependency-free quicksort).
func depthSort(a []float64) {
	depthQuicksort(a, 0, len(a)-1)
}

func depthQuicksort(a []float64, lo, hi int) {
	for lo < hi {
		// Median-of-three pivot to avoid worst case on sorted/duplicate-heavy data.
		mid := lo + (hi-lo)/2
		if a[mid] < a[lo] {
			a[lo], a[mid] = a[mid], a[lo]
		}
		if a[hi] < a[lo] {
			a[lo], a[hi] = a[hi], a[lo]
		}
		if a[hi] < a[mid] {
			a[mid], a[hi] = a[hi], a[mid]
		}
		pivot := a[mid]
		i, j := lo, hi
		for i <= j {
			for a[i] < pivot {
				i++
			}
			for a[j] > pivot {
				j--
			}
			if i <= j {
				a[i], a[j] = a[j], a[i]
				i++
				j--
			}
		}
		// Recurse into the smaller partition, loop on the larger (bounded stack depth).
		if j-lo < hi-i {
			depthQuicksort(a, lo, j)
			lo = i
		} else {
			depthQuicksort(a, i, hi)
			hi = j
		}
	}
}

// depthRANSAC fits d ≈ a*u + b*v + c by repeatedly solving the plane through 3 random
// pixels and keeping the model with the most inliers. Uses a FIXED seed for reproducibility.
func depthRANSAC(d []float32, dw, dh int, tol float64, iters int) (a, b, c float64, ok bool) {
	n := dw * dh
	if n < 3 {
		return 0, 0, 0, false
	}
	rng := rand.New(rand.NewSource(1))

	bestInliers := -1
	for it := 0; it < iters; it++ {
		// Pick 3 distinct pixel indices.
		i0 := rng.Intn(n)
		i1 := rng.Intn(n)
		i2 := rng.Intn(n)
		if i0 == i1 || i0 == i2 || i1 == i2 {
			continue
		}
		// Skip samples on invalid (NaN) pixels — they cannot define the plane.
		if math.IsNaN(float64(d[i0])) || math.IsNaN(float64(d[i1])) || math.IsNaN(float64(d[i2])) {
			continue
		}
		ca, cb, cc, solvable := depthSolvePlane(
			float64(i0%dw), float64(i0/dw), float64(d[i0]),
			float64(i1%dw), float64(i1/dw), float64(d[i1]),
			float64(i2%dw), float64(i2/dw), float64(d[i2]),
		)
		if !solvable {
			continue
		}
		cnt := depthCountInliers(d, dw, dh, ca, cb, cc, tol)
		if cnt > bestInliers {
			bestInliers, a, b, c, ok = cnt, ca, cb, cc, true
		}
	}
	return a, b, c, ok
}

// depthSolvePlane solves the 3x3 system for the plane through three (u,v,d) samples.
// Returns ok=false if the three pixels are colinear (singular system).
func depthSolvePlane(u0, v0, d0, u1, v1, d1, u2, v2, d2 float64) (a, b, c float64, ok bool) {
	// | u0 v0 1 | |a|   |d0|
	// | u1 v1 1 | |b| = |d1|
	// | u2 v2 1 | |c|   |d2|
	det := u0*(v1-v2) - v0*(u1-u2) + (u1*v2 - u2*v1)
	if math.Abs(det) < 1e-9 {
		return 0, 0, 0, false
	}
	// Cramer's rule.
	da := d0*(v1-v2) - v0*(d1-d2) + (d1*v2 - d2*v1)
	db := u0*(d1-d2) - d0*(u1-u2) + (u1*d2 - u2*d1)
	dc := u0*(v1*d2-v2*d1) - v0*(u1*d2-u2*d1) + d0*(u1*v2-u2*v1)
	return da / det, db / det, dc / det, true
}

func depthCountInliers(d []float32, dw, dh int, a, b, c, tol float64) int {
	cnt := 0
	for v := 0; v < dh; v++ {
		row := v * dw
		for u := 0; u < dw; u++ {
			pred := a*float64(u) + b*float64(v) + c
			if math.Abs(float64(d[row+u])-pred) <= tol {
				cnt++
			}
		}
	}
	return cnt
}

// depthRefit re-estimates (a,b,c) by ordinary least squares over the current inlier set.
// Falls back to the input model if the inlier normal matrix is singular.
func depthRefit(d []float32, dw, dh int, a, b, c, tol float64) (float64, float64, float64) {
	// Accumulate the normal equations A^T A x = A^T y over inliers, with A rows = [u, v, 1].
	var suu, suv, su, svv, sv, s float64
	var sdu, sdv, sd float64
	for vy := 0; vy < dh; vy++ {
		row := vy * dw
		for ux := 0; ux < dw; ux++ {
			pred := a*float64(ux) + b*float64(vy) + c
			val := float64(d[row+ux])
			if math.IsNaN(val) || math.Abs(val-pred) > tol {
				continue
			}
			fu, fv := float64(ux), float64(vy)
			suu += fu * fu
			suv += fu * fv
			su += fu
			svv += fv * fv
			sv += fv
			s++
			sdu += val * fu
			sdv += val * fv
			sd += val
		}
	}
	if s < 3 {
		return a, b, c
	}
	// Solve the 3x3 symmetric normal system via Cramer's rule.
	na, nb, nc, ok := depthSolve3(
		suu, suv, su, sdu,
		suv, svv, sv, sdv,
		su, sv, s, sd,
	)
	if !ok {
		return a, b, c
	}
	return na, nb, nc
}

// depthSolve3 solves the 3x3 linear system [m..]·x = [r0,r1,r2] via Cramer's rule.
func depthSolve3(
	m00, m01, m02, r0,
	m10, m11, m12, r1,
	m20, m21, m22, r2 float64,
) (x0, x1, x2 float64, ok bool) {
	det := m00*(m11*m22-m12*m21) - m01*(m10*m22-m12*m20) + m02*(m10*m21-m11*m20)
	if math.Abs(det) < 1e-12 {
		return 0, 0, 0, false
	}
	d0 := r0*(m11*m22-m12*m21) - m01*(r1*m22-m12*r2) + m02*(r1*m21-m11*r2)
	d1 := m00*(r1*m22-m12*r2) - r0*(m10*m22-m12*m20) + m02*(m10*r2-r1*m20)
	d2 := m00*(m11*r2-r1*m21) - m01*(m10*r2-r1*m20) + r0*(m10*m21-m11*m20)
	return d0 / det, d1 / det, d2 / det, true
}

// depthLargestComponent returns a mask containing only the largest 4-connected component
// of the input. Uses an iterative flood fill (no recursion) over a scratch stack.
func depthLargestComponent(mask []bool, w, h int) []bool {
	label := make([]int32, w*h)
	stack := make([]int, 0, w*h)

	bestLabel := int32(0)
	bestSize := 0
	cur := int32(0)
	for start := 0; start < w*h; start++ {
		if !mask[start] || label[start] != 0 {
			continue
		}
		cur++
		size := 0
		stack = stack[:0]
		stack = append(stack, start)
		label[start] = cur
		for len(stack) > 0 {
			p := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			size++
			px, py := p%w, p/w
			if px > 0 {
				if q := p - 1; mask[q] && label[q] == 0 {
					label[q] = cur
					stack = append(stack, q)
				}
			}
			if px < w-1 {
				if q := p + 1; mask[q] && label[q] == 0 {
					label[q] = cur
					stack = append(stack, q)
				}
			}
			if py > 0 {
				if q := p - w; mask[q] && label[q] == 0 {
					label[q] = cur
					stack = append(stack, q)
				}
			}
			if py < h-1 {
				if q := p + w; mask[q] && label[q] == 0 {
					label[q] = cur
					stack = append(stack, q)
				}
			}
		}
		if size > bestSize {
			bestSize, bestLabel = size, cur
		}
	}

	if bestLabel == 0 {
		return mask
	}
	out := make([]bool, w*h)
	for i := range out {
		if label[i] == bestLabel {
			out[i] = true
		}
	}
	return out
}

func depthCount(mask []bool) int {
	n := 0
	for _, v := range mask {
		if v {
			n++
		}
	}
	return n
}
