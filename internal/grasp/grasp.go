// Package grasp implements a pure-Go analytic planar parallel-jaw grasp
// synthesizer. It is a faithful port of the Python `mask2grasps` routine from
// the pyinterfaces project (instances.mask2grasps + utils.mask2normalmap /
// sampling / twopoints2theta + grasppose.GraspGroup width/theta derivation).
//
// The whole pipeline is math-only (stdlib `math`): a binary mask is turned into
// a boundary normal map via a small hand-rolled Sobel-like convolution, the
// nonzero-normal boundary pixels are decimated on a polar grid about the
// centroid, every finger combination is enumerated, and each candidate is
// scored with a force-closure / friction-cone test. Results are returned as
// api.Grasp in the mask's pixel coordinates (= original image coordinates).
//
// No cgo, no ONNX Runtime, no third-party deps — it builds and tests without
// libonnxruntime.so.
package grasp

import (
	"math"
	"sort"

	"visionserve/pkg/api"
)

// Bitmap is a binary mask. Data is row-major, length W*H; a pixel is "set" when
// the corresponding entry is true.
type Bitmap struct {
	W, H int
	Data []bool
}

// at reports whether pixel (x,y) is set; out-of-bounds reads as false (the
// convolution treats the border as background, matching cv2.filter2D's default
// of an implicit zero/replicate boundary closely enough for boundary normals).
func (b Bitmap) at(x, y int) bool {
	if x < 0 || y < 0 || x >= b.W || y >= b.H {
		return false
	}
	return b.Data[y*b.W+x]
}

// Params configures the analytic grasp search. Use DefaultParams for sane values.
type Params struct {
	NFingers     int     // number of contacts (default 2 = antipodal pair)
	FrictionCoef float64 // friction cone coefficient (default 0.5)
	Dmin, Dmax   float64 // gripper opening bounds in pixels
	AngleStepDeg int     // boundary decimation angular step in degrees (default 5)
	StridePx     int     // boundary decimation radial stride in pixels (default 5)
	MaxGrasps    int     // cap on returned grasps (0 = all)
}

// DefaultParams returns the default search parameters. They mirror the Python
// defaults: nfingers=2, friction cone ~0.5, opening 10..150 px, 5deg/5px
// decimation. Friction_coef is 0.5 here (a tighter cone than the Python default
// of 1.0) to keep the antipodal test meaningful on synthetic masks; callers can
// raise it toward 1.0 to admit more candidates.
func DefaultParams() Params {
	return Params{
		NFingers:     2,
		FrictionCoef: 0.5,
		Dmin:         10,
		Dmax:         150,
		AngleStepDeg: 5,
		StridePx:     5,
		MaxGrasps:    0,
	}
}

// vec2 is a 2D vector.
type vec2 struct{ X, Y float64 }

func (a vec2) sub(b vec2) vec2    { return vec2{a.X - b.X, a.Y - b.Y} }
func (a vec2) norm() float64      { return math.Hypot(a.X, a.Y) }
func (a vec2) dot(b vec2) float64 { return a.X*b.X + a.Y*b.Y }

// boundaryPoint is a decimated boundary contact: location + unit outward normal.
type boundaryPoint struct {
	loc vec2 // pixel location (x,y), integer-valued
	nrm vec2 // unit boundary normal
}

// normalRadius is the Sobel-like kernel radius (matches utils.mask2normalmap r=2).
const normalRadius = 2

// boundingBox returns the tight inclusive bbox [x0,y0]..[x1,y1] of the set pixels.
// ok is false when the mask is empty.
func boundingBox(b Bitmap) (x0, y0, x1, y1 int, ok bool) {
	x0, y0, x1, y1 = b.W, b.H, -1, -1
	for y := 0; y < b.H; y++ {
		row := y * b.W
		for x := 0; x < b.W; x++ {
			if b.Data[row+x] {
				if x < x0 {
					x0 = x
				}
				if x > x1 {
					x1 = x
				}
				if y < y0 {
					y0 = y
				}
				if y > y1 {
					y1 = y
				}
			}
		}
	}
	return x0, y0, x1, y1, x1 >= 0
}

// collectBoundary returns the nonzero-normal boundary pixels as unit-normal
// contacts (no decimation yet). Equivalent to the torch.nonzero(normalmap_norm)
// step in mask2grasps, using the same separable Sobel-like kernels as
// utils.mask2normalmap with r=2:
//
//	kernel_y rows [0..r-1]=+1, row r=0, rows [r+1..2r]=-1; kernel_x = transpose.
//
// cv2.filter2D correlates (no kernel flip), so we correlate too. Only boundary
// pixels carry a nonzero normal, and every such pixel lies within r of a set
// pixel — so we compute normals ONLY inside the mask's bounding box expanded by r
// (clamped to the image). This is exact (pixels outside have zero normal) while
// avoiding a full-frame O(W·H·(2r+1)²) sweep for a small object in a large image.
func collectBoundary(b Bitmap) []boundaryPoint {
	const eps = 1e-10
	const r = normalRadius
	x0, y0, x1, y1, ok := boundingBox(b)
	if !ok {
		return nil
	}
	// expand by the kernel radius and clamp to the image bounds.
	if x0 -= r; x0 < 0 {
		x0 = 0
	}
	if y0 -= r; y0 < 0 {
		y0 = 0
	}
	if x1 += r; x1 >= b.W {
		x1 = b.W - 1
	}
	if y1 += r; y1 >= b.H {
		y1 = b.H - 1
	}

	var pts []boundaryPoint
	for y := y0; y <= y1; y++ {
		for x := x0; x <= x1; x++ {
			var sx, sy float64
			for ky := -r; ky <= r; ky++ {
				for kx := -r; kx <= r; kx++ {
					if !b.at(x+kx, y+ky) {
						continue
					}
					// kernel_y weight depends on the row offset ky.
					switch {
					case ky < 0:
						sy += 1 // top rows -> +1
					case ky > 0:
						sy -= 1 // bottom rows -> -1
					}
					// kernel_x = transpose -> weight depends on column offset kx.
					switch {
					case kx < 0:
						sx += 1
					case kx > 0:
						sx -= 1
					}
				}
			}
			n := math.Hypot(sx, sy)
			if n <= 0 {
				continue
			}
			pts = append(pts, boundaryPoint{
				loc: vec2{float64(x), float64(y)},
				nrm: vec2{sx / (n + eps), sy / (n + eps)},
			})
		}
	}
	return pts
}

// decimate thins the boundary contacts on a polar grid about their centroid,
// matching the intent of utils.sampling(deg, stride): the boundary is binned by
// polar angle (degStep) and radius (stridePx) about the centroid, and ONE
// representative is kept per (angle-bin, radius-bin) cell. This keeps the
// combinatorics tractable exactly as the Python decimation does, but is robust
// and deterministic on a discrete pixel boundary (where the Python's exact
// "rho % stride < 1" float test would be fragile). Points snap to the integer
// polar grid (Xc + Rho*[cos,sin]) like the Python's int64 cast.
//
// Faithfulness note: the Python keeps points whose polar angle/radius land near
// a multiple of the step; here we keep one-per-cell. Both produce a sparse polar
// lattice of boundary contacts about the centroid — the same downstream effect.
func decimate(pts []boundaryPoint, degStep, stridePx int) []boundaryPoint {
	if len(pts) == 0 {
		return pts
	}
	if degStep <= 0 {
		degStep = 5
	}
	if stridePx <= 0 {
		stridePx = 5
	}
	// centroid of the locations
	var cx, cy float64
	for _, p := range pts {
		cx += p.loc.X
		cy += p.loc.Y
	}
	cx /= float64(len(pts))
	cy /= float64(len(pts))

	type cell struct{ a, r int }
	seen := make(map[cell]bool)
	out := make([]boundaryPoint, 0, len(pts))
	for _, p := range pts {
		dx := p.loc.X - cx
		dy := p.loc.Y - cy
		rho := math.Hypot(dx, dy)
		// Phi in [0,360): torch_arctan2(dy,dx) in degrees, +180, wrapped.
		phi := pyMod(arctan2Py(dy, dx)*180.0/math.Pi+180.0, 360.0)
		c := cell{
			a: int(phi) / degStep,
			r: int(rho) / stridePx,
		}
		if seen[c] {
			continue
		}
		seen[c] = true
		// snap onto integer polar grid (matches X = Xc + Rho*[cos,sin], int64)
		phiR := (phi - 180.0) * math.Pi / 180.0
		nx := math.Trunc(cx + rho*math.Cos(phiR))
		ny := math.Trunc(cy + rho*math.Sin(phiR))
		out = append(out, boundaryPoint{
			loc: vec2{nx, ny},
			nrm: p.nrm,
		})
	}
	return out
}

// arctan2Py mirrors the Python torch_arctan2 lambda used by utils.sampling:
//   arctan(y/(x+eps)) + (x<0)*pi*sign(y)
// This differs from a standard atan2 only in its branch handling but reproduces
// the exact binning the Python decimation uses.
func arctan2Py(y, x float64) float64 {
	const eps = 1e-10
	v := math.Atan(y / (x + eps))
	if x < 0 {
		v += math.Pi * sign(y)
	}
	return v
}

func sign(v float64) float64 {
	switch {
	case v > 0:
		return 1
	case v < 0:
		return -1
	default:
		return 0
	}
}

// pyMod is Python's % (result has the sign of the divisor, non-negative for
// positive divisor). Go's math.Mod can return negatives.
func pyMod(a, b float64) float64 {
	m := math.Mod(a, b)
	if m < 0 {
		m += b
	}
	return m
}

// FromMask runs the analytic antipodal force-closure search on the mask's
// boundary normals and returns parallel-jaw grasps in the mask's pixel
// coordinates (= original image coordinates). Port of pyinterfaces
// instances.mask2grasps. Returns an empty (non-nil-safe) slice when the mask is
// empty or no candidate satisfies the force-closure constraints; never panics.
func FromMask(mask Bitmap, p Params) []api.Grasp {
	if p.NFingers < 2 {
		p.NFingers = 2
	}
	if p.AngleStepDeg <= 0 {
		p.AngleStepDeg = 5
	}
	if p.StridePx <= 0 {
		p.StridePx = 5
	}
	if mask.W <= 0 || mask.H <= 0 || len(mask.Data) != mask.W*mask.H {
		return nil
	}

	pts := decimate(collectBoundary(mask), p.AngleStepDeg, p.StridePx)
	n := len(pts)
	if n < p.NFingers {
		return nil
	}

	// object centroid over the decimated locations (Python: mean of `locs`).
	var center vec2
	for _, pt := range pts {
		center.X += pt.loc.X
		center.Y += pt.loc.Y
	}
	center.X /= float64(n)
	center.Y /= float64(n)

	kf := math.Cos(math.Atan(p.FrictionCoef)) // friction-cone threshold

	type cand struct {
		pts     []int   // contact indices
		minDist float64 // min pairwise finger distance
		insAvg  float64 // mean force-closure inner product
		dcenter float64 // |grasp center - object center|
	}
	var cands []cand

	// enumerate all nfingers-combinations of boundary points.
	combinations(n, p.NFingers, func(idx []int) {
		// grasp center = mean of contacts (Xmean).
		var xmean vec2
		for _, i := range idx {
			xmean.X += pts[i].loc.X
			xmean.Y += pts[i].loc.Y
		}
		xmean.X /= float64(p.NFingers)
		xmean.Y /= float64(p.NFingers)

		// force directions: each finger toward the finger centroid.
		// force-closure: F·n > kf for every contact (friction cone).
		minD := math.Inf(1)
		insSum := 0.0
		ok := true
		for _, i := range idx {
			d := pts[i].loc.sub(xmean)
			dn := d.norm()
			if dn < 1e-12 {
				ok = false
				break
			}
			f := vec2{d.X / (dn + 1e-10), d.Y / (dn + 1e-10)}
			ins := f.dot(pts[i].nrm)
			if ins <= kf {
				ok = false
				break
			}
			insSum += ins
		}
		if !ok {
			return
		}

		// pairwise finger distances (gripper widths); all must be in (dmin,dmax).
		for a := 0; a < len(idx); a++ {
			for b := a + 1; b < len(idx); b++ {
				dist := pts[idx[a]].loc.sub(pts[idx[b]].loc).norm()
				if !(p.Dmin < dist && dist < p.Dmax) {
					ok = false
				}
				if dist < minD {
					minD = dist
				}
			}
		}
		if !ok {
			return
		}

		cands = append(cands, cand{
			pts:     append([]int(nil), idx...),
			minDist: minD,
			insAvg:  insSum / float64(p.NFingers),
			dcenter: xmean.sub(center).norm(),
		})
	})

	if len(cands) == 0 {
		return nil
	}

	// normalize Dmin and Dcenter by their maxima (Python: /max+eps).
	const eps = 1e-10
	var maxMinDist, maxDcenter float64
	for _, c := range cands {
		if c.minDist > maxMinDist {
			maxMinDist = c.minDist
		}
		if c.dcenter > maxDcenter {
			maxDcenter = c.dcenter
		}
	}

	// weights = [0.4,0.3,0.15,0.05,0.05,0] normalized; contact_score & obj_score
	// default to 1 (no contact map). orientation term weight is 0 -> dropped.
	w := [6]float64{0.4, 0.3, 0.15, 0.05, 0.05, 0.0}
	var wsum float64
	for _, x := range w {
		wsum += x
	}
	for i := range w {
		w[i] /= wsum
	}
	const contactScore = 1.0
	const objScore = 1.0

	out := make([]api.Grasp, 0, len(cands))
	for _, c := range cands {
		dminN := c.minDist / (maxMinDist + eps)
		dcenterN := c.dcenter / (maxDcenter + eps)
		score := w[0]*c.insAvg +
			w[1]*contactScore +
			w[2]*objScore +
			w[3]*(1-dminN) +
			w[4]*(1-dcenterN)
		if score < 0 {
			score = 0
		} else if score > 1 {
			score = 1
		}

		// 2-finger grasp geometry: center = midpoint, theta = atan2 of finger
		// vector (twopoints2theta first column), width = pixel distance.
		p0 := pts[c.pts[0]].loc
		p1 := pts[c.pts[1]].loc
		v := p1.sub(p0)
		theta := math.Atan2(v.Y, v.X)
		width := v.norm()
		cx := (p0.X + p1.X) / 2
		cy := (p0.Y + p1.Y) / 2

		out = append(out, api.Grasp{
			X:       cx,
			Y:       cy,
			Theta:   theta,
			Width:   width,
			Quality: score,
		})
	}

	// sort by quality descending (stable for deterministic output).
	sort.SliceStable(out, func(i, j int) bool {
		return out[i].Quality > out[j].Quality
	})

	if p.MaxGrasps > 0 && len(out) > p.MaxGrasps {
		out = out[:p.MaxGrasps]
	}
	return out
}

// combinations enumerates all k-combinations of [0,n) in lexicographic order,
// invoking fn with the current index set (which it must not retain).
func combinations(n, k int, fn func(idx []int)) {
	if k <= 0 || k > n {
		return
	}
	idx := make([]int, k)
	for i := range idx {
		idx[i] = i
	}
	for {
		fn(idx)
		// advance to the next combination.
		i := k - 1
		for i >= 0 && idx[i] == n-k+i {
			i--
		}
		if i < 0 {
			return
		}
		idx[i]++
		for j := i + 1; j < k; j++ {
			idx[j] = idx[j-1] + 1
		}
	}
}
