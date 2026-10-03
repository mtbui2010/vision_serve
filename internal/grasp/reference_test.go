package grasp

import (
	"math"
	"reflect"
	"sort"
	"testing"

	"visionserve/pkg/api"
)

// fromMaskReference is FromMask exactly as it was before the top-K / allocation rewrite (P8),
// kept verbatim so the rewrite is held to it bit for bit.
func fromMaskReference(mask Bitmap, p Params) []api.Grasp {
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

// The rewrite must return exactly what the original did — same grasps, same order, same floats —
// for every cap, on masks with ties (rectangle, disk) and on the many-candidate star.
func TestFromMaskMatchesReference(t *testing.T) {
	masks := map[string]Bitmap{
		"rect": filledRect(100, 80, 30, 30, 40, 20),
		"disk": filledDisk(120, 120, 60, 60, 35),
		"star": filledStar(400, 400, 200, 200, 5, 180, 70),
		"tiny": filledRect(20, 20, 8, 8, 3, 3),
	}
	for name, m := range masks {
		for _, k := range []int{0, 1, 3, 20, 100000} {
			p := starParams()
			p.Dmin = 8
			p.MaxGrasps = k
			want := fromMaskReference(m, p)
			got := FromMask(m, p)
			if !reflect.DeepEqual(got, want) {
				t.Errorf("%s cap=%d: got %d grasps, want %d (or contents differ)", name, k, len(got), len(want))
			}
		}
	}
	_ = math.Pi
}

// BenchmarkFromMaskStarReference is the pre-rewrite search on the same mask, for comparison
// against BenchmarkFromMaskStar in one run.
func BenchmarkFromMaskStarReference(b *testing.B) {
	mask := filledStar(400, 400, 200, 200, 5, 180, 70)
	for _, k := range []int{0, 20} {
		p := starParams()
		p.MaxGrasps = k
		b.Run(map[int]string{0: "uncapped", 20: "top20"}[k], func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				fromMaskReference(mask, p)
			}
		})
	}
}
