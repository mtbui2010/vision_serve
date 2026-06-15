package background

import (
	"math"
	"testing"
)

// TestDepthPlaneFitExcludesRaisedObject synthesizes a 32×32 disparity ramp (a tilted plane)
// with a raised rectangular "object" patch and asserts the RANSAC+refit plane fit recovers
// the plane as inliers while the raised object is excluded as outliers.
func TestDepthPlaneFitExcludesRaisedObject(t *testing.T) {
	const (
		w, h = 32, 32
		// Ground-truth plane d = a*u + b*v + c. MiDaS disparity: larger = nearer.
		gtA, gtB, gtC = 0.5, 0.25, 3.0
		// Raised object is NEARER → disparity well above the plane.
		objBump = 20.0
	)
	// Object rectangle (a clear raised patch, well under the inlier threshold area).
	objX0, objY0, objX1, objY1 := 20, 4, 27, 11

	inObject := func(u, v int) bool {
		return u >= objX0 && u <= objX1 && v >= objY0 && v <= objY1
	}

	d := make([]float32, w*h)
	for v := 0; v < h; v++ {
		for u := 0; u < w; u++ {
			val := gtA*float64(u) + gtB*float64(v) + gtC
			if inObject(u, v) {
				val += objBump
			}
			d[v*w+u] = float32(val)
		}
	}

	// Tolerance derived the same way the method does (MAD-based), but small enough that the
	// object (bump=20) is far outside it.
	mad := depthMAD(d)
	tol := 1.5 * mad
	if tol < 1e-4 {
		tol = 1e-4
	}
	// Sanity: the object bump must exceed the tolerance, else the test would be vacuous.
	if objBump <= tol {
		t.Fatalf("test setup: objBump %.3f not above tol %.3f", objBump, tol)
	}

	a, b, c, ok := depthRANSAC(d, w, h, tol, 200)
	if !ok {
		t.Fatal("depthRANSAC failed to find any plane")
	}
	a, b, c = depthRefit(d, w, h, a, b, c, tol)

	// Recovered plane should be close to ground truth.
	if math.Abs(a-gtA) > 1e-2 || math.Abs(b-gtB) > 1e-2 || math.Abs(c-gtC) > 1e-1 {
		t.Fatalf("plane fit off: got a=%.4f b=%.4f c=%.4f want a=%.4f b=%.4f c=%.4f",
			a, b, c, gtA, gtB, gtC)
	}

	// Classify every pixel; plane pixels must be inliers, object pixels outliers.
	planeInliers, planeTotal := 0, 0
	objInliers, objTotal := 0, 0
	for v := 0; v < h; v++ {
		for u := 0; u < w; u++ {
			pred := a*float64(u) + b*float64(v) + c
			isIn := math.Abs(float64(d[v*w+u])-pred) <= tol
			if inObject(u, v) {
				objTotal++
				if isIn {
					objInliers++
				}
			} else {
				planeTotal++
				if isIn {
					planeInliers++
				}
			}
		}
	}

	if planeInliers != planeTotal {
		t.Errorf("expected all %d plane pixels to be inliers, got %d", planeTotal, planeInliers)
	}
	if objInliers != 0 {
		t.Errorf("expected 0 object pixels to be inliers (raised → outliers), got %d/%d",
			objInliers, objTotal)
	}
}

// TestDepthLargestComponent checks that a fragmented mask is reduced to its largest
// 4-connected region.
func TestDepthLargestComponent(t *testing.T) {
	const w, h = 8, 8
	mask := make([]bool, w*h)
	// Big block: rows 0..4, cols 0..4 (25 px).
	for v := 0; v <= 4; v++ {
		for u := 0; u <= 4; u++ {
			mask[v*w+u] = true
		}
	}
	// Small isolated speck at (7,7).
	mask[7*w+7] = true

	out := depthLargestComponent(mask, w, h)
	if got := depthCount(out); got != 25 {
		t.Fatalf("largest component size = %d, want 25", got)
	}
	if out[7*w+7] {
		t.Error("isolated speck should have been dropped")
	}
	if !out[0] || !out[4*w+4] {
		t.Error("big block pixels should be retained")
	}
}
