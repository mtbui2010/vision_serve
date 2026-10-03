// Package nms is greedy non-maximum suppression over api.Detection boxes, for the models
// that NEED it (anchor/grid/patch detectors: SCRFD, OWLv2, …).
//
// NOTE (CLAUDE.md): DETR-style models (RF-DETR, RT-DETR) are NMS-free — never call this
// for them.
package nms

import (
	"math"
	"sort"

	"visionserve/pkg/api"
)

// Options configures Detections.
type Options struct {
	// IoU is the suppression threshold: a box is suppressed when its overlap with an
	// already-kept, higher-scoring box is STRICTLY greater than IoU.
	IoU float64
	// ClassAgnostic lets boxes of different Class suppress each other. The default
	// (false) only suppresses within the same Class.
	ClassAgnostic bool
	// Containment also suppresses when the intersection covers more than IoU of the
	// SMALLER box (overlap = max(IoU, inter/min(areaA, areaB))), so a large box that
	// encloses a tight one is removed even when their plain IoU is low (OWLv2).
	Containment bool
	// TopK stops after keeping TopK boxes (0 = keep all). The result equals running the
	// full NMS and truncating to TopK.
	TopK int
}

// Detections is the standard greedy NMS. Boxes are [x,y,w,h]. The result is sorted by
// descending Conf (ties keep input order) and keeps exactly the boxes the textbook
// all-pairs greedy loop keeps (TestMatchesNaive).
//
// Complexity: a kept box only needs to be compared with boxes LATER in score order that
// are not yet suppressed (an earlier box was either kept — and already compared — or
// suppressed). For large inputs a uniform grid index restricts that to boxes sharing a
// grid cell (any suppression needs a positive-area intersection, so such boxes always
// share a cell). The old all-pairs version took 2.1 s for 16.8k scattered SCRFD-sized
// proposals (BenchmarkNMS16800).
func Detections(dets []api.Detection, o Options) []api.Detection {
	if len(dets) == 0 {
		return dets
	}
	order := make([]int, len(dets))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return dets[order[a]].Conf > dets[order[b]].Conf })

	// Flatten corners/areas/class ids in score order so the hot loop touches contiguous
	// memory and compares ints instead of strings.
	boxes := make([]box, len(order))
	classID := map[string]int{}
	for p, i := range order {
		b := dets[i].BBox
		id := 0
		if !o.ClassAgnostic {
			var ok bool
			if id, ok = classID[dets[i].Class]; !ok {
				id = len(classID)
				classID[dets[i].Class] = id
			}
		}
		boxes[p] = box{b[0], b[1], b[0] + b[2], b[1] + b[3], b[2] * b[3], id}
	}

	suppressed := make([]bool, len(order)) // indexed by position in score order
	var keep []api.Detection
	full := func() bool { return o.TopK > 0 && len(keep) >= o.TopK }
	var g *grid
	if len(boxes) >= gridMin {
		g = newGrid(boxes) // nil when the boxes' extent is not finite
	}
	if g == nil {
		for p := range boxes {
			if suppressed[p] {
				continue
			}
			keep = append(keep, dets[order[p]])
			if full() {
				break
			}
			for q := p + 1; q < len(boxes); q++ {
				if !suppressed[q] && boxes[p].overlaps(&boxes[q], o.IoU, o.Containment) {
					suppressed[q] = true
				}
			}
		}
		return keep
	}

	for p := range boxes {
		if suppressed[p] {
			continue
		}
		keep = append(keep, dets[order[p]])
		if full() {
			break
		}
		cx0, cy0, cx1, cy1 := g.cellRange(&boxes[p])
		for cy := cy0; cy <= cy1; cy++ {
			for cx := cx0; cx <= cx1; cx++ {
				for _, q := range g.cells[cy*g.nx+cx] {
					if int(q) > p && !suppressed[q] && boxes[p].overlaps(&boxes[q], o.IoU, o.Containment) {
						suppressed[q] = true
					}
				}
			}
		}
	}
	return keep
}

// gridMin: below this many boxes the plain triangular scan is cheaper than building the
// grid.
const gridMin = 512

type box struct {
	x1, y1, x2, y2, area float64
	cls                  int
}

// overlaps reports same class && (IoU > thr || (containment && inter/min(area) > thr)).
func (a *box) overlaps(b *box, thr float64, containment bool) bool {
	if a.cls != b.cls {
		return false
	}
	iw := fmin(a.x2, b.x2) - fmax(a.x1, b.x1)
	if iw <= 0 {
		return false
	}
	ih := fmin(a.y2, b.y2) - fmax(a.y1, b.y1)
	if ih <= 0 {
		return false
	}
	inter := iw * ih
	union := a.area + b.area - inter
	if union > 0 && inter/union > thr {
		return true
	}
	if containment {
		if m := fmin(a.area, b.area); m > 0 && inter/m > thr {
			return true
		}
	}
	return false
}

// grid buckets box positions by the uniform cells their extent covers. Cell size is the
// mean box side, so a typical box covers ~4 cells; the grid is capped at gridMaxCells per
// axis.
type grid struct {
	minX, minY, cell float64
	nx, ny           int
	cells            [][]int32
}

const gridMaxCells = 256

func newGrid(boxes []box) *grid {
	minX, minY := math.Inf(1), math.Inf(1)
	maxX, maxY := math.Inf(-1), math.Inf(-1)
	var sumSide float64
	for i := range boxes {
		b := &boxes[i]
		minX, minY = fmin(minX, b.x1), fmin(minY, b.y1)
		maxX, maxY = fmax(maxX, b.x2), fmax(maxY, b.y2)
		sumSide += math.Abs(b.x2-b.x1) + math.Abs(b.y2-b.y1)
	}
	// An infinite coordinate (a model emitting Inf) leaves no finite grid: int(+Inf) is
	// undefined, and such a box would sit in one cell while overlapping boxes in others. The
	// caller then uses the triangular scan, which handles any input.
	if spanX, spanY := maxX-minX, maxY-minY; math.IsInf(spanX, 0) || math.IsInf(spanY, 0) ||
		math.IsNaN(spanX) || math.IsNaN(spanY) {
		return nil
	}
	cell := sumSide / float64(2*len(boxes))
	extent := fmax(maxX-minX, maxY-minY)
	if c := extent / gridMaxCells; cell < c {
		cell = c
	}
	if !(cell > 0) || math.IsInf(cell, 0) { // degenerate (all points) or non-finite input
		cell = fmax(extent, 1)
		if math.IsInf(cell, 0) || math.IsNaN(cell) {
			cell = 1
		}
	}
	g := &grid{minX: minX, minY: minY, cell: cell}
	g.nx = clampCell(int((maxX-minX)/cell)+1, gridMaxCells+1)
	g.ny = clampCell(int((maxY-minY)/cell)+1, gridMaxCells+1)
	g.cells = make([][]int32, g.nx*g.ny)
	for i := range boxes {
		cx0, cy0, cx1, cy1 := g.cellRange(&boxes[i])
		for cy := cy0; cy <= cy1; cy++ {
			for cx := cx0; cx <= cx1; cx++ {
				g.cells[cy*g.nx+cx] = append(g.cells[cy*g.nx+cx], int32(i))
			}
		}
	}
	return g
}

// cellRange returns the inclusive cell range covered by b's closed extent (NaN / inverted
// boxes collapse to a clamped cell; they never overlap anything anyway).
func (g *grid) cellRange(b *box) (cx0, cy0, cx1, cy1 int) {
	cx0 = clampCell(int((b.x1-g.minX)/g.cell), g.nx)
	cx1 = clampCell(int((b.x2-g.minX)/g.cell), g.nx)
	cy0 = clampCell(int((b.y1-g.minY)/g.cell), g.ny)
	cy1 = clampCell(int((b.y2-g.minY)/g.cell), g.ny)
	return
}

func clampCell(v, n int) int {
	if v < 0 {
		return 0
	}
	if v >= n {
		return n - 1
	}
	return v
}

// fmax / fmin deliberately keep the "a > b ? a : b" semantics (NOT the builtins, which
// propagate NaN): the suppression decisions of degenerate boxes must not change.
func fmax(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func fmin(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
