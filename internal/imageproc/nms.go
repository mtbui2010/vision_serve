package imageproc

import (
	"math"
	"sort"

	"visionserve/pkg/api"
)

// NMS (Non-Maximum Suppression) is the standard greedy IoU-based suppression, for models
// that NEED it. Boxes are [x,y,w,h]; suppression only happens within the same Class.
// The result is sorted by descending Conf (ties keep input order). It keeps exactly the
// boxes the textbook all-pairs greedy loop keeps (TestNMSMatchesNaive).
//
// Complexity: a kept box only needs to be compared with boxes LATER in score order that
// are not yet suppressed (an earlier box was either kept — and already compared — or
// suppressed). For large inputs a uniform grid index restricts that to boxes sharing a
// grid cell (IoU > 0 needs a positive-area intersection, so such boxes always share a
// cell). The old version rescanned all n indices for every kept box: 2.1 s for 16.8k
// scattered SCRFD-sized proposals (BenchmarkNMS16800).
//
// NOTE (CLAUDE.md): RF-DETR is NMS-free — do NOT call NMS for RF-DETR. This function
// is a shared utility for anchor/grid-based models.
func NMS(dets []api.Detection, iouThresh float64) []api.Detection {
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
	boxes := make([]nmsBox, len(order))
	classID := map[string]int{}
	for p, i := range order {
		b := dets[i].BBox
		id, ok := classID[dets[i].Class]
		if !ok {
			id = len(classID)
			classID[dets[i].Class] = id
		}
		boxes[p] = nmsBox{b[0], b[1], b[0] + b[2], b[1] + b[3], b[2] * b[3], id}
	}

	suppressed := make([]bool, len(order)) // indexed by position in score order
	var keep []api.Detection
	if len(boxes) < nmsGridMin {
		for p := range boxes {
			if suppressed[p] {
				continue
			}
			keep = append(keep, dets[order[p]])
			for q := p + 1; q < len(boxes); q++ {
				if !suppressed[q] && boxes[p].overlaps(&boxes[q], iouThresh) {
					suppressed[q] = true
				}
			}
		}
		return keep
	}

	g := newNMSGrid(boxes)
	for p := range boxes {
		if suppressed[p] {
			continue
		}
		keep = append(keep, dets[order[p]])
		cx0, cy0, cx1, cy1 := g.cellRange(&boxes[p])
		for cy := cy0; cy <= cy1; cy++ {
			for cx := cx0; cx <= cx1; cx++ {
				for _, q := range g.cells[cy*g.nx+cx] {
					if int(q) > p && !suppressed[q] && boxes[p].overlaps(&boxes[q], iouThresh) {
						suppressed[q] = true
					}
				}
			}
		}
	}
	return keep
}

// nmsGridMin: below this many boxes the plain triangular scan is cheaper than building
// the grid.
const nmsGridMin = 512

type nmsBox struct {
	x1, y1, x2, y2, area float64
	cls                  int
}

// overlaps reports same class && IoU > thr (same arithmetic as iou()).
func (a *nmsBox) overlaps(b *nmsBox, thr float64) bool {
	if a.cls != b.cls {
		return false
	}
	iw := min(a.x2, b.x2) - max(a.x1, b.x1)
	if iw <= 0 {
		return false
	}
	ih := min(a.y2, b.y2) - max(a.y1, b.y1)
	if ih <= 0 {
		return false
	}
	inter := iw * ih
	union := a.area + b.area - inter
	return union > 0 && inter/union > thr
}

// nmsGrid buckets box positions by the uniform cells their extent covers. Cell size is
// the mean box side, so a typical box covers ~4 cells; the grid is capped at
// nmsGridMaxCells per axis.
type nmsGrid struct {
	minX, minY, cell float64
	nx, ny           int
	cells            [][]int32
}

const nmsGridMaxCells = 256

func newNMSGrid(boxes []nmsBox) *nmsGrid {
	minX, minY := math.Inf(1), math.Inf(1)
	maxX, maxY := math.Inf(-1), math.Inf(-1)
	var sumSide float64
	for i := range boxes {
		b := &boxes[i]
		minX, minY = min(minX, b.x1), min(minY, b.y1)
		maxX, maxY = max(maxX, b.x2), max(maxY, b.y2)
		sumSide += math.Abs(b.x2-b.x1) + math.Abs(b.y2-b.y1)
	}
	cell := sumSide / float64(2*len(boxes))
	extent := max(maxX-minX, maxY-minY)
	if c := extent / nmsGridMaxCells; cell < c {
		cell = c
	}
	if !(cell > 0) || math.IsInf(cell, 0) { // degenerate (all points) or non-finite input
		cell = max(extent, 1)
		if math.IsInf(cell, 0) || math.IsNaN(cell) {
			cell = 1
		}
	}
	g := &nmsGrid{minX: minX, minY: minY, cell: cell}
	g.nx = clampCell(int((maxX-minX)/cell)+1, nmsGridMaxCells+1)
	g.ny = clampCell(int((maxY-minY)/cell)+1, nmsGridMaxCells+1)
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
func (g *nmsGrid) cellRange(b *nmsBox) (cx0, cy0, cx1, cy1 int) {
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

// iou computes the Intersection-over-Union of two boxes in [x,y,w,h] form.
func iou(a, b [4]float64) float64 {
	ax1, ay1, ax2, ay2 := a[0], a[1], a[0]+a[2], a[1]+a[3]
	bx1, by1, bx2, by2 := b[0], b[1], b[0]+b[2], b[1]+b[3]

	ix1, iy1 := max(ax1, bx1), max(ay1, by1)
	ix2, iy2 := min(ax2, bx2), min(ay2, by2)
	iw, ih := ix2-ix1, iy2-iy1
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	union := a[2]*a[3] + b[2]*b[3] - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
