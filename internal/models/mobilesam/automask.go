package mobilesam

// Automatic Mask Generator (AMG) for MobileSAM.
//
// When no bbox/point prompt is given, autoSegment tiles the image with a 16×16 grid
// of foreground point prompts, calls the decoder once per point (encoder runs once,
// shared), filters by predicted IoU quality, then suppresses duplicates via pixel-IoU
// NMS. This mirrors the SAM automatic_mask_generator pipeline.
//
// Parallelism: all 256 decoder calls are launched as goroutines; they serialise on
// the engine session's internal mutex, but tensor allocation and result parsing
// genuinely overlap. On CPU expect ~20–60 s; on GPU ~0.5–2 s.

import (
	"fmt"
	"image"
	"sort"
	"sync"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

const (
	autoGridSize    = 16   // N×N grid → N² decoder calls
	autoMinIoU      = 0.85 // discard masks below this predicted IoU
	autoDedupeIoU   = 0.70 // suppress mask if pixel-IoU with a better mask exceeds this
	autoMinAreaFrac = 1e-4 // discard masks covering < 0.01% of image pixels (noise)
	autoMaxAreaFrac = 0.95 // discard masks covering > 95% of image (background)
)

type aCandidate struct {
	bin  []bool // h×w binary mask (row-major)
	conf float64
	bbox [4]float64
	h, w int
}

type amgResult struct {
	cand  aCandidate
	valid bool
	err   error
}

// autoSegment runs the AMG pipeline and returns all surviving masks sorted by
// descending confidence. Returns an empty slice (not an error) if nothing passes
// the quality filters.
func autoSegment(
	img image.Image,
	embedding engine.Tensor,
	scale float64,
	decRun func(map[string]engine.Tensor) ([]engine.Tensor, error),
	decOutNames []string,
) ([]models.Mask, error) {
	origW := img.Bounds().Dx()
	origH := img.Bounds().Dy()
	totalPx := origW * origH
	minArea := max(1, int(float64(totalPx)*autoMinAreaFrac))
	maxArea := int(float64(totalPx) * autoMaxAreaFrac)

	const N = autoGridSize
	allResults := make([]amgResult, N*N)

	var wg sync.WaitGroup
	for j := 0; j < N; j++ {
		for i := 0; i < N; i++ {
			idx := j*N + i
			wg.Add(1)
			go func(i, j, idx int) {
				defer wg.Done()
				allResults[idx] = runGridPoint(
					i, j, origW, origH, minArea, maxArea,
					scale, embedding, decRun, decOutNames,
				)
			}(i, j, idx)
		}
	}
	wg.Wait()

	var cands []aCandidate
	for _, r := range allResults {
		if r.err != nil {
			return nil, r.err
		}
		if r.valid {
			cands = append(cands, r.cand)
		}
	}

	if len(cands) == 0 {
		return []models.Mask{}, nil
	}

	// Greedy IoU NMS: keep higher-confidence masks, suppress heavily overlapping ones.
	sort.Slice(cands, func(a, b int) bool { return cands[a].conf > cands[b].conf })
	keep := make([]bool, len(cands))
	for i := range keep {
		keep[i] = true
	}
	for i := 0; i < len(cands); i++ {
		if !keep[i] {
			continue
		}
		for j := i + 1; j < len(cands); j++ {
			if !keep[j] {
				continue
			}
			if binIoU(cands[i].bin, cands[j].bin) > autoDedupeIoU {
				keep[j] = false
			}
		}
	}

	out := make([]models.Mask, 0, len(cands))
	for i, c := range cands {
		if !keep[i] {
			continue
		}
		out = append(out, models.Mask{
			RLE:  encodeRLEColumnMajor(c.bin, c.h, c.w),
			BBox: c.bbox,
			Conf: c.conf,
		})
	}
	return out, nil
}

// runGridPoint runs a single decoder call for grid cell (i,j) and returns a candidate.
func runGridPoint(
	i, j, origW, origH, minArea, maxArea int,
	scale float64,
	embedding engine.Tensor,
	decRun func(map[string]engine.Tensor) ([]engine.Tensor, error),
	decOutNames []string,
) amgResult {
	cx := (float64(i) + 0.5) / float64(autoGridSize) * float64(origW)
	cy := (float64(j) + 0.5) / float64(autoGridSize) * float64(origH)
	px := float32(cx * scale)
	py := float32(cy * scale)

	zeros := make([]float32, 256*256)
	dec := map[string]engine.Tensor{
		"image_embeddings": embedding,
		"point_coords":     engine.F32([]float32{px, py, 0, 0}, 1, 2, 2),
		"point_labels":     engine.F32([]float32{1, -1}, 1, 2),
		"mask_input":       engine.F32(zeros, 1, 1, 256, 256),
		"has_mask_input":   engine.F32([]float32{0}, 1),
		"orig_im_size":     engine.F32([]float32{float32(origH), float32(origW)}, 2),
	}
	outs, err := decRun(dec)
	if err != nil {
		return amgResult{err: fmt.Errorf("mobilesam: amg decoder at (%d,%d): %w", i, j, err)}
	}
	maskT, iouT := pickMaskAndIoU(decOutNames, outs)
	if maskT == nil {
		return amgResult{}
	}

	bestCh, conf := bestChannel(maskT, iouT)
	if conf < autoMinIoU {
		return amgResult{}
	}

	mh := int(maskT.Dim(2))
	mw := int(maskT.Dim(3))
	off := bestCh * mh * mw
	bin := make([]bool, mh*mw)
	area := 0
	minX, minY, maxX, maxY := mw, mh, -1, -1
	for y := 0; y < mh; y++ {
		for x := 0; x < mw; x++ {
			if maskT.Data[off+y*mw+x] > 0 {
				bin[y*mw+x] = true
				area++
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

	if area < minArea || area > maxArea {
		return amgResult{}
	}

	var bbox [4]float64
	if maxX >= 0 {
		bbox = [4]float64{
			float64(minX), float64(minY),
			float64(maxX-minX+1), float64(maxY-minY+1),
		}
	}
	return amgResult{
		cand:  aCandidate{bin: bin, conf: conf, bbox: bbox, h: mh, w: mw},
		valid: true,
	}
}

// bestChannel returns the index and predicted IoU of the best mask channel.
// Used by both runGridPoint and maskToResult.
func bestChannel(maskT, iouT *engine.Tensor) (int, float64) {
	n := int(maskT.Dim(1))
	best, score := 0, float64(0)
	if iouT != nil && len(iouT.Data) > 0 {
		bestF := float32(-1e30)
		lim := n
		if len(iouT.Data) < lim {
			lim = len(iouT.Data)
		}
		for i := 0; i < lim; i++ {
			if iouT.Data[i] > bestF {
				bestF = iouT.Data[i]
				best = i
				score = float64(bestF)
			}
		}
	}
	return best, score
}

// binIoU computes pixel-level IoU between two binary masks of the same length.
func binIoU(a, b []bool) float64 {
	if len(a) != len(b) {
		return 0
	}
	inter, union := 0, 0
	for i := range a {
		if a[i] || b[i] {
			union++
		}
		if a[i] && b[i] {
			inter++
		}
	}
	if union == 0 {
		return 0
	}
	return float64(inter) / float64(union)
}
