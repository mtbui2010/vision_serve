package mobilesam

// Automatic Mask Generator (AMG) for MobileSAM.
//
// When no bbox/point prompt is given, autoSegment tiles the image with an N×N grid of
// foreground point prompts, calls the decoder once per point (encoder runs once,
// shared), filters by predicted IoU quality and area, then suppresses duplicates via
// pixel-IoU NMS. This mirrors the SAM automatic_mask_generator pipeline.
//
// Two resolutions (memory/time): the decoder upsamples its masks to orig_im_size INSIDE
// the graph, so asking it for full-resolution masks on every grid point costs N² full-res
// float maps (a 3200×2400 image: 30 MB per call, ~4.8 GB peak RSS before this split).
// Instead:
//
//  1. Filter pass — every grid point is decoded with orig_im_size set to a small WORK frame
//     (long side autoWorkLongSide, i.e. the native 256-px resolution of SAM's low-res mask
//     logits over the 1024 input). Quality, area (as a FRACTION of the frame) and the
//     pixel-IoU NMS all run on these small bitmaps; the NMS first rejects pairs whose
//     bboxes cannot reach the IoU threshold and counts intersections only inside the bbox
//     overlap.
//  2. Final pass — only the KEPT grid points are decoded again with the real orig_im_size,
//     so every returned mask is exactly what the decoder produces at original resolution
//     (bit-identical to the old full-res path for the same point prompt).
//
// When the image is already no larger than the work frame the filter-pass bitmaps ARE the
// original-resolution masks and the final pass is skipped.
//
// Parallelism: a fixed number of workers (not N² goroutines) feed the decoder session
// pool, which bounds the number of decoder outputs alive at once. All calls share ONE
// read-only zero mask_input buffer.

import (
	"fmt"
	"image"
	"math"
	"sort"
	"sync"

	"visionserve/internal/engine"
	"visionserve/internal/vision/mask"
)

const (
	autoGridSize     = 16   // N×N grid → N² decoder calls
	autoMinIoU       = 0.85 // discard masks below this predicted IoU
	autoDedupeIoU    = 0.70 // suppress mask if pixel-IoU with a better mask exceeds this
	autoMinAreaFrac  = 1e-4 // discard masks covering < 0.01% of image pixels (noise)
	autoMaxAreaFrac  = 0.95 // discard masks covering > 95% of image (background)
	autoWorkLongSide = 256  // filter-pass frame: SAM low-res logits are 256 px over the 1024 input
	autoWorkers      = 8    // concurrent decoder callers (decoder pool is 4; extra overlap Go-side work)
)

// zeroMaskInput is the shared, READ-ONLY all-zero mask_input [1,1,256,256] passed with
// has_mask_input=0 on every decoder call. ORT never writes to input buffers, so one buffer
// is safe to share across concurrent calls (like the shared image embedding). Never write
// to it.
var zeroMaskInput = make([]float32, 256*256)

func zeroMaskTensor() engine.Tensor { return engine.F32(zeroMaskInput, 1, 1, 256, 256) }

type aCandidate struct {
	bm   mask.Bitmap // binary mask in the frame it was decoded at
	ext  mask.Extent // its inclusive pixel extent + area
	conf float64
	// px, py: the grid point prompt in the decoder's resized-1024 space (re-used by the
	// final pass), idx: grid index (deterministic tie-break).
	px, py float32
	idx    int
}

type amgResult struct {
	cand  aCandidate
	valid bool
	err   error
}

// workFrame returns the filter-pass frame: the original size when its long side is
// already ≤ autoWorkLongSide, else the aspect-preserving downscale to that long side.
func workFrame(origW, origH int) (w, h int) {
	long := origW
	if origH > long {
		long = origH
	}
	if long <= autoWorkLongSide {
		return origW, origH
	}
	s := float64(autoWorkLongSide) / float64(long)
	w = int(math.Round(float64(origW) * s))
	h = int(math.Round(float64(origH) * s))
	if w < 1 {
		w = 1
	}
	if h < 1 {
		h = 1
	}
	return w, h
}

// areaBounds converts the area fractions to pixel bounds for a w×h frame.
func areaBounds(w, h int) (minArea, maxArea int) {
	total := w * h
	return max(1, int(float64(total)*autoMinAreaFrac)), int(float64(total) * autoMaxAreaFrac)
}

// autoSegment runs the AMG pipeline and returns all surviving masks (as raw
// bitmaps at original-image resolution) sorted by descending confidence. Returns an
// empty slice (not an error) if nothing passes the quality filters. Callers encode
// to RLE (models.Mask) only when needed for the API response.
func autoSegment(
	img image.Image,
	embedding engine.Tensor,
	scale float64,
	decRun func(map[string]engine.Tensor) ([]engine.Tensor, error),
	decOutNames []string,
	gridSize int,
) ([]MaskBitmap, error) {
	origW := img.Bounds().Dx()
	origH := img.Bounds().Dy()
	workW, workH := workFrame(origW, origH)
	minArea, maxArea := areaBounds(workW, workH)

	N := gridSize
	if N <= 0 {
		N = autoGridSize
	}

	// 1) Filter pass at the work frame.
	results := make([]amgResult, N*N)
	parallelFor(N*N, autoWorkers, func(idx int) {
		i, j := idx%N, idx/N
		cx := (float64(i) + 0.5) / float64(N) * float64(origW)
		cy := (float64(j) + 0.5) / float64(N) * float64(origH)
		px, py := float32(cx*scale), float32(cy*scale)
		c, ok, err := decodePoint(px, py, workW, workH, embedding, decRun, decOutNames)
		if err != nil {
			results[idx] = amgResult{err: fmt.Errorf("mobilesam: amg decoder at (%d,%d): %w", i, j, err)}
			return
		}
		if !ok || c.conf < autoMinIoU || c.ext.Area < minArea || c.ext.Area > maxArea {
			return
		}
		c.idx = idx
		results[idx] = amgResult{cand: c, valid: true}
	})

	var cands []aCandidate
	for _, r := range results {
		if r.err != nil {
			return nil, r.err
		}
		if r.valid {
			cands = append(cands, r.cand)
		}
	}
	if len(cands) == 0 {
		return []MaskBitmap{}, nil
	}

	// 2) Greedy NMS on the work-frame bitmaps.
	kept := nmsCandidates(cands, autoDedupeIoU)

	// 3) Final pass: original-resolution masks for the kept points only.
	if workW == origW && workH == origH {
		out := make([]MaskBitmap, 0, len(kept))
		for _, c := range kept {
			out = append(out, c.bitmap())
		}
		return out, nil
	}

	fullMin, fullMax := areaBounds(origW, origH)
	finals := make([]amgResult, len(kept))
	parallelFor(len(kept), autoWorkers, func(k int) {
		c, ok, err := decodePoint(kept[k].px, kept[k].py, origW, origH, embedding, decRun, decOutNames)
		if err != nil {
			finals[k] = amgResult{err: fmt.Errorf("mobilesam: amg full-res decoder: %w", err)}
			return
		}
		// Same area gate as the filter pass, now on the exact full-res mask.
		if !ok || c.ext.Area < fullMin || c.ext.Area > fullMax {
			return
		}
		c.conf = kept[k].conf // identical prompt → identical iou_predictions; keep the ranking value
		finals[k] = amgResult{cand: c, valid: true}
	})
	out := make([]MaskBitmap, 0, len(kept))
	for _, r := range finals {
		if r.err != nil {
			return nil, r.err
		}
		if r.valid {
			out = append(out, r.cand.bitmap())
		}
	}
	return out, nil
}

// parallelFor runs fn(0..n-1) on at most `workers` goroutines.
func parallelFor(n, workers int, fn func(i int)) {
	if workers > n {
		workers = n
	}
	var wg sync.WaitGroup
	next := make(chan int)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				fn(i)
			}
		}()
	}
	for i := 0; i < n; i++ {
		next <- i
	}
	close(next)
	wg.Wait()
}

// decodePoint runs one single-point decoder call with orig_im_size = (frameH, frameW)
// and thresholds the best mask channel (logit > 0) in that frame. ok=false when the
// decoder returned no mask tensor or the mask is empty.
func decodePoint(
	px, py float32,
	frameW, frameH int,
	embedding engine.Tensor,
	decRun func(map[string]engine.Tensor) ([]engine.Tensor, error),
	decOutNames []string,
) (aCandidate, bool, error) {
	dec := map[string]engine.Tensor{
		"image_embeddings": embedding,
		"point_coords":     engine.F32([]float32{px, py, 0, 0}, 1, 2, 2),
		"point_labels":     engine.F32([]float32{1, -1}, 1, 2),
		"mask_input":       zeroMaskTensor(),
		"has_mask_input":   engine.F32([]float32{0}, 1),
		"orig_im_size":     engine.F32([]float32{float32(frameH), float32(frameW)}, 2),
	}
	outs, err := decRun(dec)
	if err != nil {
		return aCandidate{}, false, err
	}
	maskT, iouT := pickMaskAndIoU(decOutNames, outs)
	if maskT == nil || len(maskT.Shape) != 4 {
		return aCandidate{}, false, nil
	}
	c := thresholdBest(maskT, iouT)
	c.px, c.py = px, py
	return c, c.ext.Area > 0, nil
}

// thresholdBest binarizes the best-IoU channel of a [1,M,H,W] mask-logit tensor.
func thresholdBest(maskT, iouT *engine.Tensor) aCandidate {
	bestCh, conf := bestChannel(maskT, iouT)
	mh := int(maskT.Dim(2))
	mw := int(maskT.Dim(3))
	bm, ext := mask.ThresholdExtent(maskT.Data, bestCh*mh*mw, mh, mw, 0)
	return aCandidate{bm: bm, ext: ext, conf: conf}
}

// bitmap converts a candidate to the public MaskBitmap ([x,y,w,h] bbox in its frame).
func (c aCandidate) bitmap() MaskBitmap {
	return MaskBitmap{Data: c.bm.Data, W: c.bm.W, H: c.bm.H, BBox: c.ext.XYWH(), Conf: c.conf}
}

// nmsCandidates keeps higher-confidence candidates and suppresses any later one whose
// pixel IoU with a kept candidate exceeds thresh. Ties in confidence break by grid index
// (deterministic). Pairs are rejected cheaply first: disjoint bboxes → IoU 0, and
// IoU ≤ min(area)/max(area) (one mask fully inside the other is the best case).
func nmsCandidates(cands []aCandidate, thresh float64) []aCandidate {
	sort.SliceStable(cands, func(a, b int) bool {
		if cands[a].conf != cands[b].conf {
			return cands[a].conf > cands[b].conf
		}
		return cands[a].idx < cands[b].idx
	})
	suppressed := make([]bool, len(cands))
	kept := make([]aCandidate, 0, len(cands))
	for i := range cands {
		if suppressed[i] {
			continue
		}
		kept = append(kept, cands[i])
		for j := i + 1; j < len(cands); j++ {
			a, b := &cands[i], &cands[j]
			if !suppressed[j] && mask.IoUMayExceed(a.ext, b.ext, thresh) &&
				mask.PixelIoU(a.bm, a.ext, b.bm, b.ext) > thresh {
				suppressed[j] = true
			}
		}
	}
	return kept
}

// bestChannel returns the index and predicted IoU of the best mask channel.
// Used by both the AMG and maskToBitmap.
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
