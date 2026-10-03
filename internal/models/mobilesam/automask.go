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
//
// Final-pass memory: each full-resolution call returns one float logit map per mask channel
// (30 MB at 3200×2400) that is thresholded to a bitmap and dropped at once. When the Runner can
// write outputs into caller buffers (models.IntoRunner), each final-pass worker owns ONE such
// buffer, mapped outside the Go heap and unmapped when the pass ends (scratch_unix.go), and the
// decoder writes into it on every call: no ORT arena block, no Go copy and no Go garbage per
// call. When the caller does not keep the bitmaps (Infer's RLE, InferMasksEach), each worker
// also thresholds into one reused bitmap. The pass runs one worker per decoder session (not
// autoWorkers): a worker past the pool size would only hold its buffers while it waits.
//
// Measured on CPU, 3200×2400, fresh server, 5 identical requests (VmHWM): mobile-sam automask
// 1.63 → 0.80 GB, grasp 2.06 → 0.87 GB, background method=automask 1.98 → 0.98 GB (0.36 GB
// of it the loaded models, 0.54 GB with MiDaS); outputs byte-identical, latency unchanged
// within noise. Part of that is the engine's memory-pattern
// change (ort.go); the rest is the buffers above and the streaming consumers.

import (
	"fmt"
	"image"
	"math"
	"sort"
	"strings"
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
	autoFinalWorkers = 4    // final-pass callers: one per decoder session, each owning one full-res buffer
)

// decodeFunc runs the decoder with its inputs bound by name. into, when non-nil, names outputs
// to write into caller-owned buffers (models.IntoRunner); a decodeFunc that cannot do that
// ignores it and returns fresh tensors with the same values.
type decodeFunc func(inputs, into map[string]engine.Tensor) ([]engine.Tensor, error)

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
	// nch: mask channels of the decoder output it came from (1 single, 4 multi export).
	nch int
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

// amgOut is one final-pass result after emit has converted it.
type amgOut[T any] struct {
	v     T
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
	decRun decodeFunc,
	decOutNames []string,
	gridSize int,
) ([]MaskBitmap, error) {
	return autoSegmentAs(img, embedding, scale, decRun, decOutNames, gridSize, keepBitmap, false)
}

// keepBitmap is the identity emit: the caller wants the bitmaps themselves.
func keepBitmap(b MaskBitmap) MaskBitmap { return b }

// autoSegmentAs is autoSegment with every surviving mask passed through emit as soon as it is
// final, in the same order. With emit = MaskBitmap.ToMask the full-resolution bitmap of each
// mask is dropped right after its RLE is made, instead of all of them being held until the end
// (a 3200×2400 image keeps ~60 masks of 7.7 MB each).
//
// borrow says emit does not keep the bitmap's Data after it returns (ToMask, a streaming
// consumer): the final pass then thresholds every mask of a worker into one reused buffer
// instead of a new 7.7 MB bitmap per mask.
func autoSegmentAs[T any](
	img image.Image,
	embedding engine.Tensor,
	scale float64,
	decRun decodeFunc,
	decOutNames []string,
	gridSize int,
	emit func(MaskBitmap) T,
	borrow bool,
) ([]T, error) {
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
		c, ok, err := decodePoint(px, py, workW, workH, embedding, decRun, decOutNames, nil, nil)
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
		return []T{}, nil
	}

	// 2) Greedy NMS on the work-frame bitmaps.
	kept := nmsCandidates(cands, autoDedupeIoU)

	// 3) Final pass: original-resolution masks for the kept points only.
	if workW == origW && workH == origH {
		out := make([]T, 0, len(kept))
		for _, c := range kept {
			out = append(out, emit(c.bitmap()))
		}
		return out, nil
	}

	fullMin, fullMax := areaBounds(origW, origH)
	finals := make([]amgOut[T], len(kept))
	// The decoder writes the masks into a worker's buffer only when they are found by name and
	// the filter pass showed it renders them at the frame orig_im_size asks for, so the
	// buffer's shape is exactly the one the run produces.
	maskName := maskOutputName(decOutNames)
	nch := kept[0].nch
	intoOK := maskName != "" && nch > 0 && kept[0].bm.W == workW && kept[0].bm.H == workH
	logits := make([][]float32, autoFinalWorkers) // per worker: the decoder's mask output (off-heap)
	frees := make([]func(), autoFinalWorkers)     // unmaps logits[w]
	bits := make([][]bool, autoFinalWorkers)      // per worker: the thresholded mask (borrow only)
	parallelForW(len(kept), autoFinalWorkers, func(w, k int) {
		var into map[string]engine.Tensor
		if intoOK {
			if logits[w] == nil {
				logits[w], frees[w] = allocLogits(nch * origH * origW)
			}
			into = map[string]engine.Tensor{maskName: engine.F32(logits[w], 1, int64(nch), int64(origH), int64(origW))}
		}
		var dst []bool
		if borrow {
			if bits[w] == nil {
				bits[w] = make([]bool, origH*origW)
			}
			dst = bits[w]
		}
		c, ok, err := decodePoint(kept[k].px, kept[k].py, origW, origH, embedding, decRun, decOutNames, into, dst)
		if err != nil {
			finals[k] = amgOut[T]{err: fmt.Errorf("mobilesam: amg full-res decoder: %w", err)}
			return
		}
		// Same area gate as the filter pass, now on the exact full-res mask.
		if !ok || c.ext.Area < fullMin || c.ext.Area > fullMax {
			return
		}
		c.conf = kept[k].conf // identical prompt → identical iou_predictions; keep the ranking value
		finals[k] = amgOut[T]{v: emit(c.bitmap()), valid: true}
	})
	// Every worker has returned, and with it every decoder run writing into its buffer.
	for _, free := range frees {
		if free != nil {
			free()
		}
	}
	out := make([]T, 0, len(kept))
	for _, r := range finals {
		if r.err != nil {
			return nil, r.err
		}
		if r.valid {
			out = append(out, r.v)
		}
	}
	return out, nil
}

// parallelFor runs fn(0..n-1) on at most `workers` goroutines.
func parallelFor(n, workers int, fn func(i int)) {
	parallelForW(n, workers, func(_, i int) { fn(i) })
}

// parallelForW is parallelFor that also tells fn which worker (0..workers-1) runs item i, so a
// worker can reuse state of its own across its items.
func parallelForW(n, workers int, fn func(w, i int)) {
	if workers > n {
		workers = n
	}
	var wg sync.WaitGroup
	next := make(chan int)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := range next {
				fn(w, i)
			}
		}(w)
	}
	for i := 0; i < n; i++ {
		next <- i
	}
	close(next)
	wg.Wait()
}

// decodePoint runs one single-point decoder call with orig_im_size = (frameH, frameW)
// and thresholds the best mask channel (logit > 0) in that frame. ok=false when the
// decoder returned no mask tensor or the mask is empty. into is passed to decRun (caller
// buffers for the mask output, or nil); the candidate never aliases it. dst, when non-nil
// and large enough, receives the thresholded bitmap instead of a new allocation.
func decodePoint(
	px, py float32,
	frameW, frameH int,
	embedding engine.Tensor,
	decRun decodeFunc,
	decOutNames []string,
	into map[string]engine.Tensor,
	dst []bool,
) (aCandidate, bool, error) {
	dec := map[string]engine.Tensor{
		"image_embeddings": embedding,
		"point_coords":     engine.F32([]float32{px, py, 0, 0}, 1, 2, 2),
		"point_labels":     engine.F32([]float32{1, -1}, 1, 2),
		"mask_input":       zeroMaskTensor(),
		"has_mask_input":   engine.F32([]float32{0}, 1),
		"orig_im_size":     engine.F32([]float32{float32(frameH), float32(frameW)}, 2),
	}
	outs, err := decRun(dec, into)
	if err != nil {
		return aCandidate{}, false, err
	}
	maskT, iouT := pickMaskAndIoU(decOutNames, outs)
	if maskT == nil || len(maskT.Shape) != 4 {
		return aCandidate{}, false, nil
	}
	c := thresholdBestInto(maskT, iouT, dst)
	c.px, c.py = px, py
	c.nch = int(maskT.Dim(1))
	return c, c.ext.Area > 0, nil
}

// maskOutputName is the decoder output pickMaskAndIoU takes as the masks BY NAME, or "" when
// none is named like one (pickMaskAndIoU then picks by shape, and the final pass does not
// pass a buffer for it).
func maskOutputName(names []string) string {
	name := ""
	for _, n := range names {
		l := strings.ToLower(n)
		switch {
		case strings.Contains(l, "iou"):
		case l == "masks" || (strings.Contains(l, "mask") && !strings.Contains(l, "low_res")):
			name = n // the last match wins, as in pickMaskAndIoU
		}
	}
	return name
}

// thresholdBest binarizes the best-IoU channel of a [1,M,H,W] mask-logit tensor.
func thresholdBest(maskT, iouT *engine.Tensor) aCandidate {
	return thresholdBestInto(maskT, iouT, nil)
}

// thresholdBestInto is thresholdBest writing the bitmap into dst when it holds H*W pixels.
func thresholdBestInto(maskT, iouT *engine.Tensor, dst []bool) aCandidate {
	bestCh, conf := bestChannel(maskT, iouT)
	mh := int(maskT.Dim(2))
	mw := int(maskT.Dim(3))
	var bm mask.Bitmap
	var ext mask.Extent
	if len(dst) >= mh*mw {
		bm, ext = mask.ThresholdExtentInto(dst, maskT.Data, bestCh*mh*mw, mh, mw, 0)
	} else {
		bm, ext = mask.ThresholdExtent(maskT.Data, bestCh*mh*mw, mh, mw, 0)
	}
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
