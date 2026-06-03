package efficientsam

import (
	"fmt"
	"strconv"
	"strings"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// pointSet holds one decoder invocation's prompts in ORIGINAL image coordinates.
type pointSet struct {
	coords []float64 // flat [x0,y0, x1,y1, ...] in original image pixels
	labels []int64   // EfficientSAM labels: 2=box TL, 3=box BR, 1=fg, 0=bg
}

func (p pointSet) n() int { return len(p.labels) }

// batchedTensors returns the two decoder input tensors for EfficientSAM:
//
//   - batched_point_coords: [1, 1, N, 2] float32, coordinates scaled to 1024-space
//   - batched_point_labels: [1, 1, N]    int64
//
// The outer two batch dims (1, 1) follow the onnx-community/EfficientSAM export
// convention (batch=1, one prompt-set per call).
//
// TODO: verify the [1, 1, N, 2] / [1, 1, N] shape expectation against the actual ONNX.
// Some exports may use [1, N, 2] / [1, N] (no extra batch dim).
func (p pointSet) batchedTensors(scale float64) (coords, labels engine.Tensor) {
	n := p.n()
	coordData := make([]float32, n*2)
	for i, v := range p.coords {
		coordData[i] = float32(v * scale)
	}
	// EfficientSAM decoder expects batched_point_labels as float32 (verified against ONNX).
	labelData := make([]float32, n)
	for i, l := range p.labels {
		labelData[i] = float32(l)
	}

	coords = engine.F32(coordData, 1, 1, int64(n), 2)
	labels = engine.F32(labelData, 1, 1, int64(n))
	return
}

// promptToPointSets converts a Prompt to a slice of decoder invocation descriptors.
//
//   - Each box [x,y,w,h] → 2 points: top-left (label 2) + bottom-right (label 3).
//   - Points-only prompt → single set of those points.
//     EfficientSAM does not use a padding (0,0,-1) point; omit it.
func promptToPointSets(p models.Prompt) ([]pointSet, error) {
	var sets []pointSet
	for _, b := range p.Boxes {
		x, y, w, h := b[0], b[1], b[2], b[3]
		sets = append(sets, pointSet{
			coords: []float64{x, y, x + w, y + h},
			labels: []int64{2, 3},
		})
	}
	if len(p.Points) > 0 {
		ps := pointSet{}
		for _, pt := range p.Points {
			ps.coords = append(ps.coords, pt.X, pt.Y)
			ps.labels = append(ps.labels, int64(pt.Label))
		}
		sets = append(sets, ps)
	}
	if len(sets) == 0 {
		if p.Text != "" {
			return nil, fmt.Errorf("efficientsam: text prompts are not supported — "+
				"use a BOX or POINT prompt, e.g. --box x,y,w,h. "+
				"For text-driven segmentation use the 'grounded-sam' model with --prompt %q", p.Text)
		}
		return nil, fmt.Errorf("efficientsam: a prompt (box or point) is required — " +
			"EfficientSAM segments around a prompt, e.g. --box x,y,w,h")
	}
	return sets, nil
}

// pickBestMask locates the low_res_masks and iou_predictions tensors in the decoder
// output list, then selects the mask channel with the highest IoU score.
//
// EfficientSAM decoder outputs:
//   - low_res_masks:   [1, 1, 4, 256, 256] — 4 candidate masks at 256×256
//   - iou_predictions: [1, 1, 4]           — per-mask IoU scores
//
// Returns: selected 256×256 mask data (length 256*256), best IoU score, error.
//
// TODO: verify tensor names against the actual ONNX decoder file.
func pickBestMask(names []string, outs []engine.Tensor) (maskData []float32, bestIoU float64, err error) {
	var maskT, iouT *engine.Tensor

	for i := range outs {
		name := ""
		if i < len(names) {
			name = strings.ToLower(names[i])
		}
		switch {
		case strings.Contains(name, "iou"):
			iouT = &outs[i]
		case strings.Contains(name, "mask"):
			// Prefer low_res_masks (shape 5-D: [1,1,4,256,256]) over any other mask output.
			if maskT == nil || len(outs[i].Shape) == 5 {
				maskT = &outs[i]
			}
		}
	}

	// Fallback by shape if name matching failed.
	if maskT == nil {
		for i := range outs {
			if len(outs[i].Shape) == 5 {
				maskT = &outs[i]
				break
			}
		}
	}
	if iouT == nil {
		for i := range outs {
			if len(outs[i].Shape) == 3 {
				iouT = &outs[i]
				break
			}
		}
	}

	if maskT == nil {
		shapes := make([]string, len(outs))
		for i, t := range outs {
			shapes[i] = fmt.Sprint(t.Shape)
		}
		return nil, 0, fmt.Errorf("efficientsam: no mask tensor found in decoder outputs (shapes: %s)", strings.Join(shapes, ", "))
	}

	// Expected shape: [1, 1, 4, 256, 256] — but be lenient about leading batch dims.
	// Find numCandidates and mask spatial dims from the last 3 dims.
	rank := len(maskT.Shape)
	if rank < 3 {
		return nil, 0, fmt.Errorf("efficientsam: unexpected mask tensor rank %d, shape %v", rank, maskT.Shape)
	}
	numCandidates := int(maskT.Shape[rank-3])
	mH := int(maskT.Shape[rank-2])
	mW := int(maskT.Shape[rank-1])
	planeSize := mH * mW

	if numCandidates < 1 || mH <= 0 || mW <= 0 {
		return nil, 0, fmt.Errorf("efficientsam: degenerate mask shape %v", maskT.Shape)
	}

	// Find the best candidate by IoU score.
	best := 0
	bestScore := float64(0)
	if iouT != nil && len(iouT.Data) >= numCandidates {
		bestF := float32(-1e30)
		for i := 0; i < numCandidates; i++ {
			if iouT.Data[i] > bestF {
				bestF = iouT.Data[i]
				best = i
			}
		}
		bestScore = float64(bestF)
	}

	// Extract that candidate's mask plane.
	// Total elements before the [numCandidates, mH, mW] suffix may have leading batch dims.
	leadingElems := int(maskT.NumElements()) / (numCandidates * planeSize)
	_ = leadingElems // structural check; we always take the first batch

	off := best * planeSize
	if off+planeSize > len(maskT.Data) {
		return nil, 0, fmt.Errorf("efficientsam: mask data too short (need offset %d+%d, have %d)", off, planeSize, len(maskT.Data))
	}
	selected := maskT.Data[off : off+planeSize]
	return selected, bestScore, nil
}

// maskToResult upsamples the low-res mask (typically 256×256) to the original image
// size using nearest-neighbor interpolation, thresholds at logit > 0, encodes as
// column-major RLE, and computes a tight BBox.
//
// EfficientSAM's decoder outputs low-resolution masks (256×256) — unlike MobileSAM's
// decoder which upsamples to the original size via orig_im_size. We upsample here in Go.
func maskToResult(maskData []float32, iou float64, origW, origH int) (models.Mask, error) {
	n := len(maskData)
	if n == 0 {
		return models.Mask{}, fmt.Errorf("efficientsam: empty mask data")
	}

	// Low-res spatial dimensions — assume 256×256 per the EfficientSAM spec.
	// (pickBestMask already extracted a single candidate plane of this size.)
	const lowResH, lowResW = 256, 256

	// Nearest-neighbor upsample: for each output pixel (ox, oy), map back to low-res.
	bin := make([]bool, origH*origW)
	minX, minY, maxX, maxY := origW, origH, -1, -1
	for oy := 0; oy < origH; oy++ {
		// Map output row to low-res row.
		ly := oy * lowResH / origH
		if ly >= lowResH {
			ly = lowResH - 1
		}
		for ox := 0; ox < origW; ox++ {
			lx := ox * lowResW / origW
			if lx >= lowResW {
				lx = lowResW - 1
			}
			idx := ly*lowResW + lx
			if idx < n && maskData[idx] > 0 {
				bin[oy*origW+ox] = true
				if ox < minX {
					minX = ox
				}
				if ox > maxX {
					maxX = ox
				}
				if oy < minY {
					minY = oy
				}
				if oy > maxY {
					maxY = oy
				}
			}
		}
	}

	var bbox [4]float64
	if maxX >= 0 {
		bbox = [4]float64{float64(minX), float64(minY), float64(maxX - minX + 1), float64(maxY - minY + 1)}
	}

	return models.Mask{
		RLE:  encodeRLEColumnMajor(bin, origH, origW),
		BBox: bbox,
		Conf: iou,
	}, nil
}

// encodeRLEColumnMajor encodes a binary mask as COCO-style uncompressed RLE: counts of
// alternating runs read in COLUMN-major (Fortran) order, always starting with a background
// (0) run. Serialized as space-separated decimal counts.
func encodeRLEColumnMajor(bin []bool, h, w int) string {
	if len(bin) == 0 {
		return ""
	}
	var counts []int
	prev := false // runs start with background
	run := 0
	for x := 0; x < w; x++ {
		for y := 0; y < h; y++ {
			v := bin[y*w+x]
			if v == prev {
				run++
			} else {
				counts = append(counts, run)
				prev = v
				run = 1
			}
		}
	}
	counts = append(counts, run)

	var sb strings.Builder
	for i, c := range counts {
		if i > 0 {
			sb.WriteByte(' ')
		}
		sb.WriteString(strconv.Itoa(c))
	}
	return sb.String()
}
