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
//   - batched_point_coords: [1, 1, N, 2] float32, ORIGINAL-image pixel coordinates
//   - batched_point_labels: [1, 1, N]    float32
//
// Coordinates are passed UNSCALED: the decoder graph rescales them itself
// (x·1024/orig_W, y·1024/orig_H, from the orig_im_size input — verified by graph
// inspection). Pre-scaling here would apply the resize twice.
func (p pointSet) batchedTensors() (coords, labels engine.Tensor) {
	n := p.n()
	coordData := make([]float32, n*2)
	for i, v := range p.coords {
		coordData[i] = float32(v)
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

// pickBestMask locates the output_masks and iou_predictions tensors in the decoder
// output list, then selects the mask candidate with the highest predicted IoU.
//
// EfficientSAM decoder outputs (verified on the real export):
//   - output_masks:    [1, 1, 3, H, W] — 3 candidate logit maps at ORIGINAL resolution
//   - iou_predictions: [1, 1, 3]       — per-candidate IoU scores (NOT sorted)
//   - a third 4-D low-res tensor (ignored)
//
// Returns the selected plane (length mH*mW), its dims, and its IoU score.
func pickBestMask(names []string, outs []engine.Tensor) (plane []float32, mH, mW int, bestIoU float64, err error) {
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
			// Prefer the 5-D output_masks over any other mask-named output.
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
		return nil, 0, 0, 0, fmt.Errorf("efficientsam: no mask tensor found in decoder outputs (shapes: %s)", strings.Join(shapes, ", "))
	}

	// [.., numCandidates, mH, mW] — leading batch dims are 1; take the first batch.
	rank := len(maskT.Shape)
	if rank < 3 {
		return nil, 0, 0, 0, fmt.Errorf("efficientsam: unexpected mask tensor rank %d, shape %v", rank, maskT.Shape)
	}
	numCandidates := int(maskT.Shape[rank-3])
	mH = int(maskT.Shape[rank-2])
	mW = int(maskT.Shape[rank-1])
	planeSize := mH * mW
	if numCandidates < 1 || mH <= 0 || mW <= 0 {
		return nil, 0, 0, 0, fmt.Errorf("efficientsam: degenerate mask shape %v", maskT.Shape)
	}

	best := 0
	if iouT != nil && len(iouT.Data) >= numCandidates {
		bestF := iouT.Data[0]
		for i := 1; i < numCandidates; i++ {
			if iouT.Data[i] > bestF {
				bestF = iouT.Data[i]
				best = i
			}
		}
		bestIoU = float64(bestF)
	}

	off := best * planeSize
	if off+planeSize > len(maskT.Data) {
		return nil, 0, 0, 0, fmt.Errorf("efficientsam: mask data too short (need offset %d+%d, have %d)", off, planeSize, len(maskT.Data))
	}
	return maskT.Data[off : off+planeSize], mH, mW, bestIoU, nil
}

// maskToResult thresholds the selected logit plane (logit >= 0, as in the official
// EfficientSAM example's torch.ge(logits, 0)), encodes it as column-major RLE at the
// ORIGINAL image size, and computes a tight BBox in original-image pixels.
//
// The decoder already returns masks at orig_im_size, so normally mW×mH == origW×origH
// and no resampling happens. If an export ever returns a different size, the plane is
// nearest-neighbour resampled to the original size so the RLE/BBox contract still holds.
func maskToResult(plane []float32, mW, mH int, iou float64, origW, origH int) (models.Mask, error) {
	if len(plane) == 0 || mW <= 0 || mH <= 0 {
		return models.Mask{}, fmt.Errorf("efficientsam: empty mask data")
	}
	if len(plane) != mW*mH {
		return models.Mask{}, fmt.Errorf("efficientsam: mask plane has %d values, want %dx%d", len(plane), mW, mH)
	}
	if origW <= 0 || origH <= 0 {
		return models.Mask{}, fmt.Errorf("efficientsam: invalid original size %dx%d", origW, origH)
	}

	bin := make([]bool, origH*origW)
	minX, minY, maxX, maxY := origW, origH, -1, -1
	for oy := 0; oy < origH; oy++ {
		ly := oy
		if mH != origH {
			ly = oy * mH / origH
		}
		for ox := 0; ox < origW; ox++ {
			lx := ox
			if mW != origW {
				lx = ox * mW / origW
			}
			if plane[ly*mW+lx] >= 0 {
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
