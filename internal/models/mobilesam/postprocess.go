package mobilesam

import (
	"fmt"
	"strconv"
	"strings"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// pointSet is one prompt for a single decoder run: point coordinates in ORIGINAL
// image space (flat x,y pairs) + a label per point.
type pointSet struct {
	coords []float64 // [x0,y0, x1,y1, ...] in original-image pixels
	labels []float32 // SAM labels: 2=box top-left, 3=box bottom-right, 1=fg, 0=bg, -1=pad
}

func (p pointSet) n() int { return len(p.labels) }

// scaledCoords maps original coords into the resized-1024 space the decoder expects.
func (p pointSet) scaledCoords(scale float64) []float32 {
	out := make([]float32, len(p.coords))
	for i, v := range p.coords {
		out[i] = float32(v * scale)
	}
	return out
}

// promptToPointSets turns a Prompt into one decoder prompt per object.
//
//   - Each box [x,y,w,h] → 2 points: top-left(label 2) + bottom-right(label 3).
//   - Points-only (no box) → a single set of the given points, padded with a (0,0)
//     label -1 point (SAM convention when there is no box).
func promptToPointSets(p models.Prompt) ([]pointSet, error) {
	var sets []pointSet
	for _, b := range p.Boxes {
		x, y, w, h := b[0], b[1], b[2], b[3]
		sets = append(sets, pointSet{
			coords: []float64{x, y, x + w, y + h},
			labels: []float32{2, 3},
		})
	}
	if len(p.Points) > 0 {
		ps := pointSet{}
		for _, pt := range p.Points {
			ps.coords = append(ps.coords, pt.X, pt.Y)
			ps.labels = append(ps.labels, float32(pt.Label))
		}
		// SAM padding point when no box accompanies the points.
		ps.coords = append(ps.coords, 0, 0)
		ps.labels = append(ps.labels, -1)
		sets = append(sets, ps)
	}
	if len(sets) == 0 {
		if p.Text != "" {
			return nil, fmt.Errorf("mobilesam: this model needs a BOX or POINT prompt, not text — "+
				"a text prompt like %q is for the 'grounded-sam' model (text → boxes → masks). "+
				"Either run `grounded-sam ... --prompt %q`, or give mobile-sam a box: `mobile-sam ... --box x,y,w,h`",
				p.Text, p.Text)
		}
		// No prompt at all → caller uses the Automatic Mask Generator.
		return nil, nil
	}
	return sets, nil
}

// pickMaskAndIoU finds the masks and iou_predictions tensors among decoder outputs.
// Prefers matching by name; falls back to shape (masks = 4-D with the largest area so
// it is not confused with low_res_masks [.,.,256,256]; iou = 2-D).
func pickMaskAndIoU(names []string, outs []engine.Tensor) (mask, iou *engine.Tensor) {
	for i := range outs {
		name := ""
		if i < len(names) {
			name = strings.ToLower(names[i])
		}
		switch {
		case strings.Contains(name, "iou"):
			iou = &outs[i]
		case name == "masks" || (strings.Contains(name, "mask") && !strings.Contains(name, "low_res")):
			mask = &outs[i]
		}
	}
	if mask == nil {
		var bestArea int64 = -1
		for i := range outs {
			if len(outs[i].Shape) == 4 {
				area := outs[i].Dim(2) * outs[i].Dim(3)
				if area > bestArea {
					bestArea = area
					mask = &outs[i]
				}
			}
		}
	}
	if iou == nil {
		for i := range outs {
			if len(outs[i].Shape) == 2 {
				iou = &outs[i]
				break
			}
		}
	}
	return mask, iou
}

// maskToResult thresholds the best mask (logit>0), encodes it as a column-major RLE,
// and computes a tight bbox + confidence (from iou_predictions).
func maskToResult(mask, iou *engine.Tensor, origW, origH int) (models.Mask, error) {
	n := int(mask.Dim(1))
	h := int(mask.Dim(2))
	w := int(mask.Dim(3))
	if n < 1 || h <= 0 || w <= 0 {
		return models.Mask{}, fmt.Errorf("mobilesam: unexpected mask shape %v", mask.Shape)
	}

	best, conf := bestChannel(mask, iou)
	off := best * h * w
	bin := make([]bool, h*w)
	minX, minY, maxX, maxY := w, h, -1, -1
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if mask.Data[off+y*w+x] > 0 {
				bin[y*w+x] = true
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

	var bbox [4]float64
	if maxX >= 0 {
		bbox = [4]float64{float64(minX), float64(minY), float64(maxX - minX + 1), float64(maxY - minY + 1)}
	}

	return models.Mask{
		RLE:  encodeRLEColumnMajor(bin, h, w),
		BBox: bbox,
		Conf: conf,
	}, nil
}

// encodeRLEColumnMajor encodes a binary mask as COCO-style uncompressed RLE: counts of
// alternating runs read in COLUMN-major (Fortran) order, always starting with a
// background (0) run. Serialized as space-separated decimal counts.
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
