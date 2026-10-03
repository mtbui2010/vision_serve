package mobilesam

import (
	"fmt"
	"strings"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/vision/mask"
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
func pickMaskAndIoU(names []string, outs []engine.Tensor) (maskT, iouT *engine.Tensor) {
	for i := range outs {
		name := ""
		if i < len(names) {
			name = strings.ToLower(names[i])
		}
		switch {
		case strings.Contains(name, "iou"):
			iouT = &outs[i]
		case name == "masks" || (strings.Contains(name, "mask") && !strings.Contains(name, "low_res")):
			maskT = &outs[i]
		}
	}
	if maskT == nil {
		var bestArea int64 = -1
		for i := range outs {
			if len(outs[i].Shape) == 4 {
				area := outs[i].Dim(2) * outs[i].Dim(3)
				if area > bestArea {
					bestArea = area
					maskT = &outs[i]
				}
			}
		}
	}
	if iouT == nil {
		for i := range outs {
			if len(outs[i].Shape) == 2 {
				iouT = &outs[i]
				break
			}
		}
	}
	return maskT, iouT
}

// MaskBitmap is a binary mask at ORIGINAL-image resolution (row-major, len W*H),
// with its tight bbox ([x,y,w,h] original-image pixels) and confidence. It is the
// un-encoded form behind models.Mask.RLE: in-process consumers (the grasp pipeline)
// take these directly to avoid the RLE encode→decode round-trip — the decoder already
// upsamples masks to original size, so no rescaling is needed.
type MaskBitmap struct {
	Data []bool // row-major, len = W*H, true = foreground
	W, H int
	BBox [4]float64
	Conf float64
}

// ToMask encodes a MaskBitmap into the public models.Mask (column-major RLE + bbox + conf).
func (b MaskBitmap) ToMask() models.Mask {
	return models.Mask{
		RLE:  mask.EncodeRLE(mask.Bitmap{Data: b.Data, W: b.W, H: b.H}),
		BBox: b.BBox,
		Conf: b.Conf,
	}
}

// maskToBitmap thresholds the best mask (logit>0) and computes a tight bbox +
// confidence (from iou_predictions). It stops short of RLE encoding so callers can
// consume the raw bitmap directly.
func maskToBitmap(maskT, iou *engine.Tensor) (MaskBitmap, error) {
	n := int(maskT.Dim(1))
	h := int(maskT.Dim(2))
	w := int(maskT.Dim(3))
	if n < 1 || h <= 0 || w <= 0 {
		return MaskBitmap{}, fmt.Errorf("mobilesam: unexpected mask shape %v", maskT.Shape)
	}

	best, conf := bestChannel(maskT, iou)
	if len(maskT.Data) < (best+1)*h*w {
		return MaskBitmap{}, fmt.Errorf("mobilesam: mask data length %d < %d (shape %v)", len(maskT.Data), (best+1)*h*w, maskT.Shape)
	}
	bm, bbox := mask.Threshold(maskT.Data, best*h*w, h, w, 0)
	return MaskBitmap{Data: bm.Data, W: w, H: h, BBox: bbox, Conf: conf}, nil
}
