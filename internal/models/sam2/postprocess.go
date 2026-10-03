package sam2

import (
	"fmt"
	"strings"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/vision/mask"
)

// pointSet is one prompt for a single decoder run: point coordinates in ORIGINAL
// image space (flat x,y pairs) and SAM2 integer labels.
type pointSet struct {
	coords []float64 // [x0,y0, x1,y1, ...] in original-image pixels
	labels []int64   // SAM2 labels: 2=box top-left, 3=box bottom-right, 1=fg, 0=bg
}

// scaledPrompt maps original coords into the 1024-space expected by the SAM2 decoder.
// The encoder input is the image SQUASHED to 1024×1024 (upstream SAM2Transforms), so the
// mapping is per axis: x·scaleX, y·scaleY with scaleX = 1024/W, scaleY = 1024/H.
func (p pointSet) scaledPrompt(scaleX, scaleY float32) (coords [][2]float32, labels []int64) {
	coords = make([][2]float32, len(p.labels))
	for i := 0; i < len(p.labels); i++ {
		coords[i] = [2]float32{
			float32(p.coords[i*2]) * scaleX,
			float32(p.coords[i*2+1]) * scaleY,
		}
	}
	labels = p.labels
	return coords, labels
}

// promptToPointSets turns a Prompt into one decoder prompt per object.
//
//   - Each box [x,y,w,h] → 2 points: top-left (label 2) + bottom-right (label 3).
//   - Points-only (no box) → a single set with the given points (no padding needed
//     for SAM2; it handles variable-length point lists natively).
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
			return nil, fmt.Errorf("sam2: this model needs a BOX or POINT prompt, not text — "+
				"for text-driven segmentation use 'grounded-sam'. "+
				"Got text prompt: %q", p.Text)
		}
		return nil, models.BadPrompt(fmt.Errorf("sam2: a prompt (box or point) is required — " +
			"e.g. `run sam2 img.jpg --box x,y,w,h`"))
	}
	return sets, nil
}

// pickEncoderOutputs extracts the three encoder outputs from the Run result.
// Matches by output name first, falls back to output order (index 0, 1, 2).
//
// TODO: verify output names against your SAM2 ONNX export.
func pickEncoderOutputs(names []string, outs []engine.Tensor) (imageEmbed, highResFeats0, highResFeats1 engine.Tensor, err error) {
	if len(outs) < 3 {
		return engine.Tensor{}, engine.Tensor{}, engine.Tensor{},
			fmt.Errorf("sam2 encoder: expected 3 outputs (image_embed, high_res_feats_0, high_res_feats_1), got %d", len(outs))
	}

	// Try name-based assignment first.
	byName := map[string]int{}
	for i, n := range names {
		byName[strings.ToLower(n)] = i
	}

	idxEmbed := -1
	idxHR0 := -1
	idxHR1 := -1

	for name, idx := range byName {
		switch {
		case name == "image_embed" || name == "image_embeddings":
			idxEmbed = idx
		case name == "high_res_feats_0":
			idxHR0 = idx
		case name == "high_res_feats_1":
			idxHR1 = idx
		}
	}

	// Fall back to positional order: [0]=image_embed, [1]=high_res_feats_0, [2]=high_res_feats_1.
	if idxEmbed < 0 {
		idxEmbed = 0
	}
	if idxHR0 < 0 {
		idxHR0 = 1
	}
	if idxHR1 < 0 {
		idxHR1 = 2
	}

	if idxEmbed >= len(outs) || idxHR0 >= len(outs) || idxHR1 >= len(outs) {
		return engine.Tensor{}, engine.Tensor{}, engine.Tensor{},
			fmt.Errorf("sam2 encoder: index out of range (embed=%d, hr0=%d, hr1=%d, len=%d)",
				idxEmbed, idxHR0, idxHR1, len(outs))
	}

	return outs[idxEmbed], outs[idxHR0], outs[idxHR1], nil
}

// runDecoder calls the SAM2 decoder session with the encoder outputs + point prompt.
//
// Verified I/O for SharpAI/sam2-hiera-tiny-onnx (encoder.onnx + decoder.onnx):
//   - "image_embed":      [1,256,64,64]  float32 from encoder
//   - "high_res_feats_0": [1,32,256,256] float32 from encoder
//   - "high_res_feats_1": [1,64,128,128] float32 from encoder
//   - "point_coords":     [1,N,2]        float32 — in 1024-space (per-axis scaled)
//   - "point_labels":     [1,N]          float32 (NOT int64 — verified against ONNX)
//   - "mask_input":       [1,1,256,256]  float32 zeros (no prior mask)
//   - "has_mask_input":   [1]            float32 0.0
//
// Decoder outputs:
//   - "masks":            [1,3,256,256] float32 LOW-RES logits (3 multimask
//     candidates, clamped to [-32,32]; pick by IoU, upsample to the original size)
//   - "iou_predictions":  [1,3]         float32
func runDecoder(
	r models.Runner,
	imageEmbed, highResFeats0, highResFeats1 engine.Tensor,
	coords [][2]float32,
	labels []int64,
) (maskTensor, iouTensor engine.Tensor, err error) {
	n := len(labels)
	if n == 0 {
		return engine.Tensor{}, engine.Tensor{}, fmt.Errorf("sam2: decoder requires at least one point")
	}

	// Flatten [][2]float32 to []float32 for the tensor.
	flatCoords := make([]float32, n*2)
	for i, c := range coords {
		flatCoords[i*2] = c[0]
		flatCoords[i*2+1] = c[1]
	}

	// SAM2 decoder takes point_labels as float32 (verified against ONNX), not int64.
	labelsF32 := make([]float32, n)
	for i, l := range labels {
		labelsF32[i] = float32(l)
	}

	// mask_input / has_mask_input: always pass zero tensors (no prior mask).
	maskInputZeros := make([]float32, 256*256)

	inputs := map[string]engine.Tensor{
		"image_embed":      imageEmbed,
		"high_res_feats_0": highResFeats0,
		"high_res_feats_1": highResFeats1,
		"point_coords":     engine.F32(flatCoords, 1, int64(n), 2),
		"point_labels":     engine.F32(labelsF32, 1, int64(n)),
		"mask_input":       engine.F32(maskInputZeros, 1, 1, 256, 256),
		"has_mask_input":   engine.F32([]float32{0}, 1),
	}

	outs, err := r.Run(roleDecoder, inputs)
	if err != nil {
		return engine.Tensor{}, engine.Tensor{}, err
	}

	outNames := r.OutputNames(roleDecoder)
	maskT, iouT := findMaskAndIoU(outNames, outs)
	if maskT == nil {
		return engine.Tensor{}, engine.Tensor{},
			fmt.Errorf("sam2: decoder returned no mask tensor (shapes: %v)", shapesOf(outs))
	}
	if iouT == nil {
		// iou is optional; return zero tensor
		empty := engine.F32([]float32{0}, 1, 1)
		return *maskT, empty, nil
	}
	return *maskT, *iouT, nil
}

// findMaskAndIoU selects the masks and iou_predictions tensors from decoder outputs.
// Matches by name first; falls back to shape (4-D = masks, 2-D = iou).
func findMaskAndIoU(names []string, outs []engine.Tensor) (maskT, iouT *engine.Tensor) {
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
	// Shape-based fallback.
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

// pickBestMask selects the best mask channel from the SAM2 decoder output, upsamples its
// logits to the ORIGINAL image size, thresholds at 0, computes a tight bbox and encodes
// the mask as column-major RLE.
//
// Verified decoder output (SharpAI/sam2-hiera-tiny-onnx): masks [1, 3, 256, 256] — the
// three multimask candidates (multimask_output=True is baked into the export), LOW-RES
// logits already clamped to [-32, 32] (graph ends in Clip, no Resize) — and
// iou_predictions [1, 3]. The 256×256 grid covers the whole squashed 1024×1024 input, so
// upstream SAM2Transforms.postprocess_masks simply does
// F.interpolate(masks, (H, W), mode="bilinear", align_corners=False) and then > 0.
func pickBestMask(maskTensor, iouTensor engine.Tensor, origW, origH int) (models.Mask, error) {
	if len(maskTensor.Shape) < 4 {
		return models.Mask{}, fmt.Errorf("sam2: unexpected mask shape %v (want 4-D)", maskTensor.Shape)
	}
	if origW <= 0 || origH <= 0 {
		return models.Mask{}, fmt.Errorf("sam2: invalid original size %dx%d", origW, origH)
	}

	// Shape: [batch, channels, H, W] — find best channel by IoU.
	batch := int(maskTensor.Dim(0))
	ch := int(maskTensor.Dim(1))
	h := int(maskTensor.Dim(2))
	w := int(maskTensor.Dim(3))

	if batch < 1 || ch < 1 || h <= 0 || w <= 0 {
		return models.Mask{}, fmt.Errorf("sam2: degenerate mask shape %v", maskTensor.Shape)
	}
	if len(maskTensor.Data) < ch*h*w {
		return models.Mask{}, fmt.Errorf("sam2: mask data length %d < %d (shape %v)",
			len(maskTensor.Data), ch*h*w, maskTensor.Shape)
	}

	// Pick best channel by iou_predictions.
	best, conf := 0, 0.0
	if len(iouTensor.Data) > 0 {
		bestScore := float32(-1e30)
		lim := ch
		if len(iouTensor.Data) < lim {
			lim = len(iouTensor.Data)
		}
		for i := 0; i < lim; i++ {
			if iouTensor.Data[i] > bestScore {
				bestScore = iouTensor.Data[i]
				best = i
			}
		}
		conf = float64(bestScore)
	}

	// Upsample the chosen low-res logits to (origH, origW) and threshold at 0.
	off := best * h * w
	bm := mask.UpsampleBilinearThreshold(maskTensor.Data[off:off+h*w], w, h, w, origH, origW, 0)

	return models.Mask{
		RLE:  mask.EncodeRLE(bm),
		BBox: bm.BBox(),
		Conf: conf,
	}, nil
}

func shapesOf(ts []engine.Tensor) [][]int64 {
	out := make([][]int64, len(ts))
	for i, t := range ts {
		out[i] = t.Shape
	}
	return out
}
