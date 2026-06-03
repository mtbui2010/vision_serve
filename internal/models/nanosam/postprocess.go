package nanosam

import (
	"fmt"
	"strconv"
	"strings"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// encodePrompt converts a box or point prompt into the two SAM decoder tensors:
//
//	pointCoords [1, N, 2]  — coordinates in resized-1024 space (= original × scale)
//	pointLabels [1, N]     — SAM labels per point
//
// Encoding rules (same as MobileSAM):
//   - Box [x,y,w,h]: 2 points — top-left label 2, bottom-right label 3.
//   - Points-only:   N foreground/background points + one padding (0,0) label -1.
//
// Only the FIRST box and the point list are used (one decoder run per call).
// Multi-box/multi-object is handled by the caller looping over boxes.
func encodePrompt(prompt models.Prompt, scale float32) (pointCoords, pointLabels engine.Tensor, err error) {
	var coords []float32
	var labels []float32

	switch {
	case len(prompt.Boxes) > 0:
		// Encode first box as top-left/bottom-right pair (caller loops for more boxes).
		b := prompt.Boxes[0]
		x0, y0, x1, y1 := float32(b[0])*scale, float32(b[1])*scale,
			float32(b[0]+b[2])*scale, float32(b[1]+b[3])*scale
		coords = []float32{x0, y0, x1, y1}
		labels = []float32{2, 3}

	case len(prompt.Points) > 0:
		for _, pt := range prompt.Points {
			coords = append(coords, float32(pt.X)*scale, float32(pt.Y)*scale)
			labels = append(labels, float32(pt.Label))
		}
		// SAM padding point when no box is provided.
		coords = append(coords, 0, 0)
		labels = append(labels, -1)

	default:
		err = fmt.Errorf("nanosam: a box or point prompt is required (text prompts are not supported — use grounded-sam for text-driven segmentation)")
		return
	}

	n := int64(len(labels))
	pointCoords = engine.F32(coords, 1, n, 2)
	pointLabels = engine.F32(labels, 1, n)
	return
}

// runDecoder calls runner.Run("decoder") with the NanoSAM decoder inputs and
// returns the raw output tensors.
//
// Tensor names verified from nanosam/tools/export_sam_mask_decoder_onnx.py in
// github.com/NVIDIA-AI-IOT/nanosam (input_names / output_names arguments to
// torch.onnx.export).
//
// Decoder inputs (5 total — no "orig_im_size"):
//
//	"image_embeddings" [1, 256, 64, 64]
//	"point_coords"     [1, N, 2]
//	"point_labels"     [1, N]
//	"mask_input"       [1, 1, 256, 256]  (zeros = no prior mask)
//	"has_mask_input"   [1]               (0 = no prior mask)
//
// Decoder outputs: "iou_predictions" [1, M], "low_res_masks" [1, M, 256, 256]
// NOTE: output masks are low-resolution (256×256), not upsampled to original size.
// origH/origW are accepted for API consistency but are not sent to the decoder.
func runDecoder(runner models.Runner, imageEmbed, pointCoords, pointLabels engine.Tensor, origH, origW int) ([]engine.Tensor, error) {
	zeros := make([]float32, 256*256)
	inputs := map[string]engine.Tensor{
		"image_embeddings": imageEmbed,
		"point_coords":     pointCoords,
		"point_labels":     pointLabels,
		"mask_input":       engine.F32(zeros, 1, 1, 256, 256),
		"has_mask_input":   engine.F32([]float32{0}, 1),
		// NanoSAM decoder does NOT accept "orig_im_size" — the decoder returns
		// low_res_masks [1, M, 256, 256] and the caller handles upsampling.
	}
	outs, err := runner.Run(roleDecoder, inputs)
	if err != nil {
		return nil, fmt.Errorf("nanosam: decoder failed: %w", err)
	}
	return outs, nil
}

// pickBestMask selects the highest-IoU mask from the decoder output, thresholds at
// logit > 0, and encodes it as column-major RLE (COCO uncompressed style).
func pickBestMask(outs []engine.Tensor, outNames []string, origW, origH int) (models.Mask, error) {
	maskT, iouT := pickMaskAndIoU(outNames, outs)
	if maskT == nil {
		shapes := make([][]int64, len(outs))
		for i, o := range outs {
			shapes[i] = o.Shape
		}
		return models.Mask{}, fmt.Errorf("nanosam: decoder output has no mask tensor (shapes %v)", shapes)
	}

	n := int(maskT.Dim(1))
	h := int(maskT.Dim(2))
	w := int(maskT.Dim(3))
	if n < 1 || h <= 0 || w <= 0 {
		return models.Mask{}, fmt.Errorf("nanosam: unexpected mask shape %v", maskT.Shape)
	}

	// Pick the channel with the highest IoU prediction.
	best, conf := 0, 0.0
	if iouT != nil && len(iouT.Data) > 0 {
		bestScore := float32(-1e30)
		lim := n
		if len(iouT.Data) < lim {
			lim = len(iouT.Data)
		}
		for i := 0; i < lim; i++ {
			if iouT.Data[i] > bestScore {
				bestScore = iouT.Data[i]
				best = i
			}
		}
		conf = float64(bestScore)
	}

	off := best * h * w
	bin := make([]bool, h*w)
	minX, minY, maxX, maxY := w, h, -1, -1
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if maskT.Data[off+y*w+x] > 0 {
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

// pickMaskAndIoU identifies the masks and iou_predictions tensors from decoder output,
// preferring name-based matching and falling back to shape heuristics.
func pickMaskAndIoU(names []string, outs []engine.Tensor) (mask, iou *engine.Tensor) {
	for i := range outs {
		name := ""
		if i < len(names) {
			name = strings.ToLower(names[i])
		}
		switch {
		case strings.Contains(name, "iou"):
			iou = &outs[i]
		// NanoSAM decoder outputs "low_res_masks" [1,M,256,256].
		// Also accept plain "masks" for forwards-compat with other SAM ONNX variants.
		case strings.Contains(name, "mask"):
			mask = &outs[i]
		}
	}
	// Shape-based fallback: masks = largest 4-D tensor; iou = 2-D tensor.
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

// encodeRLEColumnMajor encodes a binary mask as COCO-style uncompressed RLE: counts of
// alternating runs read in COLUMN-major (Fortran) order, always starting with a
// background (0) run, serialized as space-separated decimal counts.
func encodeRLEColumnMajor(bin []bool, h, w int) string {
	if len(bin) == 0 {
		return ""
	}
	var counts []int
	prev := false // runs start with background (0)
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
