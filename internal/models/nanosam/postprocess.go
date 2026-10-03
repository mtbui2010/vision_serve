package nanosam

import (
	"fmt"
	"strings"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/vision/mask"
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
// NOTE: output masks are low-resolution (256×256); pickBestMask upscales them to the
// original size. origH/origW are accepted for API consistency but are not sent to the decoder.
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

// pickBestMask selects the highest-IoU mask from the decoder output, upscales its
// low-res logits to the ORIGINAL image size, thresholds at logit > 0, and encodes it as
// column-major RLE (COCO uncompressed style). BBox is in original-image pixels.
//
// The decoder returns low_res_masks [1, M, 256, 256] covering the whole 1024×1024 encoder
// canvas, of which only the top-left (resized image) region holds the image — the rest is
// padding. Upscaling mirrors upstream nanosam/utils/predictor.py upscale_mask exactly:
//
//	if W > H: lim_x = 256, lim_y = int(256·H/W)   else: lim_x = int(256·W/H), lim_y = 256
//	mask = F.interpolate(mask[:, :, :lim_y, :lim_x], (H, W), mode="bilinear")  # align_corners=False
//	binary = mask > 0
//
// NOTE: not verified against real NanoSAM weights (none in models/nano-sam); the recipe
// is covered by unit tests on synthetic logits with the real [1,M,256,256] shape.
func pickBestMask(outs []engine.Tensor, outNames []string, origW, origH int) (models.Mask, error) {
	maskT, iouT := pickMaskAndIoU(outNames, outs)
	if maskT == nil {
		shapes := make([][]int64, len(outs))
		for i, o := range outs {
			shapes[i] = o.Shape
		}
		return models.Mask{}, fmt.Errorf("nanosam: decoder output has no mask tensor (shapes %v)", shapes)
	}
	if len(maskT.Shape) != 4 {
		return models.Mask{}, fmt.Errorf("nanosam: unexpected mask shape %v (want [1,M,H,W])", maskT.Shape)
	}
	if origW <= 0 || origH <= 0 {
		return models.Mask{}, fmt.Errorf("nanosam: invalid original size %dx%d", origW, origH)
	}

	n := int(maskT.Dim(1))
	h := int(maskT.Dim(2))
	w := int(maskT.Dim(3))
	if n < 1 || h <= 0 || w <= 0 {
		return models.Mask{}, fmt.Errorf("nanosam: unexpected mask shape %v", maskT.Shape)
	}
	if len(maskT.Data) < n*h*w {
		return models.Mask{}, fmt.Errorf("nanosam: mask data length %d < %d (shape %v)", len(maskT.Data), n*h*w, maskT.Shape)
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

	limX, limY := lowResCrop(origW, origH, w, h)
	off := best * h * w
	bm := mask.UpsampleBilinearThreshold(maskT.Data[off:off+h*w], w, limY, limX, origH, origW, 0)

	return models.Mask{
		RLE:  mask.EncodeRLE(bm),
		BBox: bm.BBox(),
		Conf: conf,
	}, nil
}

// lowResCrop returns the (limX, limY) extent of the image content inside the low-res
// (lrW×lrH, normally 256×256) mask grid — upstream upscale_mask, with Python int()
// truncation. The encoder input is the image resized long-side-to-1024 and padded
// bottom/right, so the content occupies the top-left corner.
func lowResCrop(origW, origH, lrW, lrH int) (limX, limY int) {
	if origW > origH {
		limX = lrW
		limY = int(float64(lrH) * float64(origH) / float64(origW))
	} else {
		limX = int(float64(lrW) * float64(origW) / float64(origH))
		limY = lrH
	}
	if limX < 1 {
		limX = 1
	}
	if limY < 1 {
		limY = 1
	}
	return limX, limY
}

// pickMaskAndIoU identifies the masks and iou_predictions tensors from decoder output,
// preferring name-based matching and falling back to shape heuristics.
func pickMaskAndIoU(names []string, outs []engine.Tensor) (maskT, iouT *engine.Tensor) {
	for i := range outs {
		name := ""
		if i < len(names) {
			name = strings.ToLower(names[i])
		}
		switch {
		case strings.Contains(name, "iou"):
			iouT = &outs[i]
		// NanoSAM decoder outputs "low_res_masks" [1,M,256,256].
		// Also accept plain "masks" for forwards-compat with other SAM ONNX variants.
		case strings.Contains(name, "mask"):
			maskT = &outs[i]
		}
	}
	// Shape-based fallback: masks = largest 4-D tensor; iouT = 2-D tensor.
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
