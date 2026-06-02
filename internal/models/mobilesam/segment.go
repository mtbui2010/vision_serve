package mobilesam

import (
	"fmt"
	"image"

	"visionserve/internal/engine"
	"visionserve/pkg/api"
)

// Segment is the REUSABLE MobileSAM core (also called by grounded-sam). It runs the
// encoder once, then the decoder once per box, returning one mask per box (index-aligned
// with the input boxes).
//
//   - boxes:        each [x,y,w,h] in ORIGINAL image coords.
//   - encRun/decRun: execute the encoder/decoder sessions, binding inputs by name.
//   - encInputName: the encoder's single input name (e.g. "input_image").
//   - decOutNames:  the decoder's output names (to pick masks vs iou_predictions).
//
// Behavior matches the original mobileSAM.Infer box path exactly: a box becomes 2 points
// (top-left label 2, bottom-right label 3) in resized-1024 space; mask threshold logit>0;
// mask encoded as column-major RLE.
func Segment(
	img image.Image,
	boxes [][4]float64,
	encRun, decRun func(map[string]engine.Tensor) ([]engine.Tensor, error),
	encInputName string,
	decOutNames []string,
) ([]api.Mask, error) {
	origW := img.Bounds().Dx()
	origH := img.Bounds().Dy()

	// 1) Encoder: image → embedding. scale maps original coords → 1024 input space.
	encIn, scale := encoderInput(img)
	encOuts, err := encRun(map[string]engine.Tensor{encInputName: encIn})
	if err != nil {
		return nil, fmt.Errorf("mobilesam: encoder failed: %w", err)
	}
	if len(encOuts) == 0 {
		return nil, fmt.Errorf("mobilesam: encoder returned no output")
	}
	embedding := encOuts[0]

	// 2) Decoder per box → mask.
	zeros := make([]float32, 256*256)
	masks := make([]api.Mask, 0, len(boxes))
	for _, b := range boxes {
		ps := pointSet{
			coords: []float64{b[0], b[1], b[0] + b[2], b[1] + b[3]},
			labels: []float32{2, 3},
		}
		coords := ps.scaledCoords(scale)
		dec := map[string]engine.Tensor{
			"image_embeddings": embedding,
			"point_coords":     engine.F32(coords, 1, int64(ps.n()), 2),
			"point_labels":     engine.F32(ps.labels, 1, int64(ps.n())),
			"mask_input":       engine.F32(zeros, 1, 1, 256, 256),
			"has_mask_input":   engine.F32([]float32{0}, 1),
			"orig_im_size":     engine.F32([]float32{float32(origH), float32(origW)}, 2),
		}
		outs, err := decRun(dec)
		if err != nil {
			return nil, fmt.Errorf("mobilesam: decoder failed: %w", err)
		}
		maskT, iouT := pickMaskAndIoU(decOutNames, outs)
		if maskT == nil {
			return nil, fmt.Errorf("mobilesam: decoder output has no mask tensor (shapes %v)", shapesOf(outs))
		}
		mk, err := maskToResult(maskT, iouT, origW, origH)
		if err != nil {
			return nil, err
		}
		masks = append(masks, mk)
	}
	return masks, nil
}
