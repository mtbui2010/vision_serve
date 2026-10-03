package detr

import (
	"fmt"

	"visionserve/internal/engine"
)

// Outputs are the tensors a DETR-style export can carry, identified by SplitOutputs.
type Outputs struct {
	Boxes  engine.Tensor // [1, Q, 4] normalized boxes
	Logits engine.Tensor // [1, Q, C] class logits, BEFORE sigmoid (zero Tensor when absent)
	Feats  engine.Tensor // [1, Q, D] query features of the "-qf" exports (zero Tensor when absent)
}

// SplitOutputs identifies boxes, class logits and query features among a DETR export's outputs
// BY SHAPE — export names differ across releases, and indexing by position broke the moment an
// export added an output (a 17-class cache once served a 22-class model that way). It is the one
// rule the router (hybrid), the text-aligned head (textalign) and their tests share; it replaced
// three heuristics that agreed on every shipped export only by luck of output order.
//
//   - Boxes: the first output whose last dimension is 4.
//   - Logits: the first other output whose last dimension is nLabels (the manifest's label
//     count) and differs from dFeat; when nLabels <= 0 or no output matches, the first output
//     other than the boxes (and, when dFeat is known, other than a [1, Q, dFeat] feature
//     tensor) — RF-DETR's own postprocess rule, so a labels file that does not match the export
//     still decodes as it always did (classes past the file are class_<id>).
//   - Feats: the first remaining [1, Q, D] output (rank 3, the boxes' query count) whose D is
//     dFeat, or any D when dFeat <= 0. Cross-attention maps ([3, 1, 16, Q, S]) are rank 5 and
//     never match.
//
// Verified on every shipped export (models/*/ ONNX headers; TestSplitOutputsShippedExports):
// rf-detr-base(-real) pred_boxes/pred_logits; rfdetr-small(-etri) dets/labels[/cross_attn];
// the "-qf" and textalign detectors dets/labels/[cross_attn/]query_feats. On each, Logits is the
// tensor RF-DETR's postprocess has always decoded, and Feats is query_feats or absent.
//
// Only Boxes is required; a missing Logits or Feats is a zero Tensor (Data == nil) for the caller
// to reject in its own words.
func SplitOutputs(outs []engine.Tensor, nLabels, dFeat int) (Outputs, error) {
	var o Outputs
	ib, il := -1, -1
	for i, t := range outs {
		if t.Dim(-1) == 4 {
			ib = i
			break
		}
	}
	if ib < 0 {
		return o, fmt.Errorf("detr: could not identify the boxes tensor (no output has a last dimension == 4; shapes %v)", shapes(outs))
	}
	o.Boxes = outs[ib]

	if nLabels > 0 {
		for i, t := range outs {
			if i != ib && t.Dim(-1) == int64(nLabels) && int(t.Dim(-1)) != dFeat {
				il = i
				break
			}
		}
	}
	q := o.Boxes.Dim(1)
	isFeats := func(t engine.Tensor) bool { // a [1, Q, dFeat] query-feature tensor (dFeat known)
		return dFeat > 0 && len(t.Shape) == 3 && t.Dim(1) == q && t.Dim(-1) == int64(dFeat)
	}
	if il < 0 {
		for i, t := range outs {
			if i != ib && !isFeats(t) {
				il = i
				break
			}
		}
	}
	if il >= 0 {
		o.Logits = outs[il]
	}

	for i, t := range outs {
		if i == ib || i == il || len(t.Shape) != 3 || t.Dim(1) != q {
			continue
		}
		if dFeat <= 0 || t.Dim(-1) == int64(dFeat) {
			o.Feats = t
			break
		}
	}
	return o, nil
}

func shapes(ts []engine.Tensor) [][]int64 {
	out := make([][]int64, len(ts))
	for i, t := range ts {
		out[i] = t.Shape
	}
	return out
}
