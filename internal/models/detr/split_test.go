package detr

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"visionserve/internal/engine"
)

// The three heuristics SplitOutputs replaced, kept here verbatim as references: SplitOutputs must
// pick the SAME tensors on every shipped export, or the router / head silently score the wrong
// columns.

// refHybrid is hybrid.detectRFDETRWithFeats: boxes = the LAST rank-3 output with last dim 4,
// feats = the LAST rank-3 output whose last dim is neither 4 nor the label count.
func refHybrid(outs []engine.Tensor, nCls int) (boxes, feats int) {
	boxes, feats = -1, -1
	for i, o := range outs {
		if len(o.Shape) == 3 && o.Dim(-1) == 4 {
			boxes = i
		}
	}
	for i, o := range outs {
		if len(o.Shape) == 3 && o.Dim(-1) != 4 && int(o.Dim(-1)) != nCls {
			feats = i
		}
	}
	return boxes, feats
}

// refTextalign is textalign.splitOutputs + classLogits.
func refTextalign(outs []engine.Tensor, dFeat, nLabels int) (boxes, feats, logits int) {
	boxes, feats, logits = -1, -1, -1
	for i, t := range outs {
		if t.Dim(-1) == 4 && boxes < 0 {
			boxes = i
		}
	}
	if boxes < 0 {
		return
	}
	q := outs[boxes].Dim(1)
	for i, t := range outs {
		if len(t.Shape) == 3 && t.Dim(-1) == int64(dFeat) && t.Dim(1) == q && feats < 0 {
			feats = i
		}
	}
	for i, t := range outs {
		if len(t.Shape) == 3 && t.Dim(1) == q && t.Dim(-1) == int64(nLabels) && int(t.Dim(-1)) != dFeat {
			logits = i
			break
		}
	}
	return
}

// export is one shipped detector's runtime output signature (models/*/ ONNX headers, symbolic
// dims resolved to what ORT returns at 300 queries) and the label count its manifests pair it with.
type export struct {
	name    string
	shapes  [][]int64
	nLabels int
}

var crossAttn = []int64{3, 1, 16, 300, 1024}

// shippedExports are the RF-DETR exports in models/ (read with onnx on 2026-10-03).
var shippedExports = []export{
	{"rf-detr/rf-detr-base-real (coco91)", [][]int64{{1, 300, 4}, {1, 300, 91}}, 91},
	{"rf-detr/rf-detr-base (dummy)", [][]int64{{1, 5, 91}, {1, 5, 4}}, 91},
	{"rf-detr-nano/rf-detr-base", [][]int64{{1, 300, 4}, {1, 300, 91}}, 91},
	{"rfdetr-small (91)", [][]int64{{1, 300, 4}, {1, 300, 91}, crossAttn}, 91},
	{"rfdetr-small-qf (91)", [][]int64{{1, 300, 4}, {1, 300, 91}, crossAttn, {1, 300, 256}}, 91},
	{"rfdetr-small-etri (23)", [][]int64{{1, 300, 4}, {1, 300, 23}}, 23},
	{"rfdetr-small-etri/model-explain (23)", [][]int64{{1, 300, 4}, {1, 300, 23}, crossAttn}, 23},
	{"rfdetr-small-etri-qf (23)", [][]int64{{1, 300, 4}, {1, 300, 23}, crossAttn, {1, 300, 256}}, 23},
	{"rfdetr-textalign-dec1 detector (23)", [][]int64{{1, 300, 4}, {1, 300, 23}, {1, 300, 256}}, 23},
	{"rfdetr-textalign-dec1-siglip detector = dec1_holdout (18)", [][]int64{{1, 300, 4}, {1, 300, 18}, {1, 300, 256}}, 18},
	{"rfdetr-textalign-dec1-siglip-prod detector (23)", [][]int64{{1, 300, 4}, {1, 300, 23}, {1, 300, 256}}, 23},
	{"pf-dec1-holdout-qf (18)", [][]int64{{1, 300, 4}, {1, 300, 18}, {1, 300, 256}}, 18},
}

// fakeOuts builds tensors of the given shapes, each with its own one-element buffer so the
// tests can tell them apart by identity.
func fakeOuts(shapes [][]int64) []engine.Tensor {
	outs := make([]engine.Tensor, len(shapes))
	for i, s := range shapes {
		outs[i] = engine.F32(make([]float32, 1), s...)
	}
	return outs
}

func indexOf(outs []engine.Tensor, t engine.Tensor) int {
	if t.Data == nil {
		return -1
	}
	for i := range outs {
		if &outs[i].Data[0] == &t.Data[0] {
			return i
		}
	}
	return -2
}

func checkExport(t *testing.T, e export) {
	t.Helper()
	outs := fakeOuts(e.shapes)
	rb, rl, err := splitRF(outs)
	if err != nil {
		t.Fatalf("%s: splitRF: %v", e.name, err)
	}
	hb, hf := refHybrid(outs, e.nLabels)
	for _, dFeat := range []int{0, 256} {
		o, err := SplitOutputs(outs, e.nLabels, dFeat)
		if err != nil {
			t.Fatalf("%s: SplitOutputs: %v", e.name, err)
		}
		b, l, f := indexOf(outs, o.Boxes), indexOf(outs, o.Logits), indexOf(outs, o.Feats)
		// RF-DETR's own postprocess (splitRF).
		if b != indexOf(outs, rb) || l != indexOf(outs, rl) {
			t.Errorf("%s dFeat=%d: boxes/logits = %d/%d, splitRF picks %d/%d", e.name, dFeat, b, l,
				indexOf(outs, rb), indexOf(outs, rl))
		}
		// The router (feature width unknown when the head is an ONNX session).
		if b != hb || f != hf {
			t.Errorf("%s dFeat=%d: boxes/feats = %d/%d, hybrid picked %d/%d", e.name, dFeat, b, f, hb, hf)
		}
	}
	// The text-aligned head (dFeat known from proj.bin).
	tb, tf, tl := refTextalign(outs, 256, e.nLabels)
	o, _ := SplitOutputs(outs, e.nLabels, 256)
	if b, f := indexOf(outs, o.Boxes), indexOf(outs, o.Feats); b != tb || f != tf {
		t.Errorf("%s: boxes/feats = %d/%d, textalign picked %d/%d", e.name, b, f, tb, tf)
	}
	if tl >= 0 && indexOf(outs, o.Logits) != tl {
		t.Errorf("%s: logits = %d, textalign's classLogits picked %d", e.name, indexOf(outs, o.Logits), tl)
	}
}

func TestSplitOutputsShippedExports(t *testing.T) {
	for _, e := range shippedExports {
		checkExport(t, e)
	}
}

// The same check against the ONNX headers actually on disk, when the model directories are there
// ($VISIONSERVE_ONNX_DIR, else the repository's models/; the weights are not committed): a
// re-export that reorders or adds outputs fails here instead of in a served mAP.
func TestSplitOutputsRealHeaders(t *testing.T) {
	root := os.Getenv("VISIONSERVE_ONNX_DIR")
	if root == "" {
		root = filepath.Join("..", "..", "..", "models")
	}
	cases := map[string]string{ // onnx path → its labels file
		"rf-detr/rf-detr-base-real.onnx":                  "rf-detr/coco91.txt",
		"rfdetr-small/model.onnx":                         "rfdetr-small/labels.txt",
		"rfdetr-small-qf/model.onnx":                      "rfdetr-small-qf/labels.txt",
		"rfdetr-small-etri/model.onnx":                    "rfdetr-small-etri/labels.txt",
		"rfdetr-small-etri-qf/model.onnx":                 "rfdetr-small-etri-qf/labels.txt",
		"rfdetr-textalign-dec1/detector.onnx":             "rfdetr-textalign-dec1/labels.txt",
		"rfdetr-textalign-dec1-siglip/detector.onnx":      "rfdetr-textalign-dec1-siglip/labels.txt",
		"rfdetr-textalign-etri/detector-noattn.onnx":      "rfdetr-small-etri-qf/labels.txt",
		"rfdetr-textalign-dec1-siglip-prod/detector.onnx": "rfdetr-textalign-dec1-siglip-prod/labels.txt",
	}
	seen := 0
	for onnx, labels := range cases {
		_, outs, err := engine.Inspect(filepath.Join(root, onnx))
		if err != nil {
			continue // model directory not present on this machine
		}
		raw, err := os.ReadFile(filepath.Join(root, labels))
		if err != nil {
			continue
		}
		n := 0
		for _, l := range strings.Split(string(raw), "\n") {
			if strings.TrimSpace(l) != "" {
				n++
			}
		}
		e := export{name: onnx, nLabels: n}
		for _, o := range outs {
			s := append([]int64(nil), o.Shape...)
			if len(s) == 5 {
				s = append([]int64(nil), crossAttn...) // declared symbolic; this is what ORT returns
			}
			for i := range s {
				if s[i] < 0 {
					s[i] = []int64{1, 300}[min(i, 1)]
				}
			}
			e.shapes = append(e.shapes, s)
		}
		checkExport(t, e)
		seen++
	}
	if seen == 0 {
		t.Skip("no RF-DETR model directory present")
	}
	t.Logf("checked %d real export headers", seen)
}

func TestSplitOutputsMissingPieces(t *testing.T) {
	if _, err := SplitOutputs(fakeOuts([][]int64{{1, 300, 23}}), 23, 256); err == nil {
		t.Error("an export without a boxes output was accepted")
	}
	o, err := SplitOutputs(fakeOuts([][]int64{{1, 300, 4}, {1, 300, 23}}), 23, 256)
	if err != nil || o.Feats.Data != nil {
		t.Errorf("an export without query features: feats %v, err %v; want a zero Tensor", o.Feats.Shape, err)
	}
	o, err = SplitOutputs(fakeOuts([][]int64{{1, 300, 4}}), 23, 0)
	if err != nil || o.Logits.Data != nil {
		t.Errorf("boxes only: logits %v, err %v; want a zero Tensor", o.Logits.Shape, err)
	}
	// An export with boxes and query features but no class head: with the feature width known
	// (textalign's exact/folded modes need nothing else) the features are NOT taken for logits.
	outs := fakeOuts([][]int64{{1, 300, 4}, {1, 300, 256}})
	if o, err = SplitOutputs(outs, 23, 256); err != nil || o.Logits.Data != nil || indexOf(outs, o.Feats) != 1 {
		t.Errorf("boxes+feats, dFeat known: logits %v feats %d, err %v; want no logits, feats 1", o.Logits.Shape, indexOf(outs, o.Feats), err)
	}
	// Width unknown: RF-DETR's first-non-box rule, as its own postprocess would.
	if o, err = SplitOutputs(outs, 23, 0); err != nil || indexOf(outs, o.Logits) != 1 || o.Feats.Data != nil {
		t.Errorf("boxes+feats, dFeat unknown: logits %d feats %v, err %v; want logits 1, no feats", indexOf(outs, o.Logits), o.Feats.Shape, err)
	}
}

// Where the old rules disagreed with each other, the label count decides: an export listing
// query_feats BEFORE its class head names the class head as logits (RF-DETR's first-non-box rule
// would have decoded the features as classes).
func TestSplitOutputsPrefersLabelCountOverOrder(t *testing.T) {
	outs := fakeOuts([][]int64{{1, 300, 4}, {1, 300, 256}, {1, 300, 23}})
	o, err := SplitOutputs(outs, 23, 256)
	if err != nil {
		t.Fatal(err)
	}
	if indexOf(outs, o.Logits) != 2 || indexOf(outs, o.Feats) != 1 {
		t.Errorf("logits/feats = %d/%d, want 2/1", indexOf(outs, o.Logits), indexOf(outs, o.Feats))
	}
}
