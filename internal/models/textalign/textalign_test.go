package textalign

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"

	// registers the "rf-detr" factory the sub-model is built from
	_ "visionserve/internal/models/rfdetr"
)

func TestParseMethod(t *testing.T) {
	for _, c := range []struct {
		in   string
		want scoreMode
	}{
		{"", modeExact}, {"exact", modeExact}, {" Cosine ", modeExact},
		{"folded", modeFolded}, {"LINEAR", modeFolded},
		{"gated", modeGated}, {" Split ", modeGated},
	} {
		got, err := parseMethod(c.in)
		if err != nil {
			t.Fatalf("parseMethod(%q): %v", c.in, err)
		}
		if got != c.want {
			t.Errorf("parseMethod(%q) = %v, want %v", c.in, got, c.want)
		}
	}
	if _, err := parseMethod("nonsense"); err == nil {
		t.Errorf("expected an error for an unknown method")
	}
	// Only the exact mode divides by ‖P f‖; gated deliberately uses the folded head.
	for _, c := range []struct {
		m    scoreMode
		want bool
	}{{modeExact, true}, {modeFolded, false}, {modeGated, false}} {
		if got := c.m.normalize(); got != c.want {
			t.Errorf("%v.normalize() = %v, want %v", c.m, got, c.want)
		}
	}
}

func TestBaseVocab(t *testing.T) {
	got := baseVocab([]string{"cup", " hat ", "N/A", "", "cup"})
	want := []string{"cup", "hat"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("baseVocab = %v, want %v (N/A, blanks and duplicates dropped)", got, want)
	}
}

func TestLoadTemplates(t *testing.T) {
	dir := t.TempDir()
	if got, err := loadTemplates(filepath.Join(dir, templatesFile)); err != nil ||
		!reflect.DeepEqual(got, defaultTemplates) {
		t.Errorf("missing templates.txt → %v, %v; want the default %v", got, err, defaultTemplates)
	}

	path := filepath.Join(dir, templatesFile)
	if err := os.WriteFile(path, []byte("# comment\na photo of a {}.\n\nitap of a {}.\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := loadTemplates(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []string{"a photo of a {}.", "itap of a {}."}) {
		t.Errorf("loadTemplates = %v", got)
	}

	if err := os.WriteFile(path, []byte("a photo of a cup.\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadTemplates(path); err == nil {
		t.Errorf("a template without the {} placeholder must be rejected")
	}
	if err := os.WriteFile(path, []byte("# only comments\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadTemplates(path); err == nil {
		t.Errorf("an empty templates.txt must be rejected")
	}
}

// TestSplitOutputs feeds the EXACT output signature of models/rfdetr-small-etri-qf
// (dets, labels, cross_attn_weights, query_feats) and checks we pick boxes + query_feats
// and ignore the frozen 22-class logits and the attention maps.
func TestSplitOutputs(t *testing.T) {
	const q, dFeat = 4, 8
	dets := engine.F32(make([]float32, q*4), 1, q, 4)
	labels := engine.F32(make([]float32, q*23), 1, q, 23)
	attn := engine.F32(make([]float32, 3*2*q*5), 3, 1, 2, q, 5)
	qf := engine.F32(make([]float32, q*dFeat), 1, q, dFeat)
	qf.Data[0] = 7 // identity marker

	o, err := detectorOutputs([]engine.Tensor{dets, labels, attn, qf}, dFeat, 23)
	if err != nil {
		t.Fatalf("detectorOutputs: %v", err)
	}
	if !reflect.DeepEqual(o.Boxes.Shape, dets.Shape) {
		t.Errorf("boxes shape = %v, want %v", o.Boxes.Shape, dets.Shape)
	}
	if o.Feats.Data[0] != 7 {
		t.Errorf("picked the wrong tensor as query_feats (shape %v)", o.Feats.Shape)
	}

	// A detector WITHOUT query_feats must fail loudly, not silently score garbage.
	if _, err := detectorOutputs([]engine.Tensor{dets, labels}, dFeat, 23); err == nil {
		t.Errorf("expected an error when no output has shape [1,Q,d_feat]")
	}
	if _, err := detectorOutputs([]engine.Tensor{labels}, dFeat, 23); err == nil {
		t.Errorf("expected an error when no output has a last dimension of 4")
	}
}

// TestDecodeBBoxInOriginalCoords is the load-bearing test (CLAUDE.md: "the single most
// common bug"). It drives the real decode path with a synthetic box + query feature and
// checks the emitted BBox is in ORIGINAL image coordinates, [x,y,w,h].
func TestDecodeBBoxInOriginalCoords(t *testing.T) {
	pr := testProjection(t) // dText 3, dFeat 4, scale 2, bias −0.25
	classes := []string{"cup", "hat"}

	// f = [1,1,2,2] → z = P f = [1,2,3]; ẑ = z/‖z‖.
	f := []float32{1, 1, 2, 2}
	n := float32(math.Sqrt(14))
	zhat := []float32{1 / n, 2 / n, 3 / n}
	// class 0's text embedding == ẑ  → cosine 1 → logit 1.75 → sigmoid 0.852 (kept),
	// class 1 is its opposite       → cosine −1 → logit −2.25 → sigmoid 0.095 (dropped).
	tRows := [][]float32{zhat, {-zhat[0], -zhat[1], -zhat[2]}}
	w, err := pr.Fold(tRows)
	if err != nil {
		t.Fatal(err)
	}

	cfg := models.Config{
		Name: "test", Width: 512, Height: 512, Layout: "NCHW",
		PostType: "detr", BoxFormat: "cxcywh", ConfThresh: 0.5, MaxDet: 300,
	}
	rf, err := newRFDETR(cfg, classes)
	if err != nil {
		t.Fatal(err)
	}
	m := &textAlign{cfg: cfg, proj: pr}
	h := &head{classes: classes, w: w, rf: rf}

	// One query: cxcywh (0.5, 0.5, 0.25, 0.25) of the 512×512 letterboxed input
	// → x=192, y=192, w=128, h=128 on the input.
	boxes := engine.F32([]float32{0.5, 0.5, 0.25, 0.25}, 1, 1, 4)
	feats := engine.F32(f, 1, 1, 4)
	// A 640×480 original letterboxed into 512×512: scale 0.8, 64 px of padding top/bottom.
	meta := models.PreprocessMeta{OrigWidth: 640, OrigHeight: 480, ScaleX: 0.8, ScaleY: 0.8, PadX: 0, PadY: 64}

	res, err := m.decode(h, boxes, feats, meta, true)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(res.Detections) != 1 {
		t.Fatalf("got %d detections, want 1 (the opposite class must fall below conf 0.5): %+v",
			len(res.Detections), res.Detections)
	}
	d := res.Detections[0]
	if d.Class != "cup" {
		t.Errorf("class = %q, want \"cup\" (labels come from the requested vocabulary)", d.Class)
	}
	want := [4]float64{240, 160, 160, 160} // (192−0)/0.8, (192−64)/0.8, 128/0.8, 128/0.8
	for i := range want {
		if math.Abs(d.BBox[i]-want[i]) > 1e-6 {
			t.Fatalf("BBox = %v, want %v (ORIGINAL image coords, [x,y,w,h])", d.BBox, want)
		}
	}
	if math.Abs(d.Conf-1/(1+math.Exp(-1.75))) > 1e-4 {
		t.Errorf("conf = %v, want sigmoid(a·1+b) = %v", d.Conf, 1/(1+math.Exp(-1.75)))
	}
}

// TestDecodeFoldedKeepsBoxes: the folded (plain-linear) head must return the SAME box —
// only the score calibration may move.
func TestDecodeFoldedKeepsBoxes(t *testing.T) {
	pr := testProjection(t)
	classes := []string{"cup"}
	f := []float32{1, 1, 2, 2}
	n := float32(math.Sqrt(14))
	w, _ := pr.Fold([][]float32{{1 / n, 2 / n, 3 / n}})
	cfg := models.Config{Name: "test", Width: 512, Height: 512, BoxFormat: "cxcywh", ConfThresh: 0.5, MaxDet: 300}
	rf, err := newRFDETR(cfg, classes)
	if err != nil {
		t.Fatal(err)
	}
	m := &textAlign{cfg: cfg, proj: pr}
	h := &head{classes: classes, w: w, rf: rf}
	boxes := engine.F32([]float32{0.5, 0.5, 0.25, 0.25}, 1, 1, 4)
	feats := engine.F32(f, 1, 1, 4)
	meta := models.PreprocessMeta{OrigWidth: 640, OrigHeight: 480, ScaleX: 0.8, ScaleY: 0.8, PadY: 64}

	exact, err := m.decode(h, boxes, feats, meta, true)
	if err != nil {
		t.Fatal(err)
	}
	folded, err := m.decode(h, boxes, feats, meta, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(exact.Detections) != 1 || len(folded.Detections) != 1 {
		t.Fatalf("exact=%d folded=%d detections, want 1 each", len(exact.Detections), len(folded.Detections))
	}
	if exact.Detections[0].BBox != folded.Detections[0].BBox {
		t.Errorf("folded moved the box: %v vs %v", folded.Detections[0].BBox, exact.Detections[0].BBox)
	}
}

// TestVocabCacheEviction checks the cache is bounded and hands back the same compiled
// head for a repeated vocabulary (the "no clip-text call per request" claim).
func TestVocabCacheEviction(t *testing.T) {
	m := &textAlign{cache: map[string]*head{}}
	first := vocabKey(defaultTemplates, []string{"class0"})
	for i := 0; i < maxVocabCache+5; i++ {
		m.put(vocabKey(defaultTemplates, []string{"class" + string(rune('0'+i))}), &head{})
	}
	if len(m.cache) != maxVocabCache || len(m.order) != maxVocabCache {
		t.Errorf("cache=%d order=%d, want %d each", len(m.cache), len(m.order), maxVocabCache)
	}
	if _, ok := m.cache[first]; ok {
		t.Errorf("the oldest vocabulary should have been evicted first (FIFO)")
	}

	// A repeated vocabulary must map to the same key, i.e. hit the cache and skip clip-text.
	h := &head{classes: []string{"cup"}}
	key := vocabKey(defaultTemplates, []string{"cup"})
	m.put(key, h)
	if m.cache[vocabKey(defaultTemplates, []string{"cup"})] != h {
		t.Errorf("a repeated vocabulary missed the cache")
	}
}
