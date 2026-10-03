package textalign

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// headRunner answers the roleHead session with an independent implementation of the graph.
type headRunner struct {
	in, out []string
	run     func(map[string]engine.Tensor) ([]engine.Tensor, error)
	calls   int
}

func (h *headRunner) Run(role string, in map[string]engine.Tensor) ([]engine.Tensor, error) {
	if role != roleHead {
		return nil, fmt.Errorf("unexpected role %q", role)
	}
	h.calls++
	return h.run(in)
}
func (h *headRunner) InputNames(string) []string  { return h.in }
func (h *headRunner) OutputNames(string) []string { return h.out }

// graphSpec is export_head_onnx.py's graph written independently, in float64 and in the DIRECT
// form (project every query, normalise, Scale·cos + Bias) — not the folded W + Gram form the Go
// exact head uses. A decoy output comes first, so the logits must be picked BY NAME.
func graphSpec(p *Projection) func(map[string]engine.Tensor) ([]engine.Tensor, error) {
	return func(in map[string]engine.Tensor) ([]engine.Tensor, error) {
		f, t := in[HeadInFeats], in[HeadInText]
		q, dF := int(f.Dim(1)), int(f.Dim(2))
		c, dT := int(t.Dim(0)), int(t.Dim(1))
		if dF != p.DFeat || dT != p.DText || len(f.Data) != q*dF || len(t.Data) != c*dT {
			return nil, fmt.Errorf("graph: bad input shapes %v %v", f.Shape, t.Shape)
		}
		out := make([]float32, q*c)
		z := make([]float64, dT)
		for i := 0; i < q; i++ {
			var ss float64
			for k := 0; k < dT; k++ {
				var acc float64
				for j := 0; j < dF; j++ {
					acc += float64(p.P[k*dF+j]) * float64(f.Data[i*dF+j])
				}
				z[k], ss = acc, ss+acc*acc
			}
			den := math.Sqrt(ss)
			if den == 0 {
				den = 1
			}
			for w := 0; w < c; w++ {
				var dot float64
				for k := 0; k < dT; k++ {
					dot += z[k] * float64(t.Data[w*dT+k])
				}
				out[i*c+w] = float32(float64(p.Scale)*dot/den + float64(p.Bias))
			}
		}
		return []engine.Tensor{engine.F32([]float32{0}, 1), engine.F32(out, 1, int64(q), int64(c))}, nil
	}
}

func sinGen(i int) float32 { return float32(math.Sin(float64(i)*1.7)) * 0.4 }

// synthProjection is a dense VSTXALN1 projection with a non-trivial Scale and Bias.
func synthProjection(t *testing.T, dText, dFeat int, seed int) *Projection {
	t.Helper()
	p := make([]float32, dText*dFeat)
	for i := range p {
		p[i] = sinGen(seed + i)
	}
	pr, err := ParseProjection(buildProj(dText, dFeat, 14.8, -3.8, 0, p))
	if err != nil {
		t.Fatal(err)
	}
	return pr
}

// headInputs: q query features (query 1 all-zero, the degenerate row) and an n-word head whose W
// is folded from n L2-normalised text rows, exactly as headFor builds it.
func headInputs(t *testing.T, pr *Projection, q, n int) (*head, engine.Tensor) {
	t.Helper()
	feats := make([]float32, q*pr.DFeat)
	for i := range feats {
		feats[i] = sinGen(900 + i)
	}
	for j := 0; j < pr.DFeat; j++ {
		feats[1*pr.DFeat+j] = 0
	}
	text := make([][]float32, n)
	classes := make([]string, n)
	for c := range text {
		classes[c] = fmt.Sprintf("word%d", c)
		text[c] = make([]float32, pr.DText)
		var s float64
		for k := range text[c] {
			text[c][k] = sinGen(100 + c*pr.DText + k)
			s += float64(text[c][k]) * float64(text[c][k])
		}
		for k := range text[c] {
			text[c][k] /= float32(math.Sqrt(s))
		}
	}
	w, err := pr.Fold(text)
	if err != nil {
		t.Fatal(err)
	}
	return &head{classes: classes, w: w, text: text}, engine.F32(feats, 1, int64(q), int64(pr.DFeat))
}

func maxAbsDiff(a, b []float32) float64 {
	var m float64
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > m || math.IsNaN(d) {
			m = d
		}
	}
	return m
}

func onnxModel(pr *Projection) *textAlign {
	return &textAlign{proj: pr, cfg: models.Config{Files: map[string]string{roleHead: "head.onnx"}}}
}

// The Go side of the ONNX exact head — input names, text rows passed UNSCALED, feature slicing,
// picking the output by name, adding nothing to the graph's logits — against the Go exact head,
// with the graph implemented independently. Runs without ONNX Runtime.
func TestExactHeadONNXMatchesGoFakeRunner(t *testing.T) {
	pr := synthProjection(t, 24, 8, 0)
	const q, n = 11, 4
	h, feats := headInputs(t, pr, q, n)

	want, err := h.logits(pr, feats.Data, q, true)
	if err != nil {
		t.Fatal(err)
	}
	r := &headRunner{in: []string{HeadInFeats, HeadInText}, out: []string{"decoy", HeadOutLogits}, run: graphSpec(pr)}
	got, err := onnxModel(pr).exactLogitsONNX(h, feats, q, r)
	if err != nil {
		t.Fatal(err)
	}
	if r.calls != 1 {
		t.Errorf("head session ran %d times, want once per request", r.calls)
	}
	d := maxAbsDiff(got, want)
	t.Logf("fake-runner graph vs Go exact head, %d×%d logits: max abs diff %.3g", q, n, d)
	if d > 1e-5 {
		t.Errorf("ONNX-path logits differ from the Go exact head by %.3g (max abs); want ≤ 1e-5", d)
	}
	for c := 0; c < n; c++ { // the all-zero feature: both give exactly Bias
		if got[1*n+c] != pr.Bias || want[1*n+c] != pr.Bias {
			t.Errorf("degenerate query, word %d: onnx %v, go %v, want Bias %v", c, got[1*n+c], want[1*n+c], pr.Bias)
		}
	}
}

// decode routes ONLY the exact head to the session; folded scoring never touches it, and a model
// without files.head never asks for a Runner at all (decode(…, nil) is how the Go tests call it).
func TestDecodeRoutesOnlyExactToONNX(t *testing.T) {
	pr := synthProjection(t, 24, 8, 0)
	const q, n = 6, 3
	h, feats := headInputs(t, pr, q, n)
	cfg := models.Config{Name: "t", Width: 512, Height: 512, Layout: "NCHW", PostType: "detr",
		BoxFormat: "cxcywh", ConfThresh: 0.001, MaxDet: 300}
	rf, err := newRFDETR(cfg, h.classes)
	if err != nil {
		t.Fatal(err)
	}
	h.rf = rf
	bx := make([]float32, q*4)
	for i := 0; i < q; i++ {
		copy(bx[i*4:], []float32{0.1 + 0.1*float32(i), 0.5, 0.1, 0.2})
	}
	boxes := engine.F32(bx, 1, q, 4)
	meta := models.PreprocessMeta{OrigWidth: 640, OrigHeight: 480, ScaleX: 0.8, ScaleY: 512.0 / 480}

	goModel := &textAlign{cfg: cfg, proj: pr}
	want, err := goModel.decode(h, boxes, feats, meta, true, nil)
	if err != nil {
		t.Fatal(err)
	}
	withHead := cfg
	withHead.Files = map[string]string{roleHead: "head.onnx"}
	m := &textAlign{cfg: withHead, proj: pr}
	r := &headRunner{in: []string{HeadInFeats, HeadInText}, out: []string{"decoy", HeadOutLogits}, run: graphSpec(pr)}
	got, err := m.decode(h, boxes, feats, meta, true, r)
	if err != nil {
		t.Fatal(err)
	}
	if r.calls != 1 {
		t.Fatalf("exact decode ran the head session %d times, want 1", r.calls)
	}
	if len(got.Detections) != len(want.Detections) || len(want.Detections) == 0 {
		t.Fatalf("ONNX decode gave %d detections, Go %d", len(got.Detections), len(want.Detections))
	}
	for i := range want.Detections {
		a, b := got.Detections[i], want.Detections[i]
		if a.Class != b.Class || a.BBox != b.BBox || math.Abs(a.Conf-b.Conf) > 1e-6 {
			t.Errorf("detection %d: onnx %+v, go %+v", i, a, b)
		}
	}

	if _, err := m.decode(h, boxes, feats, meta, false, r); err != nil {
		t.Fatal(err)
	}
	if r.calls != 1 {
		t.Errorf("folded decode ran the head session; only the exact head moved to ONNX")
	}
}

// files.head is what makes the head a lifecycle-owned session; without it the role is absent and
// the model is exactly what it was before (no session, no file required on disk).
func TestRolesListHeadOnlyWhenDeclared(t *testing.T) {
	if hasName((&textAlign{}).Roles(), roleHead) {
		t.Errorf("a manifest without files.%s lists the role; lifecycle would demand the file", roleHead)
	}
	if !hasName(onnxModel(nil).Roles(), roleHead) {
		t.Errorf("Roles() = %v, want %q so lifecycle creates the session", onnxModel(nil).Roles(), roleHead)
	}
}

// A head.onnx exported from another proj.bin must be refused, not served next to Go methods that
// still read this proj.bin.
func TestExactHeadONNXRejectsStaleProjection(t *testing.T) {
	pr := synthProjection(t, 24, 8, 0)
	stale := synthProjection(t, 24, 8, 5000)
	h, feats := headInputs(t, pr, 6, 3)
	r := &headRunner{in: []string{HeadInFeats, HeadInText}, out: []string{"decoy", HeadOutLogits}, run: graphSpec(stale)}
	if _, err := onnxModel(pr).exactLogitsONNX(h, feats, 6, r); err == nil ||
		!strings.Contains(err.Error(), "export_head_onnx.py") {
		t.Errorf("a head exported from a different projection was served (or not diagnosed): %v", err)
	}
}

func TestExactHeadONNXRejectsWrongGraph(t *testing.T) {
	pr := synthProjection(t, 24, 8, 0)
	h, feats := headInputs(t, pr, 4, 2)
	m := onnxModel(pr)

	wrongNames := &headRunner{in: []string{"input"}, out: []string{HeadOutLogits},
		run: func(map[string]engine.Tensor) ([]engine.Tensor, error) { return nil, nil }}
	if _, err := m.exactLogitsONNX(h, feats, 4, wrongNames); err == nil || !strings.Contains(err.Error(), "export_head_onnx.py") {
		t.Errorf("a graph without %q/%q was accepted or not diagnosed: %v", HeadInFeats, HeadInText, err)
	}
	wrongShape := &headRunner{in: []string{HeadInFeats, HeadInText}, out: []string{HeadOutLogits},
		run: func(map[string]engine.Tensor) ([]engine.Tensor, error) {
			return []engine.Tensor{engine.F32(make([]float32, 8), 1, 2, 4)}, nil
		}}
	if _, err := m.exactLogitsONNX(h, feats, 4, wrongShape); err == nil {
		t.Error("logits of shape [1 2 4] accepted for 4 queries × 2 words")
	}
	failing := &headRunner{in: []string{HeadInFeats, HeadInText}, out: []string{HeadOutLogits},
		run: func(map[string]engine.Tensor) ([]engine.Tensor, error) { return nil, fmt.Errorf("ORT: bad dims") }}
	if _, err := m.exactLogitsONNX(h, feats, 4, failing); err == nil || !strings.Contains(err.Error(), "[1 4 8]") {
		t.Errorf("a session error should name the shapes that were fed: %v", err)
	}
	if _, err := m.exactLogitsONNX(h, feats, 4, nil); err == nil {
		t.Error("files.head declared but no Runner: accepted")
	}
}

// Against the REAL exported graph on ONNX Runtime: the tiny fixture the fast-path tests use (same
// graph, same file format), and the shipped textalign head when it has been exported. Skipped
// when ORT_DYLIB_PATH is not set.
func TestExactHeadONNXMatchesGoORT(t *testing.T) {
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("ORT_DYLIB_PATH not set; the fake-runner test covers the Go side")
	}
	shipped := filepath.Join("..", "..", "..", "models", "rfdetr-textalign-dec1-siglip")
	cases := []struct {
		name, proj, onnx string
		q, n             int
	}{
		{"tiny", filepath.Join("..", "hybrid", "testdata", "tiny_head.bin"),
			filepath.Join("..", "hybrid", "testdata", "tiny_head.onnx"), 300, 5},
		{"rfdetr-textalign-dec1-siglip", filepath.Join(shipped, projFile), filepath.Join(shipped, "head.onnx"), 300, 22},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := os.Stat(c.onnx); err != nil {
				t.Skipf("%s not present (export it with models/rfdetr-textalign-dec1-siglip/export_head_onnx.py)", c.onnx)
			}
			pr, err := LoadProjection(c.proj)
			if err != nil {
				t.Fatal(err)
			}
			sess, err := engine.NewSession(c.onnx, nil, nil, []engine.Provider{engine.ProviderCPU})
			if err != nil {
				t.Fatal(err)
			}
			defer sess.Close()
			r := &headRunner{in: sess.InputNames(), out: sess.OutputNames(), run: sess.RunNamed}

			h, feats := headInputs(t, pr, c.q, c.n)
			want, err := h.logits(pr, feats.Data, c.q, true)
			if err != nil {
				t.Fatal(err)
			}
			got, err := onnxModel(pr).exactLogitsONNX(h, feats, c.q, r)
			if err != nil {
				t.Fatal(err)
			}
			d := maxAbsDiff(got, want)
			t.Logf("%s: ORT (CPU) vs Go exact head, %d×%d logits: max abs diff %.3g", c.name, c.q, c.n, d)
			if d > 1e-5 {
				t.Errorf("max abs diff %.3g > 1e-5: head.onnx is not the exact head of %s", d, c.proj)
			}
		})
	}
}
