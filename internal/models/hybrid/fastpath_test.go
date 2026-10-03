package hybrid

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/models/textalign"
	"visionserve/internal/pipeline"
)

// writeHead builds a VSTXALN1 file with a known P, a and b, so the scoring arithmetic can be
// checked against a value computed independently rather than against itself.
func writeHead(t *testing.T, dir string, dText, dFeat int, a, b float32, P []float32) {
	t.Helper()
	buf := make([]byte, 32+4*len(P))
	copy(buf, "VSTXALN1")
	binary.LittleEndian.PutUint32(buf[8:], uint32(dText))
	binary.LittleEndian.PutUint32(buf[12:], uint32(dFeat))
	binary.LittleEndian.PutUint32(buf[16:], math.Float32bits(a))
	binary.LittleEndian.PutUint32(buf[20:], math.Float32bits(b))
	for i, v := range P {
		binary.LittleEndian.PutUint32(buf[32+4*i:], math.Float32bits(v))
	}
	if err := os.WriteFile(filepath.Join(dir, headFile), buf, 0o644); err != nil {
		t.Fatal(err)
	}
}

// The feature is opt-in and its dependency is enforced: a head without the rescorer is refused at
// load, because FINDINGS §10 measures the head alone at 21.53 mAP against 46.41 with rescoring —
// less than half of what the configuration is worth, served silently.
func TestFastPathRequiresRescorer(t *testing.T) {
	dir := t.TempDir()
	writeHead(t, dir, 4, 2, 1, 0, make([]float32, 8))

	fp, err := newFastPath(models.Config{Dir: dir})
	if err != nil || fp == nil {
		t.Fatalf("newFastPath = %v, %v; want a head", fp, err)
	}
	if fp.budget != defaultBudget {
		t.Errorf("budget = %d, want the measured knee %d", fp.budget, defaultBudget)
	}

	// A manifest with no head at all is the normal case and must not error.
	none, err := newFastPath(models.Config{Dir: t.TempDir()})
	if none != nil || err != nil {
		t.Errorf("absent head → %v, %v; want nil, nil (the router keeps using GroundingDINO)", none, err)
	}
}

func TestHasFastPathNeedsBoth(t *testing.T) {
	for _, c := range []struct {
		name string
		m    *hybrid
		want bool
	}{
		{"neither", &hybrid{}, false},
		{"head only", &hybrid{fp: &fastPath{}}, false},
		{"rescorer only", &hybrid{rs: &pipeline.CropNamer{}}, false},
		{"both", &hybrid{fp: &fastPath{}, rs: &pipeline.CropNamer{}}, true},
	} {
		if got := c.m.hasFastPath(); got != c.want {
			t.Errorf("%s: hasFastPath = %v, want %v", c.name, got, c.want)
		}
	}
}

// The head's arithmetic, against a value computed by hand rather than by the code under test.
// With P = I (padded), z = normalize(f), and the score for word w is a·⟨z, ψ(w)⟩ + b — a RAW
// logit, because rfdetr/postprocess.go documents its input as "[1, Q, C] BEFORE sigmoid" and
// applies the sigmoid itself. Passing a probability would square it.
func TestHeadScoreIsARawLogit(t *testing.T) {
	dir := t.TempDir()
	const dText, dFeat = 3, 3
	P := []float32{1, 0, 0, 0, 1, 0, 0, 0, 1} // identity, row-major [dText][dFeat]
	writeHead(t, dir, dText, dFeat, 2.0, -0.5, P)
	proj, err := textalign.LoadProjection(filepath.Join(dir, headFile))
	if err != nil {
		t.Fatal(err)
	}

	f := []float32{3, 4, 0}   // ‖f‖ = 5, so z = (0.6, 0.8, 0)
	psi := []float32{1, 0, 0} // a word embedding pointing along x
	// cos(z, psi) = 0.6, so the logit is 2.0*0.6 - 0.5 = 0.7.
	var norm float64
	z := make([]float32, dText)
	for k := 0; k < dText; k++ {
		var s float32
		for j, v := range f {
			s += proj.P[k*dFeat+j] * v
		}
		z[k] = s
		norm += float64(s) * float64(s)
	}
	inv := float32(1 / math.Sqrt(norm))
	var dot float32
	for k, zv := range z {
		dot += zv * inv * psi[k]
	}
	got := proj.Scale*dot + proj.Bias
	if math.Abs(float64(got)-0.7) > 1e-6 {
		t.Errorf("logit = %.6f, want 0.700000", got)
	}
	// And the probability the decoder will report from it.
	p := 1 / (1 + math.Exp(-float64(got)))
	if math.Abs(p-0.66818777) > 1e-6 {
		t.Errorf("sigmoid(logit) = %.8f, want 0.66818777", p)
	}
}

// The head is trained against one detector's query-feature width. Serving it on another export
// must fail loudly: dec1 and dec1_holdout produce features of identical shape and mixing them has
// already silently corrupted four trained components in this project's history.
func TestFastDetectRejectsWrongFeatureWidth(t *testing.T) {
	dir := t.TempDir()
	writeHead(t, dir, 4, 8, 1, 0, make([]float32, 32)) // head wants 8-wide features
	fp, err := newFastPath(models.Config{Dir: dir})
	if err != nil {
		t.Fatal(err)
	}
	boxes := engine.F32(make([]float32, 4), 1, 1, 4)
	feats := engine.F32(make([]float32, 5), 1, 1, 5) // detector emits 5-wide
	_, err = fp.detect(boxes, feats, []string{"hat"}, models.PreprocessMeta{}, nil)
	if err == nil {
		t.Fatal("a 5-wide feature against an 8-wide head was accepted; want an error")
	}
	if !strings.Contains(err.Error(), "8-wide") || !strings.Contains(err.Error(), "5") {
		t.Errorf("error should name both widths so the mismatch is diagnosable, got: %v", err)
	}
}

// The optimised scoring path must compute the SAME function as the direct one. It did not: the
// first version multiplied by Projection.Scale after textalign.Fold had already folded it in,
// making every logit 10.28x too large and costing 9.2 mAP in a served run. The bug survived
// because the optimisation was benchmarked (2.9x faster, correctly) and never differenced
// against what it replaced.
//
// An optimisation without an equivalence test is a rewrite.
func TestFoldedScoringMatchesDirect(t *testing.T) {
	dir := t.TempDir()
	const dText, dFeat, n = 12, 6, 3
	rng := func(i int) float32 { return float32(math.Sin(float64(i)*1.7)) * 0.4 }

	P := make([]float32, dText*dFeat)
	for i := range P {
		P[i] = rng(i)
	}
	writeHead(t, dir, dText, dFeat, 10.2816, -4.9465, P) // the shipped head's own a and b
	proj, err := textalign.LoadProjection(filepath.Join(dir, headFile))
	if err != nil {
		t.Fatal(err)
	}

	text := make([][]float32, n)
	for c := range text {
		text[c] = make([]float32, dText)
		var s float64
		for k := range text[c] {
			text[c][k] = rng(100 + c*dText + k)
			s += float64(text[c][k]) * float64(text[c][k])
		}
		for k := range text[c] { // the text tower emits L2-normalised rows
			text[c][k] /= float32(math.Sqrt(s))
		}
	}
	f := make([]float32, dFeat)
	for i := range f {
		f[i] = rng(500 + i)
	}

	// Direct: z = P f, then a·cos(z, psi) + b.
	z := make([]float32, dText)
	var norm float64
	for k := 0; k < dText; k++ {
		var acc float32
		for j, v := range f {
			acc += proj.P[k*dFeat+j] * v
		}
		z[k] = acc
		norm += float64(acc) * float64(acc)
	}
	inv := float32(1 / math.Sqrt(norm))

	folded, err := proj.Fold(text)
	if err != nil {
		t.Fatal(err)
	}
	fInv := float32(1) / proj.ProjNorm(f)

	for c := 0; c < n; c++ {
		var direct float32
		for k, zv := range z {
			direct += zv * inv * text[c][k]
		}
		direct = proj.Scale*direct + proj.Bias

		var dot float32
		for j, v := range f {
			dot += folded[c*dFeat+j] * v
		}
		got := dot*fInv + proj.Bias

		if math.Abs(float64(got-direct)) > 2e-4 {
			t.Errorf("word %d: folded %.6f, direct %.6f — the two paths must agree",
				c, got, direct)
		}
	}
}

// ---------------------------------------------------------------------------------------------
// files.head: the same head on ONNX Runtime (BUGS_TO_FIX #4). An optimisation without an
// equivalence test is a rewrite, so every test below differences the ONNX path against the Go
// head.bin path it replaces, which TestFoldedScoringMatchesDirect in turn holds to the direct form.
// ---------------------------------------------------------------------------------------------

// headRunner is a models.Runner over a single roleHead "session". `run` is the graph.
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

// graphSpec is the export script's graph written independently, in float64 and in the DIRECT
// form (project every query, normalise, Scale·cos + Bias) — not the folded form the Go path uses.
func graphSpec(p *textalign.Projection) func(map[string]engine.Tensor) ([]engine.Tensor, error) {
	return func(in map[string]engine.Tensor) ([]engine.Tensor, error) {
		f, t := in[headInFeats], in[headInText]
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
		// A decoy output first, so the test also proves the logits are picked BY NAME.
		return []engine.Tensor{engine.F32([]float32{0}, 1), engine.F32(out, 1, int64(q), int64(c))}, nil
	}
}

// tinyInputs: q query features (one of them all-zero, the degenerate row) and n L2-normalised
// text rows, from the same deterministic generator as TestFoldedScoringMatchesDirect.
func tinyInputs(q, dFeat, n, dText int) (engine.Tensor, [][]float32) {
	rng := func(i int) float32 { return float32(math.Sin(float64(i)*1.7)) * 0.4 }
	feats := make([]float32, q*dFeat)
	for i := range feats {
		feats[i] = rng(900 + i)
	}
	for j := 0; j < dFeat; j++ {
		feats[1*dFeat+j] = 0
	}
	text := make([][]float32, n)
	for c := range text {
		text[c] = make([]float32, dText)
		var s float64
		for k := range text[c] {
			text[c][k] = rng(100 + c*dText + k)
			s += float64(text[c][k]) * float64(text[c][k])
		}
		for k := range text[c] {
			text[c][k] /= float32(math.Sqrt(s))
		}
	}
	return engine.F32(feats, 1, int64(q), int64(dFeat)), text
}

func maxAbsDiff(a, b []float32) float64 {
	var m float64
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > m {
			m = d
		}
	}
	return m
}

// The Go side of the ONNX path — input names, text flattening, feature slicing, picking the
// output by name, and adding NOTHING to the graph's logits (Scale lives in the graph) — against
// the Go head.bin path, with the graph implemented independently. Runs without ONNX Runtime.
func TestHeadONNXMatchesGoFakeRunner(t *testing.T) {
	proj, err := textalign.LoadProjection(filepath.Join("testdata", "tiny_head.bin"))
	if err != nil {
		t.Fatal(err)
	}
	const q, n = 9, 3
	feats, text := tinyInputs(q, proj.DFeat, n, proj.DText)

	goFP := &fastPath{proj: proj, budget: defaultBudget}
	want, err := goFP.score(feats, text, q, nil)
	if err != nil {
		t.Fatal(err)
	}
	r := &headRunner{in: []string{headInFeats, headInText}, out: []string{"decoy", headOutLogits},
		run: graphSpec(proj)}
	got, err := (&fastPath{onnx: true, budget: defaultBudget}).score(feats, text, q, r)
	if err != nil {
		t.Fatal(err)
	}
	if r.calls != 1 {
		t.Errorf("head session ran %d times, want once per request", r.calls)
	}
	if d := maxAbsDiff(got, want); d > 2e-4 {
		t.Errorf("ONNX-path logits differ from the Go head.bin path by %.3g (max abs); want ≤ 2e-4", d)
	}
	for c := 0; c < n; c++ { // the all-zero feature: both implementations give exactly Bias
		if got[1*n+c] != proj.Bias {
			t.Errorf("degenerate query, word %d: %v, want Bias %v", c, got[1*n+c], proj.Bias)
		}
	}
}

// Against the REAL exported graph on ONNX Runtime: the tiny fixture always, and the shipped head
// when its model directory is present. Skipped when ORT_DYLIB_PATH is not set.
func TestHeadONNXMatchesGoORT(t *testing.T) {
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("ORT_DYLIB_PATH not set; the fake-runner test covers the Go side")
	}
	cases := []struct{ name, bin, onnx string }{
		{"tiny", filepath.Join("testdata", "tiny_head.bin"), filepath.Join("testdata", "tiny_head.onnx")},
		{"shipped", "../../../models/rfdetr-gdino-fastpath/head.bin",
			"../../../models/rfdetr-gdino-fastpath/head.onnx"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := os.Stat(c.onnx); err != nil {
				t.Skipf("%s not present", c.onnx)
			}
			proj, err := textalign.LoadProjection(c.bin)
			if err != nil {
				t.Fatal(err)
			}
			sess, err := engine.NewSession(c.onnx, nil, nil, []engine.Provider{engine.ProviderCPU})
			if err != nil {
				t.Fatal(err)
			}
			defer sess.Close()
			r := &headRunner{in: sess.InputNames(), out: sess.OutputNames(), run: sess.RunNamed}

			const q, n = 300, 5
			feats, text := tinyInputs(q, proj.DFeat, n, proj.DText)
			want, err := (&fastPath{proj: proj}).score(feats, text, q, nil)
			if err != nil {
				t.Fatal(err)
			}
			got, err := (&fastPath{onnx: true}).score(feats, text, q, r)
			if err != nil {
				t.Fatal(err)
			}
			d := maxAbsDiff(got, want)
			t.Logf("%s: ORT (CPU) vs Go head.bin path, %d×%d logits: max abs diff %.3g", c.name, q, n, d)
			if d > 1e-4 {
				t.Errorf("max abs diff %.3g > 1e-4: head.onnx is not the head in head.bin", d)
			}
		})
	}
}

// files.head wins over head.bin, and it is what makes the head a lifecycle-owned session.
func TestFastPathPrefersONNXRole(t *testing.T) {
	dir := t.TempDir()
	writeHead(t, dir, 4, 2, 1, 0, make([]float32, 8)) // a head.bin that must NOT be read
	fp, err := newFastPath(models.Config{Dir: dir, Files: map[string]string{roleHead: "head.onnx"}})
	if err != nil || fp == nil || !fp.onnx || fp.proj != nil {
		t.Fatalf("newFastPath = %+v, %v; want the ONNX head and no head.bin", fp, err)
	}
	m := &hybrid{fp: fp, rs: &pipeline.CropNamer{}}
	if !hasName(m.Roles(), roleHead) {
		t.Errorf("Roles() = %v, want %q so lifecycle creates the session", m.Roles(), roleHead)
	}
	goOnly := &hybrid{fp: &fastPath{proj: &textalign.Projection{}}, rs: &pipeline.CropNamer{}}
	if hasName(goOnly.Roles(), roleHead) {
		t.Errorf("head.bin-only model lists %q in Roles(); lifecycle would demand files.head", roleHead)
	}
}

func TestHeadONNXRejectsWrongGraph(t *testing.T) {
	feats, text := tinyInputs(4, 6, 2, 12)
	fp := &fastPath{onnx: true}

	wrongNames := &headRunner{in: []string{"input"}, out: []string{headOutLogits},
		run: func(map[string]engine.Tensor) ([]engine.Tensor, error) { return nil, nil }}
	if _, err := fp.score(feats, text, 4, wrongNames); err == nil || !strings.Contains(err.Error(), "export_head_onnx.py") {
		t.Errorf("a graph without %q/%q was accepted or not diagnosed: %v", headInFeats, headInText, err)
	}

	wrongShape := &headRunner{in: []string{headInFeats, headInText}, out: []string{headOutLogits},
		run: func(map[string]engine.Tensor) ([]engine.Tensor, error) {
			return []engine.Tensor{engine.F32(make([]float32, 8), 1, 2, 4)}, nil
		}}
	if _, err := fp.score(feats, text, 4, wrongShape); err == nil {
		t.Error("logits of shape [1 2 4] accepted for 4 queries × 2 words")
	}

	failing := &headRunner{in: []string{headInFeats, headInText}, out: []string{headOutLogits},
		run: func(map[string]engine.Tensor) ([]engine.Tensor, error) { return nil, fmt.Errorf("ORT: bad dims") }}
	if _, err := fp.score(feats, text, 4, failing); err == nil || !strings.Contains(err.Error(), "[1 4 6]") {
		t.Errorf("a session error should name the shapes that were fed: %v", err)
	}
}
