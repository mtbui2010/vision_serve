package textalign

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// buildProj serialises a VSTXALN1 blob the way the training job must write it — the test
// is therefore also the executable specification of the file format.
func buildProj(dText, dFeat int, scale, bias float32, flags uint32, p []float32) []byte {
	buf := make([]byte, projHeader+4*len(p))
	copy(buf, projMagic)
	binary.LittleEndian.PutUint32(buf[8:], uint32(dText))
	binary.LittleEndian.PutUint32(buf[12:], uint32(dFeat))
	binary.LittleEndian.PutUint32(buf[16:], math.Float32bits(scale))
	binary.LittleEndian.PutUint32(buf[20:], math.Float32bits(bias))
	binary.LittleEndian.PutUint32(buf[24:], flags)
	binary.LittleEndian.PutUint32(buf[28:], 0)
	for i, v := range p {
		binary.LittleEndian.PutUint32(buf[projHeader+4*i:], math.Float32bits(v))
	}
	return buf
}

func testProjection(t *testing.T) *Projection {
	t.Helper()
	// dText=3, dFeat=4, row-major P.
	p := []float32{
		1, 0, 0, 0,
		0, 2, 0, 0,
		0, 0, 0.5, 1,
	}
	pr, err := ParseProjection(buildProj(3, 4, 2, -0.25, FlagUnitNorm, p))
	if err != nil {
		t.Fatalf("ParseProjection: %v", err)
	}
	return pr
}

func TestParseProjectionRoundTrip(t *testing.T) {
	pr := testProjection(t)
	if pr.DText != 3 || pr.DFeat != 4 {
		t.Fatalf("dims = %dx%d, want 3x4", pr.DText, pr.DFeat)
	}
	if pr.Scale != 2 || pr.Bias != -0.25 {
		t.Errorf("scale/bias = %v/%v, want 2/-0.25", pr.Scale, pr.Bias)
	}
	if !pr.UnitNorm() {
		t.Errorf("FlagUnitNorm not decoded")
	}
	if got, want := pr.P[5], float32(2); got != want {
		t.Errorf("P[1][1] = %v, want %v (row-major layout)", got, want)
	}
}

func TestParseProjectionRejectsBadFiles(t *testing.T) {
	good := buildProj(3, 4, 2, 0, 0, make([]float32, 12))

	cases := map[string][]byte{
		"too short":  good[:16],
		"bad magic":  append([]byte("NOTAHEAD"), good[8:]...),
		"size":       good[:len(good)-4],
		"zero dims":  buildProj(0, 4, 1, 0, 0, nil),
		"zero scale": buildProj(3, 4, 0, 0, 0, make([]float32, 12)),
		"huge dims":  buildProj(1<<20, 4, 1, 0, 0, nil),
	}
	for name, raw := range cases {
		if _, err := ParseProjection(raw); err == nil {
			t.Errorf("%s: expected an error, got nil", name)
		}
	}
}

func TestLoadProjectionFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, projFile)
	if err := os.WriteFile(path, buildProj(3, 4, 2, -0.25, 0, make([]float32, 12)), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadProjection(path); err != nil {
		t.Fatalf("LoadProjection: %v", err)
	}
	if _, err := LoadProjection(filepath.Join(dir, "missing.bin")); err == nil {
		t.Errorf("expected an error for a missing file")
	}
}

// TestFoldMatchesReference checks the fold W = a·T̂P against a straight triple loop —
// this identity IS the deployment trick, so it gets an explicit test.
func TestFoldMatchesReference(t *testing.T) {
	pr := testProjection(t)
	rows := [][]float32{
		{1, 0, 0},
		{0.6, 0.8, 0},
		{0.3, -0.4, 0.866},
	}
	w, err := pr.Fold(rows)
	if err != nil {
		t.Fatalf("Fold: %v", err)
	}
	if len(w) != len(rows)*pr.DFeat {
		t.Fatalf("W has %d values, want %d", len(w), len(rows)*pr.DFeat)
	}
	for c, tRow := range rows {
		for j := 0; j < pr.DFeat; j++ {
			var want float32
			for k := 0; k < pr.DText; k++ {
				want += tRow[k] * pr.P[k*pr.DFeat+j]
			}
			want *= pr.Scale
			if got := w[c*pr.DFeat+j]; math.Abs(float64(got-want)) > 1e-6 {
				t.Errorf("W[%d][%d] = %v, want %v", c, j, got, want)
			}
		}
	}
}

func TestFoldRejectsWrongDim(t *testing.T) {
	pr := testProjection(t)
	if _, err := pr.Fold([][]float32{{1, 0}}); err == nil {
		t.Errorf("expected an error for a text embedding of the wrong dim")
	}
	if _, err := pr.Fold(nil); err == nil {
		t.Errorf("expected an error for an empty vocabulary")
	}
}

func TestProjNorm(t *testing.T) {
	pr := testProjection(t)
	f := []float32{1, 1, 2, 2}
	// z = P f = [1, 2, 0.5*2 + 1*2] = [1, 2, 3] → ‖z‖ = sqrt(14)
	if got, want := pr.ProjNorm(f), float32(math.Sqrt(14)); math.Abs(float64(got-want)) > 1e-6 {
		t.Errorf("ProjNorm = %v, want %v", got, want)
	}
	if got := pr.ProjNorm([]float32{1, 2}); got != 0 {
		t.Errorf("ProjNorm on a wrong-length feature = %v, want 0 (no panic)", got)
	}
}

// TestProjNormMatchesDirect: ProjNorm takes the fast route ‖Pf‖² = fᵀ(PᵀP)f. A quadratic
// form can cancel, so check it against the float64 reference on the SHIPPED 512×256 P
// with realistic feature magnitudes (‖f‖ ≈ 3.7 on real query_feats).
func TestProjNormMatchesDirect(t *testing.T) {
	path := filepath.Join("..", "..", "..", "models", "rfdetr-textalign-etri", projFile)
	if _, err := os.Stat(path); err != nil {
		t.Skip("no proj.bin in models/rfdetr-textalign-etri (weights are not committed)")
	}
	pr, err := LoadProjection(path)
	if err != nil {
		t.Fatal(err)
	}
	rnd := uint32(12345)
	next := func() float32 { // xorshift, deterministic and dependency-free
		rnd ^= rnd << 13
		rnd ^= rnd >> 17
		rnd ^= rnd << 5
		return float32(int32(rnd)) / (1 << 31)
	}
	var worst float64
	for it := 0; it < 20; it++ {
		f := make([]float32, pr.DFeat)
		for i := range f {
			f[i] = next() * 0.23 // ‖f‖ ≈ 3.7, like real query_feats
		}
		got, want := float64(pr.ProjNorm(f)), float64(pr.projNormDirect(f))
		if rel := math.Abs(got-want) / want; rel > worst {
			worst = rel
		}
	}
	if worst > 1e-5 {
		t.Errorf("Gram-based ProjNorm differs from the direct one by %.3g (relative)", worst)
	}
	t.Logf("max relative error of the Gram-based ‖Pf‖ vs float64 reference: %.3g", worst)
}

// TestShippedProjection validates the side-car actually shipped in the model directory —
// it is what a request will load, so its contract (512×256, non-zero scale) is checked here.
func TestShippedProjection(t *testing.T) {
	path := filepath.Join("..", "..", "..", "models", "rfdetr-textalign-etri", projFile)
	if _, err := os.Stat(path); err != nil {
		t.Skip("no proj.bin in models/rfdetr-textalign-etri (weights are not committed)")
	}
	pr, err := LoadProjection(path)
	if err != nil {
		t.Fatalf("LoadProjection: %v", err)
	}
	if pr.DText != 512 || pr.DFeat != 256 {
		t.Errorf("shipped P is %dx%d, want 512x256 (CLIP ViT-B/32 × RFDETRSmall hidden dim)",
			pr.DText, pr.DFeat)
	}
	if pr.Scale <= 0 {
		t.Errorf("shipped scale = %v, want > 0", pr.Scale)
	}
}
