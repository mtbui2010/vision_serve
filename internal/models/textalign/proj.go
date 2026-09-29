package textalign

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

// ---------------------------------------------------------------------------
// The head-B side-car: the projection matrix P.
//
// P maps a 256-d object-query feature f into the TEACHER's text space. d_text is a field of
// the file, not a constant: 512 for CLIP ViT-B/32, 768 for SigLIP base, 1152 for so400m — see
// dim_test.go, which exercises all three through Fold and the gated decode. It is the
// ONLY trained tensor of head B and it lives NEXT TO the manifest as a small
// binary blob (512×256 f32 = 512 KB), NOT inside the ONNX graph — so retraining
// the head or changing the vocabulary never requires an ONNX re-export.
//
// FILE FORMAT "VSTXALN1" (little-endian):
//
//	offset size  field
//	     0    8  magic, ASCII "VSTXALN1"
//	     8    4  uint32  d_text  (rows of P, = the teacher's text dim; 512 CLIP ViT-B/32,
//	                              768 SigLIP base, 1152 SigLIP so400m)
//	    12    4  uint32  d_feat  (cols of P, = detector hidden dim, 256 for RFDETRSmall)
//	    16    4  float32 scale a (logit scale;   init 1/0.07 ≈ 14.2857)
//	    20    4  float32 bias  b (logit bias;    init −4.6 = focal prior π=0.01)
//	    24    4  uint32  flags  (bit0 = trained WITH the (‖P f‖−1)² penalty)
//	    28    4  uint32  reserved (must be 0)
//	    32   ..  float32 P, ROW-MAJOR [d_text][d_feat]
//
// Row-major [d_text][d_feat] means row k of P is contiguous, so z_k = ⟨P[k], f⟩
// is a contiguous dot product — the layout the scoring loop below wants.
// ---------------------------------------------------------------------------

const (
	projMagic  = "VSTXALN1"
	projHeader = 32

	// FlagUnitNorm marks a P trained with the (‖P f‖−1)² penalty, i.e. one for which
	// dropping the normalisation (the "folded", plain-linear form) is safe. Purely
	// informational: the model reports it, it never silently changes the math.
	FlagUnitNorm uint32 = 1 << 0

	// Sanity bounds, so a corrupt/foreign file fails loudly instead of allocating GBs.
	maxDim = 8192
)

// Projection is head B's trained projection plus its logit calibration (a, b).
type Projection struct {
	DText int // rows of P (CLIP text dim)
	DFeat int // cols of P (detector query-feature dim)
	Scale float32
	Bias  float32
	Flags uint32
	P     []float32 // row-major [DText][DFeat]

	// gram is Gᵀ = PᵀP (DFeat × DFeat, row-major), precomputed once at load so the
	// per-query ‖P f‖ costs a 256×256 quadratic form instead of the 512×256 product
	// P f — measured 2× cheaper, and it is what makes the EXACT (cosine) head
	// affordable. See ProjNorm.
	gram []float32
}

// UnitNorm reports whether P was trained with the unit-norm penalty (see FlagUnitNorm).
func (p *Projection) UnitNorm() bool { return p.Flags&FlagUnitNorm != 0 }

// LoadProjection reads a VSTXALN1 side-car from disk.
func LoadProjection(path string) (*Projection, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("textalign: cannot read projection %s: %w", path, err)
	}
	return ParseProjection(raw)
}

// ParseProjection parses a VSTXALN1 blob (split out from LoadProjection so it is testable
// without touching the filesystem).
func ParseProjection(raw []byte) (*Projection, error) {
	if len(raw) < projHeader {
		return nil, fmt.Errorf("textalign: projection too short (%d bytes, need ≥ %d)", len(raw), projHeader)
	}
	if string(raw[:8]) != projMagic {
		return nil, fmt.Errorf("textalign: bad projection magic %q (want %q)", string(raw[:8]), projMagic)
	}
	dText := int(binary.LittleEndian.Uint32(raw[8:12]))
	dFeat := int(binary.LittleEndian.Uint32(raw[12:16]))
	if dText <= 0 || dFeat <= 0 || dText > maxDim || dFeat > maxDim {
		return nil, fmt.Errorf("textalign: implausible projection dims %dx%d", dText, dFeat)
	}
	scale := math.Float32frombits(binary.LittleEndian.Uint32(raw[16:20]))
	bias := math.Float32frombits(binary.LittleEndian.Uint32(raw[20:24]))
	flags := binary.LittleEndian.Uint32(raw[24:28])

	n := dText * dFeat
	want := projHeader + 4*n
	if len(raw) != want {
		return nil, fmt.Errorf("textalign: projection size %d != %d (header + %dx%d f32)",
			len(raw), want, dText, dFeat)
	}
	if scale == 0 {
		return nil, fmt.Errorf("textalign: projection scale is 0 — logits would collapse to the bias")
	}

	p := make([]float32, n)
	for i := 0; i < n; i++ {
		p[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[projHeader+4*i : projHeader+4*i+4]))
	}
	pr := &Projection{DText: dText, DFeat: dFeat, Scale: scale, Bias: bias, Flags: flags, P: p}
	pr.gram = pr.computeGram()
	return pr, nil
}

// computeGram builds G = PᵀP (accumulated in float64; G is symmetric, so only the upper
// triangle is computed). One-off cost at load: d_feat²·d_text/2 MACs ≈ 17 M for 512×256.
func (p *Projection) computeGram() []float32 {
	n := p.DFeat
	g := make([]float32, n*n)
	acc := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			acc[j] = 0
		}
		for k := 0; k < p.DText; k++ {
			row := p.P[k*n : (k+1)*n]
			v := float64(row[i])
			if v == 0 {
				continue
			}
			for j := i; j < n; j++ {
				acc[j] += v * float64(row[j])
			}
		}
		for j := i; j < n; j++ {
			g[i*n+j] = float32(acc[j])
			g[j*n+i] = float32(acc[j])
		}
	}
	return g
}

// Fold precomputes the deploy-time class matrix W = a · T̂ P  (shape [C, DFeat], row-major).
//
// This is the whole point of the design: with the vocabulary known, the CLIP text tower and
// the projection collapse into ONE small matrix with exactly the role of RF-DETR's original
// class_embed. Changing the vocabulary swaps W — no ONNX re-export, no second heavy session.
//
// rows must be L2-normalised text embeddings (T̂), one per class, each of length DText.
func (p *Projection) Fold(rows [][]float32) ([]float32, error) {
	if len(rows) == 0 {
		return nil, fmt.Errorf("textalign: no text embeddings to fold")
	}
	w := make([]float32, len(rows)*p.DFeat)
	for c, t := range rows {
		if len(t) != p.DText {
			return nil, fmt.Errorf("textalign: text embedding %d has dim %d, projection expects %d",
				c, len(t), p.DText)
		}
		out := w[c*p.DFeat : (c+1)*p.DFeat]
		for k := 0; k < p.DText; k++ {
			tk := t[k] * p.Scale
			if tk == 0 {
				continue
			}
			row := p.P[k*p.DFeat : (k+1)*p.DFeat]
			for j := range out {
				out[j] += tk * row[j]
			}
		}
	}
	return w, nil
}

// ProjNorm returns ‖P f‖ for a single query feature — the per-query scalar the EXACT
// (cosine) head divides by. It uses the precomputed Gram matrix:
//
//	‖P f‖² = fᵀ(PᵀP)f
//
// which is a d_feat² quadratic form (65 k MACs) instead of the d_text×d_feat product
// P f (131 k MACs) — the whole point being that the exact head must not cost more than
// the detector it decorates.
//
// It returns 0 when len(f) != DFeat (callers validate the feature tensor once, before
// the per-query loop).
func (p *Projection) ProjNorm(f []float32) float32 {
	n := p.DFeat
	if len(f) != n || len(p.gram) != n*n {
		return 0
	}
	var sum float64
	for i, fi := range f {
		if fi == 0 {
			continue
		}
		row := p.gram[i*n : (i+1)*n]
		var s float32
		for j, fj := range f {
			s += row[j] * fj
		}
		sum += float64(fi) * float64(s)
	}
	if sum <= 0 {
		return 0 // numerically degenerate (G is PSD, so this means f ≈ ker P)
	}
	return float32(math.Sqrt(sum))
}

// projNormDirect computes ‖P f‖ straight from P, in float64. It is the reference the
// Gram-based ProjNorm is tested against (see proj_test.go).
func (p *Projection) projNormDirect(f []float32) float32 {
	if len(f) != p.DFeat {
		return 0
	}
	var sum float64
	for k := 0; k < p.DText; k++ {
		row := p.P[k*p.DFeat : (k+1)*p.DFeat]
		var z float64
		for j, fv := range f {
			z += float64(row[j]) * float64(fv)
		}
		sum += z * z
	}
	return float32(math.Sqrt(sum))
}
