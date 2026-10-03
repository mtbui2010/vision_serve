package api

import (
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"math"
	"strings"
)

// Encodings of a Result's large float arrays (DepthMap, Embeddings), chosen per request with
// encoding=... (a request field or the query parameter).
//
// JSON numbers are the default and stay the default (backward compatible), but they are slow
// and big for dense arrays: a 1920x1080 depth map measured 334 ms / 21 MB as JSON numbers and
// 56 ms / 10 MB as base64. Base64 also carries NaN and Inf, which JSON numbers cannot.
const (
	EncodingJSON   = "json"
	EncodingBase64 = "base64"
)

// ParseEncoding validates an encoding name. "" means the default, EncodingJSON.
func ParseEncoding(s string) (string, error) {
	switch e := strings.ToLower(strings.TrimSpace(s)); e {
	case "", EncodingJSON:
		return EncodingJSON, nil
	case EncodingBase64:
		return EncodingBase64, nil
	default:
		return "", fmt.Errorf("unknown encoding %q (want %q or %q)", s, EncodingJSON, EncodingBase64)
	}
}

// EncodeArraysBase64 moves DepthMap and Embeddings into their base64 fields: DepthMapBase64
// (shape [DepthHeight, DepthWidth]) and EmbeddingsBase64 + EmbeddingsShape ([N, D]), each the
// base64 of the row-major little-endian float32 bytes, and clears the number arrays.
// Embeddings whose rows differ in length have no [N, D] shape; they stay JSON numbers.
func (r *Result) EncodeArraysBase64() {
	if len(r.DepthMap) > 0 {
		r.DepthMapBase64 = encodeF32(r.DepthMap)
		r.DepthMap = nil
	}
	if n := len(r.Embeddings); n > 0 {
		d := len(r.Embeddings[0])
		flat := make([]float32, 0, n*d)
		for _, row := range r.Embeddings {
			if len(row) != d {
				return // ragged: no single shape
			}
			flat = append(flat, row...)
		}
		r.EmbeddingsBase64 = encodeF32(flat)
		r.EmbeddingsShape = []int{n, d}
		r.Embeddings = nil
	}
}

// DecodeArraysBase64 is the inverse of EncodeArraysBase64 (for Go clients): it restores
// DepthMap and Embeddings from the base64 fields and clears those. A result without base64
// fields is returned unchanged.
func (r *Result) DecodeArraysBase64() error {
	if r.DepthMapBase64 != "" {
		v, err := decodeF32(r.DepthMapBase64)
		if err != nil {
			return fmt.Errorf("depth_map_base64: %w", err)
		}
		if r.DepthWidth > 0 && r.DepthHeight > 0 && len(v) != r.DepthWidth*r.DepthHeight {
			return fmt.Errorf("depth_map_base64: %d values, want depth_height*depth_width = %d",
				len(v), r.DepthWidth*r.DepthHeight)
		}
		r.DepthMap, r.DepthMapBase64 = v, ""
	}
	if r.EmbeddingsBase64 != "" {
		v, err := decodeF32(r.EmbeddingsBase64)
		if err != nil {
			return fmt.Errorf("embeddings_base64: %w", err)
		}
		if len(r.EmbeddingsShape) != 2 || r.EmbeddingsShape[0] < 0 || r.EmbeddingsShape[1] < 0 ||
			r.EmbeddingsShape[0]*r.EmbeddingsShape[1] != len(v) {
			return fmt.Errorf("embeddings_base64: %d values do not match embeddings_shape %v", len(v), r.EmbeddingsShape)
		}
		n, d := r.EmbeddingsShape[0], r.EmbeddingsShape[1]
		r.Embeddings = make([][]float32, n)
		for i := range r.Embeddings {
			r.Embeddings[i] = v[i*d : (i+1)*d : (i+1)*d]
		}
		r.EmbeddingsBase64, r.EmbeddingsShape = "", nil
	}
	return nil
}

func encodeF32(v []float32) string {
	b := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(f))
	}
	return base64.StdEncoding.EncodeToString(b)
}

func decodeF32(s string) ([]float32, error) {
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return nil, err
	}
	if len(b)%4 != 0 {
		return nil, fmt.Errorf("%d bytes is not a whole number of float32", len(b))
	}
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v, nil
}
