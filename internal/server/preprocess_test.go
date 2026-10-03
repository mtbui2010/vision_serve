package server

import (
	"encoding/base64"
	"encoding/binary"
	"math"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/lifecycle"
	"visionserve/internal/models"
)

// The wire format is numpy.frombuffer(b64decode(data), dtype).reshape(shape); it must round-trip
// bit-for-bit, or a parity check built on it would report the encoder's error as the model's.
func TestEncodePreprocessRoundTrip(t *testing.T) {
	f := []float32{0, -2.1179039, 1.9431373, float32(math.Inf(-1))}
	ids := []int64{262, 1, -1, 1 << 40}
	res := lifecycle.PreprocessResult{
		Inputs: []lifecycle.NamedTensor{
			{Role: "model", Name: "pixel_values", Tensor: engine.F32(f, 1, 4)},
			{Role: "model", Name: "input_ids", Tensor: engine.I64(ids, 1, 4)},
		},
		Meta: &models.PreprocessMeta{OrigWidth: 848, OrigHeight: 480, ScaleX: 0.6, ScaleY: 1.06},
	}
	out := encodePreprocess("m", res)
	if out.Inputs[0].Dtype != "float32" || out.Inputs[1].Dtype != "int64" || out.Meta.OrigWidth != 848 {
		t.Fatalf("unexpected header: %+v", out)
	}
	b, _ := base64.StdEncoding.DecodeString(out.Inputs[0].Data)
	for i, want := range f {
		if got := math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:])); got != want && !(math.IsInf(float64(got), -1) && math.IsInf(float64(want), -1)) {
			t.Fatalf("float %d: got %v want %v", i, got, want)
		}
	}
	b, _ = base64.StdEncoding.DecodeString(out.Inputs[1].Data)
	for i, want := range ids {
		if got := int64(binary.LittleEndian.Uint64(b[8*i:])); got != want {
			t.Fatalf("int64 %d: got %v want %v", i, got, want)
		}
	}
}
