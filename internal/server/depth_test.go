package server

import (
	"encoding/binary"
	"math"
	"testing"
)

func u16bytes(vals ...uint16) []byte {
	b := make([]byte, len(vals)*2)
	for i, v := range vals {
		binary.LittleEndian.PutUint16(b[i*2:], v)
	}
	return b
}

func f32bytes(vals ...float32) []byte {
	b := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}

func TestParseDepthUint16(t *testing.T) {
	// 2x2: 0 (invalid→NaN), 32767 (~0.5), 65535 (1.0), 6553 (~0.1)
	d, dw, dh := parseDepth(u16bytes(0, 32767, 65535, 6553), "uint16", 2, 2, 2, 2)
	if dw != 2 || dh != 2 || len(d) != 4 {
		t.Fatalf("dims = %dx%d len=%d", dw, dh, len(d))
	}
	if !math.IsNaN(float64(d[0])) {
		t.Errorf("0 should be NaN (invalid), got %v", d[0])
	}
	if math.Abs(float64(d[1])-0.5) > 0.01 || float64(d[2]) != 1.0 {
		t.Errorf("normalization wrong: %v %v", d[1], d[2])
	}
}

func TestParseDepthFloat32(t *testing.T) {
	// float kept as-is; ≤0 / NaN → invalid
	d, _, _ := parseDepth(f32bytes(1.5, -1.0, 0.0, 2.5), "float32", 2, 2, 2, 2)
	if float64(d[0]) != 1.5 || float64(d[3]) != 2.5 {
		t.Errorf("float values changed: %v %v", d[0], d[3])
	}
	if !math.IsNaN(float64(d[1])) || !math.IsNaN(float64(d[2])) {
		t.Errorf("≤0 should be NaN: %v %v", d[1], d[2])
	}
}

func TestParseDepthResizeAndBadLen(t *testing.T) {
	// 1x1 → resized to 2x2 (all same)
	d, dw, dh := parseDepth(f32bytes(0.7), "float32", 1, 1, 2, 2)
	if dw != 2 || dh != 2 || len(d) != 4 {
		t.Fatalf("resize dims = %dx%d len=%d", dw, dh, len(d))
	}
	for _, v := range d {
		if math.Abs(float64(v)-0.7) > 1e-6 {
			t.Errorf("resized value = %v, want 0.7", v)
		}
	}
	// wrong byte length → nil
	if d, _, _ := parseDepth(u16bytes(1, 2, 3), "uint16", 2, 2, 2, 2); d != nil {
		t.Error("mismatched byte length should return nil")
	}
}
