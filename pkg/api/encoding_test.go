package api

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"
)

// awkward float32 values: every one must survive the trip bit for bit, including the ones JSON
// numbers cannot carry at all (NaN, Inf) and the ones a float64 detour could disturb.
var awkward = []float32{
	0, float32(math.Copysign(0, -1)), 1, -1, 0.1, 1.0 / 3, math.MaxFloat32, -math.MaxFloat32,
	math.SmallestNonzeroFloat32, 1e-40, // subnormal
	float32(math.Inf(1)), float32(math.Inf(-1)),
	math.Float32frombits(0x7fc00001), // NaN with a payload
	math.Float32frombits(0xffbfffff), // negative signalling-pattern NaN
}

func sameBits(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d]: bits %08x, want %08x", what, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

// Encode → JSON wire → decode must restore DepthMap and Embeddings bit-exactly.
func TestArraysBase64RoundTripBitExact(t *testing.T) {
	depth := append([]float32(nil), awkward...)
	depth = append(depth, 0.5, 0.25) // 16 values: 4x4
	emb := [][]float32{awkward[:7], awkward[7:]}
	res := Result{Task: TaskDepth, Model: "m", DepthMap: depth, DepthWidth: 4, DepthHeight: 4, Embeddings: emb}

	enc := res
	enc.EncodeArraysBase64()
	if enc.DepthMap != nil || enc.Embeddings != nil {
		t.Fatal("number arrays must be cleared once base64 is set")
	}
	if enc.EmbeddingsShape[0] != 2 || enc.EmbeddingsShape[1] != 7 {
		t.Fatalf("embeddings_shape = %v, want [2 7]", enc.EmbeddingsShape)
	}
	// The wire bytes are exactly the little-endian float32 buffer (what numpy.frombuffer reads).
	raw, _ := base64.StdEncoding.DecodeString(enc.DepthMapBase64)
	if len(raw) != 4*len(depth) || binary.LittleEndian.Uint32(raw[4*12:]) != 0x7fc00001 {
		t.Fatalf("depth bytes are not little-endian float32: % x", raw[:8])
	}

	body, err := json.Marshal(enc) // NaN/Inf would make this fail as JSON numbers
	if err != nil {
		t.Fatal(err)
	}
	var back Result
	if err := json.Unmarshal(body, &back); err != nil {
		t.Fatal(err)
	}
	if err := back.DecodeArraysBase64(); err != nil {
		t.Fatal(err)
	}
	sameBits(t, "depth", back.DepthMap, depth)
	sameBits(t, "emb[0]", back.Embeddings[0], emb[0])
	sameBits(t, "emb[1]", back.Embeddings[1], emb[1])
	if back.DepthMapBase64 != "" || back.EmbeddingsBase64 != "" || back.EmbeddingsShape != nil {
		t.Fatal("decode must clear the base64 fields")
	}
}

func TestEncodeArraysBase64KeepsRaggedEmbeddingsAsNumbers(t *testing.T) {
	res := Result{Embeddings: [][]float32{{1, 2}, {3}}}
	res.EncodeArraysBase64()
	if res.EmbeddingsBase64 != "" || len(res.Embeddings) != 2 {
		t.Fatalf("ragged embeddings must stay JSON numbers: %+v", res)
	}
}

// A result without arrays must marshal exactly as before: the new fields are omitempty.
func TestDefaultResultJSONUnchanged(t *testing.T) {
	res := Result{Task: TaskDetection, Model: "rf-detr", DurationMs: 1}
	res.EncodeArraysBase64() // nothing to encode
	b, _ := json.Marshal(res)
	if string(b) != `{"task":"detection","model":"rf-detr","duration_ms":1}` {
		t.Fatalf("unexpected wire: %s", b)
	}
}

func TestDecodeArraysBase64RejectsBadShapes(t *testing.T) {
	enc := encodeF32([]float32{1, 2, 3})
	for _, r := range []Result{
		{DepthMapBase64: enc, DepthWidth: 2, DepthHeight: 2},
		{EmbeddingsBase64: enc, EmbeddingsShape: []int{2, 2}},
		{EmbeddingsBase64: enc},
		{DepthMapBase64: "AAA"}, // 2 bytes
		{DepthMapBase64: "!!"},
	} {
		if err := r.DecodeArraysBase64(); err == nil {
			t.Errorf("accepted %+v", r)
		}
	}
}

// A 1920x1080 depth map answered as JSON numbers vs encoding=base64 (go test -bench Encode ./pkg/api).
func BenchmarkEncodeDepth2MP(b *testing.B) {
	depth := make([]float32, 1920*1080)
	for i := range depth {
		depth[i] = float32(i%1000) / 997
	}
	for _, enc := range []string{EncodingJSON, EncodingBase64} {
		b.Run(enc, func(b *testing.B) {
			var n int
			for i := 0; i < b.N; i++ {
				res := Result{Task: TaskDepth, DepthMap: depth, DepthWidth: 1920, DepthHeight: 1080}
				if enc == EncodingBase64 {
					res.EncodeArraysBase64()
				}
				body, err := json.Marshal(res)
				if err != nil {
					b.Fatal(err)
				}
				n = len(body)
			}
			b.ReportMetric(float64(n)/1e6, "MB/answer")
		})
	}
}

func TestParseEncoding(t *testing.T) {
	for in, want := range map[string]string{"": EncodingJSON, "json": EncodingJSON, "BASE64": EncodingBase64, " base64 ": EncodingBase64} {
		if got, err := ParseEncoding(in); err != nil || got != want {
			t.Errorf("ParseEncoding(%q) = %q, %v; want %q", in, got, err, want)
		}
	}
	if _, err := ParseEncoding("msgpack"); err == nil {
		t.Error("unknown encoding accepted")
	}
}
