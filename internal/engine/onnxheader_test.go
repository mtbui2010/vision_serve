package engine

import (
	"bytes"
	"encoding/hex"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
)

// A minimal protobuf encoder, enough to hand-build ModelProto bytes.

func pbVarint(x uint64) []byte {
	var b []byte
	for x >= 0x80 {
		b = append(b, byte(x)|0x80)
		x >>= 7
	}
	return append(b, byte(x))
}

func pbKey(num, wt int) []byte { return pbVarint(uint64(num)<<3 | uint64(wt)) }

func pbBytes(num int, payload ...[]byte) []byte {
	p := bytes.Join(payload, nil)
	return cat(pbKey(num, wireBytes), pbVarint(uint64(len(p))), p)
}

func pbString(num int, s string) []byte { return pbBytes(num, []byte(s)) }

func pbUint(num int, v uint64) []byte { return cat(pbKey(num, wireVarint), pbVarint(v)) }

func cat(parts ...[]byte) []byte { return bytes.Join(parts, nil) }

// ONNX builders (field numbers from onnxheader.go).

func dimValue(v int64) []byte     { return pbBytes(fieldShapeDim, pbUint(fieldDimValue, uint64(v))) }
func dimParam(s string) []byte    { return pbBytes(fieldShapeDim, pbString(fieldDimParam, s)) }
func dimUnknown() []byte          { return pbBytes(fieldShapeDim) }
func shape(dims ...[]byte) []byte { return pbBytes(fieldTensorTypeShape, dims...) }

// tensorTypeOf is a TypeProto holding tensor_type{elem_type, shape?}.
func tensorTypeOf(elem int32, shapeField []byte) []byte {
	return pbBytes(fieldTypeTensor, pbUint(fieldTensorTypeElem, uint64(elem)), shapeField)
}

func valueInfoField(num int, name string, typ []byte) []byte {
	return pbBytes(num, pbString(fieldValueInfoName, name), pbBytes(fieldValueInfoType, typ))
}

func input(name string, typ []byte) []byte  { return valueInfoField(fieldGraphInput, name, typ) }
func output(name string, typ []byte) []byte { return valueInfoField(fieldGraphOutput, name, typ) }

// initializer is a float TensorProto with dims, a name and n bytes of inline raw_data.
func initializer(name string, n int) []byte {
	return pbBytes(fieldGraphInitializer,
		pbBytes(1, pbVarint(uint64(n/4))), // dims, packed
		pbUint(2, 1),                      // data_type FLOAT
		pbString(fieldTensorName, name),
		pbBytes(9, make([]byte, n)), // raw_data
	)
}

// externalInitializer is a TensorProto whose data lives in a sidecar that does not exist.
func externalInitializer(name string) []byte {
	entry := func(k, v string) []byte { return pbBytes(13, pbString(1, k), pbString(2, v)) }
	return pbBytes(fieldGraphInitializer,
		pbBytes(1, pbVarint(1024)),
		pbUint(2, 1),
		pbString(fieldTensorName, name),
		entry("location", "missing.onnx.data"),
		entry("offset", "0"),
		entry("length", "4096"),
		pbUint(14, 1), // data_location EXTERNAL
	)
}

func node(op string, in, out string) []byte {
	return pbBytes(fieldGraphNode, pbString(1, in), pbString(2, out), pbString(3, op+"_0"), pbString(4, op))
}

// unknownFields carries one unknown field of every wire type the reader must step over.
func unknownFields(num int) []byte {
	return cat(
		pbUint(num, 1<<40),
		pbKey(num+1, wireFixed64), make([]byte, 8),
		pbKey(num+2, wireFixed32), make([]byte, 4),
		pbString(num+3, "unknown"),
	)
}

func model(graphs ...[]byte) []byte {
	parts := [][]byte{pbUint(1, 8), pbString(2, "onnxheader_test"), unknownFields(100)}
	for _, g := range graphs {
		parts = append(parts, pbBytes(fieldModelGraph, g))
	}
	return cat(parts...)
}

// syntheticModel exercises every rule the reader implements; want is what ORT would report.
func syntheticModel() (raw []byte, wantIn, wantOut []IOInfo) {
	f32 := int32(1)
	g1 := cat(
		node("Conv", "image", "a"),
		node("Relu", "a", "b"),
		pbString(2, "graph-name"),
		initializer("conv.weight", 4<<20), // 4 MiB of inline weights, never read
		externalInitializer("ext.weight"),
		// sparse_initializer: values (TensorProto) carries the name, indices and dims follow.
		pbBytes(fieldGraphSparseInitializer,
			pbBytes(fieldSparseTensorVals, pbString(fieldTensorName, "sparse.w"), pbBytes(9, make([]byte, 64))),
			pbBytes(2, pbString(fieldTensorName, "sparse.idx")),
			pbBytes(3, pbVarint(8)),
		),
		pbString(10, "doc string"),
		input("image", tensorTypeOf(f32, shape(dimParam("batch"), dimValue(3), dimUnknown(), dimValue(800)))),
		input("conv.weight", tensorTypeOf(f32, shape(dimValue(1)))), // IR<4 style: a constant, not an input
		input("sparse.w", tensorTypeOf(f32, shape(dimValue(8)))),    // a sparse constant
		input("ext.weight", tensorTypeOf(f32, shape(dimValue(1024)))),
		input("tokens", tensorTypeOf(7, shape(dimValue(1), dimParam("seq")))),
		input("scale", tensorTypeOf(f32, shape())), // scalar: shape present, no dims
		input("anyrank", tensorTypeOf(9, nil)),     // no shape: unknown rank
		input("seq", pbBytes(fieldTypeSequence, pbBytes(1, tensorTypeOf(f32, nil)))),
		// unknown fields inside ValueInfoProto, TypeProto, Tensor, shape and dimension
		pbBytes(fieldGraphInput,
			pbString(fieldValueInfoName, "noisy"),
			unknownFields(20),
			pbBytes(fieldValueInfoType,
				unknownFields(30),
				pbString(6, "denotation"),
				pbBytes(fieldTypeTensor,
					unknownFields(40),
					pbUint(fieldTensorTypeElem, 10),
					pbBytes(fieldTensorTypeShape,
						unknownFields(50),
						pbBytes(fieldShapeDim, pbUint(fieldDimValue, 4), pbString(3, "denot"), unknownFields(60)),
						pbBytes(fieldShapeDim, pbUint(fieldDimValue, 5), pbString(fieldDimParam, "later_param_wins")),
					),
				),
			),
			pbString(3, "doc"),
		),
		pbUint(fieldGraphInput, 7), // a known field number with the wrong wire type: unknown, skipped
		output("boxes", tensorTypeOf(f32, shape(dimValue(1), dimValue(900), dimValue(4)))),
		pbBytes(13, pbString(1, "b"), pbBytes(2, tensorTypeOf(f32, nil))), // value_info: ignored
		unknownFields(200),
	)
	// A second occurrence of ModelProto.graph merges into the first: its repeated fields append.
	g2 := cat(
		initializer("late.const", 16),
		input("late.const", tensorTypeOf(f32, shape(dimValue(4)))),
		output("sparse_out", pbBytes(fieldTypeSparseTensor, pbUint(fieldTensorTypeElem, 1), shape(dimValue(10), dimValue(10)))),
		output("neg", tensorTypeOf(f32, shape(dimValue(-1), dimValue(2)))),
	)
	wantIn = []IOInfo{
		{Name: "image", Shape: []int64{-1, 3, -1, 800}, ElemType: 1},
		{Name: "tokens", Shape: []int64{1, -1}, ElemType: 7},
		{Name: "scale", ElemType: 1},
		{Name: "anyrank", ElemType: 9},
		{Name: "seq"},
		{Name: "noisy", Shape: []int64{4, -1}, ElemType: 10},
	}
	wantOut = []IOInfo{
		{Name: "boxes", Shape: []int64{1, 900, 4}, ElemType: 1},
		{Name: "sparse_out", Shape: []int64{10, 10}, ElemType: 1},
		{Name: "neg", Shape: []int64{-1, 2}, ElemType: 1},
	}
	return model(g1, g2), wantIn, wantOut
}

// countingReaderAt counts the bytes read, to prove the weights are stepped over.
type countingReaderAt struct {
	r *bytes.Reader
	n atomic.Int64
}

func (c *countingReaderAt) ReadAt(p []byte, off int64) (int, error) {
	k, err := c.r.ReadAt(p, off)
	c.n.Add(int64(k))
	return k, err
}

func TestONNXHeaderSynthetic(t *testing.T) {
	raw, wantIn, wantOut := syntheticModel()
	cr := &countingReaderAt{r: bytes.NewReader(raw)}
	in, out, err := parseONNXHeader(cr, int64(len(raw)), headerBufSize)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(in, wantIn) {
		t.Errorf("inputs\n got %+v\nwant %+v", in, wantIn)
	}
	if !reflect.DeepEqual(out, wantOut) {
		t.Errorf("outputs\n got %+v\nwant %+v", out, wantOut)
	}
	// One window per region the walk lands in; the 4 MiB raw_data must not be read.
	if read := cr.n.Load(); read > 8*headerBufSize {
		t.Errorf("read %d of %d bytes: the weights were not skipped", read, len(raw))
	}

	// Window boundaries must not matter: varints, keys and names split across refills, and
	// names longer than the window (read directly), give the same answer.
	for _, size := range []int{1, 2, 3, 5, 7, 8, 13, 64, 1000} {
		in, out, err := parseONNXHeader(bytes.NewReader(raw), int64(len(raw)), size)
		if err != nil {
			t.Fatalf("window %d: %v", size, err)
		}
		if !reflect.DeepEqual(in, wantIn) || !reflect.DeepEqual(out, wantOut) {
			t.Fatalf("window %d: got %+v / %+v", size, in, out)
		}
	}
}

// The embedded identity model (device_test.go) is a real exporter's bytes: x, y float [1,3].
func TestONNXHeaderIdentityModel(t *testing.T) {
	raw, err := hex.DecodeString(identityONNX)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "identity.onnx")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	wantIn := []IOInfo{{Name: "x", Shape: []int64{1, 3}, ElemType: 1}}
	wantOut := []IOInfo{{Name: "y", Shape: []int64{1, 3}, ElemType: 1}}
	in, out, err := Inspect(path) // no ORT needed on this path
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(in, wantIn) || !reflect.DeepEqual(out, wantOut) {
		t.Fatalf("Inspect = %+v / %+v, want %+v / %+v", in, out, wantIn, wantOut)
	}

	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("needs ORT_DYLIB_PATH to compare with ONNX Runtime")
	}
	oin, oout, err := inspectORT(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(in, oin) || !reflect.DeepEqual(out, oout) {
		t.Fatalf("header %+v / %+v, ORT %+v / %+v", in, out, oin, oout)
	}
}

func TestONNXHeaderMalformed(t *testing.T) {
	raw, _, _ := syntheticModel()
	// The graph is the model's last field, so every strict prefix cuts it (or drops it).
	for _, n := range []int{0, 1, 5, 40, len(raw) / 2, len(raw) - (4 << 20), len(raw) - 1} {
		if _, _, err := parseONNXHeader(bytes.NewReader(raw[:n]), int64(n), headerBufSize); err == nil {
			t.Errorf("%d-byte prefix of a %d-byte model parsed without error", n, len(raw))
		}
	}
	// Every prefix of a small model, through a tiny window: an error, never a panic.
	small := model(cat(input("x", tensorTypeOf(1, shape(dimValue(1), dimParam("n")))),
		output("y", tensorTypeOf(1, shape(dimValue(1))))))
	for n := 0; n < len(small); n++ {
		if _, _, err := parseONNXHeader(bytes.NewReader(small[:n]), int64(n), 3); err == nil {
			t.Errorf("%d-byte prefix of the small model parsed without error", n)
		}
	}

	g := cat(output("y", tensorTypeOf(1, nil)))
	for name, b := range map[string][]byte{
		"not protobuf":         []byte("this is a text file, not an ONNX model\n"),
		"no graph":             cat(pbUint(1, 8), pbString(2, "x")),
		"graph without output": model(input("x", tensorTypeOf(1, nil))),
		"length overruns":      cat(pbKey(fieldModelGraph, wireBytes), pbVarint(1000), g),
		"inner length overruns parent": model(cat(pbKey(fieldGraphOutput, wireBytes), pbVarint(2),
			pbString(fieldValueInfoName, "y"))),
		"overlong varint":   cat(bytes.Repeat([]byte{0xff}, 11), []byte{1}),
		"field number 0":    cat(pbKey(0, wireVarint), pbVarint(1), pbBytes(fieldModelGraph, g)),
		"group wire type":   cat(pbKey(3, wireStart), pbBytes(fieldModelGraph, g)),
		"wire type 6":       cat(pbKey(3, 6), pbBytes(fieldModelGraph, g)),
		"truncated fixed64": cat(pbBytes(fieldModelGraph, g), pbKey(9, wireFixed64), []byte{1, 2}),
		"huge name": model(cat(pbBytes(fieldGraphOutput, pbBytes(fieldValueInfoName,
			make([]byte, maxNameLen+1))))),
	} {
		if in, out, err := parseONNXHeader(bytes.NewReader(b), int64(len(b)), headerBufSize); err == nil {
			t.Errorf("%s: parsed as %+v / %+v, want an error", name, in, out)
		}
	}
}

// Inspect reports a missing file directly: it neither asks ORT nor logs a header fallback.
func TestInspectMissingFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "absent.onnx")
	_, _, err := Inspect(path)
	if !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("Inspect(missing) = %v, want fs.ErrNotExist", err)
	}
	if _, logged := inspectFallbackLogged.Load(path); logged {
		t.Fatal("a missing file went down the ORT fallback")
	}
}

// A file the header reader rejects goes to ORT, and the fallback is logged once per file.
func TestInspectFallsBackToORT(t *testing.T) {
	path := filepath.Join(t.TempDir(), "garbage.onnx")
	if err := os.WriteFile(path, []byte("not a model"), 0o644); err != nil {
		t.Fatal(err)
	}
	var lines []string
	for i := 0; i < 3; i++ {
		out := captureStderrForTest(t, func() {
			if _, _, err := Inspect(path); err == nil {
				t.Error("Inspect(garbage) succeeded")
			}
		})
		if s := strings.TrimSpace(out); s != "" {
			lines = append(lines, s)
		}
	}
	var fallbacks int
	for _, l := range lines {
		fallbacks += strings.Count(l, "could not read the ONNX header")
	}
	if fallbacks != 1 {
		t.Fatalf("fallback logged %d times over 3 calls, want 1:\n%s", fallbacks, strings.Join(lines, "\n"))
	}
}

// captureStderrForTest returns what fn writes to os.Stderr (the Go-level file only).
func captureStderrForTest(t *testing.T, fn func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	old := os.Stderr
	os.Stderr = w
	done := make(chan string)
	go func() {
		var b bytes.Buffer
		_, _ = b.ReadFrom(r)
		done <- b.String()
	}()
	fn()
	os.Stderr = old
	_ = w.Close()
	return <-done
}
