// Package onnxtest writes tiny header-only ONNX files for tests: a ModelProto whose graph
// declares inputs and outputs and nothing else. engine.Inspect / InspectHeader read them like a
// real export; ONNX Runtime cannot run them (there are no nodes), so they suit load-time checks
// that read the graph's declared I/O, never a session.
//
// ModelWith adds what engine.ReadFacts reports (producer, opset, metadata, initializers with
// inline zero data), for tests of `visionserve inspect`.
package onnxtest

import (
	"bytes"
	"os"
	"sort"
	"strconv"
	"testing"
)

// Input is one graph input or output. A dimension < 0 is written as a symbolic dim_param
// (dynamic), which the header reader reports as -1.
type Input struct {
	Name string
	Dims []int64
	// Elem is the ONNX element type (TensorProto.DataType); 0 = float32.
	Elem int32
}

// Options are the model-level facts ModelWith writes.
type Options struct {
	IRVersion       uint64 // 0 = 8
	ProducerName    string // "" = "onnxtest"
	ProducerVersion string
	Opset           uint64            // default-domain opset version; 0 = none written
	Metadata        map[string]string // metadata_props, written in sorted key order
	// Initializers are float32 weights with inline zero raw_data (keep them small). An
	// initializer whose name is also an input is a constant, as in an IR<4 export.
	Initializers []Input
	// External are float32 weights stored in an external-data file (location, offset 0, length
	// 4 × elements); the file itself is not written.
	External []ExternalInit
}

// ExternalInit is an initializer whose data lives in Location.
type ExternalInit struct {
	Name     string
	Dims     []int64
	Location string
}

// Write writes the model to path (float32 tensors) and fails the test on an I/O error.
func Write(t testing.TB, path string, inputs []Input, outputs ...Input) {
	t.Helper()
	if err := os.WriteFile(path, Model(inputs, outputs...), 0o644); err != nil {
		t.Fatal(err)
	}
}

// WriteWith is Write with ModelWith's options.
func WriteWith(t testing.TB, path string, o Options, inputs []Input, outputs ...Input) {
	t.Helper()
	if err := os.WriteFile(path, ModelWith(o, inputs, outputs...), 0o644); err != nil {
		t.Fatal(err)
	}
}

// Model returns the serialized ModelProto (field numbers in engine/onnxheader.go).
func Model(inputs []Input, outputs ...Input) []byte {
	return ModelWith(Options{}, inputs, outputs...)
}

// ModelWith returns the serialized ModelProto with the model-level facts of o.
func ModelWith(o Options, inputs []Input, outputs ...Input) []byte {
	var graph [][]byte
	graph = append(graph, str(2, "onnxtest"))
	for _, w := range o.Initializers {
		n := uint64(1)
		for _, d := range w.Dims {
			n *= uint64(d)
		}
		graph = append(graph, msg(5, packed(1, w.Dims), uintField(2, 1), str(8, w.Name), msg(9, make([]byte, 4*n))))
	}
	for _, w := range o.External {
		n := uint64(1)
		for _, d := range w.Dims {
			n *= uint64(d)
		}
		graph = append(graph, msg(5, packed(1, w.Dims), uintField(2, 1), str(8, w.Name),
			msg(13, str(1, "location"), str(2, w.Location)),
			msg(13, str(1, "offset"), str(2, "0")),
			msg(13, str(1, "length"), str(2, uitoa(4*n))),
			uintField(14, 1)))
	}
	for _, in := range inputs {
		graph = append(graph, msg(11, valueInfo(in)))
	}
	if len(outputs) == 0 {
		outputs = []Input{{Name: "out", Dims: []int64{1}}}
	}
	for _, out := range outputs {
		graph = append(graph, msg(12, valueInfo(out)))
	}
	ir := o.IRVersion
	if ir == 0 {
		ir = 8
	}
	producer := o.ProducerName
	if producer == "" {
		producer = "onnxtest"
	}
	parts := [][]byte{uintField(1, ir), str(2, producer)}
	if o.ProducerVersion != "" {
		parts = append(parts, str(3, o.ProducerVersion))
	}
	parts = append(parts, msg(7, graph...))
	if o.Opset > 0 {
		parts = append(parts, msg(8, str(1, ""), uintField(2, o.Opset)))
	}
	for _, k := range sortedKeys(o.Metadata) {
		parts = append(parts, msg(14, str(1, k), str(2, o.Metadata[k])))
	}
	return bytes.Join(parts, nil)
}

func valueInfo(v Input) []byte {
	dims := make([][]byte, 0, len(v.Dims))
	for i, d := range v.Dims {
		if d < 0 {
			dims = append(dims, msg(1, str(2, "d"+string(rune('0'+i)))))
		} else {
			dims = append(dims, msg(1, uintField(1, uint64(d))))
		}
	}
	elem := uint64(1)
	if v.Elem != 0 {
		elem = uint64(v.Elem)
	}
	tensor := msg(1, uintField(1, elem), msg(2, dims...)) // TypeProto.tensor_type{elem_type, shape}
	return bytes.Join([][]byte{str(1, v.Name), msg(2, tensor)}, nil)
}

func varint(x uint64) []byte {
	var b []byte
	for x >= 0x80 {
		b = append(b, byte(x)|0x80)
		x >>= 7
	}
	return append(b, byte(x))
}

func uintField(num int, v uint64) []byte { return append(varint(uint64(num)<<3), varint(v)...) }

func msg(num int, parts ...[]byte) []byte {
	payload := bytes.Join(parts, nil)
	b := append(varint(uint64(num)<<3|2), varint(uint64(len(payload)))...)
	return append(b, payload...)
}

func str(num int, s string) []byte { return msg(num, []byte(s)) }

// packed writes a repeated int64 field in packed encoding.
func packed(num int, vals []int64) []byte {
	var p []byte
	for _, v := range vals {
		p = append(p, varint(uint64(v))...)
	}
	return msg(num, p)
}

func uitoa(n uint64) string { return strconv.FormatUint(n, 10) }

func sortedKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
