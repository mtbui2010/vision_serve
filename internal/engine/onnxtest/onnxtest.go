// Package onnxtest writes tiny header-only ONNX files for tests: a ModelProto whose graph
// declares inputs and outputs and nothing else. engine.Inspect / InspectHeader read them like a
// real export; ONNX Runtime cannot run them (there are no nodes), so they suit load-time checks
// that read the graph's declared I/O, never a session.
package onnxtest

import (
	"bytes"
	"os"
	"testing"
)

// Input is one graph input or output. A dimension < 0 is written as a symbolic dim_param
// (dynamic), which the header reader reports as -1.
type Input struct {
	Name string
	Dims []int64
}

// Write writes the model to path (float32 tensors) and fails the test on an I/O error.
func Write(t testing.TB, path string, inputs []Input, outputs ...Input) {
	t.Helper()
	if err := os.WriteFile(path, Model(inputs, outputs...), 0o644); err != nil {
		t.Fatal(err)
	}
}

// Model returns the serialized ModelProto (field numbers in engine/onnxheader.go).
func Model(inputs []Input, outputs ...Input) []byte {
	var graph [][]byte
	graph = append(graph, str(2, "onnxtest"))
	for _, in := range inputs {
		graph = append(graph, msg(11, valueInfo(in)))
	}
	if len(outputs) == 0 {
		outputs = []Input{{Name: "out", Dims: []int64{1}}}
	}
	for _, out := range outputs {
		graph = append(graph, msg(12, valueInfo(out)))
	}
	return bytes.Join([][]byte{uintField(1, 8), msg(7, graph...)}, nil)
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
	tensor := msg(1, uintField(1, 1), msg(2, dims...)) // TypeProto.tensor_type{elem_type: FLOAT, shape}
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
