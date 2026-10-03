package engine

// A pure-Go reader for the part of an ONNX file Inspect needs: the graph's input and output
// names, shapes and element types. ORT's own probe (GetInputOutputInfo) builds a whole session
// for that — 9-19 s for the 695 MB GroundingDINO graph — and NewSession paid it a second time
// for every role whose I/O names the manifest leaves out.
//
// An .onnx file is one serialized ModelProto. This walks its protobuf wire format through an
// io.ReaderAt and steps over every node, initializer and value_info by its length prefix, so the
// weights are never read: only the few bytes of keys and lengths in front of them, plus the
// inputs and outputs themselves. Weights in an external-data sidecar (.data) are never opened.
//
// The schema subset, from onnx/onnx.proto (field numbers are wire-format, so stable):
//
//	ModelProto         graph = 7
//	GraphProto         node = 1, initializer = 5, input = 11, output = 12, sparse_initializer = 15
//	TensorProto        name = 8
//	SparseTensorProto  values = 1 (a TensorProto, whose name is the sparse tensor's name)
//	ValueInfoProto     name = 1, type = 2
//	TypeProto          tensor_type = 1, sparse_tensor_type = 8 (oneof with sequence_type = 4,
//	                   map_type = 5, opaque_type = 7, optional_type = 9)
//	TypeProto.Tensor   elem_type = 1, shape = 2 (TypeProto.SparseTensor has the same layout)
//	TensorShapeProto   dim = 1
//	Dimension          dim_value = 1, dim_param = 2 (oneof)
//
// What it reports matches ORT's session inputs and outputs: a graph input that is also an
// initializer is a constant (an IR<4 export lists every weight as an input), and ORT leaves it
// out — so does this. A dimension is its dim_value, or -1 for a dim_param or an unset dimension.
// One difference is by design: an output's shape is the one declared in the file, while ORT runs
// ONNX shape inference at load and may report some declared-symbolic output dims as concrete
// (onnxheader_models_test.go checks that this is the only difference, on every local model).
// Names, element types and input shapes are identical; of the outputs, Inspect's callers (in
// this package and lifecycle) read only the names.

import (
	"errors"
	"fmt"
	"io"
	"os"
)

// ONNX field numbers used below (see the file comment).
const (
	fieldModelGraph = 7

	fieldGraphNode              = 1
	fieldGraphInitializer       = 5
	fieldGraphInput             = 11
	fieldGraphOutput            = 12
	fieldGraphSparseInitializer = 15

	fieldTensorName       = 8
	fieldSparseTensorVals = 1

	fieldValueInfoName = 1
	fieldValueInfoType = 2

	fieldTypeTensor       = 1
	fieldTypeSequence     = 4
	fieldTypeMap          = 5
	fieldTypeOpaque       = 7
	fieldTypeSparseTensor = 8
	fieldTypeOptional     = 9

	fieldTensorTypeElem  = 1
	fieldTensorTypeShape = 2

	fieldShapeDim = 1

	fieldDimValue = 1
	fieldDimParam = 2
)

// Protobuf wire types.
const (
	wireVarint  = 0
	wireFixed64 = 1
	wireBytes   = 2
	wireStart   = 3 // start group (deprecated; ONNX never uses groups)
	wireEnd     = 4
	wireFixed32 = 5
)

// maxNameLen bounds the length of a name read into memory. A real ONNX name is a few dozen
// bytes; the bound stops a corrupt length prefix from allocating the size of the file.
const maxNameLen = 1 << 20

// headerBufSize is the read window. The node list is walked sequentially, a window at a time;
// each skipped initializer costs one window read at its start.
const headerBufSize = 16 << 10

var errMalformed = errors.New("malformed protobuf")

// readONNXHeader returns the graph inputs (constant initializer inputs excluded) and outputs of
// the ONNX model at path, reading only the protobuf framing — never the weights.
func readONNXHeader(path string) (inputs, outputs []IOInfo, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}
	return parseONNXHeader(f, st.Size(), headerBufSize)
}

// parseONNXHeader is readONNXHeader over any ReaderAt; bufSize is the read window (tests shrink
// it to push varints and names across window boundaries).
func parseONNXHeader(r io.ReaderAt, size int64, bufSize int) (inputs, outputs []IOInfo, err error) {
	p := &pbReader{r: r, size: size, buf: make([]byte, 0, bufSize)}
	g := &graphHeader{initializers: map[string]bool{}}
	sawGraph := false
	// The ModelProto is the whole file. Every occurrence of the graph field is walked into the
	// same graphHeader: protobuf merges a repeated singular message field, appending its
	// repeated fields — which is what accumulating does.
	err = p.walk(size, func(num uint64, wt int) (bool, error) {
		if num != fieldModelGraph || wt != wireBytes {
			return false, nil
		}
		end, err := p.lenEnd(size)
		if err != nil {
			return true, err
		}
		sawGraph = true
		return true, p.graph(end, g)
	})
	if err != nil {
		return nil, nil, err
	}
	if !sawGraph {
		return nil, nil, errors.New("no graph in the model")
	}
	if len(g.outputs) == 0 {
		// Every valid graph has an output; zero means this was not an ONNX model at all (random
		// bytes can parse as protobuf). Let the caller fall back to ORT for a real diagnosis.
		return nil, nil, errors.New("graph declares no outputs")
	}
	for _, in := range g.inputs {
		if !g.initializers[in.name] {
			inputs = append(inputs, in.info())
		}
	}
	for _, out := range g.outputs {
		outputs = append(outputs, out.info())
	}
	return inputs, outputs, nil
}

// graphHeader accumulates what parseONNXHeader needs from a GraphProto.
type graphHeader struct {
	inputs, outputs []valueInfo
	initializers    map[string]bool
}

// valueInfo is a parsed ValueInfoProto.
type valueInfo struct {
	name     string
	member   uint64 // the TypeProto oneof member set: fieldTypeTensor, fieldTypeSequence, ...; 0 if none
	elemType int32  // TensorProto.DataType, for a (sparse) tensor
	dims     []int64
}

// tensor reports whether the value is a tensor or a sparse tensor: the types ORT reports a
// shape and an element type for.
func (v valueInfo) tensor() bool {
	return v.member == fieldTypeTensor || v.member == fieldTypeSparseTensor
}

func (v valueInfo) info() IOInfo {
	io := IOInfo{Name: v.name}
	if v.tensor() {
		io.ElemType = v.elemType
		if len(v.dims) > 0 { // a scalar and an unknown rank both read as nil, as Inspect always did
			io.Shape = v.dims
		}
	}
	return io
}

// graph walks one GraphProto occurrence ending at end.
func (p *pbReader) graph(end int64, g *graphHeader) error {
	return p.walk(end, func(num uint64, wt int) (bool, error) {
		if wt != wireBytes {
			return false, nil
		}
		switch num {
		case fieldGraphInitializer:
			e, err := p.lenEnd(end)
			if err != nil {
				return true, err
			}
			name, err := p.tensorName(e)
			if err != nil {
				return true, err
			}
			if name != "" {
				g.initializers[name] = true
			}
			return true, nil
		case fieldGraphSparseInitializer:
			e, err := p.lenEnd(end)
			if err != nil {
				return true, err
			}
			var name string
			err = p.walk(e, func(num uint64, wt int) (bool, error) {
				if num != fieldSparseTensorVals || wt != wireBytes {
					return false, nil
				}
				ve, err := p.lenEnd(e)
				if err != nil {
					return true, err
				}
				// values is a singular message: repeated occurrences merge, the last name wins.
				if n, err := p.tensorName(ve); err != nil {
					return true, err
				} else if n != "" {
					name = n
				}
				return true, nil
			})
			if err != nil {
				return true, err
			}
			if name != "" {
				g.initializers[name] = true
			}
			return true, nil
		case fieldGraphInput, fieldGraphOutput:
			e, err := p.lenEnd(end)
			if err != nil {
				return true, err
			}
			v, err := p.valueInfo(e)
			if err != nil {
				return true, err
			}
			if num == fieldGraphInput {
				g.inputs = append(g.inputs, v)
			} else {
				g.outputs = append(g.outputs, v)
			}
			return true, nil
		}
		return false, nil // node, value_info, doc_string, ...: skipped by length
	})
}

// tensorName returns the name of the TensorProto ending at end, skipping its data.
func (p *pbReader) tensorName(end int64) (string, error) {
	var name string
	err := p.walk(end, func(num uint64, wt int) (bool, error) {
		if num != fieldTensorName || wt != wireBytes {
			return false, nil
		}
		s, err := p.str(end)
		name = s
		return true, err
	})
	return name, err
}

// valueInfo parses the ValueInfoProto ending at end.
func (p *pbReader) valueInfo(end int64) (valueInfo, error) {
	var v valueInfo
	err := p.walk(end, func(num uint64, wt int) (bool, error) {
		if wt != wireBytes {
			return false, nil
		}
		switch num {
		case fieldValueInfoName:
			s, err := p.str(end)
			v.name = s
			return true, err
		case fieldValueInfoType:
			e, err := p.lenEnd(end)
			if err != nil {
				return true, err
			}
			return true, p.typeProto(e, &v)
		}
		return false, nil
	})
	return v, err
}

// typeProto parses a TypeProto into v. Its value is a oneof: a member replaces a different
// member set before it and merges into the same one — protobuf's rule, which also holds across
// repeated occurrences of ValueInfoProto.type, hence the member lives in v.
func (p *pbReader) typeProto(end int64, v *valueInfo) error {
	return p.walk(end, func(num uint64, wt int) (bool, error) {
		if wt != wireBytes {
			return false, nil
		}
		switch num {
		case fieldTypeTensor, fieldTypeSparseTensor:
			if v.member != num {
				v.elemType, v.dims = 0, nil
			}
			v.member = num
			e, err := p.lenEnd(end)
			if err != nil {
				return true, err
			}
			return true, p.tensorType(e, v)
		case fieldTypeSequence, fieldTypeMap, fieldTypeOpaque, fieldTypeOptional:
			// ORT reports neither a shape nor an element type for these: skip the payload.
			v.member, v.elemType, v.dims = num, 0, nil
		}
		return false, nil
	})
}

// tensorType parses a TypeProto.Tensor (or TypeProto.SparseTensor) into v.
func (p *pbReader) tensorType(end int64, v *valueInfo) error {
	return p.walk(end, func(num uint64, wt int) (bool, error) {
		switch {
		case num == fieldTensorTypeElem && wt == wireVarint:
			x, err := p.varint()
			v.elemType = int32(x)
			return true, err
		case num == fieldTensorTypeShape && wt == wireBytes:
			e, err := p.lenEnd(end)
			if err != nil {
				return true, err
			}
			return true, p.walk(e, func(num uint64, wt int) (bool, error) {
				if num != fieldShapeDim || wt != wireBytes {
					return false, nil
				}
				de, err := p.lenEnd(e)
				if err != nil {
					return true, err
				}
				d, err := p.dim(de)
				v.dims = append(v.dims, d)
				return true, err
			})
		}
		return false, nil
	})
}

// dim parses a TensorShapeProto.Dimension: its dim_value, or -1 for a dim_param or nothing.
func (p *pbReader) dim(end int64) (int64, error) {
	d := int64(-1)
	err := p.walk(end, func(num uint64, wt int) (bool, error) {
		switch {
		case num == fieldDimValue && wt == wireVarint:
			x, err := p.varint()
			d = int64(x)
			return true, err
		case num == fieldDimParam && wt == wireBytes:
			d = -1 // oneof: a later dim_param replaces an earlier dim_value
		}
		return false, nil
	})
	return d, err
}

// pbReader reads protobuf wire format from a ReaderAt through a small window, so stepping over
// a length-delimited field is an offset change rather than a read.
type pbReader struct {
	r      io.ReaderAt
	size   int64
	off    int64  // absolute offset of the next byte
	buf    []byte // window: the file's bytes [bufOff, bufOff+len(buf))
	bufOff int64
}

// walk reads the fields of a message ending at end. fn sees each field's number and wire type
// with the reader positioned at the field's payload; it returns true when it consumed the
// payload, false to have walk skip it.
func (p *pbReader) walk(end int64, fn func(num uint64, wt int) (bool, error)) error {
	for p.off < end {
		key, err := p.varint()
		if err != nil {
			return err
		}
		if p.off > end {
			return fmt.Errorf("%w: field key overruns its message (end %d)", errMalformed, end)
		}
		num, wt := key>>3, int(key&7)
		if num == 0 {
			return fmt.Errorf("%w: field number 0 at offset %d", errMalformed, p.off)
		}
		used, err := fn(num, wt)
		if err != nil {
			return err
		}
		if !used {
			if err := p.skip(wt, end); err != nil {
				return err
			}
		}
		if p.off > end {
			return fmt.Errorf("%w: field %d overruns its message", errMalformed, num)
		}
	}
	if p.off != end {
		return fmt.Errorf("%w: message overrun at offset %d (end %d)", errMalformed, p.off, end)
	}
	return nil
}

// skip steps over the payload of a field of wire type wt, bounded by end.
func (p *pbReader) skip(wt int, end int64) error {
	switch wt {
	case wireVarint:
		_, err := p.varint()
		return err
	case wireFixed64:
		return p.advance(8, end)
	case wireFixed32:
		return p.advance(4, end)
	case wireBytes:
		e, err := p.lenEnd(end)
		if err != nil {
			return err
		}
		p.off = e
		return nil
	}
	// Groups (3, 4) are deprecated and absent from onnx.proto; 6 and 7 are not wire types.
	return fmt.Errorf("%w: unsupported wire type %d at offset %d", errMalformed, wt, p.off)
}

func (p *pbReader) advance(n, end int64) error {
	if p.off+n > end {
		return fmt.Errorf("%w: truncated field at offset %d", errMalformed, p.off)
	}
	p.off += n
	return nil
}

// lenEnd reads a length prefix and returns the absolute end offset of the payload it announces,
// which must lie within end.
func (p *pbReader) lenEnd(end int64) (int64, error) {
	n, err := p.varint()
	if err != nil {
		return 0, err
	}
	// p.off past end (a varint that ran over) would make end-p.off negative and the bound huge.
	if p.off > end || n > uint64(end-p.off) {
		return 0, fmt.Errorf("%w: length %d at offset %d overruns its message (end %d)", errMalformed, n, p.off, end)
	}
	return p.off + int64(n), nil
}

// str reads a length-prefixed string bounded by end.
func (p *pbReader) str(end int64) (string, error) {
	e, err := p.lenEnd(end)
	if err != nil {
		return "", err
	}
	n := e - p.off
	if n > maxNameLen {
		return "", fmt.Errorf("%w: %d-byte name at offset %d", errMalformed, n, p.off)
	}
	if int(n) <= cap(p.buf) {
		if err := p.fill(int(n)); err != nil {
			return "", err
		}
		s := string(p.buf[p.off-p.bufOff : e-p.bufOff])
		p.off = e
		return s, nil
	}
	b := make([]byte, n)
	if k, err := p.r.ReadAt(b, p.off); k < len(b) {
		if err == nil {
			err = io.ErrUnexpectedEOF
		}
		return "", err
	}
	p.off = e
	return string(b), nil
}

// varint reads a base-128 varint.
func (p *pbReader) varint() (uint64, error) {
	var x uint64
	for i := 0; i < 10; i++ {
		if err := p.fill(1); err != nil {
			return 0, err
		}
		b := p.buf[p.off-p.bufOff]
		p.off++
		x |= uint64(b&0x7f) << (7 * i)
		if b < 0x80 {
			return x, nil
		}
	}
	return 0, fmt.Errorf("%w: varint longer than 10 bytes at offset %d", errMalformed, p.off)
}

// fill makes the window cover [p.off, p.off+n); n must not exceed the window's capacity.
func (p *pbReader) fill(n int) error {
	if p.off >= p.bufOff && p.off+int64(n) <= p.bufOff+int64(len(p.buf)) {
		return nil
	}
	if p.off+int64(n) > p.size {
		return fmt.Errorf("%w: unexpected end of file at offset %d", errMalformed, p.off)
	}
	m := int64(cap(p.buf))
	if rest := p.size - p.off; rest < m {
		m = rest
	}
	k, err := p.r.ReadAt(p.buf[:m], p.off)
	if k < n {
		if err == nil {
			err = io.ErrUnexpectedEOF
		}
		return err
	}
	p.buf, p.bufOff = p.buf[:k], p.off
	return nil
}
