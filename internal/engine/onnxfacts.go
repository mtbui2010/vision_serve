package engine

// ReadFacts: the descriptive facts of an ONNX file for `visionserve inspect` — IR and opset
// versions, producer, metadata, node count, initializer statistics (parameter count, weight
// bytes, external-data references) and the symbolic names of dynamic dims. It walks the same
// protobuf framing as readONNXHeader (onnxheader.go) with the same bounds, and reads only a few
// more small fields: an initializer's dims, data_type, data_location and external_data entries,
// never its data. Extra schema used here (onnx/onnx.proto):
//
//	ModelProto              ir_version = 1, producer_name = 2, producer_version = 3,
//	                        opset_import = 8, metadata_props = 14
//	OperatorSetIdProto      domain = 1, version = 2
//	StringStringEntryProto  key = 1, value = 2
//	TensorProto             dims = 1, data_type = 2, external_data = 13, data_location = 14

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
)

const (
	fieldModelIRVersion       = 1
	fieldModelProducerName    = 2
	fieldModelProducerVersion = 3
	fieldModelOpsetImport     = 8
	fieldModelMetadataProps   = 14

	fieldOpsetDomain  = 1
	fieldOpsetVersion = 2

	fieldEntryKey   = 1
	fieldEntryValue = 2

	fieldTensorDims         = 1
	fieldTensorDataType     = 2
	fieldTensorExternalData = 13
	fieldTensorDataLocation = 14

	// dataLocationExternal is TensorProto.DataLocation EXTERNAL.
	dataLocationExternal = 1

	// maxFactEntries bounds the opsets and metadata entries kept (a real model has a handful).
	maxFactEntries = 256
	// maxTensorDims bounds the dims read per initializer (a real tensor has at most ~8).
	maxTensorDims = 64
)

// Facts describes an ONNX file without loading its weights. Counts cover the top-level graph:
// initializers inside control-flow subgraphs (If/Loop bodies) are not walked.
type Facts struct {
	Path      string
	FileBytes int64

	IRVersion       int64
	ProducerName    string
	ProducerVersion string
	Opsets          []Opset           // sorted by domain ("" = the default ai.onnx domain)
	Metadata        map[string]string // metadata_props (at most maxFactEntries)

	Inputs, Outputs []IOFacts // as Inspect reports them (constant initializer inputs excluded)

	Nodes              int // top-level graph nodes
	Initializers       int // dense initializers
	SparseInitializers int
	// Params is the element count of every dense initializer; WeightBytes their size at their
	// element type's width. ParamsUnknown counts initializers whose size could not be derived (a
	// string tensor, an unknown element type): they are left out of both totals.
	Params, WeightBytes int64
	ParamsUnknown       int
	// ParamsByType is the element count per element type name ("float32", "int8", ...).
	ParamsByType map[string]int64

	// External lists the external-data files the initializers reference, sorted by location.
	External []ExternalData

	ext map[string]*ExternalData // External while the graph is walked, by location
}

// Opset is one opset_import entry.
type Opset struct {
	Domain  string
	Version int64
}

// IOFacts is an IOInfo plus the dim_param name of each dynamic dimension.
type IOFacts struct {
	IOInfo
	// DimNames has one entry per Shape entry: the symbolic name of a dynamic dim ("batch",
	// "height"), "" for a fixed dim or an unnamed dynamic one.
	DimNames []string
}

// ExternalData is one external-data file and what the header says is stored in it.
type ExternalData struct {
	Location string // as written in the model, relative to the model's directory
	Tensors  int
	// Bytes is the sum of the declared lengths; LengthUnknown counts tensors that declare none
	// (they extend to the end of the file).
	Bytes         int64
	LengthUnknown int
	// OnDisk is the file's size next to the model (-1 = missing). Unsafe is true when Location
	// is absolute or leaves the model's directory: ORT refuses such a path, and so must an
	// importer that copies the model.
	OnDisk int64
	Unsafe bool
}

// ReadFacts reads the descriptive facts of the ONNX file at path (see Facts). It never reads
// weight data; it stats the external-data files the model references.
func ReadFacts(path string) (*Facts, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	facts := &Facts{Path: path, FileBytes: st.Size(), Metadata: map[string]string{},
		ParamsByType: map[string]int64{}, ext: map[string]*ExternalData{}}
	_, _, g, err := parseONNX(f, st.Size(), headerBufSize, facts)
	if err != nil {
		return nil, err
	}
	for _, v := range g.inputs {
		if !g.initializers[v.name] {
			facts.Inputs = append(facts.Inputs, v.ioFacts())
		}
	}
	for _, v := range g.outputs {
		facts.Outputs = append(facts.Outputs, v.ioFacts())
	}
	sort.Slice(facts.Opsets, func(i, j int) bool { return facts.Opsets[i].Domain < facts.Opsets[j].Domain })
	dir := filepath.Dir(path)
	for _, e := range facts.ext {
		e.OnDisk = -1
		e.Unsafe = !localPath(e.Location)
		if !e.Unsafe {
			if st, err := os.Stat(filepath.Join(dir, e.Location)); err == nil && st.Mode().IsRegular() {
				e.OnDisk = st.Size()
			}
		}
		facts.External = append(facts.External, *e)
	}
	sort.Slice(facts.External, func(i, j int) bool { return facts.External[i].Location < facts.External[j].Location })
	facts.ext = nil
	return facts, nil
}

// localPath reports whether rel is a relative path that stays inside its base directory.
func localPath(rel string) bool {
	return rel != "" && filepath.IsLocal(rel)
}

func (v valueInfo) ioFacts() IOFacts {
	f := IOFacts{IOInfo: v.info()}
	if f.Shape != nil {
		f.DimNames = make([]string, len(f.Shape))
		copy(f.DimNames, v.params)
	}
	return f
}

// modelFact consumes one ModelProto field other than the graph into facts. It returns false (skip)
// for fields it does not read.
func (p *pbReader) modelFact(num uint64, wt int, end int64, facts *Facts) (bool, error) {
	switch {
	case num == fieldModelIRVersion && wt == wireVarint:
		x, err := p.varint()
		facts.IRVersion = int64(x)
		return true, err
	case num == fieldModelProducerName && wt == wireBytes:
		s, err := p.str(end)
		facts.ProducerName = s
		return true, err
	case num == fieldModelProducerVersion && wt == wireBytes:
		s, err := p.str(end)
		facts.ProducerVersion = s
		return true, err
	case num == fieldModelOpsetImport && wt == wireBytes:
		e, err := p.lenEnd(end)
		if err != nil {
			return true, err
		}
		var o Opset
		err = p.walk(e, func(num uint64, wt int) (bool, error) {
			switch {
			case num == fieldOpsetDomain && wt == wireBytes:
				s, err := p.str(e)
				o.Domain = s
				return true, err
			case num == fieldOpsetVersion && wt == wireVarint:
				x, err := p.varint()
				o.Version = int64(x)
				return true, err
			}
			return false, nil
		})
		if err == nil && len(facts.Opsets) < maxFactEntries {
			facts.Opsets = append(facts.Opsets, o)
		}
		return true, err
	case num == fieldModelMetadataProps && wt == wireBytes:
		e, err := p.lenEnd(end)
		if err != nil {
			return true, err
		}
		k, v, err := p.entry(e)
		if err == nil && len(facts.Metadata) < maxFactEntries {
			facts.Metadata[k] = v
		}
		return true, err
	}
	return false, nil
}

// entry parses a StringStringEntryProto ending at end.
func (p *pbReader) entry(end int64) (key, value string, err error) {
	err = p.walk(end, func(num uint64, wt int) (bool, error) {
		if wt != wireBytes || (num != fieldEntryKey && num != fieldEntryValue) {
			return false, nil
		}
		s, err := p.str(end)
		if num == fieldEntryKey {
			key = s
		} else {
			value = s
		}
		return true, err
	})
	return key, value, err
}

// tensorFacts parses an initializer TensorProto ending at end: its name, and its size and
// external-data references into facts. The data itself is stepped over.
func (p *pbReader) tensorFacts(end int64, facts *Facts) (string, error) {
	var (
		name     string
		dims     []int64
		elem     int32
		location int64
		ext      = map[string]string{}
	)
	err := p.walk(end, func(num uint64, wt int) (bool, error) {
		switch {
		case num == fieldTensorName && wt == wireBytes:
			s, err := p.str(end)
			name = s
			return true, err
		case num == fieldTensorDims && wt == wireVarint:
			x, err := p.varint()
			if len(dims) < maxTensorDims {
				dims = append(dims, int64(x))
			}
			return true, err
		case num == fieldTensorDims && wt == wireBytes: // packed
			e, err := p.lenEnd(end)
			if err != nil {
				return true, err
			}
			for p.off < e {
				x, err := p.varint()
				if err != nil {
					return true, err
				}
				if len(dims) < maxTensorDims {
					dims = append(dims, int64(x))
				}
			}
			if p.off != e {
				return true, fmt.Errorf("%w: packed dims overrun", errMalformed)
			}
			return true, nil
		case num == fieldTensorDataType && wt == wireVarint:
			x, err := p.varint()
			elem = int32(x)
			return true, err
		case num == fieldTensorDataLocation && wt == wireVarint:
			x, err := p.varint()
			location = int64(x)
			return true, err
		case num == fieldTensorExternalData && wt == wireBytes:
			e, err := p.lenEnd(end)
			if err != nil {
				return true, err
			}
			k, v, err := p.entry(e)
			if len(ext) < maxFactEntries {
				ext[k] = v
			}
			return true, err
		}
		return false, nil
	})
	if err != nil {
		return "", err
	}
	facts.Initializers++
	n, ok := elementCount(dims)
	bits := ElemBits(elem)
	if !ok || bits == 0 {
		facts.ParamsUnknown++
	} else {
		facts.Params = satAdd(facts.Params, n)
		facts.ParamsByType[ElemTypeName(elem)] = satAdd(facts.ParamsByType[ElemTypeName(elem)], n)
		if n > math.MaxInt64/int64(bits) {
			facts.WeightBytes = math.MaxInt64
		} else {
			facts.WeightBytes = satAdd(facts.WeightBytes, (n*int64(bits)+7)/8)
		}
	}
	if location == dataLocationExternal && facts.ext != nil {
		loc := ext["location"]
		e := facts.ext[loc]
		if e == nil {
			if len(facts.ext) >= maxFactEntries {
				return name, nil
			}
			e = &ExternalData{Location: loc}
			facts.ext[loc] = e
		}
		e.Tensors++
		if l, err := strconv.ParseInt(ext["length"], 10, 64); err == nil && l >= 0 {
			e.Bytes = satAdd(e.Bytes, l)
		} else {
			e.LengthUnknown++
		}
	}
	return name, nil
}

// elementCount is the product of dims (1 for a scalar); ok is false on overflow.
func elementCount(dims []int64) (int64, bool) {
	n := int64(1)
	for _, d := range dims {
		if d < 0 {
			return 0, false
		}
		if d != 0 && n > math.MaxInt64/d {
			return 0, false
		}
		n *= d
	}
	return n, true
}

// satAdd adds two non-negative counts, saturating at MaxInt64.
func satAdd(a, b int64) int64 {
	if a > math.MaxInt64-b {
		return math.MaxInt64
	}
	return a + b
}

// elemTypes maps TensorProto.DataType to its name and width in bits (0 = no fixed width).
var elemTypes = map[int32]struct {
	name string
	bits int
}{
	1: {"float32", 32}, 2: {"uint8", 8}, 3: {"int8", 8}, 4: {"uint16", 16}, 5: {"int16", 16},
	6: {"int32", 32}, 7: {"int64", 64}, 8: {"string", 0}, 9: {"bool", 8}, 10: {"float16", 16},
	11: {"float64", 64}, 12: {"uint32", 32}, 13: {"uint64", 64}, 14: {"complex64", 64},
	15: {"complex128", 128}, 16: {"bfloat16", 16}, 17: {"float8e4m3fn", 8}, 18: {"float8e4m3fnuz", 8},
	19: {"float8e5m2", 8}, 20: {"float8e5m2fnuz", 8}, 21: {"uint4", 4}, 22: {"int4", 4},
	23: {"float4e2m1", 4},
}

// ElemTypeName is the readable name of an ONNX element type (TensorProto.DataType), e.g.
// "float32" for 1; "type<n>" for one this table does not know, "" for 0 (not a tensor).
func ElemTypeName(t int32) string {
	if t == 0 {
		return ""
	}
	if e, ok := elemTypes[t]; ok {
		return e.name
	}
	return "type" + strconv.Itoa(int(t))
}

// ElemBits is the width in bits of an ONNX element type; 0 when it has none (string, unknown).
func ElemBits(t int32) int { return elemTypes[t].bits }
