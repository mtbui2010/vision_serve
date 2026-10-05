package engine

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// TestReadFactsSynthetic checks every fact against the hand-built model of onnxheader_test.go,
// whose contents are known by construction: 2 nodes, three dense float initializers (1 Mi + 1024
// + 4 elements, one of them in a missing external file) and one sparse initializer, spread over
// two occurrences of ModelProto.graph.
func TestReadFactsSynthetic(t *testing.T) {
	raw, wantIn, wantOut := syntheticModel()
	raw = cat(raw,
		pbBytes(fieldModelOpsetImport, pbString(fieldOpsetDomain, "com.microsoft"), pbUint(fieldOpsetVersion, 1)),
		pbBytes(fieldModelOpsetImport, pbUint(fieldOpsetVersion, 17)),
		pbBytes(fieldModelMetadataProps, pbString(fieldEntryKey, "author"), pbString(fieldEntryValue, "me")),
		pbString(fieldModelProducerVersion, "2.1.0"),
	)
	path := filepath.Join(t.TempDir(), "m.onnx")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := ReadFacts(path)
	if err != nil {
		t.Fatal(err)
	}
	if f.IRVersion != 8 || f.ProducerName != "onnxheader_test" || f.ProducerVersion != "2.1.0" {
		t.Errorf("ir/producer = %d %q %q", f.IRVersion, f.ProducerName, f.ProducerVersion)
	}
	if want := []Opset{{"", 17}, {"com.microsoft", 1}}; !reflect.DeepEqual(f.Opsets, want) {
		t.Errorf("opsets = %+v, want %+v", f.Opsets, want)
	}
	if want := map[string]string{"author": "me"}; !reflect.DeepEqual(f.Metadata, want) {
		t.Errorf("metadata = %v", f.Metadata)
	}
	if f.FileBytes != int64(len(raw)) {
		t.Errorf("FileBytes = %d, want %d", f.FileBytes, len(raw))
	}
	if f.Nodes != 2 || f.Initializers != 3 || f.SparseInitializers != 1 || f.ParamsUnknown != 0 {
		t.Errorf("nodes %d, initializers %d, sparse %d, unknown %d; want 2, 3, 1, 0",
			f.Nodes, f.Initializers, f.SparseInitializers, f.ParamsUnknown)
	}
	const params = 1<<20 + 1024 + 4
	if f.Params != params || f.WeightBytes != 4*params || f.ParamsByType["float32"] != params {
		t.Errorf("params %d, bytes %d, by type %v; want %d, %d", f.Params, f.WeightBytes, f.ParamsByType, params, 4*params)
	}
	wantExt := []ExternalData{{Location: "missing.onnx.data", Tensors: 1, Bytes: 4096, OnDisk: -1}}
	if !reflect.DeepEqual(f.External, wantExt) {
		t.Errorf("external = %+v, want %+v", f.External, wantExt)
	}
	// Inputs and outputs are exactly what Inspect reports, plus dim names.
	var gotIn, gotOut []IOInfo
	for _, v := range f.Inputs {
		gotIn = append(gotIn, v.IOInfo)
	}
	for _, v := range f.Outputs {
		gotOut = append(gotOut, v.IOInfo)
	}
	if !reflect.DeepEqual(gotIn, wantIn) || !reflect.DeepEqual(gotOut, wantOut) {
		t.Errorf("I/O differs from the header reader:\n in  %+v\n out %+v", gotIn, gotOut)
	}
	wantNames := map[string][]string{
		"image":  {"batch", "", "", ""},
		"tokens": {"", "seq"},
		"noisy":  {"", "later_param_wins"},
		"scale":  nil,
	}
	for _, v := range f.Inputs {
		if want, ok := wantNames[v.Name]; ok && !reflect.DeepEqual(v.DimNames, want) {
			t.Errorf("%s dim names = %q, want %q", v.Name, v.DimNames, want)
		}
	}

	// The external file present next to the model: its size is reported.
	if err := os.WriteFile(filepath.Join(filepath.Dir(path), "missing.onnx.data"), make([]byte, 4096), 0o644); err != nil {
		t.Fatal(err)
	}
	if f, err = ReadFacts(path); err != nil || f.External[0].OnDisk != 4096 {
		t.Fatalf("with the sidecar present: %+v, %v", f.External, err)
	}
}

// An external location that leaves the model's directory is flagged and never stat'ed.
func TestReadFactsUnsafeExternalLocation(t *testing.T) {
	entry := func(k, v string) []byte { return pbBytes(13, pbString(1, k), pbString(2, v)) }
	for _, loc := range []string{"../escape.data", "/etc/passwd", ""} {
		g := cat(
			pbBytes(fieldGraphInitializer, pbBytes(1, pbVarint(2)), pbUint(2, 1), pbString(fieldTensorName, "w"),
				entry("location", loc), pbUint(14, 1)),
			output("y", tensorTypeOf(1, shape(dimValue(2)))),
		)
		path := filepath.Join(t.TempDir(), "m.onnx")
		if err := os.WriteFile(path, model(g), 0o644); err != nil {
			t.Fatal(err)
		}
		f, err := ReadFacts(path)
		if err != nil {
			t.Fatal(err)
		}
		if len(f.External) != 1 || !f.External[0].Unsafe || f.External[0].OnDisk != -1 || f.External[0].LengthUnknown != 1 {
			t.Errorf("location %q: %+v", loc, f.External)
		}
	}
}

// ReadFacts, like the header reader, returns an error (never panics) on every truncation of the
// synthetic model, and on overflowing dims it reports the size as unknown.
func TestReadFactsMalformed(t *testing.T) {
	raw, _, _ := syntheticModel()
	// Every prefix (the 4 MiB initializer is stepped over, so this is cheap): the facts reader
	// accepts exactly the prefixes the header reader accepts, and fails cleanly on the others.
	for n := 0; n < len(raw); n++ {
		if n > 4096 && n < len(raw)-4096 && n%4093 != 0 {
			continue // inside the weights: one sample per window is enough
		}
		facts := &Facts{Metadata: map[string]string{}, ParamsByType: map[string]int64{}, ext: map[string]*ExternalData{}}
		_, _, _, ferr := parseONNX(bytes.NewReader(raw[:n]), int64(n), 64, facts)
		_, _, herr := parseONNXHeader(bytes.NewReader(raw[:n]), int64(n), 64)
		if (ferr == nil) != (herr == nil) {
			t.Fatalf("prefix %d: facts reader err %v, header reader err %v", n, ferr, herr)
		}
	}
	dir := t.TempDir()
	huge := cat(
		pbBytes(fieldGraphInitializer, pbBytes(1, pbVarint(1<<40), pbVarint(1<<40)), pbUint(2, 1), pbString(fieldTensorName, "w")),
		pbBytes(fieldGraphInitializer, pbUint(1, 3), pbUint(2, 8), pbString(fieldTensorName, "s")), // string tensor
		output("y", tensorTypeOf(1, shape(dimValue(2)))),
	)
	path := filepath.Join(dir, "huge.onnx")
	if err := os.WriteFile(path, model(huge), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := ReadFacts(path)
	if err != nil {
		t.Fatal(err)
	}
	if f.ParamsUnknown != 2 || f.Params != 0 || f.Initializers != 2 {
		t.Errorf("overflowing / string initializers: %+v", f)
	}
}

// On every real model file (where weights were pulled): ReadFacts reports exactly the I/O the
// header reader reports, a dim name for every dynamic dim it can name, and some parameters.
func TestReadFactsOnModels(t *testing.T) {
	dir := os.Getenv("VISIONSERVE_ONNX_DIR")
	if dir == "" {
		dir = filepath.Join("..", "..", "models")
	}
	files := onnxFilesUnder(t, dir)
	if len(files) == 0 {
		t.Skipf("no .onnx files under %s (weights not pulled; set VISIONSERVE_ONNX_DIR)", dir)
	}
	for _, path := range files {
		in, out, herr := readONNXHeader(path)
		f, err := ReadFacts(path)
		if (err == nil) != (herr == nil) {
			t.Errorf("%s: ReadFacts err %v, header err %v", path, err, herr)
			continue
		}
		if err != nil {
			continue
		}
		var gotIn, gotOut []IOInfo
		for _, v := range f.Inputs {
			gotIn = append(gotIn, v.IOInfo)
			if len(v.DimNames) != len(v.Shape) {
				t.Errorf("%s: %s has %d dim names for %d dims", path, v.Name, len(v.DimNames), len(v.Shape))
			}
		}
		for _, v := range f.Outputs {
			gotOut = append(gotOut, v.IOInfo)
		}
		if !reflect.DeepEqual(gotIn, in) || !reflect.DeepEqual(gotOut, out) {
			t.Errorf("%s: I/O differs from the header reader", path)
		}
		if f.Initializers > 0 && f.Params == 0 && f.ParamsUnknown == 0 {
			t.Errorf("%s: %d initializers but no parameters", path, f.Initializers)
		}
	}
}

func TestElemTypeName(t *testing.T) {
	for typ, want := range map[int32]string{0: "", 1: "float32", 7: "int64", 10: "float16", 16: "bfloat16", 99: "type99"} {
		if got := ElemTypeName(typ); got != want {
			t.Errorf("ElemTypeName(%d) = %q, want %q", typ, got, want)
		}
	}
	if ElemBits(1) != 32 || ElemBits(22) != 4 || ElemBits(8) != 0 {
		t.Error("ElemBits")
	}
}
