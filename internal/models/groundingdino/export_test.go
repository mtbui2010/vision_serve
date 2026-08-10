package groundingdino

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The probe must key on the NonZero op_type wherever it sits in the file, including across a
// read-buffer boundary, and must default to "not joint" whenever it sees the marker.
func TestSupportsJointTextPassSyntheticGraphs(t *testing.T) {
	dir := t.TempDir()
	const chunk = 1 << 20

	for _, tc := range []struct {
		name  string
		body  []byte
		joint bool
	}{
		{"clean graph", []byte("ReduceMaxReduceMinEqualOrWhereClip"), true},
		{"baked-loop graph", []byte("EqualOrNonZeroTransposeGatherScatterND"), false},
		{"marker straddling the buffer boundary", func() []byte {
			b := make([]byte, chunk+16)
			for i := range b {
				b[i] = 'x'
			}
			copy(b[chunk-3:], "NonZero") // 3 bytes before the boundary, 4 after
			return b
		}(), false},
		{"marker just past the first chunk", func() []byte {
			b := make([]byte, 2*chunk)
			for i := range b {
				b[i] = 'x'
			}
			copy(b[chunk+100:], "NonZero")
			return b
		}(), false},
		{"empty file", nil, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := filepath.Join(dir, strings.ReplaceAll(tc.name, " ", "_")+".onnx")
			if err := os.WriteFile(p, tc.body, 0o644); err != nil {
				t.Fatal(err)
			}
			got, err := SupportsJointTextPass(p)
			if err != nil {
				t.Fatalf("probe: %v", err)
			}
			if got != tc.joint {
				t.Errorf("SupportsJointTextPass = %v, want %v", got, tc.joint)
			}
		})
	}
}

func TestSupportsJointTextPassMissingFile(t *testing.T) {
	got, err := SupportsJointTextPass(filepath.Join(t.TempDir(), "absent.onnx"))
	if err == nil {
		t.Fatal("probing a missing file returned no error")
	}
	if got { // callers fall back to the safe path on error; never claim joint
		t.Error("probe reported joint=true for a missing file")
	}
}

// The real weights, when present, must be classified correctly: the community export is the
// baked-loop one, the re-export is not.
func TestSupportsJointTextPassRealWeights(t *testing.T) {
	for _, tc := range []struct {
		file  string
		joint bool
	}{
		{"model.onnx", false},
		{"model-fixedmask.onnx", true},
	} {
		p := filepath.Join("..", "..", "..", "models", "grounding-dino", tc.file)
		if _, err := os.Stat(p); err != nil {
			t.Logf("skip %s: not present", tc.file)
			continue
		}
		got, err := SupportsJointTextPass(p)
		if err != nil {
			t.Fatalf("%s: %v", tc.file, err)
		}
		if got != tc.joint {
			t.Errorf("%s: SupportsJointTextPass = %v, want %v", tc.file, got, tc.joint)
		}
	}
}
