package mobilesam

import (
	"image"
	_ "image/jpeg"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"testing"

	"golang.org/x/image/draw"

	"visionserve/internal/engine"
)

// TestEncoderBitwiseRepeatableOnGPU guards the GPU nondeterminism fixed in engine/deterministic.go.
// On the CUDA EP, without deterministic compute, a ReduceMean in the MobileSAM encoder's last
// LayerNorm2d sums in an order that depends on what else the GPU is running: with other work on the
// GPU about 1 run in 100 returned an embedding differing in the last bit of a few hundred values,
// which moved boundary pixels of the mask. On an otherwise idle GPU it did not show in 2000 runs, so
// the test makes its own load: a second encoder session runs the same input concurrently.
//
// Opt-in, since it needs a GPU, the weights and about a minute: set
// VISIONSERVE_GPU_DETERMINISM_RUNS (1500 is plenty: without the fix ~17 of 1500 runs differed),
// ORT_DYLIB_PATH to a CUDA build, and VISIONSERVE_ONNX_DIR to the models directory if it is not the
// repository's models/. The test turns deterministic kernels on (VISIONSERVE_DETERMINISTIC=1, the
// opt-in) unless the variable is already set; with VISIONSERVE_DETERMINISTIC=0 it is expected to fail.
func TestEncoderBitwiseRepeatableOnGPU(t *testing.T) {
	runs, _ := strconv.Atoi(os.Getenv("VISIONSERVE_GPU_DETERMINISM_RUNS"))
	if runs <= 0 {
		t.Skip("set VISIONSERVE_GPU_DETERMINISM_RUNS to run (needs a CUDA GPU and the MobileSAM weights)")
	}
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("needs ORT_DYLIB_PATH (a CUDA build of libonnxruntime.so)")
	}
	if _, set := os.LookupEnv("VISIONSERVE_DETERMINISTIC"); !set {
		t.Setenv("VISIONSERVE_DETERMINISTIC", "1")
	}
	dir := os.Getenv("VISIONSERVE_ONNX_DIR")
	if dir == "" {
		dir = filepath.Join("..", "..", "..", "models")
	}
	path := filepath.Join(dir, "mobile-sam", "mobile_sam_encoder.onnx")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("MobileSAM encoder weights not found: %v", err)
	}
	var sessions [2]*engine.Session // [0] is checked, [1] only keeps the GPU busy
	for i := range sessions {
		s, err := engine.NewSession(path, nil, nil, []engine.Provider{engine.ProviderCUDA})
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		if s.ActiveEP() != engine.ProviderCUDA {
			t.Skipf("session runs on %s, not CUDA: nothing to test", s.ActiveEP())
		}
		sessions[i] = s
	}
	enc := sessions[0]

	f, err := os.Open(filepath.Join("..", "..", "..", "test", "testdata", "sample.jpg"))
	if err != nil {
		t.Fatal(err)
	}
	src, _, err := image.Decode(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	img := image.NewRGBA(image.Rect(0, 0, 801, 601))
	draw.BiLinear.Scale(img, img.Bounds(), src, src.Bounds(), draw.Src, nil)
	in, _, err := encoderInput(img)
	if err != nil {
		t.Fatal(err)
	}
	feed := map[string]engine.Tensor{enc.InputNames()[0]: in}
	ref, err := enc.RunNamed(feed)
	if err != nil {
		t.Fatal(err)
	}

	done := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-done:
				return
			default:
			}
			if _, err := sessions[1].RunNamed(feed); err != nil {
				t.Error(err)
				return
			}
		}
	}()
	defer func() { close(done); wg.Wait() }()

	bad := 0
	for k := 0; k < runs; k++ {
		out, err := enc.RunNamed(feed)
		if err != nil {
			t.Fatal(err)
		}
		diff := 0
		for i, v := range out[0].Data {
			if v != ref[0].Data[i] {
				diff++
			}
		}
		if diff > 0 {
			bad++
			if bad <= 5 {
				t.Logf("run %d: %d of %d embedding values differ from run 0", k, diff, len(out[0].Data))
			}
		}
	}
	if bad > 0 {
		t.Errorf("%d of %d encoder runs were not bit-identical to the first", bad, runs)
	}
}
