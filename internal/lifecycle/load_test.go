package lifecycle

import (
	"context"
	"image"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"visionserve/internal/models"
	"visionserve/internal/registry"
)

var fakeBuilds atomic.Int32

// slowPipeline is a session-less pipeline model whose construction is slow and counted.
type slowPipeline struct{}

func (slowPipeline) Name() string      { return "slow" }
func (slowPipeline) Task() models.Task { return models.TaskEmbed }
func (slowPipeline) Roles() []string   { return nil }
func (slowPipeline) Infer(image.Image, models.Prompt, models.Runner) (models.Result, error) {
	return models.Result{}, nil
}

func init() {
	models.Register("test-slow-build", func(models.Config) (models.Base, error) {
		fakeBuilds.Add(1)
		time.Sleep(50 * time.Millisecond) // widen the window two cold loads used to race through
		return slowPipeline{}, nil
	})
}

// Concurrent first requests for one model must build it ONCE: each extra build hashed the
// weights again and created every ONNX session again (2x VRAM at peak) only to discard it.
func TestConcurrentLoadsBuildOnce(t *testing.T) {
	dir := t.TempDir()
	md := filepath.Join(dir, "slow")
	if err := os.MkdirAll(md, 0o755); err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(filepath.Join(md, "x.onnx"), []byte("x"), 0o644)
	man := "name: slow\ntask: embed\nlicense: MIT\narchitecture: test-slow-build\nfiles:\n  x: x.onnx\n" +
		"input:\n  width: 8\n  height: 8\nruntime:\n  prefer: [cpu]\n"
	if err := os.WriteFile(filepath.Join(md, "manifest.yaml"), []byte(man), 0o644); err != nil {
		t.Fatal(err)
	}
	reg := registry.New(dir)
	if _, err := reg.Scan(); err != nil {
		t.Fatal(err)
	}
	m := NewManager(reg)
	defer m.Close()

	fakeBuilds.Store(0)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := m.Load(context.Background(), "slow"); err != nil {
				t.Errorf("Load: %v", err)
			}
		}()
	}
	wg.Wait()
	if n := fakeBuilds.Load(); n != 1 {
		t.Fatalf("model built %d times for 8 concurrent cold loads, want 1", n)
	}
}
