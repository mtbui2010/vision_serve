package background

import (
	"fmt"
	"image"
	"strings"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// depthRunner answers role "depth" with a fixed output tensor and counts calls.
type depthRunner struct {
	out   engine.Tensor
	calls int
}

func (d *depthRunner) Run(role string, _ map[string]engine.Tensor) ([]engine.Tensor, error) {
	if role != roleDepth {
		return nil, fmt.Errorf("unexpected role %q", role)
	}
	d.calls++
	return []engine.Tensor{d.out}, nil
}
func (d *depthRunner) InputNames(string) []string  { return []string{"image"} }
func (d *depthRunner) OutputNames(string) []string { return []string{"depth"} }

// ramp is a w×h disparity map that is ONE affine plane everywhere — the monocular degenerate
// case the >70 % guard exists to reject.
func ramp(w, h int) []float32 {
	d := make([]float32, w*h)
	for v := 0; v < h; v++ {
		for u := 0; u < w; u++ {
			d[v*w+u] = float32(0.01*float64(u) + 0.02*float64(v) + 1)
		}
	}
	return d
}

// B10a: an external depth map whose dims do not match its data used to fall back to MiDaS
// SILENTLY — and, because `external` was decided from prompt.Depth != nil alone, MiDaS's
// output then skipped the monocular >70 % guard and a whole-frame "surface" came back. A
// malformed depth map is the caller's error and must say so.
func TestBackgroundDepthRejectsMalformedExternalDepth(t *testing.T) {
	m := &backgroundModel{hasDepth: true}
	r := &depthRunner{out: engine.F32(ramp(256, 256), 1, 256, 256)}
	img := image.NewRGBA(image.Rect(0, 0, 64, 48))
	p := models.Prompt{Depth: make([]float32, 10), DepthW: 64, DepthH: 48}
	data, err := m.backgroundDepth(img, p, r)
	if err == nil {
		t.Fatalf("malformed depth accepted (mask set=%v, MiDaS calls=%d)", data != nil && anySet(data), r.calls)
	}
	if !strings.Contains(err.Error(), "depth") {
		t.Errorf("error %q does not name the depth map", err)
	}
	if r.calls != 0 {
		t.Errorf("MiDaS ran %d times for a request that supplied its own depth", r.calls)
	}
}

// The monocular guard still applies to MiDaS output: a frame that is one plane is "no surface".
func TestBackgroundDepthMiDaSKeepsDegenerateGuard(t *testing.T) {
	m := &backgroundModel{hasDepth: true}
	r := &depthRunner{out: engine.F32(ramp(256, 256), 1, 256, 256)}
	data, err := m.backgroundDepth(image.NewRGBA(image.Rect(0, 0, 64, 48)), models.Prompt{}, r)
	if err != nil {
		t.Fatal(err)
	}
	if data != nil {
		t.Fatal("a whole-frame plane from MiDaS was returned as the support surface")
	}
}

// B10b: runDepth used to take "the last 256*256 values" of whatever the session returned. Any
// other output size was silently reinterpreted as 256x256 — a 288x288 map read as rows of the
// wrong width. The map's own [.., H, W] dims must be used, and a shape that is not one map an
// error.
func TestRunDepthUsesOutputShape(t *testing.T) {
	m := &backgroundModel{hasDepth: true}
	img := image.NewRGBA(image.Rect(0, 0, 64, 48))
	for _, shape := range [][]int64{{1, 256, 256}, {1, 1, 256, 256}, {256, 256}, {1, 288, 320}} {
		h, w := int(shape[len(shape)-2]), int(shape[len(shape)-1])
		r := &depthRunner{out: engine.F32(ramp(w, h), shape...)}
		d, dw, dh, err := m.runDepth(img, models.Prompt{}, r)
		if err != nil {
			t.Errorf("shape %v: %v", shape, err)
			continue
		}
		if dw != w || dh != h || len(d) != w*h {
			t.Errorf("shape %v: got %dx%d (%d values), want %dx%d", shape, dw, dh, len(d), w, h)
		}
	}
	for _, shape := range [][]int64{{1, 3, 256, 256}, {2, 256, 256}, {65536}} {
		n := int64(1)
		for _, s := range shape {
			n *= s
		}
		r := &depthRunner{out: engine.F32(make([]float32, n), shape...)}
		if _, _, _, err := m.runDepth(img, models.Prompt{}, r); err == nil {
			t.Errorf("shape %v accepted as a single depth map", shape)
		}
	}
}
