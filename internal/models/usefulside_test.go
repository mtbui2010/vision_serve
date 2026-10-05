package models

import (
	"errors"
	"sync"
	"testing"

	"visionserve/internal/vision/preprocess"
)

// The hint per resize mode: fit-inside modes bound the longer side, fill-both-axes modes the
// shorter one, both at UsefulSideFactor × the LARGER target side (a non-square target must leave
// both of its axes down-scaled by the factor); modes whose tensor follows the image get none.
func TestFixedTargetUsefulSide(t *testing.T) {
	cases := []struct {
		mode preprocess.Mode
		w, h int
		want UsefulSide
	}{
		{preprocess.Squash, 560, 560, UsefulSide{Short: 1120}},
		{preprocess.CenterCrop, 224, 224, UsefulSide{Short: 448}},
		{preprocess.Squash, 640, 384, UsefulSide{Short: 1280}},
		{preprocess.Letterbox, 640, 640, UsefulSide{Long: 1280}},
		{preprocess.Letterbox, 384, 640, UsefulSide{Long: 1280}},
		{preprocess.TopLeftPad, 640, 640, UsefulSide{Long: 1280}},
		{preprocess.LongSide, 1024, 1024, UsefulSide{Long: 2048}},
		{preprocess.LongSidePad, 960, 960, UsefulSide{Long: 1920}},
		{preprocess.KeepAspect, 518, 518, UsefulSide{}},
		{preprocess.None, 0, 0, UsefulSide{}},
		{"", 560, 560, UsefulSide{}}, // unresolved: no hint rather than a guess
		{preprocess.Squash, 0, 560, UsefulSide{}},
	}
	for _, c := range cases {
		got := FixedTargetUsefulSide(preprocess.Spec{Resize: c.mode, Width: c.w, Height: c.h})
		if got != c.want {
			t.Errorf("%s %dx%d: %+v, want %+v", c.mode, c.w, c.h, got, c.want)
		}
	}
}

var failingOnce sync.Once

// An architecture that registers nothing, or whose function refuses the spec, gets no hint.
func TestUsefulSideOfUnregisteredOrFailing(t *testing.T) {
	s := preprocess.Spec{Resize: preprocess.Squash, Width: 8, Height: 8}
	if got := UsefulSideOf("no-such-arch-usefulside", s); !got.IsZero() {
		t.Errorf("unregistered: %+v", got)
	}
	failingOnce.Do(func() { // once per process: -count=N reruns must not register twice
		RegisterUsefulSide("test-failing-usefulside", func(preprocess.Spec) (UsefulSide, error) {
			return UsefulSide{Short: 99}, errors.New("refused")
		})
	})
	if got := UsefulSideOf("test-failing-usefulside", s); !got.IsZero() {
		t.Errorf("failing function: %+v", got)
	}
	// ResolvedUsefulSide resolves through the architecture: a legacy mode it never honoured
	// falls back to its default, a declared one it does not support is refused (no hint).
	a := preprocess.Arch{Name: "t", Modes: []preprocess.Mode{preprocess.Squash, preprocess.Letterbox}}
	f := ResolvedUsefulSide(a)
	if u, err := f(preprocess.Spec{Resize: preprocess.CenterCrop, Width: 100, Height: 100, Legacy: true}); err != nil || u != (UsefulSide{Short: 200}) {
		t.Errorf("legacy center_crop on a squash arch: %+v, %v", u, err)
	}
	if _, err := f(preprocess.Spec{Resize: preprocess.CenterCrop, Width: 100, Height: 100}); err == nil {
		t.Error("declared center_crop on a squash/letterbox arch: no error")
	}
}
