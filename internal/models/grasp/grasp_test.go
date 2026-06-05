package grasp

import (
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	graspcore "visionserve/internal/grasp"
	"visionserve/internal/models"
	"visionserve/internal/registry"
)

// TestRegistered confirms init() registered the "grasp" architecture.
func TestRegistered(t *testing.T) {
	if !models.IsRegistered("grasp") {
		t.Fatalf("grasp not registered; registered = %v", models.Registered())
	}
}

// encodeRLEColumnMajor mirrors the SAM models' encoder so the test asserts
// decodeRLEColumnMajor is its exact inverse.
func encodeRLEColumnMajor(bin []bool, h, w int) string {
	if len(bin) == 0 {
		return ""
	}
	var counts []int
	prev := false
	run := 0
	for x := 0; x < w; x++ {
		for y := 0; y < h; y++ {
			if bin[y*w+x] == prev {
				run++
			} else {
				counts = append(counts, run)
				prev = bin[y*w+x]
				run = 1
			}
		}
	}
	counts = append(counts, run)
	parts := make([]string, len(counts))
	for i, c := range counts {
		parts[i] = strconv.Itoa(c)
	}
	return strings.Join(parts, " ")
}

func TestDecodeRoundTrip(t *testing.T) {
	cases := []struct {
		name string
		w, h int
		set  func(x, y int) bool
	}{
		{"empty", 8, 6, func(x, y int) bool { return false }},
		{"full", 8, 6, func(x, y int) bool { return true }},
		{"rect", 12, 10, func(x, y int) bool { return x >= 3 && x <= 8 && y >= 2 && y <= 6 }},
		{"single_col", 7, 7, func(x, y int) bool { return x == 3 }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			orig := make([]bool, c.w*c.h)
			for y := 0; y < c.h; y++ {
				for x := 0; x < c.w; x++ {
					orig[y*c.w+x] = c.set(x, y)
				}
			}
			bm, err := decodeRLEColumnMajor(encodeRLEColumnMajor(orig, c.h, c.w), c.w, c.h)
			if err != nil {
				t.Fatalf("decode error: %v", err)
			}
			for i := range orig {
				if bm.Data[i] != orig[i] {
					t.Fatalf("pixel %d (x=%d y=%d): got %v want %v", i, i%c.w, i/c.w, bm.Data[i], orig[i])
				}
			}
		})
	}
}

// TestGraspParamsOverride checks the gripper-bound precedence:
// core default < manifest default < request (Prompt).
func TestGraspParamsOverride(t *testing.T) {
	def := graspcore.DefaultParams()

	// No manifest, no request → core defaults.
	g0 := &graspModel{}
	if p := g0.graspParams(models.Prompt{}); p.Dmin != def.Dmin || p.Dmax != def.Dmax {
		t.Fatalf("defaults: got Dmin=%v Dmax=%v want %v/%v", p.Dmin, p.Dmax, def.Dmin, def.Dmax)
	}

	// Manifest defaults applied.
	gm := &graspModel{gripMin: 20, gripMax: 80}
	if p := gm.graspParams(models.Prompt{}); p.Dmin != 20 || p.Dmax != 80 {
		t.Fatalf("manifest: got Dmin=%v Dmax=%v want 20/80", p.Dmin, p.Dmax)
	}

	// Request overrides manifest.
	if p := gm.graspParams(models.Prompt{GripperMin: 33, GripperMax: 99}); p.Dmin != 33 || p.Dmax != 99 {
		t.Fatalf("request: got Dmin=%v Dmax=%v want 33/99", p.Dmin, p.Dmax)
	}

	// MaxDet → MaxGrasps cap.
	gc := &graspModel{}
	gc.cfg.MaxDet = 7
	if p := gc.graspParams(models.Prompt{}); p.MaxGrasps != 7 {
		t.Fatalf("MaxGrasps: got %v want 7", p.MaxGrasps)
	}
}

// TestManifestsValid loads the two reference manifests and checks they pass
// registry validation (license/task/dims) and carry the grasp composition fields.
func TestManifestsValid(t *testing.T) {
	root := filepath.Join("..", "..", "..", "models")

	agnostic, err := registry.LoadManifest(filepath.Join(root, "grasp", "manifest.yaml"))
	if err != nil {
		t.Fatalf("grasp manifest invalid: %v", err)
	}
	if agnostic.ArchOrName() != "grasp" {
		t.Fatalf("grasp architecture = %q, want grasp", agnostic.ArchOrName())
	}
	if agnostic.Detector != "" {
		t.Fatalf("class-agnostic manifest should have no detector, got %q", agnostic.Detector)
	}
	if agnostic.Grasp.GripperMin != 10 || agnostic.Grasp.GripperMax != 150 {
		t.Fatalf("grasp gripper bounds = %v/%v, want 10/150", agnostic.Grasp.GripperMin, agnostic.Grasp.GripperMax)
	}

	aware, err := registry.LoadManifest(filepath.Join(root, "grasp-rfdetr", "manifest.yaml"))
	if err != nil {
		t.Fatalf("grasp-rfdetr manifest invalid: %v", err)
	}
	if aware.Detector != "rf-detr" {
		t.Fatalf("grasp-rfdetr detector = %q, want rf-detr", aware.Detector)
	}
	if aware.ArchOrName() != "grasp" {
		t.Fatalf("grasp-rfdetr architecture = %q, want grasp", aware.ArchOrName())
	}

	// open-vocab variant: a grounding-dino detector (text-prompted).
	gd, err := registry.LoadManifest(filepath.Join(root, "grasp-gd", "manifest.yaml"))
	if err != nil {
		t.Fatalf("grasp-gd manifest invalid: %v", err)
	}
	if gd.Detector != "grounding-dino" {
		t.Fatalf("grasp-gd detector = %q, want grounding-dino", gd.Detector)
	}
	if gd.ArchOrName() != "grasp" {
		t.Fatalf("grasp-gd architecture = %q, want grasp", gd.ArchOrName())
	}
}
