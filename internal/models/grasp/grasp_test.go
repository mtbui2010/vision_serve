package grasp

import (
	"path/filepath"
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

	// B11: the per-mask grasp cap is its own default, NOT the detector's max_detections.
	// grasp-rfdetr ships max_detections: 300 for RF-DETR; reusing it let one star-shaped mask
	// return thousands of grasps (multi-MB responses).
	for _, maxDet := range []int{0, 7, 300} {
		gc := &graspModel{}
		gc.cfg.MaxDet = maxDet
		if p := gc.graspParams(models.Prompt{}); p.MaxGrasps != defaultMaxGraspsPerMask {
			t.Fatalf("max_detections=%d: MaxGrasps = %v, want the per-mask default %d",
				maxDet, p.MaxGrasps, defaultMaxGraspsPerMask)
		}
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
