package lifecycle

import (
	"path/filepath"
	"testing"

	"visionserve/internal/models"
	"visionserve/internal/registry"
)

// The client-resize hint of every shipped manifest (GET /api/models). The table is the decision,
// written out: a model is listed with a hint only when its result cannot depend on pixels past
// it, and every other shipped model must get none. A new manifest that gets a hint fails here
// until it is added on purpose. Manifests only — no weights needed.
func TestUsefulSideShippedManifests(t *testing.T) {
	reg := registry.New(filepath.Join("..", "..", "models"))
	if _, err := reg.Scan(); err != nil {
		t.Fatalf("scan: %v", err)
	}
	long := func(n int) models.UsefulSide { return models.UsefulSide{Long: n} }
	short := func(n int) models.UsefulSide { return models.UsefulSide{Short: n} }
	never := models.UsefulSide{}
	want := map[string]models.UsefulSide{
		// Detectors on a fixed input: squash fills both axes (shorter side), SCRFD's top-left
		// pad fits inside it (longer side). 2 × the input side. (rf-detr-nano and rt-detr squash
		// since the 2026-10-05 check audit.)
		"rf-detr":                 short(1120),
		"rf-detr-nano":            short(768),
		"rfdetr-small":            short(1024),
		"rfdetr-small-etri":       short(1024),
		"rfdetr-small-etri-probe": short(1024),
		"rfdetr-small-etri-qf":    short(1024),
		"rfdetr-small-qf":         short(1024),
		"rt-detr":                 short(1280),
		"scrfd":                   long(1280),
		"grounding-dino":          short(1600),
		"grounding-dino-fixed":    short(1600),
		// Whole-image classifiers / embedders / a squash depth model (its map is model-sized).
		"efficientnet-b0":   short(448),
		"mobilenet-v3":      short(448),
		"clip":              short(448),
		"siglip-image":      short(448),
		"siglip-image-fp16": short(448),
		"midas":             short(512),
		// keep_aspect: the tensor follows the image's aspect ratio.
		"depth-anything-v2": never,
		// Masks at the original resolution, OCR, depth-aligned grasping, templates, crop namers
		// (they crop the ORIGINAL image), text towers: never resized.
		"mobile-sam": never, "nano-sam": never, "efficient-sam": never, "sam2": never,
		"grounded-sam": never, "background": never, "paddle-ocr": never,
		"grasp": never, "grasp-gd": never, "grasp-rfdetr": never,
		"owlv2_base_patch16": never, "clip-text": never, "siglip-text": never, "siglip-text-fp16": never,
		"gdino-siglip": never, "gdino-siglip-sam": never,
		"rfdetr-gdino": never, "rfdetr-gdino-etri": never, "rfdetr-gdino-fastpath": never,
		"rfdetr-gdino-sam": never, "rfdetr-gdino-sam-etri": never, "rfdetr-gdino-siglip": never,
		"rfdetr-gdino-siglip-etri": never, "rfdetr-gdino-siglip-sam-etri": never,
		"rfdetr-dualhead-dec1": never, "rfdetr-textalign-dec1": never, "rfdetr-textalign-dec1-probe": never,
		"rfdetr-textalign-dec1-siglip": never, "rfdetr-textalign-dec1-siglip-probe": never,
		"rfdetr-textalign-dec1-siglip-prod": never, "rfdetr-textalign-etri": never,
		"rfdetr-textalign-etri-probe": never,
	}
	seen := map[string]bool{}
	for _, e := range reg.List() {
		name := e.Manifest.Name
		seen[name] = true
		w, ok := want[name]
		if !ok {
			// A manifest of an architecture that registers no hint is sent at full resolution,
			// which is always correct; one that gets a hint must be listed on purpose.
			if got := UsefulSide(e.Manifest); !got.IsZero() {
				t.Errorf("%s: not in this table but gets the client-resize hint %+v: check it and add it", name, got)
			}
			continue
		}
		if got := UsefulSide(e.Manifest); got != w {
			t.Errorf("%s: UsefulSide = %+v, want %+v", name, got, w)
		}
	}
	for name := range want {
		if !seen[name] {
			t.Errorf("%s: in the table but not shipped", name)
		}
	}
}
