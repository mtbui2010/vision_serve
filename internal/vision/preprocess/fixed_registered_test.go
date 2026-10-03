package preprocess_test

import (
	"strings"
	"testing"

	"visionserve/internal/models"
	_ "visionserve/internal/models/efficientsam"
	_ "visionserve/internal/models/mobilesam"
	_ "visionserve/internal/models/nanosam"
	_ "visionserve/internal/models/paddleocr"
	_ "visionserve/internal/models/sam2"
	"visionserve/internal/vision/preprocess"
)

// The architectures whose preprocessing the export fixes used to load a declared preprocess:
// block and silently feed something else (a mobile-sam block `squash 512` still fed long_side
// 1024 HWC raw). Their registered factories refuse it; the legacy input.* fields, reference only
// for them, keep loading.
func TestFixedByExportArchitecturesRefuseABlock(t *testing.T) {
	files := map[string]string{"encoder": "e.onnx", "decoder": "d.onnx", "det": "det.onnx", "rec": "rec.onnx", "model": "m.onnx"}
	for _, arch := range []string{"mobile-sam", "nano-sam", "sam2", "efficient-sam", "paddle-ocr"} {
		block := preprocess.Spec{Resize: preprocess.Squash, Width: 512, Height: 512}
		cfg := models.Config{Name: arch, Width: 1024, Height: 1024, Files: files, Preprocess: &block}
		_, err := models.New(arch, cfg)
		if err == nil || !strings.Contains(err.Error(), "fixed by its export") {
			t.Errorf("%s with a preprocess: block: err = %v, want a refusal", arch, err)
		}
		legacy := preprocess.FromLegacy(preprocess.LegacyFields{Width: 1024, Height: 1024})
		cfg.Preprocess = &legacy
		if _, err := models.New(arch, cfg); err != nil && strings.Contains(err.Error(), "fixed by its export") {
			t.Errorf("%s with legacy fields only: refused: %v", arch, err)
		}
	}
}
