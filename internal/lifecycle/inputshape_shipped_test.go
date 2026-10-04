package lifecycle

import (
	"os"
	"path/filepath"
	"testing"

	"visionserve/internal/registry"

	// Every architecture cmd/visionserve serves, so each shipped manifest builds its real model.
	_ "visionserve/internal/models/background"
	_ "visionserve/internal/models/classification"
	_ "visionserve/internal/models/clip"
	_ "visionserve/internal/models/depth"
	_ "visionserve/internal/models/detr"
	_ "visionserve/internal/models/efficientsam"
	_ "visionserve/internal/models/grasp"
	_ "visionserve/internal/models/groundedsam"
	_ "visionserve/internal/models/groundingdino"
	_ "visionserve/internal/models/hybrid"
	_ "visionserve/internal/models/mobilesam"
	_ "visionserve/internal/models/nanosam"
	_ "visionserve/internal/models/owlvit"
	_ "visionserve/internal/models/paddleocr"
	_ "visionserve/internal/models/sam2"
	_ "visionserve/internal/models/scrfd"
	_ "visionserve/internal/models/siglip"
	_ "visionserve/internal/models/textalign"
)

// Every shipped manifest passes the load-time input shape check against its real export: the
// check must refuse only configurations that cannot run. The weights are not committed, so it
// judges only the models whose weights were pulled — $VISIONSERVE_ONNX_DIR if set, else the
// repository's models/. Header reads and three small preprocess runs per model: no ONNX Runtime.
func TestShippedManifestsFitTheirGraphs(t *testing.T) {
	if testing.Short() {
		t.Skip("builds every shipped model (hashes pinned weights)")
	}
	dir := os.Getenv("VISIONSERVE_ONNX_DIR")
	if dir == "" {
		dir = filepath.Join("..", "..", "models")
	}
	reg := registry.New(dir)
	if _, err := reg.Scan(); err != nil {
		t.Skipf("no registry at %s: %v", dir, err)
	}
	m := NewManager(reg)
	t.Cleanup(m.Close)
	checked, judged, refused := 0, 0, 0
	for _, e := range reg.List() {
		if !e.Manifest.WeightsExist() {
			continue
		}
		name := e.Manifest.Name
		base, man, err := m.buildModel(name)
		if err != nil {
			t.Logf("%s: not built here (%v)", name, err)
			continue
		}
		skipped, err := judgeInputShape(man, base)
		switch {
		case err != nil:
			t.Errorf("%s: %v", name, err)
			refused++
		case skipped != "":
			t.Logf("%s: not judged: %s", name, skipped)
		default:
			judged++
		}
		checked++
	}
	if checked == 0 {
		t.Skipf("no model weights under %s (set VISIONSERVE_ONNX_DIR)", dir)
	}
	t.Logf("%d shipped models built, %d judged against their graph input, %d refused", checked, judged+refused, refused)
}
