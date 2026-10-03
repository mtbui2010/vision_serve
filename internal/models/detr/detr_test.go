package detr

import (
	"strings"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// Both architecture names resolve to this decoder, each with its own output contract:
// RF-DETR ignores extra outputs (rfdetr-small's cross_attn_weights), RT-DETR requires
// exactly two.
func TestRegistrationsAndOutputContracts(t *testing.T) {
	cfg := models.Config{Width: 64, Height: 64, ConfThresh: 0.5, Labels: []string{"a"}}
	logits := engine.F32([]float32{5}, 1, 1, 1)
	boxes := engine.F32([]float32{0.5, 0.5, 0.5, 0.5}, 1, 1, 4)
	attn := engine.F32(make([]float32, 3*1*2*1*8), 3, 1, 2, 1, 8)
	meta := models.PreprocessMeta{OrigWidth: 64, OrigHeight: 64, ScaleX: 1, ScaleY: 1}

	for _, c := range []struct {
		arch      string
		threeOK   bool
		errPrefix string
	}{{"rf-detr", true, "rfdetr:"}, {"rt-detr", false, "rtdetr:"}} {
		b, err := models.New(c.arch, cfg)
		if err != nil {
			t.Fatalf("%s: %v", c.arch, err)
		}
		m, ok := b.(models.Model)
		if !ok || m.Task() != models.TaskDetection {
			t.Fatalf("%s: not a detection Model", c.arch)
		}
		res, err := m.Postprocess([]engine.Tensor{boxes, logits}, meta)
		if err != nil || len(res.Detections) != 1 || res.Detections[0].Class != "a" {
			t.Fatalf("%s two outputs: %+v %v", c.arch, res, err)
		}
		_, err = m.Postprocess([]engine.Tensor{logits, boxes, attn}, meta)
		if (err == nil) != c.threeOK {
			t.Fatalf("%s three outputs: err=%v, want accepted=%v", c.arch, err, c.threeOK)
		}
		if _, err := m.Postprocess([]engine.Tensor{logits}, meta); err == nil || !strings.HasPrefix(err.Error(), c.errPrefix) {
			t.Fatalf("%s one output: err=%v, want prefix %q", c.arch, err, c.errPrefix)
		}
		if _, err := models.New(c.arch, models.Config{}); err == nil || !strings.HasPrefix(err.Error(), c.errPrefix) {
			t.Fatalf("%s zero size: err=%v", c.arch, err)
		}
	}
}
