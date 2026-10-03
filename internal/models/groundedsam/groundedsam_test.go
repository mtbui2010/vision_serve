package groundedsam

import (
	"fmt"
	"image"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// vocabDir holds the GroundingDINO tokenizer's vocab.txt (committed); the weights path only needs
// to name a file next to it — the joint-pass probe of a missing file safely answers "per phrase".
var vocabDir = filepath.Join("..", "..", "..", "models", "grounding-dino")

func testConfig(conf float64) models.Config {
	return models.Config{
		Name:       "grounded-sam",
		Dir:        filepath.Join("..", "..", "..", "models", "grounded-sam"),
		ConfThresh: conf,
		Files: map[string]string{
			roleGDINO:   filepath.Join(vocabDir, "model-missing.onnx"),
			roleEncoder: "enc.onnx",
			roleDecoder: "dec.onnx",
		},
	}
}

func newModel(t *testing.T, conf float64) *groundedSAM {
	t.Helper()
	b, err := New(testConfig(conf))
	if err != nil {
		t.Skipf("GroundingDINO vocab not present: %v", err)
	}
	return b.(*groundedSAM)
}

func TestRegistered(t *testing.T) {
	if !models.IsRegistered("grounded-sam") {
		t.Fatal("grounded-sam is not registered")
	}
}

func TestNewRequiresEveryRole(t *testing.T) {
	for _, role := range []string{roleGDINO, roleEncoder, roleDecoder} {
		cfg := testConfig(0)
		delete(cfg.Files, role)
		if _, err := New(cfg); err == nil || !strings.Contains(err.Error(), "files."+role) {
			t.Errorf("without files.%s: %v, want an error naming it", role, err)
		}
	}
}

// Roles drive what lifecycle loads; PoolSizes and Exclusive drive how it serves them.
func TestRolesPoolsAndExclusivity(t *testing.T) {
	m := newModel(t, 0)
	if got := strings.Join(m.Roles(), ","); got != "gdino,encoder,decoder" {
		t.Errorf("Roles = %s", got)
	}
	if got := m.PoolSizes(); len(got) != 1 || got[roleDecoder] != 4 {
		t.Errorf("PoolSizes = %v, want 4 decoders", got)
	}
	var b models.Base = m
	if ex, ok := b.(models.Exclusive); !ok || !ex.Exclusive() {
		t.Error("grounded-sam must ask lifecycle for one request at a time (it replaced PipelineMu)")
	}
}

func TestInferRequiresAPhrase(t *testing.T) {
	m := newModel(t, 0)
	r := &fakeRunner{t: t}
	for _, text := range []string{"", "   "} {
		if _, err := m.Infer(canvas(), models.Prompt{Text: text}, r); err == nil || !strings.Contains(err.Error(), "requires a text prompt") {
			t.Errorf("%q: %v", text, err)
		}
	}
	if _, err := m.Infer(canvas(), models.Prompt{Text: " . .. "}, r); err == nil || !strings.Contains(err.Error(), "holds no class phrase") {
		t.Errorf("punctuation-only prompt: %v", err)
	}
	if r.calls() != 0 {
		t.Errorf("a session ran %d times for an invalid prompt", r.calls())
	}
}

// fakeRunner: GroundingDINO finds one confident query over the prompt's first phrase (box
// cxcywh 0.5,0.5,0.5,0.5); the MobileSAM decoder returns a mask that is set inside that box.
type fakeRunner struct {
	t     *testing.T
	mu    sync.Mutex
	roles []string
}

func (f *fakeRunner) calls() int { f.mu.Lock(); defer f.mu.Unlock(); return len(f.roles) }

func (f *fakeRunner) Run(role string, in map[string]engine.Tensor) ([]engine.Tensor, error) {
	f.mu.Lock()
	f.roles = append(f.roles, role)
	f.mu.Unlock()
	switch role {
	case roleGDINO:
		lg := make([]float32, 256)
		for i := range lg {
			lg[i] = -6
		}
		lg[1] = 6 // the first phrase's first token: [CLS] cup . [SEP]
		return []engine.Tensor{engine.F32(lg, 1, 1, 256), engine.F32([]float32{0.5, 0.5, 0.5, 0.5}, 1, 1, 4)}, nil
	case roleEncoder:
		return []engine.Tensor{engine.F32(make([]float32, 4), 1, 1, 2, 2)}, nil
	case roleDecoder:
		size := in["orig_im_size"].Data
		h, w := int(size[0]), int(size[1])
		m := make([]float32, h*w)
		for y := h / 4; y < 3*h/4; y++ {
			for x := w / 4; x < 3*w/4; x++ {
				m[y*w+x] = 1
			}
		}
		return []engine.Tensor{engine.F32(m, 1, 1, int64(h), int64(w)), engine.F32([]float32{0.9}, 1, 1)}, nil
	}
	return nil, fmt.Errorf("unexpected role %q", role)
}
func (f *fakeRunner) InputNames(role string) []string {
	if role == roleEncoder {
		return []string{"input_image"}
	}
	return nil
}
func (f *fakeRunner) OutputNames(role string) []string {
	switch role {
	case roleGDINO:
		return []string{"logits", "pred_boxes"}
	case roleDecoder:
		return []string{"masks", "iou_predictions"}
	}
	return nil
}

func canvas() image.Image { return image.NewRGBA(image.Rect(0, 0, 40, 20)) }

// Detect then segment: one mask per detection, index-aligned, carrying the detection's box and
// score; the detection's box is in original-image pixels.
func TestInferDetectsThenSegments(t *testing.T) {
	m := newModel(t, 0)
	r := &fakeRunner{t: t}
	res, err := m.Infer(canvas(), models.Prompt{Text: "cup."}, r)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Detections) != 1 || len(res.Masks) != 1 {
		t.Fatalf("%d detections, %d masks; want 1 and 1", len(res.Detections), len(res.Masks))
	}
	d, mk := res.Detections[0], res.Masks[0]
	if d.Class != "cup" || d.BBox != [4]float64{10, 5, 20, 10} {
		t.Errorf("detection %+v, want cup at [10 5 20 10] (original pixels)", d)
	}
	if mk.BBox != d.BBox || mk.Conf != d.Conf || mk.RLE == "" {
		t.Errorf("mask %+v does not carry its detection's box/score", mk)
	}
	if got := strings.Join(r.roles, ","); got != "gdino,encoder,decoder" {
		t.Errorf("sessions ran as %s", got)
	}
}

// The manifest's conf_threshold is GroundingDINO's box threshold, and a request's box_threshold
// wins over it. No detection → no MobileSAM pass at all.
func TestThresholdPrecedence(t *testing.T) {
	m := newModel(t, 0.999) // above sigmoid(6) = 0.9975
	r := &fakeRunner{t: t}
	res, err := m.Infer(canvas(), models.Prompt{Text: "cup."}, r)
	if err != nil || len(res.Detections) != 0 || res.Masks != nil {
		t.Fatalf("manifest threshold 0.999: %+v, %v; want nothing", res, err)
	}
	if got := strings.Join(r.roles, ","); got != "gdino" {
		t.Errorf("sessions ran as %s; MobileSAM must not run without a detection", got)
	}
	res, err = m.Infer(canvas(), models.Prompt{Text: "cup.", BoxThresh: 0.5}, &fakeRunner{t: t})
	if err != nil || len(res.Detections) != 1 {
		t.Fatalf("request box_threshold 0.5: %d detections, %v; want the request to win", len(res.Detections), err)
	}
}
