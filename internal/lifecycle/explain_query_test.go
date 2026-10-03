package lifecycle

import (
	"context"
	"errors"
	"image"
	"testing"
	"time"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/models/detr"
	"visionserve/internal/registry"
)

// scriptedEngine is a session that always returns the same outputs.
type scriptedEngine struct {
	outs  []engine.Tensor
	names []string
}

func (e *scriptedEngine) Run(context.Context, []engine.Tensor) ([]engine.Tensor, error) {
	return e.outs, nil
}
func (e *scriptedEngine) RunNamed(context.Context, map[string]engine.Tensor) ([]engine.Tensor, error) {
	return e.outs, nil
}
func (e *scriptedEngine) InputNames() []string      { return []string{"input"} }
func (e *scriptedEngine) OutputNames() []string     { return e.names }
func (e *scriptedEngine) ActiveEP() engine.Provider { return engine.ProviderCPU }
func (e *scriptedEngine) Close() error              { return nil }

// explainFixture is an RF-DETR session (the real decoder) over scripted outputs whose
// confidence order is NOT the query order:
//
//	query  box (cx,cy) in the input   class  conf (sigmoid)   attends to token
//	0      top-left                   a      0.12 (dropped)   0 (top-left)
//	1      top-right                  a      0.62             1 (top-right)
//	2      bottom-left                a      0.95             2 (bottom-left)
//	3      bottom-right               b      0.82             3 (bottom-right)
//
// so /api/predict returns [query 2, query 3, query 1]. The attention tensor makes each query
// attend to its own quadrant of a 2x2 token grid, which the heatmap shows.
func explainFixture(t *testing.T) (*Manager, image.Image) {
	t.Helper()
	const w, h = 64, 64
	base, err := detr.NewRFDETR(models.Config{Width: w, Height: h, ConfThresh: 0.5, Labels: []string{"a", "b"}})
	if err != nil {
		t.Fatal(err)
	}
	mdl := base.(models.Model)

	boxes := engine.F32([]float32{
		0.25, 0.25, 0.3, 0.3,
		0.75, 0.25, 0.3, 0.3,
		0.25, 0.75, 0.3, 0.3,
		0.75, 0.75, 0.3, 0.3,
	}, 1, 4, 4)
	logit := func(p float32) float32 { // inverse sigmoid, enough precision for the test
		switch p {
		case 0.12:
			return -2
		case 0.62:
			return 0.5
		case 0.95:
			return 3
		case 0.82:
			return 1.5
		}
		return -10
	}
	logits := engine.F32([]float32{
		logit(0.12), -10,
		logit(0.62), -10,
		logit(0.95), -10,
		-10, logit(0.82),
	}, 1, 4, 2)
	// [L=1, B=1, H=1, Q=4, S=4]: query q puts all its attention on token q.
	attn := make([]float32, 16)
	for q := 0; q < 4; q++ {
		attn[q*4+q] = 1
	}
	attnT := engine.F32(attn, 1, 1, 1, 4, 4)

	man := &registry.Manifest{Name: "rfx"}
	man.Input.Width, man.Input.Height = w, h
	man.Postprocess.BoxFormat = "cxcywh"
	man.Explain = &registry.ExplainConfig{Type: "attention", Outputs: map[string]string{"attention": "cross_attn_weights"}}

	detect := &scriptedEngine{outs: []engine.Tensor{boxes, logits}, names: []string{"pred_boxes", "pred_logits"}}
	explainEng := &scriptedEngine{
		outs:  []engine.Tensor{boxes, logits, attnT},
		names: []string{"pred_boxes", "pred_logits", "cross_attn_weights"},
	}
	s := newSimpleSession("rfx", models.TaskDetection, mdl, detect, 0, time.Now())
	s.man = man
	s.SetExplainEngine(explainEng)
	m := &Manager{live: map[string]*Session{"rfx": s}, stop: make(chan struct{})}
	// A non-square original (128x64) so the box has to be mapped back through meta.
	return m, image.NewRGBA(image.Rect(0, 0, 128, 64))
}

// quadrantMax returns which quadrant of a w×h heatmap holds its maximum: 0 top-left,
// 1 top-right, 2 bottom-left, 3 bottom-right.
func quadrantMax(hm []float32, w, h int) int {
	best, at := float32(-1), 0
	for i, v := range hm {
		if v > best {
			best, at = v, i
		}
	}
	x, y := at%w, at/w
	q := 0
	if x >= w/2 {
		q++
	}
	if y >= h/2 {
		q += 2
	}
	return q
}

// detection_idx N must explain the N-th detection /api/predict returns, not query N: the decoder
// drops and sorts queries, so the two differ (here: detection 0 is query 2). The old code read
// the attention of query N and labelled it as detection N.
func TestExplainUsesTheDetectionsQuery(t *testing.T) {
	m, img := explainFixture(t)
	pred, err := m.Predict(context.Background(), "rfx", img)
	if err != nil {
		t.Fatal(err)
	}
	if len(pred.Detections) != 3 {
		t.Fatalf("fixture: predict returned %d detections, want 3", len(pred.Detections))
	}

	for idx, wantQuery := range []int{2, 3, 1} {
		res, err := m.Explain(context.Background(), "rfx", img, ExplainRequest{DetectionIdx: idx})
		if err != nil {
			t.Fatalf("detection %d: %v", idx, err)
		}
		if res.Detection != pred.Detections[idx] {
			t.Errorf("detection %d: explained %+v, predict returned %+v", idx, res.Detection, pred.Detections[idx])
		}
		if res.Query != wantQuery {
			t.Errorf("detection %d: query %d, want %d", idx, res.Query, wantQuery)
		}
		if got := quadrantMax(res.Heatmap, res.Width, res.Height); got != wantQuery {
			t.Errorf("detection %d: heatmap peaks in quadrant %d, want %d (its query's)", idx, got, wantQuery)
		}
	}
}

// class picks the first detection of that class (by /api/predict's order) and explains ITS
// query; past the end of the list, or an absent class, is a request error, not another object.
func TestExplainClassAndRange(t *testing.T) {
	m, img := explainFixture(t)
	res, err := m.Explain(context.Background(), "rfx", img, ExplainRequest{Class: "b"})
	if err != nil {
		t.Fatal(err)
	}
	if res.Detection.Class != "b" || res.Query != 3 || quadrantMax(res.Heatmap, res.Width, res.Height) != 3 {
		t.Fatalf("class b: detection %+v query %d, want class b on query 3", res.Detection, res.Query)
	}

	for _, req := range []ExplainRequest{{DetectionIdx: 3}, {Class: "zebra"}} {
		if _, err := m.Explain(context.Background(), "rfx", img, req); !errors.Is(err, ErrInvalidRequest) {
			t.Errorf("%+v: err = %v, want ErrInvalidRequest", req, err)
		}
	}
}
