package classification

import (
	"math"
	"strings"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// refSoftmax is the textbook definition exp(x_i) / sum_j exp(x_j), in float64 and without the
// max-shift, as an independent reference (inputs here are small enough not to overflow).
func refSoftmax(logits []float32) []float64 {
	var sum float64
	out := make([]float64, len(logits))
	for i, v := range logits {
		out[i] = math.Exp(float64(v))
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func logits(shape []int64, data ...float32) []engine.Tensor {
	return []engine.Tensor{{Shape: shape, Data: data}}
}

func TestPostprocessTopKSortedWithSoftmaxProbabilities(t *testing.T) {
	data := []float32{1, 3, 2, 0, -1}
	cfg := models.Config{Labels: []string{"a", "b", "c", "d", "e"}, MaxDet: 3}
	res, err := postprocess(logits([]int64{1, 5}, data...), models.PreprocessMeta{}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if res.Task != models.TaskClassification || len(res.Detections) != 0 || len(res.Masks) != 0 {
		t.Fatalf("want a classification-only result, got %+v", res)
	}
	ref := refSoftmax(data)
	want := []struct {
		class string
		idx   int
	}{{"b", 1}, {"c", 2}, {"a", 0}}
	if len(res.Classifications) != len(want) {
		t.Fatalf("got %d classes, want %d", len(res.Classifications), len(want))
	}
	for i, w := range want {
		c := res.Classifications[i]
		if c.Class != w.class || math.Abs(c.Conf-ref[w.idx]) > 1e-6 {
			t.Errorf("rank %d: got %s %.7f, want %s %.7f", i, c.Class, c.Conf, w.class, ref[w.idx])
		}
	}
}

func TestPostprocessKDefaultsAndClamps(t *testing.T) {
	data := make([]float32, 8)
	for i := range data {
		data[i] = float32(i)
	}
	res, err := postprocess(logits([]int64{1, 8}, data...), models.PreprocessMeta{}, models.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Classifications) != 5 {
		t.Fatalf("max_detections unset: want the default K=5, got %d", len(res.Classifications))
	}
	// No labels: classes are named by index; highest logit first.
	if res.Classifications[0].Class != "class_7" || res.Classifications[4].Class != "class_3" {
		t.Fatalf("unexpected order/naming: %+v", res.Classifications)
	}

	res, err = postprocess(logits([]int64{3}, 0, 1, 2), models.PreprocessMeta{}, models.Config{MaxDet: 10})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Classifications) != 3 {
		t.Fatalf("K above the class count must clamp to it ([C] shape), got %d", len(res.Classifications))
	}
	var sum float64
	for _, c := range res.Classifications {
		sum += c.Conf
	}
	if math.Abs(sum-1) > 1e-6 {
		t.Fatalf("all classes returned: probabilities must sum to 1, got %v", sum)
	}
}

// Large logits do not overflow (the max is subtracted first) and keep the exact ratios.
func TestPostprocessLargeLogitsAreStable(t *testing.T) {
	res, err := postprocess(logits([]int64{1, 2}, 1000, 999), models.PreprocessMeta{}, models.Config{MaxDet: 2})
	if err != nil {
		t.Fatal(err)
	}
	p0, p1 := res.Classifications[0].Conf, res.Classifications[1].Conf
	want0 := 1 / (1 + math.Exp(-1))
	if math.IsNaN(p0) || math.Abs(p0-want0) > 1e-6 || math.Abs(p1-(1-want0)) > 1e-6 {
		t.Fatalf("got %v %v, want %v %v", p0, p1, want0, 1-want0)
	}
}

// Equal probabilities keep class-index order, so the output is deterministic.
func TestPostprocessTiesKeepIndexOrder(t *testing.T) {
	cfg := models.Config{Labels: []string{"a", "b", "c", "d"}, MaxDet: 4}
	res, err := postprocess(logits([]int64{1, 4}, 0, 5, 0, 5), models.PreprocessMeta{}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, c := range res.Classifications {
		got = append(got, c.Class)
	}
	if strings.Join(got, ",") != "b,d,a,c" {
		t.Fatalf("got %v, want b,d,a,c", got)
	}
}

// conf_threshold is documented as not read: low-probability classes are still returned.
func TestPostprocessIgnoresConfThreshold(t *testing.T) {
	res, err := postprocess(logits([]int64{1, 3}, 0, 0, 0), models.PreprocessMeta{}, models.Config{MaxDet: 3, ConfThresh: 0.9})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Classifications) != 3 {
		t.Fatalf("want all 3 classes (conf_threshold ignored), got %d", len(res.Classifications))
	}
}

func TestPostprocessRejectsBadOutputs(t *testing.T) {
	for name, outs := range map[string][]engine.Tensor{
		"no outputs":      nil,
		"batch of 2":      logits([]int64{2, 2}, 1, 2, 3, 4),
		"rank 3":          logits([]int64{1, 1, 2}, 1, 2),
		"length mismatch": logits([]int64{1, 3}, 1, 2),
	} {
		if _, err := postprocess(outs, models.PreprocessMeta{}, models.Config{}); err == nil {
			t.Errorf("%s: want an error", name)
		}
	}
}
