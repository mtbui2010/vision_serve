package pipeline

import (
	"fmt"
	"image"
	"math"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// cropTower answers the "crop" role with preset rows (one per crop, in order) and fails the test
// if the text tower is consulted: the namer's text rows are pre-seeded in its cache.
type cropTower struct {
	t     *testing.T
	rows  [][]float32
	calls int
}

func (c *cropTower) Run(role string, in map[string]engine.Tensor) ([]engine.Tensor, error) {
	if role != "crop" {
		c.t.Errorf("unexpected tower %q", role)
		return nil, fmt.Errorf("unexpected tower %q", role)
	}
	c.calls++
	n := int(in["pixel_values"].Shape[0])
	if n > len(c.rows) {
		return nil, fmt.Errorf("asked for %d crops, %d prepared", n, len(c.rows))
	}
	dim := len(c.rows[0])
	data := make([]float32, 0, n*dim)
	for i := 0; i < n; i++ {
		data = append(data, c.rows[i]...)
	}
	return []engine.Tensor{engine.F32(data, int64(n), int64(dim))}, nil
}
func (c *cropTower) InputNames(string) []string  { return []string{"pixel_values"} }
func (c *cropTower) OutputNames(string) []string { return nil }

// testNamer seeds one basis-vector text row per word, so a crop row's cosine against word k is
// its (normalised) k-th component.
func testNamer(words []string, temp float64) *CropNamer {
	e := NewTextEmbedder("text", nil, nil, 0)
	for k, w := range words {
		row := make([]float32, len(words))
		row[k] = 1
		e.cache.Add(e.Key(w), row)
	}
	return &CropNamer{CropRole: "crop", Text: e, Temp: temp, Floor: 0}
}

func canvas() image.Image { return image.NewRGBA(image.Rect(0, 0, 100, 100)) }

func unit(v ...float32) []float32 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	inv := float32(1 / math.Sqrt(s))
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x * inv
	}
	return out
}

func TestNameArgmaxSoftmaxAndTemperature(t *testing.T) {
	words := []string{"towel", "hat"}
	rows := [][]float32{{0.1, 0.4}, {0.5, 0.2}}
	boxes := [][4]float64{{10, 10, 20, 20}, {40, 40, 20, 20}}
	for _, c := range []struct {
		reqTemp, want float64
	}{{0, 0.05}, {0.2, 0.2}} {
		n := testNamer(words, 0.05)
		got, err := n.Name(Call{Img: canvas(), Runner: &cropTower{t: t, rows: rows}}, boxes, words, c.reqTemp)
		if err != nil {
			t.Fatal(err)
		}
		for i, wantWord := range []int{1, 0} {
			cos := unit(rows[i]...)
			want := SoftmaxAt(cos, wantWord, c.want)
			if got[i].Word != wantWord || got[i].P != want {
				t.Errorf("T=%v box %d: %+v, want word %d p %v", c.reqTemp, i, got[i], wantWord, want)
			}
		}
	}
}

func TestNameRejectsBelowFloorAndDegenerateBoxes(t *testing.T) {
	words := []string{"towel", "hat"}
	tower := &cropTower{t: t, rows: [][]float32{{-0.3, -0.4}, {0.2, 0.6}}}
	boxes := [][4]float64{
		{10, 10, 20, 20},
		{50, 50, 0, 0}, // degenerate: never embedded
		{40, 40, 20, 20},
	}
	got, err := testNamer(words, 0.05).Name(Call{Img: canvas(), Runner: tower}, boxes, words, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got[0].Word != -1 || got[1].Word != -1 || got[2].Word != 1 {
		t.Errorf("names = %+v; want [rejected below floor, rejected degenerate, hat]", got)
	}
	if tower.calls != 1 {
		t.Errorf("crop tower ran %d times, want one batch", tower.calls)
	}
}

// Nothing usable → nothing runs, everything is rejected, and the request does not fail. A nil
// namer is fine on that path (textalign's zero-value tests rely on it).
func TestNameNothingUsableRunsNoTower(t *testing.T) {
	var n *CropNamer
	got, err := n.Name(Call{Img: canvas(), Runner: &cropTower{t: t}}, [][4]float64{{99.8, 10, 0.2, 5}}, []string{"zebra"}, 0)
	if err != nil || len(got) != 1 || got[0].Word != -1 {
		t.Fatalf("got %+v, %v; want one rejected box and no error", got, err)
	}
}

func TestCropRescorerRenamesReweightsAndDrops(t *testing.T) {
	words := []string{"towel", "hat"}
	rows := [][]float32{{0.1, 0.4}, {-0.5, -0.2}}
	dets := []models.Detection{
		{Class: "towel", Conf: 0.9, BBox: [4]float64{10, 10, 20, 20}},
		{Class: "hat", Conf: 0.8, BBox: [4]float64{40, 40, 20, 20}},
	}
	rs := CropRescorer{Namer: testNamer(words, 0.05)}
	got, err := rs.Rescore(Call{Img: canvas(), Prompt: models.Prompt{CropTemp: 0.1}, Runner: &cropTower{t: t, rows: rows}}, dets, words)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Class != "hat" {
		t.Fatalf("got %+v, want only the first detection, renamed hat", got)
	}
	if want := 0.9 * float64(SoftmaxAt(unit(rows[0]...), 1, 0.1)); got[0].Conf != want {
		t.Errorf("conf = %v, want %v (request crop_temp 0.1)", got[0].Conf, want)
	}
	if dets[0].Class != "towel" {
		t.Error("Rescore modified its input")
	}
	// Nothing to do: input returned as is, no tower consulted.
	if out, err := rs.Rescore(Call{Runner: &cropTower{t: t}}, dets, nil); err != nil || len(out) != 2 {
		t.Errorf("no words: %v, %v", out, err)
	}
}

func TestSoftmaxAt(t *testing.T) {
	row := []float32{0.1, 0.4, 0.2}
	var sum float64
	for _, v := range row {
		sum += math.Exp(float64(v) / 0.05)
	}
	if got, want := float64(SoftmaxAt(row, 1, 0.05)), math.Exp(0.4/0.05)/sum; math.Abs(got-want) > 1e-6 {
		t.Errorf("SoftmaxAt = %.9f, want %.9f", got, want)
	}
	if SoftmaxAt(row, 5, 0.05) != 0 || SoftmaxAt(nil, 0, 0.05) != 0 || SoftmaxAt([]float32{0.3}, 0, 0.05) != 1 {
		t.Error("edge cases: out of range → 0, empty → 0, single → 1")
	}
	if got := SoftmaxAt([]float32{40, 1}, 0, 0.02); math.IsNaN(float64(got)) || got != 1 {
		t.Errorf("saturating row gave %v, want 1", got)
	}
}
