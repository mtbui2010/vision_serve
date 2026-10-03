package textalign

import (
	"fmt"
	"image"
	"reflect"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/pipeline"
)

type noTowers struct{ t *testing.T }

func (n noTowers) Run(role string, _ map[string]engine.Tensor) ([]engine.Tensor, error) {
	n.t.Errorf("tower %q ran with no usable box", role)
	return nil, fmt.Errorf("no towers")
}
func (noTowers) InputNames(string) []string  { return []string{"pixel_values"} }
func (noTowers) OutputNames(string) []string { return nil }

// B6: the vocabulary must be lowercased, not only its cache key. Before, "Cup" compiled a head
// labelled "Cup" under the same (lowercased) key as "cup", and whichever request came first
// decided the label every later request received.
func TestNormalizeVocabLowercases(t *testing.T) {
	got := normalizeVocab([]string{"Cup", " towel ", "CUP", "Cola Can"})
	if want := []string{"cup", "towel", "cola can"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("normalizeVocab = %q, want %q", got, want)
	}
	te := pipeline.NewTextEmbedder(roleText, nil, nil, 0)
	if te.Key(normalizeVocab([]string{"Cup"})[0]) != te.Key(normalizeVocab([]string{"cup"})[0]) {
		t.Fatal("same words, different cache keys")
	}
}

// B8: when every marked (sentinel) detection has a degenerate box, the namer must drop them and
// return the rest — not fail the request. No tower may be consulted for nothing.
func TestNameOpenCropsAllDegenerateDropsInsteadOfFailing(t *testing.T) {
	m := &textAlign{}
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	dets := []models.Detection{
		{Class: "cup", Conf: 0.9, BBox: [4]float64{10, 10, 20, 20}},
		{Class: openSentinel, Conf: 0.8, BBox: [4]float64{99.8, 5, 0.2, 10}}, // sub-pixel edge sliver
		{Class: openSentinel, Conf: 0.7, BBox: [4]float64{150, 150, 5, 5}},   // off-frame
	}
	got, err := m.nameOpenCrops(img, dets, []string{"cup", "zebra"}, 0, noTowers{t})
	if err != nil {
		t.Fatalf("nameOpenCrops failed the request: %v", err)
	}
	if len(got) != 1 || got[0].Class != "cup" {
		t.Fatalf("got %+v, want only the closed-head cup", got)
	}
}
