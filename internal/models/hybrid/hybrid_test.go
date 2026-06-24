package hybrid

import (
	"reflect"
	"testing"

	"visionserve/internal/models"
)

func TestParseClasses(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"", nil},
		{"   ", nil},
		{"cat. remote.", []string{"cat", "remote"}},
		{"Person.  DOG ", []string{"person", "dog"}},
		{"a..b.", []string{"a", "b"}},
	}
	for _, c := range cases {
		if got := parseClasses(c.in); !reflect.DeepEqual(got, c.want) {
			t.Errorf("parseClasses(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestRouteRFDETR(t *testing.T) {
	m := &hybrid{vocab: map[string]bool{"person": true, "car": true, "dog": true}}
	cases := []struct {
		classes []string
		want    bool // true → RF-DETR, false → GroundingDINO
	}{
		{nil, true},                            // no prompt → RF-DETR detects everything it knows
		{[]string{"person"}, true},             // in vocab
		{[]string{"person", "car"}, true},      // all in vocab
		{[]string{"unicorn"}, false},           // open-vocab class
		{[]string{"person", "unicorn"}, false}, // one out-of-vocab → GroundingDINO
	}
	for _, c := range cases {
		if got := m.routeRFDETR(c.classes); got != c.want {
			t.Errorf("routeRFDETR(%v) = %v, want %v", c.classes, got, c.want)
		}
	}
}

func TestFilterByClass(t *testing.T) {
	dets := []models.Detection{
		{Class: "person", Conf: 0.9},
		{Class: "Car", Conf: 0.8}, // case-insensitive match
		{Class: "dog", Conf: 0.7},
	}
	got := filterByClass(dets, []string{"person", "car"})
	if len(got) != 2 {
		t.Fatalf("filterByClass kept %d, want 2: %+v", len(got), got)
	}
	for _, d := range got {
		if d.Class == "dog" {
			t.Errorf("filterByClass should have dropped 'dog'")
		}
	}
}
