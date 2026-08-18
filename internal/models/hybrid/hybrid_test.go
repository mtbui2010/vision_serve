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

// The router reads text and dispatches; it never detects. The case that matters is the mixed
// prompt: it used to send the WHOLE request to GroundingDINO as soon as one word was out of
// vocabulary, which threw away the in-domain detector for the words it was trained on.
func TestPartition(t *testing.T) {
	m := &hybrid{vocab: map[string]bool{"person": true, "car": true, "dog": true}}
	cases := []struct {
		name          string
		classes       []string
		known, unknwn []string
	}{
		{"no prompt", nil, nil, nil},
		{"all in vocab", []string{"person", "car"}, []string{"person", "car"}, nil},
		{"all out of vocab", []string{"unicorn"}, nil, []string{"unicorn"}},
		{"mixed splits, order preserved", []string{"person", "unicorn", "dog", "zebra"},
			[]string{"person", "dog"}, []string{"unicorn", "zebra"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			known, unknown := m.partition(c.classes)
			if !reflect.DeepEqual(known, c.known) {
				t.Errorf("known = %v, want %v", known, c.known)
			}
			if !reflect.DeepEqual(unknown, c.unknwn) {
				t.Errorf("unknown = %v, want %v", unknown, c.unknwn)
			}
		})
	}
}

func TestJoinClasses(t *testing.T) {
	cases := []struct {
		in   []string
		want string
	}{
		{nil, ""},
		{[]string{"zebra"}, "zebra."},
		{[]string{"zebra", "snack bag"}, "zebra. snack bag."},
	}
	for _, c := range cases {
		if got := joinClasses(c.in); got != c.want {
			t.Errorf("joinClasses(%v) = %q, want %q", c.in, got, c.want)
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
