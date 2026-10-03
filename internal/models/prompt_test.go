package models

import (
	"strings"
	"testing"
)

func TestParsePromptValid(t *testing.T) {
	p, err := ParsePrompt(" cat. ", "10,20,30,40; 0,0,0,0", "5,6;7,8,0")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p.Text != "cat." || len(p.Boxes) != 2 || p.Boxes[0] != [4]float64{10, 20, 30, 40} {
		t.Fatalf("bad prompt: %+v", p)
	}
	if len(p.Points) != 2 || p.Points[0].Label != 1 || p.Points[1].Label != 0 {
		t.Fatalf("bad points: %+v", p.Points)
	}
}

// TestParsePromptRejectsNonFinite: strconv.ParseFloat accepts "NaN", "Inf", "-Inf",
// "+Inf", "infinity" — none of them may reach a model.
func TestParsePromptRejectsNonFinite(t *testing.T) {
	for _, v := range []string{"NaN", "nan", "Inf", "-Inf", "+inf", "infinity", "1e400"} {
		if _, err := ParsePrompt("", "1,2,3,"+v, ""); err == nil {
			t.Errorf("box with %q: expected an error", v)
		} else if !strings.Contains(err.Error(), "invalid box") {
			t.Errorf("box with %q: unclear error %v", v, err)
		}
		if _, err := ParsePrompt("", "", v+",2"); err == nil {
			t.Errorf("point with %q: expected an error", v)
		}
		if _, err := ParsePrompt("", "", "1,2,"+v); err == nil {
			t.Errorf("point label %q: expected an error", v)
		}
	}
}

func TestParsePromptRejectsNegativeSize(t *testing.T) {
	for _, b := range []string{"0,0,-1,5", "0,0,5,-1", "0,0,-3,-3"} {
		_, err := ParsePrompt("", b, "")
		if err == nil || !strings.Contains(err.Error(), "width and height") {
			t.Errorf("box %q: want a width/height error, got %v", b, err)
		}
	}
	// Negative x/y is a valid (partly out-of-frame) box.
	if _, err := ParsePrompt("", "-5,-5,10,10", ""); err != nil {
		t.Errorf("negative origin should be accepted: %v", err)
	}
}
