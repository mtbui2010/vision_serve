package models

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// ParsePrompt builds a Prompt from string inputs shared by the CLI flags and the HTTP
// form/JSON fields:
//   - text:  open-vocab text prompt, e.g. "cat. remote." (GroundingDINO / Grounded-SAM)
//   - boxStr:  SAM box prompt(s) "x,y,w,h", multiple separated by ';'
//   - pointStr: SAM point prompt(s) "x,y[,label]" (label 1=fg 0=bg), separated by ';'
//
// All inputs are optional; an empty result is valid for models that need no prompt.
// Every number must be finite (NaN/±Inf are rejected — they would otherwise reach the
// ONNX decoders and the mask/bbox arithmetic), and box width/height must be >= 0.
func ParsePrompt(text, boxStr, pointStr string) (Prompt, error) {
	p := Prompt{Text: strings.TrimSpace(text)}

	for _, part := range splitList(boxStr) {
		nums, err := parseFloats(part)
		if err != nil {
			return p, fmt.Errorf("invalid box %q: %v", part, err)
		}
		if len(nums) != 4 {
			return p, fmt.Errorf("invalid box %q (want \"x,y,w,h\")", part)
		}
		if nums[2] < 0 || nums[3] < 0 {
			return p, fmt.Errorf("invalid box %q: width and height must be >= 0 (got w=%v h=%v)", part, nums[2], nums[3])
		}
		p.Boxes = append(p.Boxes, [4]float64{nums[0], nums[1], nums[2], nums[3]})
	}

	for _, part := range splitList(pointStr) {
		nums, err := parseFloats(part)
		if err != nil {
			return p, fmt.Errorf("invalid point %q: %v", part, err)
		}
		if len(nums) != 2 && len(nums) != 3 {
			return p, fmt.Errorf("invalid point %q (want \"x,y[,label]\")", part)
		}
		label := 1 // default foreground
		if len(nums) == 3 {
			label = int(nums[2])
		}
		p.Points = append(p.Points, Point{X: nums[0], Y: nums[1], Label: label})
	}
	return p, nil
}

func splitList(s string) []string {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	var out []string
	for _, part := range strings.Split(s, ";") {
		if t := strings.TrimSpace(part); t != "" {
			out = append(out, t)
		}
	}
	return out
}

func parseFloats(s string) ([]float64, error) {
	fields := strings.Split(s, ",")
	nums := make([]float64, 0, len(fields))
	for _, f := range fields {
		v, err := strconv.ParseFloat(strings.TrimSpace(f), 64)
		if err != nil {
			return nil, err
		}
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("non-finite number %q", strings.TrimSpace(f))
		}
		nums = append(nums, v)
	}
	return nums, nil
}
