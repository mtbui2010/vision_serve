package server

import (
	"strings"
	"testing"

	"visionserve/pkg/api"
)

// A float form field that parses to NaN or ±Inf is a 400, not a silent "unset" (NaN) or a
// filter that drops everything (Inf); malformed numbers keep meaning "use the default".
func TestFormIntoRejectsNonFiniteFloats(t *testing.T) {
	for _, s := range []string{"NaN", "nan", "Inf", "-Inf", "+inf"} {
		var q api.PredictJSONRequest
		err := formInto(&q, func(name string) string {
			if name == "min_size" {
				return s
			}
			return ""
		})
		if err == nil || !strings.Contains(err.Error(), "min_size") {
			t.Errorf("min_size=%q: err = %v, want an error naming the field", s, err)
		}
	}
	var q api.PredictJSONRequest
	if err := formInto(&q, func(name string) string {
		switch name {
		case "min_size":
			return "0.5"
		case "max_size":
			return "abc" // malformed: stays the default, as before
		}
		return ""
	}); err != nil || q.MinSize != 0.5 || q.MaxSize != 0 {
		t.Fatalf("finite/malformed values: err %v, min %v max %v", err, q.MinSize, q.MaxSize)
	}
}
