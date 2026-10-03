package server

import (
	"encoding/json"
	"log"
	"net/http"

	"visionserve/pkg/api"
)

// writeJSON writes a JSON value with the given status code.
//
// The body is encoded BEFORE the status is written: encoding can fail (a NaN or Inf anywhere in a
// result — JSON has no such numbers), and once WriteHeader(200) has gone out the client would get
// a 200 with an empty or truncated body. On failure it answers 500 with a JSON error instead.
func writeJSON(w http.ResponseWriter, status int, v any) {
	body, err := json.Marshal(v)
	if err != nil {
		log.Printf("server: failed to encode JSON: %v", err)
		status = http.StatusInternalServerError
		body, _ = json.Marshal(api.ErrorResponse{Error: "server: result could not be encoded as JSON: " + err.Error()})
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(append(body, '\n'))
}

// writeError returns JSON { "error": "..." } + the correct HTTP status (section 7 of the spec).
func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, api.ErrorResponse{Error: err.Error()})
}
