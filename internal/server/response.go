package server

import (
	"encoding/json"
	"log"
	"net/http"

	"visionserve/pkg/api"
)

// writeJSON writes a JSON value with the given status code.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("server: failed to encode JSON: %v", err)
	}
}

// writeError returns JSON { "error": "..." } + the correct HTTP status (section 7 of the spec).
func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, api.ErrorResponse{Error: err.Error()})
}
