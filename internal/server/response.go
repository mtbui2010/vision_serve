package server

import (
	"encoding/json"
	"log"
	"net/http"

	"visionserve/pkg/api"
)

// writeJSON ghi một giá trị JSON với status code cho trước.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("server: encode JSON thất bại: %v", err)
	}
}

// writeError trả JSON { "error": "..." } + HTTP status đúng (mục 7 của spec).
func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, api.ErrorResponse{Error: err.Error()})
}
