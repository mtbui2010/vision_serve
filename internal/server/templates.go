package server

import (
	"encoding/json"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"mime/multipart"
	"net/http"
	"sort"
)

// handleTemplateRegister: POST /api/templates
// Multipart form: name=<string>, images=<files (multiple)>
// Registers 1–N template images under the given name.
func (s *Server) handleTemplateRegister(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(64 << 20); err != nil { // 64 MiB
		http.Error(w, "failed to parse form: "+err.Error(), http.StatusBadRequest)
		return
	}
	name := r.FormValue("name")
	if name == "" {
		http.Error(w, `"name" is required`, http.StatusBadRequest)
		return
	}

	// Accept multiple files under the key "images" (or "image" for single).
	var imgs []image.Image
	for _, key := range []string{"images", "image"} {
		files := r.MultipartForm.File[key]
		for _, fh := range files {
			img, err := decodeUploadedImage(fh)
			if err != nil {
				http.Error(w, "failed to decode image: "+err.Error(), http.StatusBadRequest)
				return
			}
			imgs = append(imgs, img)
		}
	}
	if len(imgs) == 0 {
		http.Error(w, `at least one "images" file is required`, http.StatusBadRequest)
		return
	}

	if err := s.tmpl.Register(name, imgs); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{ //nolint:errcheck
		"name":  name,
		"count": len(imgs),
	})
}

// handleTemplateList: GET /api/templates
func (s *Server) handleTemplateList(w http.ResponseWriter, r *http.Request) {
	names := s.tmpl.List()
	sort.Strings(names)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"templates": names}) //nolint:errcheck
}

// handleTemplateDelete: DELETE /api/templates/{name}
func (s *Server) handleTemplateDelete(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if name == "" {
		http.Error(w, "name is required", http.StatusBadRequest)
		return
	}
	s.tmpl.Delete(name)
	w.WriteHeader(http.StatusNoContent)
}

func decodeUploadedImage(fh *multipart.FileHeader) (image.Image, error) {
	f, err := fh.Open()
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(io.LimitReader(f, maxImageBytes))
	return img, err
}
