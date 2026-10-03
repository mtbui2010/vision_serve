package server

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"mime/multipart"
	"net/http"
	"sort"
)

// maxTemplateFormMemory is the part of a template upload kept in memory; the rest spills to
// temporary files (the whole body is capped by the route's limitBody).
const maxTemplateFormMemory = 64 << 20

// handleTemplateRegister: POST /api/templates
// Multipart form: name=<string>, images=<files (multiple)>
// Registers 1–N template images under the given name.
func (s *Server) handleTemplateRegister(w http.ResponseWriter, r *http.Request) {
	name, n, err := s.registerTemplates(r)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"name": name, "count": n})
}

func (s *Server) registerTemplates(r *http.Request) (string, int, error) {
	if err := r.ParseMultipartForm(maxTemplateFormMemory); err != nil {
		return "", 0, badRequest(fmt.Errorf("failed to parse form: %w", err))
	}
	name := r.FormValue("name")
	if name == "" {
		return "", 0, badRequest(fmt.Errorf(`"name" is required`))
	}
	// Accept multiple files under the key "images" (or "image" for single).
	var imgs []image.Image
	for _, key := range []string{"images", "image"} {
		for _, fh := range r.MultipartForm.File[key] {
			img, err := decodeUploadedImage(fh)
			if err != nil {
				return "", 0, err
			}
			imgs = append(imgs, img)
		}
	}
	if len(imgs) == 0 {
		return "", 0, badRequest(fmt.Errorf(`at least one "images" file is required`))
	}
	if err := s.tmpl.Register(name, imgs); err != nil {
		return "", 0, badRequest(err)
	}
	return name, len(imgs), nil
}

// handleTemplateList: GET /api/templates
func (s *Server) handleTemplateList(w http.ResponseWriter, r *http.Request) {
	names := s.tmpl.List()
	sort.Strings(names)
	writeJSON(w, http.StatusOK, map[string]any{"templates": names})
}

// handleTemplateDelete: DELETE /api/templates/{name}
func (s *Server) handleTemplateDelete(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if name == "" {
		writeError(w, badRequest(fmt.Errorf("name is required")))
		return
	}
	s.tmpl.Delete(name)
	w.WriteHeader(http.StatusNoContent)
}

func decodeUploadedImage(fh *multipart.FileHeader) (image.Image, error) {
	f, err := fh.Open()
	if err != nil {
		return nil, badRequest(fmt.Errorf("failed to decode image: %w", err))
	}
	defer f.Close()
	img, err := decodeImage(f)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %w", err) // keeps decodeImage's 400/413
	}
	return img, nil
}
