package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"net/http"
	"strconv"

	// register decoders for common image formats
	_ "image/jpeg"
	_ "image/png"

	"visionserve/internal/models"
	"visionserve/pkg/api"
)

// maxImageBytes limits the uploaded image size (to prevent OOM). 32 MiB.
const maxImageBytes = 32 << 20

// GET /api/health
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// GET /api/models — list models + state (available / loaded).
func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	infos := make([]api.ModelInfo, 0)
	for _, e := range s.reg.List() {
		state := "not_downloaded"
		if e.Manifest.WeightsExist() {
			state = "available"
		}
		if s.mgr.IsLoaded(e.Manifest.Name) {
			state = "loaded"
		}
		infos = append(infos, api.ModelInfo{
			Name:    e.Manifest.Name,
			Task:    api.Task(e.Manifest.Task),
			License: e.Manifest.License,
			State:   state,
		})
	}
	writeJSON(w, http.StatusOK, infos)
}

// POST /api/load { "model": "rf-detr" }
func (s *Server) handleLoad(w http.ResponseWriter, r *http.Request) {
	var req api.LoadRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("invalid JSON body: %w", err))
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("missing 'model' field"))
		return
	}
	if err := s.mgr.Load(req.Model); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"model": req.Model, "state": "loaded"})
}

// POST /api/unload { "model": "rf-detr" }
func (s *Server) handleUnload(w http.ResponseWriter, r *http.Request) {
	var req api.LoadRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("invalid JSON body: %w", err))
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("missing 'model' field"))
		return
	}
	if err := s.mgr.Unload(req.Model); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"model": req.Model, "state": "unloaded"})
}

// POST /api/predict
//   - multipart: model=<name>, image=<file>
//   - or JSON: { "model": "...", "image_base64": "..." }
func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	model, img, prompt, minSize, maxSize, err := s.parsePredictRequest(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	res, err := s.mgr.PredictPrompt(model, img, prompt)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if minSize > 0 || maxSize > 0 {
		res = api.FilterBySizePct(res, minSize, maxSize, img.Bounds().Dx(), img.Bounds().Dy())
	}
	writeJSON(w, http.StatusOK, res)
}

func (s *Server) parsePredictRequest(r *http.Request) (string, image.Image, models.Prompt, float64, float64, error) {
	ct := r.Header.Get("Content-Type")

	// JSON branch (image_base64 + optional prompt fields)
	if len(ct) >= 16 && ct[:16] == "application/json" {
		var req api.PredictJSONRequest
		if err := json.NewDecoder(io.LimitReader(r.Body, maxImageBytes)).Decode(&req); err != nil {
			return "", nil, models.Prompt{}, 0, 0, fmt.Errorf("invalid JSON body: %w", err)
		}
		if req.Model == "" || req.ImageBase64 == "" {
			return "", nil, models.Prompt{}, 0, 0, fmt.Errorf("both 'model' and 'image_base64' are required")
		}
		raw, err := base64.StdEncoding.DecodeString(req.ImageBase64)
		if err != nil {
			return "", nil, models.Prompt{}, 0, 0, fmt.Errorf("invalid image_base64: %w", err)
		}
		img, _, err := image.Decode(bytes.NewReader(raw))
		if err != nil {
			return "", nil, models.Prompt{}, 0, 0, fmt.Errorf("failed to decode image: %w", err)
		}
		prompt, err := models.ParsePrompt(req.Prompt, req.Box, req.Point)
		if err != nil {
			return "", nil, models.Prompt{}, 0, 0, err
		}
		return req.Model, img, prompt, req.MinSize, req.MaxSize, nil
	}

	// Multipart branch (form fields: model, image, optional prompt/box/point/min_size/max_size)
	if err := r.ParseMultipartForm(maxImageBytes); err != nil {
		return "", nil, models.Prompt{}, 0, 0, fmt.Errorf("failed to parse multipart form: %w", err)
	}
	model := r.FormValue("model")
	if model == "" {
		return "", nil, models.Prompt{}, 0, 0, fmt.Errorf("missing 'model' field")
	}
	file, _, err := r.FormFile("image")
	if err != nil {
		return "", nil, models.Prompt{}, 0, 0, fmt.Errorf("missing 'image' file: %w", err)
	}
	defer file.Close()
	img, _, err := image.Decode(io.LimitReader(file, maxImageBytes))
	if err != nil {
		return "", nil, models.Prompt{}, 0, 0, fmt.Errorf("failed to decode image: %w", err)
	}
	prompt, err := models.ParsePrompt(r.FormValue("prompt"), r.FormValue("box"), r.FormValue("point"))
	if err != nil {
		return "", nil, models.Prompt{}, 0, 0, err
	}
	// parse size filters; ignore parse errors (default 0 = no limit)
	minSize, _ := strconv.ParseFloat(r.FormValue("min_size"), 64)
	maxSize, _ := strconv.ParseFloat(r.FormValue("max_size"), 64)
	return model, img, prompt, minSize, maxSize, nil
}
