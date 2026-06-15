package server

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"math"
	"net/http"
	"strconv"
	"strings"

	// register decoders for common image formats
	_ "image/jpeg"
	_ "image/png"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/morph"
	roipkg "visionserve/internal/roi"
	"visionserve/pkg/api"
)

// maxImageBytes limits the uploaded image size (to prevent OOM). 32 MiB.
const maxImageBytes = 32 << 20

// maxTensorBytes limits a raw tensor body (to prevent OOM). 128 MiB.
const maxTensorBytes = 128 << 20

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

	// Region of interest (generic, crop semantics): crop the image to the ROI, run the
	// model on the crop (it only sees the ROI), then map results back to original coords.
	fullW, fullH := img.Bounds().Dx(), img.Bounds().Dy()
	rect, hasROI := roipkg.Clamp(prompt.ROI, fullW, fullH)
	if hasROI {
		img = roipkg.Crop(img, rect)
		roipkg.ShiftPrompt(&prompt, rect)
	}

	res, err := s.mgr.PredictPrompt(model, img, prompt)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	if hasROI {
		res = roipkg.MapResult(res, rect, fullW, fullH)
	}
	// Mask morphology (enlarge/shrink) in ORIGINAL-image terms, then size filter.
	morph.ApplyToMasks(res.Masks, fullW, fullH, prompt.Dilate)
	if minSize > 0 || maxSize > 0 {
		res = api.FilterBySizePct(res, minSize, maxSize, fullW, fullH)
	}
	writeJSON(w, http.StatusOK, res)
}

// parseDepth turns a raw little-endian depth array into a normalized float map (row-major,
// NaN = invalid), resized to the RGB image (imgW×imgH). dtype is "float32" (kept as-is, ≤0/NaN
// → invalid) or "uint16" (default; /65535, 0 → invalid). dw/dh default to the image size.
// Returns nil when the bytes don't match the declared dtype×dims.
func parseDepth(raw []byte, dtype string, dw, dh, imgW, imgH int) ([]float32, int, int) {
	if len(raw) == 0 || imgW <= 0 || imgH <= 0 {
		return nil, 0, 0
	}
	if dw <= 0 || dh <= 0 {
		dw, dh = imgW, imgH
	}
	n := dw * dh
	depth := make([]float32, n)
	switch strings.ToLower(strings.TrimSpace(dtype)) {
	case "float32", "float", "f32":
		if len(raw) != n*4 {
			return nil, 0, 0
		}
		for i := 0; i < n; i++ {
			v := math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
			if f := float64(v); math.IsNaN(f) || f <= 0 {
				depth[i] = float32(math.NaN())
			} else {
				depth[i] = v
			}
		}
	default: // uint16
		if len(raw) != n*2 {
			return nil, 0, 0
		}
		for i := 0; i < n; i++ {
			if u := binary.LittleEndian.Uint16(raw[i*2:]); u == 0 {
				depth[i] = float32(math.NaN()) // 0 = no measurement
			} else {
				depth[i] = float32(u) / 65535.0
			}
		}
	}
	if dw != imgW || dh != imgH {
		depth = resizeDepthNearestF(depth, dw, dh, imgW, imgH)
		dw, dh = imgW, imgH
	}
	return depth, dw, dh
}

// resizeDepthNearestF nearest-neighbor resizes a float depth map (NaN-preserving).
func resizeDepthNearestF(src []float32, sw, sh, dw, dh int) []float32 {
	out := make([]float32, dw*dh)
	for y := 0; y < dh; y++ {
		sy := y * sh / dh
		if sy >= sh {
			sy = sh - 1
		}
		for x := 0; x < dw; x++ {
			sx := x * sw / dw
			if sx >= sw {
				sx = sw - 1
			}
			out[y*dw+x] = src[sy*sw+sx]
		}
	}
	return out
}

// POST /api/infer_tensor?model=<name>&shape=N,C,H,W
//
//	body = raw little-endian float32, row-major NCHW (an ALREADY-PREPROCESSED tensor).
//
// Tensor-in path: skips server-side image decode + preprocess. For clients that already hold a
// pixel/feature tensor in memory (no JPEG/PNG round-trip) and for benchmarking the serving +
// inference layer head-to-head with tensor-in servers (e.g. Triton). Simple models only.
func (s *Server) handleInferTensor(w http.ResponseWriter, r *http.Request) {
	model := r.URL.Query().Get("model")
	if model == "" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("missing 'model' query param"))
		return
	}
	shape, err := parseShape(r.URL.Query().Get("shape"))
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	raw, err := io.ReadAll(io.LimitReader(r.Body, maxTensorBytes))
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("reading tensor body: %w", err))
		return
	}
	data, err := bytesToFloat32(raw)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	var want int64 = 1
	for _, d := range shape {
		want *= d
	}
	if int64(len(data)) != want {
		writeError(w, http.StatusBadRequest,
			fmt.Errorf("tensor has %d float32 but shape %v implies %d", len(data), shape, want))
		return
	}
	res, err := s.mgr.InferTensor(model, engine.Tensor{Data: data, Shape: shape, Dtype: "f32"})
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, res)
}

// parseShape parses "N,C,H,W" into positive int64 dims.
func parseShape(s string) ([]int64, error) {
	if strings.TrimSpace(s) == "" {
		return nil, fmt.Errorf("missing 'shape' query param (e.g. shape=1,3,224,224)")
	}
	parts := strings.Split(s, ",")
	out := make([]int64, len(parts))
	for i, p := range parts {
		v, err := strconv.ParseInt(strings.TrimSpace(p), 10, 64)
		if err != nil || v <= 0 {
			return nil, fmt.Errorf("invalid shape %q (want positive ints, e.g. 1,3,224,224)", s)
		}
		out[i] = v
	}
	return out, nil
}

// bytesToFloat32 reinterprets a little-endian byte slice as []float32.
func bytesToFloat32(b []byte) ([]float32, error) {
	if len(b)%4 != 0 {
		return nil, fmt.Errorf("tensor body length %d is not a multiple of 4 (float32)", len(b))
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out, nil
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
		// Per-request predict options threaded to the model (the grasp model reads them).
		prompt.MinSize, prompt.MaxSize = req.MinSize, req.MaxSize
		prompt.GripperMin, prompt.GripperMax = req.GripperMin, req.GripperMax
		prompt.BoxThresh, prompt.TextThresh = req.BoxThreshold, req.TextThreshold
		prompt.BgMaxArea, prompt.FgMinArea = req.BgMaxArea, req.FgMinArea
		prompt.GridSize = req.GridSize
		prompt.Method = req.Method
		prompt.ROI = roipkg.Parse(req.ROI)
		prompt.Dilate = req.Dilate
		if req.DepthBase64 != "" {
			if raw, e := base64.StdEncoding.DecodeString(req.DepthBase64); e == nil {
				prompt.Depth, prompt.DepthW, prompt.DepthH = parseDepth(
					raw, req.DepthDtype, req.DepthWidth, req.DepthHeight, img.Bounds().Dx(), img.Bounds().Dy())
			}
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
	// grasp gripper bounds (px); ignore parse errors (default 0 = manifest default)
	gripperMin, _ := strconv.ParseFloat(r.FormValue("gripper_min"), 64)
	gripperMax, _ := strconv.ParseFloat(r.FormValue("gripper_max"), 64)
	// GroundingDINO threshold overrides; ignore parse errors (default 0 = manifest/default)
	boxThresh, _ := strconv.ParseFloat(r.FormValue("box_threshold"), 64)
	textThresh, _ := strconv.ParseFloat(r.FormValue("text_threshold"), 64)
	// foreground background/noise area thresholds (% of image); 0 = model default
	bgMaxArea, _ := strconv.ParseFloat(r.FormValue("bg_max_area"), 64)
	fgMinArea, _ := strconv.ParseFloat(r.FormValue("fg_min_area"), 64)
	gridSize, _ := strconv.Atoi(r.FormValue("grid_size"))
	prompt.MinSize, prompt.MaxSize = minSize, maxSize
	prompt.GripperMin, prompt.GripperMax = gripperMin, gripperMax
	prompt.BoxThresh, prompt.TextThresh = boxThresh, textThresh
	prompt.BgMaxArea, prompt.FgMinArea = bgMaxArea, fgMinArea
	prompt.GridSize = gridSize
	prompt.Method = r.FormValue("method")
	prompt.ROI = roipkg.Parse(r.FormValue("roi"))
	dilate, _ := strconv.Atoi(r.FormValue("dilate"))
	prompt.Dilate = dilate
	// Optional external depth map (RGB-D) as a raw little-endian array file.
	if df, _, derr := r.FormFile("depth"); derr == nil {
		defer df.Close()
		if raw, e := io.ReadAll(io.LimitReader(df, maxTensorBytes)); e == nil {
			dw, _ := strconv.Atoi(r.FormValue("depth_width"))
			dh, _ := strconv.Atoi(r.FormValue("depth_height"))
			prompt.Depth, prompt.DepthW, prompt.DepthH = parseDepth(
				raw, r.FormValue("depth_dtype"), dw, dh, img.Bounds().Dx(), img.Bounds().Dy())
		}
	}
	return model, img, prompt, minSize, maxSize, nil
}
