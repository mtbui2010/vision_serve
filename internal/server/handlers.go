package server

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"strings"

	// register decoders for common image formats
	_ "image/jpeg"
	_ "image/png"

	"visionserve/internal/engine"
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
		infos = append(infos, withUsefulSide(api.ModelInfo{
			Name:    e.Manifest.Name,
			Task:    api.Task(e.Manifest.Task),
			License: e.Manifest.License,
			State:   state,
		}, e.Manifest))
	}
	writeJSON(w, http.StatusOK, infos)
}

// decodeLoadRequest reads the {"model": "..."} body of /api/load and /api/unload.
func decodeLoadRequest(r *http.Request) (string, error) {
	var req api.LoadRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		return "", badRequest(fmt.Errorf("invalid JSON body: %w", err))
	}
	if req.Model == "" {
		return "", badRequest(fmt.Errorf("missing 'model' field"))
	}
	return req.Model, nil
}

// POST /api/load { "model": "rf-detr" }
func (s *Server) handleLoad(w http.ResponseWriter, r *http.Request) {
	model, err := decodeLoadRequest(r)
	if err == nil {
		// A client that leaves stops waiting; the load itself carries on (lifecycle.Manager.Load).
		err = orClientGone(r.Context(), s.mgr.Load(r.Context(), model))
	}
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"model": model, "state": "loaded"})
}

// POST /api/unload { "model": "rf-detr" }
func (s *Server) handleUnload(w http.ResponseWriter, r *http.Request) {
	model, err := decodeLoadRequest(r)
	if err == nil {
		err = s.mgr.Unload(model)
	}
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"model": model, "state": "unloaded"})
}

// admit takes an admission slot for model on behalf of a request with context ctx, and checks
// that the client is still there. Every inference handler calls it BEFORE decoding the image (a
// multipart request before its image part is even read), so the memory a queued request holds
// stays bounded by its (compressed) upload. The caller must call release when err is nil.
func (s *Server) admit(ctx context.Context, model string) (release func(), err error) {
	release, err = s.mgr.Admit(ctx, model)
	if err != nil {
		if ctx.Err() != nil { // refused because the client had already left
			return nil, errClientGone
		}
		return nil, err
	}
	if ctx.Err() != nil { // the client left while the request was being admitted
		release()
		return nil, errClientGone
	}
	return release, nil
}

// closeUnreadBody is for a request refused before its body was read (admission comes first): it
// asks net/http to close the connection after the answer. Otherwise net/http reads up to 256 KiB
// more of the upload before sending the 503 — a client that pauses its upload never sees it — and
// past that size it closes the connection anyway. A body known to be smaller is left to net/http,
// which discards it and keeps the connection alive.
func closeUnreadBody(w http.ResponseWriter, r *http.Request) {
	const maxPostHandlerReadBytes = 256 << 10 // net/http's discard limit
	if r.ContentLength < 0 || r.ContentLength > maxPostHandlerReadBytes {
		w.Header().Set("Connection", "close")
	}
}

// admitter is s.admit for request r, the form decodeFields takes.
func (s *Server) admitter(r *http.Request) func(model string) (func(), error) {
	return func(model string) (func(), error) { return s.admit(r.Context(), model) }
}

// POST /api/predict
//   - multipart: model=<name>, image=<file>, depth=<file>, and the options of api.PredictJSONRequest
//   - or JSON: api.PredictJSONRequest { "model": "...", "image_base64": "...", ... }
//
// ?encoding=base64 (or the "encoding" field) returns depth_map / embeddings as base64 float32.
func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	res, enc, err := s.predict(w, r)
	if err != nil {
		writeError(w, err)
		return
	}
	writeResult(w, res, enc)
}

func (s *Server) predict(w http.ResponseWriter, r *http.Request) (api.Result, string, error) {
	q, err := decodeRequest(w, r, s.admitter(r))
	if err != nil {
		return api.Result{}, "", err
	}
	defer q.Close()
	if err := q.validate(true); err != nil {
		return api.Result{}, "", err
	}
	if err := q.admit(); err != nil {
		return api.Result{}, "", err
	}

	img, err := q.decodeImage()
	if err != nil {
		return api.Result{}, "", err
	}
	prompt, err := q.ToPrompt(img.Bounds().Dx(), img.Bounds().Dy())
	if err != nil {
		return api.Result{}, "", err
	}
	res, err := Predict(r.Context(), s.mgr, q.Model, img, prompt)
	return res, q.Encoding, err
}

// errBadDepth: a depth map was sent but does not match its declared size/dtype. It used to be
// dropped silently, so the request "worked" without the depth the client meant to use.
var errBadDepth = fmt.Errorf("invalid depth map: its byte length must equal depth_width*depth_height*"+
	"(2 for uint16, 4 for float32), each side <= %d", maxDepthSide)

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
	// Validate the declared size against the bytes BEFORE allocating: dw*dh came from the client,
	// and a 2-byte upload declaring 1e6 x 1e6 used to request 4 TB — a fatal, unrecoverable OOM.
	if dw > maxDepthSide || dh > maxDepthSide {
		return nil, 0, 0
	}
	n := dw * dh
	elem := 2
	if dt := strings.ToLower(strings.TrimSpace(dtype)); dt == "float32" || dt == "float" || dt == "f32" {
		elem = 4
	}
	if len(raw) != n*elem {
		return nil, 0, 0
	}
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

// POST /api/infer_tensor?model=<name>&shape=N,C,H,W[&encoding=base64]
//
//	body = raw little-endian float32, row-major NCHW (an ALREADY-PREPROCESSED tensor).
//
// Tensor-in path: skips server-side image decode + preprocess. For clients that already hold a
// pixel/feature tensor in memory (no JPEG/PNG round-trip) and for benchmarking the serving +
// inference layer head-to-head with tensor-in servers (e.g. Triton). Simple models only.
func (s *Server) handleInferTensor(w http.ResponseWriter, r *http.Request) {
	res, enc, err := s.inferTensor(w, r)
	if err != nil {
		writeError(w, err)
		return
	}
	writeResult(w, res, enc)
}

func (s *Server) inferTensor(w http.ResponseWriter, r *http.Request) (api.Result, string, error) {
	query := r.URL.Query()
	model := query.Get("model")
	if model == "" {
		return api.Result{}, "", badRequest(fmt.Errorf("missing 'model' query param"))
	}
	shape, err := parseShape(query.Get("shape"))
	if err != nil {
		return api.Result{}, "", badRequest(err)
	}
	enc, err := api.ParseEncoding(query.Get("encoding"))
	if err != nil {
		return api.Result{}, "", badRequest(err)
	}
	// Everything the slot depends on is in the URL, so admission comes before the body is read.
	release, err := s.admit(r.Context(), model)
	if err != nil {
		closeUnreadBody(w, r)
		return api.Result{}, "", err
	}
	defer release()

	raw, err := io.ReadAll(io.LimitReader(r.Body, maxTensorBytes+1))
	if err != nil {
		return api.Result{}, "", badRequest(fmt.Errorf("reading tensor body: %w", err))
	}
	if len(raw) > maxTensorBytes {
		return api.Result{}, "", tooLargeError{fmt.Sprintf("tensor body is larger than %d MiB", maxTensorBytes>>20)}
	}
	data, err := bytesToFloat32(raw)
	if err != nil {
		return api.Result{}, "", badRequest(err)
	}
	var want int64 = 1
	for _, d := range shape {
		want *= d
	}
	if int64(len(data)) != want {
		return api.Result{}, "", badRequest(fmt.Errorf("tensor has %d float32 but shape %v implies %d", len(data), shape, want))
	}
	if r.Context().Err() != nil {
		return api.Result{}, "", errClientGone
	}
	res, err := s.mgr.InferTensor(r.Context(), model, engine.Tensor{Data: data, Shape: shape, Dtype: "f32"})
	return res, enc, orClientGone(r.Context(), err)
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
