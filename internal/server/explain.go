package server

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"io"
	"math"
	"net/http"
	"strconv"

	// register image format decoders
	_ "image/jpeg"
	_ "image/png"

	"visionserve/internal/explain"
	"visionserve/internal/lifecycle"
)

// POST /api/explain
//
// Multipart form fields:
//
//	model          string  required   model name (must be loaded and have an explain manifest block)
//	image          file    required   JPEG/PNG image
//	detection_idx  int     optional   0-based detection index to explain (default 0)
//	top_channels   int     optional   Score-CAM: max channels to sample (0 = manifest default)
//	alpha          float   optional   overlay opacity [0,1] (default 0.5)
//	format         string  optional   "png" (default) or "numpy" (raw float32 bytes)
//	class          string  optional   class name hint — passed to the lifecycle for scoring
//
// Responses:
//   - format=png:   Content-Type: image/png — heatmap overlaid on the input image
//   - format=numpy: Content-Type: application/octet-stream — raw little-endian float32
//     with headers X-Heatmap-Shape (H,W) and X-Heatmap-Dtype (float32).
func (s *Server) handleExplain(w http.ResponseWriter, r *http.Request) {
	// --- 1. Parse multipart form.
	if err := r.ParseMultipartForm(maxImageBytes); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("failed to parse form: %w", err))
		return
	}

	modelName := r.FormValue("model")
	if modelName == "" {
		writeError(w, http.StatusBadRequest, fmt.Errorf(`"model" field is required`))
		return
	}

	f, _, err := r.FormFile("image")
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf(`"image" file is required: %w`, err))
		return
	}
	defer f.Close()

	img, _, err := image.Decode(io.LimitReader(f, maxImageBytes))
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("failed to decode image: %w", err))
		return
	}

	// --- 2. Parse optional parameters.
	detectionIdx := 0
	if v := r.FormValue("detection_idx"); v != "" {
		if n, err2 := strconv.Atoi(v); err2 == nil && n >= 0 {
			detectionIdx = n
		}
	}

	topChannels := 0 // 0 = use manifest default
	if v := r.FormValue("top_channels"); v != "" {
		if n, err2 := strconv.Atoi(v); err2 == nil && n > 0 {
			topChannels = n
		}
	}

	alpha := float32(0.5)
	if v := r.FormValue("alpha"); v != "" {
		if f2, err2 := strconv.ParseFloat(v, 32); err2 == nil {
			a := float32(f2)
			if a >= 0 && a <= 1 {
				alpha = a
			}
		}
	}

	format := r.FormValue("format")
	if format == "" {
		format = "png"
	}

	class := r.FormValue("class")

	// --- 3. Delegate to lifecycle.Manager.Explain.
	req := lifecycle.ExplainRequest{
		Class:        class,
		DetectionIdx: detectionIdx,
		TopChannels:  topChannels,
		Alpha:        alpha,
	}

	result, err := s.mgr.Explain(modelName, img, req)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	// --- 4. Render output.
	switch format {
	case "numpy":
		// Raw little-endian float32 bytes with shape in headers.
		buf := make([]byte, len(result.Heatmap)*4)
		for i, v := range result.Heatmap {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("X-Heatmap-Shape", fmt.Sprintf("%d,%d", result.Height, result.Width))
		w.Header().Set("X-Heatmap-Dtype", "float32")
		w.Header().Set("Content-Length", strconv.Itoa(len(buf)))
		_, _ = w.Write(buf)

	default: // "png"
		var buf bytes.Buffer
		if err := explain.RenderPNG(&buf, img, result.Heatmap, result.Width, result.Height, alpha); err != nil {
			writeError(w, http.StatusInternalServerError, fmt.Errorf("failed to render PNG: %w", err))
			return
		}
		w.Header().Set("Content-Type", "image/png")
		w.Header().Set("Content-Length", strconv.Itoa(buf.Len()))
		_, _ = w.Write(buf.Bytes())
	}
}
