package server

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"strconv"

	// register image format decoders
	_ "image/jpeg"
	_ "image/png"

	"visionserve/internal/explain"
	"visionserve/internal/lifecycle"
)

// explainRequest is the /api/explain request: multipart fields, or a JSON body with the same
// names (image_base64 instead of the "image" file part).
type explainRequest struct {
	Model        string   `json:"model"`
	ImageBase64  string   `json:"image_base64"`
	DetectionIdx int      `json:"detection_idx"` // 0-based; negative = 0
	TopChannels  int      `json:"top_channels"`  // Score-CAM channels; <= 0 = manifest default
	Alpha        *float32 `json:"alpha"`         // overlay opacity in [0,1]; absent/out of range = 0.5
	Format       string   `json:"format"`        // "png" (default) | "numpy"
	Class        string   `json:"class"`
}

// explainFiles: /api/explain reads only the image part (see requestFiles).
var explainFiles = map[string]int64{"image": maxImageBytes + 1}

// POST /api/explain
//
// Multipart form fields (or the same names as a JSON body, with image_base64):
//
//	model          string  required   model name (must be loaded and have an explain manifest block)
//	image          file    required   JPEG/PNG image
//	detection_idx  int     optional   0-based position in /api/predict's detections for this image (default 0)
//	top_channels   int     optional   Score-CAM: max channels to sample (0 = manifest default)
//	alpha          float   optional   overlay opacity [0,1] (default 0.5)
//	format         string  optional   "png" (default) or "numpy" (raw float32 bytes)
//	class          string  optional   explain the first detection of this class instead of detection_idx
//
// The explained detection is one of those /api/predict returns for the same image and model
// (without a prompt); an index past the end, or a class not detected, is a 400.
//
// Responses:
//   - format=png:   Content-Type: image/png — heatmap overlaid on the input image
//   - format=numpy: Content-Type: application/octet-stream — raw little-endian float32
//     with headers X-Heatmap-Shape (H,W) and X-Heatmap-Dtype (float32).
//
// Both carry X-Explain-Detection (the explained detection as JSON: bbox, class, conf) and, for
// attention, X-Explain-Query (the object query whose attention is shown).
func (s *Server) handleExplain(w http.ResponseWriter, r *http.Request) {
	var q explainRequest
	data, err := decodeFields(w, r, &q, explainFiles, s.admitter(r))
	if err != nil {
		writeError(w, err)
		return
	}
	defer data.Close()
	switch {
	case q.Model == "":
		err = badRequest(fmt.Errorf(`"model" field is required`))
	case !data.multipart && q.ImageBase64 == "":
		err = badRequest(fmt.Errorf(`"image_base64" is required`))
	case data.file("image") == nil && q.ImageBase64 == "":
		err = badRequest(fmt.Errorf(`"image" file is required: %w`, http.ErrMissingFile))
	}
	if err == nil {
		err = data.admit(q.Model)
	}
	if err != nil {
		writeError(w, err)
		return
	}

	img, err := decodeUpload(data, q.ImageBase64)
	if err != nil {
		writeError(w, err)
		return
	}

	req := lifecycle.ExplainRequest{
		Class:        q.Class,
		DetectionIdx: max(q.DetectionIdx, 0),
		TopChannels:  max(q.TopChannels, 0), // 0 = use manifest default
		Alpha:        0.5,
	}
	if q.Alpha != nil && *q.Alpha >= 0 && *q.Alpha <= 1 {
		req.Alpha = *q.Alpha
	}
	if r.Context().Err() != nil {
		writeError(w, errClientGone)
		return
	}
	result, err := s.mgr.Explain(r.Context(), q.Model, img, req)
	if err != nil {
		writeError(w, orClientGone(r.Context(), err))
		return
	}

	// Which detection the heatmap explains, exactly as /api/predict reports it, so a client can
	// label the heatmap without a second request (and see a class/index mix-up at once).
	if det, err := json.Marshal(result.Detection); err == nil {
		w.Header().Set("X-Explain-Detection", string(det))
	}
	if result.Query >= 0 {
		w.Header().Set("X-Explain-Query", strconv.Itoa(result.Query))
	}

	switch q.Format {
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

	default: // "" or "png"
		var buf bytes.Buffer
		if err := explain.RenderPNG(&buf, img, result.Heatmap, result.Width, result.Height, req.Alpha); err != nil {
			writeError(w, fmt.Errorf("failed to render PNG: %w", err))
			return
		}
		w.Header().Set("Content-Type", "image/png")
		w.Header().Set("Content-Length", strconv.Itoa(buf.Len()))
		_, _ = w.Write(buf.Bytes())
	}
}
