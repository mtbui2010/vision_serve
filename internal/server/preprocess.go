package server

import (
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"image"
	"math"
	"net/http"

	"visionserve/internal/lifecycle"
	"visionserve/internal/models"
)

// preprocessResponse is the body of POST /api/preprocess.
type preprocessResponse struct {
	Model  string            `json:"model"`
	Inputs []preprocessInput `json:"inputs"`
	// Meta maps model-input coordinates back to the original image:
	// input_x = orig_x * scale_x + pad_x. Null when the model's preprocessing has none.
	Meta *preprocessMeta `json:"meta"`
}

type preprocessInput struct {
	Role  string  `json:"role"`  // manifest role of the session ("model" for single-session models)
	Name  string  `json:"name"`  // ONNX input name
	Shape []int64 `json:"shape"` //
	Dtype string  `json:"dtype"` // "float32" | "int64"
	// Data is the raw little-endian buffer, base64-encoded: numpy.frombuffer(b64decode(data),
	// dtype).reshape(shape) reconstructs it bit-for-bit.
	Data string `json:"data"`
}

type preprocessMeta struct {
	OrigWidth  int     `json:"orig_width"`
	OrigHeight int     `json:"orig_height"`
	ScaleX     float64 `json:"scale_x"`
	ScaleY     float64 `json:"scale_y"`
	PadX       int     `json:"pad_x"`
	PadY       int     `json:"pad_y"`
}

// handlePreprocess — POST /api/preprocess (multipart: model, image?, prompt?, box?, point?)
//
// Returns exactly what the model would feed its first ONNX session for this request, without
// running inference: the resized/normalised pixels, and for text models the token ids. It is a
// DEBUG endpoint for train/serve parity — compare it with the tensor your training code builds
// for the same image. The image is optional for text-only models (siglip-text, clip-text).
func (s *Server) handlePreprocess(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(maxImageBytes); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("failed to parse multipart form: %w", err))
		return
	}
	name := r.FormValue("model")
	if name == "" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("missing 'model' field"))
		return
	}
	var img image.Image
	if file, _, err := r.FormFile("image"); err == nil {
		defer file.Close()
		if img, err = decodeImage(file); err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
	}
	prompt, err := models.ParsePrompt(r.FormValue("prompt"), r.FormValue("box"), r.FormValue("point"))
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	res, err := s.mgr.Preprocess(name, img, prompt)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusOK, encodePreprocess(name, res))
}

func encodePreprocess(name string, res lifecycle.PreprocessResult) preprocessResponse {
	out := preprocessResponse{Model: name, Inputs: make([]preprocessInput, 0, len(res.Inputs))}
	for _, in := range res.Inputs {
		t := in.Tensor
		pi := preprocessInput{Role: in.Role, Name: in.Name, Shape: t.Shape}
		if t.Dtype == "i64" {
			buf := make([]byte, 8*len(t.DataI64))
			for i, v := range t.DataI64 {
				binary.LittleEndian.PutUint64(buf[8*i:], uint64(v))
			}
			pi.Dtype, pi.Data = "int64", base64.StdEncoding.EncodeToString(buf)
		} else {
			buf := make([]byte, 4*len(t.Data))
			for i, v := range t.Data {
				binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(v))
			}
			pi.Dtype, pi.Data = "float32", base64.StdEncoding.EncodeToString(buf)
		}
		out.Inputs = append(out.Inputs, pi)
	}
	if m := res.Meta; m != nil {
		out.Meta = &preprocessMeta{OrigWidth: m.OrigWidth, OrigHeight: m.OrigHeight,
			ScaleX: m.ScaleX, ScaleY: m.ScaleY, PadX: m.PadX, PadY: m.PadY}
	}
	return out
}
