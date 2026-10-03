package server

import (
	"encoding/base64"
	"encoding/binary"
	"image"
	"math"
	"net/http"

	"visionserve/internal/lifecycle"
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

// handlePreprocess — POST /api/preprocess (the /api/predict request: multipart or JSON, with
// model, image?, prompt?, box?, point? and the other predict options)
//
// Returns exactly what the model would feed its first ONNX session for this request, without
// running inference: the resized/normalised pixels, and for text models the token ids. It is a
// DEBUG endpoint for train/serve parity — compare it with the tensor your training code builds
// for the same image. The image is optional for text-only models (siglip-text, clip-text).
// With a roi the tensors are those of the crop the model would see, and meta maps to the crop.
func (s *Server) handlePreprocess(w http.ResponseWriter, r *http.Request) {
	res, name, err := s.preprocess(w, r)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, encodePreprocess(name, res))
}

func (s *Server) preprocess(w http.ResponseWriter, r *http.Request) (lifecycle.PreprocessResult, string, error) {
	q, err := decodeRequest(w, r, s.admitter(r))
	if err != nil {
		return lifecycle.PreprocessResult{}, "", err
	}
	defer q.Close()
	if err := q.validate(false); err != nil {
		return lifecycle.PreprocessResult{}, "", err
	}
	if err := q.admit(); err != nil {
		return lifecycle.PreprocessResult{}, "", err
	}

	var img image.Image
	var imgW, imgH int
	if q.hasImage() {
		if img, err = q.decodeImage(); err != nil {
			return lifecycle.PreprocessResult{}, "", err
		}
		imgW, imgH = img.Bounds().Dx(), img.Bounds().Dy()
	}
	prompt, err := q.ToPrompt(imgW, imgH)
	if err != nil {
		return lifecycle.PreprocessResult{}, "", err
	}
	if img != nil {
		img, _, _ = cropROI(img, &prompt)
	}
	if r.Context().Err() != nil {
		return lifecycle.PreprocessResult{}, "", errClientGone
	}
	res, err := s.mgr.Preprocess(q.Model, img, prompt)
	return res, q.Model, err
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
