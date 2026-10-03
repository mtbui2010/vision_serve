package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"mime/multipart"
	"net/http"
	"reflect"
	"strconv"
	"strings"

	"visionserve/internal/models"
	roipkg "visionserve/internal/roi"
	"visionserve/pkg/api"
)

// maxJSONBody caps a JSON request envelope (image_base64 and depth_base64 included). It is read
// whole BEFORE admission control (the model name is inside it), so it must stay bounded; large
// uploads belong in multipart, where the image is a file part.
const maxJSONBody = maxImageBytes

// Request is one /api/predict or /api/preprocess call, decoded from either content type.
//
// Its option list is api.PredictJSONRequest. A multipart field and the JSON key of the same name
// are the same option by construction (formInto fills the struct by json tag), and ToPrompt is
// the one place an option becomes a model input — so a new option is one field in
// api.PredictJSONRequest plus one line in ToPrompt, for both content types and the CLI.
//
// decodeRequest reads only the envelope: the image and the depth map stay encoded (a form file
// part, or a base64 string) until the handler holds an admission slot, because decoding is where
// a request's memory grows (40 MP of pixels is 160 MB).
type Request struct {
	api.PredictJSONRequest

	form *multipart.Form // multipart only: holds the "image" and "depth" file parts
}

// decodeRequest reads a predict/preprocess request envelope from a JSON body or a multipart form.
// The encoding may also come from the query string (?encoding=base64).
func decodeRequest(w http.ResponseWriter, r *http.Request) (*Request, error) {
	q := &Request{}
	form, err := decodeFields(w, r, &q.PredictJSONRequest)
	if err != nil {
		return nil, err
	}
	q.form = form
	if q.Encoding == "" {
		q.Encoding = r.URL.Query().Get("encoding") // multipart already saw it via FormValue
	}
	enc, err := api.ParseEncoding(q.Encoding)
	if err != nil {
		return nil, badRequest(err)
	}
	q.Encoding = enc
	return q, nil
}

// isMultipart reports whether the request came as a multipart form (vs a JSON body).
func (q *Request) isMultipart() bool { return q.form != nil }

// validate checks the fields every predict request needs, with the messages each content type
// always used.
func (q *Request) validate(needImage bool) error {
	switch {
	case !q.isMultipart() && needImage && (q.Model == "" || q.ImageBase64 == ""):
		return badRequest(fmt.Errorf("both 'model' and 'image_base64' are required"))
	case q.Model == "":
		return badRequest(fmt.Errorf("missing 'model' field"))
	case needImage && !q.hasImage():
		return badRequest(fmt.Errorf("missing 'image' file: %w", http.ErrMissingFile))
	}
	return nil
}

// hasImage reports whether the request carries an image (a multipart "image" file part, or
// image_base64 in either content type).
func (q *Request) hasImage() bool {
	return filePart(q.form, "image") != nil || q.ImageBase64 != ""
}

// decodeImage decodes the request's image (call it only after admission). The base64 copy is
// dropped as soon as it is decoded, so it is not held alongside the pixels.
func (q *Request) decodeImage() (image.Image, error) {
	b64 := q.ImageBase64
	q.ImageBase64 = ""
	return decodeUpload(q.form, b64)
}

// filePart returns the first file part called name, or nil (also for a JSON request: form nil).
func filePart(form *multipart.Form, name string) *multipart.FileHeader {
	if form == nil || len(form.File[name]) == 0 {
		return nil
	}
	return form.File[name][0]
}

// decodeUpload decodes a request's image: the multipart "image" file part when there is one,
// else the base64 text (image_base64). Shared by every endpoint that takes an image.
func decodeUpload(form *multipart.Form, b64 string) (image.Image, error) {
	if fh := filePart(form, "image"); fh != nil {
		f, err := fh.Open()
		if err != nil {
			return nil, badRequest(fmt.Errorf("missing 'image' file: %w", err))
		}
		defer f.Close()
		return decodeImage(f)
	}
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return nil, badRequest(fmt.Errorf("invalid image_base64: %w", err))
	}
	return decodeImage(bytes.NewReader(raw))
}

// depthBytes returns the raw depth map: the multipart "depth" file part, or depth_base64. nil
// when the request has none.
func (q *Request) depthBytes() ([]byte, error) {
	if fh := filePart(q.form, "depth"); fh != nil {
		f, err := fh.Open()
		if err != nil {
			return nil, badRequest(fmt.Errorf("reading depth: %w", err))
		}
		defer f.Close()
		raw, err := io.ReadAll(io.LimitReader(f, maxTensorBytes))
		if err != nil {
			return nil, badRequest(fmt.Errorf("reading depth: %w", err))
		}
		return raw, nil
	}
	if q.DepthBase64 == "" {
		return nil, nil
	}
	raw, err := base64.StdEncoding.DecodeString(q.DepthBase64)
	if err != nil {
		return nil, badRequest(fmt.Errorf("invalid depth_base64: %w", err))
	}
	q.DepthBase64 = ""
	return raw, nil
}

// ToPrompt builds the models.Prompt for this request: the ONE place a request option becomes a
// model input, for both content types and `visionserve run`. imgW/imgH (the decoded image's
// size) are what an uploaded depth map is resized to.
func (q *Request) ToPrompt(imgW, imgH int) (models.Prompt, error) {
	p, err := models.ParsePrompt(q.Prompt, q.Box, q.Point)
	if err != nil {
		return models.Prompt{}, badRequest(err)
	}
	p.MinSize, p.MaxSize = q.MinSize, q.MaxSize
	p.GripperMin, p.GripperMax = q.GripperMin, q.GripperMax
	p.BoxThresh, p.TextThresh = q.BoxThreshold, q.TextThreshold
	p.BgMaxArea, p.FgMinArea = q.BgMaxArea, q.FgMinArea
	p.GridSize = clampGrid(q.GridSize)
	p.Method = q.Method
	p.ClaimThresh = q.ClaimThreshold
	p.CropTemp = q.CropTemp
	p.ROI = roipkg.Parse(q.ROI)
	p.Dilate = q.Dilate
	p.TemplateName = q.TemplateName

	raw, err := q.depthBytes()
	if err != nil {
		return models.Prompt{}, err
	}
	if raw != nil {
		p.Depth, p.DepthW, p.DepthH = parseDepth(raw, q.DepthDtype, q.DepthWidth, q.DepthHeight, imgW, imgH)
		if p.Depth == nil {
			return models.Prompt{}, badRequest(errBadDepth)
		}
	}
	return p, nil
}

// maxGridSize bounds the automatic-mask grid: N×N decoder calls (and goroutines/allocations),
// so an unchecked grid_size=1000 asked for a million decoder runs in one request.
const maxGridSize = 64

func clampGrid(n int) int {
	if n > maxGridSize {
		return maxGridSize
	}
	return n
}

// isJSONRequest reports whether the body is JSON (anything else is parsed as a multipart form).
func isJSONRequest(r *http.Request) bool {
	return strings.HasPrefix(strings.ToLower(r.Header.Get("Content-Type")), "application/json")
}

// decodeFields fills dst — a pointer to a struct with json tags — from a JSON body or a
// multipart form, whichever the request is, and returns the multipart form (nil for JSON) for
// its file parts. Both paths read the whole body under a size cap; an oversized body is a 413.
func decodeFields(w http.ResponseWriter, r *http.Request, dst any) (*multipart.Form, error) {
	if isJSONRequest(r) {
		body := http.MaxBytesReader(w, r.Body, maxJSONBody)
		if err := json.NewDecoder(body).Decode(dst); err != nil {
			return nil, badRequest(fmt.Errorf("invalid JSON body: %w", err))
		}
		drainBody(body)
		return nil, nil
	}
	if err := r.ParseMultipartForm(maxImageBytes); err != nil {
		return nil, badRequest(fmt.Errorf("failed to parse multipart form: %w", err))
	}
	formInto(dst, r.FormValue)
	drainBody(r.Body)
	return r.MultipartForm, nil
}

// drainBody reads what is left of a body (a trailing newline after the JSON value, the epilogue
// after the last multipart boundary). net/http only notices that a client disconnected — and
// cancels r.Context() — once the handler has read the body to EOF, so without this a request
// whose client is gone could still run its inference.
func drainBody(body io.Reader) {
	_, _ = io.Copy(io.Discard, io.LimitReader(body, 64<<10))
}

// formInto sets the fields of the struct dst points to from form values, matched by each field's
// json name — the multipart twin of json.Unmarshal. Embedded structs are walked. Supported field
// kinds: string, int, float32/64, and pointers to them (set only when the value is present).
// A number that does not parse leaves the field at its zero value, which every option treats as
// "use the default" — the way these handlers always treated a malformed optional number.
func formInto(dst any, value func(name string) string) {
	formFields(reflect.ValueOf(dst).Elem(), value)
}

func formFields(v reflect.Value, value func(string) string) {
	t := v.Type()
	for i := 0; i < t.NumField(); i++ {
		f, fv := t.Field(i), v.Field(i)
		if f.Anonymous && f.Type.Kind() == reflect.Struct {
			formFields(fv, value)
			continue
		}
		name, _, _ := strings.Cut(f.Tag.Get("json"), ",")
		if !f.IsExported() || name == "" || name == "-" {
			continue
		}
		if s := value(name); s != "" {
			setFromString(fv, s)
		}
	}
}

// setFromString parses s into fv; it reports whether fv was set.
func setFromString(fv reflect.Value, s string) bool {
	switch fv.Kind() {
	case reflect.Pointer:
		p := reflect.New(fv.Type().Elem())
		if !setFromString(p.Elem(), s) {
			return false
		}
		fv.Set(p)
	case reflect.String:
		fv.SetString(s)
	case reflect.Int:
		n, err := strconv.Atoi(s)
		if err != nil {
			return false
		}
		fv.SetInt(int64(n))
	case reflect.Float32, reflect.Float64:
		x, err := strconv.ParseFloat(s, fv.Type().Bits())
		if err != nil {
			return false
		}
		fv.SetFloat(x)
	default:
		return false // a kind formSupported rejects in tests, so it cannot ship unnoticed
	}
	return true
}

// formSupported reports whether formInto can fill a field of type t (used by tests to make sure
// every option of api.PredictJSONRequest is reachable from a multipart form).
func formSupported(t reflect.Type) bool {
	switch t.Kind() {
	case reflect.Pointer:
		return formSupported(t.Elem())
	case reflect.String, reflect.Int, reflect.Float32, reflect.Float64:
		return true
	}
	return false
}
