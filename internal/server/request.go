package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"io"
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
// uploads belong in multipart, where the image is a file part read only after admission.
//
// A JSON body is not streamed the way a multipart one is: encoding/json fills the struct from the
// whole value, a streaming tokenizer would still hold the image_base64 string whole (it is ONE
// token), and nothing orders "model" before it. The cap already bounds what a queued JSON request
// holds, and the base64 text is dropped as soon as it is decoded (decodeImage).
const maxJSONBody = maxImageBytes

// requestFiles are the file parts /api/predict and /api/preprocess read, each with the most bytes
// of it that matter: decodeImage reads at most maxImageBytes+1 (one more is how it knows the image
// is over the limit), depthBytes at most maxTensorBytes.
var requestFiles = map[string]int64{"image": maxImageBytes + 1, "depth": maxTensorBytes}

// Request is one /api/predict or /api/preprocess call, decoded from either content type.
//
// Its option list is api.PredictJSONRequest. A multipart field and the JSON key of the same name
// are the same option by construction (formInto fills the struct by json tag), and ToPrompt is
// the one place an option becomes a model input — so a new option is one field in
// api.PredictJSONRequest plus one line in ToPrompt, for both content types and the CLI.
//
// decodeRequest reads only the envelope: the image and the depth map stay encoded (a form file
// part, or a base64 string) until the handler holds an admission slot, because decoding is where
// a request's memory grows (40 MP of pixels is 160 MB). A multipart request may already hold its
// slot when decodeRequest returns (see readMultipart); the handler calls admit either way.
type Request struct {
	api.PredictJSONRequest

	data *formData // nil for a Request built in code (`visionserve run`)
}

// decodeRequest reads a predict/preprocess request envelope from a JSON body or a multipart form.
// The encoding may also come from the query string (?encoding=base64). admit takes an admission
// slot (Server.admit for this request). On success the caller must defer q.Close.
func decodeRequest(w http.ResponseWriter, r *http.Request, admit func(model string) (func(), error)) (*Request, error) {
	q := &Request{}
	data, err := decodeFields(w, r, &q.PredictJSONRequest, requestFiles, admit)
	if err != nil {
		return nil, err
	}
	q.data = data
	if q.Encoding == "" {
		q.Encoding = r.URL.Query().Get("encoding") // multipart already saw it (query first)
	}
	enc, err := api.ParseEncoding(q.Encoding)
	if err != nil {
		q.Close()
		return nil, badRequest(err)
	}
	q.Encoding = enc
	return q, nil
}

// admit takes the request's admission slot on its model; a no-op when the multipart body was
// already admitted while it was read. Call it after validate.
func (q *Request) admit() error { return q.data.admit(q.Model) }

// Close releases the admission slot and the uploaded parts' temp files.
func (q *Request) Close() { q.data.Close() }

// isMultipart reports whether the request came as a multipart form (vs a JSON body).
func (q *Request) isMultipart() bool { return q.data != nil && q.data.multipart }

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
	return q.data.file("image") != nil || q.ImageBase64 != ""
}

// decodeImage decodes the request's image (call it only after admission). The base64 copy is
// dropped as soon as it is decoded, so it is not held alongside the pixels.
func (q *Request) decodeImage() (image.Image, error) {
	b64 := q.ImageBase64
	q.ImageBase64 = ""
	return decodeUpload(q.data, b64)
}

// decodeUpload decodes a request's image: the multipart "image" file part when there is one,
// else the base64 text (image_base64). Shared by every endpoint that takes an image.
func decodeUpload(data *formData, b64 string) (image.Image, error) {
	if u := data.file("image"); u != nil {
		f, err := u.open()
		if err != nil {
			return nil, badRequest(fmt.Errorf("missing 'image' file: %w", err))
		}
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
	if u := q.data.file("depth"); u != nil {
		f, err := u.open()
		if err != nil {
			return nil, badRequest(fmt.Errorf("reading depth: %w", err))
		}
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
// multipart form, whichever the request is, and returns the request's formData: the kept file
// parts (keep, see readMultipart) and its admission slot, taken through admit. Both paths read
// the body under a size cap; an oversized body is a 413.
//
// A JSON request is read whole and is not admitted here. A multipart request is admitted while it
// is read, before its first kept file part, when the model is known by then; otherwise (no kept
// file part, or one before the model field) it is not admitted yet. Either way the caller
// validates the fields, then calls admit on the formData (a no-op if it already holds the slot),
// and must defer Close on success. On error nothing is held.
func decodeFields(w http.ResponseWriter, r *http.Request, dst any, keep map[string]int64,
	admit func(model string) (func(), error)) (*formData, error) {
	d := &formData{files: map[string]*upload{}, admitFn: admit}
	if isJSONRequest(r) {
		body := http.MaxBytesReader(w, r.Body, maxJSONBody)
		if err := json.NewDecoder(body).Decode(dst); err != nil {
			return nil, badRequest(fmt.Errorf("invalid JSON body: %w", err))
		}
		drainBody(body)
		return d, nil
	}
	d.multipart = true
	value, err := readMultipart(r, d, keep)
	if err != nil {
		if d.refused {
			closeUnreadBody(w, r) // refused before a file part: the rest of the body is unread
		}
		d.Close()
		return nil, err
	}
	formInto(dst, value)
	return d, nil
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
