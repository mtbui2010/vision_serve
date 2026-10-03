package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/lifecycle"
	"visionserve/internal/models"
	"visionserve/internal/templates"
	"visionserve/pkg/api"
)

// fakeRuntime stands in for lifecycle.Manager: it records the order of calls and what each
// inference call received, and returns canned answers.
type fakeRuntime struct {
	mu     sync.Mutex
	events []string

	admitErr  error
	admitWait chan struct{} // when set, Admit blocks until it is closed
	admitted  chan struct{} // when set, closed when Admit is entered
	runErr    error         // returned by every inference call
	result    api.Result

	prompt  models.Prompt
	img     image.Image
	explain lifecycle.ExplainRequest
	tensor  engine.Tensor
}

func (f *fakeRuntime) event(e string) {
	f.mu.Lock()
	f.events = append(f.events, e)
	f.mu.Unlock()
}

func (f *fakeRuntime) Events() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.events...)
}

// Admit follows lifecycle.Manager.Admit's contract: a context already done on entry is refused
// with its error. admitWait stands in for time spent getting the slot.
func (f *fakeRuntime) Admit(ctx context.Context, name string) (func(), error) {
	f.event("admit:" + name)
	if err := ctx.Err(); err != nil {
		return func() {}, fmt.Errorf("fake: not admitted: %w", err)
	}
	if f.admitted != nil {
		close(f.admitted)
	}
	if f.admitWait != nil {
		<-f.admitWait
	}
	if f.admitErr != nil {
		return nil, f.admitErr
	}
	return func() { f.event("release") }, nil
}

func (f *fakeRuntime) Load(name string) error   { f.event("load:" + name); return f.runErr }
func (f *fakeRuntime) Unload(name string) error { f.event("unload:" + name); return f.runErr }
func (f *fakeRuntime) IsLoaded(string) bool     { return false }
func (f *fakeRuntime) Close()                   {}

func (f *fakeRuntime) PredictPrompt(name string, img image.Image, p models.Prompt) (api.Result, error) {
	f.event("predict:" + name)
	f.mu.Lock()
	f.prompt, f.img = p, img
	f.mu.Unlock()
	if f.runErr != nil {
		return api.Result{}, f.runErr
	}
	return f.result, nil
}

func (f *fakeRuntime) InferTensor(name string, in engine.Tensor) (api.Result, error) {
	f.event("tensor:" + name)
	f.tensor = in
	if f.runErr != nil {
		return api.Result{}, f.runErr
	}
	return f.result, nil
}

func (f *fakeRuntime) Explain(name string, img image.Image, req lifecycle.ExplainRequest) (lifecycle.ExplainResult, error) {
	f.event("explain:" + name)
	f.explain, f.img = req, img
	if f.runErr != nil {
		return lifecycle.ExplainResult{}, f.runErr
	}
	return lifecycle.ExplainResult{Heatmap: []float32{0, 0.5, 1, 0.25}, Width: 2, Height: 2}, nil
}

func (f *fakeRuntime) Preprocess(name string, img image.Image, p models.Prompt) (lifecycle.PreprocessResult, error) {
	f.event("preprocess:" + name)
	f.prompt, f.img = p, img
	if f.runErr != nil {
		return lifecycle.PreprocessResult{}, f.runErr
	}
	return lifecycle.PreprocessResult{Inputs: []lifecycle.NamedTensor{
		{Role: "model", Name: "pixel_values", Tensor: engine.F32([]float32{1, 2}, 1, 2)},
	}}, nil
}

// newTestServer returns a server over f, and its full handler (routes, body limits, logging).
func newTestServer(f *fakeRuntime) (*Server, http.Handler) {
	s := newServer(nil, f, templates.New(), "")
	return s, s.http.Handler
}

// do runs one request through h and returns the recorder.
func do(h http.Handler, req *http.Request) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

// part is a multipart file part.
type part struct {
	field, filename string
	data            []byte
}

// multipartRequest builds a multipart POST with text fields and file parts.
func multipartRequest(t *testing.T, url string, fields map[string]string, files ...part) *http.Request {
	t.Helper()
	var b bytes.Buffer
	mw := multipart.NewWriter(&b)
	for k, v := range fields {
		if err := mw.WriteField(k, v); err != nil {
			t.Fatal(err)
		}
	}
	for _, p := range files {
		fw, err := mw.CreateFormFile(p.field, p.filename)
		if err != nil {
			t.Fatal(err)
		}
		fw.Write(p.data) //nolint:errcheck
	}
	if err := mw.Close(); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, url, &b)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	return req
}

// formPart is one part of an orderedMultipart body: a text field when filename is "".
type formPart struct {
	name, filename string
	data           []byte
}

// orderedMultipart renders parts in the order given (multipartRequest writes fields first) and
// returns the body and its Content-Type.
func orderedMultipart(t *testing.T, parts ...formPart) ([]byte, string) {
	t.Helper()
	var b bytes.Buffer
	mw := multipart.NewWriter(&b)
	for _, p := range parts {
		var w io.Writer
		var err error
		if p.filename == "" {
			w, err = mw.CreateFormField(p.name)
		} else {
			w, err = mw.CreateFormFile(p.name, p.filename)
		}
		if err != nil {
			t.Fatal(err)
		}
		w.Write(p.data) //nolint:errcheck
	}
	if err := mw.Close(); err != nil {
		t.Fatal(err)
	}
	return b.Bytes(), mw.FormDataContentType()
}

// jsonRequest builds a JSON POST.
func jsonRequest(url string, body []byte) *http.Request {
	req := httptest.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	return req
}

func b64(b []byte) string { return base64.StdEncoding.EncodeToString(b) }
