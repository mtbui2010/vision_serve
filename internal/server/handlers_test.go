package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"visionserve/internal/lifecycle"
	"visionserve/internal/models"
	"visionserve/internal/templates"
	"visionserve/pkg/api"
)

// ---------------------------------------------------------------------------------------------
// Status mapping
// ---------------------------------------------------------------------------------------------

func TestStatusOf(t *testing.T) {
	maxBytes := &http.MaxBytesError{Limit: 10}
	cases := []struct {
		name string
		err  error
		want int
	}{
		{"not found (wrapped by the runtime)", fmt.Errorf("lifecycle: model %q: %w", "x", lifecycle.ErrModelNotFound), 404},
		{"invalid request (wrapped)", fmt.Errorf("lifecycle: template: %w", lifecycle.ErrInvalidRequest), 400},
		{"overloaded (wrapped)", fmt.Errorf("lifecycle: queue full: %w", lifecycle.ErrOverloaded), 503},
		{"request parse error", badRequest(errors.New("invalid box")), 400},
		{"oversized body", maxBytes, 413},
		{"form too large to parse", badRequest(fmt.Errorf("failed to parse multipart form: %w", maxBytes)), 413},
		{"image over the byte limit", tooLargeError{"image is larger than 32 MiB"}, 413},
		{"model refused its prompt (wrapped by the pipeline)",
			fmt.Errorf("rfdetr-gdino: %w", models.BadPrompt(errors.New("a text prompt is required"))), 400},
		{"client gone", errClientGone, 499},
		{"anything else", errors.New("onnx: run failed"), 500},
	}
	for _, c := range cases {
		if got := statusOf(c.err); got != c.want {
			t.Errorf("%s: status %d, want %d", c.name, got, c.want)
		}
	}
	// A request error is ErrInvalidRequest for errors.Is but keeps its message.
	if e := badRequest(errors.New("invalid box")); !errors.Is(e, lifecycle.ErrInvalidRequest) || e.Error() != "invalid box" {
		t.Errorf("badRequest: %v", e)
	}
}

// Every endpoint answers a failure through writeError: the right status AND the JSON shape.
func TestHandlersMapErrorsToStatus(t *testing.T) {
	png := pngBytes(t, 8, 8)
	notFound := fmt.Errorf("lifecycle: model %q not found in registry: %w", "nope", lifecycle.ErrModelNotFound)
	cases := []struct {
		name     string
		f        *fakeRuntime
		req      func() *http.Request
		want     int
		wantBody string
	}{
		{"predict: unknown model", &fakeRuntime{runErr: notFound},
			func() *http.Request {
				return multipartRequest(t, "/api/predict", map[string]string{"model": "nope"}, part{"image", "i.png", png})
			}, 404, "not found"},
		{"predict: bad prompt (runtime)", &fakeRuntime{runErr: fmt.Errorf("gdino: %w", lifecycle.ErrInvalidRequest)},
			func() *http.Request {
				return multipartRequest(t, "/api/predict", map[string]string{"model": "m"}, part{"image", "i.png", png})
			}, 400, "invalid request"},
		{"predict: bad box (parse)", &fakeRuntime{},
			func() *http.Request {
				return multipartRequest(t, "/api/predict", map[string]string{"model": "m", "box": "1,2"}, part{"image", "i.png", png})
			}, 400, "invalid box"},
		{"predict: bad depth", &fakeRuntime{},
			func() *http.Request {
				return multipartRequest(t, "/api/predict", map[string]string{"model": "m"}, part{"image", "i.png", png}, part{"depth", "d", []byte{1, 2, 3}})
			}, 400, "invalid depth map"},
		{"predict: undecodable image", &fakeRuntime{},
			func() *http.Request {
				return multipartRequest(t, "/api/predict", map[string]string{"model": "m"}, part{"image", "i.png", []byte("not an image")})
			}, 400, "failed to decode image"},
		{"predict: runtime failure", &fakeRuntime{runErr: errors.New("onnx: boom")},
			func() *http.Request {
				return multipartRequest(t, "/api/predict", map[string]string{"model": "m"}, part{"image", "i.png", png})
			}, 500, "onnx: boom"},
		{"predict: overloaded", &fakeRuntime{admitErr: fmt.Errorf("lifecycle: %w", lifecycle.ErrOverloaded)},
			func() *http.Request {
				return multipartRequest(t, "/api/predict", map[string]string{"model": "m"}, part{"image", "i.png", png})
			}, 503, "overloaded"},
		{"predict: JSON over the envelope cap", &fakeRuntime{},
			func() *http.Request {
				return jsonRequest("/api/predict", append([]byte(`{"model":"m","image_base64":"`), bytes.Repeat([]byte("A"), maxJSONBody)...))
			}, 413, "too large"},
		{"load: unknown model", &fakeRuntime{runErr: notFound},
			func() *http.Request { return jsonRequest("/api/load", []byte(`{"model":"nope"}`)) }, 404, "not found"},
		{"load: no model", &fakeRuntime{},
			func() *http.Request { return jsonRequest("/api/load", []byte(`{}`)) }, 400, "missing 'model' field"},
		{"unload: bad JSON", &fakeRuntime{},
			func() *http.Request { return jsonRequest("/api/unload", []byte(`{`)) }, 400, "invalid JSON body"},
		{"explain: unknown model", &fakeRuntime{runErr: notFound},
			func() *http.Request {
				return multipartRequest(t, "/api/explain", map[string]string{"model": "nope"}, part{"image", "i.png", png})
			}, 404, "not found"},
		{"explain: no image", &fakeRuntime{},
			func() *http.Request { return multipartRequest(t, "/api/explain", map[string]string{"model": "m"}) }, 400, `"image" file is required`},
		{"preprocess: unknown model", &fakeRuntime{runErr: notFound},
			func() *http.Request {
				return multipartRequest(t, "/api/preprocess", map[string]string{"model": "nope"})
			}, 404, "not found"},
		{"infer_tensor: overloaded", &fakeRuntime{admitErr: lifecycle.ErrOverloaded},
			func() *http.Request {
				return httptest.NewRequest("POST", "/api/infer_tensor?model=m&shape=1,1", bytes.NewReader([]byte{0, 0, 0, 0}))
			}, 503, "overloaded"},
		{"infer_tensor: wrong length", &fakeRuntime{},
			func() *http.Request {
				return httptest.NewRequest("POST", "/api/infer_tensor?model=m&shape=1,2", bytes.NewReader([]byte{0, 0, 0, 0}))
			}, 400, "implies 2"},
		{"templates: no name", &fakeRuntime{},
			func() *http.Request { return multipartRequest(t, "/api/templates", nil, part{"images", "i.png", png}) }, 400, `"name" is required`},
		{"templates: no images", &fakeRuntime{},
			func() *http.Request { return multipartRequest(t, "/api/templates", map[string]string{"name": "cup"}) }, 400, `at least one "images" file`},
		{"templates: bad image", &fakeRuntime{},
			func() *http.Request {
				return multipartRequest(t, "/api/templates", map[string]string{"name": "cup"}, part{"images", "i.png", []byte("junk")})
			}, 400, "failed to decode image"},
		{"templates: not a form", &fakeRuntime{},
			func() *http.Request { return jsonRequest("/api/templates", []byte(`{}`)) }, 400, "failed to parse form"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, h := newTestServer(c.f)
			rec := do(h, c.req())
			if rec.Code != c.want {
				t.Fatalf("status %d, want %d: %s", rec.Code, c.want, rec.Body)
			}
			if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
				t.Fatalf("Content-Type %q: errors must be JSON", ct)
			}
			var e api.ErrorResponse
			if err := json.Unmarshal(rec.Body.Bytes(), &e); err != nil || !strings.Contains(e.Error, c.wantBody) {
				t.Fatalf(`body %q, want {"error": "...%s..."}`, rec.Body, c.wantBody)
			}
			if c.want == 503 && rec.Header().Get("Retry-After") == "" {
				t.Fatal("503 without Retry-After")
			}
		})
	}
}

// A multipart body over the route cap is a 413, not a 400 "failed to parse".
func TestOversizedMultipartIs413(t *testing.T) {
	s, _ := newTestServer(&fakeRuntime{})
	h := limitBody(1<<10, s.handlePredict)
	req := multipartRequest(t, "/api/predict", map[string]string{"model": "m"}, part{"image", "i.png", make([]byte, 4<<10)})
	rec := httptest.NewRecorder()
	h(rec, req)
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
}

// ---------------------------------------------------------------------------------------------
// Admission control
// ---------------------------------------------------------------------------------------------

// Admit must run BEFORE the image is decoded: with the model's queue full, a request carrying an
// image that cannot even be decoded gets the 503 — had it been decoded first, it would be a 400.
func TestAdmitBeforeDecode(t *testing.T) {
	junk := []byte("this is not an image")
	overloaded := fmt.Errorf("lifecycle: rf-detr queue full: %w", lifecycle.ErrOverloaded)
	reqs := map[string]func() *http.Request{
		"predict multipart": func() *http.Request {
			return multipartRequest(t, "/api/predict", map[string]string{"model": "rf-detr"}, part{"image", "i.png", junk})
		},
		"predict JSON": func() *http.Request {
			return jsonRequest("/api/predict", []byte(`{"model":"rf-detr","image_base64":"!!not base64!!"}`))
		},
		"explain multipart": func() *http.Request {
			return multipartRequest(t, "/api/explain", map[string]string{"model": "rf-detr"}, part{"image", "i.png", junk})
		},
		"explain JSON": func() *http.Request {
			return jsonRequest("/api/explain", []byte(`{"model":"rf-detr","image_base64":"!!not base64!!"}`))
		},
		"preprocess multipart": func() *http.Request {
			return multipartRequest(t, "/api/preprocess", map[string]string{"model": "rf-detr"}, part{"image", "i.png", junk})
		},
		"preprocess JSON": func() *http.Request {
			return jsonRequest("/api/preprocess", []byte(`{"model":"rf-detr","image_base64":"!!not base64!!"}`))
		},
	}
	for name, mk := range reqs {
		t.Run(name, func(t *testing.T) {
			f := &fakeRuntime{admitErr: overloaded}
			_, h := newTestServer(f)
			rec := do(h, mk())
			if rec.Code != http.StatusServiceUnavailable {
				t.Fatalf("status %d (%s): the image was decoded before admission", rec.Code, rec.Body)
			}
			if ev := f.Events(); len(ev) != 1 || ev[0] != "admit:rf-detr" {
				t.Fatalf("events %v, want exactly [admit:rf-detr]", ev)
			}
		})
	}
}

// On success the slot is held for the whole inference and released after it.
func TestAdmitWrapsInference(t *testing.T) {
	png := pngBytes(t, 8, 8)
	for _, c := range []struct {
		req  func() *http.Request
		want []string
	}{
		{func() *http.Request {
			return multipartRequest(t, "/api/predict", map[string]string{"model": "m"}, part{"image", "i.png", png})
		}, []string{"admit:m", "predict:m", "release"}},
		{func() *http.Request {
			return jsonRequest("/api/explain", []byte(`{"model":"m","image_base64":"`+b64(png)+`","format":"numpy"}`))
		}, []string{"admit:m", "explain:m", "release"}},
		{func() *http.Request {
			return multipartRequest(t, "/api/preprocess", map[string]string{"model": "m", "prompt": "cat"})
		},
			[]string{"admit:m", "preprocess:m", "release"}},
		{func() *http.Request {
			return httptest.NewRequest("POST", "/api/infer_tensor?model=m&shape=1,1", bytes.NewReader([]byte{0, 0, 128, 63}))
		}, []string{"admit:m", "tensor:m", "release"}},
	} {
		f := &fakeRuntime{}
		_, h := newTestServer(f)
		req := c.req()
		if rec := do(h, req); rec.Code != http.StatusOK {
			t.Fatalf("%s: status %d %s", req.URL, rec.Code, rec.Body)
		}
		if got := strings.Join(f.Events(), " "); got != strings.Join(c.want, " ") {
			t.Errorf("%s: events %q, want %q", req.URL, got, strings.Join(c.want, " "))
		}
	}
}

// ---------------------------------------------------------------------------------------------
// Cancellation
// ---------------------------------------------------------------------------------------------

// A request whose client has already gone does not run inference.
func TestCanceledRequestSkipsInference(t *testing.T) {
	png := pngBytes(t, 8, 8)
	for _, mk := range []func() *http.Request{
		func() *http.Request {
			return multipartRequest(t, "/api/predict", map[string]string{"model": "m"}, part{"image", "i.png", png})
		},
		func() *http.Request {
			return multipartRequest(t, "/api/explain", map[string]string{"model": "m"}, part{"image", "i.png", png})
		},
		func() *http.Request { return multipartRequest(t, "/api/preprocess", map[string]string{"model": "m"}) },
		func() *http.Request {
			return httptest.NewRequest("POST", "/api/infer_tensor?model=m&shape=1", bytes.NewReader([]byte{0, 0, 0, 0}))
		},
	} {
		f := &fakeRuntime{}
		_, h := newTestServer(f)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		req := mk().WithContext(ctx)
		rec := do(h, req)
		if rec.Code != statusClientClosedRequest {
			t.Errorf("%s: status %d, want 499", req.URL, rec.Code)
		}
		if got := strings.Join(f.Events(), " "); got != "admit:m release" {
			t.Errorf("%s: events %q: inference ran for a client that was gone", req.URL, got)
		}
	}
}

// Predict checks the context after the (possibly slow) decode, right before inference.
func TestPredictChecksContextBeforeInference(t *testing.T) {
	f := &fakeRuntime{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Predict(ctx, f, "m", image.NewGray(image.Rect(0, 0, 8, 8)), models.Prompt{}); !errors.Is(err, errClientGone) {
		t.Fatalf("err %v, want errClientGone", err)
	}
	if len(f.Events()) != 0 {
		t.Fatalf("inference ran: %v", f.Events())
	}
}

// End to end over a real connection: the client disconnects while its request waits for an
// admission slot; once the slot frees up, the server must notice and not run the model.
func TestClientDisconnectWhileQueued(t *testing.T) {
	png := pngBytes(t, 8, 8)
	for name, mk := range map[string]func(url string) *http.Request{
		"multipart": func(url string) *http.Request {
			req := multipartRequest(t, url, map[string]string{"model": "m"}, part{"image", "i.png", png})
			req.RequestURI = ""
			return req
		},
		"chunked JSON": func(url string) *http.Request {
			body := `{"model":"m","image_base64":"` + b64(png) + `"}` + "\n"
			req, _ := http.NewRequest("POST", url, io.MultiReader(strings.NewReader(body))) // no length: chunked
			req.Header.Set("Content-Type", "application/json")
			return req
		},
	} {
		t.Run(name, func(t *testing.T) {
			f := &fakeRuntime{admitWait: make(chan struct{}), admitted: make(chan struct{})}
			s, _ := newTestServer(f)
			serverCtx := make(chan context.Context, 1)
			handlerDone := make(chan struct{})
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer close(handlerDone)
				serverCtx <- r.Context()
				s.routes().ServeHTTP(w, r)
			}))
			defer ts.Close()

			ctx, cancel := context.WithCancel(context.Background())
			req := mk(ts.URL + "/api/predict").WithContext(ctx)
			clientErr := make(chan error, 1)
			go func() {
				resp, err := http.DefaultClient.Do(req)
				if err == nil {
					resp.Body.Close()
				}
				clientErr <- err
			}()

			<-f.admitted // the request is queued for a slot
			cancel()     // ... and its client gives up
			if err := <-clientErr; err == nil {
				t.Fatal("client request was not canceled")
			}
			select {
			case <-(<-serverCtx).Done(): // the server noticed the disconnect
			case <-time.After(5 * time.Second):
				t.Fatal("server never noticed the client disconnect (body not read to EOF?)")
			}
			close(f.admitWait) // the slot frees up
			<-handlerDone
			if got := strings.Join(f.Events(), " "); got != "admit:m release" {
				t.Fatalf("events %q: the model ran for a client that had left", got)
			}
		})
	}
}

// net/http only watches a connection for a disconnect once the handler has read the request body
// to EOF. A JSON decoder stops at the closing brace, so when the end of the body has not arrived
// yet (here: never, until the client gives up) the server would be blind to the disconnect and
// run the model anyway — unless the handler drains the body (drainBody) before queueing.
func TestDisconnectNoticedWhenBodyTailIsLate(t *testing.T) {
	f := &fakeRuntime{admitWait: make(chan struct{})}
	s, _ := newTestServer(f)
	jsonBody := []byte(`{"model":"m","image_base64":"` + b64(pngBytes(t, 8, 8)) + `"}`)
	consumed := make(chan struct{}) // the server has read the whole JSON value
	serverCtx := make(chan context.Context, 1)
	handlerDone := make(chan struct{})
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(handlerDone)
		serverCtx <- r.Context()
		r.Body = &signalAfter{ReadCloser: r.Body, n: len(jsonBody), done: consumed}
		s.routes().ServeHTTP(w, r)
	}))
	defer ts.Close()

	pr, pw := io.Pipe()
	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, "POST", ts.URL+"/api/predict", pr)
	req.Header.Set("Content-Type", "application/json")
	go func() {
		if resp, err := http.DefaultClient.Do(req); err == nil {
			resp.Body.Close()
		}
	}()
	go pw.Write(jsonBody) //nolint:errcheck // and then nothing: the body never ends

	<-consumed
	cancel() // the client gives up
	noticed := true
	select {
	case <-(<-serverCtx).Done():
	case <-time.After(3 * time.Second):
		noticed = false
	}
	close(f.admitWait) // unblock the handler either way, so the server can shut down
	<-handlerDone
	pw.Close()
	if !noticed {
		t.Fatal("server never noticed the disconnect: the body was not drained before queueing")
	}
	if got := strings.Join(f.Events(), " "); strings.Contains(got, "predict") {
		t.Fatalf("events %q: the model ran for a client that had left", got)
	}
}

// signalAfter closes done once n bytes have been read through it.
type signalAfter struct {
	io.ReadCloser
	n, read int
	done    chan struct{}
}

func (s *signalAfter) Read(p []byte) (int, error) {
	k, err := s.ReadCloser.Read(p)
	if s.read < s.n && s.read+k >= s.n {
		close(s.done)
	}
	s.read += k
	return k, err
}

// ---------------------------------------------------------------------------------------------
// encoding=base64
// ---------------------------------------------------------------------------------------------

// The arrays a client gets back with encoding=base64 are bit-for-bit the ones the model produced,
// over both content types; NaN/Inf included (JSON numbers cannot carry them at all).
func TestBase64ResponseRoundTripBitExact(t *testing.T) {
	depth := []float32{0, 0.1, 1.0 / 3, float32(math.NaN()), float32(math.Inf(1)), -1e-40}
	emb := [][]float32{{0.25, -0.5, 1e-7}, {3, float32(math.Inf(-1)), 0.1}}
	png := pngBytes(t, 6, 4)
	for name, req := range map[string]*http.Request{
		"multipart field": multipartRequest(t, "/api/predict", map[string]string{"model": "m", "encoding": "base64"}, part{"image", "i.png", png}),
		"multipart query": multipartRequest(t, "/api/predict?encoding=base64", map[string]string{"model": "m"}, part{"image", "i.png", png}),
		"JSON field":      jsonRequest("/api/predict", []byte(`{"model":"m","image_base64":"`+b64(png)+`","encoding":"base64"}`)),
	} {
		f := &fakeRuntime{result: api.Result{Task: api.TaskDepth, Model: "m", DepthMap: depth, DepthWidth: 3, DepthHeight: 2, Embeddings: emb}}
		_, h := newTestServer(f)
		rec := do(h, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("%s: %d %s", name, rec.Code, rec.Body)
		}
		var res api.Result
		if err := json.Unmarshal(rec.Body.Bytes(), &res); err != nil {
			t.Fatal(err)
		}
		if res.DepthMap != nil || res.Embeddings != nil || res.DepthWidth != 3 || res.DepthHeight != 2 {
			t.Fatalf("%s: arrays sent as numbers or shape lost: %s", name, rec.Body)
		}
		if err := res.DecodeArraysBase64(); err != nil {
			t.Fatal(err)
		}
		for i := range depth {
			if math.Float32bits(res.DepthMap[i]) != math.Float32bits(depth[i]) {
				t.Fatalf("%s: depth[%d] %v != %v", name, i, res.DepthMap[i], depth[i])
			}
		}
		for i := range emb {
			for j := range emb[i] {
				if math.Float32bits(res.Embeddings[i][j]) != math.Float32bits(emb[i][j]) {
					t.Fatalf("%s: emb[%d][%d] %v != %v", name, i, j, res.Embeddings[i][j], emb[i][j])
				}
			}
		}
	}
}

// Without encoding the answer is the JSON-number wire it always was.
func TestDefaultEncodingIsJSONNumbers(t *testing.T) {
	f := &fakeRuntime{result: api.Result{Task: api.TaskEmbed, Model: "m", Embeddings: [][]float32{{0.5, 0.25}}}}
	_, h := newTestServer(f)
	rec := do(h, multipartRequest(t, "/api/predict", map[string]string{"model": "m"}, part{"image", "i.png", pngBytes(t, 4, 4)}))
	if got := rec.Body.String(); got != `{"task":"embed","model":"m","embeddings":[[0.5,0.25]],"duration_ms":0}`+"\n" {
		t.Fatalf("default wire changed: %s", got)
	}
}

// infer_tensor honours ?encoding=base64 as well.
func TestInferTensorBase64(t *testing.T) {
	f := &fakeRuntime{result: api.Result{Embeddings: [][]float32{{1, 2, 3}}}}
	_, h := newTestServer(f)
	rec := do(h, httptest.NewRequest("POST", "/api/infer_tensor?model=m&shape=1&encoding=base64", bytes.NewReader([]byte{0, 0, 128, 63})))
	if !strings.Contains(rec.Body.String(), `"embeddings_shape":[1,3]`) || f.tensor.Data[0] != 1 {
		t.Fatalf("got %d %s (tensor %v)", rec.Code, rec.Body, f.tensor.Data)
	}
}

// ---------------------------------------------------------------------------------------------
// explain / preprocess / templates request parsing
// ---------------------------------------------------------------------------------------------

func TestExplainOptionsBothContentTypes(t *testing.T) {
	png := pngBytes(t, 8, 8)
	cases := []struct {
		fields map[string]string
		json   string
		want   lifecycle.ExplainRequest
	}{
		{map[string]string{}, `{}`, lifecycle.ExplainRequest{Alpha: 0.5}},
		{map[string]string{"detection_idx": "3", "top_channels": "16", "alpha": "0.25", "class": "cup"},
			`{"detection_idx":3,"top_channels":16,"alpha":0.25,"class":"cup"}`,
			lifecycle.ExplainRequest{DetectionIdx: 3, TopChannels: 16, Alpha: 0.25, Class: "cup"}},
		{map[string]string{"detection_idx": "-1", "top_channels": "-4", "alpha": "2"},
			`{"detection_idx":-1,"top_channels":-4,"alpha":2}`, lifecycle.ExplainRequest{Alpha: 0.5}},
		{map[string]string{"alpha": "0"}, `{"alpha":0}`, lifecycle.ExplainRequest{Alpha: 0}},
		{map[string]string{"detection_idx": "x", "alpha": "x"}, `{}`, lifecycle.ExplainRequest{Alpha: 0.5}},
	}
	for _, c := range cases {
		fields := map[string]string{"model": "m", "format": "numpy"}
		for k, v := range c.fields {
			fields[k] = v
		}
		var body map[string]any
		_ = json.Unmarshal([]byte(c.json), &body)
		body["model"], body["format"], body["image_base64"] = "m", "numpy", b64(png)
		jb, _ := json.Marshal(body)
		for name, req := range map[string]*http.Request{
			"multipart": multipartRequest(t, "/api/explain", fields, part{"image", "i.png", png}),
			"JSON":      jsonRequest("/api/explain", jb),
		} {
			f := &fakeRuntime{}
			_, h := newTestServer(f)
			rec := do(h, req)
			if rec.Code != http.StatusOK || rec.Header().Get("X-Heatmap-Shape") != "2,2" || rec.Body.Len() != 16 {
				t.Fatalf("%s %v: %d %q", name, c.fields, rec.Code, rec.Header())
			}
			if f.explain != c.want {
				t.Errorf("%s %v: explain request %+v, want %+v", name, c.fields, f.explain, c.want)
			}
		}
	}
}

func TestExplainPNGDefault(t *testing.T) {
	_, h := newTestServer(&fakeRuntime{})
	rec := do(h, multipartRequest(t, "/api/explain", map[string]string{"model": "m"}, part{"image", "i.png", pngBytes(t, 8, 8)}))
	if rec.Code != http.StatusOK || rec.Header().Get("Content-Type") != "image/png" {
		t.Fatalf("%d %v", rec.Code, rec.Header())
	}
}

// /api/preprocess takes the predict request (both content types), image optional.
func TestPreprocessRequestBothContentTypes(t *testing.T) {
	png := pngBytes(t, 10, 10)
	for name, req := range map[string]*http.Request{
		"multipart": multipartRequest(t, "/api/preprocess", map[string]string{"model": "m", "prompt": "cup", "box": "1,1,2,2"}, part{"image", "i.png", png}),
		"JSON":      jsonRequest("/api/preprocess", []byte(`{"model":"m","prompt":"cup","box":"1,1,2,2","image_base64":"`+b64(png)+`"}`)),
	} {
		f := &fakeRuntime{}
		_, h := newTestServer(f)
		rec := do(h, req)
		if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"name":"pixel_values"`) {
			t.Fatalf("%s: %d %s", name, rec.Code, rec.Body)
		}
		if f.prompt.Text != "cup" || f.prompt.Boxes[0] != [4]float64{1, 1, 2, 2} || f.img.Bounds().Dx() != 10 {
			t.Errorf("%s: prompt %+v img %v", name, f.prompt, f.img.Bounds())
		}
	}
	// Text-only: no image at all.
	f := &fakeRuntime{}
	_, h := newTestServer(f)
	if rec := do(h, multipartRequest(t, "/api/preprocess", map[string]string{"model": "clip-text", "prompt": "a cat"})); rec.Code != 200 || f.img != nil {
		t.Fatalf("text-only preprocess: %d %s (img %v)", rec.Code, rec.Body, f.img)
	}
}

func TestTemplatesSuccessAnswersJSON(t *testing.T) {
	s, h := newTestServer(&fakeRuntime{})
	rec := do(h, multipartRequest(t, "/api/templates", map[string]string{"name": "cup"},
		part{"images", "a.png", pngBytes(t, 4, 4)}, part{"image", "b.png", pngBytes(t, 4, 4)}))
	if rec.Code != 200 || rec.Body.String() != `{"count":2,"name":"cup"}`+"\n" {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	if rec = do(h, httptest.NewRequest("GET", "/api/templates", nil)); rec.Body.String() != `{"templates":["cup"]}`+"\n" {
		t.Fatalf("list: %s", rec.Body)
	}
	if rec = do(h, httptest.NewRequest("DELETE", "/api/templates/cup", nil)); rec.Code != http.StatusNoContent || s.tmpl.Len() != 0 {
		t.Fatalf("delete: %d", rec.Code)
	}
}

// The store's pixel bound is enforced from the image headers before decoding, and replacing a
// set counts the room its old images free.
func TestTemplatesPixelBoundCheckedBeforeDecoding(t *testing.T) {
	s, h := newTestServer(&fakeRuntime{})
	s.tmpl = templates.NewWithLimits(4, 100) // 100 pixels in all

	// 2 × 8×8 = 128 > 100: refused (400) before anything is stored.
	rec := do(h, multipartRequest(t, "/api/templates", map[string]string{"name": "big"},
		part{"images", "a.png", pngBytes(t, 8, 8)}, part{"images", "b.png", pngBytes(t, 8, 8)}))
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "store is full") || s.tmpl.Len() != 0 {
		t.Fatalf("over the bound: %d %s (sets %d)", rec.Code, rec.Body, s.tmpl.Len())
	}
	// A header that does not decode is a 400 too.
	rec = do(h, multipartRequest(t, "/api/templates", map[string]string{"name": "junk"},
		part{"images", "a.png", []byte("not an image")}))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("junk upload: %d %s", rec.Code, rec.Body)
	}
	// 8×8 = 64 fits; replacing it with 9×9 = 81 fits too (the old 64 are freed).
	for _, side := range []int{8, 9} {
		rec = do(h, multipartRequest(t, "/api/templates", map[string]string{"name": "cup"},
			part{"images", "a.png", pngBytes(t, side, side)}))
		if rec.Code != http.StatusOK {
			t.Fatalf("%dx%d: %d %s", side, side, rec.Code, rec.Body)
		}
	}
	if s.tmpl.Pixels() != 81 {
		t.Fatalf("store holds %d pixels, want 81", s.tmpl.Pixels())
	}
}
