package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"visionserve/internal/lifecycle"
)

// countingReader counts the bytes read through it.
type countingReader struct {
	r io.Reader
	n atomic.Int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	k, err := c.r.Read(p)
	c.n.Add(int64(k))
	return k, err
}

// A multipart request the model's queue refuses — or whose client already left — is answered
// without its image part being read: the body is consumed only up to that part's header.
func TestRefusedMultipartDoesNotReadUpload(t *testing.T) {
	big := bytes.Repeat([]byte{0xAB}, 4<<20) // a 4 MiB "image" the server must not read
	overloaded := fmt.Errorf("lifecycle: m queue full: %w", lifecycle.ErrOverloaded)
	for _, path := range []string{"/api/predict", "/api/preprocess", "/api/explain"} {
		for _, c := range []struct {
			name     string
			f        *fakeRuntime
			canceled bool
			want     int
		}{
			{"overloaded", &fakeRuntime{admitErr: overloaded}, false, http.StatusServiceUnavailable},
			{"client gone", &fakeRuntime{}, true, statusClientClosedRequest},
		} {
			t.Run(path+" "+c.name, func(t *testing.T) {
				body, ct := orderedMultipart(t, formPart{"model", "", []byte("m")}, formPart{"prompt", "", []byte("cup")},
					formPart{"image", "i.png", big})
				cr := &countingReader{r: bytes.NewReader(body)}
				req := httptest.NewRequest("POST", path, cr)
				req.Header.Set("Content-Type", ct)
				if c.canceled {
					ctx, cancel := context.WithCancel(context.Background())
					cancel()
					req = req.WithContext(ctx)
				}
				_, h := newTestServer(c.f)
				rec := do(h, req)
				if rec.Code != c.want {
					t.Fatalf("status %d, want %d: %s", rec.Code, c.want, rec.Body)
				}
				if c.want == http.StatusServiceUnavailable && rec.Header().Get("Retry-After") == "" {
					t.Fatal("503 without Retry-After")
				}
				// The unread rest of the body (length unknown here) is not discarded by net/http
				// before the answer goes out: the connection is closed instead.
				if rec.Header().Get("Connection") != "close" {
					t.Fatal("refused with an unread body of unknown length, but no Connection: close")
				}
				if n := cr.n.Load(); n > 64<<10 {
					t.Fatalf("read %d of %d body bytes: the refused upload was read", n, len(body))
				}
				if ev := eventsOf(c.f); ev != "admit:m" {
					t.Fatalf("events %q, want only the refused admit:m", ev)
				}
			})
		}
	}

	// A small refused body (known length) is left to net/http, which discards it and keeps the
	// connection alive.
	small := multipartRequest(t, "/api/predict", map[string]string{"model": "m"}, part{"image", "i.png", []byte("x")})
	if rec := do(newTestHandler(&fakeRuntime{admitErr: overloaded}), small); rec.Code != 503 || rec.Header().Get("Connection") != "" {
		t.Fatalf("small refused body: %d, Connection %q", rec.Code, rec.Header().Get("Connection"))
	}

	// The counter does see an admitted upload: the same body, admitted, is read to the end.
	body, ct := orderedMultipart(t, formPart{"model", "", []byte("m")}, formPart{"image", "i.png", big})
	cr := &countingReader{r: bytes.NewReader(body)}
	req := httptest.NewRequest("POST", "/api/predict", cr)
	req.Header.Set("Content-Type", ct)
	if rec := do(newTestHandler(&fakeRuntime{}), req); rec.Code != http.StatusBadRequest || cr.n.Load() != int64(len(body)) {
		t.Fatalf("admitted junk image: status %d, read %d of %d", rec.Code, cr.n.Load(), len(body))
	}
}

func eventsOf(r *fakeRuntime) string { return strings.Join(r.Events(), " ") }

func newTestHandler(r *fakeRuntime) http.Handler { _, h := newTestServer(r); return h }

// Field order is the client's: an image (and depth map) sent BEFORE the model field is kept,
// bounded, and the request admitted once the form is read — with the very same result as the
// model-first order.
func TestMultipartImageBeforeModel(t *testing.T) {
	full, img, depth := fullRequest(t)
	fields := formOf(t, full)
	delete(fields, "image_base64")
	delete(fields, "depth_base64")
	delete(fields, "model")
	var opts []formPart
	for k, v := range fields {
		opts = append(opts, formPart{k, "", []byte(v)})
	}
	model := formPart{"model", "", []byte("m")}
	files := []formPart{{"depth", "d.bin", depth}, {"image", "i.png", img}}

	run := func(parts ...formPart) (string, *fakeRuntime, string) {
		body, ct := orderedMultipart(t, parts...)
		req := httptest.NewRequest("POST", "/api/predict", bytes.NewReader(body))
		req.Header.Set("Content-Type", ct)
		fr := &fakeRuntime{}
		rec := do(newTestHandler(fr), req)
		if rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body)
		}
		return rec.Body.String(), fr, eventsOf(fr)
	}
	modelFirst := append(append([]formPart{model}, opts...), files...)
	modelLast := append(append(append([]formPart{}, files...), opts...), model)
	wantBody, wantF, ev1 := run(modelFirst...)
	gotBody, gotF, ev2 := run(modelLast...)
	if !reflect.DeepEqual(wantF.prompt, gotF.prompt) || wantF.img.Bounds() != gotF.img.Bounds() || wantBody != gotBody {
		t.Fatalf("model-last differs from model-first:\n first: %+v\n last:  %+v", wantF.prompt, gotF.prompt)
	}
	// Model first: admission is probed before the image is read (a slot taken and given back),
	// then taken for real after the body. Model last: nothing to probe.
	if ev1 != "admit:m release admit:m predict:m release" || ev2 != "admit:m predict:m release" {
		t.Fatalf("events: model first %q, model last %q", ev1, ev2)
	}

	// /api/explain too.
	body, ct := orderedMultipart(t, formPart{"image", "i.png", img}, formPart{"format", "", []byte("numpy")}, model)
	req := httptest.NewRequest("POST", "/api/explain", bytes.NewReader(body))
	req.Header.Set("Content-Type", ct)
	fr := &fakeRuntime{}
	if rec := do(newTestHandler(fr), req); rec.Code != http.StatusOK || eventsOf(fr) != "admit:m explain:m release" {
		t.Fatalf("explain, image first: %d %s (events %q)", rec.Code, rec.Body, eventsOf(fr))
	}
}

// End to end: a model-first request is admitted before its image is read; when its client leaves
// while it waits for the slot (the upload never finishes), the model does not run and the slot is
// given back.
func TestClientDisconnectMidUpload(t *testing.T) {
	fr := &fakeRuntime{admitWait: make(chan struct{}), admitted: make(chan struct{})}
	s, _ := newTestServer(fr)
	handlerDone := make(chan struct{})
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(handlerDone)
		s.routes().ServeHTTP(w, r)
	}))
	defer ts.Close()

	body, ct := orderedMultipart(t, formPart{"model", "", []byte("m")}, formPart{"image", "i.png", pngBytes(t, 8, 8)})
	head := body[:bytes.Index(body, []byte("\x89PNG"))+4] // the model field, the image part header, 4 bytes
	pr, pw := io.Pipe()
	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, "POST", ts.URL+"/api/predict", pr)
	req.Header.Set("Content-Type", ct)
	clientErr := make(chan error, 1)
	go func() {
		resp, err := http.DefaultClient.Do(req)
		if err == nil {
			resp.Body.Close()
		}
		clientErr <- err
	}()
	go pw.Write(head) //nolint:errcheck // and then nothing: the rest of the upload never comes

	<-fr.admitted                       // queued for a slot, its image not read yet
	cancel()                            // ... and the client gives up
	pw.CloseWithError(context.Canceled) // (the transport waits for its body reader to return)
	if err := <-clientErr; err == nil {
		t.Fatal("client request was not canceled")
	}
	close(fr.admitWait) // the slot frees up
	<-handlerDone
	if got := eventsOf(fr); got != "admit:m release" {
		t.Fatalf("events %q: the model ran for a client that had left, or the slot leaked", got)
	}
}

// A client that sends its model and then trickles its image holds no admission slot while it
// uploads: the probe before the image gives its slot straight back, and the request is admitted
// for real only after the body is read. (Holding the slot during the upload let a few dozen slow
// uploads fill a model's bound and get every other request 503 for up to the ReadTimeout.)
func TestSlowUploadHoldsNoSlot(t *testing.T) {
	fr := &fakeRuntime{}
	ts := httptest.NewServer(newTestHandler(fr))
	defer ts.Close()

	body, ct := orderedMultipart(t, formPart{"model", "", []byte("m")}, formPart{"image", "i.png", pngBytes(t, 8, 8)})
	cut := bytes.Index(body, []byte("\x89PNG")) + 4 // the model field, the image part header, 4 bytes
	pr, pw := io.Pipe()
	req, _ := http.NewRequest("POST", ts.URL+"/api/predict", pr)
	req.Header.Set("Content-Type", ct)
	status := make(chan int, 1)
	go func() {
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			status <- 0
			return
		}
		resp.Body.Close()
		status <- resp.StatusCode
	}()
	if _, err := pw.Write(body[:cut]); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for eventsOf(fr) != "admit:m release" { // probed, and the slot already given back
		if time.Now().After(deadline) {
			t.Fatalf("mid-upload events %q, want the probe only", eventsOf(fr))
		}
		time.Sleep(time.Millisecond)
	}
	if _, err := pw.Write(body[cut:]); err != nil {
		t.Fatal(err)
	}
	pw.Close()
	if code := <-status; code != http.StatusOK {
		t.Fatalf("status %d", code)
	}
	if got := eventsOf(fr); got != "admit:m release admit:m predict:m release" {
		t.Fatalf("events %q", got)
	}
}

// The admission slot and the parse keep http.Request.FormValue's precedence, which these handlers
// used before: a URL query value comes before the form's.
func TestMultipartQueryPrecedence(t *testing.T) {
	body, ct := orderedMultipart(t, formPart{"model", "", []byte("form")}, formPart{"image", "i.png", pngBytes(t, 4, 4)})
	req := httptest.NewRequest("POST", "/api/predict?model=query", bytes.NewReader(body))
	req.Header.Set("Content-Type", ct)
	fr := &fakeRuntime{}
	if rec := do(newTestHandler(fr), req); rec.Code != http.StatusOK || eventsOf(fr) != "admit:query release admit:query predict:query release" {
		t.Fatalf("%d %s (events %q)", rec.Code, rec.Body, eventsOf(fr))
	}
}

// The parse errors ParseMultipartForm gave are kept: a non-form body, multipart/mixed, too many
// parts.
func TestMultipartParseErrors(t *testing.T) {
	many := make([]formPart, maxFormParts+1)
	for i := range many {
		many[i] = formPart{"x", "", []byte("1")}
	}
	tooMany, ct := orderedMultipart(t, many...)
	for _, c := range []struct {
		name, ct string
		body     []byte
		want     string
	}{
		{"urlencoded", "application/x-www-form-urlencoded", []byte("model=m"), "request Content-Type isn't multipart/form-data"},
		{"mixed", "multipart/mixed; boundary=b", []byte("--b--\r\n"), "request Content-Type isn't multipart/form-data"},
		{"no boundary", "multipart/form-data", []byte("--b--\r\n"), "no multipart boundary param"},
		{"too many parts", ct, tooMany, "message too large"},
	} {
		req := httptest.NewRequest("POST", "/api/predict", bytes.NewReader(c.body))
		req.Header.Set("Content-Type", c.ct)
		fr := &fakeRuntime{}
		rec := do(newTestHandler(fr), req)
		var e struct{ Error string }
		_ = json.Unmarshal(rec.Body.Bytes(), &e)
		if rec.Code != http.StatusBadRequest || !strings.HasPrefix(e.Error, "failed to parse multipart form: ") ||
			!strings.Contains(e.Error, c.want) || len(fr.Events()) != 0 {
			t.Errorf("%s: %d %q (events %v)", c.name, rec.Code, e.Error, fr.Events())
		}
	}
}

// spool keeps a part in memory while the budget lasts and the rest in a temp file, stores at most
// limit bytes, and Close removes the file.
func TestSpoolSpillsToTempFile(t *testing.T) {
	data := make([]byte, 100)
	for i := range data {
		data[i] = byte(i)
	}
	fileMem, mem := int64(30), int64(1000)
	u, err := spool(bytes.NewReader(data), 80, &fileMem, &mem)
	if err != nil {
		t.Fatal(err)
	}
	if len(u.mem) != 30 || u.file == nil || fileMem != 0 || mem != 970 {
		t.Fatalf("mem %d file %v, budgets %d/%d", len(u.mem), u.file, fileMem, mem)
	}
	for i := 0; i < 2; i++ { // open twice: each reader starts at the first byte
		r, err := u.open()
		if err != nil {
			t.Fatal(err)
		}
		if got, _ := io.ReadAll(r); !bytes.Equal(got, data[:80]) {
			t.Fatalf("read back %d bytes, want the first 80", len(got))
		}
	}
	name := u.file.Name()
	(&formData{files: map[string]*upload{"image": u}}).Close()
	if _, err := os.Stat(name); !os.IsNotExist(err) {
		t.Fatalf("temp file %s not removed: %v", name, err)
	}

	// Within the budget nothing touches the disk.
	fileMem = 1000
	if u, err = spool(bytes.NewReader(data), 1<<20, &fileMem, &mem); err != nil || u.file != nil || len(u.mem) != 100 {
		t.Fatalf("small part: %v file=%v mem=%d", err, u.file, len(u.mem))
	}
}
