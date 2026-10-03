package catalog

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// testWeights is a deterministic, non-repeating body (so a misplaced byte range changes the hash).
func testWeights(n int) []byte {
	b := make([]byte, n)
	x := uint32(2463534242)
	for i := range b {
		x ^= x << 13
		x ^= x >> 17
		x ^= x << 5
		b[i] = byte(x)
	}
	return b
}

// cutConnection answers with the full Content-Length, sends only body[:n] and then drops the TCP
// connection — what a flaky link or a killed proxy looks like to the client.
func cutConnection(t *testing.T, w http.ResponseWriter, body []byte, n int) {
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.Header().Set("Accept-Ranges", "bytes")
	w.WriteHeader(http.StatusOK)
	w.Write(body[:n])
	w.(http.Flusher).Flush()
	conn, _, err := w.(http.Hijacker).Hijack()
	if err != nil {
		t.Errorf("hijack: %v", err)
		return
	}
	conn.Close()
}

// rangeLog records the Range header of every request a test server saw.
type rangeLog struct {
	mu     sync.Mutex
	ranges []string
}

func (l *rangeLog) add(r *http.Request) int {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.ranges = append(l.ranges, r.Header.Get("Range"))
	return len(l.ranges)
}

func (l *rangeLog) get() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]string(nil), l.ranges...)
}

// An interrupted pull keeps its partial file, and the next pull fetches only the missing bytes
// (Range), then verifies the WHOLE file against the pin before installing it.
func TestPullResumesInterruptedDownload(t *testing.T) {
	body := testWeights(256 << 10)
	half := len(body) / 2
	var log rangeLog
	var served atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if log.add(r) == 1 {
			cutConnection(t, w, body, half)
			return
		}
		cw := &countingWriter{ResponseWriter: w, n: &served}
		http.ServeContent(cw, r, "model.onnx", time.Time{}, bytes.NewReader(body))
	}))
	defer srv.Close()
	withTestEntry(t, dlEntry(srv.URL+"/model.onnx", sha(body)))

	dir := t.TempDir()
	dest := filepath.Join(dir, "test-dl", "model.onnx")
	err := Pull("test-dl", PullOptions{ModelsDir: dir, Out: &bytes.Buffer{}})
	if err == nil {
		t.Fatal("the first pull must fail: the connection was cut")
	}
	if !strings.Contains(err.Error(), "run the same pull again to resume") {
		t.Errorf("error does not say the download can be resumed: %v", err)
	}
	if _, err := os.Stat(dest); !os.IsNotExist(err) {
		t.Fatalf("an incomplete download was installed under the real name (stat err %v)", err)
	}
	st, err := os.Stat(partialPath(dest))
	if err != nil || st.Size() != int64(half) {
		t.Fatalf("partial file: %v, size %v; want %d bytes kept", err, st, half)
	}

	var out bytes.Buffer
	if err := Pull("test-dl", PullOptions{ModelsDir: dir, Out: &out}); err != nil {
		t.Fatalf("resumed pull: %v\n%s", err, out.String())
	}
	got, err := os.ReadFile(dest)
	if err != nil || !bytes.Equal(got, body) {
		t.Fatalf("resumed file differs from the original (err=%v, %d bytes)", err, len(got))
	}
	if r := log.get(); len(r) != 2 || r[1] != "bytes="+strconv.Itoa(half)+"-" {
		t.Fatalf("requests' Range headers = %q, want [\"\" \"bytes=%d-\"]", r, half)
	}
	if n := served.Load(); n != int64(len(body)-half) {
		t.Errorf("the resumed request transferred %d bytes, want only the missing %d", n, len(body)-half)
	}
	if !strings.Contains(out.String(), "resuming model.onnx") {
		t.Errorf("no resume message:\n%s", out.String())
	}
	assertNoTempFiles(t, filepath.Join(dir, "test-dl"))
}

// A server that ignores Range (200 + the whole body) must not get its body appended to the
// partial file: the download restarts from zero.
func TestResumeServerWithoutRangeSupportRestarts(t *testing.T) {
	body := testWeights(64 << 10)
	var log rangeLog
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		log.add(r)
		w.Write(body) // never honours Range
	}))
	defer srv.Close()
	dest := filepath.Join(t.TempDir(), "model.onnx")
	mustWrite(t, partialPath(dest), string(body[:1000]))

	if _, err := downloadResumable(srv.URL, dest, expect{SHA256: sha(body), Floor: 1}, &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(dest); !bytes.Equal(got, body) {
		t.Fatalf("got %d bytes, want the original %d", len(got), len(body))
	}
	if r := log.get(); len(r) != 1 || r[0] != "bytes=1000-" {
		t.Fatalf("Range headers = %q", r)
	}
	assertNoTempFiles(t, filepath.Dir(dest))
}

// A partial file whose bytes are wrong (an earlier upstream version, a torn write after a crash)
// fails the pin once resumed; it is then discarded and the file downloaded once from zero.
func TestResumeCorruptPrefixRetriesFromZero(t *testing.T) {
	body := testWeights(64 << 10)
	var log rangeLog
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		log.add(r)
		http.ServeContent(w, r, "model.onnx", time.Time{}, bytes.NewReader(body))
	}))
	defer srv.Close()
	dest := filepath.Join(t.TempDir(), "model.onnx")
	bad := append([]byte(nil), body[:4096]...)
	bad[100] ^= 0xff
	mustWrite(t, partialPath(dest), string(bad))

	if _, err := downloadResumable(srv.URL, dest, expect{SHA256: sha(body), Floor: 1}, &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(dest); !bytes.Equal(got, body) {
		t.Fatal("the corrupt prefix survived into the installed file")
	}
	if r := log.get(); len(r) != 2 || r[0] != "bytes=4096-" || r[1] != "" {
		t.Fatalf("Range headers = %q, want a resume then a full download", r)
	}
	assertNoTempFiles(t, filepath.Dir(dest))

	// Wrong upstream bytes even from zero: refused, and nothing is kept to resume from.
	dest2 := filepath.Join(t.TempDir(), "model.onnx")
	mustWrite(t, partialPath(dest2), string(body[:4096]))
	if _, err := downloadResumable(srv.URL, dest2, expect{SHA256: sha([]byte("other")), Floor: 1}, &bytes.Buffer{}); err == nil ||
		!strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("want a sha256 mismatch, got %v", err)
	}
	if _, err := os.Stat(dest2); !os.IsNotExist(err) {
		t.Fatal("bytes failing the pin were installed")
	}
	assertNoTempFiles(t, filepath.Dir(dest2))
}

// A partial file at least as long as the upstream file gets 416 from the server: start over.
func TestResumeRangeNotSatisfiableRestarts(t *testing.T) {
	body := testWeights(8 << 10)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.ServeContent(w, r, "model.onnx", time.Time{}, bytes.NewReader(body))
	}))
	defer srv.Close()
	dest := filepath.Join(t.TempDir(), "model.onnx")
	mustWrite(t, partialPath(dest), string(testWeights(16<<10)))
	if _, err := downloadResumable(srv.URL, dest, expect{SHA256: sha(body), Floor: 1}, &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(dest); !bytes.Equal(got, body) {
		t.Fatal("wrong bytes after a 416 restart")
	}
}

// The plain (non-resumable) download keeps its old contract: an interrupted transfer leaves no
// temp file behind, and Pull uses it for unpinned files.
func TestPullUnpinnedInterruptedLeavesNothing(t *testing.T) {
	body := testWeights(64 << 10)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cutConnection(t, w, body, len(body)/2)
	}))
	defer srv.Close()
	withTestEntry(t, dlEntry(srv.URL+"/model.onnx", ""))
	dir := t.TempDir()
	if err := Pull("test-dl", PullOptions{ModelsDir: dir, Out: &bytes.Buffer{}}); err == nil {
		t.Fatal("want an error")
	}
	assertNoTempFiles(t, filepath.Join(dir, "test-dl"))
}

// A symlink planted at the partial path is never written through.
func TestResumeDoesNotFollowSymlinkedPartial(t *testing.T) {
	body := testWeights(8 << 10)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.ServeContent(w, r, "model.onnx", time.Time{}, bytes.NewReader(body))
	}))
	defer srv.Close()
	dir := t.TempDir()
	victim := filepath.Join(t.TempDir(), "victim")
	mustWrite(t, victim, "precious")
	dest := filepath.Join(dir, "model.onnx")
	if err := os.Symlink(victim, partialPath(dest)); err != nil {
		t.Skip("symlinks unsupported:", err)
	}
	if _, err := downloadResumable(srv.URL, dest, expect{SHA256: sha(body), Floor: 1}, nil); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(victim); string(got) != "precious" {
		t.Fatal("the download wrote through a symlink")
	}
	if got, _ := os.ReadFile(dest); !bytes.Equal(got, body) {
		t.Fatal("wrong bytes")
	}
}

func TestParseContentRange(t *testing.T) {
	for _, c := range []struct {
		in          string
		start, size int64
		ok          bool
	}{
		{"bytes 100-199/200", 100, 200, true},
		{"bytes 0-0/1", 0, 1, true},
		{"bytes 5-9/*", 5, -1, true},
		{" bytes 7-8/20 ", 7, 20, true},
		{"bytes 100-199/150", 0, 0, false}, // end beyond size
		{"bytes 9-5/20", 0, 0, false},
		{"bytes */200", 0, 0, false},
		{"items 0-1/2", 0, 0, false},
		{"", 0, 0, false},
	} {
		start, size, ok := parseContentRange(c.in)
		if ok != c.ok || (ok && (start != c.start || size != c.size)) {
			t.Errorf("parseContentRange(%q) = %d, %d, %v; want %d, %d, %v", c.in, start, size, ok, c.start, c.size, c.ok)
		}
	}
}

// countingWriter counts the body bytes a handler writes.
type countingWriter struct {
	http.ResponseWriter
	n *atomic.Int64
}

func (c *countingWriter) Write(p []byte) (int, error) {
	n, err := c.ResponseWriter.Write(p)
	c.n.Add(int64(n))
	return n, err
}
