package catalog

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Review #5: a hand edit that keeps the generated header line used to be overwritten on re-pull,
// although deploy/README promises an edited manifest is kept.
func TestPullKeepsHandEditsUnderTheGeneratedHeader(t *testing.T) {
	dir := t.TempDir()
	e, _ := Lookup("rfdetr-gdino")
	for _, dep := range e.Dependencies {
		mustWrite(t, filepath.Join(dir, dep, "manifest.yaml"), "stub")
	}
	for _, rel := range []string{"rf-detr/rf-detr-base.onnx", "rf-detr/coco91.txt", "grounding-dino/model-fixedmask.onnx"} {
		mustWrite(t, filepath.Join(dir, rel), "x")
	}
	path := filepath.Join(dir, "rfdetr-gdino", "manifest.yaml")

	// Edited body, header untouched -> kept.
	edited := strings.Replace(e.RenderManifest(), "\ninput:\n", "\n# my tweak\ninput:\n", 1)
	mustWrite(t, path, edited)
	var log bytes.Buffer
	if err := Pull("rfdetr-gdino", PullOptions{ModelsDir: dir, Out: &log}); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(path); string(got) != edited {
		t.Fatalf("hand edit under the generated header was overwritten:\n%s", got)
	}

	// Unedited output of a DIFFERENT (older) catalog, with its own recorded hash -> regenerated.
	old := e
	old.ConfThreshold = 0.987
	mustWrite(t, path, old.RenderManifest())
	if err := Pull("rfdetr-gdino", PullOptions{ModelsDir: dir, Out: &log}); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(path); string(got) != e.RenderManifest() {
		t.Fatalf("an unedited manifest from an older catalog was not regenerated:\n%s", got)
	}
}

// withTestEntry adds e to the catalog for the duration of the test.
func withTestEntry(t *testing.T, e Entry) {
	t.Helper()
	saved := builtin
	builtin = append(append([]Entry(nil), builtin...), e)
	t.Cleanup(func() { builtin = saved })
}

func sha(b []byte) string { s := sha256.Sum256(b); return hex.EncodeToString(s[:]) }

func dlEntry(url, digest string) Entry {
	return Entry{
		Name: "test-dl", Task: "detection", License: "Apache-2.0", Architecture: "test-arch",
		Files:      []File{{Role: "model", LocalFilename: "model.onnx", DirectURL: url, SHA256: digest}},
		InputWidth: 32, InputHeight: 32, InputLayout: "NCHW", PostprocessType: "detr",
		RuntimePrefer: []string{"cpu"}, Verified: true,
	}
}

// Review #7: verification used to run AFTER the rename, so a failed --force re-download replaced
// the good file with bad bytes and then deleted it; a running server could load them meanwhile.
func TestPullForceFailedVerificationKeepsGoodFile(t *testing.T) {
	good := bytes.Repeat([]byte("g"), 4096)
	var serve atomic.Value
	serve.Store(good)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(serve.Load().([]byte))
	}))
	defer srv.Close()
	withTestEntry(t, dlEntry(srv.URL+"/model.onnx", sha(good)))

	dir := t.TempDir()
	if err := Pull("test-dl", PullOptions{ModelsDir: dir, Out: &bytes.Buffer{}}); err != nil {
		t.Fatalf("first pull: %v", err)
	}
	serve.Store(bytes.Repeat([]byte("B"), 4096)) // upstream now serves wrong bytes
	if err := Pull("test-dl", PullOptions{ModelsDir: dir, Force: true, Out: &bytes.Buffer{}}); err == nil {
		t.Fatal("a sha256 mismatch must fail the pull")
	}
	got, err := os.ReadFile(filepath.Join(dir, "test-dl", "model.onnx"))
	if err != nil || !bytes.Equal(got, good) {
		t.Fatalf("failed --force re-download destroyed the verified file (err=%v, %d bytes)", err, len(got))
	}
	assertNoTempFiles(t, filepath.Join(dir, "test-dl"))
}

// Review #7: an HTML error page / truncated body for an UNPINNED file must not replace anything.
func TestPullRejectsHTMLBeforeReplacing(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("<!DOCTYPE html><html>" + strings.Repeat(" ", 4096)))
	}))
	defer srv.Close()
	withTestEntry(t, dlEntry(srv.URL+"/model.onnx", ""))
	dir := t.TempDir()
	prev := bytes.Repeat([]byte("p"), 2048)
	mustWrite(t, filepath.Join(dir, "test-dl", "model.onnx"), string(prev))
	err := Pull("test-dl", PullOptions{ModelsDir: dir, Force: true, Out: &bytes.Buffer{}})
	if err == nil || !strings.Contains(err.Error(), "HTML") {
		t.Fatalf("expected an HTML rejection, got %v", err)
	}
	if got, _ := os.ReadFile(filepath.Join(dir, "test-dl", "model.onnx")); !bytes.Equal(got, prev) {
		t.Fatal("an HTML error page replaced the previous file")
	}
	assertNoTempFiles(t, filepath.Join(dir, "test-dl"))
}

func assertNoTempFiles(t *testing.T, dir string) {
	t.Helper()
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.Contains(e.Name(), ".part") {
			t.Errorf("temporary download left behind: %s", e.Name())
		}
	}
}

// Review #6: two downloads of the same file shared destPath+".part"; the second os.Create
// truncated the first's file and the result mixed both bodies — reported OK when unpinned.
func TestConcurrentDownloadsDoNotMix(t *testing.T) {
	const size = 8192
	a, b := bytes.Repeat([]byte("A"), size), bytes.Repeat([]byte("B"), size)
	firstHalf := make(chan struct{})
	release := make(chan struct{})
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "8192")
		if atomic.AddInt32(&calls, 1) == 1 {
			w.Write(a[:size/2])
			w.(http.Flusher).Flush()
			close(firstHalf)
			<-release
			w.Write(a[size/2:])
			return
		}
		w.Write(b)
	}))
	defer srv.Close()

	dir := t.TempDir()
	dest := filepath.Join(dir, "model.onnx")
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_, _ = downloadURL(srv.URL, dest, expect{Floor: 1}, nil)
	}()
	<-firstHalf
	waitForPartial(dir, size/2)
	if _, err := downloadURL(srv.URL, dest, expect{Floor: 1}, nil); err != nil {
		t.Fatalf("second download: %v", err)
	}
	close(release)
	wg.Wait()

	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, a) && !bytes.Equal(got, b) {
		t.Fatalf("concurrent downloads produced a mixed file (%q...%q)", got[:4], got[len(got)-4:])
	}
}

// waitForPartial waits (bounded) until some file in dir holds n bytes, i.e. the first download
// has written its first half to disk.
func waitForPartial(dir string, n int64) {
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		entries, _ := os.ReadDir(dir)
		for _, e := range entries {
			if info, err := e.Info(); err == nil && info.Size() >= n {
				return
			}
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// Review #6: concurrent pulls of the same model serialize on a per-model lock, so the second one
// finds the verified file already in place instead of downloading over the first.
func TestConcurrentPullsSerialize(t *testing.T) {
	body := bytes.Repeat([]byte("m"), 4096)
	started := make(chan struct{}, 4)
	var gets int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&gets, 1)
		started <- struct{}{}
		time.Sleep(200 * time.Millisecond)
		w.Write(body)
	}))
	defer srv.Close()
	withTestEntry(t, dlEntry(srv.URL+"/model.onnx", sha(body)))

	dir := t.TempDir()
	errs := make(chan error, 2)
	go func() { errs <- Pull("test-dl", PullOptions{ModelsDir: dir, Out: &bytes.Buffer{}}) }()
	<-started
	go func() { errs <- Pull("test-dl", PullOptions{ModelsDir: dir, Out: &bytes.Buffer{}}) }()
	for i := 0; i < 2; i++ {
		if err := <-errs; err != nil {
			t.Fatalf("pull: %v", err)
		}
	}
	if n := atomic.LoadInt32(&gets); n != 1 {
		t.Fatalf("the model was downloaded %d times; the second pull should wait and reuse it", n)
	}
}
