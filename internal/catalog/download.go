package catalog

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"
)

// hfBaseURL is the public HuggingFace "resolve" endpoint. It returns a 302
// redirect to the CDN (which Go's http.Client follows automatically). No auth
// is required for public repos.
const hfBaseURL = "https://huggingface.co"

// ResolveURL builds the download URL for a file inside a public HF repo.
func ResolveURL(repo, filename string) string {
	return fmt.Sprintf("%s/%s/resolve/main/%s", hfBaseURL, repo, filename)
}

// Network timeouts. There is deliberately no TOTAL timeout: a multi-GB weight file on a slow
// link legitimately takes hours. What is bounded is waiting — for the response headers, and for
// the next byte of the body — so a stalled connection fails instead of hanging forever.
// Variables, not constants, so tests can shorten them.
var (
	responseHeaderTimeout = 60 * time.Second
	idleReadTimeout       = 2 * time.Minute
	headTimeout           = 30 * time.Second
)

func newDownloadClient() *http.Client {
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.ResponseHeaderTimeout = responseHeaderTimeout
	return &http.Client{Transport: tr}
}

// expect is what downloaded bytes must satisfy BEFORE they replace anything on disk.
type expect struct {
	SHA256 string // hex digest; "" = not pinned
	Floor  int64  // minimum plausible size in bytes (see sizeFloor)
}

// progressWriter wraps an io.Writer and prints a simple progress line.
type progressWriter struct {
	w        io.Writer
	name     string
	total    int64 // -1 if unknown (no Content-Length)
	done     int64
	lastTick time.Time
	out      io.Writer
}

func (p *progressWriter) Write(b []byte) (int, error) {
	n, err := p.w.Write(b)
	p.done += int64(n)
	// Throttle to avoid spamming the terminal.
	if time.Since(p.lastTick) > 200*time.Millisecond {
		p.print()
		p.lastTick = time.Now()
	}
	return n, err
}

func (p *progressWriter) print() {
	if p.total > 0 {
		pct := float64(p.done) / float64(p.total) * 100
		fmt.Fprintf(p.out, "\r  %s  %s / %s  (%.1f%%)        ",
			p.name, humanBytes(p.done), humanBytes(p.total), pct)
	} else {
		fmt.Fprintf(p.out, "\r  %s  %s        ", p.name, humanBytes(p.done))
	}
}

func (p *progressWriter) finish() {
	p.print()
	fmt.Fprintln(p.out)
}

// headBuffer keeps the first bytes written through it (for the HTML-error-page check).
type headBuffer struct{ b []byte }

func (h *headBuffer) Write(p []byte) (int, error) {
	if room := 64 - len(h.b); room > 0 {
		if len(p) < room {
			room = len(p)
		}
		h.b = append(h.b, p[:room]...)
	}
	return len(p), nil
}

// idleReader cancels the request when no body byte has arrived for d.
type idleReader struct {
	r     io.Reader
	d     time.Duration
	timer *time.Timer
	fired atomic.Bool
}

func newIdleReader(r io.Reader, d time.Duration, cancel func()) *idleReader {
	ir := &idleReader{r: r, d: d}
	ir.timer = time.AfterFunc(d, func() {
		ir.fired.Store(true)
		cancel()
	})
	return ir
}

func (ir *idleReader) Read(p []byte) (int, error) {
	n, err := ir.r.Read(p)
	if n > 0 {
		ir.timer.Reset(ir.d)
	}
	return n, err
}

// downloadURL is the shared streaming implementation used by DownloadFile, DownloadDirect,
// DownloadGDrive and Pull. It streams url into a uniquely named temp file next to destPath,
// hashing as it writes, and checks the result — size against Content-Length, the size floor, the
// HTML-error-page sniff and, when pinned, the SHA-256 — BEFORE renaming it over destPath. A file
// that fails any check never replaces what was there, and concurrent downloads of the same file
// never share a temp file. Large files are never buffered in RAM.
func downloadURL(url, destPath string, want expect, progressOut io.Writer) (int64, error) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return 0, fmt.Errorf("download %s: %w", url, err)
	}
	resp, err := newDownloadClient().Do(req)
	if err != nil {
		return 0, fmt.Errorf("download %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("download %s: unexpected status %s", url, resp.Status)
	}

	dir := filepath.Dir(destPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return 0, err
	}
	f, err := os.CreateTemp(dir, "."+filepath.Base(destPath)+".*.part")
	if err != nil {
		return 0, err
	}
	tmpPath := f.Name()
	done := false
	defer func() {
		if !done {
			f.Close()
			_ = os.Remove(tmpPath)
		}
	}()

	h := sha256.New()
	head := &headBuffer{}
	var dst io.Writer = io.MultiWriter(f, h, head)
	var pw *progressWriter
	if progressOut != nil {
		pw = &progressWriter{
			w:        dst,
			name:     filepath.Base(destPath),
			total:    resp.ContentLength,
			out:      progressOut,
			lastTick: time.Now(),
		}
		dst = pw
	}

	body := newIdleReader(resp.Body, idleReadTimeout, cancel)
	n, err := io.Copy(dst, body)
	body.timer.Stop()
	if pw != nil {
		pw.finish()
	}
	if err != nil {
		if body.fired.Load() {
			err = fmt.Errorf("no data received for %s (connection stalled)", idleReadTimeout)
		}
		return n, fmt.Errorf("download %s: %w", url, err)
	}
	name := filepath.Base(destPath)
	if err := checkDownload(name, n, resp.ContentLength, head.b, hex.EncodeToString(h.Sum(nil)), want); err != nil {
		return n, err
	}

	// Durable before visible: the rename must never expose bytes that are not on disk yet.
	if err := f.Chmod(0o644); err != nil {
		return n, err
	}
	if err := f.Sync(); err != nil {
		return n, fmt.Errorf("download %s: %w", url, err)
	}
	if err := f.Close(); err != nil {
		return n, fmt.Errorf("download %s: %w", url, err)
	}
	if err := os.Rename(tmpPath, destPath); err != nil {
		return n, err
	}
	done = true
	syncDir(dir)
	return n, nil
}

// checkDownload sanity-checks a completed download before it is put in place: non-empty, the
// size the server announced, above the minimum plausible size, not an HTML error page, and — if
// pinned — the expected SHA-256 (computed while streaming, so the bytes are read once).
func checkDownload(name string, n, contentLength int64, head []byte, gotSHA string, want expect) error {
	if n == 0 {
		return fmt.Errorf("verify %s: file is empty", name)
	}
	if contentLength >= 0 && n != contentLength {
		return fmt.Errorf("verify %s: size mismatch (received %d bytes, server announced %d)", name, n, contentLength)
	}
	if n < want.Floor {
		return fmt.Errorf("verify %s: file is suspiciously small (%d bytes)", name, n)
	}
	if looksLikeHTMLError(head) {
		return fmt.Errorf("verify %s: looks like an HTML error page, not a model file", name)
	}
	if want.SHA256 != "" && !strings.EqualFold(gotSHA, want.SHA256) {
		return fmt.Errorf("verify %s: sha256 mismatch (catalog pins %s, downloaded bytes are %s)",
			name, want.SHA256, gotSHA)
	}
	return nil
}

// syncDir fsyncs a directory so a rename into it survives a crash. Best effort: not every
// platform supports syncing a directory handle.
func syncDir(dir string) {
	if d, err := os.Open(dir); err == nil {
		_ = d.Sync()
		_ = d.Close()
	}
}

// writeFileAtomic writes data to path via a temp file + rename, so a reader (a server rescanning
// the registry) never sees a half-written file.
func writeFileAtomic(path string, data []byte) error {
	f, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".*.tmp")
	if err != nil {
		return err
	}
	tmp := f.Name()
	if _, err := f.Write(data); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Chmod(0o644); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}

// DownloadFile streams an HF file to destPath. progressOut receives
// human-readable progress (e.g. os.Stderr); pass nil to disable.
func DownloadFile(repo, hfFilename, destPath string, progressOut io.Writer) (int64, error) {
	return downloadURL(ResolveURL(repo, hfFilename), destPath, expect{Floor: 1}, progressOut)
}

// DownloadDirect streams any HTTPS URL to destPath.
// Use when File.DirectURL holds a full URL (not a gdrive:// URI).
func DownloadDirect(rawURL, destPath string, progressOut io.Writer) (int64, error) {
	return downloadURL(rawURL, destPath, expect{Floor: 1}, progressOut)
}

// DownloadGDrive streams a public Google Drive file to destPath.
// fileID is the alphanumeric Drive file identifier from the share URL.
func DownloadGDrive(fileID, destPath string, progressOut io.Writer) (int64, error) {
	return downloadURL(gdriveURL(fileID), destPath, expect{Floor: 1}, progressOut)
}

// gdriveURL is the download URL for a public Drive file. drive.usercontent.google.com with
// confirm=t bypasses the HTML virus-scan confirmation page served by the older
// drive.google.com/uc endpoint for files larger than ~100 MB.
func gdriveURL(fileID string) string {
	return fmt.Sprintf(
		"https://drive.usercontent.google.com/download?id=%s&export=download&authuser=0&confirm=t",
		fileID,
	)
}

// HeadSize issues a request and returns the Content-Length of an HF file
// without downloading the body. Useful to confirm a URL resolves (e.g. for the
// big grounding-dino model) before committing to a full download.
func HeadSize(repo, hfFilename string) (int64, error) {
	url := ResolveURL(repo, hfFilename)
	// HF's resolve endpoint answers HEAD with the final Content-Length.
	resp, err := (&http.Client{Timeout: headTimeout}).Head(url)
	if err != nil {
		return 0, fmt.Errorf("head %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("head %s: unexpected status %s", url, resp.Status)
	}
	return resp.ContentLength, nil
}

// humanBytes formats a byte count as a human-readable string.
func humanBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for x := n / unit; x >= unit; x /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(n)/float64(div), "KMGTPE"[exp])
}

// looksLikeHTMLError reports whether the first bytes of a file look like an HTML
// error page rather than a binary model (a common failure mode when a repo path
// is wrong and HF returns a 200 HTML page). Used as a sanity check.
func looksLikeHTMLError(head []byte) bool {
	s := strings.ToLower(strings.TrimSpace(string(head)))
	return strings.HasPrefix(s, "<!doctype html") || strings.HasPrefix(s, "<html")
}

// ParseGDriveURL extracts the file ID from a "gdrive://FILE_ID" URI.
// Returns ("", false) if s is not a gdrive URI.
func ParseGDriveURL(s string) (string, bool) {
	if strings.HasPrefix(s, "gdrive://") {
		return strings.TrimPrefix(s, "gdrive://"), true
	}
	return "", false
}
