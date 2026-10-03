package catalog

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
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
// DownloadGDrive and Pull. It streams url into a temp file next to destPath, hashing as it
// writes, and checks the result — size against what the server announced, the size floor, the
// HTML-error-page sniff and, when pinned, the SHA-256 — BEFORE renaming it over destPath. A file
// that fails any check never replaces what was there. Large files are never buffered in RAM.
//
// The temp file is uniquely named, so concurrent downloads of the same file never share one, and
// it is removed when the download fails. Pull, which holds the per-model lock, uses
// downloadResumable instead.
func downloadURL(url, destPath string, want expect, progressOut io.Writer) (int64, error) {
	return download(url, destPath, want, progressOut, false)
}

// downloadResumable is downloadURL with a temp file that survives an interrupted transfer: it has
// a fixed name (partialPath), is KEPT when the connection drops, and the next call continues it
// with an HTTP Range request when the server supports one (206); a server that ignores the Range
// (200) or refuses it (416) gets a fresh download. The bytes already on disk are hashed before
// the request, so the SHA-256 pin covers the whole file exactly as for a fresh download; a resumed
// file that fails verification is discarded and fetched once more from zero.
//
// The caller must hold the per-model lock (a fixed temp name is only safe for one writer) and
// should pass a pinned want.SHA256: without a pin nothing would notice a prefix that came from a
// different upstream version than the rest. Pull resumes only pinned files.
func downloadResumable(url, destPath string, want expect, progressOut io.Writer) (int64, error) {
	return download(url, destPath, want, progressOut, true)
}

// partialPath is where downloadResumable keeps an unfinished download of destPath.
func partialPath(destPath string) string {
	return filepath.Join(filepath.Dir(destPath), "."+filepath.Base(destPath)+partialSuffix)
}

const partialSuffix = ".partial"

// errRestart: the server would not continue the partial file (416, or a 206 for another range);
// the caller discards it and downloads from zero.
var errRestart = errors.New("server cannot resume this download")

// tempFile is a download in progress: the file, the running hash of its whole content and its
// first bytes (for the HTML sniff).
type tempFile struct {
	f      *os.File
	path   string
	h      hash.Hash
	head   headBuffer
	n      int64 // bytes in the file
	resets int   // times reset() emptied it
}

func (t *tempFile) Write(p []byte) (int, error) {
	n, err := t.f.Write(p)
	t.h.Write(p[:n])
	t.head.Write(p[:n])
	t.n += int64(n)
	return n, err
}

// reset empties the file (a fresh download from zero).
func (t *tempFile) reset() error {
	if err := t.f.Truncate(0); err != nil {
		return err
	}
	if _, err := t.f.Seek(0, io.SeekStart); err != nil {
		return err
	}
	t.h.Reset()
	t.head = headBuffer{}
	t.n = 0
	t.resets++
	return nil
}

// openPartial opens (or creates) the resumable temp file at path and hashes what it already
// holds, leaving the offset at its end. A non-regular file at path (a symlink, a directory) is
// never followed: it is removed (a symlink) or refused.
func openPartial(path string) (*tempFile, error) {
	if st, err := os.Lstat(path); err == nil && !st.Mode().IsRegular() {
		if st.Mode()&os.ModeSymlink == 0 {
			return nil, fmt.Errorf("%s exists and is not a regular file", path)
		}
		if err := os.Remove(path); err != nil {
			return nil, err
		}
	}
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return nil, err
	}
	t := &tempFile{f: f, path: path, h: sha256.New()}
	// Hash the prefix through the same writer path a download uses (minus the file itself).
	n, err := io.Copy(io.MultiWriter(t.h, &t.head), f)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("read partial download %s: %w", path, err)
	}
	t.n = n
	return t, nil
}

func download(url, destPath string, want expect, progressOut io.Writer, resume bool) (int64, error) {
	dir := filepath.Dir(destPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return 0, err
	}
	name := filepath.Base(destPath)

	var tf *tempFile
	if resume {
		var err error
		if tf, err = openPartial(partialPath(destPath)); err != nil {
			return 0, err
		}
	} else {
		f, err := os.CreateTemp(dir, "."+name+".*.part")
		if err != nil {
			return 0, err
		}
		tf = &tempFile{f: f, path: f.Name(), h: sha256.New()}
	}
	// keep: leave the temp file for the next attempt to resume (an interrupted transfer of a
	// resumable download). Anything else that fails removes it.
	done, keep := false, false
	defer func() {
		if !done {
			tf.f.Close()
			if !keep {
				_ = os.Remove(tf.path)
			}
		}
	}()

	for attempt := 0; ; attempt++ {
		resumedFrom, resets := tf.n, tf.resets
		if resumedFrom > 0 && progressOut != nil {
			fmt.Fprintf(progressOut, "  resuming %s from %s\n", name, humanBytes(resumedFrom))
		}
		total, err := fetchInto(url, tf, name, progressOut)
		// The file still starts with bytes from an earlier run (the server honoured the Range).
		prefixKept := resumedFrom > 0 && tf.resets == resets
		if errors.Is(err, errRestart) && attempt == 0 {
			if progressOut != nil {
				fmt.Fprintf(progressOut, "  %s: the server cannot resume (%v) — downloading from zero\n", name, err)
			}
			if err := tf.reset(); err != nil {
				return tf.n, err
			}
			continue
		}
		if err != nil {
			keep = resume
			return tf.n, fmt.Errorf("download %s: %w", url, err)
		}
		if total >= 0 && tf.n < total {
			// The body ended early without a transport error: the rest can still be fetched.
			keep = resume
			return tf.n, fmt.Errorf("download %s: connection closed after %d of %d bytes", url, tf.n, total)
		}
		err = checkDownload(name, tf.n, total, tf.head.b, hex.EncodeToString(tf.h.Sum(nil)), want)
		if err != nil && prefixKept && attempt == 0 {
			// The prefix came from an earlier run (or an earlier upstream version): do not trust
			// it, fetch the whole file once more before giving up.
			if progressOut != nil {
				fmt.Fprintf(progressOut, "  %s: resumed download failed verification (%v) — downloading from zero\n", name, err)
			}
			if err := tf.reset(); err != nil {
				return tf.n, err
			}
			continue
		}
		if err != nil {
			return tf.n, err
		}
		break
	}

	// Durable before visible: the rename must never expose bytes that are not on disk yet.
	if err := tf.f.Chmod(0o644); err != nil {
		return tf.n, err
	}
	if err := tf.f.Sync(); err != nil {
		return tf.n, fmt.Errorf("download %s: %w", url, err)
	}
	if err := tf.f.Close(); err != nil {
		return tf.n, fmt.Errorf("download %s: %w", url, err)
	}
	if err := os.Rename(tf.path, destPath); err != nil {
		return tf.n, err
	}
	done = true
	syncDir(dir)
	return tf.n, nil
}

// fetchInto issues one GET for url and appends the body to tf. When tf already holds bytes it asks
// for the rest with a Range header. It returns the size the server announced for the WHOLE file
// (-1 when unknown).
func fetchInto(url string, tf *tempFile, name string, progressOut io.Writer) (int64, error) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return -1, err
	}
	offset := tf.n
	if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	}
	resp, err := newDownloadClient().Do(req)
	if err != nil {
		return -1, err
	}
	defer resp.Body.Close()

	total := resp.ContentLength
	switch {
	case resp.StatusCode == http.StatusPartialContent && offset > 0:
		start, size, ok := parseContentRange(resp.Header.Get("Content-Range"))
		if !ok || start != offset {
			return -1, fmt.Errorf("%w: asked for bytes %d-, got Content-Range %q", errRestart, offset,
				resp.Header.Get("Content-Range"))
		}
		switch {
		case size >= 0:
			total = size
		case resp.ContentLength >= 0:
			total = offset + resp.ContentLength
		default:
			total = -1
		}
	case resp.StatusCode == http.StatusRequestedRangeNotSatisfiable && offset > 0:
		return -1, fmt.Errorf("%w: %s", errRestart, resp.Status)
	case resp.StatusCode == http.StatusOK:
		if offset > 0 { // the server ignored the Range: what follows is the whole file
			if progressOut != nil {
				fmt.Fprintf(progressOut, "  %s: the server does not support resuming — downloading from zero\n", name)
			}
			if err := tf.reset(); err != nil {
				return -1, err
			}
		}
	default:
		return -1, fmt.Errorf("unexpected status %s", resp.Status)
	}

	var dst io.Writer = tf
	var pw *progressWriter
	if progressOut != nil {
		pw = &progressWriter{w: tf, name: name, total: total, done: tf.n, out: progressOut, lastTick: time.Now()}
		dst = pw
	}
	body := newIdleReader(resp.Body, idleReadTimeout, cancel)
	_, err = io.Copy(dst, body)
	body.timer.Stop()
	if pw != nil {
		pw.finish()
	}
	if err != nil {
		if body.fired.Load() {
			err = fmt.Errorf("no data received for %s (connection stalled)", idleReadTimeout)
		}
		return total, err
	}
	return total, nil
}

// parseContentRange parses a 206 response's "bytes <start>-<end>/<size>" (size may be "*",
// returned as -1). ok=false for anything else.
func parseContentRange(v string) (start, size int64, ok bool) {
	rest, found := strings.CutPrefix(strings.TrimSpace(v), "bytes ")
	if !found {
		return 0, 0, false
	}
	rng, sz, found := strings.Cut(rest, "/")
	if !found {
		return 0, 0, false
	}
	s, e, found := strings.Cut(rng, "-")
	if !found {
		return 0, 0, false
	}
	start, err1 := strconv.ParseInt(strings.TrimSpace(s), 10, 64)
	end, err2 := strconv.ParseInt(strings.TrimSpace(e), 10, 64)
	if err1 != nil || err2 != nil || start < 0 || end < start {
		return 0, 0, false
	}
	size = -1
	if sz = strings.TrimSpace(sz); sz != "*" {
		n, err := strconv.ParseInt(sz, 10, 64)
		if err != nil || n <= end {
			return 0, 0, false
		}
		size = n
	}
	return start, size, true
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
