package catalog

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
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

// DownloadFile streams an HF file to destPath. It downloads to a ".part" temp
// file first and renames on success (so a partial/interrupted download never
// looks complete). Large files (hundreds of MB) are streamed, never buffered
// in memory. progressOut receives the human-readable progress line (e.g.
// os.Stderr); pass nil to disable.
func DownloadFile(repo, hfFilename, destPath string, progressOut io.Writer) (int64, error) {
	url := ResolveURL(repo, hfFilename)

	client := &http.Client{
		// Generous timeout for very large files (e.g. 719MB grounding-dino) on
		// slow links; 0 would mean no timeout but we keep a high ceiling.
		Timeout: 2 * time.Hour,
	}
	resp, err := client.Get(url)
	if err != nil {
		return 0, fmt.Errorf("download %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("download %s: unexpected status %s", url, resp.Status)
	}

	if err := os.MkdirAll(filepath.Dir(destPath), 0o755); err != nil {
		return 0, err
	}

	tmpPath := destPath + ".part"
	f, err := os.Create(tmpPath)
	if err != nil {
		return 0, err
	}

	var dst io.Writer = f
	var pw *progressWriter
	if progressOut != nil {
		pw = &progressWriter{
			w:        f,
			name:     filepath.Base(destPath),
			total:    resp.ContentLength,
			out:      progressOut,
			lastTick: time.Now(),
		}
		dst = pw
	}

	n, err := io.Copy(dst, resp.Body)
	if pw != nil {
		pw.finish()
	}
	if cerr := f.Close(); cerr != nil && err == nil {
		err = cerr
	}
	if err != nil {
		_ = os.Remove(tmpPath)
		return n, fmt.Errorf("download %s: %w", url, err)
	}

	if err := os.Rename(tmpPath, destPath); err != nil {
		_ = os.Remove(tmpPath)
		return n, err
	}
	return n, nil
}

// HeadSize issues a request and returns the Content-Length of an HF file
// without downloading the body. Useful to confirm a URL resolves (e.g. for the
// big grounding-dino model) before committing to a full download.
func HeadSize(repo, hfFilename string) (int64, error) {
	url := ResolveURL(repo, hfFilename)
	// HF's resolve endpoint answers HEAD with the final Content-Length.
	resp, err := http.Head(url)
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
