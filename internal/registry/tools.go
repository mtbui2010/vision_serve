package registry

// Helpers for the offline model tools (`visionserve inspect`, `visionserve import`): the same
// rules the registry enforces at load, exposed so a tool can report or apply them before a
// manifest exists. None of them relaxes a rule.

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"

	"visionserve/pkg/api"
)

// CheckLicense resolves a declared licence id to its canonical SPDX spelling, with the error the
// registry gives a manifest declaring it (validate) when it is not on the permissive allowlist.
func CheckLicense(declared string) (canonical string, err error) {
	canon, ok := canonicalLicense(declared)
	if !ok {
		return "", fmt.Errorf("license %q is not allowed — only permissive licenses accepted (Apache-2.0/MIT/BSD); AGPL is strictly forbidden", declared)
	}
	return canon, nil
}

// AllowedLicenses lists the canonical SPDX ids of the allowlist, sorted.
func AllowedLicenses() []string {
	out := make([]string, 0, len(licenseAllowlist))
	for _, c := range licenseAllowlist {
		out = append(out, c)
	}
	sort.Strings(out)
	return out
}

// ValidTask reports whether task is a task a manifest may declare; ValidTasks lists them sorted.
func ValidTask(task string) bool { return validTasks[api.Task(task)] }

// ValidTasks lists the tasks a manifest may declare, sorted.
func ValidTasks() []string {
	out := make([]string, 0, len(validTasks))
	for t := range validTasks {
		out = append(out, string(t))
	}
	sort.Strings(out)
	return out
}

// ValidName reports whether name is a valid model name (it becomes a directory name).
func ValidName(name string) bool { return validModelName.MatchString(name) && len(name) <= 128 }

// For returns the sha256 the manifest pins for a files: role ("" = the single model_file), and
// whether one is pinned.
func (s SHA256Field) For(role string) (digest string, ok bool) { return s.expectedFor(role) }

// FileSHA256 is the SHA-256 (hex) of the file at path, through the same cache the load-time
// verification uses (a file hashed once is not hashed again while unchanged).
func FileSHA256(path string) (string, error) { return weightDigests.sha256(path) }

// Put adds (or replaces) a manifest in the registry under its name, as a scan would have. For a
// tool that inspects one model directory outside any registry; a later Scan replaces it.
func (r *Registry) Put(m *Manifest) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.byName[m.Name] = &Entry{Manifest: m, Dir: m.Dir()}
}

// agplMarkers are byte strings an AGPL (Ultralytics / YOLO-World / FastSAM) export carries in its
// own metadata ("license: AGPL-3.0 License (https://ultralytics.com/license)"), lowercased. They
// mirror _STRONG_BYTES of the Python converter's licence scan (clients/python/visionserve/convert/
// common.py); TestAGPLMarkersMatchPython keeps the two equal.
var agplMarkers = [][]byte{[]byte("agpl-3.0"), []byte("ultralytics.com/license"), []byte("https://ultralytics.com")}

// agplScanWindow is how much of each end of the file is scanned, as the Python converter does
// (_SCAN_HEAD): an export's metadata sits at the start or the end, never inside the weights.
const agplScanWindow = 8 << 20

// ErrAGPLModel marks a model file whose content declares an AGPL origin.
var ErrAGPLModel = errors.New("AGPL model")

// ScanAGPLMarkers refuses a model file whose first or last 8 MiB carry an AGPL licence marker of
// an Ultralytics / YOLO-World / FastSAM export (agplMarkers), whatever licence a manifest or a
// flag declares for it — the content gate of the converter's license_scan, for a file that did
// not go through the converter. Reads at most 16 MiB. The error wraps ErrAGPLModel.
func ScanAGPLMarkers(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return err
	}
	size := st.Size()
	head := make([]byte, min(size, agplScanWindow))
	if _, err := io.ReadFull(f, head); err != nil {
		return err
	}
	buf := head
	if size > int64(len(head)) {
		tailStart := max(int64(len(head)), size-agplScanWindow)
		tail := make([]byte, size-tailStart)
		if _, err := f.ReadAt(tail, tailStart); err != nil && err != io.EOF {
			return err
		}
		// Overlap the boundary so a marker split across head and tail of a file just over the
		// window is still found.
		buf = append(head[len(head)-min(len(head), 64):len(head):len(head)], tail...)
		if m := findMarker(head); m != "" {
			return agplError(path, m)
		}
	}
	if m := findMarker(buf); m != "" {
		return agplError(path, m)
	}
	return nil
}

func findMarker(b []byte) string {
	low := bytes.ToLower(b)
	for _, m := range agplMarkers {
		if bytes.Contains(low, m) {
			return string(m)
		}
	}
	return ""
}

func agplError(path, marker string) error {
	return fmt.Errorf("%w: %s contains %q: this is an Ultralytics / YOLO-World / FastSAM (AGPL) model and cannot be "+
		"served by VisionServe whatever license is declared for it (CLAUDE.md rule 1)", ErrAGPLModel, filepath.Base(path), marker)
}
