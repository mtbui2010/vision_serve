package catalog

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// PullOptions controls a pull operation.
type PullOptions struct {
	ModelsDir string    // registry root (e.g. ./models)
	Force     bool      // redownload files even if present
	Out       io.Writer // human-readable log/progress (default os.Stderr)
}

// minSaneSize is the smallest plausible size for a real ONNX weights file.
// Anything smaller almost certainly means an error page or truncated download.
const minSaneSize = 1024 // 1 KiB

// Pull downloads a catalog model into <ModelsDir>/<name>/, writes its
// manifest.yaml and any embedded labels file, and verifies file sizes. It is
// idempotent: files already present are skipped unless Force is set. Existing
// files are never overwritten unless Force is set.
func Pull(name string, opts PullOptions) error {
	out := opts.Out
	if out == nil {
		out = os.Stderr
	}

	entry, ok := Lookup(name)
	if !ok {
		return UnknownModelError(name)
	}

	if !entry.Verified {
		fmt.Fprintf(out, "WARNING: %q is from an unverified source.\n  %s\n", entry.Name, entry.Note)
	}

	dstDir := filepath.Join(opts.ModelsDir, entry.Name)
	if err := os.MkdirAll(dstDir, 0o755); err != nil {
		return fmt.Errorf("create model dir: %w", err)
	}

	source := "huggingface.co/" + entry.HFRepo
	if entry.HFRepo == "" {
		source = "external sources (Google Drive / direct URL)"
	}
	fmt.Fprintf(out, "pulling %s (%s, %s) from %s\n",
		entry.Name, entry.Task, entry.License, source)

	for _, file := range entry.Files {
		destPath := filepath.Join(dstDir, file.LocalFilename)

		if !opts.Force {
			if info, err := os.Stat(destPath); err == nil && info.Size() >= minSaneSize {
				fmt.Fprintf(out, "  %s  already present (%s), skipping\n",
					file.LocalFilename, humanBytes(info.Size()))
				continue
			}
		}

		var n int64
		var dlErr error
		if gdriveID, ok := ParseGDriveURL(file.DirectURL); ok {
			fmt.Fprintf(out, "  downloading %s <- Google Drive\n", file.LocalFilename)
			n, dlErr = DownloadGDrive(gdriveID, destPath, out)
		} else if file.DirectURL != "" {
			fmt.Fprintf(out, "  downloading %s <- %s\n", file.LocalFilename, file.DirectURL)
			n, dlErr = DownloadDirect(file.DirectURL, destPath, out)
		} else {
			fmt.Fprintf(out, "  downloading %s <- %s\n", file.LocalFilename, file.HFFilename)
			n, dlErr = DownloadFile(entry.HFRepo, file.HFFilename, destPath, out)
		}
		if dlErr != nil {
			return fmt.Errorf("pull %s: %w", entry.Name, dlErr)
		}
		if err := verifyFile(destPath, n); err != nil {
			_ = os.Remove(destPath)
			return fmt.Errorf("pull %s: %w", entry.Name, err)
		}
	}

	// Write embedded labels file (if any), unless it already exists.
	if entry.LabelsFile != "" && entry.EmbeddedLabels != "" {
		labelsPath := filepath.Join(dstDir, entry.LabelsFile)
		if _, err := os.Stat(labelsPath); err != nil || opts.Force {
			if err := os.WriteFile(labelsPath, []byte(entry.EmbeddedLabels), 0o644); err != nil {
				return fmt.Errorf("write labels: %w", err)
			}
			fmt.Fprintf(out, "  wrote %s\n", entry.LabelsFile)
		}
	}

	// Write manifest.yaml, but DO NOT clobber an existing one (a user may have
	// customized it). Regenerate only when missing or --force.
	manifestPath := filepath.Join(dstDir, "manifest.yaml")
	if _, err := os.Stat(manifestPath); err != nil || opts.Force {
		if err := os.WriteFile(manifestPath, []byte(entry.RenderManifest()), 0o644); err != nil {
			return fmt.Errorf("write manifest: %w", err)
		}
		fmt.Fprintf(out, "  wrote manifest.yaml\n")
	} else {
		fmt.Fprintf(out, "  manifest.yaml already present, keeping it\n")
	}

	// Use the absolute path so the message is unambiguous regardless of cwd.
	absDstDir, err := filepath.Abs(dstDir)
	if err != nil {
		absDstDir = dstDir
	}
	fmt.Fprintf(out, "success: %s is ready in %s\n", entry.Name, absDstDir)
	fmt.Fprintf(out, "  run it with:  visionserve run %s <image>\n", entry.Name)
	return nil
}

// verifyFile sanity-checks a downloaded file: non-empty, above the minimum
// plausible size, and not an HTML error page.
func verifyFile(path string, reported int64) error {
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("verify %s: %w", path, err)
	}
	if info.Size() == 0 {
		return fmt.Errorf("verify %s: file is empty", path)
	}
	if info.Size() < minSaneSize {
		return fmt.Errorf("verify %s: file is suspiciously small (%d bytes)", path, info.Size())
	}
	if reported > 0 && info.Size() != reported {
		return fmt.Errorf("verify %s: size mismatch (wrote %d, expected %d)", path, info.Size(), reported)
	}
	// Peek the first bytes to reject HTML error pages saved as binaries.
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	head := make([]byte, 64)
	n, _ := io.ReadFull(f, head)
	if looksLikeHTMLError(head[:n]) {
		return fmt.Errorf("verify %s: looks like an HTML error page, not a model file", path)
	}
	return nil
}
