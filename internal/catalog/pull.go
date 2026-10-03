package catalog

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
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
	return pull(name, opts, map[string]bool{})
}

// pull is Pull with the set of entries already on the current dependency path, so a cycle in
// the catalog is reported instead of recursing forever.
func pull(name string, opts PullOptions, visiting map[string]bool) error {
	out := opts.Out
	if out == nil {
		out = os.Stderr
	}

	entry, ok := Lookup(name)
	if !ok {
		return UnknownModelError(name)
	}
	if entry.Name != name {
		fmt.Fprintf(out, "%q is an alias of %q\n", name, entry.Name)
	}
	if visiting[entry.Name] {
		return fmt.Errorf("catalog: dependency cycle through %q", entry.Name)
	}
	visiting[entry.Name] = true
	defer delete(visiting, entry.Name)

	// Composed models: pull every missing dependency first, so one `visionserve pull
	// rfdetr-gdino-siglip-etri` is enough (Ollama-style) instead of a chain of "pull X first"
	// errors. A dependency already installed is left alone — --force applies to the named model
	// only, never to shared weights other entries also point at.
	for _, dep := range entry.Dependencies {
		if _, err := os.Stat(filepath.Join(opts.ModelsDir, dep, "manifest.yaml")); err == nil {
			continue
		}
		fmt.Fprintf(out, "%s needs %s — pulling it first\n", entry.Name, dep)
		depOpts := opts
		depOpts.Force = false
		if err := pull(dep, depOpts, visiting); err != nil {
			return fmt.Errorf("pull %s: dependency %s: %w", entry.Name, dep, err)
		}
	}

	if !entry.Verified {
		fmt.Fprintf(out, "WARNING: %q is from an unverified source.\n  %s\n", entry.Name, entry.Note)
	}

	dstDir := filepath.Join(opts.ModelsDir, entry.Name)
	if err := os.MkdirAll(dstDir, 0o755); err != nil {
		return fmt.Errorf("create model dir: %w", err)
	}
	// One pull per model directory at a time: a second concurrent pull of the same model waits,
	// then finds the verified files in place instead of downloading over the first.
	unlock, err := lockDir(dstDir, out, entry.Name)
	if err != nil {
		return fmt.Errorf("pull %s: lock %s: %w", entry.Name, dstDir, err)
	}
	defer unlock()

	// Composed models reference files INSIDE their dependencies. The dependency being installed
	// is not enough: an install made by an older catalog can hold a different filename (the
	// grounding-dino entry once wrote model.onnx, now model-fixedmask.onnx), and the model would
	// then be "pulled" and fail at load with a path error that points nowhere near the cause.
	if err := checkVirtualFiles(entry, dstDir); err != nil {
		return err
	}

	source := "huggingface.co/" + entry.HFRepo
	if len(entry.Dependencies) > 0 {
		source = "local (" + strings.Join(entry.Dependencies, " + ") + ")"
	} else if entry.HFRepo == "" {
		source = "external sources (Google Drive / direct URL)"
	}
	fmt.Fprintf(out, "pulling %s (%s, %s) from %s\n",
		entry.Name, entry.Task, entry.License, source)

	// Virtual models have no files to download — they only need a manifest.
	for _, file := range entry.Files {
		destPath := filepath.Join(dstDir, file.LocalFilename)

		if !opts.Force {
			if info, err := os.Stat(destPath); err == nil && info.Size() >= sizeFloor(file) {
				// A file already on disk is still checked against the catalog digest when
				// there is one. Skipping the check here would make "already present" the
				// one way to get unverified bytes into the registry.
				if err := verifyDigest(destPath, file.SHA256); err != nil {
					return fmt.Errorf("pull %s: %w (re-download with --force)", entry.Name, err)
				}
				fmt.Fprintf(out, "  %s  already present (%s), skipping\n",
					file.LocalFilename, humanBytes(info.Size()))
				continue
			}
		}

		// downloadURL verifies size, HTML sniff and the sha256 pin on the temp file BEFORE it
		// replaces destPath, so a failed (re-)download never destroys a good file and a running
		// server never sees unverified bytes under the real name.
		want := expect{SHA256: file.SHA256, Floor: sizeFloor(file)}
		var url string
		if gdriveID, ok := ParseGDriveURL(file.DirectURL); ok {
			fmt.Fprintf(out, "  downloading %s <- Google Drive\n", file.LocalFilename)
			url = gdriveURL(gdriveID)
		} else if file.DirectURL != "" {
			fmt.Fprintf(out, "  downloading %s <- %s\n", file.LocalFilename, file.DirectURL)
			url = file.DirectURL
		} else {
			fmt.Fprintf(out, "  downloading %s <- %s\n", file.LocalFilename, file.HFFilename)
			url = ResolveURL(entry.HFRepo, file.HFFilename)
		}
		if _, err := downloadURL(url, destPath, want, out); err != nil {
			return fmt.Errorf("pull %s: %w", entry.Name, err)
		}
	}

	// Write embedded labels file (if any), unless it already exists.
	if entry.LabelsFile != "" && entry.EmbeddedLabels != "" {
		labelsPath := filepath.Join(dstDir, entry.LabelsFile)
		if _, err := os.Stat(labelsPath); err != nil || opts.Force {
			if err := writeFileAtomic(labelsPath, []byte(entry.EmbeddedLabels)); err != nil {
				return fmt.Errorf("write labels: %w", err)
			}
			fmt.Fprintf(out, "  wrote %s\n", entry.LabelsFile)
		}
	}

	// Write manifest.yaml, but DO NOT clobber an existing one (a user may have
	// customized it). Regenerate only when missing or --force.
	//
	// A manifest `pull` itself generated is not a customization, though, and keeping a stale one
	// is how a re-pull used to leave the old configuration in force: `pull grounding-dino`
	// downloaded model-fixedmask.onnx next to a manifest still pointing at the defective
	// model.onnx, and an RF-DETR re-pull kept letterbox: true. Those are regenerated in place.
	manifestPath := filepath.Join(dstDir, "manifest.yaml")
	want := entry.RenderManifest()
	existing, statErr := os.ReadFile(manifestPath)
	switch {
	case statErr != nil || opts.Force:
		if err := writeFileAtomic(manifestPath, []byte(want)); err != nil {
			return fmt.Errorf("write manifest: %w", err)
		}
		fmt.Fprintf(out, "  wrote manifest.yaml\n")
	case string(existing) == want:
		fmt.Fprintf(out, "  manifest.yaml up to date\n")
	case isUneditedGenerated(string(existing)):
		if err := writeFileAtomic(manifestPath, []byte(want)); err != nil {
			return fmt.Errorf("write manifest: %w", err)
		}
		fmt.Fprintf(out, "  updated manifest.yaml (generated by an older catalog)\n")
	default:
		fmt.Fprintf(out, "  manifest.yaml was edited by hand, keeping it (use --force to regenerate)\n")
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

// verifyDigest checks a file against the catalog's expected SHA-256. An empty want is a
// no-op: not every entry is pinned yet, and pull must keep working for those. Where a pin
// DOES exist this is the check that makes `visionserve pull` trustworthy — it is also what
// lets the generated manifest carry a sha256 the registry's verified mode can enforce.
func verifyDigest(path, want string) error {
	if want == "" {
		return nil
	}
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("verify %s: %w", path, err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return fmt.Errorf("verify %s: %w", path, err)
	}
	got := hex.EncodeToString(h.Sum(nil))
	if !strings.EqualFold(got, want) {
		return fmt.Errorf("verify %s: sha256 mismatch (catalog pins %s, downloaded bytes are %s)",
			filepath.Base(path), want, got)
	}
	return nil
}

// sizeFloor is the smallest plausible size for a downloaded file. The 1 KiB floor is a cheap
// "this is an error page, not weights" check and only makes sense for weights: a labels file is
// legitimately a few hundred bytes. A side file with a SHA-256 pin is checked by its digest.
func sizeFloor(f File) int64 {
	name := f.LocalFilename
	if strings.HasSuffix(name, ".onnx") || strings.HasSuffix(name, ".data") || f.SHA256 == "" {
		return minSaneSize
	}
	return 1
}

// checkVirtualFiles verifies every sibling path a composed entry's manifest will reference
// (files: roles and a relative labels file) exists, and names the dependency to re-pull if not.
func checkVirtualFiles(e Entry, dstDir string) error {
	paths := make([]string, 0, len(e.VirtualFiles)+1)
	for _, rel := range e.VirtualFiles {
		paths = append(paths, rel)
	}
	if strings.HasPrefix(e.LabelsFile, "../") {
		paths = append(paths, e.LabelsFile)
	}
	sort.Strings(paths)
	for _, rel := range paths {
		if _, err := os.Stat(filepath.Join(dstDir, rel)); err == nil {
			continue
		}
		dep := strings.SplitN(strings.TrimPrefix(rel, "../"), "/", 2)[0]
		return fmt.Errorf("%s needs %s, which is missing — the installed %q predates this catalog; "+
			"re-pull it:\n  visionserve pull %s --force", e.Name, rel, dep, dep)
	}
	return nil
}
