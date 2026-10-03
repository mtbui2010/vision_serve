package catalog

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"visionserve/internal/models"
	"visionserve/internal/registry"
)

// InstallLocal validates a local model folder and copies it into the registry as
// a new model — the "Modelfile for CV" equivalent of `ollama create`. It is what
// `visionserve pull <path/to/folder>` dispatches to when the argument is a local
// directory instead of a catalog model name.
//
// Validation is STATIC only (no ONNX session is opened):
//   - manifest.yaml parses and passes registry validation (permissive license,
//     valid task / input dims / EP fallback chain);
//   - the architecture maps to a registered factory (a fine-tuned model MUST
//     reuse an existing architecture, e.g. architecture: rf-detr), otherwise the
//     model could never load;
//   - all ONNX file(s) (and the labels file, if any) referenced by the manifest
//     exist inside the folder and do not escape it (so the copy is self-contained).
//
// On success every file under srcDir is copied to <ModelsDir>/<manifest.name>/. Symlinks are
// followed and their CONTENT copied (a HuggingFace cache snapshot is a folder of symlinks into a
// blob store). The copy is staged in a hidden directory next to the destination and renamed into
// place only once it is complete, so a failed install leaves nothing behind and a failed --force
// leaves the previous install intact. An existing model of the same name is only replaced when
// opts.Force is set. The source is never written to. Staging directories that an earlier, killed
// install left in the registry are removed first (cleanStaleStaging).
func InstallLocal(srcDir string, opts PullOptions) error {
	out := opts.Out
	if out == nil {
		out = os.Stderr
	}

	absSrc, err := filepath.Abs(srcDir)
	if err != nil {
		return fmt.Errorf("resolve %s: %w", srcDir, err)
	}
	// Resolve symlinks once, up front: every comparison below (same directory? registry inside
	// the source?) is about the directories the paths really name, and a symlinked folder must
	// be walked as the directory it points at.
	realSrc, err := filepath.EvalSymlinks(absSrc)
	if err != nil {
		return fmt.Errorf("local model folder: %w", err)
	}
	srcInfo, err := os.Stat(realSrc)
	if err != nil {
		return fmt.Errorf("local model folder: %w", err)
	}
	if !srcInfo.IsDir() {
		return fmt.Errorf("%s is not a directory (pull a folder containing manifest.yaml + .onnx)", srcDir)
	}

	manifestPath := filepath.Join(realSrc, "manifest.yaml")
	if _, err := os.Stat(manifestPath); err != nil {
		return fmt.Errorf("no manifest.yaml in %s — a local model folder must contain a manifest.yaml (see docs/manifest-spec.md)", srcDir)
	}

	// Structural validation: permissive license, valid task, input dims, EP chain.
	m, err := registry.LoadManifest(manifestPath)
	if err != nil {
		return err // already a clear "registry: invalid manifest ..." message
	}

	// The architecture must resolve to a registered factory, otherwise the model
	// can never be loaded even though the manifest is structurally valid.
	arch := m.ArchOrName()
	if !models.IsRegistered(arch) {
		return fmt.Errorf("architecture %q is not a built-in model type — a fine-tuned model must reuse an existing architecture (e.g. architecture: rf-detr).\n  valid architectures: %v",
			arch, models.Registered())
	}

	// Referenced ONNX file(s) + labels must live inside the folder so the copy is
	// self-contained. Reject absolute paths and ".." escapes.
	var refs []string
	if len(m.Files) > 0 {
		for role, rel := range m.Files {
			if err := ensureInside("files."+role, rel); err != nil {
				return err
			}
			refs = append(refs, rel)
		}
	} else {
		if err := ensureInside("model_file", m.ModelFile); err != nil {
			return err
		}
		refs = append(refs, m.ModelFile)
	}
	if !m.WeightsExist() {
		return fmt.Errorf("missing ONNX weights in %s — referenced file(s) %v not found on disk", srcDir, refs)
	}
	if m.Labels != "" {
		if err := ensureInside("labels", m.Labels); err != nil {
			return err
		}
		refs = append(refs, m.Labels)
	}

	if err := os.MkdirAll(opts.ModelsDir, 0o755); err != nil {
		return fmt.Errorf("create models dir: %w", err)
	}
	realModels, err := filepath.EvalSymlinks(opts.ModelsDir)
	if err != nil {
		return fmt.Errorf("resolve models dir: %w", err)
	}
	// Staging leftovers of earlier installs that were killed mid-copy or mid-swap.
	cleanStaleStaging(realModels, out)
	dstDir := filepath.Join(opts.ModelsDir, m.Name) // as the user named it, for messages
	dst := filepath.Join(realModels, m.Name)        // what is actually operated on

	dstExists := false
	if dstInfo, err := os.Stat(dst); err == nil {
		// Same directory, however it was reached (a symlinked registry, `pull .` from inside the
		// installed model): there is nothing to copy, and copying would read and write the same
		// files. That holds with --force too — it must never write into the source.
		if os.SameFile(srcInfo, dstInfo) {
			fmt.Fprintf(out, "%s is already in the registry at %s\n", m.Name, dstDir)
			return nil
		}
		if !opts.Force {
			return fmt.Errorf("model %q already exists at %s (use --force to overwrite)", m.Name, dstDir)
		}
		dstExists = true
		if realDst, err := filepath.EvalSymlinks(dst); err == nil && isWithin(realSrc, realDst) {
			return fmt.Errorf("the source folder %s is inside the installed model %s it would replace — "+
				"copy it somewhere else first", srcDir, dstDir)
		}
	}
	// A registry inside the source folder (`pull . --models ./models` run from the model folder)
	// would be copied into itself: the walk descends into the copy it is writing, forever.
	if isWithin(realModels, realSrc) {
		return fmt.Errorf("the model registry %s is inside the source folder %s — the copy would recurse "+
			"into itself; use a registry outside the model folder", opts.ModelsDir, srcDir)
	}

	// Stage next to the destination (same filesystem, so the final rename is atomic).
	stage, err := os.MkdirTemp(realModels, ".tmp-"+m.Name+"-")
	if err != nil {
		return fmt.Errorf("create staging dir: %w", err)
	}
	staged := false // true once stage has been renamed to dst
	defer func() {
		if !staged {
			_ = os.RemoveAll(stage)
		}
	}()

	n, err := copyTree(realSrc, stage)
	if err != nil {
		return fmt.Errorf("copy model files: %w", err)
	}
	for _, rel := range refs {
		st, err := os.Lstat(filepath.Join(stage, rel))
		if err != nil || !st.Mode().IsRegular() {
			return fmt.Errorf("copy model files: %s (referenced by manifest.yaml) was not copied as a regular file — "+
				"is it a symlink to a directory?", rel)
		}
	}

	// Swap: move the old install aside, move the new one in, publish its manifest, drop the old.
	aside := ""
	if dstExists {
		aside = stage + "-old"
		if err := moveAside(dst, aside); err != nil {
			return fmt.Errorf("move previous install aside: %w", err)
		}
	}
	restore := func() {
		if aside != "" {
			if err := os.Rename(aside, dst); err != nil {
				fmt.Fprintf(out, "WARNING: could not restore the previous install from %s: %v\n", aside, err)
			}
		}
	}
	if err := os.Rename(stage, dst); err != nil {
		restore()
		return fmt.Errorf("install %s: %w", dstDir, err)
	}
	staged = true
	if err := os.Rename(filepath.Join(dst, stagedManifest), filepath.Join(dst, "manifest.yaml")); err != nil {
		_ = os.RemoveAll(dst)
		restore()
		return fmt.Errorf("install %s: %w", dstDir, err)
	}
	if aside != "" {
		if err := os.RemoveAll(aside); err != nil {
			fmt.Fprintf(out, "WARNING: could not remove the previous install at %s: %v\n", aside, err)
		}
	}

	fmt.Fprintf(out, "installed %s (%s, %s) -> %s  [%d files]\n", m.Name, m.Task, m.License, dstDir, n)
	fmt.Fprintf(out, "  run it with:  visionserve run %s <image>\n", m.Name)
	return nil
}

// stagedManifest is the name manifest.yaml carries inside a staging directory. The registry scan
// recognises a model by <dir>/manifest.yaml, so a running server that rescans mid-install must not
// see the staging directory as a model; the manifest gets its real name only after the rename.
const stagedManifest = ".manifest.yaml.staged"

// isWithin reports whether path is parent itself or lies under it. Both must be cleaned,
// symlink-resolved absolute paths.
func isWithin(path, parent string) bool {
	rel, err := filepath.Rel(parent, path)
	if err != nil {
		return false
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && !filepath.IsAbs(rel)
}

// ensureInside rejects a manifest-referenced path that is absolute or escapes the
// model folder via "..", so a local install always copies a self-contained model.
func ensureInside(field, rel string) error {
	if rel == "" {
		return nil
	}
	if filepath.IsAbs(rel) {
		return fmt.Errorf("%s %q must be a relative path inside the model folder", field, rel)
	}
	if cleaned := filepath.Clean(rel); cleaned == ".." || strings.HasPrefix(cleaned, ".."+string(filepath.Separator)) {
		return fmt.Errorf("%s %q points outside the model folder; a local model must be self-contained", field, rel)
	}
	return nil
}

// copyTree recursively copies the files under src into the (fresh, empty) dst, preserving
// sub-directory structure, and returns the number of files copied. A symlink to a file is copied
// as the file's CONTENT; a symlink to a directory is refused (following those risks cycles); a
// broken symlink is an error. Other special files (devices, sockets) are skipped. The top-level
// manifest.yaml is written as stagedManifest (see there).
func copyTree(src, dst string) (int, error) {
	count := 0
	err := filepath.WalkDir(src, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)
		if rel == "manifest.yaml" {
			target = filepath.Join(dst, stagedManifest)
		}
		switch {
		case d.IsDir():
			return os.MkdirAll(target, 0o755)
		case d.Type().IsRegular():
		case d.Type()&os.ModeSymlink != 0:
			st, err := os.Stat(path)
			if err != nil {
				return fmt.Errorf("%s is a broken symlink: %w", rel, err)
			}
			if st.IsDir() {
				return fmt.Errorf("%s is a symlink to a directory, which is not copied — replace it with a real directory", rel)
			}
			if !st.Mode().IsRegular() {
				return nil
			}
		default:
			return nil // devices, sockets, pipes — a model folder should be plain files
		}
		if err := copyFile(path, target); err != nil {
			return err
		}
		// Keep the staging root's mtime fresh: a file copied into a sub-directory does not
		// update it, and cleanStaleStaging judges liveness by it.
		touchDir(dst)
		count++
		return nil
	})
	return count, err
}

// copyFile copies a single file's content (following a symlink at src), creating parent
// directories as needed. dst must not exist yet: O_EXCL guarantees a copy can never truncate an
// existing file, in particular the one it is reading from.
func copyFile(src, dst string) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	if err := out.Sync(); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}
