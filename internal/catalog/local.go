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
// On success every file under srcDir is copied to <ModelsDir>/<manifest.name>/.
// An existing model of the same name is only overwritten when opts.Force is set.
func InstallLocal(srcDir string, opts PullOptions) error {
	out := opts.Out
	if out == nil {
		out = os.Stderr
	}

	absSrc, err := filepath.Abs(srcDir)
	if err != nil {
		return fmt.Errorf("resolve %s: %w", srcDir, err)
	}
	info, err := os.Stat(absSrc)
	if err != nil {
		return fmt.Errorf("local model folder: %w", err)
	}
	if !info.IsDir() {
		return fmt.Errorf("%s is not a directory (pull a folder containing manifest.yaml + .onnx)", srcDir)
	}

	manifestPath := filepath.Join(absSrc, "manifest.yaml")
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
	if m.Labels != "" {
		if err := ensureInside("labels", m.Labels); err != nil {
			return err
		}
	}
	if !m.WeightsExist() {
		return fmt.Errorf("missing ONNX weights in %s — referenced file(s) %v not found on disk", srcDir, refs)
	}

	dstDir := filepath.Join(opts.ModelsDir, m.Name)
	absDst, _ := filepath.Abs(dstDir)
	if absSrc == absDst {
		fmt.Fprintf(out, "%s is already in the registry at %s\n", m.Name, dstDir)
		return nil
	}
	if _, err := os.Stat(dstDir); err == nil && !opts.Force {
		return fmt.Errorf("model %q already exists at %s (use --force to overwrite)", m.Name, dstDir)
	}

	if err := os.MkdirAll(dstDir, 0o755); err != nil {
		return fmt.Errorf("create model dir: %w", err)
	}
	n, err := copyTree(absSrc, dstDir)
	if err != nil {
		return fmt.Errorf("copy model files: %w", err)
	}

	fmt.Fprintf(out, "installed %s (%s, %s) -> %s  [%d files]\n", m.Name, m.Task, m.License, dstDir, n)
	fmt.Fprintf(out, "  run it with:  visionserve run %s <image>\n", m.Name)
	return nil
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

// copyTree recursively copies the regular files under src into dst (preserving
// sub-directory structure) and returns the number of files copied.
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
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		if !d.Type().IsRegular() {
			return nil // skip symlinks/devices — a model folder should be plain files
		}
		if err := copyFile(path, target); err != nil {
			return err
		}
		count++
		return nil
	})
	return count, err
}

// copyFile copies a single regular file, creating parent directories as needed.
func copyFile(src, dst string) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}
