// Package registry scans the models/ directory, reads + validates manifests, and lists
// available models. It does NOT load models into memory — that is lifecycle's job.
package registry

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// Entry is a model present in the registry (a validated manifest).
type Entry struct {
	Manifest *Manifest
	Dir      string
}

// Registry holds a map name -> Entry, scanned from a root directory (e.g. ./models).
type Registry struct {
	root   string
	mu     sync.RWMutex
	byName map[string]*Entry

	refreshMu   sync.Mutex
	lastRefresh time.Time // last Refresh that rescanned (see Refresh)
}

// New creates a registry pointing at the root directory that holds the models.
func New(root string) *Registry {
	return &Registry{root: root, byName: map[string]*Entry{}}
}

// Scan scans root/*/manifest.yaml and validates each one. A broken manifest (e.g. missing
// ONNX file, forbidden license) is SKIPPED + collected into the returned error list, without aborting the scan.
// A manifest with keys no field reads (Manifest.UnknownKeys) is loaded, and also gets a warning
// in that list.
// The root directory is created automatically if it does not yet exist.
func (r *Registry) Scan() ([]error, error) {
	if err := os.MkdirAll(r.root, 0o755); err != nil {
		return nil, fmt.Errorf("registry: cannot create models directory %s: %w", r.root, err)
	}
	entries, err := os.ReadDir(r.root)
	if err != nil {
		return nil, fmt.Errorf("registry: failed to read directory %s: %w", r.root, err)
	}
	found := map[string]*Entry{}
	var warns []error
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		// Dot-directories are never models: they are install staging (.tmp-<name>-*), the
		// previous version moved aside during a --force swap (.tmp-<name>-*-old, which still holds
		// a manifest with the SAME name), converter backups (.convert-backup) and dry runs.
		if strings.HasPrefix(e.Name(), ".") {
			continue
		}
		mpath := filepath.Join(r.root, e.Name(), "manifest.yaml")
		if _, statErr := os.Stat(mpath); statErr != nil {
			continue // directory has no manifest -> skip
		}
		m, mErr := LoadManifest(mpath)
		if mErr != nil {
			warns = append(warns, mErr)
			continue
		}
		if unknown := m.UnknownKeys(); len(unknown) > 0 {
			// Loaded anyway (a third-party manifest may carry extra keys), but a typo of a real
			// key would silently keep its default, so say which keys were ignored.
			warns = append(warns, fmt.Errorf("registry: manifest %s: unknown key(s) ignored, check for a typo: %s",
				mpath, strings.Join(unknown, ", ")))
		}
		found[m.Name] = &Entry{Manifest: m, Dir: filepath.Join(r.root, e.Name())}
	}
	r.mu.Lock()
	r.byName = found
	r.mu.Unlock()
	return warns, nil
}

// Get returns the Entry for a model name.
// Refresh rescans the directory when the last Refresh rescan is at least minInterval old, and
// reports whether it did. A server lists models with it, so a model installed while the server
// runs (pull, convert, import, a copied folder) shows up in the list at once — a client that
// checks the list before its first request no longer misses it. The interval bounds the cost
// (a ReadDir and one YAML parse per model) when a client polls the list.
func (r *Registry) Refresh(minInterval time.Duration) bool {
	r.refreshMu.Lock()
	now := time.Now()
	if !r.lastRefresh.IsZero() && now.Sub(r.lastRefresh) < minInterval {
		r.refreshMu.Unlock()
		return false
	}
	r.lastRefresh = now
	r.refreshMu.Unlock()
	r.Scan() //nolint:errcheck // scan problems are per-manifest warnings, reported at startup
	return true
}

func (r *Registry) Get(name string) (*Entry, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	e, ok := r.byName[name]
	return e, ok
}

// List returns all Entries, sorted by name.
func (r *Registry) List() []*Entry {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*Entry, 0, len(r.byName))
	for _, e := range r.byName {
		out = append(out, e)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Manifest.Name < out[j].Manifest.Name })
	return out
}

// Root returns the registry root directory.
func (r *Registry) Root() string { return r.root }
