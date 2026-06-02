// Package registry scans the models/ directory, reads + validates manifests, and lists
// available models. It does NOT load models into memory — that is lifecycle's job.
package registry

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
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
}

// New creates a registry pointing at the root directory that holds the models.
func New(root string) *Registry {
	return &Registry{root: root, byName: map[string]*Entry{}}
}

// Scan scans root/*/manifest.yaml and validates each one. A broken manifest (e.g. missing
// ONNX file, forbidden license) is SKIPPED + collected into the returned error list, without aborting the scan.
func (r *Registry) Scan() ([]error, error) {
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
		mpath := filepath.Join(r.root, e.Name(), "manifest.yaml")
		if _, statErr := os.Stat(mpath); statErr != nil {
			continue // directory has no manifest -> skip
		}
		m, mErr := LoadManifest(mpath)
		if mErr != nil {
			warns = append(warns, mErr)
			continue
		}
		found[m.Name] = &Entry{Manifest: m, Dir: filepath.Join(r.root, e.Name())}
	}
	r.mu.Lock()
	r.byName = found
	r.mu.Unlock()
	return warns, nil
}

// Get returns the Entry for a model name.
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
