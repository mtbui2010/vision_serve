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
	lastRoot    rootState // the root directory as that rescan saw it
}

// rootState is what an install changes in the root directory: its mtime (an install renames a
// staging directory into root; a --force swap renames the old one aside) and the names of its
// model directories (in case two changes land within one tick of a coarse filesystem clock).
type rootState struct {
	ok    bool
	mtime time.Time
	names string
}

func (r *Registry) statRoot() rootState {
	fi, err := os.Stat(r.root)
	if err != nil {
		return rootState{}
	}
	entries, err := os.ReadDir(r.root)
	if err != nil {
		return rootState{}
	}
	var b strings.Builder
	for _, e := range entries {
		if e.IsDir() && !strings.HasPrefix(e.Name(), ".") {
			b.WriteString(e.Name())
			b.WriteByte(0)
		}
	}
	return rootState{ok: true, mtime: fi.ModTime(), names: b.String()}
}

func (s rootState) changedFrom(prev rootState) bool {
	return s.ok && (!prev.ok || !s.mtime.Equal(prev.mtime) || s.names != prev.names)
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

// Refresh rescans the directory when the root directory changed since the last Refresh rescan
// (a model directory was added, removed or swapped: install, pull, convert, import, a copied
// folder), or else when that rescan is at least minInterval old, and reports whether it did. A
// server lists models with it, so a model installed while the server runs shows up in the list
// at once, even right after the previous listing — a converter that installs and then checks
// the list no longer misses it. The interval bounds the cost (a ReadDir and one YAML parse per
// model) when a client polls an unchanged list; a manifest edited inside an existing model
// directory does not change the root, so it waits for the interval.
//
// A Refresh that returns has seen a scan at least as new as the root it observed: the rescan
// runs under the refresh lock, so a concurrent caller never lists the registry from before it.
func (r *Registry) Refresh(minInterval time.Duration) bool {
	r.refreshMu.Lock()
	defer r.refreshMu.Unlock()
	now := time.Now()
	root := r.statRoot()
	recent := !r.lastRefresh.IsZero() && now.Sub(r.lastRefresh) < minInterval
	if recent && !root.changedFrom(r.lastRoot) {
		return false
	}
	r.lastRefresh, r.lastRoot = now, root // observed BEFORE the scan: a change during it rescans next time
	r.Scan()                              //nolint:errcheck // scan problems are per-manifest warnings, reported at startup
	return true
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
