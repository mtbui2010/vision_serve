// Package templates provides a thread-safe in-memory store for named template
// image sets used by instance_detection models (OWL-ViT, SiamRPN, …).
//
// Templates are registered via POST /api/templates and looked up by the lifecycle
// manager before invoking a model's Infer, so individual models receive resolved
// image.Image slices through models.Prompt.TemplateImages and never touch this store.
//
// The store is bounded. Every registered image stays decoded in memory until it is replaced or
// deleted, so without a bound a client could grow the server without limit one upload at a time
// (each upload is capped, the number of uploads was not). Two caps apply, both generous next to
// how templates are used (a handful of object crops per set, a few sets):
//
//   - DefaultMaxSets (256) named sets, and
//   - DefaultMaxPixels (400 M) decoded pixels across all sets: ten full-size 40-megapixel
//     uploads (the server's per-image limit), ~1.6 GB of RGBA at the cap.
//
// A registration that would exceed either is refused with an error wrapping ErrFull (the server
// answers 400) and changes nothing; replacing a set under the same name only counts the
// difference. Before the caps there was no limit at all.
package templates

import (
	"errors"
	"fmt"
	"image"
	"sync"
)

// Default bounds of New.
const (
	DefaultMaxSets   = 256
	DefaultMaxPixels = 400_000_000
)

// ErrFull is wrapped by Register when a registration would exceed the store's bounds.
var ErrFull = errors.New("templates: store is full")

// Store holds named sets of template images. All methods are thread-safe.
type Store struct {
	mu        sync.RWMutex
	templates map[string][]image.Image
	pixels    map[string]int64 // decoded pixels per set
	total     int64            // sum of pixels

	maxSets   int
	maxPixels int64
}

// New returns an empty Store bounded by DefaultMaxSets and DefaultMaxPixels.
func New() *Store { return NewWithLimits(DefaultMaxSets, DefaultMaxPixels) }

// NewWithLimits returns an empty Store holding at most maxSets sets and maxPixels decoded pixels
// in total; a non-positive bound is unlimited.
func NewWithLimits(maxSets int, maxPixels int64) *Store {
	return &Store{
		templates: make(map[string][]image.Image),
		pixels:    make(map[string]int64),
		maxSets:   maxSets,
		maxPixels: maxPixels,
	}
}

// Register stores images under name, replacing any previous set.
// images must be non-empty, and the store's bounds must hold afterwards.
func (s *Store) Register(name string, images []image.Image) error {
	if name == "" {
		return fmt.Errorf("templates: name must not be empty")
	}
	if len(images) == 0 {
		return fmt.Errorf("templates: must provide at least one image for %q", name)
	}
	var px int64
	for i, img := range images {
		if img == nil {
			return fmt.Errorf("templates: image %d of %q is nil", i, name)
		}
		b := img.Bounds()
		px += int64(b.Dx()) * int64(b.Dy())
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	old, replacing := s.pixels[name]
	if !replacing && s.maxSets > 0 && len(s.templates) >= s.maxSets {
		return fmt.Errorf("%w: %d template sets registered, the limit is %d — delete one first", ErrFull, len(s.templates), s.maxSets)
	}
	if total := s.total - old + px; s.maxPixels > 0 && total > s.maxPixels {
		return fmt.Errorf("%w: %q would bring the stored templates to %d pixels, the limit is %d — delete a set first",
			ErrFull, name, total, s.maxPixels)
	}
	s.templates[name] = images
	s.pixels[name] = px
	s.total += px - old
	return nil
}

// Get returns the template images for name, or nil if not found.
func (s *Store) Get(name string) []image.Image {
	s.mu.RLock()
	imgs := s.templates[name]
	s.mu.RUnlock()
	return imgs
}

// Delete removes a named template set. No-op if not found.
func (s *Store) Delete(name string) {
	s.mu.Lock()
	s.total -= s.pixels[name]
	delete(s.templates, name)
	delete(s.pixels, name)
	s.mu.Unlock()
}

// List returns all registered template names.
func (s *Store) List() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	names := make([]string, 0, len(s.templates))
	for k := range s.templates {
		names = append(names, k)
	}
	return names
}

// Len returns the number of registered template sets.
func (s *Store) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.templates)
}

// Pixels returns the decoded pixels held across all sets.
func (s *Store) Pixels() int64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.total
}
