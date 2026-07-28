// Package templates provides a thread-safe in-memory store for named template
// image sets used by instance_detection models (OWL-ViT, SiamRPN, …).
//
// Templates are registered via POST /api/templates and looked up by the lifecycle
// manager before invoking a model's Infer, so individual models receive resolved
// image.Image slices through models.Prompt.TemplateImages and never touch this store.
package templates

import (
	"fmt"
	"image"
	"sync"
)

// Store holds named sets of template images. All methods are thread-safe.
type Store struct {
	mu        sync.RWMutex
	templates map[string][]image.Image
}

// New returns an empty Store.
func New() *Store {
	return &Store{templates: make(map[string][]image.Image)}
}

// Register stores images under name, replacing any previous set.
// images must be non-empty.
func (s *Store) Register(name string, images []image.Image) error {
	if name == "" {
		return fmt.Errorf("templates: name must not be empty")
	}
	if len(images) == 0 {
		return fmt.Errorf("templates: must provide at least one image for %q", name)
	}
	s.mu.Lock()
	s.templates[name] = images
	s.mu.Unlock()
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
	delete(s.templates, name)
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
