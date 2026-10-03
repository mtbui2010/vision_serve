package pipeline

import (
	"container/list"
	"sync"
)

// LRU is a bounded, thread-safe least-recently-used map. Get refreshes an entry; Add inserts one,
// evicting the least recently used entry once Len reaches the capacity. It holds per-WORD values
// (text embeddings, folded head rows), so a new word list costs only its new words and memory
// stays bounded under adversarial prompts.
type LRU[V any] struct {
	mu    sync.Mutex
	cap   int
	ll    *list.List // front = most recently used
	items map[string]*list.Element
}

type lruEntry[V any] struct {
	key string
	val V
}

// NewLRU returns an empty cache holding at most capacity entries (capacity < 1 is treated as 1).
func NewLRU[V any](capacity int) *LRU[V] {
	if capacity < 1 {
		capacity = 1
	}
	return &LRU[V]{cap: capacity, ll: list.New(), items: map[string]*list.Element{}}
}

// Get returns the value under key and marks it most recently used.
func (c *LRU[V]) Get(key string) (V, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if e, ok := c.items[key]; ok {
		c.ll.MoveToFront(e)
		return e.Value.(*lruEntry[V]).val, true
	}
	var zero V
	return zero, false
}

// Add inserts val under key unless the key is already present, and returns the value the cache
// now holds for key. Two requests that both missed and computed the same word therefore end up
// sharing ONE published value — the first — exactly as the whole-list caches it replaces did.
func (c *LRU[V]) Add(key string, val V) V {
	c.mu.Lock()
	defer c.mu.Unlock()
	if e, ok := c.items[key]; ok {
		c.ll.MoveToFront(e)
		return e.Value.(*lruEntry[V]).val
	}
	c.items[key] = c.ll.PushFront(&lruEntry[V]{key: key, val: val})
	c.evict()
	return val
}

// Set stores val under key, replacing any value already there, and marks it most recently used.
func (c *LRU[V]) Set(key string, val V) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if e, ok := c.items[key]; ok {
		e.Value.(*lruEntry[V]).val = val
		c.ll.MoveToFront(e)
		return
	}
	c.items[key] = c.ll.PushFront(&lruEntry[V]{key: key, val: val})
	c.evict()
}

// evict drops least recently used entries until the cache is within capacity. Callers hold mu.
func (c *LRU[V]) evict() {
	for c.ll.Len() > c.cap {
		last := c.ll.Back()
		c.ll.Remove(last)
		delete(c.items, last.Value.(*lruEntry[V]).key)
	}
}

// Len reports the number of entries held.
func (c *LRU[V]) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.ll.Len()
}
