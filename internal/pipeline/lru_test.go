package pipeline

import (
	"fmt"
	"sync"
	"testing"
)

func TestLRUEvictsLeastRecentlyUsed(t *testing.T) {
	c := NewLRU[int](3)
	c.Add("a", 1)
	c.Add("b", 2)
	c.Add("c", 3)
	if _, ok := c.Get("a"); !ok { // a is now the most recently used
		t.Fatal("a missing")
	}
	c.Add("d", 4) // evicts b, the least recently used
	if _, ok := c.Get("b"); ok {
		t.Error("b survived; the least recently used entry must go first")
	}
	for _, k := range []string{"a", "c", "d"} {
		if _, ok := c.Get(k); !ok {
			t.Errorf("%s was evicted", k)
		}
	}
	if c.Len() != 3 {
		t.Errorf("Len = %d, want the capacity 3", c.Len())
	}
}

// Two requests that both missed a word must end up sharing the FIRST published value.
func TestLRUAddKeepsPublishedValue(t *testing.T) {
	c := NewLRU[string](2)
	if got := c.Add("w", "first"); got != "first" {
		t.Fatalf("Add = %q", got)
	}
	if got := c.Add("w", "second"); got != "first" {
		t.Errorf("Add on an existing key returned %q, want the published %q", got, "first")
	}
	c.Set("w", "third")
	if got, _ := c.Get("w"); got != "third" {
		t.Errorf("Set did not replace: %q", got)
	}
}

func TestLRUCapacityFloor(t *testing.T) {
	c := NewLRU[int](0)
	c.Add("a", 1)
	c.Add("b", 2)
	if c.Len() != 1 {
		t.Errorf("Len = %d, want 1 (capacity < 1 is treated as 1)", c.Len())
	}
}

// The server serves requests in parallel (CLAUDE.md). Run with -race.
func TestLRUConcurrent(t *testing.T) {
	c := NewLRU[int](16)
	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 500; i++ {
				k := fmt.Sprint((g + i) % 40)
				if _, ok := c.Get(k); !ok {
					c.Add(k, i)
				}
				if i%7 == 0 {
					c.Set(k, -i)
				}
			}
		}(g)
	}
	wg.Wait()
	if n := c.Len(); n > 16 {
		t.Errorf("Len = %d exceeds the capacity", n)
	}
}
