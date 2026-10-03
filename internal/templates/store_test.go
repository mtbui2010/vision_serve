package templates

import (
	"errors"
	"fmt"
	"image"
	"sort"
	"sync"
	"testing"
)

func img(w, h int) image.Image { return image.NewRGBA(image.Rect(0, 0, w, h)) }

func TestRegisterGetListDelete(t *testing.T) {
	s := New()
	if err := s.Register("mug", []image.Image{img(10, 10), img(4, 5)}); err != nil {
		t.Fatal(err)
	}
	if err := s.Register("can", []image.Image{img(3, 3)}); err != nil {
		t.Fatal(err)
	}
	if got := s.Get("mug"); len(got) != 2 {
		t.Errorf("Get(mug) = %d images, want 2", len(got))
	}
	if s.Get("nope") != nil {
		t.Error("Get of an unknown name returned images")
	}
	names := s.List()
	sort.Strings(names)
	if fmt.Sprint(names) != "[can mug]" || s.Len() != 2 || s.Pixels() != 100+20+9 {
		t.Errorf("List %v, Len %d, Pixels %d", names, s.Len(), s.Pixels())
	}
	s.Delete("mug")
	s.Delete("mug") // no-op
	if s.Get("mug") != nil || s.Len() != 1 || s.Pixels() != 9 {
		t.Errorf("after Delete: Len %d, Pixels %d", s.Len(), s.Pixels())
	}
}

func TestRegisterRejectsBadInput(t *testing.T) {
	s := New()
	for name, imgs := range map[string][]image.Image{"": {img(1, 1)}, "empty": nil, "nil": {nil}} {
		if err := s.Register(name, imgs); err == nil {
			t.Errorf("Register(%q, %d images) accepted", name, len(imgs))
		}
	}
	if s.Len() != 0 {
		t.Error("a refused registration was stored")
	}
}

// The number of sets is bounded; replacing an existing set is always allowed.
func TestSetLimit(t *testing.T) {
	s := NewWithLimits(2, 0)
	for _, n := range []string{"a", "b"} {
		if err := s.Register(n, []image.Image{img(1, 1)}); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.Register("c", []image.Image{img(1, 1)}); !errors.Is(err, ErrFull) {
		t.Fatalf("third set: %v, want ErrFull", err)
	}
	if err := s.Register("a", []image.Image{img(2, 2)}); err != nil {
		t.Fatalf("replacing a set at the limit: %v", err)
	}
	s.Delete("b")
	if err := s.Register("c", []image.Image{img(1, 1)}); err != nil {
		t.Fatalf("after a delete: %v", err)
	}
}

// Decoded pixels are bounded across sets; a refused registration changes nothing, and a
// replacement only counts the difference.
func TestPixelLimit(t *testing.T) {
	s := NewWithLimits(0, 1000)
	if err := s.Register("a", []image.Image{img(20, 20), img(10, 10)}); err != nil { // 500
		t.Fatal(err)
	}
	if err := s.Register("b", []image.Image{img(30, 20)}); !errors.Is(err, ErrFull) { // 1100
		t.Fatalf("over the pixel cap: %v, want ErrFull", err)
	}
	if s.Get("b") != nil || s.Pixels() != 500 {
		t.Fatalf("a refused set left traces: Pixels %d", s.Pixels())
	}
	if err := s.Register("a", []image.Image{img(30, 30)}); err != nil { // 900 replaces 500
		t.Fatalf("replacement within the cap: %v", err)
	}
	if s.Pixels() != 900 {
		t.Errorf("Pixels = %d, want 900", s.Pixels())
	}
	if err := s.Register("b", []image.Image{img(10, 10)}); err != nil { // exactly 1000: at the cap is fine
		t.Fatalf("a set reaching the cap exactly: %v", err)
	}
	if err := s.Register("c", []image.Image{img(1, 1)}); !errors.Is(err, ErrFull) {
		t.Fatalf("one pixel past the cap: %v, want ErrFull", err)
	}
}

func TestDefaultsAreGenerous(t *testing.T) {
	s := New()
	// 64 sets of eight 512×512 crops: far beyond the usual handful of object crops.
	for i := 0; i < 64; i++ {
		imgs := make([]image.Image, 8)
		for j := range imgs {
			imgs[j] = image.NewGray(image.Rect(0, 0, 512, 512))
		}
		if err := s.Register(fmt.Sprint(i), imgs); err != nil {
			t.Fatalf("set %d: %v", i, err)
		}
	}
}

// Run with -race: the server registers, reads and deletes from many requests at once.
func TestConcurrent(t *testing.T) {
	s := NewWithLimits(8, 0)
	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				n := fmt.Sprint((g + i) % 12)
				_ = s.Register(n, []image.Image{img(2, 2)})
				_ = s.Get(n)
				_ = s.List()
				if i%5 == 0 {
					s.Delete(n)
				}
			}
		}(g)
	}
	wg.Wait()
	if s.Len() > 8 || s.Pixels() != int64(4*s.Len()) {
		t.Errorf("Len %d (limit 8), Pixels %d (want 4 per set)", s.Len(), s.Pixels())
	}
}
