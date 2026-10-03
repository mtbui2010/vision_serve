package registry

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// countingCache installs a fresh digest cache for the test whose hash function counts the files it
// actually reads. future=true moves the cache's clock an hour ahead, so files the test just wrote
// are not "too fresh to trust" (see digestCache.racy) — the tests of the racy guard leave it false.
func countingCache(t *testing.T, future bool) (*digestCache, *atomic.Int32) {
	t.Helper()
	if !fileIDSupported {
		t.Skip("no file identity (dev/inode/ctime) on this platform: digests are never cached")
	}
	var reads atomic.Int32
	c := newDigestCache()
	c.hash = func(p string) (string, error) {
		reads.Add(1)
		return fileSHA256(p)
	}
	if future {
		c.now = func() time.Time { return time.Now().Add(time.Hour) }
	}
	prev := weightDigests
	weightDigests = c
	t.Cleanup(func() { weightDigests = prev })
	return c, &reads
}

func pinnedManifest(t *testing.T, content string) (*Manifest, string) {
	t.Helper()
	dir := t.TempDir()
	path, digest := writeWeight(t, dir, "model.onnx", []byte(content))
	m := &Manifest{Name: "x", License: "Apache-2.0", Task: "detection", ModelFile: "model.onnx", dir: dir}
	m.SHA256 = SHA256Field{single: digest}
	return m, path
}

// The point of the cache: every load after an idle unload, and every /api/preprocess, used to
// re-hash every pinned file (3-4 s for 695 MB). An unchanged file must be read once.
func TestVerifyCacheReadsAnUnchangedFileOnce(t *testing.T) {
	_, reads := countingCache(t, true)
	m, _ := pinnedManifest(t, "weights v1")
	for i := 0; i < 3; i++ {
		if err := m.VerifyWeights(); err != nil {
			t.Fatal(err)
		}
	}
	if n := reads.Load(); n != 1 {
		t.Fatalf("file hashed %d times for 3 verifications of unchanged bytes, want 1", n)
	}
}

// A file rewritten after it was cached must be hashed again — and refused if it no longer matches.
func TestVerifyCacheRehashesAModifiedFile(t *testing.T) {
	_, reads := countingCache(t, true)
	m, path := pinnedManifest(t, "weights v1")
	if err := m.VerifyWeights(); err != nil {
		t.Fatal(err)
	}

	// Different size: the cheapest change to notice.
	if err := os.WriteFile(path, []byte("weights v2, longer"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := m.VerifyWeights(); err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("modified file: err = %v, want a sha256 mismatch", err)
	}
	if n := reads.Load(); n != 2 {
		t.Fatalf("hashed %d times, want 2 (the modification must invalidate the cache)", n)
	}
}

// The rewrite an attacker would try: same size, in place (same inode), and the modification time
// put back with `touch -d`. The change time cannot be set back, so it still invalidates.
func TestVerifyCacheSeesSameSizeRewriteWithRestoredMtime(t *testing.T) {
	_, reads := countingCache(t, true)
	m, path := pinnedManifest(t, "weights v1")
	st, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.VerifyWeights(); err != nil {
		t.Fatal(err)
	}

	time.Sleep(50 * time.Millisecond) // past the filesystem's timestamp granularity
	f, err := os.OpenFile(path, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteAt([]byte("WEIGHTS v1"), 0); err != nil { // same length
		t.Fatal(err)
	}
	f.Close()
	if err := os.Chtimes(path, st.ModTime(), st.ModTime()); err != nil {
		t.Fatal(err)
	}
	if st2, _ := os.Stat(path); st2.Size() != st.Size() || !st2.ModTime().Equal(st.ModTime()) {
		t.Fatal("fixture: size or mtime differ, the test would not exercise the change time")
	}

	if err := m.VerifyWeights(); err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("same-size rewrite with restored mtime: err = %v, want a sha256 mismatch", err)
	}
	if n := reads.Load(); n != 2 {
		t.Fatalf("hashed %d times, want 2", n)
	}
}

// A different file moved into place (new inode) with the same size and mtime is a different file.
func TestVerifyCacheDetectsASwappedFile(t *testing.T) {
	_, reads := countingCache(t, true)
	m, path := pinnedManifest(t, "weights v1")
	st, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.VerifyWeights(); err != nil {
		t.Fatal(err)
	}

	other := filepath.Join(filepath.Dir(path), "other.onnx")
	if err := os.WriteFile(other, []byte("weights v9"), 0o644); err != nil { // same length
		t.Fatal(err)
	}
	if err := os.Chtimes(other, st.ModTime(), st.ModTime()); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(other, path); err != nil {
		t.Fatal(err)
	}

	if err := m.VerifyWeights(); err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("swapped file: err = %v, want a sha256 mismatch", err)
	}
	if n := reads.Load(); n != 2 {
		t.Fatalf("hashed %d times, want 2", n)
	}
}

// A file modified moments before it was hashed could be modified AGAIN within the same timestamp
// tick, leaving size/mtime/ctime unchanged. Such a digest is used but not remembered; once the
// file is older than the window it is cached normally.
func TestVerifyCacheDoesNotTrustAFreshFile(t *testing.T) {
	c, reads := countingCache(t, false)
	m, _ := pinnedManifest(t, "weights v1")
	for i := 0; i < 2; i++ {
		if err := m.VerifyWeights(); err != nil {
			t.Fatal(err)
		}
	}
	if n := reads.Load(); n != 2 {
		t.Fatalf("a just-written file was hashed %d times in 2 verifications, want 2 (never cached)", n)
	}

	c.now = func() time.Time { return time.Now().Add(time.Hour) } // the file is now old
	for i := 0; i < 2; i++ {
		if err := m.VerifyWeights(); err != nil {
			t.Fatal(err)
		}
	}
	if n := reads.Load(); n != 3 {
		t.Fatalf("hashed %d times, want 3 (cached once old enough)", n)
	}
}

// Concurrent loads and /api/preprocess calls verify concurrently.
func TestVerifyCacheConcurrent(t *testing.T) {
	_, reads := countingCache(t, true)
	m, _ := pinnedManifest(t, "weights v1")
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := m.VerifyWeights(); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if n := reads.Load(); n < 1 || n > 16 {
		t.Fatalf("hashed %d times", n)
	}
	before := reads.Load()
	if err := m.VerifyWeights(); err != nil {
		t.Fatal(err)
	}
	if reads.Load() != before {
		t.Fatal("not cached after the concurrent verifications")
	}
}
