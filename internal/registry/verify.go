package registry

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"gopkg.in/yaml.v3"
)

// SHA256Field is the manifest `sha256:` field. It is OPTIONAL and accepts EITHER
// of two YAML shapes, so single-file and multi-file models share one field:
//
//	sha256: "ab12…"              # single-file model (model_file)
//	sha256:                       # multi-file model (files: map)
//	  encoder: "ab12…"
//	  decoder: "cd34…"
//
// An empty SHA256Field (field absent) means "no content pinning" — weights load
// without a hash check (backward compatible with every existing manifest).
type SHA256Field struct {
	// single is the lone digest when the manifest gives a scalar sha256.
	single string
	// byRole maps a files: role to its expected digest when the manifest gives a map.
	byRole map[string]string
}

// UnmarshalYAML accepts a scalar string OR a role→digest mapping.
func (s *SHA256Field) UnmarshalYAML(value *yaml.Node) error {
	switch value.Kind {
	case 0: // empty / null node
		return nil
	case yaml.ScalarNode:
		var str string
		if err := value.Decode(&str); err != nil {
			return err
		}
		s.single = normalizeDigest(str)
		return nil
	case yaml.MappingNode:
		var m map[string]string
		if err := value.Decode(&m); err != nil {
			return err
		}
		s.byRole = make(map[string]string, len(m))
		for k, v := range m {
			s.byRole[k] = normalizeDigest(v)
		}
		return nil
	default:
		return fmt.Errorf("sha256 must be a hex string or a role→hex map")
	}
}

// IsEmpty reports whether no sha256 was declared (the common, backward-compatible case).
func (s SHA256Field) IsEmpty() bool {
	return s.single == "" && len(s.byRole) == 0
}

// declared returns every digest this field pins, for cross-checking the whole set at once.
func (s SHA256Field) declared() []string {
	if s.single != "" {
		return []string{s.single}
	}
	out := make([]string, 0, len(s.byRole))
	for _, d := range s.byRole {
		out = append(out, d)
	}
	return out
}

// expectedFor returns the declared digest for a given files: role (or for the
// single model_file when role is ""). ok is false when nothing is pinned for it.
func (s SHA256Field) expectedFor(role string) (digest string, ok bool) {
	if s.single != "" && (role == "" || len(s.byRole) == 0) {
		return s.single, true
	}
	if d, found := s.byRole[role]; found {
		return d, true
	}
	return "", false
}

func normalizeDigest(s string) string {
	s = strings.TrimSpace(strings.ToLower(s))
	// Tolerate an explicit "sha256:" algorithm prefix (OMS / SPDX style).
	s = strings.TrimPrefix(s, "sha256:")
	return s
}

// VerifiedSourcePrefixes is an OPTIONAL, opt-in allowlist hook. When non-empty,
// VerifyWeights additionally requires a manifest's source_url to begin with one
// of these curated, human-audited prefixes (e.g. the official upstream repos).
//
// It is intentionally EMPTY by default so existing behavior is unchanged and the
// project stays local-first (no network, no surprise rejections). A deployer who
// wants to restrict installs to audited origins populates it at startup, e.g.:
//
//	registry.VerifiedSourcePrefixes = []string{
//	    "https://huggingface.co/lyuwenyu/RT-DETR/",
//	    "https://github.com/IDEA-Research/GroundingDINO/",
//	}
//
// This binds the declared license not only to specific bytes (sha256) but to an
// audited origin. It does NOT fetch or verify anything over the network — it is a
// string-prefix policy gate over the author-declared source_url.
var VerifiedSourcePrefixes []string

// VerifyWeights enforces the content-pin (sha256) and, if configured, the
// verified-source allowlist. It is called at LOAD time, AFTER weights are known
// to exist on disk. Behavior:
//
//   - Default gate, no sha256 declared → no hash check. Fully backward compatible.
//   - sha256 declared → every covered weight file's computed SHA-256 must match the
//     declared digest, else a clear error (refuse to load).
//   - Hardened gate (verified mode, or a configured source allowlist) → a content pin is
//     REQUIRED, for every weight file. See the comment at the check below for why.
//
// This converts "the author typed Apache-2.0" into "these exact bytes came from an
// audited source" — it closes the relabeling hole but still TRUSTS the declared
// license itself (see docs/manifest-spec.md threat model).
func (m *Manifest) VerifyWeights() error { return m.verifyWeights(0) }

// maxComposeDepth bounds the dependency walk for composed models. The real chains are one deep
// (grounded-sam → grounding-dino + mobile-sam); the bound exists so a manifest cycle produces a
// clear error instead of a stack overflow.
const maxComposeDepth = 4

func (m *Manifest) verifyWeights(depth int) error {
	hardened := VerifiedModeEnabled() || len(VerifiedSourcePrefixes) > 0

	// A COMPOSED model (grounded-sam, rfdetr-gdino, grasp-gd) downloads nothing of its own: every
	// weight it names lives in another model's directory. It therefore has no source_url and no
	// digests to declare, and the checks below would refuse it — which would mean turning verified
	// mode on disables exactly the pipelines this project exists to serve. Its admission comes from
	// the models that DO own those bytes, which is also the honest answer: a composition is as
	// audited as what it composes.
	if hardened {
		if paths, ok := m.composedWeights(); ok {
			return m.verifyComposed(paths, depth)
		}
		// A PARTLY composed model (rfdetr-textalign-*: its own detector and projection, plus a
		// text tower in ../siglip-text/) answers for its own files through the checks below, and
		// for the borrowed ones the way a composed model does: through the model that owns them.
		if foreign := m.foreignWeights(); len(foreign) > 0 {
			if err := m.verifyComposed(foreign, depth); err != nil {
				return err
			}
		}
	}

	// Verified mode: cross-check the contributor-declared license against the maintainer-audited
	// ledger (see ledger.go). No-op when verified mode is off. This is the control that catches a
	// manifest mislabeled by its author even when bytes/origin are consistent.
	if err := m.VerifyLicenseProvenance(); err != nil {
		return err
	}

	if len(VerifiedSourcePrefixes) > 0 {
		if err := checkSourceAllowlist(m.SourceURL); err != nil {
			return fmt.Errorf("model %q: %w", m.Name, err)
		}
	}

	// The manifest's own sha256 lives in the file an attacker rewriting the model directory would
	// rewrite too, so on its own it proves only self-consistency. Require every declared digest to
	// appear in the maintainer's in-binary record for this audited upstream (ledger.go), which that
	// attacker cannot reach. Upstreams with no recorded digests fall back to the manifest pin alone.
	if hardened {
		if err := m.checkAnchoredPins(); err != nil {
			return err
		}
	}

	if err := m.verifySessionPins(hardened); err != nil {
		return err
	}
	return m.verifySideFiles(hardened)
}

// verifySessionPins checks the `sha256:` pins of the ONNX session file(s).
func (m *Manifest) verifySessionPins(hardened bool) error {
	// Under a hardened gate an UNPINNED model must not load. An audited source_url records
	// where the bytes were SUPPOSED to come from; it says nothing about the bytes now on disk.
	// Binding the declared license to specific bytes is the whole point of verified mode, so
	// leaving the pin optional there would let a manifest opt out of the control by omission —
	// the quietest possible failure. The default gate keeps sha256 optional, unchanged.
	if m.SHA256.IsEmpty() {
		if hardened {
			return fmt.Errorf("model %q: no sha256 content pin — refusing to load "+
				"(verified mode binds the declared license to specific weight bytes; add sha256 to the manifest)", m.Name)
		}
		return nil // no content pin — backward-compatible path
	}

	// Multi-file model: verify each role that has a declared digest.
	if files := m.FilesAbs(); len(files) > 0 {
		for role, path := range files {
			want, ok := m.SHA256.expectedFor(role)
			if !ok {
				// A role borrowed from another model's directory was admitted through that model
				// (verifyWeights); its pin lives in that model's manifest.
				if hardened && !m.isForeign(path) {
					return fmt.Errorf("model %q: role %q has no sha256 content pin — refusing to load "+
						"(verified mode requires every weight file to be pinned)", m.Name, role)
				}
				continue // this role is not pinned — skip (optional per-role pinning)
			}
			if err := verifyFile(path, want); err != nil {
				return fmt.Errorf("model %q role %q: %w", m.Name, role, err)
			}
		}
		return nil
	}

	// Single-file model.
	want, ok := m.SHA256.expectedFor("")
	if !ok {
		if hardened {
			return fmt.Errorf("model %q: sha256 declares no digest for the model file — refusing to load "+
				"(verified mode requires a content pin)", m.Name)
		}
		return nil
	}
	if err := verifyFile(m.ModelFilePath(), want); err != nil {
		return fmt.Errorf("model %q: %w", m.Name, err)
	}
	return nil
}

// verifySideFiles checks the `sha256_files:` pins, and — under a hardened gate — refuses ONNX
// external weight data that is not pinned.
//
// The session pin covers the graph file only. A graph saved with external data keeps its tensors
// (the actual weights) in a separate file, conventionally <graph>.data, that ORT reads by the name
// recorded inside the graph. Pinning the graph therefore fixes WHICH file is read but not its
// bytes: without this check the weights could be swapped under a pinned graph and still load.
func (m *Manifest) verifySideFiles(hardened bool) error {
	rels := make([]string, 0, len(m.SHA256Files))
	for rel := range m.SHA256Files {
		rels = append(rels, rel)
	}
	sort.Strings(rels)
	for _, rel := range rels {
		if err := verifyFile(filepath.Join(m.dir, rel), m.SHA256Files[rel]); err != nil {
			return fmt.Errorf("model %q file %q: %w", m.Name, rel, err)
		}
	}
	if !hardened {
		return nil
	}
	pinned := make(map[string]bool, len(rels))
	for _, rel := range rels {
		pinned[filepath.Join(m.dir, rel)] = true
	}
	for _, w := range m.sessionPaths() {
		if m.isForeign(w) {
			continue // its external data is pinned, and checked, by the model that owns it
		}
		data := w + ".data"
		if _, err := os.Stat(data); err == nil && !pinned[filepath.Clean(data)] {
			return fmt.Errorf("model %q: %s holds external weight data but sha256_files does not pin it — "+
				"refusing to load (verified mode requires every weight byte to be pinned)", m.Name, filepath.Base(data))
		}
	}
	return nil
}

// sessionPaths returns the ONNX session file path(s) the manifest names.
func (m *Manifest) sessionPaths() []string {
	if files := m.FilesAbs(); len(files) > 0 {
		out := make([]string, 0, len(files))
		for _, p := range files {
			out = append(out, filepath.Clean(p))
		}
		return out
	}
	return []string{filepath.Clean(m.ModelFilePath())}
}

// PinnedDigests returns every SHA-256 digest the manifest pins: the session pins (sha256:) and
// the side-file pins (sha256_files:).
func (m *Manifest) PinnedDigests() []string {
	out := m.SHA256.declared()
	for _, d := range m.SHA256Files {
		out = append(out, d)
	}
	return out
}

// checkAnchoredPins requires every digest the manifest declares to be one the maintainer
// recorded for this upstream in the in-binary ledger.
//
// It is a set membership test, not a per-role equality test, because local filenames are not
// unique within an upstream (rf-detr and rf-detr-nano both land as rf-detr-base.onnx from the
// same repo). What it establishes is the property that matters for a LICENCE gate: the bytes
// about to be loaded are bytes a human audited. Distinguishing WHICH audited file belongs in
// which role is the manifest pin's job, and that check still runs.
func (m *Manifest) checkAnchoredPins() error {
	led, ok := lookupLedger(m.SourceURL)
	if !ok || len(led.WeightSHA256) == 0 {
		return nil // upstream carries no recorded digests; the manifest pin is all there is
	}
	audited := make(map[string]bool, len(led.WeightSHA256))
	for _, d := range led.WeightSHA256 {
		audited[strings.ToLower(d)] = true
	}
	for _, d := range m.PinnedDigests() {
		if !audited[strings.ToLower(d)] {
			return fmt.Errorf("model %q: sha256 %s is not among the digests the maintainer audited "+
				"for %s — refusing to load (a manifest cannot vouch for its own bytes; the audited "+
				"record lives in the binary)", m.Name, shortDigest(d), led.SourcePrefix)
		}
	}
	return nil
}

func shortDigest(d string) string {
	if len(d) > 12 {
		return d[:12] + "…"
	}
	return d
}

// composedWeights reports whether EVERY weight this manifest names lives outside its own
// directory, and returns those paths. That is the signature of a composed pipeline: it declares
// files: entries like "../grounding-dino/model-fixedmask.onnx" and ships no weights itself.
//
// A model with even one file of its own is not composed — it has bytes to answer for, and must
// go through the normal source_url + pin path.
func (m *Manifest) composedWeights() ([]string, bool) {
	files := m.FilesAbs()
	if len(files) == 0 {
		return nil, false
	}
	own, err := filepath.Abs(m.dir)
	if err != nil {
		return nil, false
	}
	paths := make([]string, 0, len(files))
	for _, p := range files {
		abs, err := filepath.Abs(p)
		if err != nil {
			return nil, false
		}
		if filepath.Dir(abs) == own {
			return nil, false
		}
		paths = append(paths, abs)
	}
	return paths, true
}

// foreignWeights returns the session files this manifest borrows from outside its own directory
// (absolute paths, sorted), for a model that also has files of its own. Nil when there are none.
func (m *Manifest) foreignWeights() []string {
	var out []string
	for _, p := range m.FilesAbs() {
		if m.isForeign(p) {
			abs, err := filepath.Abs(p)
			if err != nil {
				abs = p
			}
			out = append(out, abs)
		}
	}
	sort.Strings(out)
	return out
}

// isForeign reports whether path lies outside the manifest's own directory tree, i.e. belongs to
// another model ("../siglip-text/model.onnx"). Undecidable paths count as own, so they keep going
// through the strict own-file checks.
func (m *Manifest) isForeign(path string) bool {
	own, err := filepath.Abs(m.dir)
	if err != nil {
		return false
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return false
	}
	rel, err := filepath.Rel(own, abs)
	if err != nil {
		return false
	}
	return rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

// verifyComposed admits a composed model exactly when every model that owns one of its weights
// is itself admitted. Two conditions, and the second is the one worth stating: the owning
// manifest must actually DECLARE the file. Without that, a composed manifest could point at any
// stray file sitting inside an audited model's directory and inherit its admission.
func (m *Manifest) verifyComposed(paths []string, depth int) error {
	if depth >= maxComposeDepth {
		return fmt.Errorf("model %q: composed-model chain deeper than %d — refusing to load "+
			"(a manifest cycle?)", m.Name, maxComposeDepth)
	}
	owners := make(map[string]*Manifest)
	for _, p := range paths {
		dir := filepath.Dir(p)
		owner, seen := owners[dir]
		if !seen {
			var err error
			owner, err = LoadManifest(filepath.Join(dir, "manifest.yaml"))
			if err != nil {
				return fmt.Errorf("model %q: weight %s belongs to no model (%v) — refusing to load "+
					"(a composed model may only reference another model's declared weights)",
					m.Name, filepath.Base(p), err)
			}
			owners[dir] = owner
		}
		if !owner.declaresFile(p) {
			return fmt.Errorf("model %q: references %s, which model %q does not declare — "+
				"refusing to load", m.Name, filepath.Base(p), owner.Name)
		}
	}
	for _, owner := range owners {
		if err := owner.verifyWeights(depth + 1); err != nil {
			return fmt.Errorf("model %q: dependency %q is not admitted: %w", m.Name, owner.Name, err)
		}
	}
	return nil
}

// declaresFile reports whether abs is one of the weight files this manifest names.
func (m *Manifest) declaresFile(abs string) bool {
	for _, p := range m.FilesAbs() {
		if a, err := filepath.Abs(p); err == nil && a == abs {
			return true
		}
	}
	if m.ModelFile != "" {
		if a, err := filepath.Abs(m.ModelFilePath()); err == nil && a == abs {
			return true
		}
	}
	return false
}

// checkSourceAllowlist requires source_url to start with an audited prefix.
func checkSourceAllowlist(sourceURL string) error {
	if sourceURL == "" {
		return fmt.Errorf("source_url is required but missing (verified-source allowlist is enforced)")
	}
	for _, p := range VerifiedSourcePrefixes {
		if strings.HasPrefix(sourceURL, p) {
			return nil
		}
	}
	return fmt.Errorf("source_url %q is not under any audited prefix in the verified-source allowlist", sourceURL)
}

// verifyFile computes a file's SHA-256 (cached while the file is unchanged, see digestCache) and
// compares it to the expected digest.
func verifyFile(path, want string) error {
	got, err := weightDigests.sha256(path)
	if err != nil {
		return err
	}
	if got != want {
		return fmt.Errorf(
			"sha256 mismatch for %s: manifest declares %s but file is %s — refusing to load (the declared license is bound to the audited bytes)",
			path, want, got,
		)
	}
	return nil
}

// weightDigests is the process-wide digest cache VerifyWeights uses.
var weightDigests = newDigestCache()

// digestCache remembers the SHA-256 of weight files for the life of the process, so verifying an
// unchanged file costs a stat instead of a full read. Without it every reload after the idle
// reaper unloaded a model, and every /api/preprocess call, re-hashed every pinned file: 3-4 s for
// a 695 MB model.
//
// An entry is keyed by the file's absolute path and is valid only while the file's identity is
// unchanged: size, modification time, change time, device and inode (symlinks followed). A
// rewrite changes size or mtime; one that restores both (same length, `touch -d`) still moves the
// change time, which cannot be set back; a different file renamed into place has another inode.
//
// Two more rules keep it honest:
//   - the file must have the same identity before and after hashing, or the digest (of bytes that
//     were changing under the read) is used once and not remembered;
//   - a file modified within racyWindow of the hash is not remembered either. Timestamps have a
//     granularity (a kernel tick, up to 2 s on FAT), so a second rewrite in the same tick would
//     leave the identity unchanged — the "racily clean" problem git solves the same way.
//
// On platforms without inode/change time (fileIDSupported == false) nothing is cached: the
// previous behaviour, a full hash every time.
type digestCache struct {
	mu      sync.Mutex
	entries map[string]cachedDigest

	hash func(path string) (string, error) // fileSHA256; tests count the reads
	now  func() time.Time
}

type cachedDigest struct {
	id     fileID
	digest string
}

// fileID is what must stay equal for a cached digest to be reused (see digestCache).
type fileID struct {
	size         int64
	mtime, ctime int64 // nanoseconds since the epoch
	dev, ino     uint64
}

// racyWindow bounds filesystem timestamp granularity (FAT: 2 s; ext4/xfs/apfs: a tick or finer).
const racyWindow = 2 * time.Second

func newDigestCache() *digestCache {
	return &digestCache{entries: map[string]cachedDigest{}, hash: fileSHA256, now: time.Now}
}

// sha256 returns the file's digest, from the cache when the file is unchanged.
func (c *digestCache) sha256(path string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		abs = filepath.Clean(path)
	}
	before, err := statFileID(abs)
	if err != nil { // missing file, or no file identity on this platform: just hash (or fail) as before
		c.forget(abs)
		return c.hash(abs)
	}
	c.mu.Lock()
	e, ok := c.entries[abs]
	c.mu.Unlock()
	if ok && e.id == before {
		return e.digest, nil
	}

	start := c.now()
	digest, err := c.hash(abs)
	if err != nil {
		c.forget(abs)
		return "", err
	}
	after, err := statFileID(abs)
	if err != nil || after != before || c.racy(before, start) {
		c.forget(abs)
		return digest, nil
	}
	c.mu.Lock()
	c.entries[abs] = cachedDigest{id: before, digest: digest}
	c.mu.Unlock()
	return digest, nil
}

// racy reports whether the file changed too close to the hash for its timestamps to tell a later
// rewrite apart (see digestCache).
func (c *digestCache) racy(id fileID, hashStart time.Time) bool {
	limit := hashStart.Add(-racyWindow).UnixNano()
	return id.mtime >= limit || id.ctime >= limit
}

func (c *digestCache) forget(abs string) {
	c.mu.Lock()
	delete(c.entries, abs)
	c.mu.Unlock()
}

// fileSHA256 streams a file through crypto/sha256 (never buffers it in RAM).
func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("cannot open weight file for hashing: %w", err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", fmt.Errorf("cannot hash weight file %s: %w", path, err)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
