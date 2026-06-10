package registry

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"strings"

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
//   - No sha256 declared  → no hash check (returns nil unless the source allowlist
//     is configured and rejects the source_url). Fully backward compatible.
//   - sha256 declared      → every covered weight file's computed SHA-256 must match
//     the declared digest, else a clear error (refuse to load).
//
// This converts "the author typed Apache-2.0" into "these exact bytes came from an
// audited source" — it closes the relabeling hole but still TRUSTS the declared
// license itself (see docs/manifest-spec.md threat model).
func (m *Manifest) VerifyWeights() error {
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

	if m.SHA256.IsEmpty() {
		return nil // no content pin — backward-compatible path
	}

	// Multi-file model: verify each role that has a declared digest.
	if files := m.FilesAbs(); len(files) > 0 {
		for role, path := range files {
			want, ok := m.SHA256.expectedFor(role)
			if !ok {
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
		return nil
	}
	if err := verifyFile(m.ModelFilePath(), want); err != nil {
		return fmt.Errorf("model %q: %w", m.Name, err)
	}
	return nil
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

// verifyFile computes a file's SHA-256 and compares it to the expected digest.
func verifyFile(path, want string) error {
	got, err := fileSHA256(path)
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
