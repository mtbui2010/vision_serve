package catalog

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"visionserve/internal/registry"
)

// repoModels is the repo's models/ directory, which tracks the hand-written manifests (and the
// small side files: proj.bin, labels, templates) the textalign catalog entries were made from.
var repoModels = filepath.Join("..", "..", "models")

var textalignEntries = []string{
	"rfdetr-textalign-dec1",
	"rfdetr-textalign-dec1-siglip",
	"rfdetr-textalign-dec1-siglip-prod",
	"rfdetr-textalign-etri",
}

// A pulled rfdetr-textalign-* model (or the clip-text tower two of them borrow) must behave exactly like the one served from the repo's
// models/ directory, where every number in its README was measured. So the generated manifest
// must parse to the same input/postprocess/runtime/explain settings as the tracked one, name the
// same roles, and its pinned side files (proj.bin, labels, templates) must be the very bytes the
// repo tracks. A threshold or a threads block lost in the catalog copy would change outputs or
// latency with no error anywhere.
func TestTextalignEntriesMatchRepoManifests(t *testing.T) {
	for _, name := range append([]string{"clip-text"}, textalignEntries...) {
		t.Run(name, func(t *testing.T) {
			e, ok := Lookup(name)
			if !ok {
				t.Fatalf("%s is not in the catalog", name)
			}
			repo, err := registry.LoadManifest(filepath.Join(repoModels, name, "manifest.yaml"))
			if err != nil {
				t.Fatalf("repo manifest: %v", err)
			}
			got := loadRendered(t, filepath.Join(t.TempDir(), name, "manifest.yaml"), render(t, e))

			if got.Task != repo.Task || got.License != repo.License || got.Architecture != repo.Architecture {
				t.Errorf("task/license/architecture %s/%s/%s, repo has %s/%s/%s",
					got.Task, got.License, got.Architecture, repo.Task, repo.License, repo.Architecture)
			}
			for what, pair := range map[string][2]any{
				"input":       {got.Input, repo.Input},
				"postprocess": {got.Postprocess, repo.Postprocess},
				"runtime":     {got.Runtime, repo.Runtime},
				"explain":     {got.Explain, repo.Explain},
			} {
				if !reflect.DeepEqual(pair[0], pair[1]) {
					t.Errorf("%s differs from the repo manifest:\ncatalog: %+v\nrepo:    %+v", what, pair[0], pair[1])
				}
			}
			if g, r := sortedKeys(got.Files), sortedKeys(repo.Files); !reflect.DeepEqual(g, r) {
				t.Fatalf("roles %v, repo manifest has %v", g, r)
			}

			// Same file per role: the same relative path, or — where the catalog renames a file the
			// repo borrows from a sibling directory (etri's explain graph) — the digest the repo
			// manifest pins for that role.
			pins := map[string]string{}
			for _, f := range e.Files {
				if f.ManifestRole != "" {
					pins[f.ManifestRole] = f.SHA256
				}
			}
			for role, rel := range got.Files {
				if rel == repo.Files[role] {
					continue
				}
				want, ok := repoRolePin(t, repo, role)
				if !ok || !strings.EqualFold(want, pins[role]) {
					t.Errorf("role %s: catalog %q (sha256 %s) vs repo %q (pin %q)", role, rel, pins[role], repo.Files[role], want)
				}
			}

			// The side files the repo tracks must be the bytes the catalog pins, and the ones the
			// model reads must all be downloaded.
			need := map[string]bool{}
			if e.Architecture == "rfdetr-textalign" {
				need = map[string]bool{"proj.bin": true, "templates.txt": true, e.LabelsFile: true}
			}
			for _, f := range e.Files {
				if f.ManifestRole != "" {
					continue // weights: not tracked by the repo, pinned per role above
				}
				src := filepath.Join(repo.Dir(), f.LocalFilename)
				if f.LocalFilename == e.LabelsFile {
					src = repo.LabelsPath()
				}
				delete(need, f.LocalFilename)
				if _, err := os.Stat(src); err != nil {
					continue // not tracked in the repo (clip-text's LICENSE, vocab-all22.txt's twin)
				}
				if d := fileDigest(t, src); d != f.SHA256 {
					t.Errorf("%s: catalog pins %s…, the repo's %s is %s…", f.LocalFilename, f.SHA256[:12], src, d[:12])
				}
			}
			if len(need) > 0 {
				t.Errorf("catalog entry does not download %v", sortedKeys(need))
			}
		})
	}
}

// repoRolePin returns the sha256 a repo manifest pins for role, via the rendered digest list.
func repoRolePin(t *testing.T, m *registry.Manifest, role string) (string, bool) {
	t.Helper()
	// registry.SHA256Field keeps its map unexported; read the repo manifest's own sha256 block.
	raw, err := os.ReadFile(filepath.Join(m.Dir(), "manifest.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	in := false
	for _, line := range strings.Split(string(raw), "\n") {
		switch {
		case strings.HasPrefix(line, "sha256:"):
			in = true
		case in && strings.HasPrefix(line, "  "):
			k, v, ok := strings.Cut(strings.TrimSpace(line), ":")
			if ok && strings.TrimSpace(k) == role {
				return strings.TrimSpace(v), true
			}
		case in && strings.TrimSpace(line) != "" && !strings.HasPrefix(line, "#"):
			in = false
		}
	}
	return "", false
}

func fileDigest(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Under verified mode a pulled textalign model is admitted through two records: its own folder's
// files through the rfdetr-textalign-ONNX ledger entry, and the borrowed text tower through the
// dependency that owns it. This builds that layout from the rendered manifests with stand-in bytes
// whose digests are swapped into the pins, then checks the gate admits it — and refuses it once
// the borrowed tower is replaced by a file its owner does not declare.
func TestTextalignVerifiedModeAdmitsPulledLayout(t *testing.T) {
	for _, name := range textalignEntries {
		t.Run(name, func(t *testing.T) {
			e, _ := Lookup(name)
			root := t.TempDir()
			for _, dep := range append([]string{name}, e.Dependencies...) {
				de, _ := Lookup(dep)
				src := render(t, de)
				for _, f := range de.Files {
					// Keyed by the pin, so two files the catalog pins to one digest stay equal.
					content := []byte("stand-in for " + f.SHA256 + f.LocalFilename)
					if f.SHA256 != "" {
						content = []byte("stand-in for " + f.SHA256)
					}
					mustWrite(t, filepath.Join(root, dep, f.LocalFilename), string(content))
					if f.SHA256 != "" {
						sum := sha256.Sum256(content)
						src = strings.ReplaceAll(src, f.SHA256, hex.EncodeToString(sum[:]))
					}
				}
				mustWrite(t, filepath.Join(root, dep, "manifest.yaml"), src)
			}
			// The ledger anchors digests and stand-in bytes are not in it, so for this test every
			// manifest points at a test-only audited upstream (same licence) that records none.
			for _, dir := range append([]string{name}, e.Dependencies...) {
				p := filepath.Join(root, dir, "manifest.yaml")
				b, _ := os.ReadFile(p)
				mustWrite(t, p, ledgerFreeSource(string(b)))
			}
			saved := registry.LicenseLedger
			registry.LicenseLedger = append(append([]registry.LedgerEntry(nil), saved...),
				registry.LedgerEntry{SourcePrefix: testSourceApache, License: "Apache-2.0"},
				registry.LedgerEntry{SourcePrefix: testSourceMIT, License: "MIT"})
			defer func() { registry.LicenseLedger = saved }()

			registry.EnableVerifiedMode()
			defer registry.DisableVerifiedMode()
			m, err := registry.LoadManifest(filepath.Join(root, name, "manifest.yaml"))
			if err != nil {
				t.Fatal(err)
			}
			if err := m.VerifyWeights(); err != nil {
				t.Fatalf("verified mode refuses the pulled layout: %v", err)
			}

			dep := e.Dependencies[0]
			mustWrite(t, filepath.Join(root, dep, "stray.onnx"), "unvetted")
			m.Files["text"] = "../" + dep + "/stray.onnx"
			if err := m.VerifyWeights(); err == nil || !strings.Contains(err.Error(), "does not declare") {
				t.Fatalf("a borrowed file the owner does not declare must be refused, got %v", err)
			}
		})
	}
}

const (
	testSourceApache = "https://example.test/apache/"
	testSourceMIT    = "https://example.test/mit/"
)

// ledgerFreeSource repoints source_url at the test-only upstream recording the manifest's licence.
func ledgerFreeSource(manifest string) string {
	var out []string
	for _, line := range strings.Split(manifest, "\n") {
		if strings.HasPrefix(line, "source_url:") {
			if strings.Contains(manifest, "\nlicense: MIT\n") {
				line = "source_url: " + testSourceMIT
			} else {
				line = "source_url: " + testSourceApache
			}
		}
		out = append(out, line)
	}
	return strings.Join(out, "\n")
}

// HFSubdir narrows source_url to a folder; a file outside that folder would be audited under a
// record that does not cover it.
func TestHFSubdirHoldsEveryFile(t *testing.T) {
	for _, e := range builtin {
		if e.HFSubdir == "" {
			continue
		}
		for _, f := range e.Files {
			if !strings.HasPrefix(f.HFFilename, strings.Trim(e.HFSubdir, "/")+"/") {
				t.Errorf("%s: %s is outside HFSubdir %q", e.Name, f.HFFilename, e.HFSubdir)
			}
		}
		if !strings.HasPrefix(e.SourceURL(), "https://huggingface.co/"+e.HFRepo+"/") {
			t.Errorf("%s: source_url %s is not under the repo", e.Name, e.SourceURL())
		}
	}
}
