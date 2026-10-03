package catalog

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"visionserve/internal/registry"
)

// A catalog entry whose upstream appears in the maintainer-audited ledger is one we intend to
// be usable under verified mode. That mode refuses a manifest declaring no sha256, and refuses
// one whose license disagrees with the ledger — both at LOAD time, i.e. after a several-hundred-
// megabyte download has already succeeded and the user believes the model is installed. This
// test moves that discovery to `go test`.
//
// Entries with no ledger entry are skipped on purpose: verified mode refuses them by design,
// and that is a deployment choice rather than a catalog defect.
func TestAuditedEntriesArePinned(t *testing.T) {
	for _, e := range builtin {
		src := e.SourceURL()
		if src == "" {
			continue
		}
		led := longestLedgerMatch(src)
		if led == nil {
			continue
		}
		if !e.Verified {
			// `pull` already warns loudly for these, and an entry whose upstream has vanished
			// cannot be pinned at all — the bytes to hash no longer exist. Its Note carries
			// the reason.
			continue
		}
		t.Run(e.Name, func(t *testing.T) {
			if led.License != e.License {
				t.Errorf("declares %q but the audited ledger records %q for %s — verified mode "+
					"refuses this at load", e.License, led.License, led.SourcePrefix)
			}
			if !strings.Contains(render(t, e), "sha256:") {
				t.Errorf("upstream %s is audited, but the entry pins no sha256 — verified mode "+
					"refuses an unpinned model, so `visionserve pull %s` would download "+
					"successfully and then fail to load", led.SourcePrefix, e.Name)
			}
		})
	}
}

// The catalog pins the digest `pull` verifies after download; the ledger pins the digest the
// LOADER trusts, because it is compiled into the binary and a manifest on disk is not. Those are
// two copies of the same fact, so they can drift — and drifting silently is the bad case: `pull`
// would happily install a model the loader then refuses. Keep them equal here.
func TestCatalogPinsMatchTheLedger(t *testing.T) {
	for _, e := range builtin {
		led := longestLedgerMatch(e.SourceURL())
		if led == nil {
			continue
		}
		audited := make(map[string]bool, len(led.WeightSHA256))
		for _, d := range led.WeightSHA256 {
			audited[strings.ToLower(d)] = true
		}
		for _, f := range e.Files {
			if f.SHA256 == "" {
				continue
			}
			if !audited[strings.ToLower(f.SHA256)] {
				t.Errorf("%s: catalog pins %s… for %s, but %s has no such digest in "+
					"registry.LicenseLedger — `pull` would install a model the loader refuses",
					e.Name, f.SHA256[:12], f.LocalFilename, led.SourcePrefix)
			}
		}
	}
}

func longestLedgerMatch(src string) *registry.LedgerEntry {
	var best *registry.LedgerEntry
	for i := range registry.LicenseLedger {
		p := registry.LicenseLedger[i].SourcePrefix
		if strings.HasPrefix(src, p) && (best == nil || len(p) > len(best.SourcePrefix)) {
			best = &registry.LicenseLedger[i]
		}
	}
	return best
}

// Review #8: every file the catalog pins must also be pinned by the manifest `pull` writes —
// including files with no ManifestRole (ONNX external data, tokenizer, labels), which used to be
// verified at download time and then never again. Parsed through the registry, the same path the
// loader takes, so the rendered YAML shape is checked too.
func TestRenderedManifestPinsEveryPinnedFile(t *testing.T) {
	for _, e := range builtin {
		if len(e.VirtualFiles) > 0 {
			continue // composed: the pins live in the dependencies
		}
		var pinned []File
		for _, f := range e.Files {
			if f.SHA256 != "" {
				pinned = append(pinned, f)
			}
		}
		if len(pinned) == 0 {
			continue
		}
		t.Run(e.Name, func(t *testing.T) {
			p := filepath.Join(t.TempDir(), "manifest.yaml")
			if err := os.WriteFile(p, []byte(render(t, e)), 0o644); err != nil {
				t.Fatal(err)
			}
			m, err := registry.LoadManifest(p)
			if err != nil {
				t.Fatalf("rendered manifest does not parse: %v", err)
			}
			have := map[string]bool{}
			for _, d := range m.PinnedDigests() {
				have[strings.ToLower(d)] = true
			}
			for _, f := range pinned {
				if !have[strings.ToLower(f.SHA256)] {
					t.Errorf("%s is pinned in the catalog but not in the generated manifest — the loader never re-checks it",
						f.LocalFilename)
				}
			}
		})
	}
}
