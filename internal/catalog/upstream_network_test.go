//go:build network

// This file is behind the `network` build tag so it NEVER runs in a normal `go test ./...`:
// it talks to huggingface.co, and a unit-test run must stay offline and deterministic.
//
// Run it explicitly (a few seconds, no weights are downloaded):
//
//	go test -tags network ./internal/catalog/ -run TestCatalogUpstreamsReachable -v
//
// WHY IT EXISTS. A catalog entry's upstream can vanish (repo deleted/gated → 401) or a file can
// be renamed inside a live repo, and nothing in the codebase notices: the breakage only surfaces
// as a failed `visionserve pull` in a user's terminal. That is exactly how
// onnx-community/RT-DETR-l-hf and onnx-community/depth-anything-v2-small-hf went dead
// unnoticed. This test is the cheap periodic check that keeps the curated list honest.

package catalog

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"
)

// hfModelInfoURL is the metadata endpoint for a repo. It returns the repo's file listing
// ("siblings") in one small JSON response — orders of magnitude cheaper than touching the
// weights, and enough to validate both the repo and every declared filename.
func hfModelInfoURL(repo string) string {
	return "https://huggingface.co/api/models/" + repo
}

// hfModelInfo is the subset of the HF model-info payload this test needs.
type hfModelInfo struct {
	Siblings []struct {
		RFilename string `json:"rfilename"`
	} `json:"siblings"`
}

// requestPause keeps the run polite: entries are checked sequentially with a short gap so the
// whole sweep is one request per repo, spread over a couple of seconds.
const requestPause = 150 * time.Millisecond

func TestCatalogUpstreamsReachable(t *testing.T) {
	client := &http.Client{Timeout: 15 * time.Second}

	var skipped, offHub []string
	for _, e := range builtin {
		if e.HFRepo == "" {
			// No HF upstream to check: either a virtual entry (grounded-sam & friends,
			// composed from already-pulled dependencies) or one hosted elsewhere
			// (nano-sam ships from Google Drive via File.DirectURL).
			if len(e.Files) > 0 {
				offHub = append(offHub, e.Name)
			}
			continue
		}
		if !e.Verified {
			// Known-broken by design: the entry is already flagged Verified:false and carries a
			// Note explaining why, and `pull` warns before downloading it. Do not fail on these
			// — but keep them visible so the dead list does not quietly grow.
			skipped = append(skipped, fmt.Sprintf("%s (%s)", e.Name, e.HFRepo))
			t.Run(e.Name, func(t *testing.T) {
				t.Skipf("Verified:false, upstream not checked — %s: %s",
					e.HFRepo, firstSentence(e.Note))
			})
			continue
		}

		t.Run(e.Name, func(t *testing.T) { checkUpstream(t, client, e) })
		time.Sleep(requestPause)
	}

	if len(skipped) > 0 {
		t.Logf("SKIPPED %d entry/entries marked Verified:false (upstream known broken): %s",
			len(skipped), strings.Join(skipped, ", "))
	}
	if len(offHub) > 0 {
		t.Logf("NOT CHECKED, not HuggingFace-hosted (File.DirectURL): %s", strings.Join(offHub, ", "))
	}
}

// checkUpstream asserts that one entry's HF repo still exists and still contains every file the
// entry declares. It downloads no weights — only the repo's small model-info JSON.
func checkUpstream(t *testing.T, client *http.Client, e Entry) {
	t.Helper()

	resp, err := client.Get(hfModelInfoURL(e.HFRepo))
	if err != nil {
		t.Fatalf("model %q: upstream %s unreachable: %v (is the machine offline?)",
			e.Name, e.HFRepo, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("model %q: upstream https://huggingface.co/%s returned HTTP %d (%s) — the repo "+
			"is gone, renamed or gated, so `visionserve pull %s` is broken. Fix: repoint HFRepo "+
			"to a live mirror AFTER verifying its real tensor shapes, or mark the entry "+
			"Verified:false with a Note explaining it.",
			e.Name, e.HFRepo, resp.StatusCode, http.StatusText(resp.StatusCode), e.Name)
	}

	var info hfModelInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		t.Fatalf("model %q: cannot decode %s model-info JSON: %v", e.Name, e.HFRepo, err)
	}
	have := make(map[string]bool, len(info.Siblings))
	for _, s := range info.Siblings {
		have[s.RFilename] = true
	}

	for _, f := range e.Files {
		if f.DirectURL != "" || f.HFFilename == "" {
			// Hosted outside HuggingFace (gdrive:// or a raw HTTPS URL) — not listed in this
			// repo's siblings, so there is nothing to cross-check here.
			continue
		}
		if !have[f.HFFilename] {
			t.Errorf("model %q: file %q (role %q) is NOT in https://huggingface.co/%s "+
				"(HTTP 200, %d files listed) — it was renamed or removed upstream; "+
				"`visionserve pull %s` will 404 on this file.",
				e.Name, f.HFFilename, f.Role, e.HFRepo, len(info.Siblings), e.Name)
		}
	}
}

// firstSentence trims a long catalog Note down to its first sentence for the skip message.
func firstSentence(note string) string {
	note = strings.TrimSpace(note)
	if i := strings.Index(note, ". "); i > 0 {
		return note[:i+1]
	}
	if len(note) > 160 {
		return note[:160] + "…"
	}
	return note
}
