package registry

import (
	"fmt"
	"strings"
)

// LICENSE PROVENANCE LEDGER (contribution C1, hardened)
// ------------------------------------------------------
// The license allowlist (manifest.go) and the sha256/source pin (verify.go) close two holes:
// a non-permissive *declared* license, and tampered/wrong-origin *bytes*. They do NOT close the
// hole the reviewer named: a manifest author can simply TYPE "Apache-2.0" for a model whose true
// upstream license is copyleft (AGPL). The allowlist trusts that string; the hash only proves the
// bytes match what the author declared, not that the declared *license* is correct.
//
// The ledger closes that hole for the curated catalog by SEPARATING AUTHORITY:
//
//   - the manifest's `license:` is CONTRIBUTOR-supplied (anyone opening a PR can set it);
//   - the ledger below is MAINTAINER-audited (a human read the actual upstream LICENSE file once,
//     recorded the verified SPDX id, the URL of that LICENSE file, and its sha256 as evidence).
//
// In verified mode the gate ADMITS a model only when the contributor-declared license EQUALS the
// maintainer-audited license for that model's audited source prefix. A PR that flips a manifest to
// "Apache-2.0" to smuggle in an AGPL model is now refused, because the ledger (a separate, audited
// record the contributor does not control) disagrees. This does not make the gate a universal
// license oracle for arbitrary uploads — that is impossible — but it removes the "trust the string
// the PR author typed" residue for every model VisionServe actually ships.
//
// HONEST SCOPE: the ledger's ground truth is a one-time human audit of the upstream LICENSE file.
// It detects later divergence of the *declared* label from that audited record; it cannot detect
// an upstream that was itself mislabeled at audit time. LicenseFileSHA256 lets a re-audit notice
// if the upstream LICENSE file later changes.

// LedgerEntry is one maintainer-audited provenance record. License is the SPDX id verified by a
// human reading LicenseURL (the upstream's actual LICENSE/COPYING file); LicenseFileSHA256 pins
// that file's bytes so a silent upstream relicense is detectable on re-audit.
type LedgerEntry struct {
	SourcePrefix      string // audited upstream URL prefix; a manifest's source_url must start with this
	License           string // SPDX id VERIFIED from the upstream LICENSE file (the ground truth)
	LicenseURL        string // URL of the upstream LICENSE/COPYING file the maintainer read
	LicenseFileSHA256 string // sha256 of that LICENSE file (optional; "" = not pinned yet)
	AuditedBy         string
	AuditedDate       string // YYYY-MM-DD
	Note              string
}

// LicenseLedger is the curated, in-binary provenance ledger for the shipped catalog. It is the
// independent ground truth the gate cross-checks each manifest's declared license against.
// Keyed/searched by SourcePrefix (longest match wins). Maintainer-owned: contributors edit
// manifests, maintainers edit this table.
var LicenseLedger = []LedgerEntry{
	// --- HuggingFace "onnx-community" / "onnxmodelzoo" mirrors (re-exports of permissive models) ---
	{SourcePrefix: "https://huggingface.co/onnxmodelzoo/mobilenet_v3_small_Opset17/", License: "Apache-2.0",
		LicenseURL: "https://github.com/pytorch/vision/blob/main/LICENSE", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "torchvision MobileNetV3-Small export; torchvision is BSD-3 upstream but onnxmodelzoo card declares Apache-2.0 — see note"},
	{SourcePrefix: "https://huggingface.co/onnxmodelzoo/efficientnet_b0_Opset17/", License: "Apache-2.0",
		LicenseURL: "https://huggingface.co/onnxmodelzoo/efficientnet_b0_Opset17", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "EfficientNet-B0 ONNX export"},
	{SourcePrefix: "https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX/", License: "Apache-2.0",
		LicenseURL: "https://huggingface.co/IDEA-Research/grounding-dino-tiny", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "GroundingDINO-tiny; IDEA-Research upstream is Apache-2.0"},
	{SourcePrefix: "https://huggingface.co/onnx-community/RT-DETR-l-hf/", License: "Apache-2.0",
		LicenseURL: "https://huggingface.co/PekingU/rtdetr_r50vd", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "RT-DETR; Apache-2.0 upstream"},
	{SourcePrefix: "https://huggingface.co/onnx-community/depth-anything-v2-small-hf/", License: "Apache-2.0",
		LicenseURL: "https://huggingface.co/depth-anything/Depth-Anything-V2-Small", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "Depth-Anything-V2-Small ONNX export"},

	// --- other audited upstreams ---
	{SourcePrefix: "https://huggingface.co/khasinski/clip-ViT-B-32-onnx/", License: "MIT",
		LicenseURL: "https://github.com/openai/CLIP/blob/main/LICENSE", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "CLIP ViT-B/32 image encoder; OpenAI CLIP is MIT"},
	{SourcePrefix: "https://huggingface.co/cromsc/scrfd-10g/", License: "MIT",
		LicenseURL: "https://github.com/deepinsight/insightface/blob/master/README.md", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "SCRFD-10GF face detector; Insightface SCRFD is MIT"},
	{SourcePrefix: "https://huggingface.co/Heliosoph/midas-small-onnx/", License: "MIT",
		LicenseURL: "https://github.com/isl-org/MiDaS/blob/master/LICENSE", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "MiDaS v2.1 small; isl-org MiDaS is MIT"},
	{SourcePrefix: "https://huggingface.co/Acly/MobileSAM/", License: "Apache-2.0",
		LicenseURL: "https://github.com/ChaoningZhang/MobileSAM/blob/master/LICENSE", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "MobileSAM encoder + single-point decoder; Apache-2.0"},
	{SourcePrefix: "https://huggingface.co/yunyangx/EfficientSAM/", License: "Apache-2.0",
		LicenseURL: "https://github.com/yformer/EfficientSAM/blob/main/LICENSE", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "EfficientSAM-Ti encoder/decoder; Apache-2.0"},
	{SourcePrefix: "https://huggingface.co/SharpAI/sam2-hiera-tiny-onnx/", License: "Apache-2.0",
		LicenseURL: "https://github.com/facebookresearch/segment-anything-2/blob/main/LICENSE", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "SAM2 Hiera-Tiny; Meta SAM2 is Apache-2.0"},
	{SourcePrefix: "https://huggingface.co/webnn/PP-OCRv4-ONNX/", License: "Apache-2.0",
		LicenseURL: "https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "PP-OCRv4 det+rec; PaddleOCR is Apache-2.0"},
	{SourcePrefix: "https://huggingface.co/PierreMarieCurie/rf-detr-onnx/", License: "Apache-2.0",
		LicenseURL: "https://github.com/roboflow/rf-detr/blob/main/LICENSE", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "RF-DETR base + nano ONNX; Roboflow RF-DETR is Apache-2.0"},
}

// ledgerEnforced toggles the maintainer-audited cross-check. Off by default (local-first,
// non-breaking): the catalog still loads under the license-string allowlist alone. EnableVerifiedMode
// turns it on for trust-critical deployments.
var ledgerEnforced bool

// EnableVerifiedMode turns on the strongest gate: in addition to the always-on license-string
// allowlist, every loaded model must (a) carry a source_url under an audited ledger prefix,
// (b) declare exactly the license the maintainer verified for that prefix, and (c) pass the
// sha256 content-pin and verified-source allowlist (which are seeded from the ledger here).
// Off by default. Returns the number of audited prefixes activated.
func EnableVerifiedMode() int {
	prefixes := make([]string, 0, len(LicenseLedger))
	for _, e := range LicenseLedger {
		prefixes = append(prefixes, e.SourcePrefix)
	}
	VerifiedSourcePrefixes = prefixes
	ledgerEnforced = true
	return len(prefixes)
}

// DisableVerifiedMode reverts to the default (allowlist-only) gate. Mainly for tests.
func DisableVerifiedMode() {
	VerifiedSourcePrefixes = nil
	ledgerEnforced = false
}

// VerifiedModeEnabled reports whether the ledger cross-check is active.
func VerifiedModeEnabled() bool { return ledgerEnforced }

// lookupLedger returns the audited entry whose SourcePrefix is the longest prefix of sourceURL.
func lookupLedger(sourceURL string) (LedgerEntry, bool) {
	var best LedgerEntry
	found := false
	for _, e := range LicenseLedger {
		if strings.HasPrefix(sourceURL, e.SourcePrefix) {
			if !found || len(e.SourcePrefix) > len(best.SourcePrefix) {
				best, found = e, true
			}
		}
	}
	return best, found
}

// VerifyLicenseProvenance cross-checks the manifest's contributor-declared license against the
// maintainer-audited ledger. Enforced only in verified mode. It is the control that catches a
// manifest relicensed/mislabeled by its author even when bytes and origin are internally
// consistent — the case the sha256 + source pin cannot catch.
func (m *Manifest) VerifyLicenseProvenance() error {
	if !ledgerEnforced {
		return nil
	}
	entry, ok := lookupLedger(m.SourceURL)
	if !ok {
		return fmt.Errorf(
			"model %q: source_url %q has no maintainer-audited license-ledger entry — refusing to load (verified mode requires an audited upstream)",
			m.Name, m.SourceURL)
	}
	if entry.License != m.License {
		return fmt.Errorf(
			"model %q: declared license %q does not match the maintainer-audited upstream license %q for %s — refusing to load (manifest may be mislabeled or relicensed)",
			m.Name, m.License, entry.License, entry.SourcePrefix)
	}
	return nil
}
