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

	// WeightSHA256 is the SET of weight digests the maintainer recorded for this upstream.
	//
	// It exists because the manifest's own `sha256:` is self-referential: it lives in the same
	// file an attacker with write access to the model directory would rewrite, so it proves the
	// bytes match what the manifest CLAIMS, never that the claim itself is honest. This table is
	// compiled into the binary, like the licence record above, and is therefore outside that
	// attacker's reach. In verified mode every digest a manifest declares must appear here, so
	// forging a pin for substituted weights now requires forging an entry in the binary.
	//
	// It is a SET rather than a filename→digest map on purpose: local filenames are not unique
	// per upstream (`rf-detr` and `rf-detr-nano` both write `rf-detr-base.onnx` from the same
	// repo), and the property worth enforcing is "these bytes are bytes a maintainer audited",
	// not "this role holds this particular file". The manifest's own role→digest pin still
	// catches a swap BETWEEN two audited files; the two checks compose.
	//
	// Empty means this upstream has no recorded digests and only the manifest-level pin applies.
	WeightSHA256 []string
}

// LicenseLedger is the curated, in-binary provenance ledger for the shipped catalog. It is the
// independent ground truth the gate cross-checks each manifest's declared license against.
// Keyed/searched by SourcePrefix (longest match wins). Maintainer-owned: contributors edit
// manifests, maintainers edit this table.
var LicenseLedger = []LedgerEntry{
	// --- HuggingFace "onnx-community" / "onnxmodelzoo" mirrors (re-exports of permissive models) ---
	{SourcePrefix: "https://huggingface.co/onnxmodelzoo/mobilenet_v3_small_Opset17/", License: "Apache-2.0",
		LicenseURL: "https://github.com/pytorch/vision/blob/main/LICENSE", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "torchvision MobileNetV3-Small export; torchvision is BSD-3 upstream but onnxmodelzoo card declares Apache-2.0 — see note",
		WeightSHA256: []string{
			"9152343d120cf7b03b6b775a5fccd53813cc21891e376060a8edd2dfc0c35193",
		},
	},
	{SourcePrefix: "https://huggingface.co/onnxmodelzoo/efficientnet_b0_Opset17/", License: "Apache-2.0",
		LicenseURL: "https://huggingface.co/onnxmodelzoo/efficientnet_b0_Opset17", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "EfficientNet-B0 ONNX export",
		WeightSHA256: []string{
			"e76596a2b9e27c7c734c38550859105b43fec926f13447a84dad175eb994068a",
		},
	},
	{SourcePrefix: "https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX/", License: "Apache-2.0",
		LicenseURL: "https://huggingface.co/IDEA-Research/grounding-dino-tiny", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "GroundingDINO-tiny; IDEA-Research upstream is Apache-2.0"},
	{SourcePrefix: "https://huggingface.co/mtbui2010/grounding-dino-tiny-fixedmask-ONNX/", License: "Apache-2.0",
		LicenseURL: "https://huggingface.co/IDEA-Research/grounding-dino-tiny", AuditedBy: "tmbui", AuditedDate: "2026-08-18",
		Note: "FIRST-PARTY re-export of the SAME IDEA-Research weights audited on the line above — " +
			"only the ONNX graph differs (correct block-diagonal text mask). The permissive licence " +
			"therefore rests on the identical upstream LICENSE, not on a new third party. Auditing " +
			"a repo you control is weaker evidence than auditing a stranger's: what it certifies is " +
			"the weight lineage, and that is exactly what the sha256 pin then binds.",
		WeightSHA256: []string{
			"07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
			"ae9a0026953c6d5ce5a97b421af84c065d7f07173971cb105edb0071868fc180",
		},
	},
	{SourcePrefix: "https://huggingface.co/onnx-community/RT-DETR-l-hf/", License: "Apache-2.0",
		LicenseURL: "https://huggingface.co/PekingU/rtdetr_r50vd", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "RT-DETR; Apache-2.0 upstream"},
	{SourcePrefix: "https://huggingface.co/onnx-community/depth-anything-v2-small-hf/", License: "Apache-2.0",
		LicenseURL: "https://huggingface.co/depth-anything/Depth-Anything-V2-Small", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "Depth-Anything-V2-Small ONNX export"},

	// --- other audited upstreams ---
	{SourcePrefix: "https://huggingface.co/khasinski/clip-ViT-B-32-onnx/", License: "MIT",
		LicenseURL: "https://github.com/openai/CLIP/blob/main/LICENSE", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "CLIP ViT-B/32 image encoder; OpenAI CLIP is MIT",
		WeightSHA256: []string{
			"78e896b2c7301d01eda84e280d7c7297299aa6f8bacc0f5f8fe5bd60d42d8aae",
		},
	},
	{SourcePrefix: "https://huggingface.co/cromsc/scrfd-10g/", License: "MIT",
		LicenseURL: "https://github.com/deepinsight/insightface/blob/master/README.md", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "SCRFD-10GF face detector; Insightface SCRFD is MIT",
		WeightSHA256: []string{
			"5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
		},
	},
	{SourcePrefix: "https://huggingface.co/Heliosoph/midas-small-onnx/", License: "MIT",
		LicenseURL: "https://github.com/isl-org/MiDaS/blob/master/LICENSE", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "MiDaS v2.1 small; isl-org MiDaS is MIT",
		WeightSHA256: []string{
			"b0a5b3f12625137e626805167907fe0410665bec671685d59daaa2daab19f977",
		},
	},
	{SourcePrefix: "https://huggingface.co/Acly/MobileSAM/", License: "Apache-2.0",
		LicenseURL: "https://github.com/ChaoningZhang/MobileSAM/blob/master/LICENSE", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "MobileSAM encoder + single-point decoder; Apache-2.0",
		WeightSHA256: []string{
			"580f5fb648ea1062c0aabc26217aed56921985f03f0cbbd852bba81d760cc749",
			"93915fc7c993ab9d59ab8c9ccd3bce37f7509c81ab4150a74abd4d2abbd8570d",
		},
	},
	{SourcePrefix: "https://huggingface.co/yunyangx/EfficientSAM/", License: "Apache-2.0",
		LicenseURL: "https://github.com/yformer/EfficientSAM/blob/main/LICENSE", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "EfficientSAM-Ti encoder/decoder; Apache-2.0",
		WeightSHA256: []string{
			"84ed466ffcc5c1f8d08409bc34a23bb364ab2c15e402cb12d4335a42be0e0951",
			"a62f8fa5ea080447c0689418d69e58f1e83e0b7adf9c142e2bd9bcc8045c0b11",
		},
	},
	{SourcePrefix: "https://huggingface.co/SharpAI/sam2-hiera-tiny-onnx/", License: "Apache-2.0",
		LicenseURL: "https://github.com/facebookresearch/segment-anything-2/blob/main/LICENSE", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "SAM2 Hiera-Tiny; Meta SAM2 is Apache-2.0",
		WeightSHA256: []string{
			"63198f1f1e273d8f2f4a9d1baf926e53a01d78dc50e0674640e1513dc00d9927",
			"df265cb552475e1b3a6cb57c939e57c95ed849bfc2f985c06efab85d8bca6db9",
		},
	},
	{SourcePrefix: "https://huggingface.co/webnn/PP-OCRv4-ONNX/", License: "Apache-2.0",
		LicenseURL: "https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "PP-OCRv4 det+rec; PaddleOCR is Apache-2.0",
		WeightSHA256: []string{
			"06b3e6af6c59a1ba5d53790ed8c2e4b2de389870b6cf5a97f349f3412cb269c0",
			"30a86f5731181461d08021402766601e4302a9b9b9666be8aff402696339cdff",
		},
	},
	{SourcePrefix: "https://huggingface.co/PierreMarieCurie/rf-detr-onnx/", License: "Apache-2.0",
		LicenseURL: "https://github.com/roboflow/rf-detr/blob/main/LICENSE", AuditedBy: "tmbui", AuditedDate: "2026-06-09",
		Note: "RF-DETR base + nano ONNX; Roboflow RF-DETR is Apache-2.0",
		WeightSHA256: []string{
			"3fcbba0f68bad4939fdf1c38f432783b95691e2869af3be369780aa5be67abb2",
			"b3321965003f11020701987a2de6e3d88f7c9a1298c1a7d4fec2d32e7f179987",
		},
	},
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
