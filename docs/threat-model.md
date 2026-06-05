# VisionServe license-gate: threat model (contribution C1)

VisionServe enforces a **permissive-license allowlist at model-load time**, inside the inference
runtime, offline, in a single binary. This document states precisely what that gate defends
against, what it does **not** guarantee, and where it sits relative to existing tooling. The
claims below are exercised by the reproducible demo `TestGate_AdversarialDemo`
(`internal/registry/gate_demo_test.go`; run `go test ./internal/registry -run TestGate_AdversarialDemo -v`).

## Why a gate is needed (measured motivation)

Declared licenses on model hubs are, empirically, not trustworthy:

- **95.8% of models lack the required license text** and only **3.2% satisfy both license-text and
  copyright requirements**; only **5.75% of downstream applications preserve a compliant model
  notice** — over 124,278 dataset→model→app chains (Jewitt et al., "Permissive-Washing",
  arXiv:2602.08816, 2026).
- **35.5% of model→application transitions strip restrictive license clauses** by relicensing as
  permissive (Jewitt et al., "License Drift", arXiv:2509.09873, 2025; 1.6M models audited).

The specific copyleft VisionServe forbids (AGPL-3.0, e.g. Ultralytics YOLO / FastSAM / YOLO-World)
virally relicenses any network service that serves it; a permissive project that loads one is
contaminated. The gate makes that load **impossible by construction**, not by audit-after-the-fact.

## Controls and what each catches

| Adversarial condition | Control | Enforcement point | Demo case |
|---|---|---|---|
| A model under a non-permissive / copyleft license (AGPL) is declared | **License allowlist** (Apache-2.0 / MIT / BSD only) | `manifest.validate()` — registry scan, before load | `A_agpl_license_refused` |
| Weights are swapped/tampered but the manifest keeps a permissive label ("relabeling") | **Content pin (sha256)** — computed digest must equal the declared digest | `Manifest.VerifyWeights()` — load time, after weights exist | `B_hash_mismatch_refused` |
| Weights come from an unvetted mirror (right hash, wrong origin) | **Verified-source allowlist** (`source_url` must start with an audited prefix; opt-in) | `Manifest.VerifyWeights()` / `checkSourceAllowlist` | `C_unaudited_source_refused` |
| Permissive license + correct bytes + audited origin | — (admitted) | both checks pass | `0_baseline_admitted` |

Together: the gate **binds a declared-permissive license to specific bytes from an audited origin**,
closing the relabeling and wrong-origin holes — entirely offline, no network call.

## What the gate does NOT guarantee (honest limits)

- It still **trusts the declared license string itself**. If an upstream author mislabels a truly
  copyleft model as Apache-2.0 *and* the bytes/origin are consistent, the gate admits it. The
  sha256 + source allowlist mitigate this by binding the label to **audited** bytes/origin (a human
  vets the prefix once), but the gate is a **policy-enforcement point, not a license oracle**. It is
  therefore "load-time license *policy* enforcement, hardened by content-hashing and a verified
  source", **not** a "machine-checkable license guarantee".
- It does not detect license obligations that require source/weight disclosure post-hoc; it prevents
  the load, which is the relevant control for an edge deployer.
- sha256 is integrity, not authenticity — pair with model signing (below) for signer identity.

## Position vs existing tooling (the novelty boundary)

License/provenance enforcement today lives in two places; **neither is the model loader**:

- **Build/CI-time SCA scanners** (ScanCode, FOSSology, Snyk, GitLab License Compliance): scan
  *source dependency trees* at build time, emit reports/SBOMs, leave the decision to a human/policy
  file. They do not run at model-load time and do not gate weights.
- **Deploy-time admission controllers** (Kyverno, OPA Gatekeeper, Sigstore policy-controller):
  verify *container image* signatures/attestations at K8s pod admission — integrity/identity of
  containers, not a *license policy* over model weights.
- **Model signing** (OpenSSF / Sigstore Model Signing v1.0): proves *who signed* and *that bytes are
  unmodified* — provenance/integrity, **not** license policy. VisionServe's sha256+source pin is a
  lightweight, framework-free cousin and is **complementary**: a future hook can require an OMS/
  Sigstore attestation in addition to the license+hash gate.

To our knowledge, **no inference server enforces a license-allowlist policy at model-load time,
offline, in a single binary.** That is contribution C1.

## Regulatory alignment (why now)

EU Cyber Resilience Act mandates machine-readable SBOMs with integrity (hash) + license fields
(reporting from Sep 2026); EU AI Act GPAI transparency obligations from Aug 2026. A load-time
hash+license gate is a concrete, edge-deployable enforcement point for these regimes.
