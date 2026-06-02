// Package catalog defines a built-in, curated list of permissively-licensed
// computer-vision models that `visionserve pull` can download from the
// HuggingFace Hub into the local model registry (Ollama-style).
//
// The catalog is intentionally a small Go data structure (no remote registry —
// see CLAUDE.md MVP scope). Each Entry carries everything needed to:
//   - download the model's ONNX weights (+ side files) from a public HF repo,
//   - write a manifest.yaml matching internal/registry.Manifest so that
//     `visionserve list` / `run` work afterwards,
//   - optionally materialize a small embedded labels file (e.g. coco91.txt).
//
// LICENSE POLICY (CLAUDE.md principle #1): ONLY permissive licenses
// (Apache-2.0 / MIT / BSD). No AGPL models (YOLO/Ultralytics, FastSAM,
// YOLO-World) are ever listed here.
package catalog

import (
	_ "embed"
	"fmt"
	"sort"
)

// coco91 is embedded so `pull rf-detr` can write the labels file without any
// network round-trip (the file is tiny and license-clean to redistribute).
//
//go:embed labels/coco91.txt
var coco91 string

// File describes one downloadable artifact of a model.
type File struct {
	// Role is the logical role used by multi-session models (e.g. "encoder",
	// "decoder", "model", "vocab"). For single-file models use "model".
	Role string
	// HFFilename is the path of the file inside the HF repo, relative to the
	// repo root (may contain a subfolder, e.g. "onnx/model.onnx").
	HFFilename string
	// LocalFilename is the name written under <modelsdir>/<name>/.
	LocalFilename string
	// ManifestRole, if non-empty, is the role key written into the manifest's
	// `files:` map. Empty means the file is not an ONNX session referenced by
	// the manifest `files`/`model_file` (e.g. a vocab side-file is downloaded
	// but is not an ONNX graph). For single-file models, ManifestRole is empty
	// and the file is wired via Manifest.ModelFile instead.
	ManifestRole string
}

// Normalize is the optional mean/std normalization baked into the manifest.
type Normalize struct {
	Mean []float32
	Std  []float32
}

// Entry is one model the catalog can pull.
type Entry struct {
	Name         string
	Task         string // detection | segmentation | open_vocab
	License      string // MUST be permissive (Apache-2.0/MIT/BSD)
	Architecture string // selects the model factory (registry.Manifest.Architecture)
	Description  string

	// HFRepo is the HuggingFace repo id, e.g. "PierreMarieCurie/rf-detr-onnx".
	HFRepo string
	Files  []File

	// Manifest fields (mirrors registry.Manifest).
	InputWidth  int
	InputHeight int
	InputLayout string // NCHW | NHWC
	Letterbox   bool
	Normalize   *Normalize

	PostprocessType string  // detr | sam | grounding-dino ...
	BoxFormat       string  // e.g. cxcywh
	ConfThreshold   float64 // 0 => omit
	TextThreshold   float64 // 0 => omit (GroundingDINO)
	MaxDetections   int     // 0 => omit

	// LabelsFile is the local labels filename to write (e.g. "coco91.txt"),
	// empty if none. EmbeddedLabels holds the content to write for it.
	LabelsFile     string
	EmbeddedLabels string

	RuntimePrefer     []string
	IdleUnloadSeconds int

	// Verified is false when the exact HF source (filenames/license) could not
	// be fully confirmed; `pull` warns the user before downloading such a model.
	Verified bool
	// Note is an optional human-readable caveat shown for unverified entries.
	Note string
}

// builtin is the curated catalog. Keep entries permissive-only.
var builtin = []Entry{
	{
		Name:         "rf-detr",
		Task:         "detection",
		License:      "Apache-2.0",
		Architecture: "rf-detr",
		Description:  "RF-DETR base (COCO) — NMS-free DETR detector.",
		HFRepo:       "PierreMarieCurie/rf-detr-onnx",
		Files: []File{
			{
				Role:          "model",
				HFFilename:    "rf-detr-base-coco.onnx",
				LocalFilename: "rf-detr-base.onnx",
			},
		},
		InputWidth:        560,
		InputHeight:       560,
		InputLayout:       "NCHW",
		Letterbox:         true,
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "detr",
		BoxFormat:         "cxcywh",
		ConfThreshold:     0.5,
		MaxDetections:     300,
		LabelsFile:        "coco91.txt",
		EmbeddedLabels:    coco91,
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
	},
	{
		Name:         "grounding-dino",
		Task:         "open_vocab",
		License:      "Apache-2.0",
		Architecture: "grounding-dino",
		Description:  "Grounding DINO tiny — open-vocabulary detection (text-prompted).",
		HFRepo:       "onnx-community/grounding-dino-tiny-ONNX",
		Files: []File{
			{
				Role:          "model",
				HFFilename:    "onnx/model.onnx",
				LocalFilename: "model.onnx",
				ManifestRole:  "model",
			},
			{
				// Tokenizer vocab side-file (not an ONNX graph). Resolved by the
				// model package via Manifest.Dir() at runtime.
				Role:          "vocab",
				HFFilename:    "vocab.txt",
				LocalFilename: "vocab.txt",
			},
		},
		InputWidth:        800,
		InputHeight:       800,
		InputLayout:       "NCHW",
		Letterbox:         false,
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "grounding-dino",
		BoxFormat:         "cxcywh",
		ConfThreshold:     0.3,
		TextThreshold:     0.25,
		MaxDetections:     300,
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
	},
	{
		Name:         "mobile-sam",
		Task:         "segmentation",
		License:      "Apache-2.0",
		Architecture: "mobile-sam",
		Description:  "MobileSAM — promptable segmentation (encoder + single-mask decoder).",
		// NOTE: no public HF repo was found that ships BOTH samexporter files
		// (mobile_sam_encoder.onnx + mobile_sam_decoder_single.onnx) under a
		// clearly stated permissive license. We use a best-effort source that
		// provides the encoder under the exact name and a compatible decoder,
		// mapping HF filenames -> the local names the manifest expects. This
		// entry is marked Verified:false so `pull` warns before downloading.
		HFRepo: "hkn1304/mobilesam-onnx",
		Files: []File{
			{
				Role:          "encoder",
				HFFilename:    "mobile_sam_encoder.onnx",
				LocalFilename: "mobile_sam_encoder.onnx",
				ManifestRole:  "encoder",
			},
			{
				Role:          "decoder",
				HFFilename:    "mobile_sam_decoder.onnx",
				LocalFilename: "mobile_sam_decoder_single.onnx",
				ManifestRole:  "decoder",
			},
		},
		InputWidth:        1024,
		InputHeight:       1024,
		InputLayout:       "NHWC",
		Letterbox:         false,
		PostprocessType:   "sam",
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          false,
		Note: "HF source for the samexporter single-mask decoder could not be license-verified; " +
			"filenames are mapped best-effort. Confirm the upstream MobileSAM license (Apache-2.0) before production use.",
	},
}

// List returns all catalog entries sorted by name.
func List() []Entry {
	out := make([]Entry, len(builtin))
	copy(out, builtin)
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// Lookup returns the catalog entry for name (ok=false if not found).
func Lookup(name string) (Entry, bool) {
	for _, e := range builtin {
		if e.Name == name {
			return e, true
		}
	}
	return Entry{}, false
}

// Names returns the sorted list of catalog model names (for error messages).
func Names() []string {
	names := make([]string, 0, len(builtin))
	for _, e := range builtin {
		names = append(names, e.Name)
	}
	sort.Strings(names)
	return names
}

// UnknownModelError builds a helpful error listing available names.
func UnknownModelError(name string) error {
	return fmt.Errorf("unknown model %q; available models to pull: %v", name, Names())
}
