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

	"visionserve/internal/registry"
)

// coco91 is embedded so `pull rf-detr` can write the labels file without any
// network round-trip (the file is tiny and license-clean to redistribute).
//
//go:embed labels/coco91.txt
var coco91 string

// imagenet1k embeds the standard 1000-class ILSVRC2012 labels (PyTorch order).
//
//go:embed labels/imagenet1k.txt
var imagenet1k string

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
	// DirectURL is used instead of HFRepo+HFFilename when the file is hosted
	// outside HuggingFace. Supported schemes:
	//   "gdrive://FILE_ID"  — Google Drive public file
	//   "https://..."       — any direct HTTPS URL
	// Leave empty for normal HF downloads.
	DirectURL string
	// SHA256 is the expected hex digest of the downloaded bytes. Optional, but WITHOUT it
	// a pulled model carries no content pin, and the registry's verified mode refuses to
	// load an unpinned model (registry.Manifest.VerifyWeights). Pull checks it after
	// downloading; RenderManifest writes it into the generated manifest.
	SHA256 string
}

// Normalize is the optional mean/std normalization baked into the manifest.
type Normalize struct {
	Mean []float32
	Std  []float32
}

// Entry is one model the catalog can pull.
type Entry struct {
	Name string
	// Aliases are extra names `pull` accepts for this entry (e.g. "groundingdino" for
	// "grounding-dino"). The model is always installed under Name, because composed entries
	// reference their dependencies by that directory ("../grounding-dino/...").
	Aliases      []string
	Task         string // detection | segmentation | open_vocab
	License      string // MUST be permissive (Apache-2.0/MIT/BSD)
	Architecture string // selects the model factory (registry.Manifest.Architecture)
	Description  string

	// HFRepo is the HuggingFace repo id, e.g. "PierreMarieCurie/rf-detr-onnx".
	HFRepo string
	// HFSubdir, when set, is the folder of HFRepo that holds this entry's files (each
	// File.HFFilename still gives the full path, and must lie under it). It narrows SourceURL to
	// that folder, so one repo can host several models — under different licences — and the
	// registry's ledger, which matches by the longest URL prefix, audits each folder on its own.
	HFSubdir string
	Files    []File

	// Manifest fields (mirrors registry.Manifest).
	InputWidth  int
	InputHeight int
	InputLayout string // NCHW | NHWC
	Letterbox   bool
	Crop        string // "" | "center" (manifest input.crop)
	KeepAspect  bool   // manifest input.keep_aspect (graph needs dynamic H/W)
	MultipleOf  int    // manifest input.multiple_of; 0 => omit
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
	// RuntimeThreads is the manifest's runtime.threads: role → ONNX Runtime intra-op threads for
	// that role's session. Nil leaves every role on the default. Keys must be roles of files:.
	RuntimeThreads map[string]int

	// Verified is false when the exact HF source (filenames/license) could not
	// be fully confirmed; `pull` warns the user before downloading such a model.
	Verified bool
	// Note is an optional human-readable caveat shown for unverified entries.
	Note string

	// Dependencies lists model names that must already be downloaded before this
	// virtual model can be "pulled" (e.g. grounded-sam needs grounding-dino + mobile-sam).
	Dependencies []string
	// VirtualFiles are role→relative-path pairs (relative to <modelsDir>/<name>/) that point into
	// a dependency's directory, written into the manifest's files: block. Dependencies are pulled
	// and checked first. An entry with VirtualFiles and no Files is COMPOSED: it downloads nothing
	// and its pins live in its dependencies (grounded-sam). An entry with both is PARTLY composed:
	// it downloads and pins its own Files, and borrows the rest (rfdetr-textalign-*: own detector
	// and head, a shared text tower in ../siglip-text/).
	VirtualFiles map[string]string

	// Grasp-pipeline fields (architecture "grasp").
	// Detector is the optional box-stage model name ("rf-detr", "grounding-dino", …).
	// Empty means class-agnostic (whole-image automask, no detector session).
	Detector   string
	Segmenter  string  // mask backbone; default "mobile-sam" when empty
	GripperMin float64 // default parallel-jaw opening lower bound in original-image px
	GripperMax float64 // default parallel-jaw opening upper bound in original-image px

	// Explain: optional heatmap visualization config. Nil means model does not support /api/explain.
	Explain *registry.ExplainConfig
	// Instance: optional one-shot detection config. Nil means model does not accept template prompts.
	Instance *registry.InstanceConfig
}

// textalignRepo hosts the rfdetr-textalign-* models (one folder each) and the clip-text tower
// two of them need. One repo, several licences: see Entry.HFSubdir.
const textalignRepo = "mtbui2010/rfdetr-textalign-ONNX"

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
				SHA256:        "b3321965003f11020701987a2de6e3d88f7c9a1298c1a7d4fec2d32e7f179987",
			},
		},
		InputWidth:        560,
		InputHeight:       560,
		InputLayout:       "NCHW",
		Letterbox:         false, // squash, as trained — BUGS_TO_FIX.md #1 (+1.68 mAP, full COCO val2017)
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "detr",
		BoxFormat:         "cxcywh",
		ConfThreshold:     0.5,
		MaxDetections:     300,
		LabelsFile:        "coco91.txt",
		EmbeddedLabels:    coco91,
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
	},
	{
		Name:         "grounding-dino",
		Aliases:      []string{"groundingdino", "gdino"},
		Task:         "open_vocab",
		License:      "Apache-2.0",
		Architecture: "grounding-dino",
		Description:  "Grounding DINO tiny — open-vocabulary detection (text-prompted), corrected text-mask export.",
		// The CORRECTED re-export, not onnx-community's. That export baked a Python loop's trip
		// count into the graph and only builds the text self-attention mask for the first
		// "."-separated phrase; VisionServe serves it correctly but must then run one pass per
		// class. Measured on a 12-class prompt: ~15x slower AND a different (wrong) detection
		// list. Nobody should be pulling the defective graph by default — models/grounding-dino/
		// README.md keeps the analysis and the recipe for reproducing it.
		HFRepo: "mtbui2010/grounding-dino-tiny-fixedmask-ONNX",
		Files: []File{
			{
				Role:          "model",
				HFFilename:    "model-fixedmask.onnx",
				LocalFilename: "model-fixedmask.onnx",
				ManifestRole:  "model",
				SHA256:        "ae9a0026953c6d5ce5a97b421af84c065d7f07173971cb105edb0071868fc180",
			},
			{
				// Tokenizer vocab side-file (not an ONNX graph). Resolved by the
				// model package via Manifest.Dir() at runtime.
				Role:          "vocab",
				HFFilename:    "vocab.txt",
				LocalFilename: "vocab.txt",
				SHA256:        "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
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
		RuntimePrefer:     []string{"cuda", "cpu"}, // TensorRT: -6.83 held-out mAP, BUGS_TO_FIX.md #3
		IdleUnloadSeconds: 300,
		Verified:          true,
	},
	{
		// DEPRECATED ALIAS of `grounding-dino`, which now ships these same corrected weights.
		// Kept because the name was published and manifests/scripts may already reference it;
		// it pulls the identical files into its own directory. Prefer `grounding-dino`.
		Name:         "grounding-dino-fixed",
		Task:         "open_vocab",
		License:      "Apache-2.0",
		Architecture: "grounding-dino",
		Description:  "DEPRECATED alias of grounding-dino (same corrected text-mask weights) — pull `grounding-dino` instead.",
		HFRepo:       "mtbui2010/grounding-dino-tiny-fixedmask-ONNX",
		Files: []File{
			{
				Role:          "model",
				HFFilename:    "model-fixedmask.onnx",
				LocalFilename: "model-fixedmask.onnx",
				ManifestRole:  "model",
				SHA256:        "ae9a0026953c6d5ce5a97b421af84c065d7f07173971cb105edb0071868fc180",
			},
			{
				// Same tokenizer vocab as grounding-dino, re-downloaded here so a
				// pulled model directory is self-contained (no cross-model deps).
				Role:          "vocab",
				HFFilename:    "vocab.txt",
				LocalFilename: "vocab.txt",
				SHA256:        "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
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
		RuntimePrefer:     []string{"cuda", "cpu"}, // TensorRT: -6.83 held-out mAP, BUGS_TO_FIX.md #3
		IdleUnloadSeconds: 300,
		Verified:          true,
		Note: "Re-export of IDEA-Research/grounding-dino-tiny (Apache-2.0), same 5 inputs / 2 outputs as " +
			"onnx-community/grounding-dino-tiny-ONNX. The community export baked the trip count of the " +
			"transformers-4.48 mask loop, so only the FIRST '.'-separated phrase got its attention block; " +
			"this graph rebuilds the mask with a vectorized subgraph. sha256 " +
			"ae9a0026953c6d5ce5a97b421af84c065d7f07173971cb105edb0071868fc180 (694.8 MB, opset 17). " +
			"internal/models/groundingdino probes the graph (SupportsJointTextPass) and picks the regime, " +
			"so old and new weights are both safe.",
	},
	{
		Name:         "rf-detr-nano",
		Task:         "detection",
		License:      "Apache-2.0",
		Architecture: "rf-detr",
		Description:  "RF-DETR nano (COCO) — smallest/fastest RF-DETR variant, 384×384 input, ~23 ms on GPU.",
		HFRepo:       "PierreMarieCurie/rf-detr-onnx",
		Files: []File{
			{
				Role:          "model",
				HFFilename:    "rf-detr-nano.onnx",
				LocalFilename: "rf-detr-base.onnx",
				SHA256:        "3fcbba0f68bad4939fdf1c38f432783b95691e2869af3be369780aa5be67abb2",
			},
		},
		// Letterbox false = squash, like every RF-DETR (BUGS_TO_FIX.md #1): `visionserve check`
		// against the official rf-detr-nano.pth on 200 COCO val2017 photos measured letterbox mAP
		// 40.92 vs 44.09 (rfdetr), squash 43.80 (2026-10-05). An unedited manifest from an older
		// pull is regenerated on re-pull.
		InputWidth:        384,
		InputHeight:       384,
		InputLayout:       "NCHW",
		Letterbox:         false,
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "detr",
		BoxFormat:         "cxcywh",
		ConfThreshold:     0.5,
		MaxDetections:     300,
		LabelsFile:        "coco91.txt",
		EmbeddedLabels:    coco91,
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
	},
	{
		Name:         "mobile-sam",
		Task:         "segmentation",
		License:      "Apache-2.0",
		Architecture: "mobile-sam",
		Description:  "MobileSAM — promptable segmentation (encoder + single-mask decoder).",
		HFRepo:       "Acly/MobileSAM",
		Files: []File{
			{
				Role:          "encoder",
				HFFilename:    "mobile_sam_image_encoder.onnx",
				LocalFilename: "mobile_sam_encoder.onnx",
				ManifestRole:  "encoder",
				SHA256:        "580f5fb648ea1062c0aabc26217aed56921985f03f0cbbd852bba81d760cc749",
			},
			{
				Role:          "decoder",
				HFFilename:    "sam_mask_decoder_single.onnx",
				LocalFilename: "mobile_sam_decoder_single.onnx",
				ManifestRole:  "decoder",
				SHA256:        "93915fc7c993ab9d59ab8c9ccd3bce37f7509c81ab4150a74abd4d2abbd8570d",
			},
		},
		InputWidth:        1024,
		InputHeight:       1024,
		InputLayout:       "NHWC",
		Letterbox:         false,
		PostprocessType:   "sam",
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
		Note: "Encoder input 'input_image' [H,W,3] HWC raw 0-255 (normalize+pad baked in graph). " +
			"Decoder outputs: masks (upsampled to orig size), iou_predictions, low_res_masks.",
	},
	{
		Name:         "grounded-sam",
		Task:         "open_vocab",
		License:      "Apache-2.0",
		Architecture: "grounded-sam",
		Description:  "Grounded-SAM — text-prompted segmentation (GroundingDINO → MobileSAM).",
		Dependencies: []string{"grounding-dino", "mobile-sam"},
		VirtualFiles: map[string]string{
			"gdino":   "../grounding-dino/model-fixedmask.onnx",
			"encoder": "../mobile-sam/mobile_sam_encoder.onnx",
			"decoder": "../mobile-sam/mobile_sam_decoder_single.onnx",
		},
		InputWidth:        800,
		InputHeight:       800,
		InputLayout:       "NCHW",
		Letterbox:         false,
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "grounded-sam",
		BoxFormat:         "cxcywh",
		ConfThreshold:     0.3,
		TextThreshold:     0.25,
		MaxDetections:     300,
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
	},
	{
		Name:         "rfdetr-gdino",
		Task:         "open_vocab",
		License:      "Apache-2.0",
		Architecture: "rfdetr-gdino",
		Description:  "Hybrid router — RF-DETR for in-vocabulary COCO prompts (fast), GroundingDINO for open-vocab prompts.",
		Dependencies: []string{"rf-detr", "grounding-dino"},
		VirtualFiles: map[string]string{
			"rfdetr": "../rf-detr/rf-detr-base.onnx",
			"gdino":  "../grounding-dino/model-fixedmask.onnx",
		},
		InputWidth:        560,
		InputHeight:       560,
		InputLayout:       "NCHW",
		Letterbox:         false, // squash, as trained — BUGS_TO_FIX.md #1 (+1.68 mAP, full COCO val2017)
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "detr",
		BoxFormat:         "cxcywh",
		ConfThreshold:     0.5,
		MaxDetections:     300,
		LabelsFile:        "../rf-detr/coco91.txt", // reuse rf-detr's labels (no file written; dep provides it)
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
	},
	{
		Name:         "rfdetr-gdino-sam",
		Task:         "open_vocab",
		License:      "Apache-2.0",
		Architecture: "rfdetr-gdino",
		Description:  "Hybrid router + MobileSAM masks — RF-DETR/GroundingDINO boxes → one mask per box.",
		Dependencies: []string{"rf-detr", "grounding-dino", "mobile-sam"},
		VirtualFiles: map[string]string{
			"rfdetr":  "../rf-detr/rf-detr-base.onnx",
			"gdino":   "../grounding-dino/model-fixedmask.onnx",
			"encoder": "../mobile-sam/mobile_sam_encoder.onnx",
			"decoder": "../mobile-sam/mobile_sam_decoder_single.onnx",
		},
		InputWidth:        560,
		InputHeight:       560,
		InputLayout:       "NCHW",
		Letterbox:         false, // squash, as trained — BUGS_TO_FIX.md #1 (+1.68 mAP, full COCO val2017)
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "detr",
		BoxFormat:         "cxcywh",
		ConfThreshold:     0.5,
		MaxDetections:     300,
		LabelsFile:        "../rf-detr/coco91.txt",
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
	},
	{
		Name:         "rfdetr-small-etri",
		Task:         "detection",
		License:      "Apache-2.0",
		Architecture: "rf-detr",
		Description:  "RF-DETR Small fine-tuned on 22 tabletop classes (ETRI) — 512×512, NMS-free.",
		HFRepo:       "mtbui2010/rfdetr-small-etri-ONNX",
		Files: []File{
			{Role: "model", HFFilename: "model.onnx", LocalFilename: "model.onnx", SHA256: "c0373d2b8823767955e649fe03e43c3428cd9358c8ccf0ca906cad9fc1e5d3a9"},
			{Role: "labels", HFFilename: "labels.txt", LocalFilename: "labels.txt", SHA256: "fdd0d4e9dc1965b37e46959d1fd2f3964fb407e5b176ea84a357dbeb921d183e"},
		},
		InputWidth:  512,
		InputHeight: 512,
		InputLayout: "NCHW",
		// Squash: the fine-tune used rfdetr's square_resize_div_64=True (BUGS_TO_FIX.md #1).
		Letterbox:         false,
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "detr",
		BoxFormat:         "cxcywh",
		ConfThreshold:     0.5,
		MaxDetections:     300,
		LabelsFile:        "labels.txt", // downloaded with the weights (23 rows: 22 names + N/A)
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
		Note:              "Input 'input' [1,3,512,512]; outputs dets [1,300,4] cxcywh + labels [1,300,23] sigmoid logits.",
	},
	{
		Name:         "siglip-image",
		Task:         "embed",
		License:      "Apache-2.0",
		Architecture: "siglip-image",
		Description:  "SigLIP base-patch16-224 image tower — 768-d crop embeddings (rescorer for rfdetr-gdino-siglip).",
		HFRepo:       "mtbui2010/siglip-base-patch16-224-ONNX",
		Files: []File{
			{Role: "model", HFFilename: "image/model.onnx", LocalFilename: "model.onnx", ManifestRole: "model", SHA256: "d13787ca7b0c0c3b780478f36b18bf729394752c0e868b0fc729e87122a338c9"},
			// External weights. The graph refers to this file BY NAME, so the local name is fixed.
			{Role: "weights", HFFilename: "image/model.onnx.data", LocalFilename: "model.onnx.data", SHA256: "4fbafa23edb2db76ee79def6879ea1717481380341b526890c8188363a298f84"},
		},
		InputWidth:  224,
		InputHeight: 224,
		InputLayout: "NCHW",
		// SigLIP squashes (bicubic) with mean = std = 0.5 — NOT CLIP's constants.
		Letterbox:         false,
		Normalize:         &Normalize{Mean: []float32{0.5, 0.5, 0.5}, Std: []float32{0.5, 0.5, 0.5}},
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
		Note:              "Input pixel_values [N,3,224,224]; output image_embeds [N,768], not L2-normalised.",
	},
	{
		Name:         "siglip-text",
		Task:         "embed",
		License:      "Apache-2.0",
		Architecture: "siglip-text",
		Description:  "SigLIP base-patch16-224 text tower + pure-Go tokenizer — 768-d text embeddings.",
		HFRepo:       "mtbui2010/siglip-base-patch16-224-ONNX",
		Files: []File{
			{Role: "model", HFFilename: "text/model.onnx", LocalFilename: "model.onnx", ManifestRole: "model", SHA256: "7840f8ffa18f2d4b39f836ff8773c6dd8d01621703ce1675e7a56a44322f8878"},
			{Role: "weights", HFFilename: "text/model.onnx.data", LocalFilename: "model.onnx.data", SHA256: "e8ccd846fac6ceacb8add0fa809be25bfb91cc991fd06c86d1e8450f2afc2771"},
			{Role: "tokenizer", HFFilename: "text/tokenizer.json", LocalFilename: "tokenizer.json", SHA256: "c6e405cb7c670d56636a9402c81023a55bc6c3c53d89cf02b92f5c5005bfe920"},
		},
		// No image input: the validator needs width/height > 0, so these describe the token
		// input (64 = SigLIP's context length). Nothing reads them.
		InputWidth:        64,
		InputHeight:       1,
		InputLayout:       "NCHW",
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
		Note:              "Input input_ids [N,64] padded with </s> (id 1); output text_embeds [N,768].",
	},
	{
		Name:         "rfdetr-gdino-siglip",
		Task:         "open_vocab",
		License:      "Apache-2.0",
		Architecture: "rfdetr-gdino",
		Description:  "Hybrid router + SigLIP-crop rescoring — GroundingDINO detections re-scored by SigLIP (+12.6 held-out mAP).",
		Dependencies: []string{"rf-detr", "grounding-dino", "siglip-image", "siglip-text"},
		VirtualFiles: map[string]string{
			"rfdetr": "../rf-detr/rf-detr-base.onnx",
			"gdino":  "../grounding-dino/model-fixedmask.onnx",
			"crop":   "../siglip-image/model.onnx",
			"text":   "../siglip-text/model.onnx",
		},
		InputWidth:        560,
		InputHeight:       560,
		InputLayout:       "NCHW",
		Letterbox:         false, // squash, as trained — BUGS_TO_FIX.md #1
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "detr",
		BoxFormat:         "cxcywh",
		ConfThreshold:     0.5,
		MaxDetections:     300,
		LabelsFile:        "../rf-detr/coco91.txt",
		RuntimePrefer:     []string{"cuda", "cpu"}, // every number for this entry was measured on CUDA
		IdleUnloadSeconds: 300,
		Verified:          true,
	},
	{
		Name:         "rfdetr-gdino-etri",
		Task:         "open_vocab",
		License:      "Apache-2.0",
		Architecture: "rfdetr-gdino",
		Description:  "Hybrid router, tabletop — rfdetr-small-etri for its 22 trained names, GroundingDINO for everything else.",
		Dependencies: []string{"rfdetr-small-etri", "grounding-dino"},
		VirtualFiles: map[string]string{
			"rfdetr": "../rfdetr-small-etri/model.onnx",
			"gdino":  "../grounding-dino/model-fixedmask.onnx",
		},
		InputWidth:        512,
		InputHeight:       512,
		InputLayout:       "NCHW",
		Letterbox:         false,
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "detr",
		BoxFormat:         "cxcywh",
		ConfThreshold:     0.5,
		MaxDetections:     300,
		LabelsFile:        "../rfdetr-small-etri/labels.txt",
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
	},
	{
		Name:         "rfdetr-gdino-sam-etri",
		Task:         "open_vocab",
		License:      "Apache-2.0",
		Architecture: "rfdetr-gdino",
		Description:  "Tabletop hybrid router + MobileSAM masks — one mask per box.",
		Dependencies: []string{"rfdetr-small-etri", "grounding-dino", "mobile-sam"},
		VirtualFiles: map[string]string{
			"rfdetr":  "../rfdetr-small-etri/model.onnx",
			"gdino":   "../grounding-dino/model-fixedmask.onnx",
			"encoder": "../mobile-sam/mobile_sam_encoder.onnx",
			"decoder": "../mobile-sam/mobile_sam_decoder_single.onnx",
		},
		InputWidth:        512,
		InputHeight:       512,
		InputLayout:       "NCHW",
		Letterbox:         false,
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "detr",
		BoxFormat:         "cxcywh",
		ConfThreshold:     0.5,
		MaxDetections:     300,
		LabelsFile:        "../rfdetr-small-etri/labels.txt",
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
	},
	{
		Name:         "rfdetr-gdino-siglip-etri",
		Task:         "open_vocab",
		License:      "Apache-2.0",
		Architecture: "rfdetr-gdino",
		Description:  "Tabletop hybrid router + SigLIP-crop rescoring — the configuration the +12.6 held-out mAP was measured in.",
		Dependencies: []string{"rfdetr-small-etri", "grounding-dino", "siglip-image", "siglip-text"},
		VirtualFiles: map[string]string{
			"rfdetr": "../rfdetr-small-etri/model.onnx",
			"gdino":  "../grounding-dino/model-fixedmask.onnx",
			"crop":   "../siglip-image/model.onnx",
			"text":   "../siglip-text/model.onnx",
		},
		InputWidth:        512,
		InputHeight:       512,
		InputLayout:       "NCHW",
		Letterbox:         false,
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "detr",
		BoxFormat:         "cxcywh",
		ConfThreshold:     0.5,
		MaxDetections:     300,
		LabelsFile:        "../rfdetr-small-etri/labels.txt",
		RuntimePrefer:     []string{"cuda", "cpu"}, // measured on CUDA; TensorRT rebuilds per prompt length
		IdleUnloadSeconds: 300,
		Verified:          true,
	},
	{
		Name:         "rfdetr-gdino-siglip-sam-etri",
		Task:         "open_vocab",
		License:      "Apache-2.0",
		Architecture: "rfdetr-gdino",
		Description:  "Tabletop hybrid router + SigLIP-crop rescoring + MobileSAM masks — one mask per returned box.",
		Dependencies: []string{"rfdetr-small-etri", "grounding-dino", "siglip-image", "siglip-text", "mobile-sam"},
		VirtualFiles: map[string]string{
			"rfdetr":  "../rfdetr-small-etri/model.onnx",
			"gdino":   "../grounding-dino/model-fixedmask.onnx",
			"crop":    "../siglip-image/model.onnx",
			"text":    "../siglip-text/model.onnx",
			"encoder": "../mobile-sam/mobile_sam_encoder.onnx",
			"decoder": "../mobile-sam/mobile_sam_decoder_single.onnx",
		},
		InputWidth:        512,
		InputHeight:       512,
		InputLayout:       "NCHW",
		Letterbox:         false,
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "detr",
		BoxFormat:         "cxcywh",
		ConfThreshold:     0.5,
		MaxDetections:     300,
		LabelsFile:        "../rfdetr-small-etri/labels.txt",
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
	},
	{
		Name:         "gdino-siglip",
		Task:         "open_vocab",
		License:      "Apache-2.0",
		Architecture: "gdino-siglip",
		Description:  "GroundingDINO + SigLIP-crop rescoring, no closed-set detector — every word open-vocabulary, SigLIP rejects the misses.",
		Dependencies: []string{"grounding-dino", "siglip-image", "siglip-text"},
		VirtualFiles: map[string]string{
			"gdino": "../grounding-dino/model-fixedmask.onnx",
			"crop":  "../siglip-image/model.onnx",
			"text":  "../siglip-text/model.onnx",
		},
		// Not read (GroundingDINO preprocesses itself); the validator needs a positive size.
		InputWidth:        800,
		InputHeight:       800,
		InputLayout:       "NCHW",
		Letterbox:         false,
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
	},
	{
		Name:         "gdino-siglip-sam",
		Task:         "open_vocab",
		License:      "Apache-2.0",
		Architecture: "gdino-siglip",
		Description:  "GroundingDINO + SigLIP-crop rescoring + MobileSAM masks — Grounded-SAM with a rejector before the segmenter.",
		Dependencies: []string{"grounding-dino", "siglip-image", "siglip-text", "mobile-sam"},
		VirtualFiles: map[string]string{
			"gdino":   "../grounding-dino/model-fixedmask.onnx",
			"crop":    "../siglip-image/model.onnx",
			"text":    "../siglip-text/model.onnx",
			"encoder": "../mobile-sam/mobile_sam_encoder.onnx",
			"decoder": "../mobile-sam/mobile_sam_decoder_single.onnx",
		},
		InputWidth:        800,
		InputHeight:       800,
		InputLayout:       "NCHW",
		Letterbox:         false,
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
	},
	{
		// The text half of CLIP ViT-B/32 (models/clip ships the image half). First-party export,
		// published next to the textalign models that need it: no public ONNX of it existed.
		Name:         "clip-text",
		Task:         "embed",
		License:      "MIT",
		Architecture: "clip-text",
		Description:  "CLIP ViT-B/32 text tower + pure-Go BPE tokenizer — 512-d text embeddings, same space as clip (MIT, OpenAI).",
		HFRepo:       textalignRepo,
		HFSubdir:     "clip-text",
		Files: []File{
			{Role: "model", HFFilename: "clip-text/model.onnx", LocalFilename: "model.onnx", ManifestRole: "model", SHA256: "a104b96e1a9ce466e24dac4e32f406ffc412eb1a459049b4040eab97b196b580"},
			// The BPE tokenizer reads these two from the model directory (internal/models/clip).
			{Role: "vocab", HFFilename: "clip-text/vocab.json", LocalFilename: "vocab.json", SHA256: "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363"},
			{Role: "merges", HFFilename: "clip-text/merges.txt", LocalFilename: "merges.txt", SHA256: "f526393189112391ce6f9795d4695f704121ce452c3aad1f5335cc41337eba85"},
			// MIT asks for the notice to travel with copies.
			{Role: "license", HFFilename: "clip-text/LICENSE", LocalFilename: "LICENSE", SHA256: "987e63b32f6c89ff5160e429458a872ff048e6860b590a3912e938f9da8f14db"},
		},
		// No image input: the validator needs width/height > 0, so these describe the token input
		// (77 = CLIP's context length). Nothing reads them.
		InputWidth:        77,
		InputHeight:       1,
		InputLayout:       "NCHW",
		PostprocessType:   "embed",
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
		Note:              "Input input_ids [N,77] int64 (exported from openai/clip-vit-base-patch32 CLIPTextModelWithProjection, opset 18); output text_embeds [N,512].",
	},
	{
		// Text-aligned open-vocabulary heads on RF-DETR Small (internal/models/textalign). Each
		// downloads its own detector, projection and head.onnx, and borrows a text tower from a
		// dependency. The generated manifests mirror models/rfdetr-textalign-*/manifest.yaml in
		// the repo (TestTextalignEntriesMatchRepoManifests).
		Name:         "rfdetr-textalign-dec1",
		Task:         "open_vocab",
		License:      "Apache-2.0",
		Architecture: "rfdetr-textalign",
		Description:  "RF-DETR Small (22 tabletop classes, partial fine-tune) + CLIP-aligned head — open-vocabulary; tuned for method=gated.",
		HFRepo:       textalignRepo,
		HFSubdir:     "dec1",
		Dependencies: []string{"clip-text"},
		VirtualFiles: map[string]string{"text": "../clip-text/model.onnx"},
		Files: []File{
			{Role: "rfdetr", HFFilename: "dec1/detector.onnx", LocalFilename: "detector.onnx", ManifestRole: "rfdetr", SHA256: "cd4cb2166978635de3ab2323ed0d7198cc1ae5b77dc6e21c4c90845877579125"},
			{Role: "head", HFFilename: "dec1/head.onnx", LocalFilename: "head.onnx", ManifestRole: "head", SHA256: "ab6db90d0e921931be6303d06c2691ee777ce56fd9a7750e7741f9a8d667b3ab"},
			{Role: "proj", HFFilename: "dec1/proj.bin", LocalFilename: "proj.bin", SHA256: "1d81d692de4c7cd3b3345cfe317afd78f752be0347181798847709cc3a2aff2b"},
			{Role: "labels", HFFilename: "dec1/labels.txt", LocalFilename: "labels.txt", SHA256: "fdd0d4e9dc1965b37e46959d1fd2f3964fb407e5b176ea84a357dbeb921d183e"},
			{Role: "templates", HFFilename: "dec1/templates.txt", LocalFilename: "templates.txt", SHA256: "dd37dd428e8c0700e26b86a6c7701a9e50a26a9c32b932febdbcb9ebb45c663c"},
		},
		InputWidth:      512,
		InputHeight:     512,
		InputLayout:     "NCHW",
		Letterbox:       false, // squash, as RF-DETR is trained (square_resize_div_64) — BUGS_TO_FIX.md #1
		Normalize:       &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType: "detr",
		BoxFormat:       "cxcywh",
		// F1 peak (0.768) for method gated on the 62 held-out images. Under the default method
		// exact the thresholded score is head B's, on another scale: few or no detections.
		ConfThreshold:     0.35,
		MaxDetections:     300,
		LabelsFile:        "labels.txt", // the detector's own order: gated finds N/A by position
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		RuntimeThreads:    map[string]int{"head": 1}, // ~1 ms head; ORT's default pool slows the detector ~3x on CPU
		Verified:          true,
		Note:              "Detector input 'input' [1,3,512,512]; outputs dets [1,300,4], labels [1,300,23], query_feats [1,300,256] (in that order). proj.bin P [512,256]; head.onnx is proj.bin as a graph.",
	},
	{
		Name:         "rfdetr-textalign-dec1-siglip",
		Task:         "open_vocab",
		License:      "Apache-2.0",
		Architecture: "rfdetr-textalign",
		Description:  "RF-DETR Small (17 tabletop classes, 5 held out) + SigLIP-distilled head — open-vocabulary; tuned for method=gated.",
		HFRepo:       textalignRepo,
		HFSubdir:     "dec1-siglip",
		Dependencies: []string{"siglip-text"},
		VirtualFiles: map[string]string{"text": "../siglip-text/model.onnx"},
		Files: []File{
			{Role: "rfdetr", HFFilename: "dec1-siglip/detector.onnx", LocalFilename: "detector.onnx", ManifestRole: "rfdetr", SHA256: "7de8ca150390b8e5d64d4541695a6b167c76793fdc0887e60c7e1481572c5a09"},
			{Role: "head", HFFilename: "dec1-siglip/head.onnx", LocalFilename: "head.onnx", ManifestRole: "head", SHA256: "ec724f1a1c338795e1db37dcb9892d27b8ffb6d1f69f2c47c0c2558f281cce5e"},
			{Role: "proj", HFFilename: "dec1-siglip/proj.bin", LocalFilename: "proj.bin", SHA256: "302640c92684e78b64e7c0fd89b4f1c2761184408c7735dbb74ab43257372c85"},
			// 17 base names + N/A — must match the detector's class tensor width (18).
			{Role: "labels", HFFilename: "dec1-siglip/labels.txt", LocalFilename: "labels.txt", SHA256: "321bf1eb6803aa638016b48b7597f4fd56e73df11c36ebcf52dacbea65daa787"},
			{Role: "templates", HFFilename: "dec1-siglip/templates.txt", LocalFilename: "templates.txt", SHA256: "dd37dd428e8c0700e26b86a6c7701a9e50a26a9c32b932febdbcb9ebb45c663c"},
			// All 22 names, for pasting into a prompt; not read by the model.
			{Role: "vocab", HFFilename: "dec1-siglip/vocab-all22.txt", LocalFilename: "vocab-all22.txt", SHA256: "fdd0d4e9dc1965b37e46959d1fd2f3964fb407e5b176ea84a357dbeb921d183e"},
		},
		InputWidth:      512,
		InputHeight:     512,
		InputLayout:     "NCHW",
		Letterbox:       false,
		Normalize:       &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType: "detr",
		BoxFormat:       "cxcywh",
		// Inherited from rfdetr-textalign-dec1 (method gated); not re-tuned for this detector.
		ConfThreshold:     0.35,
		MaxDetections:     300,
		LabelsFile:        "labels.txt",
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		RuntimeThreads:    map[string]int{"head": 1},
		Verified:          true,
		Note:              "Detector outputs dets [1,300,4], labels [1,300,18], query_feats [1,300,256]. proj.bin P [768,256] in siglip-text space.",
	},
	{
		Name:         "rfdetr-textalign-dec1-siglip-prod",
		Task:         "open_vocab",
		License:      "Apache-2.0",
		Architecture: "rfdetr-textalign",
		Description:  "RF-DETR Small (22 tabletop classes, partial fine-tune) + SigLIP-space head — open-vocabulary, conf_threshold 0.001.",
		HFRepo:       textalignRepo,
		HFSubdir:     "dec1-siglip-prod",
		Dependencies: []string{"siglip-text"},
		VirtualFiles: map[string]string{"text": "../siglip-text/model.onnx"},
		Files: []File{
			// Same bytes as dec1/detector.onnx.
			{Role: "rfdetr", HFFilename: "dec1-siglip-prod/detector.onnx", LocalFilename: "detector.onnx", ManifestRole: "rfdetr", SHA256: "cd4cb2166978635de3ab2323ed0d7198cc1ae5b77dc6e21c4c90845877579125"},
			{Role: "head", HFFilename: "dec1-siglip-prod/head.onnx", LocalFilename: "head.onnx", ManifestRole: "head", SHA256: "3f19241baa19cafb2f673a101f639aba55f445a944d055516745ab966e7fb804"},
			{Role: "proj", HFFilename: "dec1-siglip-prod/proj.bin", LocalFilename: "proj.bin", SHA256: "57ceaa1539bca398f3e495353f1761422594c11b8c683064a650a4fc6dcea91c"},
			{Role: "labels", HFFilename: "dec1-siglip-prod/labels.txt", LocalFilename: "labels.txt", SHA256: "fdd0d4e9dc1965b37e46959d1fd2f3964fb407e5b176ea84a357dbeb921d183e"},
			{Role: "templates", HFFilename: "dec1-siglip-prod/templates.txt", LocalFilename: "templates.txt", SHA256: "dd37dd428e8c0700e26b86a6c7701a9e50a26a9c32b932febdbcb9ebb45c663c"},
		},
		InputWidth:        512,
		InputHeight:       512,
		InputLayout:       "NCHW",
		Letterbox:         false,
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "detr",
		BoxFormat:         "cxcywh",
		ConfThreshold:     0.001, // as the local manifest: every query comes back, the client thresholds
		MaxDetections:     300,
		LabelsFile:        "labels.txt",
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		RuntimeThreads:    map[string]int{"head": 1},
		Verified:          true,
		Note:              "Detector outputs dets [1,300,4], labels [1,300,23], query_feats [1,300,256]. proj.bin P [768,256] in siglip-text space.",
	},
	{
		Name:         "rfdetr-textalign-etri",
		Task:         "open_vocab",
		License:      "Apache-2.0",
		Architecture: "rfdetr-textalign",
		Description:  "Frozen rfdetr-small-etri (22 tabletop classes) + CLIP-aligned head — open-vocabulary, with /api/explain.",
		HFRepo:       textalignRepo,
		HFSubdir:     "etri",
		Dependencies: []string{"clip-text"},
		VirtualFiles: map[string]string{"text": "../clip-text/model.onnx"},
		Files: []File{
			// The qf export minus cross_attn_weights: 59 MB per inference the head never reads.
			{Role: "rfdetr", HFFilename: "etri/detector-noattn.onnx", LocalFilename: "detector-noattn.onnx", ManifestRole: "rfdetr", SHA256: "efcf3af08d5e0512095946b69866425645fca81dc5e05aaea775eff9abc11b4c"},
			// The un-stripped export, for /api/explain only: not in textalign's Roles(), so it is
			// opened lazily on the first explain call.
			{Role: "explain", HFFilename: "etri/detector-qf.onnx", LocalFilename: "detector-qf.onnx", ManifestRole: "explain", SHA256: "5d87e22067458c9af1f679a8eeb85588569a84881a060ac0f1d8f8f252379818"},
			{Role: "head", HFFilename: "etri/head.onnx", LocalFilename: "head.onnx", ManifestRole: "head", SHA256: "5484c2cdd32deb74776b9b8d7b9621354a330ac417eee2ba4cf0b497333f3c38"},
			{Role: "proj", HFFilename: "etri/proj.bin", LocalFilename: "proj.bin", SHA256: "314302bf7549d85ef2467375fa2d415dca2eb1dbfee6bf6e3f8efddafd89392b"},
			{Role: "labels", HFFilename: "etri/labels.txt", LocalFilename: "labels.txt", SHA256: "fdd0d4e9dc1965b37e46959d1fd2f3964fb407e5b176ea84a357dbeb921d183e"},
			{Role: "templates", HFFilename: "etri/templates.txt", LocalFilename: "templates.txt", SHA256: "dd37dd428e8c0700e26b86a6c7701a9e50a26a9c32b932febdbcb9ebb45c663c"},
		},
		InputWidth:      512,
		InputHeight:     512,
		InputLayout:     "NCHW",
		Letterbox:       false,
		Normalize:       &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType: "detr",
		BoxFormat:       "cxcywh",
		// F1 peak (0.730) for method exact on the 62 held-out images; this head's confidence
		// never reaches 0.5, so the detector's usual 0.5 returns nothing.
		ConfThreshold:     0.28,
		MaxDetections:     300,
		LabelsFile:        "labels.txt",
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		RuntimeThreads:    map[string]int{"head": 1},
		Explain: &registry.ExplainConfig{Type: "attention", Role: "explain",
			Outputs: map[string]string{"attention": "cross_attn_weights"}, SpatialStride: 16},
		Verified: true,
		Note:     "Detector input 'input' [1,3,512,512]; outputs dets, labels [1,300,23], query_feats [1,300,256]. proj.bin P [512,256], trained with the unit-norm penalty.",
	},
	{
		Name:         "background",
		Task:         "segmentation",
		License:      "Apache-2.0",
		Architecture: "background",
		Description:  "Background (class-agnostic support surface) — table/floor mask via method=depth (MiDaS plane, default) | sam | cv | automask.",
		Dependencies: []string{"mobile-sam", "midas"},
		Segmenter:    "mobile-sam",
		VirtualFiles: map[string]string{
			"encoder": "../mobile-sam/mobile_sam_encoder.onnx",
			"decoder": "../mobile-sam/mobile_sam_decoder_single.onnx",
			"depth":   "../midas/midas_v21_small_256.onnx",
		},
		InputWidth:        1024,
		InputHeight:       1024,
		InputLayout:       "NCHW",
		Letterbox:         false,
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
	},
	{
		Name:         "grasp",
		Task:         "grasp",
		License:      "Apache-2.0",
		Architecture: "grasp",
		Description:  "Grasp (class-agnostic) — MobileSAM automask → analytic mask2grasp. No detector; grasps cover every object.",
		Dependencies: []string{"mobile-sam"},
		Segmenter:    "mobile-sam",
		GripperMin:   10,
		GripperMax:   150,
		VirtualFiles: map[string]string{
			"encoder": "../mobile-sam/mobile_sam_encoder.onnx",
			"decoder": "../mobile-sam/mobile_sam_decoder_single.onnx",
		},
		InputWidth:        1024,
		InputHeight:       1024,
		InputLayout:       "NCHW",
		Letterbox:         false,
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
	},
	{
		Name:         "grasp-gd",
		Task:         "grasp",
		License:      "Apache-2.0",
		Architecture: "grasp",
		Description:  "Grasp (open-vocab) — GroundingDINO text → boxes → MobileSAM masks → analytic mask2grasp. Requires a text prompt.",
		Dependencies: []string{"grounding-dino", "mobile-sam"},
		Detector:     "grounding-dino",
		Segmenter:    "mobile-sam",
		GripperMin:   10,
		GripperMax:   150,
		VirtualFiles: map[string]string{
			"det":     "../grounding-dino/model-fixedmask.onnx",
			"encoder": "../mobile-sam/mobile_sam_encoder.onnx",
			"decoder": "../mobile-sam/mobile_sam_decoder_single.onnx",
		},
		InputWidth:        800,
		InputHeight:       800,
		InputLayout:       "NCHW",
		Letterbox:         false,
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		ConfThreshold:     0.3,
		TextThreshold:     0.25,
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
	},
	{
		Name:         "grasp-rfdetr",
		Task:         "grasp",
		License:      "Apache-2.0",
		Architecture: "grasp",
		Description:  "Grasp (class-aware COCO-80) — RF-DETR detect → MobileSAM segment → analytic mask2grasp.",
		Dependencies: []string{"rf-detr", "mobile-sam"},
		Detector:     "rf-detr",
		Segmenter:    "mobile-sam",
		GripperMin:   10,
		GripperMax:   150,
		VirtualFiles: map[string]string{
			"det":     "../rf-detr/rf-detr-base.onnx",
			"encoder": "../mobile-sam/mobile_sam_encoder.onnx",
			"decoder": "../mobile-sam/mobile_sam_decoder_single.onnx",
		},
		// Letterbox false = squash, as RF-DETR is trained and as the rf-detr entry serves the same
		// weights (BUGS_TO_FIX.md #1): on 200 COCO val2017 photos this model's detections scored
		// mAP 45.50 letterboxed, 47.77 squashed (= rf-detr), 2026-10-05. An unedited manifest from
		// an older pull is regenerated on re-pull.
		InputWidth:        560,
		InputHeight:       560,
		InputLayout:       "NCHW",
		Letterbox:         false,
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "detr",
		BoxFormat:         "cxcywh",
		ConfThreshold:     0.5,
		MaxDetections:     300,
		LabelsFile:        "../rf-detr/coco91.txt",
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
	},
	{
		Name:         "rt-detr",
		Task:         "detection",
		License:      "Apache-2.0",
		Architecture: "rt-detr",
		Description:  "RT-DETR-l (COCO) — real-time NMS-free detector, 640×640. UPSTREAM GONE, see Note.",
		HFRepo:       "onnx-community/RT-DETR-l-hf",
		Files: []File{
			{Role: "model", HFFilename: "onnx/model.onnx", LocalFilename: "model.onnx"},
		},
		// Squash + [0, 1] with no mean/std, as RT-DETR is trained (RTDetrImageProcessor
		// do_normalize false); letterbox + ImageNet mean/std scored 7.16 mAP vs 50.40 on
		// rtdetr_r50vd (models/rt-detr/manifest.yaml, BUGS_TO_FIX.md #1).
		InputWidth:        640,
		InputHeight:       640,
		InputLayout:       "NCHW",
		Letterbox:         false,
		Normalize:         &Normalize{Mean: []float32{0, 0, 0}, Std: []float32{1, 1, 1}},
		PostprocessType:   "rt-detr",
		BoxFormat:         "cxcywh",
		ConfThreshold:     0.5,
		MaxDetections:     300,
		LabelsFile:        "coco80.txt",
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          false,
		Note: "UPSTREAM REMOVED: huggingface.co/onnx-community/RT-DETR-l-hf returns 401 as of " +
			"2026-08-18, so this entry cannot be pulled and cannot be sha256-pinned. Candidate " +
			"replacements exist under different names (onnx-community/rtdetr_r50vd, rtdetr_v2_r18vd-ONNX). " +
			"Do NOT repoint blindly — a different export can change input/output names and the " +
			"postprocess contract; verify the real tensor shapes first (CLAUDE.md).",
	},
	{
		Name:         "efficient-sam",
		Task:         "segmentation",
		License:      "Apache-2.0",
		Architecture: "efficient-sam",
		Description:  "EfficientSAM ViT-Tiny — promptable segmentation, lighter than MobileSAM.",
		HFRepo:       "yunyangx/EfficientSAM",
		Files: []File{
			{Role: "encoder", HFFilename: "efficientsam_ti_encoder.onnx", LocalFilename: "efficient_sam_encoder.onnx", ManifestRole: "encoder", SHA256: "84ed466ffcc5c1f8d08409bc34a23bb364ab2c15e402cb12d4335a42be0e0951"},
			{Role: "decoder", HFFilename: "efficientsam_ti_decoder.onnx", LocalFilename: "efficient_sam_decoder.onnx", ManifestRole: "decoder", SHA256: "a62f8fa5ea080447c0689418d69e58f1e83e0b7adf9c142e2bd9bcc8045c0b11"},
		},
		InputWidth:        1024,
		InputHeight:       1024,
		InputLayout:       "NCHW",
		PostprocessType:   "sam",
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
		Note: "Encoder input 'batched_images' [batch,3,H,W] NCHW ImageNet-norm. " +
			"Decoder inputs: image_embeddings, batched_point_coords [1,1,N,2] float32, " +
			"batched_point_labels [1,1,N] float32, orig_im_size [2] int64. " +
			"Output 'output_masks' 5-D [1,1,M,H,W], iou_predictions [1,1,M].",
	},
	{
		Name:         "sam2",
		Task:         "segmentation",
		License:      "Apache-2.0",
		Architecture: "sam2",
		Description:  "SAM2-Tiny — promptable segmentation with multi-scale features (Meta AI).",
		HFRepo:       "SharpAI/sam2-hiera-tiny-onnx",
		Files: []File{
			{Role: "encoder", HFFilename: "encoder.onnx", LocalFilename: "sam2_tiny_encoder.onnx", ManifestRole: "encoder", SHA256: "df265cb552475e1b3a6cb57c939e57c95ed849bfc2f985c06efab85d8bca6db9"},
			{Role: "decoder", HFFilename: "decoder.onnx", LocalFilename: "sam2_tiny_decoder.onnx", ManifestRole: "decoder", SHA256: "63198f1f1e273d8f2f4a9d1baf926e53a01d78dc50e0674640e1513dc00d9927"},
		},
		InputWidth:        1024,
		InputHeight:       1024,
		InputLayout:       "NCHW",
		PostprocessType:   "sam",
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
		Note: "Encoder input 'image' [1,3,1024,1024] NCHW ImageNet-norm. " +
			"Encoder outputs: image_embed [1,256,64,64], high_res_feats_0 [1,32,256,256], high_res_feats_1 [1,64,128,128]. " +
			"Decoder point_labels are float32 (not int64); mask_input/has_mask_input required.",
	},
	{
		Name:         "depth-anything-v2",
		Task:         "depth",
		License:      "Apache-2.0",
		Architecture: "depth-anything-v2",
		Description:  "Depth Anything V2 small — monocular depth estimation, ~518 keep-aspect (Apache-2.0; Base/Large are CC-BY-NC).",
		// First-party export with DYNAMIC H/W (the onnx-community repo it replaces is gone, 401).
		// Produced and verified by `visionserve convert hf depth-anything/Depth-Anything-V2-Small-hf`:
		// ONNX vs PyTorch 1.05e-05 at three input shapes; served map vs predicted_depth Pearson
		// r >= 0.9999 on 7 non-square photos.
		HFRepo: "mtbui2010/depth-anything-v2-small-ONNX",
		Files: []File{
			{Role: "model", HFFilename: "model.onnx", LocalFilename: "model.onnx",
				SHA256: "4e456781eac92f7f8e79da50f721f65eb1876a10ed90d59ab2b7d7df04f59e71"},
		},
		InputWidth:  518,
		InputHeight: 518,
		InputLayout: "NCHW",
		Letterbox:   false,
		// DPTImageProcessor's own geometry (keep_aspect_ratio, ensure_multiple_of 14): an 848x480
		// photo is fed as 910x518, not squashed to 518x518. Needs a graph with dynamic H/W.
		KeepAspect:        true,
		MultipleOf:        14,
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "depth",
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
		Note:              "Input 'pixel_values' [1,3,H,W] (H, W multiples of 14), output 'predicted_depth' [1,H,W].",
	},
	{
		Name:         "midas",
		Task:         "depth",
		License:      "MIT",
		Architecture: "midas",
		Description:  "MiDaS v2.1-small — monocular depth estimation, 256×256, MIT license.",
		HFRepo:       "Heliosoph/midas-small-onnx",
		Files: []File{
			{Role: "model", HFFilename: "midas_v21_small_256.onnx", LocalFilename: "midas_v21_small_256.onnx", SHA256: "b0a5b3f12625137e626805167907fe0410665bec671685d59daaa2daab19f977"},
		},
		InputWidth:        256,
		InputHeight:       256,
		InputLayout:       "NCHW",
		Letterbox:         false,
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "depth",
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
	},
	{
		Name:         "efficientnet-b0",
		Task:         "classification",
		License:      "Apache-2.0",
		Architecture: "efficientnet",
		Description:  "EfficientNet-B0 — ImageNet-1k classification, 224×224.",
		HFRepo:       "onnxmodelzoo/efficientnet_b0_Opset17",
		Files: []File{
			{Role: "model", HFFilename: "efficientnet_b0_Opset17.onnx", LocalFilename: "model.onnx", SHA256: "e76596a2b9e27c7c734c38550859105b43fec926f13447a84dad175eb994068a"},
		},
		InputWidth:        224,
		InputHeight:       224,
		InputLayout:       "NCHW",
		Letterbox:         false,
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "classification",
		MaxDetections:     5,
		LabelsFile:        "imagenet1k.txt",
		EmbeddedLabels:    imagenet1k,
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
		Note:              "Input 'x' [1,3,224,224], output logits [1,1000]. imagenet1k.txt written by pull.",
	},
	{
		Name:         "mobilenet-v3",
		Task:         "classification",
		License:      "Apache-2.0",
		Architecture: "mobilenet-v3",
		Description:  "MobileNetV3-Small — ImageNet-1k classification, 224×224, ultra-lightweight.",
		HFRepo:       "onnxmodelzoo/mobilenet_v3_small_Opset17",
		Files: []File{
			{Role: "model", HFFilename: "mobilenet_v3_small_Opset17.onnx", LocalFilename: "model.onnx", SHA256: "9152343d120cf7b03b6b775a5fccd53813cc21891e376060a8edd2dfc0c35193"},
		},
		InputWidth:        224,
		InputHeight:       224,
		InputLayout:       "NCHW",
		Letterbox:         false,
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "classification",
		MaxDetections:     5,
		LabelsFile:        "imagenet1k.txt",
		EmbeddedLabels:    imagenet1k,
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
		Note:              "Input 'x' [1,3,224,224], output logits [1,1000]. imagenet1k.txt written by pull.",
	},
	{
		Name:         "clip",
		Task:         "embed",
		License:      "MIT",
		Architecture: "clip",
		Description:  "CLIP ViT-B/32 image encoder — 512-d embeddings for zero-shot classification (MIT, OpenAI).",
		// Short-side resize + centre crop, as CLIP is trained: squashing measured cosine 0.88-0.92
		// against the reference embedding on non-square photos, the crop 0.998.
		HFRepo: "khasinski/clip-ViT-B-32-onnx",
		Files: []File{
			{Role: "model", HFFilename: "visual.onnx", LocalFilename: "model.onnx", SHA256: "78e896b2c7301d01eda84e280d7c7297299aa6f8bacc0f5f8fe5bd60d42d8aae"},
		},
		InputWidth:        224,
		InputHeight:       224,
		InputLayout:       "NCHW",
		Letterbox:         false,
		Crop:              "center",
		Normalize:         &Normalize{Mean: []float32{0.48145466, 0.4578275, 0.40821073}, Std: []float32{0.26862954, 0.26130258, 0.27577711}},
		PostprocessType:   "embed",
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
		Note:              "Input 'input' [batch,3,224,224], output 'output' [batch,512] float32. V1: image encoder only (512-d L2-normalized). Text encoder planned for v2.",
	},
	{
		Name:         "nano-sam",
		Task:         "segmentation",
		License:      "Apache-2.0",
		Architecture: "nano-sam",
		Description:  "NanoSAM — NVIDIA edge-optimized SAM (ResNet-18 encoder), Apache-2.0.",
		// Weights are hosted on Google Drive (not HuggingFace) by NVIDIA.
		// DirectURL uses the "gdrive://FILE_ID" scheme; pull handles the download.
		HFRepo: "",
		Files: []File{
			{
				Role:          "encoder",
				DirectURL:     "gdrive://14-SsvoaTl-esC3JOzomHDnI9OGgdO2OR",
				LocalFilename: "resnet18_image_encoder.onnx",
				ManifestRole:  "encoder",
			},
			{
				Role:          "decoder",
				DirectURL:     "gdrive://1jYNvnseTL49SNRx9PDcbkZ9DwsY8up7n",
				LocalFilename: "mobile_sam_mask_decoder.onnx",
				ManifestRole:  "decoder",
			},
		},
		InputWidth:        1024,
		InputHeight:       1024,
		InputLayout:       "NCHW",
		PostprocessType:   "sam",
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
		Note: "Encoder: resnet18_image_encoder.onnx (input 'image' [1,3,1024,1024] NCHW ImageNet-norm, " +
			"output 'image_embeddings' [1,256,64,64]). " +
			"Decoder: mobile_sam_mask_decoder.onnx (inputs: image_embeddings, point_coords, " +
			"point_labels, mask_input, has_mask_input; NO orig_im_size input; " +
			"outputs: iou_predictions [1,M], low_res_masks [1,M,256,256]).",
	},
	{
		Name:         "scrfd",
		Task:         "detection",
		License:      "MIT",
		Architecture: "scrfd",
		Description:  "SCRFD-10GF — InsightFace face detector, 640×640, with keypoints, MIT license.",
		HFRepo:       "cromsc/scrfd-10g",
		Files: []File{
			{Role: "model", HFFilename: "scrfd_10g_bnkps.onnx", LocalFilename: "det_10g.onnx", SHA256: "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"},
		},
		InputWidth:        640,
		InputHeight:       640,
		InputLayout:       "NCHW",
		Letterbox:         true,
		Normalize:         &Normalize{Mean: []float32{127.5, 127.5, 127.5}, Std: []float32{128.0, 128.0, 128.0}},
		PostprocessType:   "scrfd",
		ConfThreshold:     0.5,
		MaxDetections:     1000,
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
		Note: "Input 'input.1' [1,3,H,W] dynamic. 9 outputs (3 strides × scores+bbox+kps): " +
			"shape-routed (12800=s8, 3200=s16, 800=s32). kps outputs [N,10] are 5 keypoints (unused in v1). " +
			"cromsc/scrfd-10g has no explicit HF license; upstream InsightFace detection is MIT.",
	},
	{
		Name:         "paddle-ocr",
		Task:         "detection",
		License:      "Apache-2.0",
		Architecture: "paddle-ocr",
		Description:  "PP-OCRv4 — Chinese+English OCR (text detection + recognition).",
		HFRepo:       "webnn/PP-OCRv4-ONNX",
		Files: []File{
			{Role: "det", HFFilename: "ch_PP-OCRv4_det.onnx", LocalFilename: "det_model.onnx", ManifestRole: "det", SHA256: "30a86f5731181461d08021402766601e4302a9b9b9666be8aff402696339cdff"},
			{Role: "rec", HFFilename: "ch_PP-OCRv4_rec.onnx", LocalFilename: "rec_model.onnx", ManifestRole: "rec", SHA256: "06b3e6af6c59a1ba5d53790ed8c2e4b2de389870b6cf5a97f349f3412cb269c0"},
			{Role: "keys", HFFilename: "ch_PP-OCR_keys_v1.txt", LocalFilename: "ppocr_keys_v1.txt"},
		},
		InputWidth:        960,
		InputHeight:       960,
		InputLayout:       "NCHW",
		Letterbox:         false,
		PostprocessType:   "paddle-ocr",
		ConfThreshold:     0.3,
		RuntimePrefer:     []string{"cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
		Note: "Det: input 'x' [1,3,H,W] dynamic, output 'sigmoid_0.tmp_0' [1,1,H,W]. " +
			"Rec: input 'x' [1,3,48,W], output 'softmax_11.tmp_0' [1,T,6625]. " +
			"ppocr_keys_v1.txt must be present in model dir. V1: axis-aligned boxes only.",
	},
}

// List returns all catalog entries sorted by name.
func List() []Entry {
	out := make([]Entry, len(builtin))
	copy(out, builtin)
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// Lookup returns the catalog entry for name or one of its aliases (ok=false if not found).
func Lookup(name string) (Entry, bool) {
	for _, e := range builtin {
		if e.Name == name {
			return e, true
		}
	}
	for _, e := range builtin {
		for _, a := range e.Aliases {
			if a == name {
				return e, true
			}
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
