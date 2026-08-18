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

	// Dependencies lists model names that must already be downloaded before this
	// virtual model can be "pulled" (e.g. grounded-sam needs grounding-dino + mobile-sam).
	Dependencies []string
	// VirtualFiles, if non-nil, makes Pull skip all network downloads and instead
	// write a manifest whose files: block contains these role→relative-path pairs
	// (relative to <modelsDir>/<name>/). Dependencies are checked first.
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
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
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
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
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
		InputWidth:        384,
		InputHeight:       384,
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
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
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
			"gdino":   "../grounding-dino/model.onnx",
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
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
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
			"gdino":  "../grounding-dino/model.onnx",
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
		LabelsFile:        "../rf-detr/coco91.txt", // reuse rf-detr's labels (no file written; dep provides it)
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
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
			"gdino":   "../grounding-dino/model.onnx",
			"encoder": "../mobile-sam/mobile_sam_encoder.onnx",
			"decoder": "../mobile-sam/mobile_sam_decoder_single.onnx",
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
		LabelsFile:        "../rf-detr/coco91.txt",
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
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
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
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
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
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
			"det":     "../grounding-dino/model.onnx",
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
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
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
		InputWidth:        560,
		InputHeight:       560,
		InputLayout:       "NCHW",
		Letterbox:         true,
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "detr",
		BoxFormat:         "cxcywh",
		ConfThreshold:     0.5,
		MaxDetections:     300,
		LabelsFile:        "../rf-detr/coco91.txt",
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
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
		InputWidth:        640,
		InputHeight:       640,
		InputLayout:       "NCHW",
		Letterbox:         true,
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "rt-detr",
		BoxFormat:         "cxcywh",
		ConfThreshold:     0.5,
		MaxDetections:     300,
		LabelsFile:        "coco80.txt",
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
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
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
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
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
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
		Description:  "Depth Anything V2 small — monocular depth estimation, 518×518. UPSTREAM GONE, see Note.",
		HFRepo:       "onnx-community/depth-anything-v2-small-hf",
		Files: []File{
			{Role: "model", HFFilename: "onnx/model.onnx", LocalFilename: "model.onnx"},
		},
		InputWidth:        518,
		InputHeight:       518,
		InputLayout:       "NCHW",
		Letterbox:         false,
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "depth",
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          false,
		Note: "UPSTREAM REMOVED: huggingface.co/onnx-community/depth-anything-v2-small-hf returns " +
			"401 as of 2026-08-18, so this entry cannot be pulled and cannot be sha256-pinned. " +
			"Candidate replacements exist under different names (onnx-community/depth-anything-v2-small, " +
			"…-small-ONNX). Do NOT repoint blindly — verify the real tensor shapes first (CLAUDE.md). " +
			"`midas` is a working MIT-licensed depth model in the meantime.",
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
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
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
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
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
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
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
		HFRepo:       "khasinski/clip-ViT-B-32-onnx",
		Files: []File{
			{Role: "model", HFFilename: "visual.onnx", LocalFilename: "model.onnx", SHA256: "78e896b2c7301d01eda84e280d7c7297299aa6f8bacc0f5f8fe5bd60d42d8aae"},
		},
		InputWidth:        224,
		InputHeight:       224,
		InputLayout:       "NCHW",
		Letterbox:         false,
		Normalize:         &Normalize{Mean: []float32{0.48145466, 0.4578275, 0.40821073}, Std: []float32{0.26862954, 0.26130258, 0.27577711}},
		PostprocessType:   "embed",
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
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
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
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
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
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
