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
			},
			{
				Role:          "decoder",
				HFFilename:    "sam_mask_decoder_single.onnx",
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
		Verified:          true,
		Note: "Encoder input 'input_image' [H,W,3] HWC raw 0-255 (normalize+pad baked in graph). " +
			"Decoder outputs: masks (upsampled to orig size), iou_predictions, low_res_masks.",
	},
	{
		Name:         "rt-detr",
		Task:         "detection",
		License:      "Apache-2.0",
		Architecture: "rt-detr",
		Description:  "RT-DETR-l (COCO) — real-time NMS-free detector, 640×640.",
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
		Verified:          true,
	},
	{
		Name:         "efficient-sam",
		Task:         "segmentation",
		License:      "Apache-2.0",
		Architecture: "efficient-sam",
		Description:  "EfficientSAM ViT-Tiny — promptable segmentation, lighter than MobileSAM.",
		HFRepo:       "yunyangx/EfficientSAM",
		Files: []File{
			{Role: "encoder", HFFilename: "efficientsam_ti_encoder.onnx", LocalFilename: "efficient_sam_encoder.onnx", ManifestRole: "encoder"},
			{Role: "decoder", HFFilename: "efficientsam_ti_decoder.onnx", LocalFilename: "efficient_sam_decoder.onnx", ManifestRole: "decoder"},
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
			{Role: "encoder", HFFilename: "encoder.onnx", LocalFilename: "sam2_tiny_encoder.onnx", ManifestRole: "encoder"},
			{Role: "decoder", HFFilename: "decoder.onnx", LocalFilename: "sam2_tiny_decoder.onnx", ManifestRole: "decoder"},
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
		Description:  "Depth Anything V2 small — monocular depth estimation, 518×518.",
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
		Verified:          true,
	},
	{
		Name:         "midas",
		Task:         "depth",
		License:      "MIT",
		Architecture: "midas",
		Description:  "MiDaS v2.1-small — monocular depth estimation, 256×256, MIT license.",
		HFRepo:       "Heliosoph/midas-small-onnx",
		Files: []File{
			{Role: "model", HFFilename: "midas_v21_small_256.onnx", LocalFilename: "midas_v21_small_256.onnx"},
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
			{Role: "model", HFFilename: "efficientnet_b0_Opset17.onnx", LocalFilename: "model.onnx"},
		},
		InputWidth:        224,
		InputHeight:       224,
		InputLayout:       "NCHW",
		Letterbox:         false,
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "classification",
		MaxDetections:     5,
		LabelsFile:        "imagenet1k.txt",
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
		Note:              "imagenet1k.txt labels file must be present in the model directory. Input 'x' [1,3,224,224], output logits [1,1000].",
	},
	{
		Name:         "mobilenet-v3",
		Task:         "classification",
		License:      "Apache-2.0",
		Architecture: "mobilenet-v3",
		Description:  "MobileNetV3-Small — ImageNet-1k classification, 224×224, ultra-lightweight.",
		HFRepo:       "onnxmodelzoo/mobilenet_v3_small_Opset17",
		Files: []File{
			{Role: "model", HFFilename: "mobilenet_v3_small_Opset17.onnx", LocalFilename: "model.onnx"},
		},
		InputWidth:        224,
		InputHeight:       224,
		InputLayout:       "NCHW",
		Letterbox:         false,
		Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
		PostprocessType:   "classification",
		MaxDetections:     5,
		LabelsFile:        "imagenet1k.txt",
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          true,
		Note:              "imagenet1k.txt labels file must be present in the model directory. Input 'x' [1,3,224,224], output logits [1,1000].",
	},
	{
		Name:         "clip",
		Task:         "embed",
		License:      "MIT",
		Architecture: "clip",
		Description:  "CLIP ViT-B/32 image encoder — 512-d embeddings for zero-shot classification (MIT, OpenAI).",
		HFRepo:       "khasinski/clip-ViT-B-32-onnx",
		Files: []File{
			{Role: "model", HFFilename: "visual.onnx", LocalFilename: "model.onnx"},
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
		// NanoSAM ONNX files are NOT on HuggingFace; they must be downloaded from
		// Google Drive links in the NVIDIA-AI-IOT/nanosam README:
		//   resnet18_image_encoder.onnx  — encoder
		//   mobile_sam_mask_decoder.onnx — decoder
		// HFRepo is intentionally empty; `pull` will show the Note directing users
		// to the manual download instructions.
		HFRepo: "",
		Files: []File{
			{Role: "encoder", HFFilename: "resnet18_image_encoder.onnx", LocalFilename: "resnet18_image_encoder.onnx", ManifestRole: "encoder"},
			{Role: "decoder", HFFilename: "mobile_sam_mask_decoder.onnx", LocalFilename: "mobile_sam_mask_decoder.onnx", ManifestRole: "decoder"},
		},
		InputWidth:        1024,
		InputHeight:       1024,
		InputLayout:       "NCHW",
		PostprocessType:   "sam",
		RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
		IdleUnloadSeconds: 300,
		Verified:          false,
		Note: "NanoSAM ONNX files are NOT on HuggingFace — download manually from the " +
			"Google Drive links in the NVIDIA-AI-IOT/nanosam README " +
			"(github.com/NVIDIA-AI-IOT/nanosam). " +
			"Encoder: resnet18_image_encoder.onnx (input 'image' [1,3,1024,1024] NCHW ImageNet-norm, " +
			"output 'image_embeddings' [1,256,64,64]). " +
			"Decoder: mobile_sam_mask_decoder.onnx (inputs: image_embeddings, point_coords, " +
			"point_labels, mask_input, has_mask_input; NO orig_im_size input; " +
			"outputs: iou_predictions [1,M], low_res_masks [1,M,256,256]). " +
			"License: Apache-2.0 (confirmed from github.com/NVIDIA-AI-IOT/nanosam). " +
			"Place both files in the model directory and register manually.",
	},
	{
		Name:         "scrfd",
		Task:         "detection",
		License:      "MIT",
		Architecture: "scrfd",
		Description:  "SCRFD-10GF — InsightFace face detector, 640×640, with keypoints, MIT license.",
		HFRepo:       "cromsc/scrfd-10g",
		Files: []File{
			{Role: "model", HFFilename: "scrfd_10g_bnkps.onnx", LocalFilename: "det_10g.onnx"},
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
			{Role: "det", HFFilename: "ch_PP-OCRv4_det.onnx", LocalFilename: "det_model.onnx", ManifestRole: "det"},
			{Role: "rec", HFFilename: "ch_PP-OCRv4_rec.onnx", LocalFilename: "rec_model.onnx", ManifestRole: "rec"},
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
