package registry

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"visionserve/internal/engine"
	"visionserve/pkg/api"

	"gopkg.in/yaml.v3"
)

// licenseAllowlist — ONLY permissive licenses are accepted (CLAUDE.md principle #1).
// AGPL is strictly rejected (YOLO/Ultralytics, FastSAM, YOLO-World).
var licenseAllowlist = map[string]bool{
	"Apache-2.0":   true,
	"MIT":          true,
	"BSD-3-Clause": true,
	"BSD-2-Clause": true,
}

var validTasks = map[api.Task]bool{
	api.TaskDetection:         true,
	api.TaskSegmentation:      true,
	api.TaskOpenVocab:         true,
	api.TaskDepth:             true,
	api.TaskClassification:    true,
	api.TaskEmbed:             true,
	api.TaskGrasp:             true,
	api.TaskInstanceDetection: true,
}

// InstanceConfig is optional — when present, the model supports one-shot / template-based
// detection via /api/predict with a TemplateName prompt field.
type InstanceConfig struct {
	// MaxTemplates is the max number of template images to use per inference (0 = no limit).
	MaxTemplates int `yaml:"max_templates"`
	// SimThreshold is the minimum similarity score to report a detection (0 = use model default).
	SimThreshold float64 `yaml:"sim_threshold"`
	// PatchSize is the ViT patch size (e.g. 16 for owlv2-base-patch16, 32 for base-patch32).
	PatchSize int `yaml:"patch_size"`
}

// ExplainConfig khai báo khả năng heatmap visualization của model.
// Khi không có block này, model không support /api/explain.
type ExplainConfig struct {
	Type          string            `yaml:"type"`           // "attention" | "score_cam"
	Outputs       map[string]string `yaml:"outputs"`        // role → ONNX output node name
	                                                        // attention: {"attention": "cross_attn_weights"}
	                                                        // score_cam: {"features": "backbone_features"}
	SpatialStride int               `yaml:"spatial_stride"` // backbone downsample factor (default 32 if 0)
	TopChannels   int               `yaml:"top_channels"`   // Score-CAM: max channels per request (default 64 if 0)
}

// ExplainOutputNames trả về set các output node names dành riêng cho explain.
// Lifecycle dùng để tạo detect_session với các tên này bị loại ra.
func (c *ExplainConfig) ExplainOutputNames() map[string]bool {
	if c == nil {
		return nil
	}
	s := make(map[string]bool, len(c.Outputs))
	for _, v := range c.Outputs {
		s[v] = true
	}
	return s
}

// EffectiveSpatialStride trả về stride thực tế (default 32).
func (c *ExplainConfig) EffectiveSpatialStride() int {
	if c == nil || c.SpatialStride <= 0 {
		return 32
	}
	return c.SpatialStride
}

// EffectiveTopChannels trả về số kênh tối đa cho Score-CAM (default 64).
func (c *ExplainConfig) EffectiveTopChannels() int {
	if c == nil || c.TopChannels <= 0 {
		return 64
	}
	return c.TopChannels
}

// Manifest is the "Modelfile for CV" — specified in docs/manifest-spec.md.
type Manifest struct {
	Name      string `yaml:"name"`
	Task      string `yaml:"task"`
	License   string `yaml:"license"`
	ModelFile string `yaml:"model_file"`

	// Files: multi-session models (SAM, Grounded-SAM) declare multiple ONNX files by "role"
	// (e.g. encoder/decoder). If Files is set, model_file is optional.
	Files map[string]string `yaml:"files"`

	// Architecture: the architecture type used to select the factory (models.New). Defaults to Name.
	Architecture string `yaml:"architecture"`

	// SHA256 (OPTIONAL): content hash(es) that bind the declared license to specific
	// weight bytes. Accepts EITHER a single hex digest (single-file model) OR a
	// role→digest map (multi-file model). When present, weights whose computed
	// SHA-256 does not match are refused at load time. See VerifyWeights + docs/manifest-spec.md.
	SHA256 SHA256Field `yaml:"sha256"`

	// SourceURL (OPTIONAL): the audited upstream the weights were obtained from
	// (e.g. the official RF-DETR/MobileSAM/GroundingDINO repo). Recorded for
	// provenance and optionally checked against a curated allowlist (see
	// VerifiedSourcePrefixes). Purely informational unless an allowlist is supplied.
	SourceURL string `yaml:"source_url"`

	Input struct {
		Width     int    `yaml:"width"`
		Height    int    `yaml:"height"`
		Layout    string `yaml:"layout"`
		Letterbox bool   `yaml:"letterbox"`
		Normalize struct {
			Mean []float32 `yaml:"mean"`
			Std  []float32 `yaml:"std"`
		} `yaml:"normalize"`
	} `yaml:"input"`

	Postprocess struct {
		Type          string  `yaml:"type"`
		BoxFormat     string  `yaml:"box_format"`
		ConfThreshold float64 `yaml:"conf_threshold"`
		TextThreshold float64 `yaml:"text_threshold"` // GroundingDINO: token→label threshold
		MaxDetections int     `yaml:"max_detections"`
	} `yaml:"postprocess"`

	Labels string `yaml:"labels"` // labels file, one class per line

	// Detector/Segmenter: composition for the "grasp" architecture. Segmenter picks the
	// mask backbone (default "mobile-sam"); Detector is OPTIONAL — set it (e.g. "rf-detr",
	// "grounding-dino") for class-aware grasps, omit it for class-agnostic (whole-image
	// automask). The detector/segmenter ONNX graphs are referenced via the files map by
	// role (det / encoder / decoder), like grounded-sam references sibling weights.
	Detector  string `yaml:"detector"`
	Segmenter string `yaml:"segmenter"`

	// Grasp: parallel-jaw defaults for the "grasp" architecture (gripper opening bounds in
	// original-image pixels). A request may override these per call (gripper_min/gripper_max).
	Grasp struct {
		GripperMin float64 `yaml:"gripper_min"`
		GripperMax float64 `yaml:"gripper_max"`
	} `yaml:"grasp"`

	// Explain: optional heatmap visualization config. Khi nil, model không support /api/explain.
	Explain *ExplainConfig `yaml:"explain"`

	// Instance: optional one-shot detection config (OWL-ViT, SiamRPN, …).
	// When nil, model does not accept template prompts.
	Instance *InstanceConfig `yaml:"instance"`

	Runtime struct {
		Prefer            []string `yaml:"prefer"`
		IdleUnloadSeconds int      `yaml:"idle_unload_seconds"`
	} `yaml:"runtime"`

	// dir is the directory containing the manifest (filled at load time, not in the YAML).
	dir string `yaml:"-"`
}

// LoadManifest reads + parses + validates a manifest.yaml file.
func LoadManifest(path string) (*Manifest, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("registry: failed to read manifest %s: %w", path, err)
	}
	var m Manifest
	if err := yaml.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("registry: failed to parse YAML %s: %w", path, err)
	}
	m.dir = filepath.Dir(path)
	if err := m.validate(); err != nil {
		return nil, fmt.Errorf("registry: invalid manifest %s: %w", path, err)
	}
	return &m, nil
}

func (m *Manifest) validate() error {
	if strings.TrimSpace(m.Name) == "" {
		return fmt.Errorf("missing 'name' field")
	}
	// License: required + must be in the permissive allowlist.
	if !licenseAllowlist[m.License] {
		return fmt.Errorf("license %q is not allowed — only permissive licenses accepted (Apache-2.0/MIT/BSD); AGPL is strictly forbidden", m.License)
	}
	if !validTasks[api.Task(m.Task)] {
		return fmt.Errorf("task %q is invalid (detection/segmentation/open_vocab)", m.Task)
	}
	if m.ModelFile == "" && len(m.Files) == 0 {
		return fmt.Errorf("missing 'model_file' (or a 'files' map for multi-session models)")
	}
	// NOTE: the EXISTENCE of the ONNX file is NOT checked here. validate() only checks
	// STRUCTURAL validity (license/task/dims/EP) so a model can still be LISTED even
	// without downloaded weights (like Ollama: see the model before you pull). Weights are
	// checked at Load time via WeightsExist() — see lifecycle.
	if m.Input.Width <= 0 || m.Input.Height <= 0 {
		return fmt.Errorf("input.width/height must be > 0")
	}
	if layout := strings.ToUpper(m.Input.Layout); layout != "NCHW" && layout != "NHWC" && layout != "" {
		return fmt.Errorf("input.layout %q is invalid (NCHW/NHWC)", m.Input.Layout)
	}
	// Validate the fallback chain (normalizes + ensures CPU is present).
	if _, err := engine.ResolveProviders(m.Runtime.Prefer); err != nil {
		return err
	}
	if m.Explain != nil {
		if m.Explain.Type != "attention" && m.Explain.Type != "score_cam" {
			return fmt.Errorf("explain.type %q is invalid (attention/score_cam)", m.Explain.Type)
		}
		if len(m.Explain.Outputs) == 0 {
			return fmt.Errorf("explain.outputs must declare at least one output node name")
		}
		switch m.Explain.Type {
		case "attention":
			if _, ok := m.Explain.Outputs["attention"]; !ok {
				return fmt.Errorf("explain.type=attention requires explain.outputs.attention")
			}
		case "score_cam":
			if _, ok := m.Explain.Outputs["features"]; !ok {
				return fmt.Errorf("explain.type=score_cam requires explain.outputs.features")
			}
		}
	}
	return nil
}

// WeightsExist reports whether the ONNX file(s) are present on disk (weights are not committed).
// For multi-session models: ALL files in Files must exist.
func (m *Manifest) WeightsExist() bool {
	if len(m.Files) > 0 {
		for _, abs := range m.FilesAbs() {
			if _, err := os.Stat(abs); err != nil {
				return false
			}
		}
		return true
	}
	if _, err := os.Stat(m.ModelFilePath()); err != nil {
		return false
	}
	return true
}

// FilesAbs returns a map role → absolute ONNX path (empty if not multi-session).
func (m *Manifest) FilesAbs() map[string]string {
	if len(m.Files) == 0 {
		return nil
	}
	out := make(map[string]string, len(m.Files))
	for role, rel := range m.Files {
		if filepath.IsAbs(rel) {
			out[role] = rel
		} else {
			out[role] = filepath.Join(m.dir, rel)
		}
	}
	return out
}

// Dir returns the directory containing the manifest (so a model can resolve side files like vocab.txt).
func (m *Manifest) Dir() string { return m.dir }

// ModelFilePath returns the absolute path to the ONNX file.
func (m *Manifest) ModelFilePath() string {
	if filepath.IsAbs(m.ModelFile) {
		return m.ModelFile
	}
	return filepath.Join(m.dir, m.ModelFile)
}

// LabelsPath returns the labels file path (empty if the manifest does not declare it).
func (m *Manifest) LabelsPath() string {
	if m.Labels == "" {
		return ""
	}
	if filepath.IsAbs(m.Labels) {
		return m.Labels
	}
	return filepath.Join(m.dir, m.Labels)
}

// LoadLabels reads the labels file (one class per line). Returns nil if not declared.
func (m *Manifest) LoadLabels() ([]string, error) {
	p := m.LabelsPath()
	if p == "" {
		return nil, nil
	}
	raw, err := os.ReadFile(p)
	if err != nil {
		return nil, fmt.Errorf("registry: failed to read labels %s: %w", p, err)
	}
	var labels []string
	for _, line := range strings.Split(string(raw), "\n") {
		s := strings.TrimSpace(line)
		if s != "" {
			labels = append(labels, s)
		}
	}
	return labels, nil
}

// Providers returns the normalized fallback chain.
func (m *Manifest) Providers() ([]engine.Provider, error) {
	return engine.ResolveProviders(m.Runtime.Prefer)
}

// ArchOrName returns the architecture type used to select the factory (defaults to Name).
func (m *Manifest) ArchOrName() string {
	if m.Architecture != "" {
		return m.Architecture
	}
	return m.Name
}
