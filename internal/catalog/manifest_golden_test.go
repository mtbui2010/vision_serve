package catalog

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"visionserve/internal/registry"
)

// The manifest renderer used to concatenate YAML by hand. It now marshals a schema-shaped value,
// and this file pins that the switch changed no manifest's MEANING: for every catalog entry (and a
// few synthetic ones that reach the fields no catalog entry uses yet), the manifest the new
// renderer writes must parse through registry.LoadManifest — the loader's own path — to exactly
// the Manifest the old renderer's output parsed to.
//
// legacyRenderManifest below is a frozen copy of the old renderer. Do not "fix" it: it is the
// reference, and it is also what a user's registry may still hold from older releases (see
// TestPullRegeneratesLegacyGeneratedManifest).

// goldenEntries is the catalog plus synthetic entries covering what no catalog entry sets today.
func goldenEntries() []Entry {
	out := append([]Entry(nil), builtin...)
	out = append(out,
		Entry{
			Name: "synthetic-explain", Task: "detection", License: "MIT", Architecture: "rf-detr",
			Description: "synthetic: explain + instance + grasp + crop", HFRepo: "org/repo",
			Files: []File{
				{Role: "model", HFFilename: "m.onnx", LocalFilename: "m.onnx", SHA256: strings.Repeat("ab", 32)},
				{Role: "weights", HFFilename: "m.onnx.data", LocalFilename: "m.onnx.data", SHA256: strings.Repeat("cd", 32)},
				{Role: "labels", HFFilename: "labels.txt", LocalFilename: "my labels.txt", SHA256: strings.Repeat("ef", 32)},
			},
			InputWidth: 224, InputHeight: 224, InputLayout: "NCHW", Crop: "center",
			Normalize:       &Normalize{Mean: []float32{0.48145466, 0.4578275, 0.40821073}, Std: []float32{0.26862954, 0.26130258, 0.27577711}},
			PostprocessType: "detr", BoxFormat: "cxcywh", ConfThreshold: 0.35, TextThreshold: 0.2, MaxDetections: 50,
			GripperMin: 12.5, GripperMax: 140,
			Explain:       &registry.ExplainConfig{Type: "attention", Outputs: map[string]string{"attention": "attn", "b": "z"}, SpatialStride: 16, TopChannels: 32},
			Instance:      &registry.InstanceConfig{MaxTemplates: 4, SimThreshold: 0.7, PatchSize: 16},
			RuntimePrefer: []string{"cuda", "cpu"}, IdleUnloadSeconds: 60, Verified: true,
		},
		Entry{
			Name: "synthetic-multi", Task: "segmentation", License: "BSD-3-Clause", Architecture: "mobile-sam",
			Description: "synthetic: multi-session, keep-aspect, unverified", HFRepo: "org/multi",
			Files: []File{
				{Role: "encoder", HFFilename: "enc.onnx", LocalFilename: "enc.onnx", ManifestRole: "encoder", SHA256: strings.Repeat("12", 32)},
				{Role: "decoder", HFFilename: "dec.onnx", LocalFilename: "dec.onnx", ManifestRole: "decoder"},
				{Role: "vocab", HFFilename: "vocab.txt", LocalFilename: "vocab.txt", SHA256: strings.Repeat("34", 32)},
			},
			InputWidth: 518, InputHeight: 518, KeepAspect: true, MultipleOf: 14,
			PostprocessType: "sam", Verified: false, Note: "synthetic: weights not audited",
		},
		Entry{
			Name: "synthetic-direct", Task: "embed", License: "Apache-2.0", Architecture: "clip",
			Files:      []File{{Role: "model", DirectURL: "https://example.com/x.onnx", LocalFilename: "x.onnx"}},
			InputWidth: 224, InputHeight: 224, Verified: true,
		},
	)
	return out
}

// TestRenderManifestMatchesLegacyRenderer is the golden check: new output parses to the same
// registry.Manifest as the old output, for every entry.
func TestRenderManifestMatchesLegacyRenderer(t *testing.T) {
	for _, e := range goldenEntries() {
		t.Run(e.Name, func(t *testing.T) {
			if postLegacy(e) {
				// The frozen renderer predates these fields and cannot express them;
				// TestTextalignEntriesMatchRepoManifests is these entries' reference instead.
				t.Skip("uses fields added after the legacy renderer (own files + VirtualFiles, runtime.threads)")
			}
			// Both files are loaded from the SAME path, so the unexported dir field compares equal too.
			p := filepath.Join(t.TempDir(), e.Name, "manifest.yaml")
			legacy := loadRendered(t, p, legacyRenderManifest(e))
			got, err := e.RenderManifest()
			if err != nil {
				t.Fatalf("RenderManifest: %v", err)
			}
			current := loadRendered(t, p, got)
			if !reflect.DeepEqual(legacy, current) {
				t.Errorf("new manifest parses differently from the legacy one\nlegacy: %+v\nnew:    %+v\n--- new YAML ---\n%s",
					*legacy, *current, got)
			}
		})
	}
}

// Every key the renderer writes must be a field of registry.Manifest in the block it was written
// to. A field that lands in the wrong block (the hand-written renderer could put box_format under
// input: when no postprocess header was emitted) is silently ignored by the loader; strict decoding
// turns that into a failure.
func TestRenderManifestEmitsOnlySchemaFields(t *testing.T) {
	for _, e := range goldenEntries() {
		got, err := e.RenderManifest()
		if err != nil {
			t.Fatalf("%s: RenderManifest: %v", e.Name, err)
		}
		dec := yaml.NewDecoder(strings.NewReader(got))
		dec.KnownFields(true)
		var m registry.Manifest
		if err := dec.Decode(&m); err != nil {
			t.Errorf("%s: generated manifest has a key outside the registry schema: %v\n%s", e.Name, err, got)
		}
	}
}

// The drift the rewrite removes: an entry with box_format / text_threshold but no type,
// conf_threshold or max_detections. The legacy renderer emitted no "postprocess:" header for it, so
// both keys landed in the input: block and the loader dropped them.
func TestRenderManifestKeepsPostprocessFieldsInTheirBlock(t *testing.T) {
	e := Entry{
		Name: "drift", Task: "open_vocab", License: "Apache-2.0", Architecture: "grounding-dino",
		Files:      []File{{Role: "model", LocalFilename: "m.onnx"}},
		InputWidth: 800, InputHeight: 800,
		Normalize: &Normalize{Mean: []float32{0.5, 0.5, 0.5}, Std: []float32{0.5, 0.5, 0.5}},
		BoxFormat: "cxcywh", TextThreshold: 0.25, Verified: true,
	}
	p := filepath.Join(t.TempDir(), "manifest.yaml")
	m := loadRendered(t, p, render(t, e))
	if m.Postprocess.BoxFormat != "cxcywh" || m.Postprocess.TextThreshold != 0.25 {
		t.Fatalf("postprocess fields lost: %+v", m.Postprocess)
	}
	if old := loadRendered(t, p, legacyRenderManifest(e)); old.Postprocess.BoxFormat != "" {
		t.Fatalf("the legacy renderer was expected to lose box_format here (it is the bug this guards); got %+v", old.Postprocess)
	}
}

// The renderer's document type mirrors registry.Manifest by hand (registry.Manifest has no
// omitempty tags and its sha256 field cannot be built outside the registry package). This keeps
// the mirror honest: every yaml key path in manifestDoc must exist in registry.Manifest.
func TestManifestDocMirrorsRegistrySchema(t *testing.T) {
	schema := yamlPaths(reflect.TypeOf(registry.Manifest{}), "")
	doc := yamlPaths(reflect.TypeOf(manifestDoc{}), "")
	if !schema["postprocess.box_format"] || !doc["postprocess.box_format"] || !doc["input.normalize.mean"] {
		t.Fatalf("yamlPaths does not descend into nested blocks: schema=%v doc=%v", schema, doc)
	}
	for p := range doc {
		if !schema[p] {
			t.Errorf("manifestDoc writes %q, which registry.Manifest does not declare", p)
		}
	}
}

// yamlPaths lists the dotted yaml key paths of a struct type (maps and custom types are leaves).
func yamlPaths(t reflect.Type, prefix string) map[string]bool {
	out := map[string]bool{}
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		tag := strings.Split(f.Tag.Get("yaml"), ",")[0]
		if tag == "-" || !f.IsExported() {
			continue
		}
		if tag == "" {
			tag = strings.ToLower(f.Name)
		}
		key := prefix + tag
		out[key] = true
		ft := f.Type
		for ft.Kind() == reflect.Pointer {
			ft = ft.Elem()
		}
		// SHA256Field decodes itself (scalar or map): a leaf, like the maps.
		if ft.Kind() == reflect.Struct && ft != reflect.TypeOf(registry.SHA256Field{}) {
			for k := range yamlPaths(ft, key+".") {
				out[k] = true
			}
		}
	}
	return out
}

// An installed manifest written by the OLD renderer must still be recognised as generated (its
// recorded body hash matches its body) and replaced by the new rendering on re-pull — not kept as
// if a user had edited it.
func TestPullRegeneratesLegacyGeneratedManifest(t *testing.T) {
	dir := t.TempDir()
	// grasp-gd: the legacy renderer wrote an empty "type:" under postprocess, so its bytes differ
	// from the new rendering (most entries render byte-identically, which would make this vacuous).
	e, _ := Lookup("grasp-gd")
	want := render(t, e)
	legacy := legacyRenderManifest(e)
	if legacy == want {
		t.Fatal("precondition: pick an entry whose legacy rendering differs from the new one")
	}
	for _, dep := range e.Dependencies {
		mustWrite(t, filepath.Join(dir, dep, "manifest.yaml"), "stub")
	}
	for _, rel := range e.VirtualFiles {
		mustWrite(t, filepath.Join(dir, e.Name, rel), "x")
	}
	path := filepath.Join(dir, e.Name, "manifest.yaml")
	if !isUneditedGenerated(legacy) {
		t.Fatal("a legacy-rendered manifest is not recognised as unedited generated output")
	}
	mustWrite(t, path, legacy)
	var log bytes.Buffer
	if err := Pull(e.Name, PullOptions{ModelsDir: dir, Out: &log}); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(path); string(got) != want {
		t.Fatalf("legacy generated manifest was not regenerated:\n%s\nlog:\n%s", got, log.String())
	}
	if !strings.Contains(log.String(), "generated by an older catalog") {
		t.Errorf("unexpected log:\n%s", log.String())
	}
	// And the new rendering is itself recognised as generated (and as edited once touched).
	if !isUneditedGenerated(want) {
		t.Fatal("the new rendering is not recognised as unedited generated output")
	}
	if isUneditedGenerated(strings.Replace(want, "letterbox: false", "letterbox: true", 1)) {
		t.Fatal("an edited new rendering is still treated as unedited")
	}
}

// postLegacy reports whether e uses a field the frozen legacy renderer never knew: a partly
// composed entry (own Files and VirtualFiles) or runtime.threads. (HFSubdir only changes
// SourceURL(), which the legacy renderer calls too.)
func postLegacy(e Entry) bool {
	return (len(e.Files) > 0 && len(e.VirtualFiles) > 0) || len(e.RuntimeThreads) > 0 ||
		e.Resize != "" || e.CropPct != 0 || e.Resample != ""
}

// render is RenderManifest for tests: a rendering error fails the test.
func render(t *testing.T, e Entry) string {
	t.Helper()
	s, err := e.RenderManifest()
	if err != nil {
		t.Fatalf("%s: RenderManifest: %v", e.Name, err)
	}
	return s
}

func loadRendered(t *testing.T, path, content string) *registry.Manifest {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := registry.LoadManifest(path)
	if err != nil {
		t.Fatalf("rendered manifest does not load: %v\n%s", err, content)
	}
	return m
}

// ---------------------------------------------------------------------------------------------
// Frozen copy of the hand-built renderer (catalog/manifest.go before the yaml.v3 rewrite).
// ---------------------------------------------------------------------------------------------

func legacyRenderManifest(e Entry) string {
	body := legacyRenderManifestBody(e)
	return fmt.Sprintf("%s %s` [body %s] — hand edits are kept on re-pull (--force regenerates).\n%s",
		generatedHeader, e.Name, bodyHash(body), body)
}

func legacyRenderManifestBody(e Entry) string {
	var b strings.Builder

	if e.Description != "" {
		fmt.Fprintf(&b, "# %s\n", e.Description)
	}
	if !e.Verified {
		fmt.Fprintf(&b, "# WARNING: pulled from an unverified HF source. %s\n", e.Note)
	}
	fmt.Fprintf(&b, "name: %s\n", e.Name)
	fmt.Fprintf(&b, "task: %s\n", e.Task)
	fmt.Fprintf(&b, "license: %s\n", e.License)
	if src := e.SourceURL(); src != "" {
		fmt.Fprintf(&b, "source_url: %s\n", src)
	}
	b.WriteString(legacyRenderSHA256(e))
	b.WriteString(legacyRenderSHA256Files(e))
	if e.Architecture != "" {
		fmt.Fprintf(&b, "architecture: %s\n", e.Architecture)
	}
	if e.Detector != "" {
		fmt.Fprintf(&b, "detector: %s\n", e.Detector)
	}
	if e.Segmenter != "" {
		fmt.Fprintf(&b, "segmenter: %s\n", e.Segmenter)
	}

	if len(e.VirtualFiles) > 0 {
		b.WriteString("\nfiles:\n")
		roles := make([]string, 0, len(e.VirtualFiles))
		for k := range e.VirtualFiles {
			roles = append(roles, k)
		}
		sort.Strings(roles)
		for _, role := range roles {
			fmt.Fprintf(&b, "  %s: %s\n", role, e.VirtualFiles[role])
		}
	} else if manifestFiles := legacyManifestFiles(e); len(manifestFiles) > 0 {
		b.WriteString("\nfiles:\n")
		for _, f := range manifestFiles {
			fmt.Fprintf(&b, "  %s: %s\n", f.ManifestRole, f.LocalFilename)
		}
	} else if mf := legacyModelFile(e); mf != "" {
		fmt.Fprintf(&b, "model_file: %s\n", mf)
	}

	b.WriteString("\ninput:\n")
	fmt.Fprintf(&b, "  width: %d\n", e.InputWidth)
	fmt.Fprintf(&b, "  height: %d\n", e.InputHeight)
	if e.InputLayout != "" {
		fmt.Fprintf(&b, "  layout: %s\n", e.InputLayout)
	}
	fmt.Fprintf(&b, "  letterbox: %t\n", e.Letterbox)
	if e.Crop != "" {
		fmt.Fprintf(&b, "  crop: %s\n", e.Crop)
	}
	if e.KeepAspect {
		b.WriteString("  keep_aspect: true\n")
	}
	if e.MultipleOf > 0 {
		fmt.Fprintf(&b, "  multiple_of: %d\n", e.MultipleOf)
	}
	if e.Normalize != nil {
		b.WriteString("  normalize:\n")
		fmt.Fprintf(&b, "    mean: %s\n", legacyFloatList(e.Normalize.Mean))
		fmt.Fprintf(&b, "    std: %s\n", legacyFloatList(e.Normalize.Std))
	}

	if e.PostprocessType != "" || e.ConfThreshold > 0 || e.MaxDetections > 0 {
		b.WriteString("\npostprocess:\n")
		fmt.Fprintf(&b, "  type: %s\n", e.PostprocessType)
	}
	if e.BoxFormat != "" {
		fmt.Fprintf(&b, "  box_format: %s\n", e.BoxFormat)
	}
	if e.ConfThreshold > 0 {
		fmt.Fprintf(&b, "  conf_threshold: %g\n", e.ConfThreshold)
	}
	if e.TextThreshold > 0 {
		fmt.Fprintf(&b, "  text_threshold: %g\n", e.TextThreshold)
	}
	if e.MaxDetections > 0 {
		fmt.Fprintf(&b, "  max_detections: %d\n", e.MaxDetections)
	}

	if e.GripperMin > 0 || e.GripperMax > 0 {
		b.WriteString("\ngrasp:\n")
		if e.GripperMin > 0 {
			fmt.Fprintf(&b, "  gripper_min: %g\n", e.GripperMin)
		}
		if e.GripperMax > 0 {
			fmt.Fprintf(&b, "  gripper_max: %g\n", e.GripperMax)
		}
	}

	if e.Explain != nil {
		b.WriteString("\nexplain:\n")
		fmt.Fprintf(&b, "  type: %s\n", e.Explain.Type)
		if len(e.Explain.Outputs) > 0 {
			b.WriteString("  outputs:\n")
			keys := make([]string, 0, len(e.Explain.Outputs))
			for k := range e.Explain.Outputs {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				fmt.Fprintf(&b, "    %s: %s\n", k, e.Explain.Outputs[k])
			}
		}
		if e.Explain.SpatialStride > 0 {
			fmt.Fprintf(&b, "  spatial_stride: %d\n", e.Explain.SpatialStride)
		}
		if e.Explain.TopChannels > 0 {
			fmt.Fprintf(&b, "  top_channels: %d\n", e.Explain.TopChannels)
		}
	}

	if e.Instance != nil {
		b.WriteString("\ninstance:\n")
		if e.Instance.MaxTemplates > 0 {
			fmt.Fprintf(&b, "  max_templates: %d\n", e.Instance.MaxTemplates)
		}
		if e.Instance.SimThreshold > 0 {
			fmt.Fprintf(&b, "  sim_threshold: %g\n", e.Instance.SimThreshold)
		}
		if e.Instance.PatchSize > 0 {
			fmt.Fprintf(&b, "  patch_size: %d\n", e.Instance.PatchSize)
		}
	}

	if e.LabelsFile != "" {
		fmt.Fprintf(&b, "\nlabels: %s\n", e.LabelsFile)
	}

	if len(e.RuntimePrefer) > 0 || e.IdleUnloadSeconds > 0 {
		b.WriteString("\nruntime:\n")
		if len(e.RuntimePrefer) > 0 {
			fmt.Fprintf(&b, "  prefer: [%s]\n", strings.Join(e.RuntimePrefer, ", "))
		}
		if e.IdleUnloadSeconds > 0 {
			fmt.Fprintf(&b, "  idle_unload_seconds: %d\n", e.IdleUnloadSeconds)
		}
	}

	return b.String()
}

func legacyRenderSHA256(e Entry) string {
	if len(e.VirtualFiles) > 0 {
		return ""
	}
	if mf := legacyModelFile(e); mf != "" {
		for _, f := range e.Files {
			if f.LocalFilename == mf && f.SHA256 != "" {
				return fmt.Sprintf("sha256: %s\n", f.SHA256)
			}
		}
		return ""
	}
	var pinned []File
	for _, f := range legacyManifestFiles(e) {
		if f.SHA256 != "" {
			pinned = append(pinned, f)
		}
	}
	if len(pinned) == 0 {
		return ""
	}
	sort.Slice(pinned, func(i, j int) bool { return pinned[i].ManifestRole < pinned[j].ManifestRole })
	var b strings.Builder
	b.WriteString("sha256:\n")
	for _, f := range pinned {
		fmt.Fprintf(&b, "  %s: %s\n", f.ManifestRole, f.SHA256)
	}
	return b.String()
}

func legacyRenderSHA256Files(e Entry) string {
	if len(e.VirtualFiles) > 0 {
		return ""
	}
	mf := legacyModelFile(e)
	var side []File
	for _, f := range e.Files {
		if f.SHA256 != "" && f.ManifestRole == "" && f.LocalFilename != mf {
			side = append(side, f)
		}
	}
	if len(side) == 0 {
		return ""
	}
	sort.Slice(side, func(i, j int) bool { return side[i].LocalFilename < side[j].LocalFilename })
	var b strings.Builder
	b.WriteString("sha256_files:\n")
	for _, f := range side {
		fmt.Fprintf(&b, "  %q: %s\n", f.LocalFilename, f.SHA256)
	}
	return b.String()
}

func legacyManifestFiles(e Entry) []File {
	var out []File
	for _, f := range e.Files {
		if f.ManifestRole != "" {
			out = append(out, f)
		}
	}
	return out
}

func legacyModelFile(e Entry) string {
	if len(legacyManifestFiles(e)) > 0 {
		return ""
	}
	for _, f := range e.Files {
		if f.Role == "model" {
			return f.LocalFilename
		}
	}
	if len(e.Files) == 1 {
		return e.Files[0].LocalFilename
	}
	return ""
}

func legacyFloatList(xs []float32) string {
	parts := make([]string, len(xs))
	for i, x := range xs {
		parts[i] = fmt.Sprintf("%g", x)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}
