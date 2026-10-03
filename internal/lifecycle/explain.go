package lifecycle

import (
	"fmt"
	"image"
	"time"

	"visionserve/internal/engine"
	"visionserve/internal/explain"
	"visionserve/internal/models"
)

// ExplainRequest carries explain parameters from the HTTP handler.
type ExplainRequest struct {
	Class        string  // filter by class name (e.g. "cup"); empty = use DetectionIdx
	DetectionIdx int     // 0-based index; used when Class is empty (default 0)
	TopChannels  int     // Score-CAM: number of channels to sample (0 = use manifest default)
	Alpha        float32 // PNG overlay opacity [0,1] (0 = use default 0.5)
}

// ExplainResult holds the raw heatmap (H×W float32 in [0,1]).
type ExplainResult struct {
	Heatmap []float32 // row-major [H*W], values in [0,1]
	Width   int
	Height  int
}

// loadExplainSession lazily creates the explain session (all outputs including explain tensors).
// Thread-safe — multiple concurrent /api/explain calls race to create it; only one wins.
func (m *Manager) loadExplainSession(name string) error {
	s, release, err := m.acquire(name)
	if err != nil {
		return err
	}
	defer release()
	if s.ExplainEngine() != nil {
		return nil // already created
	}

	entry, ok := m.reg.Get(name)
	if !ok {
		return fmt.Errorf("lifecycle: model %q not in registry", name)
	}
	man := entry.Manifest

	if man.Explain == nil {
		return fmt.Errorf("model %q does not support explain (no explain block in manifest)", name)
	}

	providers, err := man.Providers()
	if err != nil {
		return err
	}

	var explainEng engine.Runnable

	if s.pipeline != nil {
		// PipelineModel: create explain session for the role that owns the explain outputs.
		if man.Explain.Role == "" {
			return fmt.Errorf("model %q is a pipeline model — explain block must set 'role' (e.g. role: rfdetr)", name)
		}
		filesAbs := man.FilesAbs()
		rolePath, ok := filesAbs[man.Explain.Role]
		if !ok {
			return fmt.Errorf("lifecycle: explain role %q not in files map for model %q", man.Explain.Role, name)
		}
		explainEng, err = engine.NewSession(rolePath, nil, nil, providers)
		if err != nil {
			return fmt.Errorf("lifecycle: failed to create explain session for role %q in %q: %w", man.Explain.Role, name, err)
		}
	} else {
		// Plain Model: create session with ALL outputs (detect + explain tensors).
		explainEng, err = engine.NewSession(man.ModelFilePath(), nil, nil, providers)
		if err != nil {
			return fmt.Errorf("lifecycle: failed to create explain session for %q: %w", name, err)
		}
	}

	if !s.SetExplainEngine(explainEng) {
		_ = explainEng.Close() // another request won the race, discard ours
	}
	return nil
}

// Explain runs heatmap inference for the named model.
// Returns raw float32 heatmap; rendering (PNG / numpy response) is done by the handler.
func (m *Manager) Explain(name string, img image.Image, req ExplainRequest) (ExplainResult, error) {
	// Ensure the model is loaded.
	if err := m.Load(name); err != nil {
		return ExplainResult{}, err
	}
	// Ensure explain session exists (lazy).
	if err := m.loadExplainSession(name); err != nil {
		return ExplainResult{}, err
	}

	s, release, err := m.acquire(name)
	if err != nil {
		return ExplainResult{}, err
	}
	defer release()

	entry, ok := m.reg.Get(name)
	if !ok {
		return ExplainResult{}, fmt.Errorf("lifecycle: model %q not in registry", name)
	}
	man := entry.Manifest

	// Build the Explainer from the manifest config.
	exp, err := explain.New(man.Explain)
	if err != nil {
		return ExplainResult{}, err
	}

	// Preprocess: plain Model uses its own Preprocess(); PipelineModel uses ExplainPreprocessor.
	var inputTensor engine.Tensor
	var meta models.PreprocessMeta
	detectionIdx := req.DetectionIdx

	if s.pipeline != nil {
		ep, ok := s.pipeline.(models.ExplainPreprocessor)
		if !ok {
			return ExplainResult{}, fmt.Errorf(
				"lifecycle: pipeline model %q does not implement ExplainPreprocessor", name)
		}
		inputTensor, meta, err = ep.ExplainPreprocess(img)
		if err != nil {
			return ExplainResult{}, fmt.Errorf("lifecycle: preprocess for explain failed: %w", err)
		}
		// Class-based detection index not supported for pipeline models; use req.DetectionIdx.
	} else {
		mdl := s.model
		if mdl == nil {
			return ExplainResult{}, fmt.Errorf("lifecycle: model %q has no simple Model", name)
		}
		inputTensor, meta, err = mdl.Preprocess(img)
		if err != nil {
			return ExplainResult{}, fmt.Errorf("lifecycle: preprocess for explain failed: %w", err)
		}
		// Resolve class → detectionIdx (plain models only).
		if req.Class != "" {
			// A class that is not detected is an error, not "explain detection 0": that used to
			// return a heatmap for some other object, labelled as the requested class.
			detectOuts, derr := s.engine.Run([]engine.Tensor{inputTensor})
			if derr != nil {
				return ExplainResult{}, fmt.Errorf("lifecycle: explain: detection pass failed: %w", derr)
			}
			res, derr := mdl.Postprocess(detectOuts, meta)
			if derr != nil {
				return ExplainResult{}, fmt.Errorf("lifecycle: explain: %w", derr)
			}
			found := false
			for i, d := range res.Detections {
				if d.Class == req.Class {
					detectionIdx, found = i, true
					break
				}
			}
			if !found {
				return ExplainResult{}, fmt.Errorf("lifecycle: explain: no %q detection in this image", req.Class)
			}
		}
	}

	// Run the explain session (all outputs: detect + explain tensors).
	explainEng := s.ExplainEngine()
	outputs, err := explainEng.Run([]engine.Tensor{inputTensor})
	if err != nil {
		return ExplainResult{}, fmt.Errorf("lifecycle: explain session inference failed: %w", err)
	}
	outputNames := explainEng.OutputNames()

	origW := meta.OrigWidth
	origH := meta.OrigHeight

	// Override topChannels in the explain config for this request.
	if req.TopChannels > 0 && man.Explain.TopChannels != req.TopChannels {
		// Build a temporary config copy with the per-request top_channels.
		cfgCopy := *man.Explain
		cfgCopy.TopChannels = req.TopChannels
		exp, err = explain.New(&cfgCopy)
		if err != nil {
			return ExplainResult{}, err
		}
	}

	var heatmap []float32
	var W, H int

	if man.Explain.Type == "score_cam" {
		// Full Score-CAM: re-run detect session once per top-K channel with a masked image.
		// detectRunner is a closure that captures the lifecycle-owned detect session —
		// this avoids any circular import between internal/explain and internal/lifecycle.
		featName := man.Explain.Outputs["features"]
		var featTensor engine.Tensor
		for i, n := range outputNames {
			if n == featName && i < len(outputs) {
				featTensor = outputs[i]
				break
			}
		}
		if featTensor.Shape == nil {
			return ExplainResult{}, fmt.Errorf("lifecycle: feature tensor %q not found in explain session outputs", featName)
		}

		topK := man.Explain.EffectiveTopChannels()
		if req.TopChannels > 0 {
			topK = req.TopChannels
		}

		plainMdl := s.model // Score-CAM requires a plain Model (pipeline models not supported)
		if plainMdl == nil {
			return ExplainResult{}, fmt.Errorf("lifecycle: score_cam explain requires a plain Model, not a pipeline")
		}
		detectRunner := func(masked image.Image) (float32, error) {
			in, meta2, err2 := plainMdl.Preprocess(masked)
			if err2 != nil {
				return 0, err2
			}
			outs, err2 := s.engine.Run([]engine.Tensor{in})
			if err2 != nil {
				return 0, err2
			}
			res, err2 := plainMdl.Postprocess(outs, meta2)
			if err2 != nil {
				return 0, err2
			}
			if detectionIdx < len(res.Detections) {
				return float32(res.Detections[detectionIdx].Conf), nil
			}
			return 0, nil
		}

		heatmap, W, H, err = explain.ScoreCAMHeatmap(featTensor, img, detectRunner, topK, origW, origH)
	} else {
		// Attention map: single inference already done, extract from outputs.
		heatmap, W, H, err = exp.Heatmap(outputs, outputNames, meta, detectionIdx, origW, origH)
	}

	if err != nil {
		return ExplainResult{}, fmt.Errorf("lifecycle: heatmap computation failed: %w", err)
	}

	s.touch(time.Now())
	return ExplainResult{Heatmap: heatmap, Width: W, Height: H}, nil
}
