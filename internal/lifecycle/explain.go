package lifecycle

import (
	"context"
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

// explainEngineOrLoad returns the session's explain session (the same ONNX file loaded with ALL outputs,
// explain tensors included), creating it on first use. The caller must hold a lease on s, which
// keeps s from being closed while the explain session is being attached to it.
//
// It works on the leased Session itself, never by model name: looking the model up again could
// hand back a different Session (unloaded and reloaded in between) that has no explain session,
// and the Run on it would dereference nil.
//
// Thread-safe — concurrent /api/explain calls race to create it; only one wins.
func (s *Session) explainEngineOrLoad() (engine.Runnable, error) {
	if ex := s.ExplainEngine(); ex != nil {
		return ex, nil // already created
	}

	// The manifest the live sessions were built from (see Session.man), not the registry.
	man := s.man
	if man == nil || man.Explain == nil {
		return nil, fmt.Errorf("%w: model %q does not support explain (no explain block in manifest)", ErrInvalidRequest, s.name)
	}

	providers, err := man.Providers()
	if err != nil {
		return nil, err
	}

	var explainEng engine.Runnable

	if s.pipeline != nil {
		// PipelineModel: create explain session for the role that owns the explain outputs.
		if man.Explain.Role == "" {
			return nil, fmt.Errorf("model %q is a pipeline model — explain block must set 'role' (e.g. role: rfdetr)", s.name)
		}
		filesAbs := man.FilesAbs()
		rolePath, ok := filesAbs[man.Explain.Role]
		if !ok {
			return nil, fmt.Errorf("lifecycle: explain role %q not in files map for model %q", man.Explain.Role, s.name)
		}
		explainEng, err = engine.NewSession(rolePath, nil, nil, providers)
		if err != nil {
			return nil, fmt.Errorf("lifecycle: failed to create explain session for role %q in %q: %w", man.Explain.Role, s.name, err)
		}
	} else {
		// Plain Model: create session with ALL outputs (detect + explain tensors).
		explainEng, err = engine.NewSession(man.ModelFilePath(), nil, nil, providers)
		if err != nil {
			return nil, fmt.Errorf("lifecycle: failed to create explain session for %q: %w", s.name, err)
		}
	}

	if !s.SetExplainEngine(explainEng) {
		_ = explainEng.Close() // another request won the race, discard ours
	}
	return s.ExplainEngine(), nil
}

// Explain runs heatmap inference for the named model.
// Returns raw float32 heatmap; rendering (PNG / numpy response) is done by the handler.
// ctx as in PredictPrompt: a gone request stops waiting for the model and its sessions, and a
// Score-CAM run stops between its per-channel passes.
func (m *Manager) Explain(ctx context.Context, name string, img image.Image, req ExplainRequest) (ExplainResult, error) {
	if req.DetectionIdx < 0 || req.TopChannels < 0 {
		// A negative index would panic the Score-CAM runner (res.Detections[-1]).
		return ExplainResult{}, fmt.Errorf("lifecycle: explain: %w: detection index %d / top channels %d must be >= 0",
			ErrInvalidRequest, req.DetectionIdx, req.TopChannels)
	}
	// Ensure the model is loaded.
	s, release, err := m.loadAndAcquire(ctx, name)
	if err != nil {
		return ExplainResult{}, err
	}
	defer release()

	// Ensure explain session exists (lazy), on the session this request holds. Creating it is
	// heavy and cannot be interrupted: not for a request that is already gone.
	if err := ctx.Err(); err != nil {
		return ExplainResult{}, gaveUp(name, err)
	}
	explainEng, err := s.explainEngineOrLoad()
	if err != nil {
		return ExplainResult{}, err
	}

	// The load-time snapshot, not the registry: the explain session, the outputs it exposes and
	// the detect session's output filter were all derived from it.
	man := s.man

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
				"lifecycle: %w: pipeline model %q does not implement ExplainPreprocessor", ErrInvalidRequest, name)
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
			detectOuts, derr := s.engine.Run(ctx, []engine.Tensor{inputTensor})
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
				return ExplainResult{}, fmt.Errorf("lifecycle: explain: %w: no %q detection in this image", ErrInvalidRequest, req.Class)
			}
		}
	}

	// Run the explain session (all outputs: detect + explain tensors).
	outputs, err := explainEng.Run(ctx, []engine.Tensor{inputTensor})
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
			if err2 := ctx.Err(); err2 != nil { // ends the Score-CAM loop (see ScoreCAMHeatmap)
				return 0, gaveUp(name, err2)
			}
			in, meta2, err2 := plainMdl.Preprocess(masked)
			if err2 != nil {
				return 0, err2
			}
			outs, err2 := s.engine.Run(ctx, []engine.Tensor{in})
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
