package lifecycle

import (
	"context"
	"fmt"
	"image"
	"time"

	"visionserve/internal/engine"
	"visionserve/internal/explain"
	"visionserve/internal/models"
	"visionserve/pkg/api"
)

// ExplainRequest carries explain parameters from the HTTP handler.
//
// The detection to explain is one of the detections /api/predict returns for the same image and
// model (with no prompt or options): the DetectionIdx-th of that list, or the first one of Class.
type ExplainRequest struct {
	Class        string  // the first detection of this class (e.g. "cup"); empty = use DetectionIdx
	DetectionIdx int     // 0-based position in /api/predict's detections; used when Class is empty
	TopChannels  int     // Score-CAM: number of channels to sample (0 = use manifest default)
	Alpha        float32 // PNG overlay opacity [0,1] (0 = use default 0.5)
}

// ExplainResult holds the raw heatmap (H×W float32 in [0,1]) and the detection it explains.
type ExplainResult struct {
	Heatmap []float32 // row-major [H*W], values in [0,1]
	Width   int
	Height  int
	// Detection is the explained detection, exactly as /api/predict returns it.
	Detection api.Detection
	// Query is the object query of the explain session that produced Detection (attention), or
	// -1 when the method follows the object by its box instead (Score-CAM).
	Query int
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

	if s.pipeline != nil {
		ep, ok := s.pipeline.(models.ExplainPreprocessor)
		if !ok {
			return ExplainResult{}, fmt.Errorf(
				"lifecycle: %w: pipeline model %q does not implement ExplainPreprocessor", ErrInvalidRequest, name)
		}
		inputTensor, meta, err = ep.ExplainPreprocess(img)
	} else {
		if s.model == nil {
			return ExplainResult{}, fmt.Errorf("lifecycle: model %q has no simple Model", name)
		}
		inputTensor, meta, err = s.model.Preprocess(img)
	}
	if err != nil {
		return ExplainResult{}, fmt.Errorf("lifecycle: preprocess for explain failed: %w", err)
	}

	// The detection to explain, picked from the list /api/predict returns for this image.
	target, err := s.explainTarget(ctx, img, req)
	if err != nil {
		return ExplainResult{}, err
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
	query := -1

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

		if s.model == nil { // Score-CAM requires a plain Model (pipeline models not supported)
			return ExplainResult{}, fmt.Errorf("lifecycle: score_cam explain requires a plain Model, not a pipeline")
		}
		// The target is followed on each masked image by class and box: the N-th detection of
		// a masked run is in general another object (SameObjectScore).
		detectRunner := func(masked image.Image) (float32, error) {
			if err2 := ctx.Err(); err2 != nil { // ends the Score-CAM loop (see ScoreCAMHeatmap)
				return 0, gaveUp(name, err2)
			}
			res, err2 := s.predictSimple(ctx, masked)
			if err2 != nil {
				return 0, err2
			}
			return explain.SameObjectScore(res.Detections, target), nil
		}

		heatmap, W, H, err = explain.ScoreCAMHeatmap(featTensor, img, detectRunner, topK, origW, origH)
	} else {
		// Attention is indexed by object QUERY. The target's position in the detection list is
		// not its query (the decoder thresholds and sorts the queries), so find the query whose
		// box is the target's.
		query, err = explain.QueryForDetection(outputs, meta,
			explain.BoxDecode{InputW: man.Input.Width, InputH: man.Input.Height, Format: man.Postprocess.BoxFormat},
			target)
		if err != nil {
			return ExplainResult{}, fmt.Errorf("lifecycle: explain %q: %w", name, err)
		}
		heatmap, W, H, err = exp.Heatmap(outputs, outputNames, meta, query, origW, origH)
	}

	if err != nil {
		return ExplainResult{}, fmt.Errorf("lifecycle: heatmap computation failed: %w", err)
	}

	s.touch(time.Now())
	return ExplainResult{Heatmap: heatmap, Width: W, Height: H, Detection: target, Query: query}, nil
}

// explainTarget returns the detection an explain request names, from the detections
// /api/predict returns for img on this session (no prompt, no options): the DetectionIdx-th, or
// the first of Class. Explaining "detection N" means exactly that detection, so the list is
// produced by the same path as a predict, not re-derived from the explain session's outputs.
//
// A class that is not detected, or an index past the end of the list, is an error: anything
// else would return a heatmap of some other object, labelled as the requested one.
func (s *Session) explainTarget(ctx context.Context, img image.Image, req ExplainRequest) (api.Detection, error) {
	var (
		res api.Result
		err error
	)
	if s.pipeline != nil {
		res, err = s.inferPipeline(ctx, img, models.Prompt{})
	} else {
		res, err = s.predictSimple(ctx, img)
	}
	if err != nil {
		return api.Detection{}, fmt.Errorf("lifecycle: explain: detection pass failed: %w", err)
	}
	dets := res.Detections
	if req.Class != "" {
		for _, d := range dets {
			if d.Class == req.Class {
				return d, nil
			}
		}
		return api.Detection{}, fmt.Errorf("lifecycle: explain: %w: no %q detection in this image", ErrInvalidRequest, req.Class)
	}
	if req.DetectionIdx >= len(dets) {
		return api.Detection{}, fmt.Errorf("lifecycle: explain: %w: detection_idx %d is out of range: %q found %d detection(s) in this image",
			ErrInvalidRequest, req.DetectionIdx, s.name, len(dets))
	}
	return dets[req.DetectionIdx], nil
}
