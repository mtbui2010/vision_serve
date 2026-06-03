// Package scrfd implements the SCRFD face detector (MIT license, InsightFace).
//
// SCRFD (Sample and Computation Redistribution for Face Detection) is an anchor-based
// multi-scale face detector. Unlike RF-DETR it NEEDS NMS — anchor-based detectors
// generate many overlapping proposals that must be suppressed.
//
// Registered under architecture name "scrfd" via init(); adding this model does NOT
// require modifying core (server/engine/lifecycle) — per CLAUDE.md contribution path.
//
// Reference: https://github.com/deepinsight/insightface/tree/master/detection/scrfd
// License: MIT (verified — InsightFace detection sub-project, not AGPL)
package scrfd

import (
	"fmt"
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

func init() {
	models.Register("scrfd", New)
}

// Config mirrors the subset of models.Config that SCRFD uses.
// We hold a copy of the full models.Config for convenience.
type scrfdModel struct {
	cfg models.Config
}

// New is the factory invoked by lifecycle after parsing the manifest.
func New(cfg models.Config) (models.Base, error) {
	if cfg.Width <= 0 || cfg.Height <= 0 {
		return nil, fmt.Errorf("scrfd: invalid input dimensions %dx%d", cfg.Width, cfg.Height)
	}
	return &scrfdModel{cfg: cfg}, nil
}

func (m *scrfdModel) Name() string      { return m.cfg.Name }
func (m *scrfdModel) Task() models.Task { return models.TaskDetection }

// InputName/OutputNames left empty → engine auto-detects I/O names from the ONNX file.
// Postprocess identifies score/bbox/kps tensors by SHAPE, not name — more robust when
// different SCRFD export variants use different output names.
func (m *scrfdModel) InputName() string     { return "" }
func (m *scrfdModel) OutputNames() []string { return nil }

func (m *scrfdModel) Preprocess(img image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	return preprocess(img, m.cfg)
}

func (m *scrfdModel) Postprocess(outs []engine.Tensor, meta models.PreprocessMeta) (models.Result, error) {
	return postprocess(outs, meta, m.cfg)
}
