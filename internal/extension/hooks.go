// Package extension defines EXTENSION POINTS so downstream integrations can plug in
// later WITHOUT modifying the open-source core.
//
// This repository ships only the interface + a default no-op implementation.
//
// A typical example: a data-collection hook for edge devices to send "hard cases"
// (low-confidence, novel objects) to an external labeling/fine-tune service. This
// open-source repo ONLY defines the contract (interface) and a no-op; any real
// implementation lives in a separate downstream integration.
package extension

import (
	"context"

	"visionserve/pkg/api"
)

// DataCollector receives inference results so a downstream integration can filter + forward them.
// By default, use NoopCollector — it collects nothing (local-first, no telemetry).
type DataCollector interface {
	// Observe is called after every predict. A downstream implementation may filter by confidence,
	// store hard samples, or push to a labeling queue. It MUST be non-blocking / its errors must not block inference.
	Observe(ctx context.Context, modelName string, result api.Result) error
}

// NoopCollector is the default collector: it does nothing.
type NoopCollector struct{}

func (NoopCollector) Observe(context.Context, string, api.Result) error { return nil }

// defaultCollector is the default collector (no-op). Downstream integrations override it via SetDefault.
var defaultCollector DataCollector = NoopCollector{}

// SetDefault lets a downstream integration register a real collector at startup.
func SetDefault(c DataCollector) {
	if c != nil {
		defaultCollector = c
	}
}

// Default returns the collector currently in use.
func Default() DataCollector { return defaultCollector }
