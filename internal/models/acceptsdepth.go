package models

import "sync"

// AcceptsDepthFunc says whether a model of an architecture reads an uploaded depth map
// (Prompt.Depth: the request's depth / depth_base64 fields), given its manifest's files map
// (role -> path). It must be cheap and pure: GET /api/models calls it for every listed model
// without building the model.
type AcceptsDepthFunc func(files map[string]string) bool

var (
	acceptsDepthMu sync.RWMutex
	acceptsDepth   = map[string]AcceptsDepthFunc{}
)

// RegisterAcceptsDepth declares that architecture arch may read an uploaded, pixel-aligned depth
// map (GET /api/models: accepts_depth). It is the hint SDKs use to decide whether to upload a
// camera's depth frame at all (Python Client.watch(depth="auto")): a model that does not read
// depth only pays for the bytes. Register only an architecture whose Infer really reads
// Prompt.Depth, and keep f exact for its manifests (a background manifest without a MiDaS
// session never reaches the depth method, so it does not accept depth). An architecture that
// accepts depth must not register a client-resize hint (RegisterUsefulSide): the depth map is
// aligned to the full photo. Call it in the model package's init(); a duplicate panics like
// Register.
func RegisterAcceptsDepth(arch string, f AcceptsDepthFunc) {
	acceptsDepthMu.Lock()
	defer acceptsDepthMu.Unlock()
	if _, dup := acceptsDepth[arch]; dup {
		panic("models: accepts-depth of " + arch + " already registered")
	}
	acceptsDepth[arch] = f
}

// AcceptsDepthOf reports whether a model of architecture arch with these manifest files reads an
// uploaded depth map; false for an architecture that registered nothing.
func AcceptsDepthOf(arch string, files map[string]string) bool {
	acceptsDepthMu.RLock()
	f, ok := acceptsDepth[arch]
	acceptsDepthMu.RUnlock()
	return ok && f(files)
}
