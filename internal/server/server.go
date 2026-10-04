// Package server provides the HTTP REST API (JSON) for VisionServe.
// Default address 127.0.0.1:11435 (port 11435 avoids clashing with Ollama's 11434).
package server

import (
	"context"
	"fmt"
	"image"
	"log"
	"net"
	"net/http"
	"time"

	"visionserve/internal/engine"
	"visionserve/internal/lifecycle"
	"visionserve/internal/models"
	"visionserve/internal/registry"
	"visionserve/internal/templates"
	"visionserve/pkg/api"
)

// DefaultAddr is the default listen address: loopback only, like Ollama. The API has no
// authentication, so it is not exposed to the network unless asked for: --addr :11435 (or
// 0.0.0.0:11435) listens on every interface, which is what the container images pass.
const DefaultAddr = "127.0.0.1:11435"

// modelRuntime is what the HTTP layer needs from lifecycle.Manager. It is an interface so the
// handler tests can drive a fake (admission order, cancellation, status mapping) without ONNX.
//
// Every call that can wait takes the request's context: when the client leaves, the runtime stops
// waiting (for the model to load, a session, a model's lock) and returns an error wrapping
// ctx.Err(), which the handlers answer as errClientGone (499).
type modelRuntime interface {
	Admit(ctx context.Context, name string) (release func(), err error)
	Load(ctx context.Context, name string) error
	Unload(name string) error
	IsLoaded(name string) bool
	PredictPrompt(ctx context.Context, name string, img image.Image, prompt models.Prompt) (api.Result, error)
	InferTensor(ctx context.Context, name string, in engine.Tensor) (api.Result, error)
	Explain(ctx context.Context, name string, img image.Image, req lifecycle.ExplainRequest) (lifecycle.ExplainResult, error)
	Preprocess(ctx context.Context, name string, img image.Image, prompt models.Prompt) (lifecycle.PreprocessResult, error)
	Close()
}

// Server ties together the registry + lifecycle Manager + HTTP server.
type Server struct {
	reg  *registry.Registry
	mgr  modelRuntime
	tmpl *templates.Store // template store for instance_detection models
	http *http.Server
}

// New creates a server. Empty addr -> DefaultAddr.
func New(reg *registry.Registry, mgr *lifecycle.Manager, tmpl *templates.Store, addr string) *Server {
	return newServer(reg, mgr, tmpl, addr)
}

func newServer(reg *registry.Registry, mgr modelRuntime, tmpl *templates.Store, addr string) *Server {
	if addr == "" {
		addr = DefaultAddr
	}
	s := &Server{reg: reg, mgr: mgr, tmpl: tmpl}
	s.http = &http.Server{
		Addr:              addr,
		Handler:           s.routes(),
		ReadHeaderTimeout: 10 * time.Second,
		// Bound slow uploads and idle keep-alives. No WriteTimeout: a first request may wait
		// minutes for a model to load (TensorRT engine build), and that is not a client fault.
		ReadTimeout: 2 * time.Minute,
		IdleTimeout: 2 * time.Minute,
	}
	return s
}

func (s *Server) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/health", s.handleHealth)
	mux.HandleFunc("GET /api/models", s.handleModels)
	mux.HandleFunc("POST /api/load", s.handleLoad)
	mux.HandleFunc("POST /api/unload", s.handleUnload)
	// Whole-body caps (see limitBody): an image + a depth map + form fields for the multipart
	// routes, a raw tensor for infer_tensor, several template images for templates.
	const formBody = maxImageBytes + maxTensorBytes + 1<<20
	mux.HandleFunc("POST /api/predict", limitBody(formBody, s.handlePredict))
	mux.HandleFunc("POST /api/infer_tensor", limitBody(maxTensorBytes+1<<20, s.handleInferTensor))
	mux.HandleFunc("POST /api/preprocess", limitBody(maxImageBytes+1<<20, s.handlePreprocess))
	mux.HandleFunc("POST /api/explain", limitBody(maxImageBytes+1<<20, s.handleExplain))
	mux.HandleFunc("POST /api/templates", limitBody(8*maxImageBytes, s.handleTemplateRegister))
	mux.HandleFunc("GET /api/templates", s.handleTemplateList)
	mux.HandleFunc("DELETE /api/templates/{name}", s.handleTemplateDelete)
	return logRequests(corsFromEnv().wrap(mux)) // opt-in CORS (VISIONSERVE_ORIGINS), cors.go
}

// ListenAndServe starts the server (blocking).
func (s *Server) ListenAndServe() error {
	log.Printf("VisionServe listening on %s%s", s.http.Addr, listenScope(s.http.Addr))
	return s.http.ListenAndServe()
}

// listenScope is the note logged after a loopback listen address: it is reachable from this
// machine only, and the same port on every interface (--addr :<port>) would accept other hosts.
// Empty for any other address.
func listenScope(addr string) string {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return ""
	}
	if ip := net.ParseIP(host); host == "localhost" || (ip != nil && ip.IsLoopback()) {
		return fmt.Sprintf(" (this machine only; --addr :%s accepts other hosts)", port)
	}
	return ""
}

// Shutdown gracefully stops the server, THEN releases the models: in-flight requests are drained
// first (http.Server.Shutdown waits for them), so no request ever runs on a released session and
// no new request can load a model after the manager has stopped.
func (s *Server) Shutdown(ctx context.Context) error {
	err := s.http.Shutdown(ctx)
	s.mgr.Close()
	return err
}

// logRequests is middleware that logs each request (method, path, duration).
func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
	})
}
