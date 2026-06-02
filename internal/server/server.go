// Package server provides the HTTP REST API (JSON) for VisionServe.
// Default port 11435 (to avoid clashing with Ollama's 11434).
package server

import (
	"context"
	"log"
	"net/http"
	"time"

	"visionserve/internal/lifecycle"
	"visionserve/internal/registry"
)

// DefaultAddr is the default listen address.
const DefaultAddr = ":11435"

// Server ties together the registry + lifecycle Manager + HTTP server.
type Server struct {
	reg  *registry.Registry
	mgr  *lifecycle.Manager
	http *http.Server
}

// New creates a server. Empty addr -> DefaultAddr.
func New(reg *registry.Registry, mgr *lifecycle.Manager, addr string) *Server {
	if addr == "" {
		addr = DefaultAddr
	}
	s := &Server{reg: reg, mgr: mgr}
	s.http = &http.Server{
		Addr:              addr,
		Handler:           s.routes(),
		ReadHeaderTimeout: 10 * time.Second,
	}
	return s
}

func (s *Server) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/health", s.handleHealth)
	mux.HandleFunc("GET /api/models", s.handleModels)
	mux.HandleFunc("POST /api/load", s.handleLoad)
	mux.HandleFunc("POST /api/unload", s.handleUnload)
	mux.HandleFunc("POST /api/predict", s.handlePredict)
	return logRequests(mux)
}

// ListenAndServe starts the server (blocking).
func (s *Server) ListenAndServe() error {
	log.Printf("VisionServe listening on %s", s.http.Addr)
	return s.http.ListenAndServe()
}

// Shutdown gracefully stops the server + releases models.
func (s *Server) Shutdown(ctx context.Context) error {
	s.mgr.Close()
	return s.http.Shutdown(ctx)
}

// logRequests is middleware that logs each request (method, path, duration).
func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
	})
}
