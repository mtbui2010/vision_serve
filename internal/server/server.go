// Package server cung cấp HTTP REST API (JSON) cho VisionServe.
// Cổng mặc định 11435 (tránh trùng 11434 của Ollama).
package server

import (
	"context"
	"log"
	"net/http"
	"time"

	"visionserve/internal/lifecycle"
	"visionserve/internal/registry"
)

// DefaultAddr là địa chỉ lắng nghe mặc định.
const DefaultAddr = ":11435"

// Server gắn kết registry + lifecycle Manager + HTTP server.
type Server struct {
	reg  *registry.Registry
	mgr  *lifecycle.Manager
	http *http.Server
}

// New tạo server. addr rỗng -> DefaultAddr.
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

// ListenAndServe khởi động server (blocking).
func (s *Server) ListenAndServe() error {
	log.Printf("VisionServe lắng nghe tại %s", s.http.Addr)
	return s.http.ListenAndServe()
}

// Shutdown tắt server gọn gàng + giải phóng model.
func (s *Server) Shutdown(ctx context.Context) error {
	s.mgr.Close()
	return s.http.Shutdown(ctx)
}

// logRequests middleware ghi log mỗi request (method, path, thời gian).
func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
	})
}
