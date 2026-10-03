package cli

import (
	"context"
	"errors"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"visionserve/internal/lifecycle"
	"visionserve/internal/registry"
	"visionserve/internal/server"
	"visionserve/internal/templates"
)

func runServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	modelsFlag := fs.String("models", "", "model registry directory")
	addr := fs.String("addr", server.DefaultAddr, "listen address host:port; the default is loopback only (this machine, like Ollama) — the API has no authentication. Use :11435 (or 0.0.0.0:11435) to accept other hosts, e.g. in a container")
	preloadFlag := fs.String("preload", "", "comma-separated models to load at startup, e.g. mobile-sam,rf-detr")
	idleFlag := fs.Int("idle-unload-seconds", -1, "override every model's idle auto-unload (seconds); 0 = never unload (stay resident, no slow reload after an idle pause); -1 = use each manifest's value")
	trtFlag := addTensorRTFlag(fs)
	if err := fs.Parse(args); err != nil {
		return err
	}
	applyTensorRTFlag(trtFlag) // before the registry validates chains and anything loads

	// Verified mode (opt-in, trust-critical deployments): cross-check every model's declared
	// license against the maintainer-audited provenance ledger and enforce the sha256/source pin.
	// Off by default (local-first). Enable with VISIONSERVE_VERIFY=strict (or 1/true).
	switch strings.ToLower(strings.TrimSpace(os.Getenv("VISIONSERVE_VERIFY"))) {
	case "strict", "1", "true", "on":
		n := registry.EnableVerifiedMode()
		log.Printf("license gate: VERIFIED MODE on — %d audited upstream(s); declared licenses cross-checked against the provenance ledger, sha256/source pinned", n)
	}

	reg := registry.New(modelsDir(*modelsFlag))
	warns, err := reg.Scan()
	if err != nil {
		return err
	}
	for _, wn := range warns {
		log.Printf("registry warning: %v", wn)
	}

	tmpl := templates.New()
	mgr := lifecycle.NewManager(reg)
	mgr.SetTemplateStore(tmpl)
	if *idleFlag >= 0 {
		mgr.SetIdleUnloadOverride(*idleFlag)
		if *idleFlag == 0 {
			log.Printf("idle-unload: DISABLED (models stay resident; no reload after idle)")
		} else {
			log.Printf("idle-unload: overriding all models to %ds", *idleFlag)
		}
	}

	// Background: log the EP chain (CUDA → CPU by default, TensorRT → CUDA → CPU with --tensorrt
	// or VISIONSERVE_TENSORRT=1). The libnvinfer probe scans the filesystem, so it runs in a
	// goroutine and never delays server startup or the first request.
	go func() {
		for _, line := range epStatus() {
			log.Printf("GPU: %s", line)
		}
	}()

	// Pre-load models in background so the first request doesn't pay cold-start cost.
	if *preloadFlag != "" {
		for _, name := range strings.Split(*preloadFlag, ",") {
			name = strings.TrimSpace(name)
			if name == "" {
				continue
			}
			go func(n string) {
				log.Printf("preloading %s ...", n)
				if err := mgr.Load(context.Background(), n); err != nil {
					log.Printf("preload %s failed: %v", n, err)
				} else {
					log.Printf("preloaded: %s", n)
				}
			}(name)
		}
	}

	srv := server.New(reg, mgr, tmpl, *addr)

	// graceful shutdown on SIGINT/SIGTERM. ListenAndServe returns as soon as Shutdown STARTS, so
	// wait for it to finish draining requests and releasing models before returning (main exits).
	stopped := make(chan struct{})
	go func() {
		defer close(stopped)
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
		<-sig
		log.Println("shutting down server...")
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = srv.Shutdown(ctx)
	}()

	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	<-stopped
	return nil
}
