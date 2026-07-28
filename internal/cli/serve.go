package cli

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"visionserve/internal/engine"
	"visionserve/internal/lifecycle"
	"visionserve/internal/registry"
	"visionserve/internal/server"
	"visionserve/internal/templates"
)

func runServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	modelsFlag := fs.String("models", "", "model registry directory")
	addr := fs.String("addr", server.DefaultAddr, "listen address host:port")
	preloadFlag := fs.String("preload", "", "comma-separated models to load at startup, e.g. mobile-sam,rf-detr")
	idleFlag := fs.Int("idle-unload-seconds", -1, "override every model's idle auto-unload (seconds); 0 = never unload (stay resident, no slow reload after an idle pause); -1 = use each manifest's value")
	if err := fs.Parse(args); err != nil {
		return err
	}

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

	// Background: check TRT availability and log a recommendation if absent.
	// Runs in a goroutine so it never delays server startup or the first request.
	go func() {
		if engine.TRTAvailable() {
			log.Printf("GPU: TensorRT found (%s) — TRT EP enabled for maximum performance", engine.TRTLibPath())
		} else {
			log.Println("GPU: TensorRT (libnvinfer.so.10) not found — using CUDA EP only")
			log.Println("     For 10-50× faster inference on transformer models, install TensorRT:")
			log.Println("     • Check LD_LIBRARY_PATH includes the TRT lib directory")
			log.Println("     • Or install: https://developer.nvidia.com/tensorrt")
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
				if err := mgr.Load(n); err != nil {
					log.Printf("preload %s failed: %v", n, err)
				} else {
					log.Printf("preloaded: %s", n)
				}
			}(name)
		}
	}

	srv := server.New(reg, mgr, tmpl, *addr)

	// graceful shutdown on SIGINT/SIGTERM
	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
		<-sig
		log.Println("shutting down server...")
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = srv.Shutdown(ctx)
	}()

	if err := srv.ListenAndServe(); err != nil && err.Error() != "http: Server closed" {
		return err
	}
	return nil
}
