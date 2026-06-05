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
)

func runServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	modelsFlag := fs.String("models", "", "model registry directory")
	addr := fs.String("addr", server.DefaultAddr, "listen address host:port")
	preloadFlag := fs.String("preload", "", "comma-separated models to load at startup, e.g. mobile-sam,rf-detr")
	if err := fs.Parse(args); err != nil {
		return err
	}

	reg := registry.New(modelsDir(*modelsFlag))
	warns, err := reg.Scan()
	if err != nil {
		return err
	}
	for _, wn := range warns {
		log.Printf("registry warning: %v", wn)
	}

	mgr := lifecycle.NewManager(reg)

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

	srv := server.New(reg, mgr, *addr)

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
