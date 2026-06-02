package cli

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"visionserve/internal/lifecycle"
	"visionserve/internal/registry"
	"visionserve/internal/server"
)

func runServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	modelsFlag := fs.String("models", "", "model registry directory")
	addr := fs.String("addr", server.DefaultAddr, "listen address host:port")
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
