package cli

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"text/tabwriter"

	"visionserve/internal/catalog"
	"visionserve/pkg/api"
)

// The ps/rm commands operate on models LIVE inside the server process, so they
// call the HTTP API of the running server (state is not shared across processes).

func serverBase(addr string) string {
	if addr == "" {
		addr = ":11435"
	}
	if addr[0] == ':' {
		addr = "127.0.0.1" + addr
	}
	return "http://" + addr
}

// runPs: visionserve ps — which models are loaded (filtered from GET /api/models).
func runPs(args []string) error {
	fs := flag.NewFlagSet("ps", flag.ContinueOnError)
	addr := fs.String("addr", ":11435", "server address")
	if err := fs.Parse(args); err != nil {
		return err
	}
	resp, err := http.Get(serverBase(*addr) + "/api/models")
	if err != nil {
		return fmt.Errorf("could not connect to server (did you run `visionserve serve`?): %w", err)
	}
	defer resp.Body.Close()
	var infos []api.ModelInfo
	if err := json.NewDecoder(resp.Body).Decode(&infos); err != nil {
		return err
	}
	tw := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tTASK\tSTATE")
	for _, in := range infos {
		if in.State == "loaded" {
			fmt.Fprintf(tw, "%s\t%s\t%s\n", in.Name, in.Task, in.State)
		}
	}
	return tw.Flush()
}

// runRm: visionserve rm <model> — unload a model from memory (POST /api/unload).
func runRm(args []string) error {
	fs := flag.NewFlagSet("rm", flag.ContinueOnError)
	addr := fs.String("addr", ":11435", "server address")
	if err := fs.Parse(args); err != nil {
		return err
	}
	rest := fs.Args()
	if len(rest) < 1 {
		return fmt.Errorf("usage: visionserve rm <model>")
	}
	body, _ := json.Marshal(api.LoadRequest{Model: rest[0]})
	resp, err := http.Post(serverBase(*addr)+"/api/unload", "application/json", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("could not connect to server: %w", err)
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(resp.Body)
	fmt.Println(string(out))
	return nil
}

// runPull: visionserve pull <model> — download a curated, permissively-licensed
// model from the HuggingFace Hub into the local registry (Ollama-style).
//
// It does NOT contact a remote registry: the catalog of available models is
// built into the binary (internal/catalog). With no args it lists what can be
// pulled.
func runPull(args []string) error {
	fs := flag.NewFlagSet("pull", flag.ContinueOnError)
	modelsFlag := fs.String("models", "", "model registry directory")
	force := fs.Bool("force", false, "redownload files even if already present")

	// Allow flags after the positional (e.g. `pull rf-detr --models DIR`). The standard
	// flag package stops at the first positional, so we loop: parse flags -> take 1
	// positional -> parse again.
	var rest []string
	rem := args
	for len(rem) > 0 {
		if err := fs.Parse(rem); err != nil {
			return err
		}
		rem = fs.Args()
		if len(rem) > 0 {
			rest = append(rest, rem[0])
			rem = rem[1:]
		}
	}
	if len(rest) < 1 {
		// No model named: print the catalog so the user can pick one.
		tw := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
		fmt.Fprintln(tw, "NAME\tTASK\tLICENSE\tSOURCE")
		for _, e := range catalog.List() {
			src := "huggingface.co/" + e.HFRepo
			if !e.Verified {
				src += " (unverified)"
			}
			fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", e.Name, e.Task, e.License, src)
		}
		tw.Flush()
		return fmt.Errorf("usage: visionserve pull <model> [--models DIR] [--force]")
	}

	return catalog.Pull(rest[0], catalog.PullOptions{
		ModelsDir: modelsDir(*modelsFlag),
		Force:     *force,
		Out:       os.Stderr,
	})
}
