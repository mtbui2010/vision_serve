package cli

import (
	"flag"
	"fmt"
	"os"
	"text/tabwriter"

	"visionserve/internal/catalog"
	"visionserve/internal/registry"
)

// runList: visionserve list — list models in the registry (local scan, no server required).
func runList(args []string) error {
	fs := flag.NewFlagSet("list", flag.ContinueOnError)
	modelsFlag := fs.String("models", "", "model registry directory")
	if err := fs.Parse(args); err != nil {
		return err
	}

	reg := registry.New(modelsDir(*modelsFlag))
	warns, err := reg.Scan()
	if err != nil {
		return err
	}
	for _, wn := range warns {
		fmt.Fprintf(os.Stderr, "warning: %v\n", wn)
	}

	entries := reg.List()

	// Track which model names already exist locally, so catalog models that are
	// not yet downloaded can be listed as "available to pull" (Ollama-style).
	local := make(map[string]bool, len(entries))
	for _, e := range entries {
		local[e.Manifest.Name] = true
	}

	tw := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tTASK\tLICENSE\tINPUT\tWEIGHTS")
	for _, e := range entries {
		m := e.Manifest
		weights := "missing"
		if m.WeightsExist() {
			weights = "ready"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%dx%d\t%s\n", m.Name, m.Task, m.License, m.Input.Width, m.Input.Height, weights)
	}
	// Append catalog models not present locally (pullable via `visionserve pull`).
	for _, c := range catalog.List() {
		if local[c.Name] {
			continue
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%dx%d\t%s\n",
			c.Name, c.Task, c.License, c.InputWidth, c.InputHeight, "available to pull")
	}
	if err := tw.Flush(); err != nil {
		return err
	}
	if len(entries) == 0 {
		fmt.Printf("\nNo models downloaded in %s yet. Use `visionserve pull <name>` to fetch one.\n", reg.Root())
	}
	return nil
}
