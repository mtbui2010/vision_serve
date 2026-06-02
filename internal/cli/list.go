package cli

import (
	"flag"
	"fmt"
	"os"
	"text/tabwriter"

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
	if len(entries) == 0 {
		fmt.Printf("No models found in %s\n", reg.Root())
		return nil
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
	return tw.Flush()
}
