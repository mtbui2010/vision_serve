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

// runPull: TODO future phase — a remote registry is not part of the MVP (see CLAUDE.md).
func runPull(args []string) error {
	return fmt.Errorf("`pull` not supported yet: remote registry is a future phase (the MVP uses local models only). " +
		"Place the model + manifest.yaml into the ./models directory manually")
}
