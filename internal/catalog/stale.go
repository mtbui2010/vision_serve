package catalog

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"time"
)

// staleStagingAfter is how old a staging directory must be before an install treats it as left
// behind by an interrupted one. A live InstallLocal refreshes its staging directory's mtime after
// every file it copies and touches the previous install right before moving it aside, so a live
// install's directories never look this old — only a single file copy lasting longer could.
// A variable so tests can shorten it.
var staleStagingAfter = 60 * time.Minute

// stagingDirRE matches exactly the names InstallLocal creates in the registry root:
// os.MkdirTemp(root, ".tmp-<name>-") → ".tmp-<name>-<digits>", and the previous install moved
// aside next to it, "<staging>-old". Nothing else — a model directory merely named "x-old", or any
// other dot-directory, is never a candidate.
var stagingDirRE = regexp.MustCompile(`^\.tmp-(.+)-[0-9]+(-old)?$`)

// cleanStaleStaging removes what interrupted local installs (`pull <folder>` killed mid-copy or
// mid-swap) leave in the registry root: staging copies and moved-aside previous installs older than
// staleStagingAfter. Without it they stay forever, each a full copy of a model's weights.
//
// Only DIRECT children of root whose name matches stagingDirRE and which are real directories:
// the entry type comes from lstat, so a symlink (to a directory or anywhere else) is never
// followed or removed, and os.RemoveAll does not follow symlinks inside what it removes. Best
// effort: a failure is reported on out and does not fail the install.
func cleanStaleStaging(root string, out io.Writer) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return // no registry yet: nothing to clean
	}
	now := time.Now()
	for _, de := range entries {
		name := de.Name()
		m := stagingDirRE.FindStringSubmatch(name)
		if !de.IsDir() || m == nil {
			continue
		}
		// A moved-aside previous install is the ONLY copy when the swap or its roll-back was
		// interrupted: keep it unless <root>/<model> is a complete install again.
		if m[2] != "" {
			if _, err := os.Stat(filepath.Join(root, m[1], "manifest.yaml")); err != nil {
				fmt.Fprintf(out, "WARNING: %s holds the previous install of %q and %s has none; "+
					"move it back to %s (not removed)\n", name, m[1], root, m[1])
				continue
			}
		}
		p := filepath.Join(root, name)
		st, err := os.Lstat(p)
		if err != nil || !st.IsDir() || now.Sub(st.ModTime()) < staleStagingAfter {
			continue
		}
		if err := os.RemoveAll(p); err != nil {
			fmt.Fprintf(out, "WARNING: could not remove %s, left by an interrupted install: %v\n", p, err)
			continue
		}
		fmt.Fprintf(out, "removed %s, left by an interrupted install (%s old)\n",
			name, now.Sub(st.ModTime()).Round(time.Minute))
	}
}

// moveAside renames the install being replaced to aside. A rename keeps the directory's own mtime,
// which for an old install is old, so it is touched first: a concurrent install's
// cleanStaleStaging must never take it for a leftover while this install may still need it to
// roll back.
func moveAside(dst, aside string) error {
	touchDir(dst)
	return os.Rename(dst, aside)
}

// touchDir sets dir's mtime to now, marking it as in use for cleanStaleStaging. Best effort.
func touchDir(dir string) {
	now := time.Now()
	_ = os.Chtimes(dir, now, now)
}
