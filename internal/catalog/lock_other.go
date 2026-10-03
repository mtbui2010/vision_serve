//go:build !unix

package catalog

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

const pullLockName = ".pull.lock"

// Without flock the lock is an O_EXCL lock file. The holder refreshes its mtime every
// lockHeartbeat; a lock file older than lockStaleAfter belongs to a pull that died and is taken
// over.
const (
	lockHeartbeat  = 30 * time.Second
	lockStaleAfter = 5 * time.Minute
)

func lockDir(dir string, out io.Writer, what string) (func(), error) {
	path := filepath.Join(dir, pullLockName)
	warned := false
	for {
		f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
		if err == nil {
			fmt.Fprintf(f, "%d\n", os.Getpid())
			f.Close()
			stop := make(chan struct{})
			go func() {
				t := time.NewTicker(lockHeartbeat)
				defer t.Stop()
				for {
					select {
					case <-stop:
						return
					case now := <-t.C:
						_ = os.Chtimes(path, now, now)
					}
				}
			}()
			return func() {
				close(stop)
				_ = os.Remove(path)
			}, nil
		}
		if !os.IsExist(err) {
			return nil, err
		}
		if st, err := os.Stat(path); err == nil && time.Since(st.ModTime()) > lockStaleAfter {
			_ = os.Remove(path) // the holder stopped heart-beating: it is gone
			continue
		}
		if !warned {
			fmt.Fprintf(out, "  another pull of %s is in progress — waiting for it to finish\n", what)
			warned = true
		}
		time.Sleep(500 * time.Millisecond)
	}
}
