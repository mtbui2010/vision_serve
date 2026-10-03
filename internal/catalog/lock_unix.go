//go:build unix

package catalog

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"syscall"
)

// pullLockName is the per-model lock file. It is left in place after a pull: deleting a flock
// file races with a waiter that already opened it.
const pullLockName = ".pull.lock"

// lockDir takes an exclusive lock on dir for the duration of a pull, waiting for (and saying so)
// another pull that holds it. flock is released by the kernel when the process dies, so a
// crashed pull never leaves a stale lock behind.
func lockDir(dir string, out io.Writer, what string) (func(), error) {
	f, err := os.OpenFile(filepath.Join(dir, pullLockName), os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return nil, err
	}
	fd := int(f.Fd())
	if err := flock(fd, syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		if !errors.Is(err, syscall.EWOULDBLOCK) {
			f.Close()
			return nil, err
		}
		fmt.Fprintf(out, "  another pull of %s is in progress — waiting for it to finish\n", what)
		if err := flock(fd, syscall.LOCK_EX); err != nil {
			f.Close()
			return nil, err
		}
	}
	return func() {
		_ = flock(fd, syscall.LOCK_UN)
		_ = f.Close()
	}, nil
}

func flock(fd, how int) error {
	for {
		err := syscall.Flock(fd, how)
		if err != syscall.EINTR {
			return err
		}
	}
}
