//go:build linux || darwin

package registry

import "golang.org/x/sys/unix"

// fileIDSupported: this platform gives the device, inode and change time the digest cache needs.
const fileIDSupported = true

// statFileID returns the identity of the file at path (symlinks followed) for the digest cache.
func statFileID(path string) (fileID, error) {
	var st unix.Stat_t
	if err := unix.Stat(path, &st); err != nil {
		return fileID{}, err
	}
	return fileID{
		size:  st.Size,
		mtime: st.Mtim.Nano(),
		ctime: st.Ctim.Nano(),
		dev:   uint64(st.Dev),
		ino:   uint64(st.Ino),
	}, nil
}
