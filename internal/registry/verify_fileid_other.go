//go:build !(linux || darwin)

package registry

import "errors"

// fileIDSupported: without an inode and a change time a same-size rewrite with a restored mtime
// would go unnoticed, so on these platforms the digest cache is off and every verification hashes.
const fileIDSupported = false

func statFileID(string) (fileID, error) {
	return fileID{}, errors.New("registry: no file identity on this platform")
}
