package server

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"strings"
)

// A multipart body is read part by part, so that admission is checked before the first large
// part (an image or a depth map) is read: a request the model's queue refuses is answered 503
// without its upload being read or stored. http.Request.ParseMultipartForm, used before, buffered
// the whole form first.
//
// That check is a probe: it takes a slot and gives it straight back (formData.probe). The slot the
// request keeps is taken only once the whole body has been read, as before, so a client that
// trickles its upload holds no slot meanwhile — otherwise a few dozen slow uploads would fill a
// model's admission bound (32 by default) and every other request for it would get 503 for as long
// as the server's ReadTimeout. The cost is a race: a slot that frees during the upload may be taken
// by the time it ends, and that request is then refused after its upload, as before.
//
// Field order is up to the client. When a kept file part arrives before the model name is known,
// it is stored as ParseMultipartForm stored it — in memory up to the form's memory budget, the
// rest in a temporary file — and nothing is probed. The bounds
// are the ones ParseMultipartForm(maxFormMemory) applied: maxFormMemory bytes of file data in
// memory, maxFormMemory + 10 MiB for everything kept in memory (an oversized text field is
// multipart.ErrMessageTooLarge), at most maxFormParts parts; and limitBody caps the whole body.

// maxFormMemory is the file data a form keeps in memory; the rest goes to a temporary file.
const maxFormMemory = maxImageBytes

// maxFormParts is ParseMultipartForm's default part limit (GODEBUG multipartmaxparts).
const maxFormParts = 1000

// formData is what a request carries besides its option fields: the file parts kept from a
// multipart body, and the admission slot the request holds. Close releases both. It belongs to
// one handler goroutine.
type formData struct {
	multipart bool
	files     map[string]*upload // the first file part of each kept name; empty for JSON

	admitFn func(model string) (release func(), err error)
	release func() // the admission slot; nil until admitted
	probed  bool   // admission was probed while the body was read (see probe)
	refused bool   // admission was refused
}

// file returns the kept file part called name, or nil (also for a JSON request, or a nil d).
func (d *formData) file(name string) *upload {
	if d == nil {
		return nil
	}
	return d.files[name]
}

// admit takes the request's admission slot on model, unless it already holds one. It runs after
// the body is read; a multipart body only had admission probed while it was read (see probe).
func (d *formData) admit(model string) error {
	if d.release != nil {
		return nil
	}
	release, err := d.admitFn(model)
	if err != nil {
		d.refused = true
		return err
	}
	d.release = release
	return nil
}

// probe checks, once per request, that model would admit it now: it takes a slot and gives it
// straight back. A refusal is returned (and recorded in refused) so the caller stops reading the
// body; a success holds nothing — admit takes the real slot after the body is read.
func (d *formData) probe(model string) error {
	if d.probed || d.release != nil {
		return nil
	}
	d.probed = true
	release, err := d.admitFn(model)
	if err != nil {
		d.refused = true
		return err
	}
	release()
	return nil
}

// Close releases the admission slot and removes the temporary files. Safe on a nil d.
func (d *formData) Close() {
	if d == nil {
		return
	}
	if d.release != nil {
		d.release()
		d.release = nil
	}
	for _, u := range d.files {
		u.remove()
	}
}

// upload is one kept file part: its first bytes in memory, the rest (if any) in a temp file.
type upload struct {
	mem  []byte
	file *os.File
}

// open returns a reader over the whole part, from its first byte.
func (u *upload) open() (io.Reader, error) {
	if u.file == nil {
		return bytes.NewReader(u.mem), nil
	}
	if _, err := u.file.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	return io.MultiReader(bytes.NewReader(u.mem), u.file), nil
}

func (u *upload) remove() {
	if u.file != nil {
		name := u.file.Name()
		_ = u.file.Close()
		_ = os.Remove(name)
		u.file = nil
	}
}

// formParseError is how a malformed or oversized form has always been reported (400, or 413 when
// the cause is the body cap).
func formParseError(err error) error {
	return badRequest(fmt.Errorf("failed to parse multipart form: %w", err))
}

// readMultipart reads a multipart/form-data body into d and returns the form's value lookup.
// keep maps each file part the endpoint uses to the most bytes of it that matter (the rest of
// the part is skipped); other file parts are skipped unread into memory.
//
// Before the first byte of a kept file part is read, admission is probed if the model name is
// known by then (d.probe); a refusal is returned as is, and nothing more of the body is read. The
// request is not admitted here: the caller takes its slot after the body is read (formData.admit). Parse errors keep the wording ParseMultipartForm's callers used.
//
// The lookup gives a name's first value from the URL query, else from the form — the precedence
// of http.Request.FormValue after ParseMultipartForm, which these endpoints used to call.
func readMultipart(r *http.Request, d *formData, keep map[string]int64) (func(name string) string, error) {
	query, err := url.ParseQuery(r.URL.RawQuery)
	if err != nil {
		return nil, formParseError(err)
	}
	mr, err := multipartReader(r)
	if err != nil {
		return nil, formParseError(err)
	}
	values := url.Values{}
	value := func(name string) string {
		if vs := query[name]; len(vs) > 0 {
			return vs[0]
		}
		if vs := values[name]; len(vs) > 0 {
			return vs[0]
		}
		return ""
	}

	memLeft := int64(maxFormMemory) + 10<<20 // text fields + file data kept in memory
	fileMemLeft := int64(maxFormMemory)      // file data kept in memory
	for parts := 0; ; parts++ {
		p, err := mr.NextPart() // also skips whatever of the previous part was not read
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, formParseError(err)
		}
		if parts >= maxFormParts {
			return nil, formParseError(multipart.ErrMessageTooLarge)
		}
		name := p.FormName()
		if name == "" {
			continue
		}
		const mapEntryOverhead = 200 // ReadForm's per-value bookkeeping charge
		if memLeft -= int64(len(name)) + mapEntryOverhead; memLeft < 0 {
			return nil, formParseError(multipart.ErrMessageTooLarge)
		}
		if p.FileName() == "" { // a text field
			var b strings.Builder
			n, err := io.CopyN(&b, p, memLeft+1)
			if err != nil && err != io.EOF {
				return nil, formParseError(err)
			}
			if memLeft -= n; memLeft < 0 {
				return nil, formParseError(multipart.ErrMessageTooLarge)
			}
			values[name] = append(values[name], b.String())
			continue
		}
		limit, kept := keep[name]
		if !kept || d.files[name] != nil {
			continue // a file part the endpoint does not read, or a second one of a kept name
		}
		if model := value("model"); model != "" {
			if err := d.probe(model); err != nil {
				return nil, err
			}
		}
		u, err := spool(p, limit, &fileMemLeft, &memLeft)
		if u != nil {
			d.files[name] = u // recorded even on error, so Close removes its temp file
		}
		if err != nil {
			return nil, formParseError(err)
		}
	}
	drainBody(r.Body)
	return value, nil
}

// multipartReader is http.Request.MultipartReader restricted to multipart/form-data, with the
// errors ParseMultipartForm gave (it refuses multipart/mixed; MultipartReader does not).
func multipartReader(r *http.Request) (*multipart.Reader, error) {
	ct := r.Header.Get("Content-Type")
	if ct == "" {
		return nil, http.ErrNotMultipart
	}
	if r.Body == nil {
		return nil, errors.New("missing form body")
	}
	mediaType, params, err := mime.ParseMediaType(ct)
	if err != nil || mediaType != "multipart/form-data" {
		return nil, http.ErrNotMultipart
	}
	boundary, ok := params["boundary"]
	if !ok {
		return nil, http.ErrMissingBoundary
	}
	return multipart.NewReader(r.Body, boundary), nil
}

// spool stores up to limit bytes of part p: in memory while *fileMemLeft allows, the rest in a
// temporary file. It returns the upload even with an error once a temp file exists, so the caller
// can remove it.
func spool(p io.Reader, limit int64, fileMemLeft, memLeft *int64) (*upload, error) {
	inMem := min(limit, max(*fileMemLeft, 0))
	var b bytes.Buffer
	n, err := io.CopyN(&b, p, inMem)
	if err != nil && err != io.EOF {
		return nil, err
	}
	*fileMemLeft -= n
	*memLeft -= n
	u := &upload{mem: b.Bytes()}
	if err == io.EOF || n == limit {
		return u, nil // the whole part (or all of it that matters) is in memory
	}
	f, err := os.CreateTemp("", "visionserve-upload-")
	if err != nil {
		return nil, err
	}
	u.file = f
	if _, err := io.Copy(f, io.LimitReader(p, limit-n)); err != nil {
		return u, err
	}
	return u, nil
}
