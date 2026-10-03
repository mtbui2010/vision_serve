package server

import (
	"context"
	"errors"
	"net/http"
	"strconv"

	"visionserve/internal/lifecycle"
	"visionserve/internal/models"
	"visionserve/pkg/api"
)

// retryAfterSeconds is the Retry-After a 503 (model queue full) suggests.
const retryAfterSeconds = 1

// statusClientClosedRequest is nginx's 499: the client went away before the work started. Nobody
// reads the response; the code is for the access log.
const statusClientClosedRequest = 499

// errClientGone: the client disconnected before inference started, so it was not run (or, for a
// multi-stage pipeline, its remaining stages were not).
var errClientGone = errors.New("client disconnected before inference started")

// orClientGone returns errClientGone when err is the runtime giving up on a request because ctx
// ended — its client left while it waited for the model to load, for a session or for a model's
// lock — and err unchanged otherwise.
func orClientGone(ctx context.Context, err error) error {
	if err != nil && ctx.Err() != nil && errors.Is(err, ctx.Err()) {
		return errClientGone
	}
	return err
}

// requestError marks a malformed request (a form or JSON body that does not parse, a missing
// field, a bad prompt or depth map). It keeps the message as written — clients match on it — and
// counts as lifecycle.ErrInvalidRequest for errors.Is, so the server and the runtime share one
// notion of "the client's fault".
type requestError struct{ err error }

func (e requestError) Error() string        { return e.err.Error() }
func (e requestError) Unwrap() error        { return e.err }
func (e requestError) Is(target error) bool { return target == lifecycle.ErrInvalidRequest }

// badRequest wraps a request-parse error so writeError answers 400 (or 413 when the cause is an
// oversized body).
func badRequest(err error) error {
	if err == nil {
		return nil
	}
	return requestError{err}
}

// tooLargeError is an upload over one of the server's size limits that the body cap itself did
// not catch (e.g. the image part inside a form that fits the cap).
type tooLargeError struct{ msg string }

func (e tooLargeError) Error() string { return e.msg }

// statusOf maps an error to its HTTP status. It is the ONLY place a failure's status is chosen:
//
//	oversized body / upload            413
//	lifecycle.ErrModelNotFound         404
//	lifecycle.ErrOverloaded            503 (+ Retry-After)
//	lifecycle.ErrInvalidRequest, parse 400
//	client disconnected                499
//	anything else                      500
//
// Size comes first: a form that failed to parse BECAUSE it was too big is a 413, not a 400.
func statusOf(err error) int {
	var maxBytes *http.MaxBytesError
	var tooLarge tooLargeError
	switch {
	case errors.As(err, &maxBytes), errors.As(err, &tooLarge):
		return http.StatusRequestEntityTooLarge
	case errors.Is(err, lifecycle.ErrModelNotFound):
		return http.StatusNotFound
	case errors.Is(err, lifecycle.ErrOverloaded):
		return http.StatusServiceUnavailable
	case errors.Is(err, lifecycle.ErrInvalidRequest), errors.Is(err, models.ErrBadPrompt):
		return http.StatusBadRequest
	case errors.Is(err, errClientGone):
		return statusClientClosedRequest
	default:
		return http.StatusInternalServerError
	}
}

// writeError answers {"error": "..."} with the status statusOf picks for err.
func writeError(w http.ResponseWriter, err error) {
	status := statusOf(err)
	if status == http.StatusServiceUnavailable {
		w.Header().Set("Retry-After", strconv.Itoa(retryAfterSeconds))
	}
	writeJSON(w, status, api.ErrorResponse{Error: err.Error()})
}
