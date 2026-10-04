package server

import (
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

func TestParseOrigins(t *testing.T) {
	p, bad := parseOrigins(" http://localhost:5173 , HTTPS://App.Example.com/,null,ftp://x, http://a/path, ,localhost")
	if p == nil || p.any {
		t.Fatalf("policy %+v", p)
	}
	want := map[string]bool{"http://localhost:5173": true, "https://app.example.com": true, "null": true}
	if !reflect.DeepEqual(p.origins, want) {
		t.Errorf("origins %v, want %v", p.origins, want)
	}
	if !reflect.DeepEqual(bad, []string{"ftp://x", "http://a/path", "localhost"}) {
		t.Errorf("rejected %q", bad)
	}
	if p, _ := parseOrigins("*"); p == nil || !p.any {
		t.Errorf("* not parsed as any: %+v", p)
	}
	if p, bad := parseOrigins("nope, "); p != nil || len(bad) != 1 {
		t.Errorf("no valid origin must mean CORS off: %+v %v", p, bad)
	}
}

// corsDo sends one request through a policy around a handler that answers 200 "ok".
func corsDo(p *corsPolicy, method, origin string, preflight bool) *httptest.ResponseRecorder {
	h := p.wrap(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusMethodNotAllowed) // what the mux answers: no OPTIONS routes
			return
		}
		_, _ = w.Write([]byte("ok"))
	}))
	req := httptest.NewRequest(method, "/api/predict", nil)
	if origin != "" {
		req.Header.Set("Origin", origin)
	}
	if preflight {
		req.Header.Set("Access-Control-Request-Method", "POST")
		req.Header.Set("Access-Control-Request-Headers", "content-type")
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestCORS(t *testing.T) {
	list, _ := parseOrigins("http://localhost:5173")
	star, _ := parseOrigins("*")
	acao := func(r *httptest.ResponseRecorder) string { return r.Header().Get("Access-Control-Allow-Origin") }

	// Default (nil policy, VISIONSERVE_ORIGINS unset): unchanged — no CORS headers, OPTIONS goes
	// to the mux (405).
	if r := corsDo(nil, "POST", "http://localhost:5173", false); acao(r) != "" || r.Code != 200 {
		t.Errorf("default: %d %q", r.Code, acao(r))
	}
	if r := corsDo(nil, "OPTIONS", "http://localhost:5173", true); r.Code != http.StatusMethodNotAllowed || acao(r) != "" {
		t.Errorf("default preflight: %d %q", r.Code, acao(r))
	}

	// Allowed origin: echoed, Retry-After exposed; preflight answered 204 without reaching the mux.
	r := corsDo(list, "POST", "http://LOCALHOST:5173", false)
	if r.Code != 200 || acao(r) != "http://LOCALHOST:5173" || r.Header().Get("Access-Control-Expose-Headers") != "Retry-After" || r.Header().Get("Vary") != "Origin" {
		t.Errorf("allowed: %d %v", r.Code, r.Header())
	}
	r = corsDo(list, "OPTIONS", "http://localhost:5173", true)
	if r.Code != http.StatusNoContent || acao(r) != "http://localhost:5173" ||
		r.Header().Get("Access-Control-Allow-Methods") != corsMethods || r.Header().Get("Access-Control-Allow-Headers") != corsHeaders {
		t.Errorf("allowed preflight: %d %v", r.Code, r.Header())
	}

	// Other origin: no CORS headers (the browser withholds the answer), preflight refused.
	if r := corsDo(list, "POST", "http://evil.example", false); r.Code != 200 || acao(r) != "" {
		t.Errorf("other origin: %d %q", r.Code, acao(r))
	}
	if r := corsDo(list, "OPTIONS", "http://evil.example", true); r.Code != http.StatusForbidden || acao(r) != "" {
		t.Errorf("other origin preflight: %d %q", r.Code, acao(r))
	}

	// No Origin (curl, the SDKs): untouched, even with a policy.
	if r := corsDo(list, "POST", "", false); r.Code != 200 || len(r.Header()) != 1 { // Content-Type only
		t.Errorf("no origin: %d %v", r.Code, r.Header())
	}

	// "*": any origin, answered with the literal "*".
	if r := corsDo(star, "OPTIONS", "https://anything.example", true); r.Code != http.StatusNoContent || acao(r) != "*" {
		t.Errorf("star preflight: %d %q", r.Code, acao(r))
	}
}

// The server reads VISIONSERVE_ORIGINS when it builds its routes: set, an allowed origin's
// preflight to a real route is answered; unset, the routes are exactly the old ones.
func TestServerCORSFromEnv(t *testing.T) {
	preflight := func() *httptest.ResponseRecorder {
		_, h := newTestServer(&fakeRuntime{})
		req := httptest.NewRequest(http.MethodOptions, "/api/predict", nil)
		req.Header.Set("Origin", "http://localhost:5173")
		req.Header.Set("Access-Control-Request-Method", "POST")
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		return rec
	}
	t.Setenv("VISIONSERVE_ORIGINS", "http://localhost:5173")
	if r := preflight(); r.Code != http.StatusNoContent || r.Header().Get("Access-Control-Allow-Origin") != "http://localhost:5173" {
		t.Errorf("with VISIONSERVE_ORIGINS: %d %v", r.Code, r.Header())
	}
	t.Setenv("VISIONSERVE_ORIGINS", "")
	if r := preflight(); r.Code == http.StatusNoContent || r.Header().Get("Access-Control-Allow-Origin") != "" {
		t.Errorf("empty VISIONSERVE_ORIGINS must leave CORS off: %d %v", r.Code, r.Header())
	}
}
