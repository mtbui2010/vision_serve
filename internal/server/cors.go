package server

import (
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
)

// corsMethods / corsHeaders are what a preflight may ask for: every route's method, and the
// request headers the clients send (a browser adds Content-Type for JSON bodies).
const (
	corsMethods       = "GET, POST, DELETE, OPTIONS"
	corsHeaders       = "Content-Type, Accept, Authorization, X-Requested-With"
	corsExposeHeaders = "Retry-After" // the 503 back-off hint the SDKs read
	corsMaxAge        = "600"
)

// corsPolicy is the opt-in CORS policy (VISIONSERVE_ORIGINS, like Ollama's OLLAMA_ORIGINS). The
// default — the variable unset — is no policy at all: no CORS headers, so a browser page can
// only call the server from its own origin, exactly as before.
type corsPolicy struct {
	any     bool            // "*": every origin
	origins map[string]bool // normalised "scheme://host[:port]" (or "null")
}

// corsFromEnv reads VISIONSERVE_ORIGINS. It returns nil (CORS off) when the variable is unset or
// holds no valid origin, and logs every entry it ignores and a warning for "*".
func corsFromEnv() *corsPolicy {
	v := os.Getenv("VISIONSERVE_ORIGINS")
	if strings.TrimSpace(v) == "" {
		return nil
	}
	p, bad := parseOrigins(v)
	for _, b := range bad {
		log.Printf("server: ignoring VISIONSERVE_ORIGINS entry %q (want scheme://host[:port], \"null\" or \"*\")", b)
	}
	switch {
	case p == nil:
		log.Printf("server: VISIONSERVE_ORIGINS=%q holds no valid origin — CORS stays off", v)
	case p.any:
		log.Printf("server: WARNING VISIONSERVE_ORIGINS=* — ANY web page a user opens can call this API from their browser; the API has no authentication. List the origins you trust instead.")
	default:
		log.Printf("server: CORS on for %d origin(s) (VISIONSERVE_ORIGINS)", len(p.origins))
	}
	return p
}

// parseOrigins parses a comma-separated origin list. It returns nil when no entry is valid, and
// the entries it rejected.
func parseOrigins(v string) (*corsPolicy, []string) {
	p := &corsPolicy{origins: map[string]bool{}}
	var bad []string
	for _, e := range strings.Split(v, ",") {
		e = strings.TrimSpace(e)
		switch {
		case e == "":
			continue
		case e == "*":
			p.any = true
		case e == "null":
			p.origins["null"] = true // file:// pages and sandboxed iframes send Origin: null
		default:
			o, ok := normaliseOrigin(e)
			if !ok {
				bad = append(bad, e)
				continue
			}
			p.origins[o] = true
		}
	}
	if !p.any && len(p.origins) == 0 {
		return nil, bad
	}
	return p, bad
}

// normaliseOrigin turns "HTTP://LocalHost:8080/" into "http://localhost:8080". An origin has a
// scheme and a host and nothing else (no path, query or credentials).
func normaliseOrigin(s string) (string, bool) {
	u, err := url.Parse(strings.TrimSuffix(s, "/"))
	if err != nil || u.Host == "" || u.User != nil || u.Path != "" || u.RawQuery != "" || u.Fragment != "" {
		return "", false
	}
	scheme := strings.ToLower(u.Scheme)
	if scheme != "http" && scheme != "https" {
		return "", false
	}
	return scheme + "://" + strings.ToLower(u.Host), true
}

// allows reports whether a request's Origin header is on the list.
func (p *corsPolicy) allows(origin string) bool {
	if p.any {
		return true
	}
	if origin == "null" {
		return p.origins["null"]
	}
	o, ok := normaliseOrigin(origin)
	return ok && p.origins[o]
}

// wrap adds the CORS headers for an allowed Origin and answers its preflight (OPTIONS with
// Access-Control-Request-Method) with 204. A request from an origin not on the list gets no CORS
// headers (the browser then refuses to hand the answer to the page) and its preflight a 403. A
// request without Origin (curl, the SDKs, same-origin) passes through untouched. A nil policy is
// no middleware at all.
func (p *corsPolicy) wrap(next http.Handler) http.Handler {
	if p == nil {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin == "" {
			next.ServeHTTP(w, r)
			return
		}
		preflight := r.Method == http.MethodOptions && r.Header.Get("Access-Control-Request-Method") != ""
		h := w.Header()
		h.Add("Vary", "Origin")
		if !p.allows(origin) {
			if preflight {
				http.Error(w, "origin not allowed (VISIONSERVE_ORIGINS)", http.StatusForbidden)
				return
			}
			next.ServeHTTP(w, r)
			return
		}
		if p.any {
			h.Set("Access-Control-Allow-Origin", "*")
		} else {
			h.Set("Access-Control-Allow-Origin", origin)
		}
		h.Set("Access-Control-Expose-Headers", corsExposeHeaders)
		if preflight {
			h.Add("Vary", "Access-Control-Request-Method")
			h.Add("Vary", "Access-Control-Request-Headers")
			h.Set("Access-Control-Allow-Methods", corsMethods)
			h.Set("Access-Control-Allow-Headers", corsHeaders)
			h.Set("Access-Control-Max-Age", corsMaxAge)
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}
