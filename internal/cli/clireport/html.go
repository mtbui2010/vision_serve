package clireport

import (
	"encoding/base64"
	"html/template"
	"io"
	"strings"
)

// WriteHTML writes the report as one self-contained HTML page: inline CSS, no script, no
// external asset (images are base64 PNG data URLs), light and dark themes via
// prefers-color-scheme. Its layout is the shared one (see the package comment): the sections
// #verdict, #summary, #details and #next-steps, in that order.
func (r *Report) WriteHTML(w io.Writer) error {
	type img struct {
		Caption string
		Src     template.URL
	}
	type section struct {
		Title string
		Rows  []Field
		Table *Table
		Lines string
		Image *img
	}
	data := struct {
		Title, Verdict, VerdictClass, Reason string
		Summary                              []Field
		Findings                             []Finding
		Sections                             []section
		NextSteps                            []string
	}{
		Title:        strings.TrimSpace("visionserve " + r.Command + " " + r.Subject),
		Verdict:      string(r.verdict()),
		VerdictClass: strings.ToLower(string(r.verdict())),
		Reason:       r.Reason,
		Summary:      r.Summary,
		Findings:     r.sortedFindings(),
		NextSteps:    r.NextSteps,
	}
	for _, s := range r.Sections {
		sec := section{Title: s.Title, Rows: s.Rows, Table: s.Table, Lines: strings.Join(s.Lines, "\n")}
		if s.Image != nil && len(s.Image.PNG) > 0 {
			// Safe: the URL is built here from PNG bytes, never from user text.
			sec.Image = &img{Caption: s.Image.Caption,
				Src: template.URL("data:image/png;base64," + base64.StdEncoding.EncodeToString(s.Image.PNG))}
		}
		data.Sections = append(data.Sections, sec)
	}
	return htmlTemplate.Execute(w, data)
}

var htmlTemplate = template.Must(template.New("report").Funcs(template.FuncMap{
	"show":  func(f Field) string { return f.display() },
	"lower": func(v Verdict) string { return strings.ToLower(string(v)) },
}).Parse(`<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{.Title}}</title>
<style>
:root{--bg:#ffffff;--fg:#1d1f23;--muted:#5d636e;--line:#d9dce1;--panel:#f5f6f8;--code:#eef0f3;
--pass:#1a7f37;--pass-bg:#e6f4ea;--warn:#9a6700;--warn-bg:#fff4d6;--fail:#c62828;--fail-bg:#fde7e7;--info:#3b5bdb;--info-bg:#e8edff}
@media (prefers-color-scheme: dark){:root{--bg:#16181c;--fg:#e6e8eb;--muted:#a0a6b0;--line:#30343b;--panel:#1e2126;--code:#262a30;
--pass:#56d364;--pass-bg:#12291a;--warn:#e3b341;--warn-bg:#2e2510;--fail:#ff7b72;--fail-bg:#3a1618;--info:#8fa8ff;--info-bg:#1b2240}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:960px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:20px;margin:0 0 16px}h2{font-size:17px;margin:28px 0 10px;border-bottom:1px solid var(--line);padding-bottom:4px}
h3{font-size:15px;margin:20px 0 8px}
.verdict{display:flex;gap:12px;align-items:baseline;padding:14px 16px;border-radius:8px;border:1px solid var(--line)}
.badge{font-weight:700;letter-spacing:.04em;padding:2px 10px;border-radius:999px;white-space:nowrap}
.pass .badge,.badge.pass{color:var(--pass);background:var(--pass-bg)}.warn .badge,.badge.warn{color:var(--warn);background:var(--warn-bg)}
.fail .badge,.badge.fail{color:var(--fail);background:var(--fail-bg)}.badge.info{color:var(--info);background:var(--info-bg)}
.verdict.pass{background:var(--pass-bg)}.verdict.warn{background:var(--warn-bg)}.verdict.fail{background:var(--fail-bg)}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:5px 10px;border-bottom:1px solid var(--line);vertical-align:top;overflow-wrap:anywhere}
th{color:var(--muted);font-weight:600}
td.k{color:var(--muted);width:30%}
.scroll{overflow-x:auto}
ul.findings{list-style:none;padding:0;margin:0}ul.findings li{display:flex;gap:10px;align-items:baseline;padding:4px 0}
pre{background:var(--code);padding:10px 12px;border-radius:6px;overflow-x:auto;font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;margin:8px 0}
code{font:13px ui-monospace,SFMono-Regular,Menlo,monospace}
figure{margin:10px 0}figure img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;image-rendering:pixelated}
figcaption{color:var(--muted);font-size:13px}
footer{color:var(--muted);font-size:12px;margin-top:32px}
</style>
</head>
<body>
<main>
<h1>{{.Title}}</h1>
<section id="verdict">
<h2>Verdict</h2>
<div class="verdict {{.VerdictClass}}"><span class="badge">{{.Verdict}}</span><span>{{.Reason}}</span></div>
</section>
<section id="summary">
<h2>Summary</h2>
<table>{{range .Summary}}<tr><td class="k">{{.Label}}</td><td>{{show .}}</td></tr>{{end}}</table>
</section>
<section id="details">
<h2>Details</h2>
{{if .Findings}}<h3>Findings</h3>
<ul class="findings">{{range .Findings}}<li><span class="badge {{lower .Level}}">{{.Level}}</span><span>{{.Text}}</span></li>{{end}}</ul>{{end}}
{{range .Sections}}<h3>{{.Title}}</h3>
{{if .Rows}}<table>{{range .Rows}}<tr><td class="k">{{.Label}}</td><td>{{show .}}</td></tr>{{end}}</table>{{end}}
{{with .Table}}<div class="scroll"><table><tr>{{range .Header}}<th>{{.}}</th>{{end}}</tr>{{range .Rows}}<tr>{{range .}}<td>{{.}}</td>{{end}}</tr>{{end}}</table></div>{{end}}
{{if .Lines}}<pre>{{.Lines}}</pre>{{end}}
{{with .Image}}<figure><img src="{{.Src}}" alt="{{.Caption}}"><figcaption>{{.Caption}}</figcaption></figure>{{end}}
{{end}}
</section>
<section id="next-steps">
<h2>Next steps</h2>
{{if .NextSteps}}<pre>{{range .NextSteps}}{{.}}
{{end}}</pre>{{else}}<p>Nothing to do.</p>{{end}}
</section>
<footer>Generated by VisionServe. Self-contained: no external assets.</footer>
</main>
</body>
</html>
`))
