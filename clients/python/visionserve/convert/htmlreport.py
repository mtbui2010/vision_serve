"""The self-contained HTML page of `visionserve check --report FILE.html`.

One file, nothing fetched: inline CSS, figures as base64 data URIs (JPEG). Sections Verdict /
Summary / Details / Next steps, light and dark through prefers-color-scheme. The layout is shared
with the Go reports (internal/cli/clireport): a header banner coloured by verdict (green PASS,
amber WARN, red FAIL), system-ui font, max-width 960px, tables with zebra rows.

numpy + PIL only (figures are drawn with PIL; no matplotlib).
"""
from __future__ import annotations

import base64
import datetime as _dt
import html
import io
import re
from typing import List, Optional, Sequence

import numpy as np

STATUS_CLASS = {"PASS": "pass", "WARN": "warn", "FAIL": "fail", "ERROR": "warn", "SKIP": "skip", "INFO": "skip"}

CSS = """
:root{--bg:#ffffff;--fg:#1f2328;--muted:#59636e;--card:#f6f8fa;--border:#d1d9e0;--zebra:#f6f8fa;
--code:#eff2f5;--pass:#1a7f37;--warn:#9a6700;--fail:#cf222e;--skip:#6e7781}
@media (prefers-color-scheme: dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--card:#151b23;
--border:#3d444d;--zebra:#151b23;--code:#262c36;--pass:#238636;--warn:#9e6a03;--fail:#da3633;--skip:#6e7681}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,
"Helvetica Neue",Arial,sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:0 16px}
header.banner{color:#fff;padding:22px 0 18px}
header.banner.pass{background:var(--pass)}header.banner.warn{background:var(--warn)}
header.banner.fail{background:var(--fail)}header.banner.skip{background:var(--skip)}
header .verdict{font-size:13px;font-weight:700;letter-spacing:.08em;opacity:.9}
header h1{margin:4px 0 6px;font-size:22px;line-height:1.35;font-weight:600}
header .meta{font-size:13px;opacity:.9}
main{padding:8px 0 48px}
h2{font-size:18px;margin:28px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--border)}
h3{font-size:15px;margin:0 0 6px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;vertical-align:top;padding:7px 10px;border-bottom:1px solid var(--border)}
th{font-weight:600;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
tbody tr:nth-child(even){background:var(--zebra)}
.pill{display:inline-block;min-width:44px;text-align:center;padding:1px 8px;border-radius:999px;color:#fff;
font-size:12px;font-weight:700}
.pill.pass{background:var(--pass)}.pill.warn{background:var(--warn)}.pill.fail{background:var(--fail)}
.pill.skip{background:var(--skip)}
.cause{margin:6px 0 0;font-size:13.5px}.cause b{font-weight:600}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px 16px;margin:12px 0}
.card .sub{color:var(--muted);font-size:13px;margin:2px 0 8px}
.card ul{margin:6px 0 0;padding-left:20px;font-size:13.5px}
figure{margin:14px 0}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}
.panels img{width:100%;height:auto;display:block;border-radius:4px;border:1px solid var(--border)}
.panels .lbl{font-size:12.5px;color:var(--muted);margin-top:4px}
figcaption{font-size:13px;color:var(--muted);margin-top:6px}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px}
code{background:var(--code);padding:1px 4px;border-radius:4px}
pre{background:var(--code);padding:12px;border-radius:6px;overflow-x:auto;white-space:pre}
details summary{cursor:pointer;color:var(--muted);font-size:13.5px}
ol{padding-left:22px}ol li{margin:4px 0}
footer{color:var(--muted);font-size:12px;border-top:1px solid var(--border);padding:12px 0 24px}
"""


# --------------------------------------------------------------------------------------------
# figures (PIL)
# --------------------------------------------------------------------------------------------

def tensor_image(t, mean: Sequence[float], std: Sequence[float]):
    """A CHW float tensor drawn back as an RGB photo with the given normalisation."""
    from PIL import Image
    a = np.asarray(t, np.float32)
    if a.ndim == 4:
        a = a[0]
    c = a.shape[0]
    m = np.resize(np.asarray(mean, np.float32), c)[:, None, None]
    s = np.resize(np.asarray(std, np.float32), c)[:, None, None]
    img = ((a * s + m) * 255.0).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
    if c == 1:
        img = np.repeat(img, 3, axis=2)
    return Image.fromarray(np.ascontiguousarray(img[..., :3]))


# magma, sampled at 0, .25, .5, .75, 1
_MAGMA = np.array([[0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191]], np.float32)


def heatmap_image(levels: np.ndarray, vmax: Optional[float] = None):
    """A 2-D map of gray-level differences as a magma heatmap (0 = black)."""
    from PIL import Image
    lv = np.asarray(levels, np.float32)
    top = float(vmax) if vmax else max(1.0, float(np.percentile(lv, 99.5)))
    x = (lv / top).clip(0, 1)
    pos = np.linspace(0, 1, len(_MAGMA))
    rgb = np.stack([np.interp(x, pos, _MAGMA[:, k]) for k in range(3)], -1)
    return Image.fromarray(rgb.astype(np.uint8))


REF_COLOR, SRV_COLOR = (9, 105, 218), (232, 89, 12)


def draw_boxes(pil, ref_dets: List[dict], srv_dets: List[dict], max_side: int = 720):
    """The photo with the reference boxes (thick blue) and the served boxes (thin orange)."""
    from PIL import ImageDraw
    img = pil.convert("RGB").copy()
    k = min(1.0, max_side / max(img.size))
    if k < 1:
        img = img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))))
    d = ImageDraw.Draw(img)
    from PIL import ImageFont
    try:
        font = ImageFont.load_default(size=13)  # Pillow >= 10.1: a scalable built-in font
    except TypeError:
        font = ImageFont.load_default()
    th = 16

    def box(det, color, width, below):
        x, y, w, h = (float(v) * k for v in det["bbox"])
        d.rectangle([x, y, x + w, y + h], outline=color, width=width)
        text = f"{det['cls']} {float(det['conf']):.2f}"
        tw = d.textlength(text, font=font)
        ty = min(y + h + 1, img.height - th) if below else max(0, y - th)
        d.rectangle([x, ty, x + tw + 6, ty + th], fill=color)
        d.text((x + 3, ty + 1), text, fill=(255, 255, 255), font=font)

    for det in ref_dets:
        box(det, REF_COLOR, 5, False)
    for det in srv_dets:
        box(det, SRV_COLOR, 2, True)
    return img


def data_uri(img, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------------------------

def _e(s) -> str:
    return html.escape(str(s), quote=True)


def _rich(s) -> str:
    """Escaped text with `code` spans rendered as <code>."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", _e(s))


def _pill(status: str) -> str:
    return f'<span class="pill {STATUS_CLASS.get(status, "skip")}">{_e(status)}</span>'


def _scalar_metrics(m: dict) -> List[tuple]:
    out = []
    for k, v in m.items():
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float, np.integer, np.floating)):
            v = float(v)
            out.append((k, f"{v:.4g}"))
        elif isinstance(v, str) and len(v) < 160:
            out.append((k, v))
    return out


def _tier_card(t: dict) -> str:
    rows = "".join(f"<tr><td><code>{_e(k)}</code></td><td>{_e(v)}</td></tr>" for k, v in _scalar_metrics(t["metrics"]))
    per = t["metrics"].get("per_image") or []
    per_html = ""
    if per and isinstance(per, list) and isinstance(per[0], dict) and "image" in per[0]:
        body = "".join(
            f"<tr><td>{_e(p['image'])}</td><td>{_e(_num(p.get('mean_levels')))}</td>"
            f"<td>{_e(_num(p.get('max_levels')))}</td><td>{_e(', '.join(p.get('diagnosis') or []) or '-')}</td></tr>"
            for p in per)
        per_html = ("<details><summary>per photo</summary><table><thead><tr><th>photo</th><th>mean |Δ| (gray "
                    "levels)</th><th>max</th><th>diagnosis</th></tr></thead><tbody>" + body + "</tbody></table></details>")
    notes = "".join(f"<li>{_rich(n)}</li>" for n in t.get("notes") or [])
    return (f'<div class="card"><h3>{_e(t["tier"])} · {_e(t["title"])} {_pill(t["status"])}</h3>'
            f'<div class="sub">{_e(t["summary"])}</div>'
            + (f"<ul>{notes}</ul>" if notes else "")
            + (f"<details><summary>numbers</summary><table><tbody>{rows}</tbody></table></details>" if rows else "")
            + per_html + "</div>")


def _num(v) -> str:
    return "-" if v is None else f"{float(v):.2f}"


def _figure(f: dict) -> str:
    panels = "".join(f'<div><img alt="{_e(lbl)}" src="{data_uri(img)}"><div class="lbl">{_e(lbl)}</div></div>'
                     for img, lbl in f["panels"])
    return (f'<figure><h3>{_e(f["title"])}</h3><div class="panels">{panels}</div>'
            f'<figcaption>{_e(f.get("caption", ""))}</figcaption></figure>')


def render_html(out: dict, figures: Optional[list] = None) -> str:
    """out: check.summarise()'s object. figures: [{"title", "caption", "panels": [(PIL image, label)]}]."""
    s, d = out["summary"], out["details"]
    verdict = out["verdict"]
    cls = STATUS_CLASS.get(verdict, "skip")
    created = d.get("created") or _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
    rows = []
    for r in s["checks"]:
        extra = ""
        if r.get("cause"):
            extra += f'<div class="cause"><b>Likely cause:</b> {_rich(r["cause"])}.</div>'
        if r.get("fix"):
            extra += f'<div class="cause"><b>Fix:</b> {_rich(r["fix"])}.</div>'
        rows.append(f"<tr><td>{_e(r['name'])}</td><td>{_pill(r['status'])}</td><td>{_rich(r['finding'])}{extra}</td></tr>")
    ref = s.get("reference") or {}
    facts = [("Model", f"{s['model']} ({s['task']}, {s['architecture']})"), ("Server", s["server"]),
             ("Manifest", s["manifest"]), ("Photos", f"{s['images']} of {s.get('images_in_dir', '?')} in {s['images_dir']}"),
             ("Labels", s.get("labels") or "none"),
             ("Preprocessing reference", ref.get("preprocessing") or "none"),
             ("Output reference", ref.get("outputs") or "none (no reference model)")]
    facts_html = "".join(f"<tr><td>{_e(k)}</td><td>{_e(v)}</td></tr>" for k, v in facts)
    figs = "".join(_figure(f) for f in figures or [])
    cards = "".join(_tier_card(t) for t in d.get("tiers") or [])
    th = d.get("thresholds") or {}
    th_html = ", ".join(f"{_e(k)}={_e(v)}" for k, v in th.items())
    steps = "".join(f"<li>{_rich(x)}</li>" for x in s.get("next_steps") or [])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{_e(verdict)}: visionserve check {_e(s['model'])}</title>
<style>{CSS}</style></head>
<body>
<header id="verdict" class="banner {cls}"><div class="wrap">
<div class="verdict">VERDICT · {_e(verdict)}</div>
<h1>{_e(out['reason'])}</h1>
<div class="meta">visionserve check {_e(s['model'])} · {_e(s['server'])} · {_e(created)}</div>
</div></header>
<main class="wrap">
<section id="summary"><h2>Summary</h2>
<table><thead><tr><th>Check</th><th>Status</th><th>What we found</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
{figs}
</section>
<section id="details"><h2>Details</h2>
<table><tbody>{facts_html}</tbody></table>
{cards}
<details><summary>the converter's tier table</summary><pre>{_e(d.get('table', ''))}</pre></details>
</section>
<section id="next-steps"><h2>Next steps</h2><ol>{steps}</ol></section>
<footer>Generated by <code>visionserve check</code>. Gray levels are 0-255 pixel units. Thresholds: {th_html}.</footer>
</main>
</body></html>
"""
