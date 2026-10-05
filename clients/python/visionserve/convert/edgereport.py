"""Verdict / summary / details / next steps for `visionserve sensitivity` and `visionserve optimize`,
as text, as one JSON object (--json) or as one self-contained HTML file (--report).

The layout is the convention the Go diagnostic commands share (internal/cli/clireport):
first line `PASS|WARN|FAIL: <one sentence>`, a short summary table, details, next steps; exit 0 for
PASS/WARN, 1 for FAIL, 2 for a usage or setup error; JSON {"verdict","reason","summary","details"};
HTML: the page of htmlreport.py (`visionserve check`): its CSS, banner coloured by verdict, sections
#verdict / #summary / #details / #next-steps, light/dark via prefers-color-scheme; charts as base64 PNG.

Charts are drawn with Pillow (already a converter dependency): no plotting library.
"""
from __future__ import annotations

import base64
import dataclasses
import datetime as _dt
import io
import json
import math
from typing import Any, Dict, List, Optional, Sequence

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"


@dataclasses.dataclass
class Row:
    key: str                 # JSON key
    label: str               # what people read
    value: Any               # JSON value
    text: str = ""           # printed value ("" = str(value))
    note: str = ""

    def shown(self) -> str:
        if self.text:
            return self.text
        if self.value is None:
            return "n/a"
        if isinstance(self.value, float):
            return fmt_num(self.value)
        return str(self.value)


@dataclasses.dataclass
class Table:
    title: str
    header: List[str]
    rows: List[List[str]]
    notes: List[str] = dataclasses.field(default_factory=list)
    highlight: Optional[int] = None      # row index drawn as the recommended one


@dataclasses.dataclass
class Chart:
    title: str
    png: bytes
    alt: str = ""


@dataclasses.dataclass
class Report:
    title: str
    verdict: str
    reason: str
    summary: List[Row] = dataclasses.field(default_factory=list)
    tables: List[Table] = dataclasses.field(default_factory=list)
    charts: List[Chart] = dataclasses.field(default_factory=list)
    notes: List[str] = dataclasses.field(default_factory=list)
    next_steps: List[str] = dataclasses.field(default_factory=list)
    details: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return 1 if self.verdict == FAIL else 0

    def findings(self) -> List[tuple]:
        """(level, text), worst first: a note is a plain string (INFO) or a (level, text) pair."""
        out = [(n[0], n[1]) if isinstance(n, tuple) else (INFO, n) for n in self.notes]
        rank = {FAIL: 3, WARN: 2, PASS: 1, INFO: 0}
        return sorted(out, key=lambda f: -rank.get(f[0], 0))

    # -- JSON (clireport's shape: findings under details.findings) ---------------------------
    def to_json(self) -> dict:
        details = dict(self.details)
        details["findings"] = [{"level": lv, "text": tx} for lv, tx in self.findings()]
        return clean({"verdict": self.verdict, "reason": self.reason,
                      "summary": {r.key: r.value for r in self.summary}, "details": details})

    def json_text(self) -> str:
        return json.dumps(self.to_json(), indent=2, allow_nan=False)

    # -- text (clireport's layout) ------------------------------------------------------------
    def text(self) -> str:
        out = [f"{self.verdict}: {self.reason}", ""]
        out += _columns([["  " + r.label, r.shown(), r.note] for r in self.summary])
        fs = self.findings()
        if fs:
            out += ["", "Findings"] + [f"  {lv:<4}  {tx}" for lv, tx in fs]
        for t in self.tables:
            out += ["", t.title]
            rows = [t.header] + t.rows if t.header else t.rows
            out += _columns([["  " + c[0]] + c[1:] for c in rows])
            out += [f"  {n}" for n in t.notes]
        for c in self.charts:
            out.append(f"  [image: {c.title} — in the --report HTML]")
        if self.next_steps:
            out += ["", "Next steps"] + [f"  {n}" for n in self.next_steps]
        return "\n".join(out) + "\n"

    # -- HTML (htmlreport.py's page: same CSS, banner, sections) ---------------------------------
    def html(self) -> str:
        from .htmlreport import CSS, STATUS_CLASS, _e, _pill, _rich
        cls = STATUS_CLASS.get(self.verdict, "skip")
        created = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
        summ = "".join(f"<tr><td>{_e(r.label)}</td><td>{_e(r.shown())}</td><td class=\"note\">{_rich(r.note)}</td></tr>"
                       for r in self.summary)
        parts = []
        fs = self.findings()
        if fs:
            parts.append("<h3>Findings</h3><table><tbody>" + "".join(
                f"<tr><td>{_pill(lv)}</td><td>{_rich(tx)}</td></tr>" for lv, tx in fs) + "</tbody></table>")
        for t in self.tables:
            head = "".join(f"<th>{_e(h)}</th>" for h in t.header)
            rec = ' class="rec"'
            body = "".join(f"<tr{rec if t.highlight == i else ''}>"
                           + "".join(f"<td>{_e(c)}</td>" for c in row) + "</tr>" for i, row in enumerate(t.rows))
            notes = "".join(f'<p class="note">{_rich(n)}</p>' for n in t.notes)
            parts.append(f'<h3>{_e(t.title)}</h3><div class="scroll"><table><thead><tr>{head}</tr></thead>'
                         f"<tbody>{body}</tbody></table></div>{notes}")
        for c in self.charts:
            src = "data:image/png;base64," + base64.b64encode(c.png).decode("ascii")
            parts.append(f'<figure><h3>{_e(c.title)}</h3><img class="chart" alt="{_e(c.alt or c.title)}" src="{src}">'
                         "</figure>")
        steps = "".join(f"<li>{_rich(x)}</li>" for x in self.next_steps) or "<li>Nothing to do.</li>"
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{_e(self.verdict)}: {_e(self.title)}</title>
<style>{CSS}{_EXTRA_CSS}</style></head>
<body>
<header id="verdict" class="banner {cls}"><div class="wrap">
<div class="verdict">VERDICT · {_e(self.verdict)}</div>
<h1>{_e(self.reason)}</h1>
<div class="meta">{_e(self.title)} · {_e(created)}</div>
</div></header>
<main class="wrap">
<section id="summary"><h2>Summary</h2>
<table><tbody>{summ}</tbody></table>
</section>
<section id="details"><h2>Details</h2>
{''.join(parts)}
</section>
<section id="next-steps"><h2>Next steps</h2><ol>{steps}</ol></section>
<footer>Generated by <code>{_e(self.title)}</code>. Self-contained: no external assets.</footer>
</main>
</body></html>
"""

    def write_html(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.html())


# On top of htmlreport.CSS: the recommended row, notes in tables, charts (drawn on white).
_EXTRA_CSS = """
tbody tr.rec td{font-weight:700;background:rgba(26,127,55,.14)}
td.note,p.note{color:var(--muted);font-size:13px}
.scroll{overflow-x:auto}
img.chart{max-width:100%;height:auto;display:block;background:#fff;border:1px solid var(--border);border-radius:6px}
"""


def _columns(rows: Sequence[Sequence[str]]) -> List[str]:
    rows = [[str(c) for c in r] for r in rows]
    n = max((len(r) for r in rows), default=0)
    widths = [max((len(r[i]) for r in rows if i < len(r)), default=0) for i in range(n)]
    out = []
    for r in rows:
        cells = [c.ljust(widths[i]) if i < len(r) - 1 else c for i, c in enumerate(r)]
        out.append("  ".join(cells).rstrip())
    return out


def fmt_num(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    if isinstance(v, int):
        return str(v)
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    return f"{v:.{digits}g}"


def clean(v):
    """JSON-safe: non-finite floats -> None, tuples -> lists, recursively."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    return v


# ------------------------------------------------------------------------------------------
# charts (Pillow)
# ------------------------------------------------------------------------------------------

_INK, _MUTED, _BAR, _MARK, _GOOD, _GRID = ((29, 35, 39), (138, 148, 156), (47, 111, 179), (194, 65, 12),
                                           (30, 127, 59), (227, 231, 234))


def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:            # Pillow < 10.1: fixed bitmap font only
        return ImageFont.load_default()


def _tw(draw, text, font) -> int:
    l, _, r, _ = draw.textbbox((0, 0), text, font=font)
    return r - l


def bar_chart(labels: Sequence[str], values: Sequence[Optional[float]], unit: str = "",
              mark: Optional[float] = None, mark_label: str = "budget", highlight: Sequence[int] = (),
              log: bool = False) -> bytes:
    """Horizontal bars, one per label. `mark` draws a dashed reference line (a budget / threshold);
    bars in `highlight` are drawn green. log=True uses a log10 axis (errors spanning decades)."""
    from PIL import Image, ImageDraw
    s = 2                                   # drawn at 2x for high-DPI screens
    font, small = _font(13 * s), _font(11 * s)
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    label_w = max((_tw(probe, l, font) for l in labels), default=0) + 14 * s
    plot_w, row_h = 460 * s, 24 * s
    W, H = label_w + plot_w + 110 * s, len(labels) * row_h + 40 * s
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    finite = [v for v in values if v is not None and math.isfinite(v) and (v > 0 or not log)]
    vmax = max(finite + ([mark] if mark is not None else []), default=1.0) or 1.0
    vmin = min(finite + ([mark] if mark is not None else []), default=vmax)

    def x_of(v):
        if log:
            lo = math.log10(max(vmin, 1e-6)) - 0.3
            hi = math.log10(max(vmax, 1e-6)) + 0.05
            t = (math.log10(max(v, 1e-6)) - lo) / max(hi - lo, 1e-9)
        else:
            t = v / vmax
        return label_w + int(plot_w * max(0.0, min(1.0, t)))

    for i, (lab, v) in enumerate(zip(labels, values)):
        y = 10 * s + i * row_h
        d.text((label_w - 8 * s - _tw(d, lab, font), y + 2 * s), lab, fill=_INK, font=font)
        if v is None or not math.isfinite(v):
            d.text((label_w + 4 * s, y + 2 * s), "n/a" if v is None else "inf", fill=_MARK, font=font)
            continue
        x1 = max(x_of(v), label_w + s)
        d.rectangle([label_w, y + 3 * s, x1, y + row_h - 5 * s], fill=_GOOD if i in highlight else _BAR)
        d.text((x1 + 6 * s, y + 2 * s), fmt_num(v) + (f" {unit}" if unit else ""), fill=_INK, font=font)
    d.line([label_w, 6 * s, label_w, H - 26 * s], fill=_MUTED, width=s)
    if mark is not None and math.isfinite(mark) and mark > 0:
        mx = x_of(mark)
        for y in range(6 * s, H - 26 * s, 8 * s):
            d.line([mx, y, mx, y + 4 * s], fill=_MARK, width=s)
        txt = f"{mark_label} {fmt_num(mark)}"
        d.text((mx - _tw(d, txt, small) // 2, H - 22 * s), txt, fill=_MARK, font=small)
    if log:
        d.text((label_w, H - 22 * s), "log scale", fill=_MUTED, font=small)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()
