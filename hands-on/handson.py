"""Small helpers for the VisionServe hands-on notebooks.

The notebooks in this folder import this file so that each notebook can stay short:

    from handson import connect, ensure_model, photo, show

    client = connect()                    # talk to a running VisionServe server
    ensure_model(client, "rf-detr-nano")  # download the model if it is missing
    res = client.predict("rf-detr-nano", photo("cat"))
    show(photo("cat"), res)               # draw the answer on the photo

Nothing here runs a model. The server runs the models; this file only connects to it,
calls the command-line program, and draws pictures.

Environment variables (all optional):

    VS_HOST          address of the server (default http://127.0.0.1:11435)
    VISIONSERVE_CLI  a full command to run the command-line program, for example
                     "docker exec visionserve visionserve" when the server runs in Docker
    VISIONSERVE_BIN  path of the visionserve program (the Go binary)
    HANDSON_NO_PULL  set to 1 to never download models (ensure_model only reports)
"""

from __future__ import annotations

import colorsys
import hashlib
import html
import io
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_HOST = "http://127.0.0.1:11435"

# If the SDK is not installed, use the copy in this repository (clients/python).
try:  # pragma: no cover - depends on the environment
    import visionserve  # noqa: F401
except ImportError:  # pragma: no cover
    _sdk = REPO / "clients" / "python"
    if (_sdk / "visionserve" / "__init__.py").is_file():
        sys.path.insert(0, str(_sdk))
    import visionserve  # noqa: F401,E402

from visionserve import Client, Result, VisionServeError  # noqa: E402

__all__ = [
    "connect",
    "host_url",
    "ensure_model",
    "photo",
    "PHOTOS",
    "draw",
    "show",
    "show_side_by_side",
    "run_cli",
    "show_html_report",
    "table",
    "print_json",
]

# Photos in hands-on/images/: COCO val2017 photos, CC BY 2.0 (see images/CREDITS.md).
PHOTOS = {
    "cat": 177015,
    "dogs": 372819,
    "food": 389381,
    "living-room": 29596,
    "desk": 363840,
    "elephant": 564133,
    "person": 8021,
    "kitchen": 34873,
}

START_HELP = """
Could not reach the VisionServe server at {host}
({error}).

Start the server first, then run this cell again. Pick ONE way:

  * Docker, NVIDIA GPU:
      docker run -d --gpus all -p 11435:11435 -v ~/.visionserve_models:/root/.models \\
        --name visionserve mtbui2010/visionserve:latest
  * Docker, CPU only:
      docker run -d -p 11435:11435 -v ~/.visionserve_models:/root/.models \\
        --name visionserve mtbui2010/visionserve:latest-cpu
  * From source (in the repository folder):
      make build
      bin/visionserve serve

If the server runs on another address, set VS_HOST (for example
VS_HOST=http://192.168.0.10:11435) or call connect("http://...").
More help: hands-on/README.md
"""

CLI_HELP = """
Could not find the visionserve command-line program (the Go binary).

Notebooks 03 to 07 (and pulling models) need it. Pick ONE way:

  * The server runs in Docker: set
      VISIONSERVE_CLI="docker exec visionserve visionserve"
    (before you start Jupyter, or in a cell: os.environ["VISIONSERVE_CLI"] = "...")
  * You built it from source (make build): set
      VISIONSERVE_BIN=/path/to/vision_serve/bin/visionserve
    or put bin/ on your PATH.

Note: `pip install visionserve` also installs a Python command called `visionserve`.
That one only talks to a running server; it cannot pull, check or inspect models.
More help: hands-on/README.md
"""


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #
def host_url(host: str | None = None) -> str:
    """Return the server address to use.

    Order: the `host` argument, then the environment variable VS_HOST,
    then http://127.0.0.1:11435 (the default port of VisionServe).
    """
    url = host or os.environ.get("VS_HOST") or DEFAULT_HOST
    if "://" not in url:
        url = "http://" + url
    return url.rstrip("/")


def connect(host: str | None = None) -> Client:
    """Connect to a running VisionServe server and return a `visionserve.Client`.

    The address comes from `host`, else from the environment variable VS_HOST,
    else it is http://127.0.0.1:11435. The function asks the server for /api/health.
    If the server does not answer, it prints how to start one and raises RuntimeError.
    """
    url = host_url(host)
    client = Client(url)
    try:
        status = client.health()
    except VisionServeError as e:
        print(START_HELP.format(host=url, error=_short(str(e))))
        raise RuntimeError("VisionServe server not reachable at %s" % url) from None
    if not isinstance(status, dict) or status.get("status") != "ok":
        print(START_HELP.format(host=url, error="unexpected answer %r" % (status,)))
        raise RuntimeError("VisionServe server at %s did not answer ok" % url)
    print("Connected to VisionServe at %s: %s" % (url, status))
    return client


def _model_state(client: Client, name: str) -> str | None:
    """The model's state from GET /api/models, or None when the server does not know it."""
    for m in client.list_models():
        if m.name == name:
            return m.state
    return None


def ensure_model(client: Client, name: str) -> bool:
    """Make sure the model `name` is installed on the server.

    If the server already has it, this only prints a short line. If it is missing
    (state "not_downloaded", or not in the list at all), it runs
    `visionserve pull NAME` with `run_cli` and prints what happened.
    Set HANDSON_NO_PULL=1 to only report a missing model and never download.

    Returns True when the model is ready to use, False when it is still missing.
    """
    state = _model_state(client, name)
    if state in ("available", "loaded"):
        print("Model %r is installed (state: %s)." % (name, state))
        return True
    why = "is not downloaded yet" if state == "not_downloaded" else "is not known to this server"
    if os.environ.get("HANDSON_NO_PULL", "").strip() not in ("", "0", "false", "no"):
        print("Model %r %s. HANDSON_NO_PULL is set, so it is NOT downloaded.\n"
              "Download it yourself with:  visionserve pull %s" % (name, why, name))
        return False
    print("Model %r %s. Downloading it with `visionserve pull %s` (only once)..." % (name, why, name))
    run_cli("pull", name)
    state = _model_state(client, name)
    if state in ("available", "loaded"):
        print("Done: model %r is installed (state: %s)." % (name, state))
        return True
    print("The server still reports model %r as %r.\n"
          "Check that the command-line program and the server use the same models folder\n"
          "(in Docker, use VISIONSERVE_CLI=\"docker exec visionserve visionserve\")." % (name, state))
    return False


# --------------------------------------------------------------------------- #
# Photos
# --------------------------------------------------------------------------- #
def photo(key: str) -> Path:
    """Return the path of a bundled photo in hands-on/images/.

    Keys: cat, dogs, food, living-room, desk, elephant, person, kitchen.
    The path is relative to the current folder when the photo is inside it
    (so printed paths stay short), else absolute. Raises KeyError for an unknown key.
    """
    if key not in PHOTOS:
        raise KeyError("unknown photo %r; use one of: %s" % (key, ", ".join(sorted(PHOTOS))))
    path = HERE / "images" / ("%s.jpg" % key)
    try:
        rel = path.relative_to(Path.cwd().resolve())
        return rel
    except ValueError:
        return path


def _load_image(image: Any):
    """Open a path, bytes, PIL image or numpy array as an RGB PIL image (EXIF rotation applied)."""
    from PIL import Image, ImageOps

    if isinstance(image, (str, Path)):
        img = Image.open(image)
        img = ImageOps.exif_transpose(img)
    elif isinstance(image, (bytes, bytearray)):
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(image)))
    elif isinstance(image, Image.Image):
        img = image
    else:
        import numpy as np

        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.size and float(np.nanmax(arr)) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        img = Image.fromarray(arr)
    return img.convert("RGB")


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def _colour(name: str) -> tuple:
    """A stable, bright colour (r, g, b in 0..1) for a class name."""
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:6], 16) / float(0xFFFFFF)
    return colorsys.hsv_to_rgb(h, 0.75, 1.0)


def _mask_colours(n: int) -> list:
    """`n` different colours for masks without a class name."""
    return [colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.7, 1.0) for i in range(n)]


def draw(image: Any, result: Result | None = None, show_masks: bool = True):
    """Draw a result on the photo and return a new PIL image (the photo's own size).

    Draws masks (translucent colours), detection boxes with "class conf" labels and
    grasps (a line between the two jaws plus a short bar for each jaw).
    Depth maps and classifications are not drawn here; `show` puts them beside the photo.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    img = _load_image(image)
    if result is None:
        return img
    w, h = img.size
    canvas = np.asarray(img).astype(np.float32)

    # Masks: blend a colour into the pixels of each mask.
    if show_masks and result.masks:
        labels = _mask_labels(result)
        auto = _mask_colours(len(result.masks))
        for i, m in enumerate(result.masks):
            if not m.rle:
                continue
            try:
                mask = m.to_ndarray(w, h)
            except ValueError:
                continue
            col = _colour(labels[i]) if labels[i] else auto[i]
            canvas[mask] = canvas[mask] * 0.5 + np.array(col, np.float32) * 255 * 0.5
            # A thin outline makes neighbouring masks easy to tell apart.
            edge = mask & ~_erode(mask)
            canvas[edge] = np.array(col, np.float32) * 255
    out = Image.fromarray(canvas.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(out)
    font = _font(max(11, int(round(min(w, h) / 40))))
    lw = max(2, int(round(min(w, h) / 220)))

    for det in result.detections:
        x, y, bw, bh = det.bbox
        col = tuple(int(c * 255) for c in _colour(det.cls or "object"))
        d.rectangle([x, y, x + bw, y + bh], outline=col, width=lw)
        _label(d, x, y, "%s %.2f" % (det.cls or "object", det.conf), col, font)

    if result.grasps:
        import math

        for g in result.grasps:
            col = _quality_colour(g.quality)
            c, s = math.cos(g.theta), math.sin(g.theta)
            hw = g.width / 2.0
            p0 = (g.x - c * hw, g.y - s * hw)
            p1 = (g.x + c * hw, g.y + s * hw)
            plate = max(6.0, min(g.width * 0.35, 22.0)) / 2.0
            px, py = -s * plate, c * plate
            d.line([p0, p1], fill=col, width=lw)
            for q in (p0, p1):  # one bar for each jaw
                d.line([(q[0] - px, q[1] - py), (q[0] + px, q[1] + py)], fill=col, width=lw + 2)
            d.ellipse([g.x - lw, g.y - lw, g.x + lw, g.y + lw], fill=col)
    return out


def _mask_labels(result: Result) -> list:
    """The class name of each mask when a detection with the same box exists, else ""."""
    labels = []
    for i, m in enumerate(result.masks):
        name = ""
        if len(result.masks) == len(result.detections) and i < len(result.detections):
            name = result.detections[i].cls
        else:
            for det in result.detections:
                if [round(v) for v in det.bbox] == [round(v) for v in m.bbox]:
                    name = det.cls
                    break
        labels.append(name)
    return labels


def _erode(mask):
    """A one-pixel erosion of a boolean mask (numpy only)."""
    m = mask
    out = m.copy()
    out[1:, :] &= m[:-1, :]
    out[:-1, :] &= m[1:, :]
    out[:, 1:] &= m[:, :-1]
    out[:, :-1] &= m[:, 1:]
    return out


def _quality_colour(q: float) -> tuple:
    """Grasp quality 0..1 as a colour: red (low), yellow, green (high)."""
    q = max(0.0, min(1.0, float(q)))
    if q < 0.5:
        return (255, int(510 * q), 0)
    return (int(255 * (2 - 2 * q)), 220, 0)


def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # old Pillow
        return ImageFont.load_default()


def _label(d, x: float, y: float, text: str, col: tuple, font) -> None:
    """Text on a filled box, above (x, y) when there is room, else just inside."""
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    ty = y - th - 4 if y - th - 4 >= 0 else y
    d.rectangle([x, ty, x + tw + 4, ty + th + 4], fill=col)
    lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
    d.text((x + 2 - l, ty + 2 - t), text, fill=(0, 0, 0) if lum > 140 else (255, 255, 255), font=font)


def _auto_title(result: Result) -> str:
    parts = ["%s (%s)" % (result.model, result.task)]
    counts = []
    for field, word in (("detections", "detections"), ("masks", "masks"), ("grasps", "grasps"),
                        ("classifications", "labels")):
        n = len(getattr(result, field) or [])
        if n:
            counts.append("%d %s" % (n, word))
    if result.depth_width:
        counts.append("depth %dx%d" % (result.depth_width, result.depth_height))
    if result.embeddings is not None and len(result.embeddings):
        counts.append("%d embedding(s)" % len(result.embeddings))
    if counts:
        parts.append(", ".join(counts))
    if result.duration_ms:
        parts.append("%.0f ms on %s" % (result.duration_ms, result.device or "?"))
    return " | ".join(parts)


def _depth_image(result: Result, size: tuple):
    """The depth map as a colour picture (bright = near) at `size` (w, h)."""
    import numpy as np
    from PIL import Image
    import matplotlib

    d = result.depth_array()
    if d is None or d.size == 0:
        return None
    d = np.asarray(d, np.float32)
    lo, hi = float(np.nanmin(d)), float(np.nanmax(d))
    norm = (d - lo) / (hi - lo) if hi > lo else np.zeros_like(d)
    rgb = (matplotlib.colormaps["inferno"](norm)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize(size, Image.BILINEAR)


def _display_figure(fig) -> None:
    """Show a matplotlib figure inline as a JPEG (small notebooks), or with plt.show()."""
    import matplotlib.pyplot as plt

    try:
        from IPython import get_ipython
        from IPython.display import Image as IPImage, display

        if get_ipython() is not None:
            buf = io.BytesIO()
            fig.savefig(buf, format="jpeg", pil_kwargs={"quality": 82}, bbox_inches="tight",
                        pad_inches=0.05, facecolor="white")
            plt.close(fig)
            display(IPImage(data=buf.getvalue(), format="jpeg"))
            return
    except ImportError:
        pass
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":  # Agg cannot open a window
        plt.show()
    plt.close(fig)


def _figure(panels: list, titles: list, max_width: int, suptitle: str | None = None,
            bars: Any = None) -> None:
    """Lay out PIL images (and optionally one bar chart) in one row and display them."""
    import matplotlib.pyplot as plt

    dpi = 100
    n = len(panels) + (1 if bars is not None else 0)
    total_w = sum(p.size[0] for p in panels) + (panels[0].size[0] * 0.8 if bars is not None else 0)
    max_h = max(p.size[1] for p in panels)
    scale = min(1.0, max_width / float(total_w))
    fig_w = total_w * scale / dpi
    fig_h = max_h * scale / dpi + (0.35 if any(titles) or suptitle else 0.1)
    ratios = [p.size[0] for p in panels] + ([panels[0].size[0] * 0.8] if bars is not None else [])
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), dpi=dpi,
                             gridspec_kw={"width_ratios": ratios}, squeeze=False)
    axes = axes[0]
    fs = max(7, min(11, int(9 * scale + 3)))
    for ax, p, t in zip(axes, panels, titles):
        ax.imshow(p)
        ax.set_axis_off()
        if t:
            ax.set_title(t, fontsize=fs)
    if bars is not None:
        ax = axes[-1]
        names, values = bars
        ypos = list(range(len(names)))[::-1]
        ax.barh(ypos, values, color="#4c72b0")
        ax.set_yticks(ypos)
        ax.set_yticklabels(names, fontsize=fs)
        ax.set_xlim(0, 1)
        ax.set_xlabel("conf", fontsize=fs)
        ax.tick_params(axis="x", labelsize=fs - 1)
        for y, v in zip(ypos, values):
            ax.text(min(v + 0.02, 0.8), y, "%.2f" % v, va="center", fontsize=fs - 1)
        ax.set_title("top-5 labels", fontsize=fs)
    if suptitle:
        fig.suptitle(suptitle, fontsize=fs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout()
    _display_figure(fig)


def show(image: Any, result: Result | None = None, title: str | None = None,
         max_width: int = 900, show_masks: bool = True) -> None:
    """Show a photo, with a VisionServe result drawn on it, inside the notebook.

    `image` can be a file path, a PIL image or a numpy array (RGB).
    If `result` has boxes, masks or grasps, they are drawn on the photo.
    If it has a depth map, the depth picture is shown on the right (bright = near).
    If it has classifications, the top-5 labels are shown as bars on the right.
    `max_width` limits the width of the picture in pixels. `show_masks=False` hides masks.
    """
    img = draw(image, result, show_masks=show_masks)
    panels, titles = [img], [title if title is not None else (_auto_title(result) if result else "")]
    bars = None
    if result is not None:
        depth = _depth_image(result, img.size) if result.depth_width else None
        if depth is not None:
            panels.append(depth)
            titles.append("depth (bright = near)")
        if result.classifications:
            top = sorted(result.classifications, key=lambda c: -c.conf)[:5]
            bars = ([c.cls for c in top], [c.conf for c in top])
    if len(panels) > 1 or bars is not None:
        _figure(panels, ["", *titles[1:]], max_width, suptitle=titles[0], bars=bars)
    else:
        _figure(panels, titles, max_width)


def show_side_by_side(images: list, titles: list[str] | None = None, max_width: int = 1000) -> None:
    """Show several pictures next to each other, in one row.

    Each item of `images` is a path, a PIL image or a numpy array, or a pair
    `(image, result)`: then the result is drawn on the image first (like `show`).
    `titles` gives one title per picture.
    """
    panels = []
    for item in images:
        if isinstance(item, tuple) and len(item) == 2 and (item[1] is None or isinstance(item[1], Result)):
            panels.append(draw(item[0], item[1]))
        else:
            panels.append(_load_image(item))
    titles = list(titles) if titles else [""] * len(panels)
    titles += [""] * (len(panels) - len(titles))
    _figure(panels, titles, max_width)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def _is_script(path: str) -> bool:
    """True for a text script (such as the Python `visionserve` command), False for a binary."""
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"#!"
    except OSError:
        return True


def _find_cli() -> list | None:
    """The command (a list) that runs the Go program, or None when it cannot be found."""
    prefix = os.environ.get("VISIONSERVE_CLI", "").strip()
    if prefix:
        return shlex.split(prefix)
    exe = os.environ.get("VISIONSERVE_BIN", "").strip()
    if exe:
        found = shutil.which(exe) or (exe if os.path.isfile(exe) else None)
        return [found] if found else None
    built = REPO / "bin" / ("visionserve.exe" if os.name == "nt" else "visionserve")
    if built.is_file() and os.access(built, os.X_OK):
        return [str(built)]
    py_bin = Path(sys.executable).parent.resolve()
    for d in os.environ.get("PATH", "").split(os.pathsep):
        for name in ("visionserve", "visionserve.exe"):
            cand = Path(d) / name
            if cand.is_file() and os.access(cand, os.X_OK):
                # Skip the Python client's command of the same name.
                if _is_script(str(cand)) or cand.resolve().parent == py_bin:
                    continue
                return [str(cand)]
    return None


def _tidy(text: str) -> str:
    """Make long absolute paths in program output shorter: the current folder becomes '.',
    the repository folder becomes '<repo>', and the home folder becomes '~'."""
    pairs = [(str(Path.cwd().resolve()), "."), (str(REPO), "<repo>"), (str(Path.home()), "~")]
    for long, short in pairs:
        if long and long not in ("/", "."):
            text = text.replace(long + os.sep, short + os.sep).replace(long, short)
    return text


def run_cli(*args: str, check: bool = True, echo: bool = True) -> str:
    """Run the `visionserve` command-line program and return what it printed on stdout.

    The program is found like this: the environment variable VISIONSERVE_CLI (a full
    command, for example "docker exec visionserve visionserve"), else VISIONSERVE_BIN,
    else bin/visionserve in this repository, else `visionserve` on your PATH (the Python
    command of the same name is skipped). If it cannot be found, this prints how to get it
    and raises RuntimeError.

    With `echo=True` it prints the command and everything the program prints (stdout and
    stderr) while it runs. Exit code 1 means "the check FAILED"; that is a normal result and
    is NOT an error here. With `check=True`, exit code 2 (a setup or usage error) or any
    other code raises RuntimeError.
    """
    cmd = _find_cli()
    if cmd is None:
        print(CLI_HELP)
        raise RuntimeError("visionserve command-line program not found")
    args = [str(a) for a in args]
    if echo:
        shown = "visionserve" if not os.environ.get("VISIONSERVE_CLI", "").strip() else " ".join(cmd)
        print("$ " + " ".join([shown] + [shlex.quote(a) for a in args]))
        sys.stdout.flush()
    try:
        proc = subprocess.Popen(cmd + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                stdin=subprocess.DEVNULL, text=True, encoding="utf-8",
                                errors="replace", bufsize=1)
    except FileNotFoundError:
        print(CLI_HELP)
        raise RuntimeError("could not start %r" % cmd[0]) from None

    lines: "queue.Queue[tuple]" = queue.Queue()

    def _reader(stream, tag):
        for line in iter(stream.readline, ""):
            lines.put((tag, line))
        stream.close()
        lines.put((tag, None))

    threads = [threading.Thread(target=_reader, args=(proc.stdout, "out"), daemon=True),
               threading.Thread(target=_reader, args=(proc.stderr, "err"), daemon=True)]
    for t in threads:
        t.start()
    out, err, open_streams = [], [], 2
    while open_streams:
        tag, line = lines.get()
        if line is None:
            open_streams -= 1
            continue
        (out if tag == "out" else err).append(line)
        if echo:
            # Progress bars redraw one line with "\r"; keep only the last state of it.
            print(_tidy(line.rsplit("\r", 1)[-1]), end="")
            sys.stdout.flush()
    code = proc.wait()
    if echo:
        print("(exit code %d)" % code)
    if check and code not in (0, 1):
        tail = _tidy("".join(err[-5:]).strip())
        raise RuntimeError("visionserve %s failed with exit code %d%s"
                           % (" ".join(args), code, (": " + tail) if tail else ""))
    return "".join(out)


# --------------------------------------------------------------------------- #
# Reports and tables
# --------------------------------------------------------------------------- #
def show_html_report(path: Any, height: int = 700) -> None:
    """Show an HTML report file (for example from `visionserve check`) inside the notebook.

    The file is embedded in the notebook (in an <iframe>), so the saved notebook still shows it.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    frame = ('<iframe srcdoc="%s" style="width:100%%;height:%dpx;border:1px solid #ccc;" '
             'sandbox="allow-scripts"></iframe>' % (html.escape(text, quote=True), int(height)))
    try:
        from IPython.display import HTML, display

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # IPython suggests IFrame; srcdoc keeps the report in the notebook
            display(HTML(frame))
    except ImportError:  # pragma: no cover
        print("Open this file in a web browser: %s" % path)


def table(rows: list[list], headers: list[str]) -> None:
    """Print a table: a pandas table in a notebook when pandas is installed, else plain text."""
    try:
        import pandas as pd
        from IPython import get_ipython
        from IPython.display import display

        if get_ipython() is not None:
            display(pd.DataFrame([list(r) for r in rows], columns=list(headers))
                    .style.hide(axis="index"))
            return
    except Exception:
        pass
    print(_text_table(rows, headers))


def _text_table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    cells = [[str(h) for h in headers]] + [[_cell(v) for v in r] for r in rows]
    widths = [max(len(r[i]) if i < len(r) else 0 for r in cells) for i in range(len(headers))]
    lines = []
    for k, r in enumerate(cells):
        lines.append("  ".join((r[i] if i < len(r) else "").ljust(widths[i]) for i in range(len(headers))).rstrip())
        if k == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _cell(v: Any) -> str:
    if isinstance(v, float):
        return "%.3g" % v if abs(v) < 1000 else "%.0f" % v
    return str(v)


def print_json(data: Any, max_items: int = 3, max_text: int = 60) -> None:
    """Print a result (or any dict) as indented JSON, shortened so it fits on the screen.

    Lists longer than `max_items` show only their first items and a note such as
    "... 12 more". Long strings (like a mask's "rle") are cut to `max_text` characters.
    """
    if isinstance(data, Result):
        data = data.to_json()

    def _short_value(v):
        if isinstance(v, dict):
            return {k: _short_value(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            if v and all(isinstance(x, (int, float)) for x in v) and len(v) <= 6:
                return [round(x, 2) if isinstance(x, float) else x for x in v]
            head = [_short_value(x) for x in list(v)[:max_items]]
            if len(v) > max_items:
                head.append("... %d more" % (len(v) - max_items))
            return head
        if isinstance(v, float):
            return round(v, 3)
        if isinstance(v, str) and len(v) > max_text:
            return v[:max_text] + "... (%d characters)" % len(v)
        return v

    text = json.dumps(_short_value(data), indent=2, ensure_ascii=False)
    # Keep short lists of numbers (a bbox) on one line.
    text = re.sub(r"\[\s*(-?[\d.e+-]+(?:,\s*-?[\d.e+-]+)*)\s*\]",
                  lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", text)
    print(text)


def _short(text: str, n: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."
