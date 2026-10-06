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

_REPO_SDK = REPO / "clients" / "python"


def _use_repo_sdk() -> bool:  # pragma: no cover - depends on the environment
    """Put this repository's SDK (clients/python) first on sys.path and forget an imported one."""
    if not (_REPO_SDK / "visionserve" / "__init__.py").is_file():
        return False
    sys.path.insert(0, str(_REPO_SDK))
    for name in [m for m in sys.modules if m == "visionserve" or m.startswith("visionserve.")]:
        del sys.modules[name]
    return True


# If the SDK is not installed, use the copy in this repository (clients/python).
try:  # pragma: no cover - depends on the environment
    import visionserve  # noqa: F401
except ImportError:  # pragma: no cover
    _use_repo_sdk()
    import visionserve  # noqa: F401,E402

# The drawing helpers below need SDK 0.3.1 (draw options, draw_prompts). An older installed SDK:
# use the repository's copy instead.
try:  # pragma: no cover - depends on the environment
    from visionserve.visualize import draw_prompts as _sdk_draw_prompts  # noqa: F401
except ImportError:  # pragma: no cover
    if _use_repo_sdk():
        print("handson: the installed visionserve %s is older than 0.3.1; using the copy in %s"
              % (getattr(visionserve, "__version__", "?"), _REPO_SDK))
        import visionserve  # noqa: F401,E402

from visionserve import Client, Result, VisionServeError  # noqa: E402
from visionserve import visualize as _vis  # noqa: E402

__all__ = [
    "connect",
    "host_url",
    "ensure_model",
    "has_model",
    "photo",
    "PHOTOS",
    "draw",
    "mark",
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


def has_model(client: Client, name: str) -> bool:
    """True when the server has the model `name` on disk (state "available" or "loaded")."""
    return _model_state(client, name) in ("available", "loaded")


def ensure_model(client: Client, name: str) -> None:
    """Make sure the model `name` is installed on the server.

    If the server already has it, this only prints a short line. If it is missing
    (state "not_downloaded", or not in the list at all), it runs
    `visionserve pull NAME` with `run_cli` and prints what happened.
    Set HANDSON_NO_PULL=1 to only report a missing model and never download.
    Use `has_model(client, name)` when your code needs a True/False answer.
    """
    state = _model_state(client, name)
    if state in ("available", "loaded"):
        print("Model %r is installed (state: %s)." % (name, state))
        return
    why = "is not downloaded yet" if state == "not_downloaded" else "is not known to this server"
    if os.environ.get("HANDSON_NO_PULL", "").strip() not in ("", "0", "false", "no"):
        print("Model %r %s. HANDSON_NO_PULL is set, so it is NOT downloaded.\n"
              "Download it yourself with:  visionserve pull %s" % (name, why, name))
        return
    print("Model %r %s. Downloading it with `visionserve pull %s` (only once)..." % (name, why, name))
    run_cli("pull", name)
    state = _model_state(client, name)
    if state in ("available", "loaded"):
        print("Done: model %r is installed (state: %s)." % (name, state))
        return
    print("The server still reports model %r as %r.\n"
          "Check that the command-line program and the server use the same models folder\n"
          "(in Docker, use VISIONSERVE_CLI=\"docker exec visionserve visionserve\")." % (name, state))


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
# The SDK's draw() options that give the notebooks' look: class colours, mask contours, all the
# grasps the result holds (predict() already kept the best per object), no mask boxes.
_DRAW_STYLE = dict(color_by="class", mask_outline=True, alpha=0.5, max_grasps_per_object=None,
                   mask_boxes=False)


def _inferno(t):
    """matplotlib's "inferno" colour map: the depth picture is bright where it is near."""
    import matplotlib

    return matplotlib.colormaps["inferno"](t)


def _without_masks(result: Result) -> Result:
    import dataclasses

    return dataclasses.replace(result, masks=[])


def draw(image: Any, result: Result | None = None, show_masks: bool = True):
    """Draw a result on the photo and return a new PIL image (the photo's own size).

    Draws masks (translucent colours with an outline), detection boxes with "class conf%"
    labels and grasps (a line between the two jaws plus a short bar for each jaw), with the
    SDK's `visionserve.visualize.draw`.
    Depth maps and classifications are not drawn here; `show` puts them beside the photo.
    """
    img = _load_image(image)
    if result is None:
        return img
    if not show_masks:
        result = _without_masks(result)
    return _vis.draw(result, img, depth="none", classes="none", **_DRAW_STYLE)


def mark(image: Any, boxes: Any = None, points: Any = None):
    """Draw prompts on a copy of the photo and return it (a PIL image).

    `boxes`: one box `[x, y, w, h]` or a list of boxes, drawn as dashed rectangles.
    `points`: one point `[x, y]` / `[x, y, label]` or a list of them. Label 1 (the default)
    is drawn as a green dot ("on the object"), label 0 as a red cross ("not on the object").
    Use it to see the prompt you send, for example `show(mark(photo("cat"), boxes=box), res)`.
    This is the SDK's `visionserve.visualize.draw_prompts`.
    """
    return _vis.draw_prompts(_load_image(image), boxes=boxes, points=points)


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


def _figure(panels: list, titles: list, max_width: int) -> None:
    """Lay out PIL images in one row and display them."""
    import matplotlib.pyplot as plt

    dpi = 100
    total_w = sum(p.size[0] for p in panels)
    max_h = max(p.size[1] for p in panels)
    scale = min(1.0, max_width / float(total_w))
    fig_w = total_w * scale / dpi
    fig_h = max_h * scale / dpi + (0.35 if any(titles) else 0.1)
    fig, axes = plt.subplots(1, len(panels), figsize=(fig_w, fig_h), dpi=dpi,
                             gridspec_kw={"width_ratios": [p.size[0] for p in panels]}, squeeze=False)
    fs = max(7, min(11, int(9 * scale + 3)))
    for ax, p, t in zip(axes[0], panels, titles):
        ax.imshow(p)
        ax.set_axis_off()
        if t:
            ax.set_title(t, fontsize=fs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout()
    _display_figure(fig)


def show(image: Any, result: Result | None = None, title: str | None = None,
         max_width: int = 900, show_masks: bool = True) -> None:
    """Show a photo, with a VisionServe result drawn on it, inside the notebook.

    `image` can be a file path, a PIL image or a numpy array (RGB).
    If `result` has boxes, masks or grasps, they are drawn on the photo.
    If it has a depth map, the depth picture is shown on the right (bright = near, dark = far).
    If it has classifications, the top-5 labels are shown as bars on the right.
    `max_width` limits the width of the picture in pixels. `show_masks=False` hides masks.
    """
    img = _load_image(image)
    if result is not None:
        if not show_masks:
            result = _without_masks(result)
        img = _vis.draw(result, img, depth="side", classes="bars", depth_colormap=_inferno, **_DRAW_STYLE)
    _figure([img], [title if title is not None else (_auto_title(result) if result else "")], max_width)


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
                    .style.format(_cell).hide(axis="index"))
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
