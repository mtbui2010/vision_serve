"""Shared machinery for every converter family.

A family (rfdetr, hf, torch, tf) only has to turn a checkpoint into one or more ONNX graphs and
describe them with a `Bundle`. Everything that must be identical across families lives here:

  * the LICENSE GATE (CLAUDE.md rule 1) — permissive only, AGPL refused, checked before any work;
  * the I/O CONTRACT of each VisionServe architecture — a converted graph whose real tensor shapes
    do not match what the Go model decodes is refused here, not discovered as garbage at serve time;
  * the PARITY check — framework output vs ONNX Runtime output on the same input;
  * the manifest writer and the hand-off to the Go binary (`visionserve pull <folder>`), so the
    registry's own validation is the last word, not a Python re-implementation of it.

This package runs only inside the converter image. The server image stays Python-free (rule 3).
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .constants import TEXT_ARCHS

# Keep in sync with internal/registry/manifest.go::licenseAllowlist (the Go gate re-checks).
LICENSE_ALLOWLIST = {
    "apache-2.0": "Apache-2.0",
    "mit": "MIT",
    "bsd-3-clause": "BSD-3-Clause",
    "bsd-2-clause": "BSD-2-Clause",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODELS_DIR = Path(os.environ.get("VISIONSERVE_MODELS", "/root/.models"))
VISIONSERVE_BIN = os.environ.get("VISIONSERVE_BIN", "visionserve")


class ConvertError(Exception):
    """A refusal with a message meant for the user (no traceback)."""


# A registry name is a directory name under --models AND the manifest's `name:`. Anything else
# (an absolute path, "../x", a newline) would make the converter write — and on FAIL, rmtree —
# outside the registry, or inject YAML. The Go registry should enforce the same rule.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_name(name) -> str:
    n = str(name or "")
    if not _NAME_RE.match(n) or len(n) > 128:
        raise ConvertError(f"--name {name!r} is not a valid model name: use letters, digits, '.', '_' and '-' "
                           "(starting with a letter or digit, at most 128 characters) — it becomes a directory "
                           "under --models")
    return n


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------------
# License gate
# --------------------------------------------------------------------------------------------

def canonical_license(declared: Optional[str]) -> str:
    """Return the canonical SPDX id, or raise. AGPL/GPL/unknown/empty are refused."""
    key = (declared or "").strip().lower()
    if key in LICENSE_ALLOWLIST:
        return LICENSE_ALLOWLIST[key]
    if not key:
        raise ConvertError("a --license is required (Apache-2.0, MIT, BSD-3-Clause or BSD-2-Clause). "
                           "Check the ORIGINAL model's license — 'it is on HuggingFace' says nothing about it.")
    raise ConvertError(f"license {declared!r} is not allowed — VisionServe accepts only permissive licenses "
                       "(Apache-2.0 / MIT / BSD). AGPL models (Ultralytics YOLO, FastSAM, YOLO-World) are "
                       "strictly forbidden: one AGPL model would relicense every deployer (CLAUDE.md rule 1).")


def resolve_license(declared: Optional[str], detected: Optional[str], source: str) -> str:
    """Combine the user's --license with a license DETECTED from the checkpoint (e.g. an HF model
    card). A detected non-permissive license is refused even if the user typed a permissive one:
    the flag cannot launder a copyleft model. If both are permissive they must agree."""
    if detected:
        det = canonical_license(detected)  # raises on AGPL/unknown
        if declared and canonical_license(declared) != det:
            raise ConvertError(f"--license {declared} contradicts the license declared by {source} ({detected})")
        return det
    return canonical_license(declared)


# Top-level Python packages of AGPL model code: Ultralytics (YOLOv5/8/11, RT-DETR-in-ultralytics,
# its YOLO-World and FastSAM ports), the original YOLO-World (`yolo_world`) and FastSAM (`fastsam`).
_AGPL_PACKAGES = ("ultralytics", "yolo_world", "fastsam")
# A dotted QUALIFIED NAME of one of those packages (`ultralytics.nn.tasks.DetectionModel`, TorchScript's
# `__torch__.ultralytics.nn...`), not a filesystem path that merely contains the word: the character
# before it must not be part of a path or identifier (`/data/ultralytics.v2/`, `my_fastsam.x`).
_QUALNAME_RE = re.compile(r"(?<![\w/\\.\-])(?:__torch__\.)?(ultralytics|yolo_world|fastsam)\.[A-Za-z_]", re.I)
# License declarations Ultralytics embeds in its exports (metadata.yaml / metadata.json / TFLite
# metadata / TorchScript config.txt: `author: Ultralytics`, `license: AGPL-3.0 License (https://
# ultralytics.com/license)`), and any AGPL declaration in a metadata/text entry.
_LICENSE_TEXT_RE = re.compile(r"\bagpl\b|ultralytics\.com|author\W{0,6}ultralytics", re.I)
# Byte markers safe to look for anywhere in a binary file (long enough not to occur by chance).
_STRONG_BYTES = (b"agpl-3.0", b"ultralytics.com/license", b"https://ultralytics.com")
# Copyleft / non-permissive license texts in a LICENSE file next to the weights.
_COPYLEFT_RE = re.compile(r"GNU\s+(AFFERO\s+|LESSER\s+)?GENERAL\s+PUBLIC\s+LICEN[SC]E|\bAGPL|\bLGPL|\bGPL-?[23]"
                          r"|Mozilla\s+Public\s+License|Server\s+Side\s+Public\s+License|NonCommercial", re.I)

_SCAN_HEAD = 8 << 20          # bytes scanned at each end of a big binary file
_SCAN_ENTRY_MAX = 16 << 20    # zip entries larger than this are tensor data, not code/metadata
_TEXT_NAMES = {"metadata.yaml", "metadata.yml", "metadata.json", "metadata.txt", "args.yaml", "config.json",
               "config.txt", "keras_metadata.pb"}
_LICENSE_NAMES = {"license", "license.txt", "license.md", "copying", "copying.txt", "licence", "licence.txt"}


def _refuse(where: str, why: str):
    raise ConvertError(f"{where} {why}: this is an Ultralytics / YOLO-World / FastSAM (AGPL) model and cannot be "
                       "served by VisionServe whatever --license says (CLAUDE.md rule 1).")


def _check_qualname(name: str, where: str) -> None:
    """`name` is a Python/TorchScript qualified name (module path). Refuse an AGPL package."""
    parts = [p for p in str(name).lower().split(".") if p and p != "__torch__"]
    if parts and parts[0] in _AGPL_PACKAGES:
        _refuse(where, f"is built from {name!r}")


def _check_text(text: str, where: str, license_markers: bool = True) -> None:
    m = _QUALNAME_RE.search(text)
    if m:
        _refuse(where, f"references {text[m.start():m.end() + 30].split()[0]!r}")
    if license_markers:
        m = _LICENSE_TEXT_RE.search(text)
        if m:
            _refuse(where, f"declares {text[max(0, m.start() - 20):m.end() + 30].strip()!r}")


def _check_bytes(data: bytes, where: str) -> None:
    low = data.lower()
    for mk in _STRONG_BYTES:
        if mk in low:
            _refuse(where, f"contains {mk.decode()!r}")


def _pickle_names(data: bytes, max_pickles: int = 8):
    """(class paths, string constants) of the pickle(s) at the start of `data`, from the opcodes —
    so a path string elsewhere in the file cannot be mistaken for a class. Legacy torch.save files
    are several pickles back to back (magic, protocol, sys info, the object); all are read."""
    import io
    import pickletools
    classes, strings = [], []
    f = io.BytesIO(data)
    for _ in range(max_pickles):
        if f.tell() >= len(data) or data[f.tell():f.tell() + 1] != b"\x80":
            break
        recent, memo = [], {}
        try:
            for op, arg, _pos in pickletools.genops(f):
                if op.name == "GLOBAL" and isinstance(arg, str):
                    classes.append(arg.replace(" ", ".").replace("\n", "."))
                elif isinstance(arg, str) and op.name in ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE",
                                                          "BINUNICODE8", "STRING", "BINSTRING", "SHORT_BINSTRING"):
                    strings.append(arg)
                    recent.append(arg)
                elif op.name in ("BINPUT", "LONG_BINPUT", "PUT") and recent:
                    memo[arg] = recent[-1]
                elif op.name == "MEMOIZE" and recent:
                    memo[len(memo)] = recent[-1]
                elif op.name in ("BINGET", "LONG_BINGET", "GET") and arg in memo:
                    recent.append(memo[arg])
                elif op.name == "STACK_GLOBAL" and len(recent) >= 2:
                    classes.append(f"{recent[-2]}.{recent[-1]}")
        except Exception:  # noqa: BLE001 — truncated head / not a pickle after all: keep what was read
            break
    return classes, strings


def _check_pickle(data: bytes, where: str) -> None:
    classes, strings = _pickle_names(data)
    for c in classes:
        _check_qualname(c, where)
    for s in strings:
        _check_text(s, where, license_markers=False)
        _check_bytes(s.encode("utf-8", "replace"), where)


def _scan_zip(path: Path, label: str) -> None:
    import zipfile
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        # torch.save / torch.jit.save archives put every entry under one top-level directory named
        # after the FILE ("ultralytics_baseline/data.pkl"): that is a filesystem name, not content.
        tops = {n.split("/", 1)[0] for n in names}
        strip = len(tops) == 1 and all("/" in n for n in names)
        for info in infos:
            rel = info.filename.split("/", 1)[1] if strip else info.filename
            where = f"{label}:{rel}"
            for comp in rel.lower().replace("\\", "/").split("/"):
                if comp in _AGPL_PACKAGES:  # TorchScript code/__torch__/ultralytics/nn/tasks.py
                    _refuse(where, "holds TorchScript code of that package")
            if info.is_dir() or info.file_size > _SCAN_ENTRY_MAX:
                continue
            base = rel.rsplit("/", 1)[-1].lower()
            if base.endswith(".pkl") and not base.endswith(".debug_pkl"):
                _check_pickle(z.read(info), where)
            elif rel.startswith(("code/", "extra/")) or base.endswith((".py", ".json", ".yaml", ".yml", ".txt")) \
                    or base in _TEXT_NAMES:
                data = z.read(info)
                _check_text(data.decode("utf-8", "replace"), where)


def _scan_file(f: Path, label: str) -> None:
    import zipfile
    size = f.stat().st_size
    with open(f, "rb") as fh:
        head = fh.read(_SCAN_HEAD)
        tail = b""
        if size > 2 * _SCAN_HEAD:
            fh.seek(size - _SCAN_HEAD)
            tail = fh.read()
        elif size > _SCAN_HEAD:
            tail = fh.read()
    base = f.name.lower()
    if base.endswith((".md", ".markdown")):
        return  # a model card is prose ("unlike YOLO (AGPL-3.0) ..."); its license: is read by card_license
    _check_bytes(head + tail, label)
    if base in _LICENSE_NAMES:
        m = _COPYLEFT_RE.search((head + tail).decode("utf-8", "replace"))
        if m:
            raise ConvertError(f"{label} is a {m.group(0)!r} license text: only Apache-2.0 / MIT / BSD models can "
                               "be served by VisionServe (CLAUDE.md rule 1).")
        return
    if base in _TEXT_NAMES or base.endswith((".yaml", ".yml")):
        _check_text((head + tail).decode("utf-8", "replace"), label)
    if head[:1] == b"\x80":  # a (legacy torch.save) pickle stream
        _check_pickle(head, label)
    if zipfile.is_zipfile(f):  # torch.save / TorchScript / .keras, or a TFLite with appended metadata zip
        try:
            _scan_zip(f, label)
        except zipfile.BadZipFile:
            pass


def license_scan(source=None, module=None) -> None:
    """THE AGPL gate on model CONTENT, for every format (the --license / model-card gate is
    resolve_license). Refuses Ultralytics / YOLO-World / FastSAM models — AGPL whatever license the
    user declares — found through:

      module  an in-memory torch.nn.Module (Python API, a --script's build_model(), a loaded
              TorchScript module): the defining module of every submodule class (and its bases),
              and TorchScript qualified names (`__torch__.ultralytics.nn.tasks.DetectionModel`);
      source  a checkpoint file or directory: pickle class paths and dotted names in pickled strings
              (read from the opcodes, not by grepping bytes), every entry of a zip archive
              (TorchScript code/ and extra/ files, *.pkl, metadata json/yaml — wherever they sit in
              the archive), Ultralytics export metadata (metadata.yaml in a SavedModel, metadata.json
              appended to a TFLite file, config.txt in TorchScript), and a LICENSE file next to the
              weights. A filesystem path that merely contains "ultralytics" is not a marker."""
    if module is not None:
        _scan_module(module)
    if source is not None:
        p = Path(source)
        if p.is_file():
            _scan_file(p, p.name)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    _scan_file(f, str(f.relative_to(p)))


def _scan_module(module) -> None:
    mods = getattr(module, "modules", None)
    items = [module] + (list(mods()) if callable(mods) else [])
    for m in items:
        for cls in type(m).__mro__:
            _check_qualname(getattr(cls, "__module__", "") or "", f"in-memory module {type(m).__name__}")
        c = getattr(m, "_c", None)  # torch.jit.ScriptModule: the class it was scripted from
        if c is not None:
            try:
                qn = c._type().qualified_name()
            except Exception:  # noqa: BLE001
                qn = ""
            _check_qualname(qn, f"TorchScript module {getattr(m, 'original_name', '')}")


def refuse_agpl_pickle(path: Path) -> None:
    """Kept for callers of the old name: see license_scan."""
    license_scan(source=path)


# --------------------------------------------------------------------------------------------
# Bundle = one model directory to install
# --------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Bundle:
    name: str
    task: str                      # detection | segmentation | open_vocab | classification | depth | embed
    architecture: str              # a registered Go factory: rf-detr, rt-detr, efficientnet, midas, clip, ...
    license: str
    width: int
    height: int
    onnx: dict                     # manifest role -> local ONNX path ("model" for single-session)
    layout: str = "NCHW"
    letterbox: bool = False
    crop: Optional[str] = None     # "center": short-side resize + centre crop (manifest input.crop)
    # keep_aspect: the model's own keep-aspect resize (manifest input.keep_aspect; depth: DPT's rule,
    # sides rounded to multiple_of) instead of squashing to width x height. Needs dynamic H/W axes.
    keep_aspect: bool = False
    multiple_of: int = 0
    mean: Optional[Sequence[float]] = None
    std: Optional[Sequence[float]] = None
    postprocess: dict = dataclasses.field(default_factory=dict)
    labels: Optional[list] = None
    extra_files: dict = dataclasses.field(default_factory=dict)   # dest name -> source path
    prefer: Sequence[str] = ("cuda", "cpu")
    notes: list = dataclasses.field(default_factory=list)          # header comment lines
    single_file: bool = True       # model_file: vs files: map
    # The original framework pipeline for tiers B/C/speed (convert/reference.py). Not written
    # anywhere; it only keeps the in-process model alive until verification has run.
    reference: object = dataclasses.field(default=None, repr=False, compare=False)

    def write(self, out_dir: Path) -> Path:
        d = Path(out_dir) / validate_name(self.name)
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        for role, src in self.onnx.items():
            _copy_onnx(Path(src), d)
        for dest, src in self.extra_files.items():
            shutil.copy2(src, d / dest)
        if self.labels:
            (d / "labels.txt").write_text("\n".join(self.labels) + "\n")
        (d / "manifest.yaml").write_text(self.render_manifest())
        return d

    def render_manifest(self) -> str:
        L = [f"# Generated by `visionserve convert` — {line}" if i == 0 else f"# {line}"
             for i, line in enumerate(self.notes or ["converted checkpoint"])]
        L += [f"name: {self.name}", f"task: {self.task}", f"license: {self.license}",
              f"architecture: {self.architecture}"]
        digests = {role: sha256(p) for role, p in self.onnx.items()}
        if self.single_file and list(self.onnx) == ["model"]:
            L.append(f"sha256: {digests['model']}")
            L.append(f"model_file: {Path(self.onnx['model']).name}")
        else:
            L.append("sha256:")
            L += [f"  {r}: {d}" for r, d in sorted(digests.items())]
            L.append("")
            L.append("files:")
            L += [f"  {r}: {Path(p).name}" for r, p in sorted(self.onnx.items())]
        L += ["", "input:", f"  width: {self.width}", f"  height: {self.height}",
              f"  layout: {self.layout}", f"  letterbox: {'true' if self.letterbox else 'false'}"]
        if self.crop:
            L.append(f"  crop: {self.crop}")
        if self.keep_aspect:
            L.append("  keep_aspect: true")
        if self.multiple_of:
            L.append(f"  multiple_of: {int(self.multiple_of)}")
        if self.mean is not None and self.std is not None:
            L += ["  normalize:", f"    mean: {_flist(self.mean)}", f"    std: {_flist(self.std)}"]
        if self.postprocess:
            L += ["", "postprocess:"] + [f"  {k}: {v}" for k, v in self.postprocess.items()]
        if self.labels:
            L += ["", "labels: labels.txt"]
        L += ["", "runtime:", f"  prefer: [{', '.join(self.prefer)}]", "  idle_unload_seconds: 300", ""]
        return "\n".join(L)


def _copy_onnx(src: Path, dst_dir: Path) -> None:
    """Copy an ONNX file and any external-data sidecars it references (by name, same dir)."""
    import onnx
    shutil.copy2(src, dst_dir / src.name)
    m = onnx.load(str(src), load_external_data=False)
    locs = {e.value for t in m.graph.initializer for e in t.external_data if e.key == "location"}
    for loc in locs:
        # ORT resolves `location` relative to the model file: it must stay inside the model
        # directory, or the copy (and later the server) reads/writes outside the bundle.
        rel = Path(loc)
        if not loc or rel.is_absolute() or len(rel.parts) != 1 or rel.name in (".", "..") or "\\" in loc:
            raise ConvertError(f"{src.name}: external-data location {loc!r} escapes the model directory; "
                               "only plain file names next to the .onnx are accepted")
        shutil.copy2(src.parent / rel, dst_dir / rel)


def _flist(xs) -> str:
    return "[" + ", ".join(f"{float(x):g}" for x in xs) + "]"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------
# I/O contracts of the Go architectures (verify real shapes — CLAUDE.md "When unsure")
# --------------------------------------------------------------------------------------------

def onnx_io(path):
    """[(name, shape, elem_type)] for inputs and outputs; dynamic dims are strings."""
    import onnx
    m = onnx.load(str(path), load_external_data=False)
    init = {t.name for t in m.graph.initializer}

    def dims(v):
        return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
                for d in v.type.tensor_type.shape.dim]
    ins = [(v.name, dims(v), v.type.tensor_type.elem_type) for v in m.graph.input if v.name not in init]
    outs = [(v.name, dims(v), v.type.tensor_type.elem_type) for v in m.graph.output]
    return ins, outs


def _fixed(d, want) -> bool:
    return d == want or isinstance(d, str)


def check_contract(b: Bundle) -> None:
    """Refuse a bundle whose graph cannot be decoded by its Go architecture."""
    arch = b.architecture
    path = b.onnx.get("model") or next(iter(b.onnx.values()))
    ins, outs = onnx_io(path)
    n_labels = len(b.labels) if b.labels else None

    def image_input():
        if len(ins) != 1:
            raise ConvertError(f"{arch}: expected exactly 1 image input, graph has {[i[0] for i in ins]}")
        name, shp, _ = ins[0]
        if len(shp) != 4 or not _fixed(shp[1], 3):
            raise ConvertError(f"{arch}: input {name} is {shp}; VisionServe feeds NCHW [1,3,H,W] "
                               "(TensorFlow graphs are converted with inputs as NCHW)")

    if arch in ("rf-detr", "rt-detr"):
        image_input()
        # Same rule as the Go decoders: the FIRST output with last dim 4 is boxes, the first other
        # rank-3 output is logits (so a 3-class fine-tune, whose logits are also [1,Q,4], still works).
        box = [o for o in outs if len(o[1]) == 3 and o[1][-1] == 4][:1]
        cls = [o for o in outs if len(o[1]) == 3 and o not in box]
        if not box or not cls:
            raise ConvertError(f"{arch}: needs outputs boxes [1,Q,4] (cxcywh, normalised) and logits [1,Q,C]; "
                               f"graph emits {[(o[0], o[1]) for o in outs]}")
        c = cls[0][1][-1]
        if n_labels is not None and isinstance(c, int) and c != n_labels:
            raise ConvertError(f"{arch}: logits have {c} classes but {n_labels} labels were given "
                               "(RF-DETR fine-tunes carry an extra N/A row — labels must match the head exactly)")
    elif arch in ("efficientnet", "mobilenet-v3"):
        image_input()
        if len(outs) != 1 or len(outs[0][1]) != 2:
            raise ConvertError(f"classification: needs ONE output [1,C] logits; graph emits "
                               f"{[(o[0], o[1]) for o in outs]}")
        c = outs[0][1][1]
        if n_labels is not None and isinstance(c, int) and c != n_labels:
            raise ConvertError(f"classification: {c} logits but {n_labels} labels")
    elif arch in ("midas", "depth-anything-v2"):
        image_input()
        if len(outs) != 1 or len(outs[0][1]) != 3:
            raise ConvertError(f"depth: needs ONE output [1,H,W]; graph emits {[(o[0], o[1]) for o in outs]}")
        if b.keep_aspect and not all(isinstance(d, str) for d in ins[0][1][2:]):
            raise ConvertError(f"depth: keep_aspect feeds a different HxW per image, but input {ins[0][0]} is "
                               f"fixed at {ins[0][1]}; export it with dynamic height/width axes")
    elif arch in ("clip", "siglip-image"):
        image_input()
        if len(outs[0][1]) != 2:
            raise ConvertError(f"{arch}: needs an embedding output [N,D]; graph emits {[(o[0], o[1]) for o in outs]}")
    elif arch in TEXT_ARCHS:
        if len(ins) != 1 or ins[0][2] != 7:  # 7 = INT64
            raise ConvertError(f"{arch}: needs one int64 input_ids [N,L]; graph has {ins}")
    elif arch == "grounding-dino":
        names = {i[0] for i in ins}
        need = {"pixel_values", "input_ids", "attention_mask", "token_type_ids", "pixel_mask"}
        if not need <= names:
            raise ConvertError(f"grounding-dino: needs inputs {sorted(need)}; graph has {sorted(names)}")
    else:
        raise ConvertError(f"no I/O contract known for architecture {arch!r}")


# --------------------------------------------------------------------------------------------
# Parity: framework vs ONNX Runtime
# --------------------------------------------------------------------------------------------

# Every passing parity run of this process, for the report's tier A (cli resets it per conversion).
PARITY_RECORDS: list = []


def record_parity(what: str, err: float, tol: float, **extra) -> None:
    PARITY_RECORDS.append({"what": what.strip() or "parity", "max_rel_diff": float(err), "tol": float(tol), **extra})


def parity(onnx_path, feeds: dict, reference: Sequence[np.ndarray], tol: float, what: str = "") -> float:
    """Run ORT on `feeds` and compare every output with `reference` (same order). Returns the max
    absolute difference relative to the reference's scale; raises when it exceeds `tol`. An export
    that silently changed the math must never reach the registry."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, feeds)
    if len(got) < len(reference):
        raise ConvertError(f"parity{what}: ONNX graph returns {len(got)} outputs, reference has {len(reference)}")
    worst = 0.0
    for i, (g, r) in enumerate(zip(got, reference)):
        g, r = np.asarray(g, np.float64), np.asarray(r, np.float64)
        if g.shape != r.shape:
            raise ConvertError(f"parity{what}: output {i} shape {g.shape} != reference {r.shape}")
        # Non-finite values must agree EXACTLY (same NaN/±inf positions and signs) and are then
        # excluded; otherwise max() of a NaN difference is NaN, and `NaN > tol` is False — a graph
        # emitting NaN would pass. A NaN the reference does not have is always a failure.
        fin_g, fin_r = np.isfinite(g), np.isfinite(r)
        if (not np.array_equal(fin_g, fin_r) or np.isnan(g).any() != np.isnan(r).any()
                or not np.array_equal(g[~fin_g], r[~fin_r], equal_nan=True)):
            raise ConvertError(f"parity{what}: output {i} has NaN/inf where the original does not "
                               f"({int((~fin_g).sum())} non-finite values vs {int((~fin_r).sum())})")
        gf, rf = g[fin_r], r[fin_r]
        scale = max(1.0, float(np.abs(rf).max()) if rf.size else 1.0)
        err = float(np.abs(gf - rf).max()) / scale if rf.size else 0.0
        log(f"  parity{what}: output {i} {tuple(g.shape)} max|Δ|/scale = {err:.2e}")
        worst = max(worst, err)
    if worst > tol:
        raise ConvertError(f"parity{what}: ONNX differs from the original by {worst:.2e} (> tolerance {tol:g}). "
                           "Not installed. Re-export with a different opset, or raise --tolerance only if you "
                           "understand why the outputs differ.")
    record_parity(what, worst, tol)
    return worst


def sample_image(width: int, height: int, seed: int = 0) -> np.ndarray:
    """A deterministic, non-constant RGB uint8 image [H,W,3] for parity runs (smooth gradients +
    noise, so both flat and textured regions are exercised)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    img = np.stack([(x * 255 // max(1, width - 1)), (y * 255 // max(1, height - 1)),
                    ((x + y) * 127 // max(1, width + height - 2))], -1).astype(np.float32)
    img += rng.normal(0, 20, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def to_nchw(img_u8: np.ndarray, mean, std) -> np.ndarray:
    """Exactly what the Go preprocess feeds: /255, (x-mean)/std, CHW, batch 1, float32."""
    x = img_u8.astype(np.float32) / 255.0
    if mean is not None:
        x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]).astype(np.float32)


# --------------------------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------------------------

def install(bundles: Sequence[Bundle], staging: Path, models_dir: Path, force: bool, dry_run: bool) -> None:
    for b in bundles:
        check_contract(b)
    dirs = [b.write(staging) for b in bundles]
    for d in dirs:
        log(f"--- {d.name}/manifest.yaml ---\n{(d / 'manifest.yaml').read_text()}")
    if dry_run:
        log(f"dry run: bundles left in {staging}, nothing installed")
        return
    for d in dirs:
        # The Go binary validates exactly as `visionserve pull <folder>` does (license allowlist,
        # registered architecture, every referenced file present) and copies into the registry.
        cmd = [VISIONSERVE_BIN, "pull", str(d), "--models", str(models_dir)] + (["--force"] if force else [])
        log("$ " + " ".join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise ConvertError(f"visionserve refused {d.name} (see above); nothing was installed for it")
    log(f"done: {', '.join(b.name for b in bundles)} installed in {models_dir}. "
        "A running server picks new models up without a restart.")
