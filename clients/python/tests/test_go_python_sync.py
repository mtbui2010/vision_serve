"""The converter's rules must be the Go registry's rules.

The license allowlist and the model-name rule exist twice: in the Go registry
(internal/registry/manifest.go, the last word at load and at `visionserve pull <folder>`) and here
(visionserve/convert/common.py, which refuses before a multi-GB export rather than after). These
tests fail when the two drift:

  * the Go source is parsed and its allowlist / name pattern / length limit must equal ours;
  * both sides run the shared corpus internal/registry/testdata/rules_sync.json through their own
    checks and must give its answers (identical pattern text can still behave differently:
    Python's `$` also matches before a trailing newline, Go's does not);
  * every architecture and task the converter can write into a manifest must exist in Go
    (models.Register names, registry validTasks) — or the converted model could never load.

internal/registry/sync_test.go is the mirror image on the Go side.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]  # repo root (clients/python/tests/ -> ../../..)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # clients/python

from visionserve.convert import cli, common, constants  # noqa: E402

GO_MANIFEST = ROOT / "internal" / "registry" / "manifest.go"
CORPUS = ROOT / "internal" / "registry" / "testdata" / "rules_sync.json"
CONVERT = Path(common.__file__).parent

if not (ROOT / "internal").is_dir():  # an installed copy of the SDK without the Go tree
    pytest.skip("the Go sources are not next to this checkout", allow_module_level=True)


def _go(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _corpus() -> dict:
    return json.loads(CORPUS.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------
# literal comparison with the Go source
# --------------------------------------------------------------------------------------------

def test_license_allowlist_matches_go():
    src = _go(GO_MANIFEST)
    m = re.search(r"\nvar licenseAllowlist = map\[string\]string\{(.*?)\n\}", src, re.S)
    assert m, f"could not find `var licenseAllowlist = map[string]string{{...}}` in {GO_MANIFEST}"
    body = m.group(1)
    pairs = dict(re.findall(r'"((?:[^"\\]|\\.)*)"\s*:\s*"((?:[^"\\]|\\.)*)"', body))
    entries = [ln for ln in (x.strip() for x in body.splitlines()) if ln and not ln.startswith("//")]
    assert len(pairs) == len(entries), f"unparsed lines in licenseAllowlist: {entries}"
    assert pairs == common.LICENSE_ALLOWLIST


def test_name_rule_matches_go():
    src = _go(GO_MANIFEST)
    m = re.search(r"\bvalidModelName\s*=\s*regexp\.MustCompile\(`([^`]+)`\)", src)
    assert m, f"could not find validModelName in {GO_MANIFEST}"
    assert m.group(1) == common._NAME_RE.pattern
    lim = re.search(r"len\(m\.Name\)\s*>\s*(\d+)", src)
    assert lim, f"could not find the name length limit (len(m.Name) > N) in {GO_MANIFEST}"
    assert int(lim.group(1)) == common.NAME_MAX_LEN == _corpus()["name_max_len"]


# --------------------------------------------------------------------------------------------
# behaviour on the shared corpus
# --------------------------------------------------------------------------------------------

def test_license_corpus():
    for declared, want in _corpus()["licenses"].items():
        if want is None:
            with pytest.raises(common.ConvertError):
                common.canonical_license(declared)
        else:
            assert common.canonical_license(declared) == want, declared


def test_name_corpus():
    c = _corpus()
    cases = dict(c["names"])
    cases["a" * c["name_max_len"]] = True
    cases["a" * (c["name_max_len"] + 1)] = False
    for name, want in cases.items():
        try:
            common.validate_name(name)
            got = True
        except common.ConvertError:
            got = False
        assert got == want, f"name {name!r}: converter accepted={got}, the corpus (and Go) say {want}"


# --------------------------------------------------------------------------------------------
# architectures and tasks the converter writes must exist in Go
# --------------------------------------------------------------------------------------------

def _go_architectures() -> set:
    names = set()
    for f in (ROOT / "internal" / "models").rglob("*.go"):
        if not f.name.endswith("_test.go"):
            names |= set(re.findall(r'\bmodels\.Register\(\s*"([^"]+)"', _go(f)))
    assert len(names) > 10, "found almost no models.Register calls: did the Go layout change?"
    return names


def _go_tasks() -> set:
    api = "".join(_go(f) for f in (ROOT / "pkg" / "api").glob("*.go") if not f.name.endswith("_test.go"))
    consts = dict(re.findall(r"\b(Task\w+)\s+Task\s*=\s*\"([^\"]+)\"", api))
    m = re.search(r"\nvar validTasks = map\[api\.Task\]bool\{(.*?)\n\}", _go(GO_MANIFEST), re.S)
    assert m and consts, "could not find validTasks / the api.Task constants"
    keys = re.findall(r"api\.(Task\w+)\s*:\s*true", m.group(1))
    assert keys and all(k in consts for k in keys), keys
    return {consts[k] for k in keys}


def _bundle_literals():
    """(architecture, task) string literals of every Bundle(...) a converter module builds."""
    archs, tasks = set(), set()
    for py in CONVERT.rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"), str(py))):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Bundle":
                for kw in node.keywords:
                    if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        if kw.arg == "architecture":
                            archs.add(kw.value.value)
                        elif kw.arg == "task":
                            tasks.add(kw.value.value)
    return archs, tasks


def _contract_architectures() -> set:
    """Architecture names check_contract has an I/O contract for (the `arch in (...)` arms)."""
    fn = next(n for n in ast.walk(ast.parse(Path(common.__file__).read_text(encoding="utf-8")))
              if isinstance(n, ast.FunctionDef) and n.name == "check_contract")
    out = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Compare) and getattr(node.left, "id", None) == "arch":
            for comp in node.comparators:
                if isinstance(comp, ast.Constant) and isinstance(comp.value, str):
                    out.add(comp.value)
                elif isinstance(comp, (ast.Tuple, ast.List, ast.Set)):
                    out |= {e.value for e in comp.elts if isinstance(e, ast.Constant)}
                elif isinstance(comp, ast.Name) and hasattr(constants, comp.id):
                    out |= set(getattr(constants, comp.id))
    assert out, "found no architecture arms in check_contract"
    return out


def test_converter_architectures_are_registered_in_go():
    go = _go_architectures()
    archs, _ = _bundle_literals()
    archs |= {arch for arch, _task, _post in cli.TASK_PRESETS.values()}
    archs |= _contract_architectures() | set(constants.TEXT_ARCHS) | set(constants.OPEN_VOCAB_ARCHS)
    missing = sorted(archs - go)
    assert not missing, f"the converter knows architectures no Go package registers: {missing}"


def test_converter_tasks_are_valid_in_go():
    go = _go_tasks()
    _, tasks = _bundle_literals()
    tasks |= {task for _arch, task, _post in cli.TASK_PRESETS.values()}
    missing = sorted(tasks - go)
    assert not missing, f"the converter writes tasks the Go registry refuses: {missing}"


# --------------------------------------------------------------------------------------------
# preprocessing: one Spec semantics (internal/vision/preprocess <-> convert/spec.py)
# --------------------------------------------------------------------------------------------

PRE_CORPUS = ROOT / "internal" / "registry" / "testdata" / "preprocess_sync.json"
GEOMETRY_CORPUS = ROOT / "internal" / "vision" / "preprocess" / "testdata" / "geometry_sync.json"
GO_PREPROCESS = ROOT / "internal" / "vision" / "preprocess" / "spec.go"


def _canonical(s) -> dict:
    import numpy as np
    return {"resize": s.resize, "width": s.width, "height": s.height, "multiple_of": s.multiple_of,
            "no_upscale": s.no_upscale, "crop_pct": float(np.float32(s.crop_pct)), "resample": s.resample,
            "mean": [float(np.float32(v)) for v in (s.mean or [])],
            "std": [float(np.float32(v)) for v in (s.std or [])],
            "rescale": s.rescale, "layout": s.layout, "pad": float(np.float32(s.pad)), "legacy": s.legacy}


def test_preprocess_modes_match_go():
    from visionserve.convert import spec
    go_modes = re.findall(r'^\t(\w+)\s+Mode = "(\w+)"', _go(GO_PREPROCESS), re.M)
    assert go_modes, f"could not find the Mode constants in {GO_PREPROCESS}"
    assert tuple(v for _, v in go_modes) == spec.MODES


def test_preprocess_resolution_corpus():
    """registry.Manifest.PreprocessSpec and spec.spec_from_manifest resolve every manifest of the
    shared corpus the same way: legacy-only, block-only, both-consistent, both-conflicting."""
    import numpy as np
    from visionserve.convert import spec
    cases = json.loads(PRE_CORPUS.read_text(encoding="utf-8"))["cases"]
    assert len(cases) >= 30
    for c in cases:
        doc = {k: c[k] for k in ("input", "preprocess") if k in c}
        if "error" in c:
            with pytest.raises(spec.SpecError) as e:
                spec.spec_from_manifest(doc)
            for sub in c["error"]:
                assert sub in str(e.value), f"{c['name']}: {e.value!r} does not name {sub!r}"
            continue
        want = {"resize": "", "width": 0, "height": 0, "multiple_of": 0, "no_upscale": False, "crop_pct": 0.0,
                "resample": "", "mean": [], "std": [], "rescale": True, "layout": "", "pad": 0.0, "legacy": False}
        want.update(c["spec"])
        want["crop_pct"] = float(np.float32(want["crop_pct"]))
        want["mean"] = [float(np.float32(v)) for v in want["mean"]]
        want["std"] = [float(np.float32(v)) for v in want["std"]]
        assert _canonical(spec.spec_from_manifest(doc)) == want, c["name"]


def test_preprocess_geometry_corpus():
    """spec.apply_spec gives the tensor shape and Meta Go's Spec.Apply gives, for every mode."""
    import numpy as np
    from PIL import Image
    from visionserve.convert import spec
    cases = json.loads(GEOMETRY_CORPUS.read_text(encoding="utf-8"))["cases"]
    assert len(cases) >= 50
    for c in cases:
        s = spec.Spec(**c["spec"])
        w, h = c["image"]
        pil = Image.fromarray(np.random.default_rng(w * 7 + h).integers(0, 256, (h, w, 3), dtype=np.uint8))
        x, meta = spec.apply_spec(pil, s)
        assert list(x.shape) == c["shape"], (c["spec"], c["image"])
        m = c["meta"]
        assert meta == {"orig_width": m["OrigWidth"], "orig_height": m["OrigHeight"], "scale_x": m["ScaleX"],
                        "scale_y": m["ScaleY"], "pad_x": m["PadX"], "pad_y": m["PadY"]}, (c["spec"], c["image"])
        assert spec.spec_meta(s, w, h) == meta


ARCH_CORPUS = ROOT / "internal" / "vision" / "preprocess" / "testdata" / "arch_resolve_sync.json"


def test_arch_resolution_corpus():
    """spec.resolve_arch(spec_from_manifest(doc), arch) gives what the Go model resolves (models.New
    + ResolvedPreprocess, internal/vision/preprocess/arch_sync_test.go) for every row: legacy
    flags read each architecture's way, blocks with an unsupported mode refused, fixed-by-export
    architectures. `visionserve check`'s manifest reference is built this way."""
    import numpy as np
    from visionserve.convert import spec
    cases = json.loads(ARCH_CORPUS.read_text(encoding="utf-8"))["cases"]
    assert len(cases) >= 30
    for c in cases:
        doc = {k: c[k] for k in ("input", "preprocess") if k in c}
        if "error" in c:
            with pytest.raises(spec.SpecError) as e:
                spec.resolve_arch(spec.spec_from_manifest(doc), c["architecture"])
            for sub in c["error"]:
                assert sub in str(e.value), f"{c['name']}: {e.value!r} does not name {sub!r}"
            continue
        got = spec.resolve_arch(spec.spec_from_manifest(doc), c["architecture"])
        if c.get("fixed_by_export"):
            assert got is None, c["name"]
            continue
        want = {"resize": "", "width": 0, "height": 0, "multiple_of": 0, "no_upscale": False, "crop_pct": 0.0,
                "resample": "", "mean": [], "std": [], "rescale": True, "layout": "", "pad": 0.0, "legacy": False}
        want.update(c["spec"])
        want["crop_pct"] = float(np.float32(want["crop_pct"]))
        want["mean"] = [float(np.float32(v)) for v in want["mean"]]
        want["std"] = [float(np.float32(v)) for v in want["std"]]
        assert _canonical(got) == want, c["name"]
    assert {c["architecture"] for c in cases} >= set(spec.ARCHS) | set(spec.FIXED_BY_EXPORT)


def test_arch_modes_match_go():
    """spec.ARCHS names the modes each Go model package declares (`prep.Arch{Name: ..., Modes:
    ...}`), and spec.FIXED_BY_EXPORT the packages that call FixedByExport."""
    from visionserve.convert import spec
    consts = dict(re.findall(r'^\t(\w+)\s+Mode = "(\w+)"', _go(GO_PREPROCESS), re.M))
    models_dir = ROOT / "internal" / "models"
    go_archs = {}
    for f in models_dir.glob("*/*.go"):
        if f.name.endswith("_test.go"):
            continue
        for name, modes in re.findall(r'Arch\{Name:\s*"([\w-]+)",\s*Modes:\s*\[\]\w+\.Mode\{([^}]*)\}', _go(f)):
            go_archs[name] = tuple(consts[m.split(".")[-1].strip()] for m in modes.split(",") if m.strip())
    # detr builds its Arch per variant: Name is the variant prefix ("rfdetr" / "rtdetr").
    detr = _go(models_dir / "detr" / "preprocess.go")
    m = re.search(r"Arch\{Name:\s*prefix,\s*Modes:\s*\[\]\w+\.Mode\{([^}]*)\}", detr)
    assert m, "could not find detr's Arch"
    detr_modes = tuple(consts[x.split(".")[-1].strip()] for x in m.group(1).split(",") if x.strip())
    go_archs["rfdetr"] = go_archs["rtdetr"] = detr_modes
    for reg, a in spec.ARCHS.items():
        assert a.name in go_archs, f"{reg}: no Go Arch named {a.name!r}"
        assert a.modes == go_archs[a.name], f"{reg}: modes {a.modes} != Go {go_archs[a.name]}"
    fixed = set()
    for f in models_dir.glob("*/*.go"):
        if not f.name.endswith("_test.go"):
            fixed |= set(re.findall(r'FixedByExport\("([\w-]+)"', _go(f)))
    assert fixed == set(spec.FIXED_BY_EXPORT)
