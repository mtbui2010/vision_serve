"""convert/constants.py is the one copy of the names several converter modules share. These tests
fail when a module grows its own copy again (by name or by an equal literal)."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # clients/python

from visionserve.convert import api, bench, cli, common, constants, evaluate, verify  # noqa: E402

CONVERT = Path(constants.__file__).parent
SHARED_NAMES = {"IMAGE_EXT", "_HUB_ID", "HUB_ID_RE", "TEXT_ARCHS", "OPEN_VOCAB_ARCHS"}
SHARED_FUNCS = {"parse_wxh", "_parse_wxh"}


def _modules():
    return sorted(p for p in CONVERT.rglob("*.py") if p.name != "constants.py")


def _literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None


def test_no_module_redefines_a_shared_name():
    for py in _modules():
        for node in ast.walk(ast.parse(py.read_text(), str(py))):
            if isinstance(node, ast.Assign):
                names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = {node.target.id}
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.name not in SHARED_FUNCS, f"{py.name} defines its own {node.name}; import constants.parse_wxh"
                continue
            else:
                continue
            assert not names & SHARED_NAMES, f"{py.name} redefines {names & SHARED_NAMES}; import it from constants"


def test_no_module_repeats_a_shared_literal():
    shared = {
        "IMAGE_EXT": set(constants.IMAGE_EXT),
        "TEXT_ARCHS": set(constants.TEXT_ARCHS),
        "OPEN_VOCAB_ARCHS": set(constants.OPEN_VOCAB_ARCHS),
    }
    for py in _modules():
        for node in ast.walk(ast.parse(py.read_text(), str(py))):
            if isinstance(node, (ast.Tuple, ast.List, ast.Set)) and len(node.elts) >= 2:
                v = _literal(node)
                if v is None:
                    continue
                try:
                    v = set(v)
                except TypeError:
                    continue
                for name, want in shared.items():
                    assert v != want, f"{py.name}:{node.lineno} repeats constants.{name}; import it instead"
            elif isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "compile" and node.args:
                pat = _literal(node.args[0])
                assert pat != constants.HUB_ID_RE.pattern, f"{py.name}:{node.lineno} repeats constants.HUB_ID_RE"


def test_modules_use_the_shared_objects():
    assert verify.IMAGE_EXT is constants.IMAGE_EXT
    assert evaluate.IMAGE_EXT is constants.IMAGE_EXT
    assert verify.TEXT_ARCHS is bench.TEXT_ARCHS is common.TEXT_ARCHS is constants.TEXT_ARCHS
    assert api.HUB_ID_RE is constants.HUB_ID_RE
    assert cli.parse_wxh is constants.parse_wxh


def test_values_unchanged():
    # The values the modules held before they were moved here (behaviour must not change).
    assert constants.IMAGE_EXT == {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    assert constants.TEXT_ARCHS == ("siglip-text", "clip-text")
    assert constants.OPEN_VOCAB_ARCHS == ("grounding-dino", "grounded-sam", "gdino-siglip", "rfdetr-gdino")
    assert constants.HUB_ID_RE.match("google/siglip-base-patch16-224")
    assert not constants.HUB_ID_RE.match("../etc/passwd") and not constants.HUB_ID_RE.match("noslash")
    assert constants.parse_wxh("640X480") == (640, 480)
    for bad in ("640", "axb", "1x2x3"):
        with pytest.raises(common.ConvertError, match="224x224"):
            constants.parse_wxh(bad)
