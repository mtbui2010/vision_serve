"""The robot / grasp helpers are implemented twice (this SDK and the JS one): both run the shared
cases in clients/testdata/postprocess_sync.json (the JS side in clients/js/tests/postprocess.test.ts).
The file is generated from this SDK by clients/testdata/gen_postprocess_sync.py; this test checks
that the Python code still gives what the file says, so a change to the maths shows up here and
the file (and the JS port) must be updated with it."""
import importlib.util
import json
import math
from pathlib import Path

import pytest

pytest.importorskip("numpy")

TESTDATA = Path(__file__).resolve().parents[2] / "testdata"


def _close(got, want, path="$"):
    if isinstance(want, float) and isinstance(got, (int, float)):
        assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12), (path, got, want)
    elif isinstance(want, dict):
        assert isinstance(got, dict) and set(got) == set(want), (path, got, want)
        for k in want:
            _close(got[k], want[k], "%s.%s" % (path, k))
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), (path, got, want)
        for i, (g, w) in enumerate(zip(got, want)):
            _close(g, w, "%s[%d]" % (path, i))
    else:
        assert got == want, (path, got, want)


def test_python_matches_the_shared_postprocess_cases():
    spec = importlib.util.spec_from_file_location("gen_postprocess_sync", TESTDATA / "gen_postprocess_sync.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    want = json.loads((TESTDATA / "postprocess_sync.json").read_text())
    _close(json.loads(json.dumps(gen.build())), want)
