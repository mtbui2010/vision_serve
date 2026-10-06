"""The drawing colours and sizes are implemented twice (draw() here, toSVG in the JS SDK): both run
clients/testdata/class_colors.json, generated from this SDK by clients/testdata/gen_class_colors.py.
This test checks that the Python code still gives what the file says."""
import importlib.util
import json
from pathlib import Path

TESTDATA = Path(__file__).resolve().parents[2] / "testdata"


def test_python_matches_the_shared_class_colours():
    spec = importlib.util.spec_from_file_location("gen_class_colors", TESTDATA / "gen_class_colors.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    want = json.loads((TESTDATA / "class_colors.json").read_text(encoding="utf-8"))
    assert json.loads(json.dumps(gen.build())) == want
