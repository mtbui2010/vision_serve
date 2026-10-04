"""The Python and JS SDKs send the same prompt for the same (model, prompt).

normalize_prompt (here) and normalizePrompt (clients/js/src/client.ts) implement one rule; both
run the shared cases in clients/testdata/normalize_prompt.json (the JS side in
clients/js/tests/client.test.ts), so a change to one SDK that the other does not follow fails.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from visionserve.client import normalize_prompt

FIXTURES = Path(__file__).resolve().parents[2] / "testdata" / "normalize_prompt.json"

if not FIXTURES.is_file():  # an installed copy of the SDK without the repository around it
    pytest.skip("clients/testdata is not next to this checkout", allow_module_level=True)

CASES = json.loads(FIXTURES.read_text(encoding="utf-8"))["cases"]


def test_fixture_set_is_not_trivial():
    assert len(CASES) >= 20
    assert any(c["want"] == "object." for c in CASES)
    assert any(c["want"] is None for c in CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda c: "%s|%r" % (c["model"], c["prompt"]))
def test_normalize_prompt_matches_shared_fixtures(case):
    assert normalize_prompt(case["model"], case["prompt"]) == case["want"]
