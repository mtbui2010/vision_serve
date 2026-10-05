"""The client-resize rules are implemented twice (this SDK and the JS one): both run the shared
cases in clients/testdata/client_resize.json (the JS side in clients/js/tests/resize.test.ts),
so they send the same size and see the same ROI region for an image."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visionserve.resize import roi_region, target_size  # noqa: E402

FIXTURES = json.loads((Path(__file__).resolve().parents[2] / "testdata" / "client_resize.json").read_text())


def test_target_size_shared_cases():
    for c in FIXTURES["target_size"]:
        got = target_size(c["width"], c["height"], max_side=c.get("max_side"),
                          max_short_side=c.get("max_short_side"),
                          region=tuple(c["region"]) if c.get("region") else None)
        assert list(got) == c["want"], c


def test_roi_region_shared_cases():
    for c in FIXTURES["roi_region"]:
        got = roi_region(c["roi"], c["width"], c["height"])
        assert (list(got) if got is not None else None) == c["want"], c


def test_is_loopback_shared_cases():
    from visionserve.client import _is_loopback

    for c in FIXTURES["is_loopback"]:
        assert _is_loopback(c["host"]) is c["want"], c
