"""Latency percentile computation backed by HdrHistogram.

HdrHistogram is used (not a plain numpy percentile of a truncated sample) because the
revision plan requires coordinated-omission-correct tail latency over >=1000 requests.
``hdrhistogram`` records values in microseconds with bounded relative error, so p99/p99.9
are trustworthy even with millions of samples.

If ``hdrhistogram`` is not installed we fall back to numpy on the raw sample and emit a
``backend`` marker so the consumer knows the percentiles are *not* HDR-recorded. We never
silently pretend; the fallback is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

try:  # pragma: no cover - import guard
    from hdrh.histogram import HdrHistogram  # type: ignore

    _HAVE_HDR = True
except Exception:  # noqa: BLE001
    _HAVE_HDR = False

import numpy as np

# Percentiles reported everywhere in the harness (matches REVISION-PLAN W4: p50/p95/p99/p99.9).
REPORT_PERCENTILES = (50.0, 95.0, 99.0, 99.9)


@dataclass
class LatencySummary:
    """Percentile summary in milliseconds plus count and backend marker."""

    count: int
    backend: str  # "hdrhistogram" or "numpy-fallback"
    percentiles_ms: dict[str, float] = field(default_factory=dict)
    min_ms: Optional[float] = None
    max_ms: Optional[float] = None
    mean_ms: Optional[float] = None

    def as_row(self) -> dict[str, object]:
        """Flatten to a CSV-friendly row."""
        row: dict[str, object] = {
            "count": self.count,
            "backend": self.backend,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "mean_ms": self.mean_ms,
        }
        row.update({f"p{p}_ms": v for p, v in self.percentiles_ms.items()})
        return row


class LatencyRecorder:
    """Accumulates latency samples (in milliseconds) and computes percentiles.

    Internally HdrHistogram records microseconds (integer); we keep a parallel raw list only
    for min/max/mean reporting and the numpy fallback path.
    """

    def __init__(
        self,
        lowest_us: int = 1,
        highest_us: int = 60 * 1_000_000,  # 60 s ceiling
        sig_figs: int = 3,
    ) -> None:
        self._raw_us: list[float] = []
        if _HAVE_HDR:
            self._hist = HdrHistogram(lowest_us, highest_us, sig_figs)
        else:
            self._hist = None

    def record_ms(self, latency_ms: float) -> None:
        """Record one latency sample given in milliseconds."""
        us = latency_ms * 1000.0
        self._raw_us.append(us)
        if self._hist is not None:
            # HdrHistogram stores integers; round to nearest microsecond.
            self._hist.record_value(int(round(us)))

    def record_many_ms(self, values: Iterable[float]) -> None:
        for v in values:
            self.record_ms(v)

    @property
    def count(self) -> int:
        return len(self._raw_us)

    def summary(self) -> LatencySummary:
        """Return the percentile summary (ms). Empty recorder -> zeroed summary."""
        if not self._raw_us:
            return LatencySummary(count=0, backend="empty")
        arr_ms = np.array(self._raw_us, dtype=float) / 1000.0
        base = dict(
            count=len(self._raw_us),
            min_ms=float(arr_ms.min()),
            max_ms=float(arr_ms.max()),
            mean_ms=float(arr_ms.mean()),
        )
        if self._hist is not None:
            pct = {
                str(p): self._hist.get_value_at_percentile(p) / 1000.0
                for p in REPORT_PERCENTILES
            }
            return LatencySummary(backend="hdrhistogram", percentiles_ms=pct, **base)
        pct = {
            str(p): float(np.percentile(arr_ms, p)) for p in REPORT_PERCENTILES
        }
        return LatencySummary(backend="numpy-fallback", percentiles_ms=pct, **base)
