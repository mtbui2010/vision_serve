"""The engine behind :meth:`visionserve.Client.watch`: frames from a source, one ``predict`` each.

The server stays stateless: every frame is an ordinary ``POST /api/predict``. What this module
adds is the real-time plumbing around it:

* a reader thread pulls frames from the source as fast as it delivers them and keeps only the
  LATEST one (``drop_frames=True``): a frame that was not sent before the next one arrived is
  skipped, so results are never behind the camera by more than one request and memory stays
  bounded. With ``drop_frames=False`` the reader waits instead, and every frame is sent (offline
  video);
* request starts are at least ``1 / fps`` seconds apart (never a catch-up burst after a slow
  answer), and at most ``in_flight`` run at once; results come back in frame order;
* depth goes up only when the model reads it (``depth="auto"``: ``accepts_depth`` from
  ``GET /api/models``), converted to the server's depth wire format (see :func:`depth_for_upload`);
* an optional IoU tracker (:mod:`visionserve.track`) adds ``track_id`` to detections.
"""

from __future__ import annotations

import collections
import inspect
import logging
import math
import numbers
import threading
import time
from typing import Any, Dict, Iterator, Optional

from .sources import _open as _open_source
from .sources.frame import Frame, _np, as_frame
from .track import IoUTracker

log = logging.getLogger("visionserve.watch")

DEPTH_POLICIES = ("auto", "always", "never")
_READ_TIMEOUT_S = 0.25  # how long one source.read() may block before the reader checks for stop
_JOIN_TIMEOUT_S = 2.0
_RETRY_AFTER_MAX_S = 5.0
_MAX_IN_FLIGHT = 64
# predict() arguments watch() sets itself.
_RESERVED = ("model", "image", "depth")


def depth_for_upload(frame: Frame) -> Any:
    """``frame.depth`` in the form :meth:`Client.predict` sends (``depth=``):

    * ``uint16`` is sent unchanged as ``depth_dtype=uint16``. The server reads it as value / 65535
      with ``0`` = no reading, the convention of every RGB-D sensor, so no unit is lost that the
      server would use: the only model that reads depth (``background``) fits a plane to RELATIVE
      values. Half the bytes of float32.
    * ``float32`` / ``float64`` becomes ``float32`` METRES (× ``depth_scale`` when one is set),
      with NaN, ±inf and values <= 0 set to ``0`` (the server's "no reading"; ROS ``32FC1`` marks
      out-of-range pixels with NaN / +inf).
    """
    np = _np()
    d = frame.depth
    if d is None:
        return None
    if d.dtype == np.uint16:
        return np.ascontiguousarray(d)
    m = d.astype(np.float32, copy=True)
    if frame.depth_scale is not None:
        m *= np.float32(frame.depth_scale)
    m[~np.isfinite(m) | (m <= 0)] = 0.0
    return m


def _check_positive(name: str, v: Any, *, integer: bool = False) -> Any:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(float(v)) or v <= 0:
        raise ValueError("%s must be a positive number or None, got %r" % (name, v))
    if integer:
        if float(v) != int(v):
            raise ValueError("%s must be an integer, got %r" % (name, v))
        return int(v)
    return float(v)


def _takes_timeout(read: Any) -> bool:
    try:
        params = inspect.signature(read).parameters
    except (TypeError, ValueError):
        return False
    return "timeout" in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


class _Reader(threading.Thread):
    """Pulls frames from the source into a one-frame slot (see the module docstring)."""

    def __init__(self, source: Any, owned: bool, drop_frames: bool, cond: threading.Condition, stop: threading.Event):
        super().__init__(daemon=True, name="visionserve-watch-reader")
        self.source, self.owned, self.drop_frames = source, owned, drop_frames
        self.cond, self.stop = cond, stop
        self.latest: Optional[Frame] = None
        self.done = False
        self.error: Optional[BaseException] = None
        self.read_count = 0
        self.dropped = 0
        self._timeout_kw = _takes_timeout(source.read)

    def run(self) -> None:
        try:
            while not self.stop.is_set():
                try:
                    item = self.source.read(timeout=_READ_TIMEOUT_S) if self._timeout_kw else self.source.read()
                except TimeoutError:
                    continue
                if item is None:
                    break
                frame = as_frame(item).check()
                if frame.frame_id is None or frame.frame_id < 0:
                    frame.frame_id = self.read_count
                self.read_count += 1
                with self.cond:
                    if self.drop_frames:
                        if self.latest is not None:
                            self.dropped += 1
                    else:
                        while self.latest is not None and not self.stop.is_set():
                            self.cond.wait(0.1)
                    self.latest = frame
                    self.cond.notify_all()
        except BaseException as e:  # noqa: BLE001 - re-raised in the consumer
            self.error = e
        finally:
            with self.cond:
                self.done = True
                self.cond.notify_all()
            if self.owned:
                try:
                    self.source.close()
                except Exception as e:  # noqa: BLE001
                    log.warning("watch: closing the source failed: %s", e)

    def take(self) -> Optional[Frame]:
        """The latest frame, removing it from the slot (call with ``cond`` held)."""
        f, self.latest = self.latest, None
        if f is not None:
            self.cond.notify_all()  # a waiting reader (drop_frames=False) may continue
        return f


class _Job(threading.Thread):
    """One predict() request on a daemon thread: an abandoned request never blocks exit."""

    def __init__(self, fn, frame: Frame, cond: threading.Condition):
        super().__init__(daemon=True, name="visionserve-watch-request")
        self.fn, self.frame, self.cond = fn, frame, cond
        self.result: Any = None
        self.error: Optional[BaseException] = None
        self.finished = False

    def run(self) -> None:
        try:
            self.result = self.fn(self.frame)
        except BaseException as e:  # noqa: BLE001 - re-raised in the consumer, in order
            self.error = e
        finally:
            with self.cond:
                self.finished = True
                self.cond.notify_all()


def check_args(*, fps, max_frames, duration, depth, in_flight, track, predict_kwargs) -> Dict[str, Any]:
    """Validate :meth:`Client.watch` arguments eagerly (before the generator starts)."""
    if depth not in DEPTH_POLICIES:
        raise ValueError("depth must be one of %s, got %r" % (", ".join(DEPTH_POLICIES), depth))
    bad = sorted(k for k in predict_kwargs if k in _RESERVED)
    if bad:
        raise TypeError("watch() sets %s itself (the frame's colour and depth)" % ", ".join(bad))
    n = _check_positive("in_flight", in_flight, integer=True)
    if n is None or n > _MAX_IN_FLIGHT:
        raise ValueError("in_flight must be an integer in 1..%d, got %r" % (_MAX_IN_FLIGHT, in_flight))
    if track is True:
        tracker = IoUTracker()
    elif track is False or track is None:
        tracker = None
    elif hasattr(track, "update"):
        tracker = track
    else:
        raise TypeError("track must be True/False or a tracker with update(detections), got %r" % (track,))
    return {
        "fps": _check_positive("fps", fps),
        "max_frames": _check_positive("max_frames", max_frames, integer=True),
        "duration": _check_positive("duration", duration),
        "in_flight": n,
        "tracker": tracker,
    }


def run(client: Any, source: Any, model: str, *, fps: Optional[float], max_frames: Optional[int],
        duration: Optional[float], depth: str, in_flight: int, return_frames: bool, tracker: Any,
        drop_frames: bool, predict_kwargs: Dict[str, Any]) -> Iterator[Any]:
    """The generator of :meth:`Client.watch` (arguments already checked by :func:`check_args`)."""
    from .client import VisionServeError

    src, owned = _open_source(source)
    cond = threading.Condition()
    stop = threading.Event()
    reader = _Reader(src, owned, drop_frames, cond, stop)
    period = 1.0 / fps if fps else 0.0
    state: Dict[str, Any] = {"accepts": None, "resolved": False}

    def depth_arg(frame: Frame) -> Any:
        if depth == "never":
            return None
        if frame.depth is None:
            if depth == "always":
                raise ValueError("depth='always' but the source gave a frame without depth "
                                 "(frame %s); use depth='auto' or a depth camera" % frame.frame_id)
            return None
        if depth == "auto":
            if not state["resolved"]:
                state["accepts"] = client.accepts_depth(model)
                state["resolved"] = True
                if not state["accepts"]:
                    log.info("watch: %s does not read depth (accepts_depth=%s); sending colour only",
                             model, state["accepts"])
            if not state["accepts"]:
                return None
        return depth_for_upload(frame)

    def predict(frame: Frame) -> Any:
        kw = dict(predict_kwargs)
        d = depth_arg(frame)
        if d is not None:
            kw["depth"] = d  # predict() then sends the colour at full resolution (aligned to it)
        try:
            return client.predict(model, frame.color, **kw)
        except VisionServeError as e:
            if e.status != 503 or stop.is_set():
                raise
            # The model's queue is full: wait as told (bounded) and try once more.
            time.sleep(min(e.retry_after if e.retry_after is not None else 0.5, _RETRY_AFTER_MAX_S))
            return client.predict(model, frame.color, **kw)

    jobs: "collections.deque[_Job]" = collections.deque()
    sent = 0
    t_start = time.monotonic()
    next_t = t_start
    reader.start()
    try:
        while True:
            with cond:
                while True:
                    if jobs and jobs[0].finished:
                        break
                    if reader.error is not None:
                        break
                    now = time.monotonic()
                    budget_left = (max_frames is None or sent < max_frames) and \
                        (duration is None or now - t_start < duration)
                    source_left = reader.latest is not None or not reader.done
                    dispatching = budget_left and source_left
                    if not dispatching and not jobs:
                        return
                    if dispatching and len(jobs) < in_flight and now >= next_t and reader.latest is not None:
                        break
                    wait = 0.1
                    if dispatching and now < next_t:
                        wait = min(wait, next_t - now)
                    if duration is not None and budget_left:
                        wait = min(wait, max(0.0, t_start + duration - now))
                    cond.wait(max(wait, 0.001))
                if reader.error is not None:
                    raise reader.error
                frame = None
                if not (jobs and jobs[0].finished):
                    frame = reader.take()
            if jobs and jobs[0].finished:
                job = jobs.popleft()
                if job.error is not None:
                    raise job.error
                res = job.result
                res.frame_id = job.frame.frame_id
                res.timestamp = job.frame.timestamp
                if return_frames:
                    res.frame = job.frame
                if tracker is not None:
                    tracker.update(res.detections)
                yield res
                continue
            if frame is not None:
                now = time.monotonic()
                if period:
                    next_t = now + period  # request starts are at least 1/fps apart
                job = _Job(predict, frame, cond)
                jobs.append(job)
                sent += 1
                job.start()
    finally:
        stop.set()
        with cond:
            cond.notify_all()
        if reader.is_alive():
            reader.join(_JOIN_TIMEOUT_S)
            if reader.is_alive():
                log.warning("watch: the source is still blocked in read(); it is closed when that returns")
        # In-flight requests run on daemon threads; their answers are discarded.
