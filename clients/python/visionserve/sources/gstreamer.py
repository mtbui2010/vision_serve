"""GStreamer adapter: any ``gst-launch-1.0`` pipeline, run as a subprocess.

No Python binding (PyGObject) is needed: the adapter appends a raw-RGB tail to your pipeline,

    <your pipeline> ! videoconvert [! videoscale] ! video/x-raw,format=RGB[,width=W,height=H]
                    ! fdsink name=vs_sink fd=<pipe>

starts ``gst-launch-1.0 -v`` with it, learns the frame size from the caps the sink negotiated
(printed by ``-v``), and reads exact frames from a pipe. Use it for what GStreamer does best:
RTSP with hardware decoding (``nvv4l2decoder`` on Jetson, ``vaapih264dec``, ``v4l2h264dec``),
industrial cameras (``aravissrc`` for GigE Vision / USB3 Vision), ``v4l2src`` with special caps,
``nvarguscamerasrc`` (Jetson CSI cameras). GStreamer itself is a system install
(``apt install gstreamer1.0-tools gstreamer1.0-plugins-good ...``), not a pip extra.

POSIX only (Linux, macOS): the frames travel over an inherited pipe descriptor.
"""

from __future__ import annotations

import collections
import logging
import os
import re
import select
import shlex
import shutil
import subprocess
import threading
import time
from typing import List, Optional, Tuple

from .frame import BaseSource, Frame, _np

log = logging.getLogger("visionserve.sources")

SINK_NAME = "vs_sink"
# `gst-launch-1.0 -v` prints the negotiated caps of every pad; this is the line of our sink's pad.
_CAPS_RE = re.compile(r"GstFdSink:" + SINK_NAME + r"\.GstPad:sink: caps = (.*)$")
_INT_RE = r"%s=\(int\)(\d+)"


def parse_caps(line: str) -> Optional[Tuple[int, int, str]]:
    """``(width, height, format)`` from a ``gst-launch-1.0 -v`` caps line of our sink, else None."""
    m = _CAPS_RE.search(line)
    if not m:
        return None
    caps = m.group(1)
    w = re.search(_INT_RE % "width", caps)
    h = re.search(_INT_RE % "height", caps)
    f = re.search(r"format=\(string\)(\w+)", caps)
    if not (w and h):
        return None
    return int(w.group(1)), int(h.group(1)), (f.group(1) if f else "RGB")


def rgb_stride(width: int) -> int:
    """Bytes per row of a GStreamer ``video/x-raw,format=RGB`` buffer: rows are padded to a
    multiple of 4 bytes (a 65-pixel row is 196 bytes, not 195)."""
    return (width * 3 + 3) & ~3


def build_command(pipeline: str, fd: int, *, width: Optional[int] = None, height: Optional[int] = None,
                  sync: bool = False, gst_launch: str = "gst-launch-1.0") -> List[str]:
    """The ``gst-launch-1.0`` argv for ``pipeline`` with the raw-RGB tail writing to ``fd``."""
    if not pipeline or not pipeline.strip():
        raise ValueError("empty GStreamer pipeline")
    if (width is None) != (height is None):
        raise ValueError("give both width and height, or neither")
    tail = ["!", "videoconvert"]
    caps = "video/x-raw,format=RGB"
    if width is not None:
        if int(width) <= 0 or int(height) <= 0:  # type: ignore[arg-type]
            raise ValueError("width and height must be positive, got %r x %r" % (width, height))
        tail += ["!", "videoscale"]
        caps += ",width=%d,height=%d" % (int(width), int(height))  # type: ignore[arg-type]
    tail += ["!", caps, "!", "fdsink", "name=" + SINK_NAME, "fd=%d" % fd, "sync=" + ("true" if sync else "false")]
    return [gst_launch, "-v"] + shlex.split(pipeline.strip().rstrip("!").strip()) + tail


class GStreamerSource(BaseSource):
    """Frames from a ``gst-launch-1.0`` pipeline (see the module docstring).

    Args:
        pipeline: the pipeline up to the decoded video, as you would type it after
            ``gst-launch-1.0`` (without a sink), e.g. ``"rtspsrc location=rtsp://cam/stream
            latency=0 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv"``. Its last element
            must output system-memory video (on Jetson end with ``nvvidconv``, which copies NVMM
            buffers out); the adapter converts it to RGB.
        width, height: scale every frame to this size in GStreamer (``videoscale``). Without
            them the source's own size is used, read from the negotiated caps.
        sync: ``True`` makes the sink play at the stream's clock (pace a FILE like a camera);
            ``False`` (default) takes frames as fast as they decode.
        restart: when the pipeline ends or dies, start it again after ``restart_delay`` seconds
            (a dropped RTSP connection); ``False`` (default): the stream ends.
        startup_timeout: seconds to wait for the first caps before giving up with the
            pipeline's error output.
        gst_launch: the ``gst-launch-1.0`` executable.

    The pipeline's stderr (warnings, errors) is logged to the ``visionserve.sources`` logger; a
    pipeline that fails before its first frame raises ``RuntimeError`` with its last lines.
    ``timestamp`` is ``time.time()`` when a frame was read.
    """

    def __init__(
        self,
        pipeline: str,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        sync: bool = False,
        restart: bool = False,
        restart_delay: float = 1.0,
        startup_timeout: float = 15.0,
        gst_launch: str = "gst-launch-1.0",
    ):
        super().__init__()
        if os.name != "posix":
            raise OSError("GStreamerSource needs a POSIX system (Linux, macOS)")
        exe = shutil.which(gst_launch)
        if exe is None:
            raise FileNotFoundError(
                "%s not found: install GStreamer (e.g. apt install gstreamer1.0-tools "
                "gstreamer1.0-plugins-base gstreamer1.0-plugins-good)" % gst_launch
            )
        build_command(pipeline, 3, width=width, height=height)  # validate the arguments now
        self.pipeline = pipeline
        self._exe = exe
        self._size = (width, height)
        self._sync = bool(sync)
        self.restart = bool(restart)
        self.restart_delay = float(restart_delay)
        self.startup_timeout = float(startup_timeout)
        self._proc: Optional[subprocess.Popen] = None
        self._rfd = -1
        self._buf = bytearray()
        self._caps: Optional[Tuple[int, int, str]] = None
        self._caps_ready = threading.Event()
        self._tail: "collections.deque[str]" = collections.deque(maxlen=30)
        self._threads: List[threading.Thread] = []
        self._frames_this_run = 0
        self._started_at = 0.0
        self._exit_code: Optional[int] = None
        self._lock = threading.Lock()
        self._start()

    # -- process management ---------------------------------------------------------------- #
    def _start(self) -> None:
        rfd, wfd = os.pipe()
        cmd = build_command(self.pipeline, wfd, width=self._size[0], height=self._size[1],
                            sync=self._sync, gst_launch=self._exe)
        log.debug("GStreamerSource: %s", " ".join(shlex.quote(c) for c in cmd))
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, pass_fds=(wfd,), close_fds=True)
        finally:
            os.close(wfd)  # the child holds its own copy: EOF on rfd when it exits
        self._proc, self._rfd = proc, rfd
        self._buf = bytearray()
        self._caps = None
        self._caps_ready.clear()
        self._frames_this_run = 0
        self._started_at = time.monotonic()
        self._threads = [
            threading.Thread(target=self._pump, args=(proc.stdout, False), daemon=True, name="vs-gst-stdout"),
            threading.Thread(target=self._pump, args=(proc.stderr, True), daemon=True, name="vs-gst-stderr"),
        ]
        for t in self._threads:
            t.start()

    def _pump(self, stream, is_err: bool) -> None:
        """Read gst-launch's text output: caps lines set the frame size, the rest is logged."""
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if not line:
                continue
            caps = None if is_err else parse_caps(line)
            if caps is not None:
                if caps[2] != "RGB":
                    log.warning("GStreamerSource: sink negotiated %s, expected RGB", caps[2])
                if self._caps is None:
                    self._caps = caps
                    self._caps_ready.set()
                continue
            if is_err or line.startswith(("ERROR", "WARNING")):
                self._tail.append(line)
                # Before the first frame a failure is raised with these lines; after it, a
                # warning or error of a running pipeline is logged.
                (log.warning if self._frames_this_run else log.debug)("gst-launch: %s", line)
            else:
                log.debug("gst-launch: %s", line)
        stream.close()

    def _stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            self._exit_code = proc.returncode
        if self._rfd >= 0:
            os.close(self._rfd)
            self._rfd = -1
        for t in self._threads:
            t.join(timeout=2)
        self._threads = []

    def _error_text(self) -> str:
        tail = "\n  ".join(list(self._tail)[-10:]) or "(no error output)"
        return "GStreamer pipeline %r ended (exit code %s) before its first frame:\n  %s" % (
            self.pipeline, self._exit_code, tail)

    # -- reading ------------------------------------------------------------------------------ #
    def read(self, timeout: Optional[float] = None) -> Optional[Frame]:
        with self._lock:
            while True:
                if self._closed or self._proc is None:
                    return None
                status, data = self._read_frame(timeout)
                if status == "frame":
                    return data  # type: ignore[return-value]
                if status == "timeout":
                    raise TimeoutError("no frame from the GStreamer pipeline within %ss" % timeout)
                # "eof": the pipeline ended or died (or close() stopped it).
                if self._closed:
                    return None
                failed_at_start = self._frames_this_run == 0
                self._stop()  # waits for the process and the rest of its output
                if failed_at_start and (self._count == 0 or not self.restart):
                    raise RuntimeError(self._error_text())
                if not self.restart:
                    return None
                last = self._tail[-1] if self._tail else "no error output"
                log.warning("GStreamerSource: pipeline ended (%s); restarting in %.1fs", last, self.restart_delay)
                self._tail.clear()
                time.sleep(self.restart_delay)
                if self._closed:
                    return None
                self._start()

    def _read_frame(self, timeout: Optional[float]):
        deadline = None if timeout is None else time.monotonic() + timeout
        # The caps line comes before the first buffer; a pipeline that fails exits instead.
        while not self._caps_ready.wait(0.05):
            if self._proc is None or self._proc.poll() is not None:
                if not self._caps_ready.is_set():
                    return "eof", None
                break
            now = time.monotonic()
            if now >= self._started_at + self.startup_timeout:
                self._tail.append("(the sink negotiated no caps within %.0fs)" % self.startup_timeout)
                return "eof", None
            if deadline is not None and now >= deadline:
                return "timeout", None
        w, h, _ = self._caps  # type: ignore[misc]
        stride = rgb_stride(w)
        need = stride * h
        while len(self._buf) < need:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return "timeout", None
            ready, _, _ = select.select([self._rfd], [], [], remaining)
            if not ready:
                return "timeout", None
            chunk = os.read(self._rfd, max(need - len(self._buf), 1 << 16))
            if not chunk:
                return "eof", None
            self._buf += chunk
        raw = bytes(self._buf[:need])
        del self._buf[:need]
        np = _np()
        rows = np.frombuffer(raw, dtype=np.uint8).reshape(h, stride)
        color = np.ascontiguousarray(rows[:, : w * 3].reshape(h, w, 3))
        self._frames_this_run += 1
        return "frame", Frame(color=color, timestamp=time.time(), frame_id=self._next_id())

    def _release(self) -> None:
        # Stop the process first: a read() blocked on the pipe then sees EOF and returns, so the
        # descriptors are closed under the lock, never under a reader.
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
        with self._lock:
            self._stop()
