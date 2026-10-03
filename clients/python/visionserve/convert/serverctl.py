"""Find (or start) the VisionServe server that tiers B/C and the server speed level talk to.

Order:
  1. --no-server                      -> no server; tiers B/C/server-speed are SKIPPED, with the reason.
  2. --server URL / $VISIONSERVE_HOST / http://localhost:11435, if it answers /api/health AND lists
     every model just installed (GET /api/models). A server only sees ITS OWN registry directory:
     one that answers but does not list the model is serving another --models dir, so it is
     reported and not used.
  3. otherwise a TEMPORARY local server on the registry the model was installed into:
        $VISIONSERVE_BIN serve --models DIR --addr 127.0.0.1:<free port> --idle-unload-seconds 0
     (needs ORT_DYLIB_PATH, which the converter image sets), stopped when verification ends.
  4. if neither works -> None, and every message says why. Never silently.

stdlib only (+ the SDK client), dependencies injectable for offline tests.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence

DEFAULT_URL = "http://localhost:11435"


class ServerHandle:
    def __init__(self, url: str, temporary: bool, proc=None, log_path: Optional[Path] = None, client=None):
        self.url, self.temporary, self.proc, self.log_path = url, temporary, proc, log_path
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from ..client import Client
            self._client = Client(self.url, timeout=600)
        return self._client

    def describe(self) -> dict:
        return {"url": self.url, "temporary": self.temporary}

    def close(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _health(url: str, timeout: float = 2.0) -> bool:
    from ..client import Client
    try:
        return (Client(url, timeout=timeout).health() or {}).get("status") == "ok"
    except Exception:  # noqa: BLE001 — unreachable, refused, bad JSON: all "not usable"
        return False


def _listed(url: str, timeout: float = 5.0) -> List[str]:
    from ..client import Client
    return [m.name for m in Client(url, timeout=timeout).list_models()]


def registered_on_server(names: Sequence[str], *, url: Optional[str] = None, no_server: bool = False,
                         log: Callable[[str], None] = print, env=None, health=None, listed=None) -> List[str]:
    """Names in `names` that the server verification would talk to ALREADY has registered — call
    it BEFORE installing. Such a server keeps serving the manifest and sessions it loaded for
    that name (Go re-scans the registry only for names it does not know), so after this run
    replaces the model it would test the OLD one. /api/models carries no weight digest to tell
    the two apart; acquire(fresh=True) then verifies on a temporary server instead."""
    if no_server:
        return []
    health = health or _health
    listed = listed or _listed
    env = dict(os.environ if env is None else env)
    target = (url or env.get("VISIONSERVE_HOST") or DEFAULT_URL).rstrip("/")
    try:
        if not health(target):
            return []
        have = set(listed(target))
    except Exception:  # noqa: BLE001 — unreachable: nothing stale there
        return []
    stale = [n for n in names if n in have]
    if stale:
        log(f"server: {target} already has {', '.join(stale)} registered from before this run; it would keep "
            "serving that copy, so the new model is verified on a temporary server")
    return stale


def acquire(names: Sequence[str], models_dir, *, url: Optional[str] = None, no_server: bool = False,
            workdir=None, log: Callable[[str], None] = print, health=_health, listed=_listed,
            popen=subprocess.Popen, which=shutil.which, env=None, start_timeout: float = 60.0,
            sleep=time.sleep, fresh: bool = False) -> Optional[ServerHandle]:
    """Return a ServerHandle whose registry contains every model in `names`, or None (after logging
    why). The caller must close() it (or use it as a context manager).

    fresh=True: never reuse a running server (it had these names registered before they were
    replaced — see registered_on_server); always start a temporary one on `models_dir`."""
    env = dict(os.environ if env is None else env)
    if no_server:
        log("server: --no-server — tiers B/C and the server speed level are skipped")
        return None
    explicit = url or env.get("VISIONSERVE_HOST")
    target = (explicit or DEFAULT_URL).rstrip("/")
    if fresh:
        log(f"server: not reusing {target}: it may still serve the previous {', '.join(names)}; starting a "
            "temporary server on the registry instead")
        return start_temporary(models_dir, workdir=workdir, log=log, health=health, popen=popen, which=which,
                               env=env, timeout=start_timeout, sleep=sleep)
    if health(target):
        try:
            have = set(listed(target))
        except Exception as e:  # noqa: BLE001
            have = set()
            log(f"server: {target} answers /api/health but GET /api/models failed ({e})")
        missing = [n for n in names if n not in have]
        if not missing:
            log(f"server: using {target}")
            return ServerHandle(target, temporary=False)
        log(f"server: WARNING {target} is up but does not list {', '.join(missing)} — it serves a different "
            f"registry than --models {models_dir} (a server only sees its own --models dir). Not using it; "
            "starting a temporary server on the right registry instead")
    elif explicit:
        log(f"server: {target} is not reachable; starting a temporary local server instead")
    return start_temporary(models_dir, workdir=workdir, log=log, health=health, popen=popen, which=which,
                           env=env, timeout=start_timeout, sleep=sleep)


def start_temporary(models_dir, *, workdir=None, log=print, health=_health, popen=subprocess.Popen,
                    which=shutil.which, env=None, timeout: float = 60.0, sleep=time.sleep
                    ) -> Optional[ServerHandle]:
    env = dict(os.environ if env is None else env)
    binary = env.get("VISIONSERVE_BIN", "visionserve")
    exe = binary if (os.sep in binary and Path(binary).is_file()) else which(binary)
    if not exe:
        log(f"server: SKIPPED tiers B/C/server-speed — no reachable server and no `{binary}` binary to start one "
            "(set VISIONSERVE_BIN, or pass --server URL of a server whose --models is this registry)")
        return None
    if not env.get("ORT_DYLIB_PATH"):
        log("server: warning: ORT_DYLIB_PATH is not set; the temporary server may fail to load ONNX Runtime")
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    log_path = Path(workdir or ".") / f"visionserve-serve-{port}.log"
    cmd = [exe, "serve", "--models", str(models_dir), "--addr", f"127.0.0.1:{port}", "--idle-unload-seconds", "0"]
    log("server: starting a temporary server: " + " ".join(cmd))
    fh = open(log_path, "wb")
    try:
        proc = popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    except OSError as e:
        fh.close()
        log(f"server: SKIPPED tiers B/C/server-speed — could not start {exe}: {e}")
        return None
    h = ServerHandle(url, temporary=True, proc=proc, log_path=log_path)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                h.proc = None
                log(f"server: SKIPPED tiers B/C/server-speed — the temporary server exited with code "
                    f"{proc.returncode}:\n" + _tail(log_path))
                return None
            if health(url):
                log(f"server: temporary server up at {url} (log {log_path})")
                return h
            sleep(0.2)
    except BaseException:  # Ctrl-C while waiting: the child must not outlive us
        h.close()
        raise
    finally:
        fh.close()  # the child keeps its own copy of the descriptor
    h.close()
    log(f"server: SKIPPED tiers B/C/server-speed — the temporary server did not answer /api/health within "
        f"{timeout:.0f}s:\n" + _tail(log_path))
    return None


def _tail(path: Path, n: int = 12) -> str:
    try:
        lines = Path(path).read_text(errors="replace").splitlines()[-n:]
    except OSError:
        return "  (no log)"
    return "\n".join("  | " + ln for ln in lines)
