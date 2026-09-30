"""Explicit local inference pacing; no clients, credentials or AWS calls on import.

Workshop Studio requires BELOW one request/second per participant. The default
1.1-second interval spaces admissions without accumulating burst credits. Attach
to every inference path; this is not a cross-host or account-wide guarantee.
Shared lockfiles require a local filesystem and the same host's monotonic clock.
Never delete/replace a lockfile while participating processes are running.
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import inspect
import json
import math
import os
import stat
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, BinaryIO, ParamSpec, TypeVar
from urllib.parse import urlsplit

DEFAULT_MIN_INTERVAL = 1.1
BEDROCK_INFERENCE_OPERATIONS = (
    "Converse",
    "ConverseStream",
    "InvokeModel",
    "InvokeModelWithResponseStream",
    "InvokeModelWithBidirectionalStream",
    "StartAsyncInvoke",
)
_EVENTS = tuple(f"before-send.bedrock-runtime.{operation}" for operation in BEDROCK_INFERENCE_OPERATIONS)
_HTTP_PATHS = frozenset(
    {"/openai/v1/responses", "/openai/v1/chat/completions", "/v1/responses", "/v1/chat/completions"}
)
P = ParamSpec("P")
T = TypeVar("T")
_factory_lock = threading.Lock()
_factory_pid = os.getpid()
_default_pacers: dict[str | None, InferencePacer] = {}


class PacingStateError(RuntimeError):
    """Invalid shared state or a non-monotonic clock; no request was admitted."""


def _number(value: Any, name: str, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if number <= minimum:
        raise ValueError(f"{name} must be greater than {minimum}")
    return number


@contextmanager
def _locked_file(path: Path) -> Iterator[BinaryIO]:
    """Stable inode/byte-range lock, automatically released on process exit."""
    if os.name not in {"posix", "nt"}:
        raise PacingStateError(f"No process-lock implementation for {os.name}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent = path.parent.lstat()
    if not stat.S_ISDIR(parent.st_mode):
        raise PacingStateError("Pacing directory must be a real directory, not a symlink")
    if os.name == "posix" and (parent.st_uid != os.getuid() or parent.st_mode & 0o022):
        raise PacingStateError("Pacing directory must be owned by this user and not writable by other users")
    fd = os.open(
        path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(fd, "r+b") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise PacingStateError("Pacing state must be a regular file")
        if os.name == "posix" and (info.st_uid != os.getuid() or info.st_mode & 0o022):
            raise PacingStateError("Pacing state must be owned by this user and not writable by other users")
        if os.name == "nt":
            import msvcrt

            # Windows locks byte zero. State starts at byte one on every platform.
            if os.fstat(stream.fileno()).st_size == 0:
                stream.write(b"\0")
                stream.flush()
            while True:
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        raise
                    time.sleep(0.01)
            try:
                yield stream
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield stream
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read_state(stream: BinaryIO) -> tuple[float | None, float]:
    stream.seek(1)
    raw = stream.read(4097)
    if not raw:
        return None, DEFAULT_MIN_INTERVAL
    try:
        if len(raw) > 4096:
            raise ValueError("Oversized state")
        state = json.loads(raw)
        if not isinstance(state, dict) or set(state) != {"version", "last_admitted", "min_interval"}:
            raise ValueError("Unexpected state fields")
        if type(state["version"]) is not int or state["version"] != 1:
            raise ValueError("Unsupported state version")
        last = _number(state["last_admitted"], "last_admitted", minimum=-1)
        if last < 0:
            raise ValueError("Negative clock")
        interval = _number(state["min_interval"], "min_interval", minimum=1)
        return last, interval
    except (ValueError, TypeError, UnicodeError) as exc:
        raise PacingStateError("Invalid pacing lockfile; inspect it before restarting participants") from exc


def _is_async(function: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(function) or inspect.iscoroutinefunction(function.__call__)


def _validate_callable(function: Any) -> None:
    if not callable(function):
        raise TypeError("An explicit inference callable is required")
    implementation = function.__call__
    if any(
        check(candidate)
        for check in (inspect.isgeneratorfunction, inspect.isasyncgenfunction)
        for candidate in (function, implementation)
    ):
        raise TypeError("Generator functions defer execution; pace their underlying inference client instead")


class InferencePacer:
    """Thread-safe admissions; process coordination is opt-in via lock_path.

    Use one instance for all inference clients in a process. Separate instances
    and notebook processes coordinate only when they share the same lock_path.
    Runtimes on other hosts need their own limiter; their schedules are separate.
    Injected clocks/sleepers are for offline tests and must share a clock domain
    when using a lockfile. Production uses time.monotonic and time.sleep.
    """

    def __init__(
        self,
        min_interval: float = DEFAULT_MIN_INTERVAL,
        *,
        lock_path: str | os.PathLike[str] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._min_interval = _number(min_interval, "min_interval", minimum=1)
        if not callable(clock) or not callable(sleeper):
            raise TypeError("clock and sleeper must be callable")
        if lock_path is not None:
            if not isinstance(lock_path, (str, os.PathLike)) or not os.fspath(lock_path):
                raise ValueError("lock_path must be a nonempty filesystem path")
            self._lock_path = Path(lock_path).expanduser().absolute()
        else:
            self._lock_path = None
        self._clock, self._sleep = clock, sleeper
        self._lock = threading.Lock()
        self._pid = os.getpid()
        self._last: float | None = None
        self._last_clock: float | None = None

    @property
    def min_interval(self) -> float:
        return self._min_interval

    def _now(self) -> float:
        try:
            now = _number(self._clock(), "clock", minimum=-1)
            if now < 0:
                raise ValueError("Negative clock")
        except ValueError as exc:
            raise PacingStateError("The clock must return finite nonnegative monotonic seconds") from exc
        if self._last_clock is not None and now < self._last_clock:
            raise PacingStateError("The pacing clock moved backwards")
        self._last_clock = now
        return now

    def _admit(self, last: float | None, interval: float) -> float:
        now = self._now()
        if last is not None:
            # A future timestamp from an earlier boot/clock domain cannot be
            # compared. Wait a full interval conservatively before replacing it.
            deadline = min(last, now) + interval
            if not math.isfinite(deadline):
                raise PacingStateError("Pacing deadline exceeds the clock range")
            while now < deadline:
                self._sleep(deadline - now)
                now = self._now()
        return now

    def wait(self) -> float:
        """Wait for one admission and return its monotonic timestamp.

        Slots are consumed even if the following request fails or is cancelled.
        The lock stays held while sleeping so concurrent callers cannot reserve
        future slots and later dispatch a burst after scheduler delays.
        """
        pid = os.getpid()
        if pid != self._pid:
            # Do not inherit a locked Python mutex after fork. Shared file state
            # remains authoritative; a local-only limiter does not span processes.
            self._lock = threading.Lock()
            self._last_clock = None
            self._pid = pid
        with self._lock:
            if self._lock_path is None:
                self._last = self._admit(self._last, self.min_interval)
                return self._last
            with _locked_file(self._lock_path) as stream:
                last, prior_interval = _read_state(stream)
                interval = max(self.min_interval, prior_interval)
                admitted = self._admit(last, interval)
                payload = json.dumps(
                    {"version": 1, "last_admitted": admitted, "min_interval": interval},
                    allow_nan=False,
                ).encode("utf-8")
                stream.seek(0)
                stream.write(b"\0" + payload)
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
                return admitted

    def call(self, function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Pace one synchronous call. Internal retries need a transport/SDK hook."""
        _validate_callable(function)
        if _is_async(function):
            raise TypeError("Use acall or wrap for an async callable")
        self.wait()
        return function(*args, **kwargs)

    async def acall(self, function: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs) -> T:
        """Pace an async call without blocking the event loop."""
        _validate_callable(function)
        if not _is_async(function):
            raise TypeError("acall requires an async callable")
        await asyncio.to_thread(self.wait)
        return await function(*args, **kwargs)

    def wrap(self, function: Callable[P, T]) -> Callable[P, T]:
        """Explicit wrapper/decorator; use retries=0 unless each retry is hooked.

        Pace Strands' underlying Bedrock clients via boto_session rather than
        wrapping the outer Agent, which can make multiple model calls.
        """
        _validate_callable(function)
        if _is_async(function):

            @wraps(function)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                return await self.acall(function, *args, **kwargs)

            return async_wrapper

        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return self.call(function, *args, **kwargs)

        return wrapper

    def before_send(self, **kwargs: Any) -> None:
        """Botocore handler. MUST return None, never a synthetic HTTP response."""
        self.wait()

    def httpx_request_hook(self, request: Any) -> None:
        """Sync HTTPX request hook for Bedrock OpenAI-compatible inference only."""
        if _is_bedrock_http_inference(request):
            self.wait()

    async def async_httpx_request_hook(self, request: Any) -> None:
        """Async HTTPX counterpart; OpenAI SDK retries run these hooks again."""
        if _is_bedrock_http_inference(request):
            await asyncio.to_thread(self.wait)


def _is_bedrock_http_inference(request: Any) -> bool:
    url = urlsplit(str(request.url))
    host = url.hostname or ""
    return (
        request.method == "POST"
        and url.scheme == "https"
        and host.startswith(("bedrock-runtime.", "bedrock-mantle."))
        and host.endswith((".amazonaws.com", ".amazonaws.com.cn", ".api.aws"))
        and url.path in _HTTP_PATHS
    )


def workshop_lock_path() -> Path:
    """Stable per-user path in the host temp directory; independent of repo/CWD.

    No directory/file is created until the first admission. Configure the same
    local temp root for cooperating processes; network-mounted temp is unsuitable.
    """
    identity = str(os.getuid()) if hasattr(os, "getuid") else str(Path.home())
    user_key = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"bedrock-workshop-pacing-{user_key}" / "inference.lock"


def get_workshop_pacer(
    *, runtime: bool | None = None, lock_path: str | os.PathLike[str] | None = None
) -> InferencePacer:
    """Lazy 1.1s singleton per process/path, with shared host state by default.

    runtime=None reads WORKSHOP_RUNTIME lazily (true/false). runtime=True uses
    process-local state and forbids a shared lock_path. No SDKs or credentials
    are loaded here. Explicit InferencePacer injection supports offline clocks.
    """
    global _factory_lock, _factory_pid, _default_pacers
    if runtime is None:
        setting = os.environ.get("WORKSHOP_RUNTIME", "false").strip().lower()
        if setting not in {"true", "false"}:
            raise ValueError("WORKSHOP_RUNTIME must be true or false")
        runtime = setting == "true"
    if not isinstance(runtime, bool):
        raise TypeError("runtime must be a bool or None")
    if runtime and lock_path is not None:
        raise ValueError("Runtime pacing is process-local; do not supply lock_path")
    if lock_path is not None and (not isinstance(lock_path, (str, os.PathLike)) or not os.fspath(lock_path)):
        raise ValueError("lock_path must be a nonempty filesystem path")
    selected_path = None if runtime else workshop_lock_path() if lock_path is None else Path(lock_path).expanduser()
    key = None if selected_path is None else str(selected_path.absolute())
    pid = os.getpid()
    if pid != _factory_pid:
        _factory_lock = threading.Lock()
        _default_pacers = {}
        _factory_pid = pid
    with _factory_lock:
        if key not in _default_pacers:
            _default_pacers[key] = InferencePacer(lock_path=key)
        return _default_pacers[key]


def _declared_attribute(target: Any, name: str) -> Any:
    """Do not mistake dynamic Mock/proxy attributes for an SDK event emitter."""
    if inspect.getattr_static(target, name, None) is None:
        return None
    return getattr(target, name)


@dataclass(frozen=True)
class BotoPacingBinding:
    """Owns explicit registrations; detaching a session does not update old clients."""

    events: Any
    unique_id: str

    def detach(self) -> None:
        for event in _EVENTS:
            self.events.unregister(event, unique_id=f"{self.unique_id}:{event}")


def attach_boto3(target: Any, pacer: InferencePacer) -> BotoPacingBinding:
    """Attach to a boto3 Session (before client creation) or existing Runtime client.

    Only exact Bedrock Runtime inference before-send events are registered, so
    each SDK retry is gated. S3, STS, CountTokens, ApplyGuardrail and control-plane
    operations are unaffected. This does not authorize blocked operations.
    Reattaching the same pacer is idempotent. Do not layer different pacers on
    one client; use the same instance across clients and wrappers.
    """
    if not isinstance(pacer, InferencePacer):
        raise TypeError("pacer must be an InferencePacer")
    events = _declared_attribute(target, "events")
    if events is None:
        meta = _declared_attribute(target, "meta")
        service = _declared_attribute(meta, "service_model")
        if _declared_attribute(service, "service_name") != "bedrock-runtime":
            raise ValueError("Expected a boto3 Session or a bedrock-runtime client")
        events = _declared_attribute(meta, "events")
    if not callable(getattr(events, "register_last", None)) or not callable(getattr(events, "unregister", None)):
        raise TypeError("Target does not expose boto3 event registration")
    unique_id = f"workshop-inference-pacing-{id(pacer)}"
    for event in _EVENTS:
        events.register_last(event, pacer.before_send, unique_id=f"{unique_id}:{event}")
    return BotoPacingBinding(events, unique_id)


def call_converse(client: Any, *, pacer: InferencePacer | None = None, **request: Any) -> Any:
    """Converse with default pacing and no request mutation.

    SDK clients require a real declared Runtime event emitter and get per-attempt
    hooks. Minimal injected/custom clients without SDK metadata are paced at the
    callable boundary; their hidden retries, if any, must be integrated separately.
    No special Mock exemption disables pacing.
    """
    selected = get_workshop_pacer() if pacer is None else pacer
    if not isinstance(selected, InferencePacer):
        raise TypeError("pacer must be an InferencePacer")
    meta = _declared_attribute(client, "meta")
    service = _declared_attribute(meta, "service_model")
    if service is not None:
        attach_boto3(client, selected)  # Fail closed for malformed SDK metadata.
        return client.converse(**request)
    return selected.call(client.converse, **request)


def paced_boto3_session(
    *,
    region_name: str | None = None,
    pacer: InferencePacer | None = None,
    boto_session: Any = None,
) -> Any:
    """Create/attach a session for Strands and future clients, lazily.

    Caller-provided sessions retain identity/configuration. If a region is given,
    it must match an injected session; do not override a frozen experiment's
    region. Session construction does not resolve credentials or create a client.
    """
    selected = get_workshop_pacer() if pacer is None else pacer
    if not isinstance(selected, InferencePacer):
        raise TypeError("pacer must be an InferencePacer")
    if region_name is not None and (not isinstance(region_name, str) or not region_name.strip()):
        raise ValueError("region_name must be a nonempty string or None")
    if boto_session is None:
        import boto3

        boto_session = boto3.Session(region_name=region_name)
    elif region_name is not None and boto_session.region_name != region_name:
        raise ValueError("Injected session region differs from the selected experiment")
    attach_boto3(boto_session, selected)
    return boto_session
