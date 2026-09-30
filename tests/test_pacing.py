"""Offline inference pacing contracts; transports never reach the network."""

from __future__ import annotations

import asyncio
import io
import json
import multiprocessing
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest

from workshop_utils.pacing import (
    BEDROCK_INFERENCE_OPERATIONS,
    DEFAULT_MIN_INTERVAL,
    InferencePacer,
    PacingStateError,
    attach_boto3,
    call_converse,
    get_workshop_pacer,
    paced_boto3_session,
    workshop_lock_path,
)


class FakeTime:
    def __init__(self, now=100.0):
        self.now = now
        self.sleeps = []
        self.lock = threading.Lock()

    def clock(self):
        with self.lock:
            return self.now

    def sleep(self, delay):
        assert delay > 0
        with self.lock:
            self.sleeps.append(delay)
            self.now += delay

    def pacer(self, **kwargs):
        return InferencePacer(clock=self.clock, sleeper=self.sleep, **kwargs)


def assert_spaced(admissions, interval=DEFAULT_MIN_INTERVAL):
    ordered = sorted(admissions)
    assert all(later - earlier >= interval - 1e-9 for earlier, later in pairwise(ordered))


def _process_admit(path, start, results):
    if not start.wait(5):
        raise TimeoutError("Parent did not release process test")
    results.put(InferencePacer(lock_path=path).wait())


def test_import_has_no_sdk_or_filesystem_initialization():
    code = """
import sys
def audit(event, args):
    if event.startswith('socket.'):
        raise AssertionError(event)
sys.addaudithook(audit)
import workshop_utils.pacing
assert 'boto3' not in sys.modules
assert 'botocore' not in sys.modules
assert 'openai' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("interval", [True, False, 0, -1, 1, 0.5, "1.1", None, float("nan"), float("inf"), 10**1000])
def test_interval_must_be_finite_and_strictly_below_one_rps(interval):
    with pytest.raises(ValueError):
        InferencePacer(interval)


def test_constructor_does_not_create_lockfile_and_validates_dependencies(tmp_path):
    path = tmp_path / "nested" / "pacing.lock"
    InferencePacer(lock_path=path)
    assert not path.parent.exists()
    for kwargs in ({"clock": None}, {"sleeper": 1}):
        with pytest.raises(TypeError):
            InferencePacer(**kwargs)
    for path in ("", 1, True):
        with pytest.raises(ValueError):
            InferencePacer(lock_path=path)


def test_default_spacing_early_wakeup_and_idle_without_burst_credit():
    fake = FakeTime()
    early_wakeup = True

    def sleep(delay):
        nonlocal early_wakeup
        fake.sleep(delay / 2 if early_wakeup else delay)
        early_wakeup = False

    pacer = InferencePacer(clock=fake.clock, sleeper=sleep)
    assert pacer.min_interval == 1.1
    first, second = pacer.wait(), pacer.wait()
    assert second - first == pytest.approx(1.1)
    assert len(fake.sleeps) == 2
    fake.now += 100
    third, fourth = pacer.wait(), pacer.wait()
    assert fourth - third == pytest.approx(1.1)
    assert len(fake.sleeps) == 3


def test_late_wakeup_does_not_create_catchup_burst():
    fake = FakeTime()
    pacer = InferencePacer(clock=fake.clock, sleeper=lambda delay: fake.sleep(delay + 10))
    assert_spaced([pacer.wait() for _ in range(4)])
    assert fake.sleeps == pytest.approx([11.1] * 3)


@pytest.mark.parametrize("clock_value", [float("nan"), float("inf"), -1, -0.5, True, "100"])
def test_invalid_clock_fails_closed(clock_value):
    with pytest.raises(PacingStateError):
        InferencePacer(clock=lambda: clock_value).wait()


def test_backwards_clock_never_admits_another_call():
    fake = FakeTime()
    pacer = fake.pacer()
    pacer.wait()
    fake.now = 99
    with pytest.raises(PacingStateError, match="backwards"):
        pacer.wait()


@pytest.mark.parametrize("shared_file", [False, True])
def test_concurrent_threads_and_distinct_instances_share_spacing(tmp_path, shared_file):
    fake = FakeTime()
    pacer = fake.pacer()
    barrier = threading.Barrier(8)

    def admit(_):
        selected = fake.pacer(lock_path=tmp_path / "shared.lock") if shared_file else pacer
        barrier.wait(timeout=3)
        return selected.wait()

    with ThreadPoolExecutor(max_workers=8) as pool:
        admissions = list(pool.map(admit, range(8)))
    assert_spaced(admissions)
    assert len(set(admissions)) == 8
    assert len(fake.sleeps) == 7


def test_shared_state_preserves_strongest_interval_across_instances(tmp_path):
    fake = FakeTime()
    path = tmp_path / "shared.lock"
    slow, normal = fake.pacer(min_interval=2, lock_path=path), fake.pacer(lock_path=path)
    assert slow.wait() == 100
    assert normal.wait() == 102
    assert normal.wait() == 104
    assert fake.pacer(lock_path=path).wait() == 106


def test_future_state_after_reboot_waits_full_interval(tmp_path):
    path = tmp_path / "shared.lock"
    path.write_bytes(b'\0{"version":1,"last_admitted":999999,"min_interval":1.1}')
    fake = FakeTime()
    assert fake.pacer(lock_path=path).wait() == pytest.approx(101.1)
    assert fake.sleeps == pytest.approx([1.1])


@pytest.mark.parametrize(
    "state",
    [
        b"broken",
        b"{}",
        b"[]",
        b'{"version":true,"last_admitted":100,"min_interval":1.1}',
        b'{"version":1,"last_admitted":NaN,"min_interval":1.1}',
        b'{"version":1,"last_admitted":-0.5,"min_interval":1.1}',
        b'{"version":1,"last_admitted":100,"min_interval":0.1}',
        b"x" * 4097,
    ],
)
def test_corrupt_shared_state_is_not_silently_reset(tmp_path, state):
    path = tmp_path / "shared.lock"
    path.write_bytes(b"\0" + state)
    with pytest.raises(PacingStateError):
        FakeTime().pacer(lock_path=path).wait()
    assert path.read_bytes() == b"\0" + state


def test_two_spawned_processes_coordinate_one_local_lockfile(tmp_path):
    # The only real pacing sleep: two admissions, one 1.1-second interval.
    context = multiprocessing.get_context("spawn")
    start, results = context.Event(), context.Queue()
    processes = [
        context.Process(target=_process_admit, args=(str(tmp_path / "process.lock"), start, results)) for _ in range(2)
    ]
    try:
        for process in processes:
            process.start()
        start.set()
        admissions = [results.get(timeout=8) for _ in processes]
        for process in processes:
            process.join(timeout=3)
            assert process.exitcode == 0
        assert_spaced(admissions)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
        results.close()


def test_wrapper_preserves_results_errors_and_consumes_failed_slots():
    fake = FakeTime()
    pacer = fake.pacer()

    @pacer.wrap
    def model(prompt, *, suffix):
        """Model fixture."""
        return prompt + suffix

    assert model.__name__ == "model"
    assert model.__doc__ == "Model fixture."
    assert model("hello", suffix="!") == "hello!"
    failure = ValueError("model failed")

    def fail():
        raise failure

    with pytest.raises(ValueError) as caught:
        pacer.call(fail)
    assert caught.value is failure
    assert model("next", suffix=".") == "next."
    assert fake.sleeps == pytest.approx([1.1, 1.1])


def test_async_calls_and_wrappers_share_the_sync_limiter():
    fake = FakeTime()
    pacer = fake.pacer()
    pacer.wait()

    @pacer.wrap
    async def model(value):
        return value * 2

    async def run():
        return await asyncio.gather(*(model(value) for value in range(4)))

    assert asyncio.run(run()) == [0, 2, 4, 6]
    assert len(fake.sleeps) == 4
    assert sum(fake.sleeps) == pytest.approx(4.4)


def test_generator_and_wrong_sync_async_callables_rejected_before_admission():
    fake = FakeTime()
    pacer = fake.pacer()

    def generator():
        yield "deferred network call"

    async def async_generator():
        yield "deferred async call"

    async def model():
        return "OK"

    for function in (None, generator, async_generator):
        with pytest.raises(TypeError):
            pacer.wrap(function)
    with pytest.raises(TypeError):
        pacer.call(model)
    with pytest.raises(TypeError):
        asyncio.run(pacer.acall(lambda: "OK"))
    assert fake.sleeps == []
    assert pacer.wait() == 100


def test_session_hooks_cover_only_inference_and_are_idempotent():
    boto3 = pytest.importorskip("boto3")
    fake = FakeTime()
    pacer = fake.pacer()
    session = boto3.Session(region_name="us-east-1", aws_access_key_id="offline", aws_secret_access_key="offline")
    binding = attach_boto3(session, pacer)
    attach_boto3(session, pacer)
    for operation in BEDROCK_INFERENCE_OPERATIONS:
        outcomes = session.events.emit(f"before-send.bedrock-runtime.{operation}", request=object())
        assert all(response is None for _, response in outcomes)
    assert len(fake.sleeps) == len(BEDROCK_INFERENCE_OPERATIONS) - 1
    for event in (
        "before-send.s3.GetObject",
        "before-send.sts.GetCallerIdentity",
        "before-send.bedrock.CreateModelInvocationJob",
        "before-send.bedrock.GetFoundationModelAvailability",
        "before-send.bedrock-runtime.CountTokens",
        "before-send.bedrock-runtime.ApplyGuardrail",
        "before-send.bedrock-runtime.GetAsyncInvoke",
    ):
        assert session.events.emit(event, request=object()) == []
    binding.detach()
    assert session.events.emit("before-send.bedrock-runtime.Converse", request=object()) == []


@pytest.mark.parametrize("wrapper", ["raw", "converse", "traced_converse", "strands"])
def test_inherited_session_hook_paces_real_botocore_retry_loop_with_fake_transport(monkeypatch, wrapper):
    boto3 = pytest.importorskip("boto3")
    from botocore.awsrequest import AWSResponse
    from botocore.config import Config

    fake = FakeTime()
    session = boto3.Session(region_name="us-east-1", aws_access_key_id="offline", aws_secret_access_key="offline")
    pacer = fake.pacer()
    attach_boto3(session, pacer)
    config = Config(retries={"max_attempts": 1})
    if wrapper == "strands":
        strands = pytest.importorskip("strands.models")
        client = strands.BedrockModel(
            boto_session=session,
            model_id="global.anthropic.claude-sonnet-5",
            boto_client_config=config,
        ).client
    else:
        client = session.client("bedrock-runtime", config=config)
    attach_boto3(client, pacer)  # The inherited handler must not be duplicated.
    sends = []

    class Raw(io.BytesIO):
        def stream(self, **kwargs):
            yield self.getvalue()

    def send(request):
        sends.append(fake.clock())
        if len(sends) == 1:
            status, body = 500, {"message": "retry fixture"}
        else:
            status, body = (
                200,
                {
                    "output": {"message": {"role": "assistant", "content": [{"text": "OK"}]}},
                    "usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5},
                },
            )
        return AWSResponse(request.url, status, {"content-type": "application/json"}, Raw(json.dumps(body).encode()))

    monkeypatch.setattr(client._endpoint.http_session, "send", send)
    # Preserve Endpoint's retry loop and before-send events; remove only retry
    # backoff/decision delays from this offline fixture, not pacing delays.
    monkeypatch.setattr(client._endpoint, "_needs_retry", lambda attempts, *args, **kwargs: attempts == 1)
    model = "global.anthropic.claude-sonnet-5"
    messages = [{"role": "user", "content": [{"text": "offline fixture"}]}]
    if wrapper == "converse":
        from workshop_utils.bedrock import converse

        response = converse(client, model, messages, pacer=pacer).response
    elif wrapper == "traced_converse":
        from workshop_utils import observability

        monkeypatch.setattr(observability, "_state", SimpleNamespace(backend="none"))
        response = observability.traced_converse(client, modelId=model, messages=messages, pacer=pacer)
    else:
        response = client.converse(modelId=model, messages=messages)
    assert response["output"]["message"]["content"][0]["text"] == "OK"
    assert len(sends) == 2
    assert_spaced(sends)
    assert fake.sleeps == pytest.approx([1.1])
    client.close()


def test_default_factory_is_lazy_thread_safe_and_cwd_independent(tmp_path, monkeypatch):
    from workshop_utils import pacing

    monkeypatch.setattr(pacing.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(pacing, "_default_pacers", {})
    monkeypatch.delenv("WORKSHOP_RUNTIME", raising=False)
    monkeypatch.chdir(tmp_path)
    first_path = workshop_lock_path()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: get_workshop_pacer(), range(8)))
    assert all(pacer is results[0] for pacer in results)
    assert results[0].min_interval == 1.1
    assert results[0]._lock_path == first_path
    monkeypatch.chdir(Path(__file__).parents[1])
    assert workshop_lock_path() == first_path
    assert not first_path.parent.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX filesystem permission contract")
def test_default_lock_directory_is_private_and_rejects_unsafe_parent(tmp_path, monkeypatch):
    from workshop_utils import pacing

    monkeypatch.setattr(pacing.tempfile, "gettempdir", lambda: str(tmp_path))
    path = workshop_lock_path()
    FakeTime().pacer(lock_path=path).wait()
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o777 == 0o600
    path.parent.chmod(0o777)
    try:
        with pytest.raises(PacingStateError, match="directory"):
            FakeTime().pacer(lock_path=path).wait()
    finally:
        path.parent.chmod(0o700)


def test_runtime_factory_never_needs_host_filesystem_and_resets_after_fork(monkeypatch):
    from workshop_utils import pacing

    monkeypatch.setattr(pacing, "_default_pacers", {})
    monkeypatch.setenv("WORKSHOP_RUNTIME", "true")
    monkeypatch.setattr(pacing, "workshop_lock_path", lambda: pytest.fail("Runtime accessed host lock path"))
    first = get_workshop_pacer()
    assert first is get_workshop_pacer(runtime=True)
    assert first._lock_path is None
    monkeypatch.setattr(pacing, "_factory_pid", -1)
    second = get_workshop_pacer()
    assert second is not first
    assert second._lock_path is None
    with pytest.raises(ValueError):
        get_workshop_pacer(runtime=True, lock_path="not-a-runtime-path")
    with pytest.raises(TypeError):
        get_workshop_pacer(runtime="false")
    monkeypatch.setenv("WORKSHOP_RUNTIME", "typo")
    with pytest.raises(ValueError):
        get_workshop_pacer()


def test_explicit_factory_path_and_default_use_one_instance(tmp_path, monkeypatch):
    from workshop_utils import pacing

    path = tmp_path / "private" / "inference.lock"
    monkeypatch.setattr(pacing, "workshop_lock_path", lambda: path)
    monkeypatch.setattr(pacing, "_default_pacers", {})
    assert get_workshop_pacer(runtime=False) is get_workshop_pacer(runtime=False, lock_path=path)
    with pytest.raises(ValueError):
        get_workshop_pacer(runtime=False, lock_path="")


def test_plain_mock_cannot_invent_an_emitter_and_disable_pacing(monkeypatch):
    from unittest.mock import Mock

    from workshop_utils import pacing

    fake = FakeTime()
    pacer = fake.pacer()
    monkeypatch.setattr(pacing, "get_workshop_pacer", lambda: pacer)
    client = Mock()
    response = object()
    client.converse.return_value = response
    assert call_converse(client, modelId="fixture") is response
    assert call_converse(client, pacer=pacer, modelId="fixture") is response
    assert fake.sleeps == pytest.approx([1.1])
    assert "events" not in client._mock_children
    assert "meta" not in client._mock_children
    with pytest.raises(ValueError):
        attach_boto3(client, pacer)


def test_malformed_declared_sdk_client_fails_closed():
    client = SimpleNamespace(
        meta=SimpleNamespace(service_model=SimpleNamespace(service_name="bedrock-runtime")),
        converse=lambda **kwargs: pytest.fail("Unpaced request escaped"),
    )
    with pytest.raises(TypeError):
        call_converse(client, pacer=FakeTime().pacer())


def test_paced_session_is_lazy_and_preserves_injected_configuration(monkeypatch):
    boto3 = pytest.importorskip("boto3")
    from unittest.mock import Mock

    fake = FakeTime()
    session = boto3.Session(region_name="us-east-1", aws_access_key_id="offline", aws_secret_access_key="offline")
    monkeypatch.setattr(session, "get_credentials", Mock(side_effect=AssertionError("credential lookup")))
    monkeypatch.setattr(session, "client", Mock(side_effect=AssertionError("client creation")))
    constructor = Mock(return_value=session)
    monkeypatch.setattr(boto3, "Session", constructor)
    pacer = fake.pacer()
    assert paced_boto3_session(region_name="us-east-1", pacer=pacer) is session
    constructor.assert_called_once_with(region_name="us-east-1")
    assert paced_boto3_session(region_name="us-east-1", pacer=pacer, boto_session=session) is session
    session.events.emit("before-send.bedrock-runtime.Converse", request=object())
    session.events.emit("before-send.bedrock-runtime.ConverseStream", request=object())
    assert fake.sleeps == pytest.approx([1.1])
    with pytest.raises(ValueError, match="region"):
        paced_boto3_session(region_name="us-west-2", pacer=pacer, boto_session=session)
    session.get_credentials.assert_not_called()
    session.client.assert_not_called()


def test_http_hooks_only_pace_bedrock_inference_paths_including_repeated_attempts():
    fake = FakeTime()
    pacer = fake.pacer()
    for path in ("/openai/v1/responses", "/openai/v1/chat/completions", "/v1/responses"):
        request = SimpleNamespace(method="POST", url=f"https://bedrock-mantle.us-east-1.api.aws{path}")
        assert pacer.httpx_request_hook(request) is None
        assert pacer.httpx_request_hook(request) is None  # retry
    for method, url in (
        ("GET", "https://bedrock-mantle.us-east-1.api.aws/v1/responses"),
        ("POST", "https://bedrock-mantle.us-east-1.api.aws/v1/responses/id/cancel"),
        ("POST", "https://bedrock-mantle.us-east-1.api.aws/v1/files"),
        ("POST", "https://s3.us-east-1.amazonaws.com/v1/responses"),
        ("POST", "https://example.com/openai/v1/responses"),
    ):
        pacer.httpx_request_hook(SimpleNamespace(method=method, url=url))
    assert len(fake.sleeps) == 5
    request = SimpleNamespace(method="POST", url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/responses")
    asyncio.run(pacer.async_httpx_request_hook(request))
    assert len(fake.sleeps) == 6


def test_openai_responses_sdk_retries_are_paced_by_httpx_hook(monkeypatch):
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    fake = FakeTime()
    pacer = fake.pacer()
    sends = []

    def transport(request):
        sends.append(fake.clock())
        if len(sends) == 1:
            return httpx.Response(500, json={"error": {"message": "offline retry"}})
        return httpx.Response(200, json={"id": "fixture", "object": "response", "output": []})

    with (
        httpx.Client(
            transport=httpx.MockTransport(transport), event_hooks={"request": [pacer.httpx_request_hook]}
        ) as http,
        openai.OpenAI(
            api_key="offline",
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1",
            http_client=http,
            max_retries=1,
        ) as client,
    ):
        monkeypatch.setattr(client, "_sleep_for_retry", lambda *args, **kwargs: None)
        client.responses.create(model="global.openai.gpt-5.6-luna", input="offline fixture")
    assert len(sends) == 2
    assert_spaced(sends)
    assert fake.sleeps == pytest.approx([1.1])
