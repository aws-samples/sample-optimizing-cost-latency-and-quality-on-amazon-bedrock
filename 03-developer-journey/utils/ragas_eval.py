"""
RAGAS quality-evaluation helpers for Lab 07.

Runs three RAGAS metrics against agents invoked LOCALLY (in-notebook), then
pushes each score onto the agent's Langfuse trace so they show up as numeric
Scores alongside the operational metrics.

Metrics and their scope:
- tool_call_accuracy  -- tool SELECTION (name-only), all scenarios with expected
                         tool(s). No judge LLM.
- answer_correctness  -- extracted `answer` field vs an authored `reference`,
                         any scenario carrying a `reference`. LLM judge + embeddings.
- context_precision   -- retrieved KB chunks vs `reference`, RAG scenarios only
                         (`is_rag: true`, i.e. get_technical_support). LLM judge.

Judge + embeddings run on Bedrock via the langchain-aws wrappers (no OpenAI key).

Stack (verified live against Bedrock -- see CONTEXT.md and
docs/adr/0001-local-ragas-quality-evals.md):
    ragas==0.4.0, langchain-aws==0.2.35, langchain-community==0.3.27
The legacy `ragas.metrics` API is used deliberately: the newer
`ragas.metrics.collections` API reaches Bedrock via LiteLLM+Instructor, which
sends temperature+top_p together and Claude Sonnet 4.5 rejects that.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import re
import warnings

# RAGAS emits a cosmetic deprecation warning for the Langchain wrappers; the
# legacy API is the only one that works with a Bedrock judge today.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

from ragas import SingleTurnSample
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import AIMessage, HumanMessage, ToolCall as RagasToolCall
from ragas.metrics import (
    FactualCorrectness,
    LLMContextPrecisionWithReference,
    ToolCallAccuracy,
)

from utils.agent_config import MODEL_OPUS, model_settings


# --------------------------------------------------------------------------
# Judge / embeddings on Bedrock
# --------------------------------------------------------------------------
def build_bedrock_evaluator(judge_model_id: str = MODEL_OPUS, region: str | None = None):
    """Build a RAGAS judge LLM + embeddings backed by Amazon Bedrock.

    Returns (judge, embeddings) wrapped for the legacy ragas.metrics API.
    Judge = ChatBedrockConverse with model-compatible controls, embeddings
    = Titan Text v2.
    """
    import os

    from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    settings = model_settings(judge_model_id, 0.0)
    if "additional_request_fields" in settings:
        settings["additional_model_request_fields"] = settings.pop("additional_request_fields")
    judge = LangchainLLMWrapper(
        ChatBedrockConverse(model=judge_model_id, region_name=region, **settings),
        bypass_temperature="additional_model_request_fields" in settings,
    )
    embeddings = LangchainEmbeddingsWrapper(
        BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name=region)
    )
    return judge, embeddings


# --------------------------------------------------------------------------
# Local invocation + capturing tool calls / retrieved contexts
# --------------------------------------------------------------------------
def invoke_and_capture(agent, query: str) -> dict:
    """Invoke a Strands Agent locally and capture what RAGAS needs.

    Reads tool calls and tool results from `agent.messages` after the run:
    - tool_calls          -- names of every tool the agent invoked (toolUse blocks)
    - retrieved_contexts  -- text returned by get_technical_support (the KB chunks),
                             i.e. toolResult blocks for the RAG tool
    - response            -- the agent's final text answer

    A fresh agent per scenario keeps conversations isolated (no shared history).
    """
    result = agent(query)
    response_text = ""
    try:
        response_text = "".join(b["text"] for b in result.message["content"] if "text" in b)
    except (AttributeError, KeyError, IndexError, TypeError):
        response_text = str(getattr(result, "message", result))

    tool_calls: list[str] = []
    retrieved_contexts: list[str] = []
    # Map toolUseId -> tool name so we can attribute results to the RAG tool.
    rag_tool_use_ids: set[str] = set()

    for message in agent.messages:
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            if "toolUse" in block:
                name = block["toolUse"]["name"]
                tool_calls.append(name)
                if name == "get_technical_support":
                    rag_tool_use_ids.add(block["toolUse"].get("toolUseId", ""))
            if "toolResult" in block:
                tr = block["toolResult"]
                if tr.get("toolUseId", "") in rag_tool_use_ids:
                    for c in tr.get("content", []):
                        if isinstance(c, dict) and c.get("text"):
                            retrieved_contexts.append(c["text"])

    return {
        "response": response_text,
        "tool_calls": tool_calls,
        "retrieved_contexts": retrieved_contexts,
    }


# --------------------------------------------------------------------------
# Extracting the agent's answer
# --------------------------------------------------------------------------
def extract_answer(response_text: str) -> str:
    """Pull the `answer` field out of a v2+ structured response.

    v4 emits a markdown `**answer:** ...` block; v1 emits free prose. If no
    structured answer field is found, the whole response is returned so v1 and
    v4 stay comparable against the same plain-prose references.
    """
    if not response_text:
        return ""

    # Match an "answer" label with markdown bold in any position
    # (**answer:**, **answer**:, answer:) and capture up to the next field label.
    match = re.search(
        r"\**\s*answer\s*\**\s*:\s*\**\s*(.+?)(?=\n?\s*[-*]?\s*\**\s*(?:category|confidence)\s*\**\s*:|$)",
        response_text,
        re.IGNORECASE | re.DOTALL,
    )
    answer = match.group(1) if match else response_text
    # Strip stray leading/trailing markdown bold markers and list bullets.
    return answer.strip().strip("*").strip().lstrip("-").strip()


# --------------------------------------------------------------------------
# Metric runners (each returns a float in [0, 1] or None)
# --------------------------------------------------------------------------
def _run_async(coro):
    """Run an async RAGAS scorer from sync code, even inside Jupyter's live loop.

    Jupyter already runs an asyncio loop, so a bare `run_until_complete()` raises
    "This event loop is already running". `nest_asyncio` patches the running loop
    to allow re-entrant `run_until_complete`, which is the approach RAGAS itself
    uses for notebooks. (A worker-thread approach was tried and deadlocked against
    Langfuse's background threads in the kernel.)
    """
    try:
        import nest_asyncio

        nest_asyncio.apply()
    except ImportError:
        pass  # not in a nested loop (plain script) -- the calls below still work

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def score_tool_call_accuracy(query: str, actual_tools: list[str], expected_tools: list[str]) -> float | None:
    """Name-only tool selection accuracy.

    Tool calls are built with empty args so RAGAS scores tool SELECTION only --
    args are intentionally ignored (the tools normalize input, so arg matching
    would create false negatives). See CONTEXT.md.
    """
    if not expected_tools:
        return None

    ai = AIMessage(
        content="",
        tool_calls=[RagasToolCall(name=name, args={}) for name in actual_tools],
    )
    sample = MultiTurnSample(
        user_input=[HumanMessage(content=query), ai],
        reference_tool_calls=[RagasToolCall(name=name, args={}) for name in expected_tools],
    )
    return _run_async(ToolCallAccuracy().multi_turn_ascore(sample))


def score_answer_correctness(query: str, answer: str, reference: str, judge, embeddings=None) -> float | None:
    """Answer correctness via RAGAS FactualCorrectness in recall mode.

    Recall mode scores "what fraction of the reference's facts did the answer
    cover" -- it is length-NEUTRAL (a concise correct answer and a verbose
    correct answer both score 1.0). This is deliberate: AnswerCorrectness blends
    embedding similarity that rewards verbosity, which would unfairly penalize
    the optimized agent's concise answers vs. the baseline's rambling ones.
    `embeddings` is accepted but unused (kept for call-site compatibility).
    """
    scorer = FactualCorrectness(llm=judge, mode="recall")
    sample = SingleTurnSample(user_input=query, response=answer, reference=reference)
    # The Bedrock judge occasionally emits JSON the Langchain parser rejects mid-batch
    # (an intermittent OutputParserException -- the JSON is well-formed, the streaming
    # parser just trips). It's non-deterministic, so retry a few times. If every retry
    # still trips, return None ("metric unavailable") rather than raising -- one flaky
    # scenario shouldn't abort the whole Step 9 scoring loop.
    last_exc = None
    for _ in range(4):
        try:
            return _run_async(scorer.single_turn_ascore(sample))
        except Exception as exc:
            last_exc = exc
    print(f"      (factual_correctness unavailable after retries: {type(last_exc).__name__})")
    return None


def score_context_precision(query: str, reference: str, retrieved_contexts: list[str], judge) -> float | None:
    """Context Precision (with reference): RAG scenarios only."""
    if not retrieved_contexts:
        return None
    sample = SingleTurnSample(
        user_input=query,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    return _run_async(LLMContextPrecisionWithReference(llm=judge).single_turn_ascore(sample))


# --------------------------------------------------------------------------
# Reading what RAGAS needs from a DEPLOYED agent's Langfuse trace
# --------------------------------------------------------------------------
# Used by the v6-vs-v1 section (Step 9). Unlike invoke_and_capture (which runs an
# agent locally and reads agent.messages), these read the response text, the tool
# calls, and any retrieved KB chunks straight from a deployed agent's trace --
# so the quality scores describe the SAME deployed invocation as the operational
# metrics. Verified live (2026-06-02): v6 traces carry a TOOL observation per call
# plus toolUse / toolResult blocks. See docs/adr/0002.


def normalize_trace_tool_names(raw_names: list[str]) -> list[str]:
    """Map trace tool names onto the bare names the scenarios are labeled with.

    v6 reaches tools through the AgentCore Gateway, so its trace tool names are
    namespaced (``customer-support-tools___get_return_policy``) and the list also
    includes the ``skills`` plugin activation. Strip the gateway prefix and drop
    ``skills`` (an instruction-load event, not a customer-support tool selection)
    so v6 and v1 produce comparable sequences. Pure name normalization -- the
    behavior being scored ("did it call the right tool?") is unchanged.
    """
    out: list[str] = []
    for name in raw_names:
        bare = name.split("___")[-1] if "___" in name else name
        if bare == "skills":
            continue
        out.append(bare)
    return out


def find_trace_id_by_query(agent_trace_name, query, wait_seconds=5, max_retries=6, *, session_id=None):
    """Use the supplied invocation identity, not a query substring or project order."""
    from utils.langfuse_metrics import get_latest_trace_metrics

    return get_latest_trace_metrics(
        agent_name=agent_trace_name, session_id=session_id, wait_seconds=wait_seconds, max_retries=max_retries
    )["trace_id"]


def read_trace_for_eval(trace_id: str, region: str | None = None) -> dict:
    """Pull response text, tool calls, and retrieved KB chunks from a trace.

    Returns ``{"response", "tool_calls", "retrieved_contexts"}`` -- the same shape
    as invoke_and_capture, but sourced from a deployed agent's Langfuse trace
    instead of a local run. tool_calls are normalized (gateway prefix stripped,
    ``skills`` dropped). retrieved_contexts are the toolResult texts from
    get_technical_support, when present.
    """
    import json

    from utils.langfuse_metrics import trace_reader

    full = trace_reader().trace.get(trace_id, request_options={"max_retries": 0, "timeout_in_seconds": 30})
    observations = getattr(full, "observations", []) or []

    def _to_text(value):
        """Read supported trace output shapes, excluding tool proposals and reasoning."""
        if isinstance(value, str):
            if value.lstrip().startswith(("[", "{")):
                try:
                    parsed = json.loads(value)
                except ValueError:
                    return value
                text = _to_text(parsed)
                shaped = isinstance(parsed, list) or (
                    isinstance(parsed, dict)
                    and (
                        any(
                            key in parsed
                            for key in ("message", "response", "text", "content", "parts", "toolUse", "body")
                        )
                        or parsed.get("type") in ("tool_use", "tool_call")
                    )
                )
                return text if shaped or text else value
            return value
        if isinstance(value, list):
            if any(
                isinstance(block, dict) and ("toolUse" in block or block.get("type") in ("tool_use", "tool_call"))
                for block in value
            ):
                return ""
            return "\n".join(text for block in value if (text := _to_text(block)))
        if isinstance(value, dict):
            if isinstance(value.get("statusCode"), int) and 200 <= value["statusCode"] < 300 and "body" in value:
                return _to_text(value["body"])
            if "toolUse" in value or value.get("type") in ("tool_use", "tool_call"):
                return ""
            for key in ("message", "response", "text", "content", "parts"):
                if key in value:
                    return _to_text(value[key])
        return ""

    response_text = ""
    raw_tool_names: list[str] = []
    retrieved_contexts: list[str] = []

    # Final answer = the trace's top-level output (carries the agent's last message).
    response_text = _to_text(getattr(full, "output", None))

    # Prefer the latest completed answer when the trace-level output is delayed.
    ordered = sorted(
        observations,
        key=lambda o: getattr(o, "start_time", None) or _dt.datetime.min.replace(tzinfo=_dt.UTC),
        reverse=True,
    )
    if not response_text:
        for o in ordered:
            if getattr(o, "type", "") == "AGENT" or (getattr(o, "name", "") or "").startswith("invoke_agent"):
                response_text = _to_text(getattr(o, "output", None))
                if response_text:
                    break
    if not response_text:
        for o in ordered:
            if getattr(o, "type", "") == "GENERATION":
                response_text = _to_text(getattr(o, "output", None))
                if response_text:
                    break

    for o in observations:
        otype = getattr(o, "type", "")
        oname = getattr(o, "name", "") or ""
        out = getattr(o, "output", None)

        # Every TOOL observation is one tool call; its name carries the tool.
        if otype == "TOOL":
            raw_tool_names.append(oname)
            # Retrieved KB chunks: the get_technical_support tool's result text.
            if "technical_support" in oname:
                context_text = _to_text(out)
                if context_text:
                    retrieved_contexts.append(context_text)

    if not response_text.strip():
        raise ValueError("Trace has no final answer; do not score an empty response")
    return {
        "response": response_text,
        "tool_calls": normalize_trace_tool_names(raw_tool_names),
        "retrieved_contexts": retrieved_contexts,
    }


# --------------------------------------------------------------------------
# Push scores to Langfuse (v3 API, attach to an existing trace_id)
# --------------------------------------------------------------------------
def push_scores_to_langfuse(trace_id: str, scores: dict[str, float]) -> None:
    """Attach numeric RAGAS scores to a trace via Langfuse's REST ingestion API.

    Deliberately does NOT use the Langfuse SDK client: in a Jupyter kernel the
    SDK spins up background flush threads, and constructing + `shutdown()`-ing a
    client per call can deadlock against the OTEL tracing threads already running
    from `setup_langfuse_telemetry()`. A plain synchronous HTTP POST with a short
    timeout has no client lifecycle, no threads, and no event loop -- so it can't
    hang. Endpoint: POST /api/public/ingestion (batch of score-create events).
    """
    import os
    import uuid as _uuid

    import httpx

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").rstrip("/")
    if not public_key or not secret_key:
        print("      (skipping Langfuse push -- keys not set)")
        return

    events = [
        {
            "id": str(_uuid.uuid4()),
            "type": "score-create",
            "timestamp": _dt.datetime.now(_dt.UTC).isoformat(),
            "body": {
                "id": str(_uuid.uuid4()),
                "traceId": trace_id,
                "name": name,
                "value": float(value),
                "dataType": "NUMERIC",
            },
        }
        for name, value in scores.items()
        if value is not None
    ]
    if not events:
        return

    resp = httpx.post(
        f"{host}/api/public/ingestion",
        json={"batch": events},
        auth=(public_key, secret_key),
        timeout=30,
    )
    resp.raise_for_status()
