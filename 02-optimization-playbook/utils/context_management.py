"""Text context state, fixture checks and replay using the notebook's model caller."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Any

# Native JSON contracts constrain shape, not the correct fixture answers.
# Empty selections, unknown facts, and incorrect claims still reach the checks below.
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence_ids": {
            "type": "array",
            "items": {"type": "string", "description": "An ID from candidate_sentences, such as E1:S3."},
        },
    },
    "required": ["evidence_ids"],
    "additionalProperties": False,
}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "headset_model": {"type": "string"},
        "constraints": {
            "type": "array",
            "items": {"type": "string", "enum": ["no_software_install", "no_factory_reset"]},
        },
        "failed_checks": {
            "type": "array",
            "items": {"type": "string", "enum": ["cable_swap", "other_device"]},
        },
        "next_action": {
            "type": "string", "enum": ["hardware_diagnostic", "cable_swap", "other_device", "unknown"],
        },
        "diagnostic_status": {"type": "string", "enum": ["pending", "completed", "unknown"]},
        "replacement_status": {"type": "string", "enum": ["approved", "not_approved", "unknown"]},
    },
    "required": [
        "headset_model", "constraints", "failed_checks", "next_action",
        "diagnostic_status", "replacement_status",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Exchange:
    exchange_id: str
    user: str
    assistant: str


def _user_sentences(text: str) -> list[str]:
    """Use identical sentence boundaries in summary requests and validation."""
    return re.split(r"(?<=[.!?])\s+", text)


class RollingContext:
    """Keep complete exchanges; replace older ones only after evidence validation.

    The archive and required_evidence are an offline regression oracle for the
    supplied transcript. Neither is injected into answer requests. This is not a
    general semantic validator or a manager for tool/reasoning histories.
    """

    def __init__(
        self, *, keep_exchanges: int = 2,
        required_evidence: dict[str, list[str]] | None = None,
    ) -> None:
        if not isinstance(keep_exchanges, int) or keep_exchanges < 1:
            raise ValueError("Keep at least one complete exchange.")
        self.keep_exchanges = keep_exchanges
        self.required_evidence = required_evidence or {}
        self.archive: list[Exchange] = []
        self.recent: list[Exchange] = []
        self.summary: list[dict[str, str]] = []

    def append(self, exchange: Exchange) -> None:
        if not all(isinstance(v, str) and v.strip() for v in asdict(exchange).values()):
            raise ValueError("Only nonempty text exchanges are supported; never flatten tool/reasoning blocks.")
        if any(e.exchange_id == exchange.exchange_id for e in self.archive):
            raise ValueError("Exchange IDs must be unique.")
        self.archive.append(exchange)
        self.recent.append(exchange)

    def messages(self, question: str) -> list[dict[str, Any]]:
        messages = []
        for i, exchange in enumerate(self.recent):
            user_data: dict[str, Any] = {
                "exchange_id": exchange.exchange_id, "user_message": exchange.user,
            }
            if i == 0 and self.summary:
                user_data["earlier_evidence"] = self.summary
            messages.extend([
                {"role": "user", "content": [{"text": json.dumps(user_data)}]},
                {"role": "assistant", "content": [{"text": exchange.assistant}]},
            ])
        messages.append({"role": "user", "content": [{"text": question}]})
        return messages

    def context_chars(self, question: str, system: str) -> int:
        """Serialized character count: an approximate size proxy, NOT tokens."""
        return len(json.dumps({"system": system, "messages": self.messages(question)}))

    def should_compact(self, policy: str, question: str, system: str, char_budget: int) -> bool:
        if policy not in {"full", "frequent", "budget"}:
            raise ValueError("Unknown context policy.")
        if char_budget <= 0:
            raise ValueError("The approximate character budget must be positive.")
        if policy == "full" or len(self.recent) <= self.keep_exchanges:
            return False
        return policy == "frequent" or self.context_chars(question, system) >= char_budget

    def _summary_candidates(self) -> dict[str, dict[str, str]]:
        # Only expose previous retained evidence and newly eligible user sentences.
        # The full archive is a validation reference, not a recovery source for the model.
        available = {(item["exchange_id"], item["quote"]) for item in self.summary}
        available.update((e.exchange_id, sentence) for e in self.recent[:-self.keep_exchanges]
                         for sentence in _user_sentences(e.user))
        return {
            f"{exchange.exchange_id}:S{index}": {"exchange_id": exchange.exchange_id, "quote": sentence}
            for exchange in self.archive[:-self.keep_exchanges]
            for index, sentence in enumerate(_user_sentences(exchange.user), start=1)
            if (exchange.exchange_id, sentence) in available
        }

    def summary_request(self) -> str:
        candidates = self._summary_candidates()
        previous = {(item["exchange_id"], item["quote"]) for item in self.summary}
        return json.dumps({
            "retained_evidence": [
                {"id": key, "text": item["quote"]} for key, item in candidates.items()
                if (item["exchange_id"], item["quote"]) in previous
            ],
            "candidate_sentences": [
                {"id": key, "text": item["quote"]} for key, item in candidates.items()
                if (item["exchange_id"], item["quote"]) not in previous
            ],
        })

    def accept_summary(self, text: str, *, stop_reason: str) -> list[str]:
        """Add selected evidence to retained facts; mutate history only on success.

        Previously accepted facts are retained by the application, not selected
        afresh by the model. Repeating an offered retained ID is harmless.
        Newly missing facts still fail validation; the reference never repairs
        or supplies the model's selection. Corrections remain in source order.
        """
        if stop_reason != "end_turn":
            return [f"Summary did not finish normally: {stop_reason}"]
        if len(self.recent) <= self.keep_exchanges:
            return ["No complete older exchanges are eligible."]
        try:
            selection = json.loads(text)
        except (ValueError, TypeError):
            return ["Summary is not JSON."]
        if (
            not isinstance(selection, dict) or set(selection) != {"evidence_ids"}
            or not isinstance(selection["evidence_ids"], list)
            or not all(isinstance(key, str) for key in selection["evidence_ids"])
        ):
            return ["Summary needs a list of sentence IDs (empty if no new facts are needed)."]
        selected = selection["evidence_ids"]
        candidates = self._summary_candidates()
        errors = [f"Unknown or unavailable sentence ID: {key}" for key in selected if key not in candidates]
        if len(set(selected)) != len(selected):
            errors.append("Duplicate sentence IDs.")
        previous = {(item["exchange_id"], item["quote"]) for item in self.summary}
        quotes = previous | {(candidates[key]["exchange_id"], candidates[key]["quote"])
                             for key in selected if key in candidates}
        for exchange in self.archive[:-self.keep_exchanges]:
            for quote in self.required_evidence.get(exchange.exchange_id, []):
                if (exchange.exchange_id, quote) not in quotes:
                    errors.append(f"Missing required evidence: {exchange.exchange_id}: {quote}")
        if errors:
            return errors
        # Candidate order follows the original conversation, preserving corrections.
        self.summary = [
            dict(item) for item in candidates.values()
            if (item["exchange_id"], item["quote"]) in quotes
        ]
        self.recent = self.recent[-self.keep_exchanges:]
        return []


def answer_checks(text: str, stop_reason: str) -> dict[str, bool]:
    """Deterministic support-fixture checks, deliberately separate from prompts."""
    expected = {
        "headset_model": "H-220",
        "constraints": ["no_software_install", "no_factory_reset"],
        "failed_checks": ["cable_swap", "other_device"],
        "next_action": "hardware_diagnostic",
        "diagnostic_status": "pending",
        "replacement_status": "not_approved",
    }
    try:
        answer = json.loads(text)
    except (ValueError, TypeError):
        answer = None
    checks = {"complete": stop_reason == "end_turn",
              "format": isinstance(answer, dict) and set(answer) == set(expected)}
    for key, value in expected.items():
        actual = answer.get(key) if isinstance(answer, dict) else None
        if isinstance(value, list):
            checks[key] = (
                isinstance(actual, list) and all(isinstance(v, str) for v in actual)
                and sorted(actual) == sorted(value)
            )
        else:
            checks[key] = actual == value
    return checks


def call_totals(rows: list[dict[str, Any]], *, full_task_latency_ms: float) -> dict[str, Any]:
    """Aggregate returned meters, including rejected summaries; unknown stays unknown."""
    summaries = [row for row in rows if row["context_kind"] == "summary"]
    answers = [row for row in rows if row["context_kind"] == "answer"]

    def cost(group: list[dict[str, Any]]) -> float | None:
        return None if any(r.get("cost_usd") is None or r.get("sdk_retries", 0) for r in group) else sum(
            r["cost_usd"] for r in group
        )

    totals = {
        key: sum(row[key] for row in rows)
        for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
    }
    return {
        "model_calls": len(rows), "summary_calls": len(summaries),
        "answer_calls": len(rows) - len(summaries),
        **totals,
        "total_input_tokens": totals["input_tokens"] + totals["cache_read_tokens"] + totals["cache_write_tokens"],
        "total_cost_usd": cost(rows), "summary_cost_usd": cost(summaries),
        "handoff_cost_usd": cost(answers),
        "handoff_input_tokens": sum(r["input_tokens"] + r["cache_read_tokens"] + r["cache_write_tokens"] for r in answers),
        "summary_input_tokens": sum(r["input_tokens"] + r["cache_read_tokens"] + r["cache_write_tokens"] for r in summaries),
        "sdk_retries": sum(r.get("sdk_retries", 0) for r in rows),
        "usage_complete": not any(r.get("sdk_retries", 0) for r in rows),
        "summed_call_latency_ms": sum(r["latency_ms"] for r in rows),
        "full_task_latency_ms": full_task_latency_ms,
    }


def replay_context(
    policy: str, *, call, exchanges: list[Exchange], checkpoints: dict[int, str],
    required_evidence: dict[str, list[str]], model_id: str, summary_model: str,
    answer_system: str, summary_system: str,
    answer_output_config: dict, summary_output_config: dict,
    keep_exchanges: int = 2, char_budget: int = 4200,
    answer_max_tokens: int = 512, summary_max_tokens: int = 1024, call_limit: int = 18,
) -> dict[str, Any]:
    """Replay one policy with the caller's measured model function.

    No SDK/client initialization occurs here. Each requested summary and handoff
    is a separate measured call, with no hidden repair calls. Keep the input
    configuration and per-exchange context snapshots with the result for review.
    """
    if policy not in {"full", "frequent", "budget"}:
        raise ValueError("Unknown context policy.")
    if not 1 <= keep_exchanges < len(exchanges):
        raise ValueError("Keep at least one exchange and leave older exchanges eligible for compaction.")
    if char_budget <= 0:
        raise ValueError("Use a positive approximate character budget.")
    if not checkpoints or any(turn < 1 or turn > len(exchanges) for turn in checkpoints):
        raise ValueError("Checkpoints must refer to exchanges in the conversation.")
    max_calls = len(checkpoints) + (0 if policy == "full" else len(exchanges) - keep_exchanges)
    if max_calls > call_limit:
        raise ValueError(f"This policy needs up to {max_calls} calls; review the call limit first.")

    state = RollingContext(keep_exchanges=keep_exchanges, required_evidence=required_evidence)
    calls, events, answers, steps = [], [], [], []
    started = time.perf_counter()
    for turn, exchange in enumerate(exchanges, 1):
        state.append(exchange)
        question = checkpoints.get(turn, checkpoints[max(checkpoints)])
        before_chars = state.context_chars(question, answer_system)
        action = "Keep history"
        errors = []
        if state.should_compact(policy, question, answer_system, char_budget):
            previously_retained = len(state.summary)
            summary_row = call(
                summary_model, state.summary_request(), label=f"L09-{policy}-summary-{turn}",
                system=[{"text": summary_system}], max_tokens=summary_max_tokens,
                output_config=summary_output_config,
            )
            summary_row["context_kind"] = "summary"
            calls.append(summary_row)
            errors = state.accept_summary(summary_row["text"], stop_reason=summary_row["stop_reason"])
            action = "Summary accepted" if not errors else "Summary rejected; history kept"
            events.append({
                "policy": policy, "turn": turn, "accepted": not errors, "errors": errors,
                "before_chars": before_chars, "after_chars": state.context_chars(question, answer_system),
                "summary_text": summary_row["text"],
                "evidence_carried_forward": previously_retained,
                "new_evidence_added": len(state.summary) - previously_retained,
            })
        # Copy snapshots so later compaction cannot rewrite the earlier evidence.
        steps.append({
            "turn": turn, "action": action, "before_chars": before_chars,
            "after_chars": state.context_chars(question, answer_system),
            "recent_ids": [e.exchange_id for e in state.recent],
            "summary": [dict(item) for item in state.summary],
            "handoff_requested": turn in checkpoints,
            "validation_errors": list(errors),
        })
        if turn in checkpoints:
            answer_row = call(
                model_id, messages=state.messages(question), label=f"L09-{policy}-answer-{turn}",
                system=[{"text": answer_system}], max_tokens=answer_max_tokens,
                output_config=answer_output_config,
            )
            answer_row["context_kind"] = "answer"
            calls.append(answer_row)
            checks = answer_checks(answer_row["text"], answer_row["stop_reason"])
            answers.append({
                "policy": policy, "turn": turn, "passed": all(checks.values()),
                "checks": checks, "answer": answer_row["text"],
            })
    elapsed_ms = (time.perf_counter() - started) * 1000
    totals = {
        "policy": policy, **call_totals(calls, full_task_latency_ms=elapsed_ms),
        "checkpoints_passed": sum(a["passed"] for a in answers), "checkpoints": len(answers),
        "summaries_accepted": sum(e["accepted"] for e in events),
        "summaries_rejected": sum(not e["accepted"] for e in events),
    }
    return {
        "policy": policy, "state": state, "calls": calls, "events": events,
        "answers": answers, "steps": steps, "totals": totals,
        "settings": {
            "model_id": model_id, "summary_model": summary_model,
            "keep_exchanges": keep_exchanges, "char_budget": char_budget,
            "answer_max_tokens": answer_max_tokens, "summary_max_tokens": summary_max_tokens,
        },
    }
