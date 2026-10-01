"""Text-only context state and fixture checks. Model calls live in the notebook."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

# Native JSON contracts constrain shape, not the correct fixture answers.
# Empty evidence, unknown facts, and incorrect claims still reach the checks below.
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "exchange_id": {"type": "string"},
                    "quote": {
                        "type": "string",
                        "description": "One complete original user sentence, copied verbatim. Never join sentences.",
                    },
                },
                "required": ["exchange_id", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["evidence"],
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

    def summary_request(self) -> str:
        return json.dumps({
            "previous_evidence": self.summary,
            "newly_evicted_exchanges": [
                {"exchange_id": e.exchange_id, "user_sentences": _user_sentences(e.user)}
                for e in self.recent[:-self.keep_exchanges]
            ],
        })

    def accept_summary(self, text: str, *, stop_reason: str) -> list[str]:
        """Return rejection reasons; mutate retained context only on success."""
        if stop_reason != "end_turn":
            return [f"Summary did not finish normally: {stop_reason}"]
        if len(self.recent) <= self.keep_exchanges:
            return ["No complete older exchanges are eligible."]
        try:
            candidate = json.loads(text)
        except (ValueError, TypeError):
            return ["Summary is not JSON."]
        if (
            not isinstance(candidate, dict) or set(candidate) != {"evidence"}
            or not isinstance(candidate["evidence"], list) or not candidate["evidence"]
        ):
            return ["Summary needs a nonempty evidence list."]
        # Includes previously compacted exchanges, so repeated summaries must
        # preserve old constraints as well as facts in the newly evicted window.
        covered = {e.exchange_id: e for e in self.archive[:-self.keep_exchanges]}
        quotes = set()
        errors = []
        for entry in candidate["evidence"]:
            if (
                not isinstance(entry, dict) or set(entry) != {"exchange_id", "quote"}
                or not all(isinstance(v, str) for v in entry.values())
            ):
                errors.append("Each evidence item needs an exchange_id and a quote.")
                continue
            eid, quote = entry["exchange_id"], entry["quote"]
            # Full sentences retain negation; a substring such as 'approved'
            # cannot pass by matching 'No replacement has been approved.'
            sentences = _user_sentences(covered[eid].user) if eid in covered else []
            if not quote.strip() or quote not in sentences:
                errors.append(f"Not a verbatim user sentence from eligible history: {eid}: {quote}")
            if (eid, quote) in quotes:
                errors.append(f"Duplicate evidence: {eid}: {quote}")
            quotes.add((eid, quote))
        for eid in covered:
            for quote in self.required_evidence.get(eid, []):
                if (eid, quote) not in quotes:
                    errors.append(f"Missing required evidence: {eid}: {quote}")
        if errors:
            return errors
        chronology = {eid: index for index, eid in enumerate(covered)}
        self.summary = sorted(candidate["evidence"], key=lambda entry: chronology[entry["exchange_id"]])
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
        "sdk_retries": sum(r.get("sdk_retries", 0) for r in rows),
        "usage_complete": not any(r.get("sdk_retries", 0) for r in rows),
        "summed_call_latency_ms": sum(r["latency_ms"] for r in rows),
        "full_task_latency_ms": full_task_latency_ms,
    }
