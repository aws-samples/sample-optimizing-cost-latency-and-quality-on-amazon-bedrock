"""Bounded, offline scoring for LOW's label/integer reasoning exercises.

This is a deliberately small answer grammar, not a judge or a JSON validator.
Keep the original response text alongside these diagnostics when displaying results.
"""

from __future__ import annotations

import re
from collections.abc import Collection

_MAX_TEXT_CHARS = 32_768
_INTEGER = re.compile(r"[+-]?[0-9]+")
_LABEL = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
_LEADING = re.compile(
    r"(?P<wrap>\*{1,3}|_{1,3}|`)?"
    r"(?P<answer>[+-]?[0-9]+|[A-Za-z][A-Za-z0-9_]*)"
    r"(?(wrap)(?P=wrap))"
    r"(?=$|\s|[.,:;!?/|&=+\-\u2014\u2013])"
)
_FENCE = re.compile(r"(?P<fence>```|~~~)[A-Za-z0-9_-]*\n(?P<body>[\s\S]*?)\n(?P=fence)")
_ASSERTION = re.compile(
    r"\b(?:(?:(?:final|correct)\s+)?(?:answer|decision|label|result|correction)"
    r"\s*(?::|=|\bis\b)|I\s+(?:choose|select)|actually\s*,)\s*",
    re.IGNORECASE,
)
_ALTERNATIVE = re.compile(r"(?:or\b|and\b|alternatively\b|[/,|&])", re.IGNORECASE)


def _comparison_key(answer: str) -> str:
    """Normalize a bounded token's value without converting decimal strings to int."""
    if _INTEGER.fullmatch(answer):
        digits = answer.lstrip("+-").lstrip("0") or "0"
        return "-" + digits if answer.startswith("-") and digits != "0" else digits
    return answer.lower() if _LABEL.fullmatch(answer) else answer


def _leading(text: str, choices: frozenset[str]) -> tuple[str, str] | None:
    """Read one whole leading token; never use the expected answer as a pattern."""
    match = _LEADING.match(text)
    if match is None:
        return None
    answer = match["answer"]
    if answer.lower() not in choices and _INTEGER.fullmatch(answer) is None:
        return None
    return answer, text[match.end():]


def _rationale_tail(tail: str) -> bool:
    """Require a token boundary followed by a supported explanation separator."""
    tail = tail.lstrip(" \t")
    return (
        not tail
        or tail.startswith(("\n", "\r"))
        or re.match(r"[.:;\u2014\u2013](?:\s|$)", tail) is not None
        or re.match(r"because\b", tail, re.IGNORECASE) is not None
    )


def _conflicting_assertions(tail: str, answer: str, choices: frozenset[str]) -> bool:
    key = _comparison_key(answer)
    # Inspect explicit assertions anywhere, not arbitrary occurrences of answer tokens.
    for marker in _ASSERTION.finditer(tail):
        repeated = _leading(tail[marker.end():], choices)
        if repeated is None or _comparison_key(repeated[0]) != key or not _rationale_tail(repeated[1]):
            return True

    # Also catch standalone repeated answers and alternatives on subsequent lines
    # or sentences. Arithmetic such as "17 * 23 = 391" is not a bare assertion.
    for clause in re.split(r"\n+|(?<=[.!;])\s+", tail):
        clause = clause.strip().lstrip(".:;!\u2014\u2013").strip()
        if _ALTERNATIVE.match(clause):
            return True
        repeated = _leading(clause, choices)
        if repeated is not None:
            other, rest = repeated
            if _ALTERNATIVE.match(rest.strip()):
                return True
            if _rationale_tail(rest) and _comparison_key(other) != key:
                return True
    return False


def score_effort_answer(
    text: str,
    expected: str | int,
    *,
    choices: Collection[str] | None = None,
    stop_reason: str | None = None,
    require_exact_format: bool = False,
) -> dict[str, str | bool | None]:
    """Separate answer correctness, raw format, and completion without changing text.

    Answers are labels supplied in ``choices`` (recognized without regard to case)
    or ASCII integers (optional sign, one or more digits). ``answer`` preserves the
    extracted token's spelling. ``expected`` is used ONLY for comparison: labels
    ignore ASCII case, and integers compare by value using canonical strings
    without int parsing. Leading plus/zeros are ignored and signed zero equals zero.
    Expected strings are limited to 32,768 characters; supply very large expected
    integers as decimal strings to avoid Python's int-to-decimal conversion limit.
    Label choices must be identifiers of at most 64 characters, with at most 32
    entries. Invalid caller configuration raises TypeError/ValueError.

    Extraction accepts a leading token, optionally wrapped in matching Markdown
    emphasis or inline backticks, and an optional outer three-backtick/tilde fence.
    Rationale may follow on another line, after sentence punctuation, or after
    "because". Later bare answers and explicit "answer/decision/label/result/
    correction: (or is/=) ...", "I choose/select ...", and "actually, ..."
    assertions must agree in value. Alternatives require review; rationale
    mentions such as "rather than approve/deny" do not constitute assertions.
    Unsupported forms and outputs over 32,768 characters are left unscored, never
    truncated for scoring. This bounded grammar does not interpret arbitrary prose;
    a leading "The answer is ..." remains unsupported and requires review.

    ``format_valid`` means raw stripped text is an exact case-sensitive member of
    choices or a bare signed integer, independently of correctness. Case variations
    absent from choices, Markdown, and rationale make it False.
    ``complete`` means exactly stop_reason == "end_turn".
    ``passed`` requires correctness, completion, and (when requested) valid format.
    ``review_required`` means extraction was unscored OR completion is unconfirmed.
    Missing/ambiguous answers have answer=None and answer_correct=None.
    """
    if not isinstance(text, str) or not isinstance(expected, (str, int)) or isinstance(expected, bool):
        raise TypeError("text must be str and expected must be str or int")
    # Bound caller-owned integer conversion too, even if Python's digit limit is
    # disabled. Four bits per allowed character is a cheap, conservative precheck.
    if isinstance(expected, int) and expected.bit_length() > 4 * _MAX_TEXT_CHARS:
        raise ValueError("expected exceeds the supported size; supply a bounded decimal string")
    try:
        expected_text = str(expected)
    except ValueError as exc:
        raise ValueError("expected integer is too large to render; supply a bounded decimal string") from exc
    if len(expected_text) > _MAX_TEXT_CHARS:
        raise ValueError("expected must be at most 32,768 characters")
    if choices is not None and (
        isinstance(choices, (str, bytes))
        or not isinstance(choices, Collection)
        or len(choices) > 32
        or any(not isinstance(label, str) or _LABEL.fullmatch(label) is None for label in choices)
    ):
        raise ValueError("choices must contain at most 32 identifier labels of at most 64 characters")
    labels = frozenset(choices or ())
    matching_labels = frozenset(label.lower() for label in labels)
    complete = stop_reason == "end_turn"
    answer = None
    format_valid = False
    if len(text) <= _MAX_TEXT_CHARS:
        raw = text.strip()
        format_valid = raw in labels if labels else _INTEGER.fullmatch(raw) is not None
        body = raw
        if raw.startswith(("```", "~~~")):
            fence = _FENCE.fullmatch(raw.replace("\r\n", "\n"))
            body = fence["body"].strip() if fence else ""
            # Multiple/nested fences are outside this intentionally small grammar.
            if "```" in body or "~~~" in body:
                body = ""
        leading = _leading(body, matching_labels)
        if leading is not None:
            candidate, tail = leading
            if _rationale_tail(tail) and not _conflicting_assertions(tail, candidate, matching_labels):
                answer = candidate
    answer_correct = _comparison_key(answer) == _comparison_key(expected_text) if answer is not None else None
    return {
        "answer": answer,
        "answer_correct": answer_correct,
        "format_valid": format_valid,
        "complete": complete,
        "passed": answer_correct is True and complete and (format_valid or not require_exact_format),
        "review_required": answer is None or not complete,
    }
