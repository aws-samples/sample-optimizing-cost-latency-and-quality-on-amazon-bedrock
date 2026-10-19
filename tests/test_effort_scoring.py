"""Offline contracts for the bounded LOW reasoning-answer scorer."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "02-optimization-playbook/utils/effort_scoring.py"
CHOICES = ("approve", "deny", "ask_for_receipt")


@pytest.fixture(scope="module")
def score():
    # Load this helper alone; importing notebook setup would initialize external clients.
    spec = importlib.util.spec_from_file_location("effort_scoring", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score_effort_answer


@pytest.mark.parametrize(
    "text,expected,choices",
    [
        ("**6195**", "6195", None),
        ("**ask_for_receipt**\n\nexplanation...", "ask_for_receipt", CHOICES),
    ],
)
@pytest.mark.parametrize("strict", [False, True])
def test_user_examples_separate_correctness_from_format(score, text, expected, choices, strict):
    assert score(text, expected, choices=choices, stop_reason="end_turn", require_exact_format=strict) == {
        "answer": expected,
        "answer_correct": True,
        "format_valid": False,
        "complete": True,
        "passed": not strict,
        "review_required": False,
    }


@pytest.mark.parametrize(
    "text,expected,choices",
    [
        ("6196", "6195", None),
        ("approve", "ask_for_receipt", CHOICES),
        ("-12", "-13", None),
    ],
)
def test_wrong_bare_answers_have_valid_shape(score, text, expected, choices):
    assert score(text, expected, choices=choices, stop_reason="end_turn", require_exact_format=True) == {
        "answer": text,
        "answer_correct": False,
        "format_valid": True,
        "complete": True,
        "passed": False,
        "review_required": False,
    }


@pytest.mark.parametrize(
    "text,expected,choices",
    [
        (" \n6195\t", "6195", None),
        ("ask_for_receipt", "ask_for_receipt", CHOICES),
        ("-12", -12, None),
        ("+12", "+12", None),
        ("0012", "0012", None),
    ],
)
def test_exact_format_means_stripped_bare_shape(score, text, expected, choices):
    result = score(text, expected, choices=choices, stop_reason="end_turn", require_exact_format=True)
    assert result["answer"] == text.strip()
    assert result["answer_correct"] is result["format_valid"] is result["passed"] is True
    assert result["review_required"] is False


@pytest.mark.parametrize(
    "text",
    [
        "*6195*",
        "__6195__",
        "***6195***",
        "_6195_",
        "`6195`",
        "```text\n6195\n```",
        "```\r\n6195\r\n```",
        "~~~\n**6195**\n~~~",
        "6195 \u2013 calculated from the expression.",
        "6195.\n\n17 * 23 * 19 - 1234 = 6195.",
        "6195 because 17 * 23 * 19 - 1234 = 6195.",
        "6195\n\n6195\n\nFinal answer: **6195**.",
    ],
)
def test_harmless_formatting_and_math_rationale(score, text):
    result = score(text, "6195", stop_reason="end_turn")
    assert result["answer"] == "6195"
    assert result["passed"] is True
    assert result["format_valid"] is result["review_required"] is False


@pytest.mark.parametrize(
    "text",
    [
        "**ask_for_receipt**\n\nChoose this rather than approve/deny.",
        "ask_for_receipt. The alternatives approve and deny lack supporting evidence.",
        "ask_for_receipt because the receipt is missing, rather than approve or deny.",
        "ask_for_receipt\n\nThe policy says to approve when all conditions are met.",
    ],
)
def test_mentions_of_alternatives_are_not_answer_assertions(score, text):
    result = score(text, "ask_for_receipt", choices=CHOICES, stop_reason="end_turn")
    assert result["answer"] == "ask_for_receipt"
    assert result["passed"] is True
    assert result["review_required"] is False


@pytest.mark.parametrize(
    "text,expected,choices,answer",
    [
        ("6196\n\nI considered 6195 but used different arithmetic.", "6195", None, "6196"),
        ("approve\n\nI considered ask_for_receipt as an alternative.", "ask_for_receipt", CHOICES, "approve"),
        ("**deny**\n\nUse this rather than ask_for_receipt.", "ask_for_receipt", CHOICES, "deny"),
    ],
)
def test_wrong_leading_answer_is_not_rescued_by_expected_in_rationale(score, text, expected, choices, answer):
    result = score(text, expected, choices=choices, stop_reason="end_turn")
    assert result["answer"] == answer
    assert result["answer_correct"] is result["passed"] is result["review_required"] is False


@pytest.mark.parametrize(
    "text,choices",
    [
        ("6195 or 6196", None),
        ("6195/6196", None),
        ("6195, 6196", None),
        ("6195 6196", None),
        ("6195\n6196", None),
        ("6195\n6195 or 6196", None),
        ("6195.\nOr 6196.", None),
        ("6195\n\nFinal answer: 6196", None),
        ("6195\n\nThe answer is **6196**.", None),
        ("6195\n\nAfter recalculation, the correct answer is 6196.", None),
        ("6195\n\nCorrection: 6196.", None),
        ("6195. Actually, 6196.", None),
        ("ask_for_receipt or approve", CHOICES),
        ("ask_for_receipt/deny", CHOICES),
        ("**ask_for_receipt** / **deny**", CHOICES),
        ("ask_for_receipt\n\n**deny**", CHOICES),
        ("ask_for_receipt. Final decision: deny.", CHOICES),
        ("ask_for_receipt\n\nI choose deny.", CHOICES),
        ("ask_for_receipt\n\nAnswer: approve or deny", CHOICES),
        ("approve\n\nFinal answer: ask_for_receipt", CHOICES),
    ],
)
def test_competing_answers_are_unscored_for_every_expected_value(score, text, choices):
    # Changing the expected value must never resolve ambiguity in extraction.
    for expected in choices or ("6195", "6196"):
        assert score(text, expected, choices=choices, stop_reason="end_turn") == {
            "answer": None,
            "answer_correct": None,
            "format_valid": False,
            "complete": True,
            "passed": False,
            "review_required": True,
        }


@pytest.mark.parametrize(
    "text",
    [
        "",
        " \n ",
        "The answer is 6195.",
        "I cannot answer this question. 6195",
        "I cannot approve this request.",
        "unknown_label",
        "ask_for_receipt_extra",
        "61950suffix",
        "6195.0",
        "6,195",
        "6195 + 1",
        "6195 - 1",
        "6195!",
        "6195 = 6196",
        "6195e0",
        "\uff16\uff11\uff19\uff15",  # Fullwidth digits are not ASCII integers.
        "**6195*",
        "6195 is not my answer.",
        "```text\n6195",
        "```text\n6195\n```\nFinal answer: 6196",
        "```\n6195\n```\n```\n6196\n```",
        '{"answer": 6195}',
        '```json\n{"answer": 6195}\n```',
        '["6195"]',
    ],
)
def test_unsupported_or_nonleading_answers_are_not_substring_matches(score, text):
    result = score(text, "6195", choices=CHOICES, stop_reason="end_turn")
    assert result["answer"] is result["answer_correct"] is None
    assert result["format_valid"] is result["passed"] is False
    assert result["review_required"] is True


def test_labels_require_explicit_choices(score):
    for choices in (None, (), ("approve", "deny")):
        result = score("ask_for_receipt", "ask_for_receipt", choices=choices, stop_reason="end_turn")
        assert result["answer"] is result["answer_correct"] is None
        assert result["passed"] is False
    result = score("61950", "6195", stop_reason="end_turn")
    assert result["answer"] == "61950"
    assert result["answer_correct"] is False


@pytest.mark.parametrize(
    "text,expected",
    [
        ("+6195", "6195"),
        ("06195", "6195"),
        ("+0006195", 6195),
        ("6195", "+0006195"),
        ("-006195", -6195),
        ("-6195", "-006195"),
        ("-0", "0"),
        ("+000", "-0000"),
        ("0000", 0),
    ],
)
def test_integer_correctness_compares_value_and_preserves_answer_spelling(score, text, expected):
    assert score(text, expected, stop_reason="end_turn", require_exact_format=True) == {
        "answer": text,
        "answer_correct": True,
        "format_valid": True,
        "complete": True,
        "passed": True,
        "review_required": False,
    }


@pytest.mark.parametrize("text,expected", [("-06195", "6195"), ("+6195", "-6195"), ("06196", "6195")])
def test_integer_normalization_preserves_sign_and_value_differences(score, text, expected):
    result = score(text, expected, stop_reason="end_turn")
    assert result["answer"] == text
    assert result["answer_correct"] is result["passed"] is result["review_required"] is False
    assert result["format_valid"] is True


@pytest.mark.parametrize(
    "text,expected,choices,format_valid",
    [
        ("Billing", "billing", ("billing", "technical", "returns"), False),
        ("BILLING", "billing", ("billing",), False),
        ("billing", "BILLING", ("billing",), True),
        ("ask_for_receipt", "ask_for_receipt", ("ASK_FOR_RECEIPT",), False),
        ("ASK_FOR_RECEIPT", "ask_for_receipt", CHOICES, False),
    ],
)
@pytest.mark.parametrize("strict", [False, True])
def test_label_semantics_ignore_case_but_raw_format_uses_choices(score, text, expected, choices, format_valid, strict):
    assert score(text, expected, choices=choices, stop_reason="end_turn", require_exact_format=strict) == {
        "answer": text,
        "answer_correct": True,
        "format_valid": format_valid,
        "complete": True,
        "passed": format_valid or not strict,
        "review_required": False,
    }


@pytest.mark.parametrize(
    "text,expected,choices,answer",
    [
        ("**+06195**", "6195", None, "+06195"),
        ("+06195\n6195\nFinal answer: 0006195.", "6195", None, "+06195"),
        ("-00\n+0\nAnswer: 0.", "0", None, "-00"),
        ("**Billing**\n\nExplanation.", "billing", ("billing",), "Billing"),
        ("Billing\nBILLING\nFinal answer: billing.", "billing", ("billing",), "Billing"),
    ],
)
def test_semantically_equal_repeated_assertions_are_not_conflicts(score, text, expected, choices, answer):
    result = score(text, expected, choices=choices, stop_reason="end_turn")
    assert result["answer"] == answer
    assert result["answer_correct"] is result["passed"] is True
    assert result["format_valid"] is result["review_required"] is False


@pytest.mark.parametrize(
    "text,choices",
    [
        ("+06195\n\nFinal answer: -6195", None),
        ("06195\n6196", None),
        ("06195 or 6196", None),
        ("ASK_FOR_RECEIPT\n\nFinal answer: APPROVE", CHOICES),
        ("Ask_for_receipt\nDENY", CHOICES),
        ("Ask_for_receipt or APPROVE", CHOICES),
    ],
)
def test_semantic_comparison_keeps_conflicting_assertions_unscored(score, text, choices):
    result = score(text, "ask_for_receipt" if choices else "6195", choices=choices, stop_reason="end_turn")
    assert result["answer"] is result["answer_correct"] is None
    assert result["passed"] is False
    assert result["review_required"] is True


def test_wrong_leading_label_stays_wrong_despite_case_and_expected_mention(score):
    result = score(
        "**APPROVE**\n\nI considered ASK_FOR_RECEIPT as an alternative.",
        "ask_for_receipt",
        choices=CHOICES,
        stop_reason="end_turn",
    )
    assert result["answer"] == "APPROVE"
    assert result["answer_correct"] is result["passed"] is result["review_required"] is False


@pytest.mark.parametrize("text", ["The answer is Billing.", "Billing_extra", "billing123", "b\u0131lling"])
def test_case_normalization_does_not_broaden_leading_label_grammar(score, text):
    result = score(text, "billing", choices=("billing",), stop_reason="end_turn")
    assert result["answer"] is result["answer_correct"] is None
    assert result["format_valid"] is result["passed"] is False
    assert result["review_required"] is True


@pytest.mark.parametrize(
    "text,expected,correct",
    [
        ("+" + "0" * 5_000 + "6195", "6195", True),
        ("-0" + "9" * 5_000, "-" + "9" * 5_000, True),
        ("-" + "0" * 5_000, "+0", True),
        ("9" * 5_000, "9" * 4_999 + "8", False),
    ],
)
def test_huge_digit_strings_are_compared_without_integer_conversion(score, text, expected, correct):
    result = score(text, expected, stop_reason="end_turn")
    assert result["answer"] == text
    assert result["answer_correct"] is result["passed"] is correct
    assert result["format_valid"] is True
    assert result["review_required"] is False


def test_expected_normalization_is_bounded_and_never_truncated(score):
    assert score("0", "0" * 32_768, stop_reason="end_turn")["passed"] is True
    with pytest.raises(ValueError, match="expected"):
        score("0", "0" * 32_769, stop_reason="end_turn")


def test_oversized_expected_integer_is_rejected_before_decimal_rendering(score):
    with pytest.raises(ValueError, match="expected exceeds"):
        score("0", 1 << (4 * 32_768), stop_reason="end_turn")


def test_large_expected_integer_can_be_supplied_as_a_decimal_string(score):
    limit = sys.get_int_max_str_digits()
    if not 0 < limit < 32_768:
        pytest.skip("Python has no decimal conversion limit within the scorer's bound")
    with pytest.raises(ValueError, match=r"expected integer.*decimal string"):
        score("0", 10**limit, stop_reason="end_turn")
    decimal = "1" + "0" * limit
    assert score(decimal, decimal, stop_reason="end_turn")["passed"] is True


@pytest.mark.parametrize("expected", ["6195.0", "6_195", "+ 6195", "The answer is 6195"])
def test_semantic_comparison_does_not_coerce_noninteger_expected_strings(score, expected):
    result = score("6195", expected, stop_reason="end_turn")
    assert result["answer"] == "6195"
    assert result["answer_correct"] is result["passed"] is False
    assert result["format_valid"] is True


@pytest.mark.parametrize("stop_reason", [None, "max_tokens", "stop_sequence", "refusal", "guardrail_intervened", "END_TURN"])
def test_incomplete_output_never_passes_even_with_a_correct_bare_answer(score, stop_reason):
    assert score("6195", "6195", stop_reason=stop_reason) == {
        "answer": "6195",
        "answer_correct": True,
        "format_valid": True,
        "complete": False,
        "passed": False,
        "review_required": True,
    }


def test_expected_affects_comparison_only(score):
    text = "**approve**\n\nThe rationale mentions ask_for_receipt."
    right = score(text, "approve", choices=CHOICES, stop_reason="end_turn")
    wrong = score(text, "ask_for_receipt", choices=CHOICES, stop_reason="end_turn")
    assert right["answer_correct"] is right["passed"] is True
    assert wrong["answer_correct"] is wrong["passed"] is False
    for key in ("answer", "format_valid", "complete", "review_required"):
        assert right[key] == wrong[key]


def test_raw_output_and_caller_choices_are_preserved(score):
    row = {"text": "**ask_for_receipt**\n\nexplanation...", "choices": list(CHOICES)}
    before = copy.deepcopy(row)
    result = score(row["text"], "ask_for_receipt", choices=row["choices"], stop_reason="end_turn")
    assert row == before
    result["answer"] = "caller change"
    assert score(row["text"], "ask_for_receipt", choices=row["choices"], stop_reason="end_turn")["passed"]
    assert row == before


def test_overlong_output_requires_review_instead_of_scoring_a_prefix(score):
    text = "6195\n\n" + "rationale " * 4_000 + "\nFinal answer: 6196"
    result = score(text, "6195", stop_reason="end_turn")
    assert result["answer"] is result["answer_correct"] is None
    assert result["passed"] is False
    assert result["review_required"] is True


def test_conflicts_are_checked_through_the_full_supported_text(score):
    prefix = "6195\n\n" + "Rationale. " * 2_000
    text = prefix + "\nFinal answer: 6196"
    assert len(text) < 32_768
    result = score(text, "6195", stop_reason="end_turn")
    assert result["answer"] is result["answer_correct"] is None
    assert result["review_required"] is True


def test_length_limit_is_inclusive_and_never_truncates_for_scoring(score):
    prefix = "6195\n\n"
    text = prefix + "x" * (32_768 - len(prefix))
    assert score(text, "6195", stop_reason="end_turn")["passed"] is True
    result = score(text + "x", "6195", stop_reason="end_turn")
    assert result["answer"] is None
    assert result["passed"] is False
    assert result["review_required"] is True


@pytest.mark.parametrize("choices", ["approve", ["approve or deny"], ["x"] * 33, ["x" * 65]])
def test_invalid_choice_configuration_fails_explicitly(score, choices):
    with pytest.raises(ValueError, match="choices"):
        score("approve", "approve", choices=choices)


@pytest.mark.parametrize("text,expected", [(None, "6195"), ("6195", None), ("6195", True)])
def test_invalid_input_types_fail_explicitly(score, text, expected):
    with pytest.raises(TypeError):
        score(text, expected)


def test_integer_is_not_valid_format_for_a_label_task(score):
    result = score("123", "billing", choices=("billing", "technical", "returns"), stop_reason="end_turn")
    assert result["answer_correct"] is False
    assert result["format_valid"] is False
    assert result["passed"] is False


def test_label_choices_define_format_even_when_integer_value_matches(score):
    result = score("12", "12", choices=CHOICES, stop_reason="end_turn", require_exact_format=True)
    assert result["answer_correct"] is True
    assert result["format_valid"] is False
    assert result["passed"] is False
