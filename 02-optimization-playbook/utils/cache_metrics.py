"""Extract and display cache token meters.

For cost comparisons, use workshop_utils.pricing.calculate_cost with an exact
model/profile and normalized per-request usage, preserving cache write TTLs.
"""

from __future__ import annotations


def extract_cache_metrics(response):
    """
    Extract cache metrics from Bedrock response.

    Args:
        response: Response from bedrock_runtime.converse()

    Returns:
        dict with keys:
            - input_tokens: Regular input tokens
            - output_tokens: Output tokens generated
            - cache_write: Cache write input tokens (first occurrence)
            - cache_read: Cache read input tokens (cached content reused)
    """
    usage = response.get("usage", {})

    metrics = {
        "input_tokens": usage.get("inputTokens", 0),
        "output_tokens": usage.get("outputTokens", 0),
        "cache_write": usage.get("cacheWriteInputTokens", 0),
        "cache_read": usage.get("cacheReadInputTokens", 0),
    }

    return metrics


def print_cache_metrics(metrics, request_num=None):
    """
    Pretty print cache metrics with color-coded status.

    Args:
        metrics: Dict from extract_cache_metrics()
        request_num: Optional request number to display
    """
    header = f"Request {request_num}" if request_num else "Cache Metrics"

    print(f"\n{'=' * 60}")
    print(f"{header}")
    print(f"{'=' * 60}")
    print(f"Input tokens:       {metrics['input_tokens']:,}")
    print(f"Output tokens:      {metrics['output_tokens']:,}")
    print(f"Cache write tokens: {metrics['cache_write']:,}")
    print(f"Cache read tokens:  {metrics['cache_read']:,}")

    print(f"{'=' * 60}\n")  # Extra newline for spacing
