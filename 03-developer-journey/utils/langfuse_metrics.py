"""
Langfuse metrics helper - Query traces directly to get accurate metrics.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

METRICS_FILE = Path(".lab_metrics.json")

# Global list to collect metrics across invocations
_collected_metrics = []


def clear_metrics():
    """Clear all collected metrics."""
    global _collected_metrics
    _collected_metrics = []


def collect_metric(metrics: dict, test_name: str = ""):
    """Add a metric result to the collection."""
    global _collected_metrics
    entry = {"test_name": test_name, **metrics}
    _collected_metrics.append(entry)


def get_collected_metrics():
    """Return all collected metrics."""
    return _collected_metrics.copy()


def print_metrics_table():
    """Print all collected metrics as a formatted table."""
    global _collected_metrics

    if not _collected_metrics:
        print("No metrics collected yet.")
        return

    # Try to use pandas for nicer display
    try:
        import pandas as pd

        rows = []
        for m in _collected_metrics:
            if "error" not in m:
                rows.append(
                    {
                        "Test": m.get("test_name", ""),
                        "Latency": f"{m.get('latency_seconds', 0):.2f}s" if m.get("latency_seconds") else "N/A",
                        "Cost": f"${m.get('cost_usd', 0):.4f}",
                        "Input": f"{m.get('input_tokens', 0):,}",
                        "Output": f"{m.get('output_tokens', 0):,}",
                        "Cache Read Tokens": f"{m.get('cache_read_tokens', 0):,}",
                        "Cache Write Tokens": f"{m.get('cache_write_tokens', 0):,}",
                    }
                )
            else:
                rows.append(
                    {
                        "Test": m.get("test_name", ""),
                        "Latency": "ERROR",
                        "Cost": "-",
                        "Input": "-",
                        "Output": "-",
                        "Cache Read Tokens": "-",
                        "Cache Write Tokens": "-",
                    }
                )

        df = pd.DataFrame(rows)

        # Calculate totals
        total_cost = sum(m.get("cost_usd", 0) for m in _collected_metrics if "error" not in m)
        total_input = sum(m.get("input_tokens", 0) for m in _collected_metrics if "error" not in m)
        total_output = sum(m.get("output_tokens", 0) for m in _collected_metrics if "error" not in m)
        total_cache_read = sum(m.get("cache_read_tokens", 0) for m in _collected_metrics if "error" not in m)
        total_cache_write = sum(m.get("cache_write_tokens", 0) for m in _collected_metrics if "error" not in m)
        avg_latency = sum(m.get("latency_seconds", 0) or 0 for m in _collected_metrics if "error" not in m) / max(
            len([m for m in _collected_metrics if "error" not in m]), 1
        )

        print("\n" + "=" * 105)
        print("                                  METRICS SUMMARY")
        print("=" * 105)
        print(df.to_string(index=False))
        print("-" * 105)
        print(
            f"  TOTALS: Latency(avg): {avg_latency:.2f}s | Cost: ${total_cost:.4f} | Input: {total_input:,} | Output: {total_output:,}"
        )
        print(f"          Cache Read Tokens: {total_cache_read:,} | Cache Write Tokens: {total_cache_write:,}")
        print("=" * 105 + "\n")

        return df

    except ImportError:
        # Fallback to manual formatting
        print("\n" + "=" * 120)
        print("                                           METRICS SUMMARY")
        print("=" * 120)
        print(
            f"{'Test':<20} {'Latency':>8} {'Cost':>10} {'Input':>10} {'Output':>10} {'Cache Read Tokens':>18} {'Cache Write Tokens':>19}"
        )
        print("-" * 120)

        total_cost = 0
        total_input = 0
        total_output = 0
        total_cache_read = 0
        total_cache_write = 0
        total_latency = 0
        count = 0

        for m in _collected_metrics:
            test = m.get("test_name", "")[:19]
            if "error" not in m:
                lat = f"{m.get('latency_seconds', 0):.2f}s" if m.get("latency_seconds") else "N/A"
                cost = f"${m.get('cost_usd', 0):.4f}"
                inp = f"{m.get('input_tokens', 0):,}"
                out = f"{m.get('output_tokens', 0):,}"
                c_read = f"{m.get('cache_read_tokens', 0):,}"
                c_write = f"{m.get('cache_write_tokens', 0):,}"

                total_cost += m.get("cost_usd", 0)
                total_input += m.get("input_tokens", 0)
                total_output += m.get("output_tokens", 0)
                total_cache_read += m.get("cache_read_tokens", 0)
                total_cache_write += m.get("cache_write_tokens", 0)
                total_latency += m.get("latency_seconds", 0) or 0
                count += 1
            else:
                lat, cost, inp, out, c_read, c_write = "ERROR", "-", "-", "-", "-", "-"

            print(f"{test:<20} {lat:>8} {cost:>10} {inp:>10} {out:>10} {c_read:>18} {c_write:>19}")

        print("-" * 120)
        avg_lat = total_latency / max(count, 1)
        print(
            f"{'TOTALS':<20} {f'{avg_lat:.2f}s':>8} {f'${total_cost:.4f}':>10} {f'{total_input:,}':>10} {f'{total_output:,}':>10} {f'{total_cache_read:,}':>18} {f'{total_cache_write:,}':>19}"
        )
        print("=" * 120 + "\n")


def trace_reader():
    """Read-only SDK client; does not create or shut down a tracing provider."""
    from langfuse.api import LangfuseAPI

    return LangfuseAPI(
        base_url=os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        username=os.environ["LANGFUSE_PUBLIC_KEY"],
        password=os.environ["LANGFUSE_SECRET_KEY"],
    )


def get_latest_trace_metrics(
    agent_name="customer-support-v1-baseline", wait_seconds=5, max_retries=24, timeout_seconds=120, *, session_id=None
):
    """Read this invocation's generations; keep the old call name for notebook compatibility."""
    if not session_id:
        raise ValueError("Supply the Runtime session ID; newest-project-trace lookup is not reliable")
    reader = trace_reader()
    deadline = time.monotonic() + timeout_seconds
    for attempt in range(max_retries):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if wait_seconds:
            time.sleep(min(wait_seconds, remaining))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        options = {"timeout_in_seconds": min(remaining, 30), "max_retries": 0}
        traces = reader.trace.list(session_id=session_id, limit=100, request_options=options).data
        # List summaries can lag behind the complete record during ingestion.
        matches = []
        for summary in traces:
            full = reader.trace.get(summary.id, request_options=options)
            if agent_name in (full.name or "") or any(
                agent_name in (observation.name or "") for observation in full.observations
            ):
                matches.append(full)
        if len(matches) > 1:
            raise ValueError("Multiple matching traces for this invocation; inspect the session")
        if not matches:
            if attempt == 0:
                print("Waiting for this invocation to arrive in Langfuse...")
            continue
        trace = matches[0]
        generations = [o.dict() for o in trace.observations if o.type == "GENERATION"]
        if not generations or any(
            not g.get("usageDetails") or g.get("calculatedTotalCost") is None for g in generations
        ):
            continue
        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cost_usd": 0.0,
        }
        for generation in generations:
            usage = generation.get("usageDetails") or {}
            if not usage or generation.get("calculatedTotalCost") is None:
                raise ValueError("Generation has missing usage/pricing; it is not a free call")
            totals["input_tokens"] += usage.get("input", 0)
            totals["output_tokens"] += usage.get("output", 0)
            totals["cache_read_tokens"] += usage.get("cache_read_input_tokens", 0)
            totals["cache_write_tokens"] += sum(v for k, v in usage.items() if k.startswith("cache_write_"))
            totals["cost_usd"] += generation["calculatedTotalCost"]
        return {
            **totals,
            "trace_id": trace.id,
            "trace_name": trace.name,
            "session_id": session_id,
            "latency_seconds": trace.latency,
            "model_calls": len(generations),
            "total_tokens": sum(
                totals[k] for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
            ),
            "langfuse_url": os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com") + trace.html_path,
        }
    raise TimeoutError(f"No complete trace for {agent_name}, session {session_id}; do not evaluate missing data")


def calculate_totals_from_collected() -> dict:
    """Calculate totals from collected metrics."""
    global _collected_metrics
    metrics = [m for m in _collected_metrics if "error" not in m]

    if not metrics:
        return {
            "total_cost": 0,
            "avg_latency": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_read_tokens": 0,
            "total_cache_write_tokens": 0,
        }

    latencies = [m.get("latency_seconds", 0) or 0 for m in metrics]
    return {
        "total_cost": sum(m.get("cost_usd", 0) for m in metrics),
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
        "total_input_tokens": sum(m.get("input_tokens", 0) for m in metrics),
        "total_output_tokens": sum(m.get("output_tokens", 0) for m in metrics),
        "total_cache_read_tokens": sum(m.get("cache_read_tokens", 0) for m in metrics),
        "total_cache_write_tokens": sum(m.get("cache_write_tokens", 0) for m in metrics),
    }


def save_metrics(version: str) -> None:
    """Save current collected metrics totals for cross-notebook comparison.

    Saves to .lab_metrics.json in the working directory so later notebooks
    can load previous lab results automatically.
    """
    totals = calculate_totals_from_collected()

    data = {}
    if METRICS_FILE.exists():
        data = json.loads(METRICS_FILE.read_text())

    data[version] = totals
    METRICS_FILE.write_text(json.dumps(data, indent=2))
    print(f"Metrics saved as '{version}' → {METRICS_FILE}")


def load_metrics(version: str) -> dict:
    """Load saved metrics from a previous lab for comparison.

    Returns a dict with: total_cost, avg_latency, total_input_tokens,
    total_output_tokens, total_cache_read_tokens, total_cache_write_tokens.
    """
    empty = {
        "total_cost": 0,
        "avg_latency": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cache_read_tokens": 0,
        "total_cache_write_tokens": 0,
    }

    if not METRICS_FILE.exists():
        print("No saved metrics found. Run the previous lab first.")
        return empty

    data = json.loads(METRICS_FILE.read_text())
    if version not in data:
        available = ", ".join(data.keys()) or "none"
        print(f"No metrics for '{version}'. Available: {available}")
        return empty

    metrics = data[version]
    print(
        f"Loaded '{version}' metrics: cost=${metrics['total_cost']:.4f}, "
        f"latency={metrics['avg_latency']:.2f}s, "
        f"input={metrics['total_input_tokens']:,}, output={metrics['total_output_tokens']:,}"
    )
    return metrics


def print_comparison(
    prev_name: str,
    curr_name: str,
    prev_cost: float,
    prev_latency: float,
    prev_input_tokens: int,
    prev_output_tokens: int,
    curr_cost: float | None = None,
    curr_latency: float | None = None,
    curr_input_tokens: int | None = None,
    curr_output_tokens: int | None = None,
) -> None:
    """
    Print a comparison table between two versions.

    If curr_* values are not provided, they are calculated from collected metrics.

    Args:
        prev_name: Name of previous version (e.g., "v1 (Baseline)")
        curr_name: Name of current version (e.g., "v2 (Quick Wins)")
        prev_cost: Total cost from previous version
        prev_latency: Average latency from previous version
        prev_input_tokens: Total input tokens from previous version
        prev_output_tokens: Total output tokens from previous version
        curr_*: Current version metrics (optional, calculated if not provided)
    """
    # Calculate current metrics from collected if not provided
    if curr_cost is None or curr_latency is None:
        totals = calculate_totals_from_collected()
        curr_cost = curr_cost if curr_cost is not None else totals["total_cost"]
        curr_latency = curr_latency if curr_latency is not None else totals["avg_latency"]
        curr_input_tokens = curr_input_tokens if curr_input_tokens is not None else totals["total_input_tokens"]
        curr_output_tokens = curr_output_tokens if curr_output_tokens is not None else totals["total_output_tokens"]

    # Header
    print("=" * 70)
    print(f"  {prev_name.upper()} vs {curr_name.upper()} COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<20} {prev_name:>18} {curr_name:>18} {'Change':>12}")
    print("-" * 70)

    has_prev_metrics = prev_cost > 0 and prev_latency > 0

    # Cost comparison
    if prev_cost > 0:
        cost_change = ((curr_cost - prev_cost) / prev_cost) * 100
        cost_str = f"{cost_change:+.1f}%"
    else:
        cost_change = 0
        cost_str = "N/A"
    print(f"{'Total Cost':<20} ${prev_cost:>17.4f} ${curr_cost:>17.4f} {cost_str:>12}")

    # Latency comparison
    if prev_latency > 0:
        latency_change = ((curr_latency - prev_latency) / prev_latency) * 100
        latency_str = f"{latency_change:+.1f}%"
    else:
        latency_change = 0
        latency_str = "N/A"
    print(f"{'Avg Latency (s)':<20} {prev_latency:>18.2f} {curr_latency:>18.2f} {latency_str:>12}")

    # Input tokens comparison
    if prev_input_tokens > 0:
        input_change = ((curr_input_tokens - prev_input_tokens) / prev_input_tokens) * 100
        input_str = f"{input_change:+.1f}%"
    else:
        input_str = "N/A"
    print(f"{'Input Tokens':<20} {prev_input_tokens:>18,} {curr_input_tokens:>18,} {input_str:>12}")

    # Output tokens comparison
    if prev_output_tokens > 0:
        output_change = ((curr_output_tokens - prev_output_tokens) / prev_output_tokens) * 100
        output_str = f"{output_change:+.1f}%"
    else:
        output_str = "N/A"
    print(f"{'Output Tokens':<20} {prev_output_tokens:>18,} {curr_output_tokens:>18,} {output_str:>12}")

    print("=" * 70)
    if has_prev_metrics:
        # Report the magnitude with the matching direction word, so the sign and
        # the label never disagree (e.g. a latency rise reads "21.8% latency increase",
        # never "-21.8% latency increase").
        cost_word = "reduction" if cost_change < 0 else "increase"
        latency_word = "improvement" if latency_change < 0 else "increase"
        print(f"\nResult: {abs(cost_change):.1f}% cost {cost_word}, {abs(latency_change):.1f}% latency {latency_word}")
    else:
        print(f"\n⚠️  Enter your {prev_name} metrics above to see the comparison")


def print_metrics(metrics: dict, test_name: str = "") -> None:
    """Pretty print metrics from get_latest_trace_metrics."""
    host = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    print("\n" + "=" * 60)
    print("                    LANGFUSE METRICS")
    print("=" * 60)

    if test_name:
        print(f"  Test:          {test_name}")

    if "error" in metrics:
        print("  Status:        ERROR")
        print(f"  Message:       {metrics['error']}")
        print("-" * 60)
        print(f"  Dashboard:     {host}")
    else:
        print(f"  Trace ID:      {metrics['trace_id']}")
        print("-" * 60)
        if metrics.get("latency_seconds") is not None:
            print(f"  Latency:       {metrics['latency_seconds']:.2f}s")
        else:
            print("  Latency:       N/A")
        print(f"  Cost:          ${metrics.get('cost_usd', 0):.6f}")
        print(f"  Input tokens:  {metrics.get('input_tokens', 0):,}")
        print(f"  Output tokens: {metrics.get('output_tokens', 0):,}")
        print(f"  Total tokens:  {metrics.get('total_tokens', 0):,}")
        print("-" * 60)
        print(f"  Cache read tokens:   {metrics.get('cache_read_tokens', 0):,}")
        print(f"  Cache write tokens:  {metrics.get('cache_write_tokens', 0):,}")
        print("-" * 60)
        print(f"  View trace:    {metrics.get('langfuse_url', host)}")

    print("=" * 60 + "\n")
