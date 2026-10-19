"""Cache metric extraction and display.

For cost comparisons, use workshop_utils.pricing.calculate_cost with an exact
model/profile and normalized per-request usage, preserving cache write TTLs.
"""

from __future__ import annotations

from .cache_metrics import extract_cache_metrics, print_cache_metrics

__all__ = [
    "extract_cache_metrics",
    "print_cache_metrics",
]
