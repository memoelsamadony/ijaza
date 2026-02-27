"""
Legacy Quran-scanning approaches archived for reference.

These modules preserve older strategies that were tried during optimization
work. They are intentionally separate from the main package code path.
"""

from .sliding_window_fullscan_legacy import scan_for_verses_sliding_window_fullscan
from .indexed_ngram_scan_legacy import scan_for_verses_indexed_legacy
from .streaming_scanner_legacy import (
    LegacyStreamingScanner,
    LegacyStreamingScannerOptions,
    LegacyStreamingResult,
    LegacyScannedVerse,
    LegacyPartialVerse,
)
from .stream_matcher_token_dp_legacy import TokenDPStreamingQuranMatcher

__all__ = [
    "scan_for_verses_sliding_window_fullscan",
    "scan_for_verses_indexed_legacy",
    "LegacyStreamingScanner",
    "LegacyStreamingScannerOptions",
    "LegacyStreamingResult",
    "LegacyScannedVerse",
    "LegacyPartialVerse",
    "TokenDPStreamingQuranMatcher",
]
