"""
Legacy approach: exhaustive sliding-window scan with global validate().

This is the slowest historical approach. It checks every candidate window
with `QuranValidator.validate()` and then greedily selects non-overlapping
matches.
"""

from __future__ import annotations

from ijaza.validator import QuranValidator


def scan_for_verses_sliding_window_fullscan(
    validator: QuranValidator,
    text: str,
    min_words: int = 3,
    max_words: int = 50,
    confidence_threshold: float = 0.85,
) -> list[dict]:
    """Exhaustive legacy scanner based on validate() per window."""
    words = text.split()
    if len(words) < min_words:
        return []

    provisional: list[dict] = []
    total_words = len(words)
    checked_windows: set[tuple[int, int]] = set()

    for start in range(total_words):
        for end in range(
            start + min_words,
            min(start + max_words + 1, total_words + 1),
        ):
            if end <= start:
                continue
            window_key = (start, end)
            if window_key in checked_windows:
                continue
            checked_windows.add(window_key)

            window_text = " ".join(words[start:end])
            result = validator.validate(window_text)
            if not result.is_valid or result.confidence < confidence_threshold:
                continue

            provisional.append({
                "start_word": start,
                "end_word": end,
                "result": result,
            })

    if not provisional:
        return []

    provisional.sort(
        key=lambda entry: (
            entry["start_word"],
            -entry["result"].confidence,
            -(entry["end_word"] - entry["start_word"]),
        )
    )

    selected: list[dict] = []
    covered: set[int] = set()
    seen_matches: set[tuple[int, int, str]] = set()

    for entry in provisional:
        start = int(entry["start_word"])
        end = int(entry["end_word"])
        result = entry["result"]

        match_key = (start, end, result.reference or "")
        if match_key in seen_matches:
            continue
        if any(pos in covered for pos in range(start, end)):
            continue

        verse = result.matched_verse
        start_char = len(" ".join(words[:start])) + (1 if start > 0 else 0)
        end_char = start_char + len(" ".join(words[start:end]))

        selected.append({
            "original_text": " ".join(words[start:end]),
            "correct_text": verse.text if verse else "",
            "reference": result.reference or "",
            "confidence": result.confidence,
            "start_pos": start_char,
            "end_pos": end_char,
            "verses": [verse] if verse else [],
            "needs_correction": result.match_type != "exact",
            "translations": result.translations,
        })

        seen_matches.add(match_key)
        for pos in range(start, end):
            covered.add(pos)

    return selected
