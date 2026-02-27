"""
Legacy approach: indexed n-gram anchor scanner.

This snapshots the previous default strategy that used indexed n-gram anchors
plus targeted candidate window validation.
"""

from __future__ import annotations

from ijaza.validator import (
    QuranValidator,
    _iter_word_ngrams_with_pos,
    _normalize_scan_token,
)


def scan_for_verses_indexed_legacy(
    validator: QuranValidator,
    text: str,
    min_words: int = 3,
    max_words: int = 50,
    confidence_threshold: float = 0.85,
) -> list[dict]:
    """
    Legacy hit-driven scanner from the indexed n-gram phase.
    """
    words = text.split()
    if len(words) < min_words:
        return []

    normalized_input_words = tuple(_normalize_scan_token(w) for w in words)
    total_words = len(words)

    anchor_scores: dict[tuple[int, int], float] = {}
    saw_strong_hits = False
    if total_words >= 3:
        n_plan = ((3, 6.0), (2, 2.5), (1, 0.5))
    elif total_words == 2:
        n_plan = ((2, 2.0), (1, 0.6))
    else:
        n_plan = ((1, 1.0),)

    for n, weight in n_plan:
        if total_words < n:
            continue

        local_hits = 0
        for input_pos, gram in _iter_word_ngrams_with_pos(normalized_input_words, n):
            if not gram.strip():
                continue
            for verse_id, verse_pos in validator._ngram_pos_index_by_n[n].get(gram, ()):
                verse_wc = validator._verse_word_count_by_id.get(verse_id, 0)
                if verse_wc < max(1, min_words - 2) or verse_wc > max_words + 6:
                    continue

                base_start = input_pos - verse_pos
                if base_start < 0 or base_start >= total_words:
                    continue

                min_candidate_len = max(min_words, verse_wc - 3)
                if base_start + min_candidate_len > total_words:
                    continue

                key = (base_start, verse_id)
                anchor_scores[key] = anchor_scores.get(key, 0.0) + weight
                local_hits += 1

        if local_hits and n >= 2:
            saw_strong_hits = True
            if n == 2:
                break

    if not anchor_scores:
        return []

    if saw_strong_hits:
        filtered_anchor_scores: dict[tuple[int, int], float] = {}
        for key, score in anchor_scores.items():
            if score >= 1.0:
                filtered_anchor_scores[key] = score
        if filtered_anchor_scores:
            anchor_scores = filtered_anchor_scores

    ranked_anchors = sorted(
        anchor_scores.items(),
        key=lambda item: (
            -item[1],
            item[0][0],
            abs(validator._verse_word_count_by_id.get(item[0][1], 0) - total_words),
            item[0][1],
        ),
    )

    max_anchors = 32 if validator.options.asr_tolerant else 20
    ranked_anchors = ranked_anchors[:max_anchors]

    if validator.options.asr_tolerant:
        start_deltas = (0, -1, 1, -2, 2)
        len_deltas = (0, -1, 1, -2, 2, -3, 3)
    else:
        start_deltas = (0, -1, 1)
        len_deltas = (0, -1, 1)

    checked_windows: set[tuple[int, int, int]] = set()
    provisional: list[dict] = []

    for (base_start, verse_id), anchor_score in ranked_anchors:
        verse = validator.verse_by_id[verse_id]
        verse_wc = validator._verse_word_count_by_id.get(verse_id, 0) or len(
            validator._normalized_verse_words_by_id.get(verse_id, ())
        )
        if verse_wc <= 0:
            continue

        best_entry = None
        stop_anchor = False
        for start_delta in start_deltas:
            if stop_anchor:
                break
            start = base_start + start_delta
            if start < 0 or start >= total_words:
                continue

            for len_delta in len_deltas:
                candidate_len = verse_wc + len_delta
                if candidate_len < min_words or candidate_len > max_words:
                    continue

                end = start + candidate_len
                if end > total_words or end <= start:
                    continue

                window_key = (start, end, verse_id)
                if window_key in checked_windows:
                    continue
                checked_windows.add(window_key)

                window_text = " ".join(words[start:end])
                result = validator._validate_against_verse(window_text, verse)
                if not result.is_valid or result.confidence < confidence_threshold:
                    continue

                candidate_entry = {
                    "start_word": start,
                    "end_word": end,
                    "result": result,
                    "anchor_score": anchor_score,
                }

                if best_entry is None:
                    best_entry = candidate_entry
                else:
                    prev = best_entry["result"]
                    curr = result
                    prev_span = best_entry["end_word"] - best_entry["start_word"]
                    curr_span = end - start
                    if (
                        curr.confidence,
                        anchor_score,
                        curr_span,
                    ) > (
                        prev.confidence,
                        best_entry["anchor_score"],
                        prev_span,
                    ):
                        best_entry = candidate_entry

                if result.match_type in ("exact", "normalized") and result.confidence >= 0.95:
                    stop_anchor = True
                    break

        if best_entry is not None:
            provisional.append(best_entry)

    if not provisional:
        return []

    provisional.sort(
        key=lambda entry: (
            entry["start_word"],
            -entry["result"].confidence,
            -(entry["end_word"] - entry["start_word"]),
            -entry["anchor_score"],
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
