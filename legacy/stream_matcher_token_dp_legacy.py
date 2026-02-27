"""
Legacy approach: early StreamingQuranMatcher with token-DP verifier.

This keeps the pre-chain verifier variant for reference and comparison.
"""

from __future__ import annotations

import math
from typing import Optional

from ijaza.asr_tolerance import FUNCTION_WORDS, calculate_asr_similarity
from ijaza.stream_matcher import (
    StreamingQuranMatcher,
    StreamingQuranHit,
    _ApproxMatch,
)
from ijaza.types import QuranVerse


class TokenDPStreamingQuranMatcher(StreamingQuranMatcher):
    """
    Legacy streaming matcher variant using bounded token edit distance.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._token_sub_cost_cache: dict[tuple[str, str], float] = {}

    def reset(self) -> None:
        super().reset()
        self._token_sub_cost_cache = {}

    def _verify_hypotheses(self, *, flush: bool) -> tuple[list[StreamingQuranHit], dict[str, int]]:
        if not self._active:
            return [], {"active": 0, "verified": 0, "full_validated": 0, "emitted": 0}

        ranked = sorted(
            self._active.values(),
            key=lambda h: (
                -(h.anchor_score + h.best_confidence * 8.0),
                -h.last_anchor_token,
                h.candidate_start_token,
                h.verse_id,
            ),
        )
        ranked = ranked[:self.options.beam_size]

        emitted: list[StreamingQuranHit] = []
        emitted_refs_this_pass: set[str] = set()
        verified_count = 0
        full_validated_count = 0
        checked_spans: set[tuple[int, int, int]] = set()

        for h in ranked:
            if verified_count >= self.options.max_verifications_per_chunk:
                break
            if h.last_verified_chunk == self._chunk_index and not flush:
                continue
            if h.anchor_score < self.options.min_anchor_score and h.best_confidence < self.options.min_confidence:
                continue

            verse = self.validator.verse_by_id[h.verse_id]
            verse_wc = self.validator._verse_word_count_by_id.get(h.verse_id, 0)
            if verse_wc <= 0:
                continue

            best_match: Optional[_ApproxMatch] = None
            start_deltas = range(-self.options.max_start_delta, self.options.max_start_delta + 1)
            len_deltas = range(-self.options.max_len_delta, self.options.max_len_delta + 1)

            for start_delta in start_deltas:
                start_global = h.candidate_start_token + start_delta
                start_local = start_global - self._buffer_start_token
                if start_local < 0 or start_local >= len(self._buffer_words):
                    continue

                for len_delta in len_deltas:
                    candidate_len = verse_wc + len_delta
                    if candidate_len < self.options.min_words or candidate_len > self.options.max_words:
                        continue

                    end_local = start_local + candidate_len
                    if end_local <= start_local:
                        continue
                    if end_local > len(self._buffer_words):
                        if not flush:
                            continue
                        end_local = len(self._buffer_words)
                        if end_local - start_local < self.options.min_words:
                            continue

                    end_global = self._buffer_start_token + end_local
                    span_key = (h.verse_id, start_global, end_global)
                    if span_key in checked_spans:
                        continue
                    checked_spans.add(span_key)

                    verified_count += 1
                    approx = self._approx_verify_against_verse(
                        self._buffer_norm_words[start_local:end_local],
                        verse,
                    )
                    if approx is None or approx.approx_confidence < self.options.min_approx_confidence:
                        continue

                    if (
                        best_match is None
                        or approx.approx_confidence > best_match.approx_confidence
                        or (
                            approx.approx_confidence == best_match.approx_confidence
                            and (end_local - start_local) > (best_match.end_local - best_match.start_local)
                        )
                    ):
                        best_match = _ApproxMatch(
                            start_local=start_local,
                            end_local=end_local,
                            approx_confidence=approx.approx_confidence,
                            match_type=approx.match_type,
                        )

            h.last_verified_chunk = self._chunk_index
            if best_match is None:
                continue

            h.best_confidence = max(h.best_confidence, best_match.approx_confidence)
            start_local = best_match.start_local
            end_local = best_match.end_local
            start_global = self._buffer_start_token + start_local
            end_global = self._buffer_start_token + end_local
            touches_end = end_local >= len(self._buffer_words)

            if touches_end and not flush and h.chunk_span < self.options.max_chunk_span:
                continue

            if full_validated_count >= self.options.max_full_validations_per_chunk and not flush:
                continue

            window_text = " ".join(self._buffer_words[start_local:end_local])
            result = self.validator._validate_against_verse(window_text, verse)
            full_validated_count += 1
            if not result.is_valid or result.confidence < self.options.min_confidence:
                continue

            ref = result.reference or ""
            if not flush and ref and ref in emitted_refs_this_pass:
                continue

            emit_key = (ref, start_global)
            if emit_key in self._emitted_keys:
                continue
            if self._is_duplicate_reference_span(ref, start_global, end_global):
                continue

            self._emitted_keys.add(emit_key)
            self._record_emitted_reference_span(ref, start_global, end_global)
            if ref:
                emitted_refs_this_pass.add(ref)

            verse_obj = result.matched_verse
            emitted.append(StreamingQuranHit(
                original_text=window_text,
                start_token=start_global,
                end_token=end_global,
                correct_text=verse_obj.text if verse_obj else "",
                reference=ref,
                confidence=result.confidence,
                verses=[verse_obj] if verse_obj else [],
                needs_correction=(result.match_type != "exact"),
                translations=result.translations,
            ))

            self._active.pop((h.verse_id, h.candidate_start_token), None)

        return emitted, {
            "active": len(self._active),
            "verified": verified_count,
            "full_validated": full_validated_count,
            "emitted": len(emitted),
        }

    def _approx_verify_against_verse(
        self,
        normalized_input_tokens: list[str],
        verse: QuranVerse,
    ) -> Optional[_ApproxMatch]:
        if not normalized_input_tokens:
            return None

        verse_tokens = list(self.validator._normalized_verse_words_by_id.get(verse.id, ()))
        if not verse_tokens:
            return None

        if normalized_input_tokens == verse_tokens:
            return _ApproxMatch(0, len(normalized_input_tokens), 0.95, "normalized")

        max_len = max(len(normalized_input_tokens), len(verse_tokens))
        max_cost = max_len * max(0.0, 1.0 - self.options.min_confidence)
        band = max(abs(len(normalized_input_tokens) - len(verse_tokens)), int(math.ceil(max_cost)) + 2)
        distance = self._bounded_token_edit_distance(
            normalized_input_tokens,
            verse_tokens,
            max_cost=max_cost,
            band=band,
        )
        if distance is None:
            return None

        confidence = max(0.0, 1.0 - distance / max_len)
        if confidence < self.options.min_approx_confidence:
            return None
        return _ApproxMatch(0, len(normalized_input_tokens), confidence, "fuzzy")

    def _bounded_token_edit_distance(
        self,
        input_tokens: list[str],
        verse_tokens: list[str],
        *,
        max_cost: float,
        band: int,
    ) -> Optional[float]:
        m = len(input_tokens)
        n = len(verse_tokens)
        if m == 0 and n == 0:
            return 0.0
        if abs(m - n) > band:
            return None

        inf = max_cost + 10.0
        prev_row = [inf] * (n + 1)
        curr_row = [inf] * (n + 1)
        prev_row[0] = 0.0
        for j in range(1, n + 1):
            prev_row[j] = prev_row[j - 1] + self._token_del_cost(verse_tokens[j - 1])

        for i in range(1, m + 1):
            for j in range(n + 1):
                curr_row[j] = inf

            j_lo = max(0, i - band)
            j_hi = min(n, i + band)
            if j_lo == 0:
                curr_row[0] = prev_row[0] + self._token_ins_cost(input_tokens[i - 1])

            row_min = curr_row[0] if j_lo == 0 else inf
            for j in range(max(1, j_lo), j_hi + 1):
                ins_cost = curr_row[j - 1] + self._token_del_cost(verse_tokens[j - 1])
                del_cost = prev_row[j] + self._token_ins_cost(input_tokens[i - 1])
                sub_cost = prev_row[j - 1] + self._token_sub_cost(input_tokens[i - 1], verse_tokens[j - 1])
                val = min(ins_cost, del_cost, sub_cost)
                curr_row[j] = val
                if val < row_min:
                    row_min = val

            if row_min > max_cost:
                return None
            prev_row, curr_row = curr_row, prev_row

        result = prev_row[n]
        if result > max_cost:
            return None
        return result

    @staticmethod
    def _token_ins_cost(token: str) -> float:
        return 0.2 if token in FUNCTION_WORDS else 1.0

    @staticmethod
    def _token_del_cost(token: str) -> float:
        return 0.2 if token in FUNCTION_WORDS else 1.0

    def _token_sub_cost(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        key = (a, b) if a <= b else (b, a)
        cached = self._token_sub_cost_cache.get(key)
        if cached is not None:
            return cached

        if self.options.asr_tolerant:
            cost = 1.0 - calculate_asr_similarity(a, b)
        else:
            cost = 1.0
        self._token_sub_cost_cache[key] = cost
        return cost
