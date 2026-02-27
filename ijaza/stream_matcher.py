"""
Experimental streaming Quran matcher.

This module implements a stateful, anchor-driven matcher for chunked ASR text.
It keeps lightweight hypotheses across chunks instead of rescanning each chunk
from scratch with many sliding windows.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .types import QuranVerse, ValidatorOptions
from .validator import (
    QuranValidator,
    _iter_word_ngrams_with_pos,
    _normalize_scan_token,
)
if TYPE_CHECKING:
    from .translations import TranslationProvider


@dataclass
class StreamingQuranMatcherOptions:
    """Configuration for the experimental streaming matcher."""

    min_confidence: float = 0.85
    min_words: int = 3
    max_words: int = 50
    asr_tolerant: bool = True

    # Anchor search
    anchor_ngram_sizes: tuple[int, ...] = (3, 2)
    beam_size: int = 48
    max_verifications_per_chunk: int = 64
    extra_verifications_dense: int = 192
    max_full_validations_per_chunk: int = 8
    extra_full_validations_dense: int = 12
    dense_active_ratio: float = 0.70
    max_emit_window_candidates: int = 8
    min_anchor_score: float = 2.0
    # Candidate gate for the fast verifier before final validation.
    # Keep lower than min_confidence to avoid pruning true positives too early.
    min_approx_confidence: float = 0.70
    max_hits_per_hypothesis: int = 64

    # Candidate window expansion around anchored verse start/length
    max_start_delta: int = 2
    max_len_delta: int = 3
    adaptive_expand_coverage_threshold: float = 0.35
    adaptive_expand_support_threshold: float = 0.70
    adaptive_extra_start_delta: int = 2
    adaptive_extra_len_delta: int = 3
    adaptive_min_approx_drop: float = 0.10

    # Fuzzy emission guard to reduce false positives.
    short_verse_words: int = 8
    min_fuzzy_lexical_overlap: float = 0.45
    min_fuzzy_lexical_overlap_short: float = 0.56

    # State retention
    max_buffer_words: int = 96
    max_chunk_span: int = 3
    max_anchor_age_tokens: int = 48
    duplicate_ref_window_tokens: int = 32


@dataclass
class StreamingQuranHit:
    """A detected Quran verse in the streaming input."""

    original_text: str
    start_token: int
    end_token: int
    correct_text: str
    reference: str
    confidence: float
    verses: list[QuranVerse] = field(default_factory=list)
    needs_correction: bool = False
    translations: dict[str, str] = field(default_factory=dict)


@dataclass
class ActiveHypothesis:
    """Anchor-seeded candidate verse hypothesis kept across chunks."""

    verse_id: int
    candidate_start_token: int
    anchor_score: float
    anchor_hits: int
    first_chunk_seen: int
    last_chunk_seen: int
    last_anchor_token: int
    best_confidence: float = 0.0
    last_verified_chunk: int = -1
    hits: list[tuple[int, int, int, float]] = field(default_factory=list)

    @property
    def chunk_span(self) -> int:
        return self.last_chunk_seen - self.first_chunk_seen + 1


@dataclass
class StreamingQuranMatcherResult:
    """Result of processing one chunk."""

    complete_verses: list[StreamingQuranHit] = field(default_factory=list)
    partial_hypotheses: int = 0
    consumed_text: str = ""
    stats: dict[str, int] = field(default_factory=dict)


@dataclass
class _ApproxMatch:
    """Internal candidate span match scored by the fast verifier."""

    start_local: int
    end_local: int
    approx_confidence: float
    match_type: str
    lexical_overlap: float = 0.0
    anchor_mass_ratio: float = 0.0
    chain_coverage: float = 0.0


@dataclass
class _ChainSummary:
    """Monotonic anchor chain summary for a hypothesis."""

    chain_hits: list[tuple[int, int, int, float]]
    chain_score: float
    coverage_ratio: float
    support_ratio: float
    est_start_global: int
    span_len_bias: int
    min_input_global: int
    max_input_end_global: int
    min_verse_pos: int
    max_verse_end_pos: int
    diag_min: int
    diag_max: int


class StreamingQuranMatcher:
    """
    Experimental online matcher for Quran verses in ASR chunks.

    Strategy:
    - Build hashed n-gram anchor hits against verse token positions
    - Maintain candidate hypotheses across chunks
    - Verify only a bounded set of candidate spans each chunk
    """

    def __init__(
        self,
        options: Optional[StreamingQuranMatcherOptions] = None,
        translation_provider: Optional['TranslationProvider'] = None,
    ):
        if options is None:
            options = StreamingQuranMatcherOptions()
        self.options = options

        validator_opts = ValidatorOptions(
            fuzzy_threshold=max(0.6, options.min_confidence * 0.9),
            include_partial=False,
            asr_tolerant=options.asr_tolerant,
        )
        self.validator = QuranValidator(
            validator_opts,
            translation_provider=translation_provider,
        )
        self._translation_provider = translation_provider

        self._buffer_words: list[str] = []
        self._buffer_norm_words: list[str] = []
        self._buffer_start_token: int = 0
        self._stream_token_count: int = 0
        self._chunk_index: int = 0

        self._active: dict[tuple[int, int], ActiveHypothesis] = {}
        self._emitted_keys: set[tuple[str, int]] = set()
        self._emitted_spans_by_ref: dict[str, list[tuple[int, int]]] = {}

    def process_chunk(self, text: str) -> StreamingQuranMatcherResult:
        """Process one ASR chunk and emit complete verse matches."""
        self._chunk_index += 1
        chunk_words = text.split()
        if not chunk_words:
            return StreamingQuranMatcherResult(consumed_text=text)

        before_len = len(self._buffer_words)
        new_norm_words = [_normalize_scan_token(w) for w in chunk_words]
        self._buffer_words.extend(chunk_words)
        self._buffer_norm_words.extend(new_norm_words)

        self._seed_hypotheses_from_new_words(before_len, len(self._buffer_words))
        complete_hits, stats = self._verify_hypotheses(flush=False)
        self._prune_hypotheses()
        self._trim_buffer()

        return StreamingQuranMatcherResult(
            complete_verses=complete_hits,
            partial_hypotheses=len(self._active),
            consumed_text=text,
            stats=stats,
        )

    def flush(self) -> StreamingQuranMatcherResult:
        """Flush any pending hypotheses at end-of-stream."""
        complete_hits, stats = self._verify_hypotheses(flush=True)
        result = StreamingQuranMatcherResult(
            complete_verses=complete_hits,
            partial_hypotheses=0,
            consumed_text=" ".join(self._buffer_words),
            stats=stats,
        )
        self.reset()
        return result

    def reset(self) -> None:
        """Reset streaming state."""
        self._buffer_words = []
        self._buffer_norm_words = []
        self._buffer_start_token = 0
        self._stream_token_count = 0
        self._chunk_index = 0
        self._active = {}
        self._emitted_keys = set()
        self._emitted_spans_by_ref = {}

    def _seed_hypotheses_from_new_words(self, new_start_local: int, new_end_local: int) -> None:
        """Seed/update hypotheses using n-grams that touch newly appended words."""
        total_local = len(self._buffer_norm_words)
        if total_local < self.options.min_words:
            self._stream_token_count = self._buffer_start_token + total_local
            return

        weighted_sizes: list[tuple[int, float]] = []
        for n in self.options.anchor_ngram_sizes:
            if n == 3:
                weighted_sizes.append((n, 6.0))
            elif n == 2:
                weighted_sizes.append((n, 2.5))
            else:
                weighted_sizes.append((n, float(n)))

        for n, weight in weighted_sizes:
            if total_local < n:
                continue

            local_start = max(0, new_start_local - n + 1)
            local_end = new_end_local - n + 1
            if local_end <= local_start:
                continue

            # Build only once for the full buffer, then slice the relevant range.
            ngrams_with_pos = _iter_word_ngrams_with_pos(tuple(self._buffer_norm_words), n)
            for input_pos_local, gram in ngrams_with_pos[local_start:local_end]:
                if not gram.strip():
                    continue

                postings = self.validator._ngram_pos_index_by_n.get(n, {}).get(gram, ())
                if not postings:
                    continue

                input_pos_global = self._buffer_start_token + input_pos_local
                for verse_id, verse_pos in postings:
                    verse_wc = self.validator._verse_word_count_by_id.get(verse_id, 0)
                    if verse_wc < self.options.min_words:
                        continue
                    if verse_wc > self.options.max_words + self.options.max_len_delta:
                        continue

                    candidate_start = input_pos_global - verse_pos
                    if candidate_start < 0:
                        continue

                    key = (verse_id, candidate_start)
                    h = self._active.get(key)
                    if h is None:
                        self._active[key] = ActiveHypothesis(
                            verse_id=verse_id,
                            candidate_start_token=candidate_start,
                            anchor_score=weight,
                            anchor_hits=1,
                            first_chunk_seen=self._chunk_index,
                            last_chunk_seen=self._chunk_index,
                            last_anchor_token=input_pos_global + n - 1,
                            hits=[(input_pos_global, verse_pos, n, weight)],
                        )
                    else:
                        h.anchor_score += weight
                        h.anchor_hits += 1
                        h.last_chunk_seen = self._chunk_index
                        h.last_anchor_token = max(h.last_anchor_token, input_pos_global + n - 1)
                        h.hits.append((input_pos_global, verse_pos, n, weight))
                        if len(h.hits) > self.options.max_hits_per_hypothesis:
                            # Keep the most recent evidence; candidate_start preserves
                            # the original alignment hypothesis for older anchors.
                            h.hits = h.hits[-self.options.max_hits_per_hypothesis:]

        self._stream_token_count = self._buffer_start_token + len(self._buffer_words)

    def _verify_hypotheses(self, *, flush: bool) -> tuple[list[StreamingQuranHit], dict[str, int]]:
        """Verify top hypotheses against nearby spans and emit complete matches."""
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
        verification_cap = self.options.max_verifications_per_chunk
        full_validated_count = 0
        full_validation_cap = self.options.max_full_validations_per_chunk
        if len(self._active) >= max(1, int(self.options.beam_size * self.options.dense_active_ratio)):
            verification_cap += self.options.extra_verifications_dense
            full_validation_cap += self.options.extra_full_validations_dense
        if flush:
            verification_cap += max(8, self.options.extra_verifications_dense // 4)
            full_validation_cap += max(2, self.options.extra_full_validations_dense // 2)
        checked_spans: set[tuple[int, int, int]] = set()

        for h in ranked:
            if verified_count >= verification_cap:
                break
            if h.last_verified_chunk == self._chunk_index and not flush:
                continue
            if h.anchor_score < self.options.min_anchor_score and h.best_confidence < self.options.min_confidence:
                continue

            verse = self.validator.verse_by_id[h.verse_id]
            verse_wc = self.validator._verse_word_count_by_id.get(h.verse_id, 0)
            if verse_wc <= 0:
                continue

            chain = self._build_chain_summary(h, verse_wc)
            if chain is None:
                h.last_verified_chunk = self._chunk_index
                continue

            candidate_matches: list[_ApproxMatch] = []
            uncertain_chain = (
                chain.coverage_ratio < self.options.adaptive_expand_coverage_threshold
                or chain.support_ratio < self.options.adaptive_expand_support_threshold
            )
            start_delta_max = self.options.max_start_delta
            len_delta_max = self.options.max_len_delta
            local_min_approx = self.options.min_approx_confidence
            if uncertain_chain:
                start_delta_max += self.options.adaptive_extra_start_delta
                len_delta_max += self.options.adaptive_extra_len_delta
                local_min_approx = max(
                    0.45,
                    self.options.min_approx_confidence - self.options.adaptive_min_approx_drop,
                )
            if verse_wc <= self.options.short_verse_words:
                local_min_approx = min(
                    local_min_approx,
                    max(0.52, self.options.min_approx_confidence - 0.14),
                )

            start_deltas = range(-start_delta_max, start_delta_max + 1)
            len_deltas = range(-len_delta_max, len_delta_max + 1)

            for start_delta in start_deltas:
                if verified_count >= verification_cap:
                    break
                start_global = chain.est_start_global + start_delta
                start_local = start_global - self._buffer_start_token
                if start_local < 0 or start_local >= len(self._buffer_words):
                    continue

                for len_delta in len_deltas:
                    if verified_count >= verification_cap:
                        break
                    candidate_len = verse_wc + chain.span_len_bias + len_delta
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
                    approx = self._score_chain_window(
                        chain,
                        verse_wc=verse_wc,
                        start_global=start_global,
                        end_global=end_global,
                    )
                    if approx is None or approx.approx_confidence < local_min_approx:
                        continue

                    candidate_matches.append(approx)

            h.last_verified_chunk = self._chunk_index
            if candidate_matches:
                h.best_confidence = max(
                    h.best_confidence,
                    max(m.approx_confidence for m in candidate_matches),
                )

                verse_tokens = tuple(self.validator._normalized_verse_words_by_id.get(h.verse_id, ()))
                if not verse_tokens:
                    verse_tokens = tuple(_normalize_scan_token(w) for w in verse.text.split())

                selected_candidates = self._select_emit_candidates(
                    candidate_matches,
                    limit=self.options.max_emit_window_candidates,
                )

                for match in selected_candidates:
                    start_local = match.start_local
                    end_local = match.end_local
                    start_global = self._buffer_start_token + start_local
                    end_global = self._buffer_start_token + end_local
                    touches_end = end_local >= len(self._buffer_words)

                    if touches_end and not flush and h.chunk_span < self.options.max_chunk_span:
                        # Keep as partial for next chunk; try a different candidate first.
                        continue

                    # Finalize with full validator only for emitted matches to keep
                    # per-candidate verification cheap.
                    if full_validated_count >= full_validation_cap:
                        break

                    window_text = " ".join(self._buffer_words[start_local:end_local])
                    result = self.validator._validate_against_verse(window_text, verse)
                    full_validated_count += 1
                    if not result.is_valid or result.confidence < self.options.min_confidence:
                        continue

                    if result.match_type == "fuzzy":
                        window_norm = tuple(self._buffer_norm_words[start_local:end_local])
                        lexical_overlap = self._token_multiset_overlap_ratio(window_norm, verse_tokens)
                        min_overlap = self._min_required_fuzzy_overlap(
                            verse_words=len(verse_tokens),
                            confidence=result.confidence,
                        )
                        if lexical_overlap < min_overlap:
                            continue

                    ref = result.reference or ""
                    if not flush and ref and ref in emitted_refs_this_pass:
                        break

                    emit_key = (ref, start_global)
                    if emit_key in self._emitted_keys:
                        break
                    if self._is_duplicate_reference_span(
                        ref,
                        start_global,
                        end_global,
                    ):
                        break
                    self._emitted_keys.add(emit_key)
                    self._record_emitted_reference_span(
                        ref,
                        start_global,
                        end_global,
                    )
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

                    # Prevent repeated detections for the same anchored hypothesis.
                    self._active.pop((h.verse_id, h.candidate_start_token), None)
                    break

        return emitted, {
            "active": len(self._active),
            "verified": verified_count,
            "full_validated": full_validated_count,
            "emitted": len(emitted),
        }

    def _build_chain_summary(
        self,
        h: ActiveHypothesis,
        verse_wc: int,
    ) -> Optional[_ChainSummary]:
        """
        Build a monotonic weighted anchor chain for one hypothesis.

        The chain acts as the fast verifier signal: if anchors align in order and
        cover a reasonable fraction of verse positions, we keep the hypothesis for
        emit-time full validation.
        """
        if verse_wc <= 0 or not h.hits:
            return None

        buffer_start = self._buffer_start_token
        buffer_end = self._buffer_start_token + len(self._buffer_words)

        hits: list[tuple[int, int, int, float]] = []
        for input_pos, verse_pos, n, weight in h.hits:
            if input_pos >= buffer_end:
                continue
            if input_pos + n <= buffer_start:
                continue
            if verse_pos >= verse_wc:
                continue
            n_eff = min(n, verse_wc - verse_pos)
            if n_eff <= 0:
                continue
            hits.append((input_pos, verse_pos, n_eff, weight))

        if not hits:
            return None

        hits.sort(key=lambda x: (x[0], x[1], -x[2]))
        k = len(hits)
        dp = [0.0] * k
        prev = [-1] * k

        for i, (ip_i, vp_i, n_i, w_i) in enumerate(hits):
            base = w_i + 0.30 * (n_i - 1)
            best_score = base
            best_prev = -1

            for j in range(i):
                ip_j, vp_j, n_j, _w_j = hits[j]
                if ip_j >= ip_i or vp_j >= vp_i:
                    continue

                ip_j_end = ip_j + n_j
                vp_j_end = vp_j + n_j
                overlap_ip = max(0, ip_j_end - ip_i)
                overlap_vp = max(0, vp_j_end - vp_i)
                if overlap_ip and overlap_vp:
                    # Allow nested/overlapping anchors, but penalize to avoid
                    # overcounting repeated n-grams at nearly the same location.
                    overlap_penalty = 0.35 * (overlap_ip + overlap_vp)
                else:
                    overlap_penalty = 0.0

                gap_ip = max(0, ip_i - ip_j_end)
                gap_vp = max(0, vp_i - vp_j_end)
                gap_mismatch = abs(gap_ip - gap_vp)
                gap_penalty = 0.20 * gap_mismatch + 0.03 * (gap_ip + gap_vp)

                cand = dp[j] + base - overlap_penalty - gap_penalty
                if cand > best_score:
                    best_score = cand
                    best_prev = j

            dp[i] = best_score
            prev[i] = best_prev

        best_idx = max(range(k), key=lambda idx: dp[idx])
        chain_rev: list[tuple[int, int, int, float]] = []
        cursor = best_idx
        while cursor >= 0:
            chain_rev.append(hits[cursor])
            cursor = prev[cursor]
        chain_hits = list(reversed(chain_rev))
        if not chain_hits:
            return None

        covered_verse: set[int] = set()
        covered_input: set[int] = set()
        diags: list[int] = []
        min_ip = None
        max_ip_end = None
        min_vp = None
        max_vp_end = None
        for ip, vp, n_hit, weight in chain_hits:
            diags.append(ip - vp)
            if min_ip is None or ip < min_ip:
                min_ip = ip
            if max_ip_end is None or (ip + n_hit) > max_ip_end:
                max_ip_end = ip + n_hit
            if min_vp is None or vp < min_vp:
                min_vp = vp
            if max_vp_end is None or (vp + n_hit) > max_vp_end:
                max_vp_end = vp + n_hit
            for pos in range(vp, min(verse_wc, vp + n_hit)):
                covered_verse.add(pos)
            for pos in range(ip, ip + n_hit):
                if buffer_start <= pos < buffer_end:
                    covered_input.add(pos)

        if min_ip is None or max_ip_end is None or min_vp is None or max_vp_end is None:
            return None

        diags.sort()
        est_start_global = diags[len(diags) // 2]

        verse_span = max(1, max_vp_end - min_vp)
        input_span = max(1, max_ip_end - min_ip)
        raw_bias = input_span - verse_span
        span_len_bias = max(-self.options.max_len_delta, min(self.options.max_len_delta, raw_bias))

        coverage_ratio = len(covered_verse) / max(1, verse_wc)
        expected_chain_hits = max(2, verse_wc // 4)
        support_ratio = min(1.0, len(chain_hits) / expected_chain_hits)

        # Keep the raw score for window ranking normalization later.
        return _ChainSummary(
            chain_hits=chain_hits,
            chain_score=max(0.0, dp[best_idx]),
            coverage_ratio=coverage_ratio,
            support_ratio=support_ratio,
            est_start_global=est_start_global,
            span_len_bias=span_len_bias,
            min_input_global=min_ip,
            max_input_end_global=max_ip_end,
            min_verse_pos=min_vp,
            max_verse_end_pos=max_vp_end,
            diag_min=diags[0],
            diag_max=diags[-1],
        )

    def _score_chain_window(
        self,
        chain: _ChainSummary,
        *,
        verse_wc: int,
        start_global: int,
        end_global: int,
    ) -> Optional[_ApproxMatch]:
        """Score a candidate window using chained anchor evidence only."""
        if end_global <= start_global:
            return None

        window_len = end_global - start_global
        if window_len <= 0:
            return None

        covered_verse: set[int] = set()
        weight_mass = 0.0
        partial_hit_count = 0
        for ip, vp, n_hit, weight in chain.chain_hits:
            hit_start = ip
            hit_end = ip + n_hit
            overlap = min(hit_end, end_global) - max(hit_start, start_global)
            if overlap <= 0:
                continue

            frac = overlap / max(1, n_hit)
            weight_mass += weight * frac
            if overlap < n_hit:
                partial_hit_count += 1

            # Approximate verse coverage from overlap share.
            verse_cov = int(round(n_hit * frac))
            verse_cov = max(1, min(n_hit, verse_cov))
            for pos in range(vp, min(verse_wc, vp + verse_cov)):
                covered_verse.add(pos)

        if not covered_verse and weight_mass <= 0.0:
            return None

        coverage_ratio = len(covered_verse) / max(1, verse_wc)
        total_chain_weight = max(1.0, sum(w for _ip, _vp, _n, w in chain.chain_hits))
        mass_ratio = min(1.0, weight_mass / total_chain_weight)

        diag_slack = max(1, chain.diag_max - chain.diag_min)
        start_penalty = 0.015 * max(0, abs(start_global - chain.est_start_global) - diag_slack)
        target_len = verse_wc + chain.span_len_bias
        len_penalty = 0.012 * abs(window_len - target_len)
        partial_penalty = 0.02 * partial_hit_count
        needed_left = max(0, chain.min_verse_pos)
        needed_right = max(0, verse_wc - chain.max_verse_end_pos)
        have_left = max(0, chain.min_input_global - start_global)
        have_right = max(0, end_global - chain.max_input_end_global)
        boundary_deficit_penalty = (
            0.07 * max(0, needed_left - have_left)
            + 0.06 * max(0, needed_right - have_right)
        )

        confidence = (
            0.10
            + 0.35 * chain.coverage_ratio
            + 0.20 * chain.support_ratio
            + 0.25 * coverage_ratio
            + 0.20 * mass_ratio
            - start_penalty
            - len_penalty
            - partial_penalty
            - boundary_deficit_penalty
        )
        confidence = max(0.0, min(0.99, confidence))

        if confidence < self.options.min_approx_confidence:
            return None

        start_local = start_global - self._buffer_start_token
        end_local = end_global - self._buffer_start_token
        if start_local < 0 or end_local <= start_local:
            return None

        return _ApproxMatch(
            start_local=start_local,
            end_local=end_local,
            approx_confidence=confidence,
            match_type="anchor-chain",
            lexical_overlap=coverage_ratio,
            anchor_mass_ratio=mass_ratio,
            chain_coverage=chain.coverage_ratio,
        )

    @staticmethod
    def _token_multiset_overlap_ratio(
        input_tokens: tuple[str, ...],
        verse_tokens: tuple[str, ...],
    ) -> float:
        """Multiset token overlap normalized by verse length."""
        if not input_tokens or not verse_tokens:
            return 0.0

        input_counts = Counter(input_tokens)
        verse_counts = Counter(verse_tokens)
        overlap = 0
        for token, v_count in verse_counts.items():
            overlap += min(v_count, input_counts.get(token, 0))
        return overlap / max(1, len(verse_tokens))

    def _min_required_fuzzy_overlap(self, verse_words: int, confidence: float) -> float:
        """Adaptive lexical-overlap requirement for fuzzy emits."""
        if verse_words <= self.options.short_verse_words:
            base = self.options.min_fuzzy_lexical_overlap_short
        else:
            base = self.options.min_fuzzy_lexical_overlap

        # Allow slightly lower overlap for very high-confidence fuzzy matches.
        confidence_bonus = max(0.0, confidence - self.options.min_confidence) * 0.35
        return max(0.40, base - confidence_bonus)

    def _select_emit_candidates(
        self,
        candidates: list[_ApproxMatch],
        *,
        limit: int,
    ) -> list[_ApproxMatch]:
        """
        Choose emit candidates using mixed ranking.

        Mixes top-by-confidence and top-by-lexical/anchor evidence to reduce
        misses when one scoring dimension over-favors truncated spans.
        """
        if not candidates or limit <= 0:
            return []

        conf_ranked = sorted(
            candidates,
            key=lambda m: (
                -m.approx_confidence,
                -(m.end_local - m.start_local),
                m.start_local,
            ),
        )
        lex_ranked = sorted(
            candidates,
            key=lambda m: (
                -m.lexical_overlap,
                -m.anchor_mass_ratio,
                -m.chain_coverage,
                -m.approx_confidence,
                -(m.end_local - m.start_local),
            ),
        )

        selected: list[_ApproxMatch] = []
        seen: set[tuple[int, int]] = set()
        i = 0
        j = 0
        while len(selected) < limit and (i < len(conf_ranked) or j < len(lex_ranked)):
            if i < len(conf_ranked):
                c = conf_ranked[i]
                i += 1
                key = (c.start_local, c.end_local)
                if key not in seen:
                    selected.append(c)
                    seen.add(key)
                    if len(selected) >= limit:
                        break
            if j < len(lex_ranked):
                c = lex_ranked[j]
                j += 1
                key = (c.start_local, c.end_local)
                if key not in seen:
                    selected.append(c)
                    seen.add(key)
        return selected

    def _is_duplicate_reference_span(
        self,
        reference: str,
        start_token: int,
        end_token: int,
    ) -> bool:
        """Check if this reference span substantially overlaps a prior emission."""
        if not reference or end_token <= start_token:
            return False

        prior_spans = self._emitted_spans_by_ref.get(reference)
        if not prior_spans:
            return False

        span_len = end_token - start_token
        for prev_start, prev_end in prior_spans:
            prev_len = max(1, prev_end - prev_start)
            overlap = min(end_token, prev_end) - max(start_token, prev_start)
            if overlap <= 0:
                # Also suppress near-adjacent repeats from chunk overlap carryover.
                if abs(start_token - prev_start) <= self.options.duplicate_ref_window_tokens:
                    length_ratio = min(span_len, prev_len) / max(span_len, prev_len)
                    if length_ratio >= 0.65:
                        return True
                continue

            shorter = max(1, min(span_len, prev_len))
            # Suppress near-identical re-emits from overlapping hypotheses,
            # but still allow the same verse later in the stream.
            if overlap >= max(2, int(0.5 * shorter)):
                return True

        return False

    def _record_emitted_reference_span(
        self,
        reference: str,
        start_token: int,
        end_token: int,
    ) -> None:
        """Remember emitted span for overlap-based duplicate suppression."""
        if not reference or end_token <= start_token:
            return

        spans = self._emitted_spans_by_ref.setdefault(reference, [])
        spans.append((start_token, end_token))
        # Keep bounded memory for long live streams.
        if len(spans) > 48:
            del spans[:-48]

    def _prune_hypotheses(self) -> None:
        """Drop stale/weak hypotheses and keep the beam bounded."""
        if not self._active:
            return

        current_end = self._buffer_start_token + len(self._buffer_words)
        to_delete: list[tuple[int, int]] = []
        for key, h in self._active.items():
            age_tokens = current_end - h.last_anchor_token
            if age_tokens > self.options.max_anchor_age_tokens:
                to_delete.append(key)
                continue
            if h.chunk_span > self.options.max_chunk_span:
                to_delete.append(key)
                continue
            # If the hypothesized start is far behind the retained buffer, we can no longer verify it.
            if h.candidate_start_token < self._buffer_start_token:
                to_delete.append(key)

        for key in to_delete:
            self._active.pop(key, None)

        if len(self._active) <= self.options.beam_size:
            return

        ranked_keys = sorted(
            self._active.keys(),
            key=lambda key: (
                -(self._active[key].anchor_score + self._active[key].best_confidence * 8.0),
                -self._active[key].last_anchor_token,
            ),
        )
        keep = set(ranked_keys[:self.options.beam_size])
        for key in list(self._active.keys()):
            if key not in keep:
                self._active.pop(key, None)

    def _trim_buffer(self) -> None:
        """Retain only the tail of the token buffer and update offsets."""
        max_keep = self.options.max_buffer_words
        if len(self._buffer_words) <= max_keep:
            return

        drop = len(self._buffer_words) - max_keep
        self._buffer_words = self._buffer_words[drop:]
        self._buffer_norm_words = self._buffer_norm_words[drop:]
        self._buffer_start_token += drop

        # Remove hypotheses whose candidate start falls before the retained buffer.
        for key in list(self._active.keys()):
            if self._active[key].candidate_start_token < self._buffer_start_token:
                self._active.pop(key, None)
