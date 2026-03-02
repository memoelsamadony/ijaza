"""
Experimental streaming Quran matcher.

This module implements a stateful, anchor-driven matcher for chunked ASR text.
It keeps lightweight hypotheses across chunks instead of rescanning each chunk
from scratch with many sliding windows.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, field
import math
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

    # `stream_v2` keeps the current high-speed hypothesis matcher.
    # `main_indexed_v1` runs chunk-local indexed scanning compatible with
    # the pre-stream-matcher main branch behavior.
    matcher_mode: str = "stream_v2"

    min_confidence: float = 0.85
    min_words: int = 3
    max_words: int = 50
    asr_tolerant: bool = True
    # Final complete-emit quality gates after local fragment alignment.
    complete_min_matched_tokens: int = 4
    complete_min_fragment_chars: int = 12
    complete_min_coverage_ratio: float = 0.80
    complete_min_boundary_ratio: float = 0.50
    complete_short_verse_words: int = 4
    complete_short_verse_min_confidence: float = 0.95
    complete_short_verse_min_coverage: float = 0.95
    complete_short_verse_require_full_boundary: bool = True

    # Anchor search
    anchor_ngram_sizes: tuple[int, ...] = (3, 2)
    beam_size: int = 48
    max_hypotheses_per_verse: int = 4
    max_hypotheses_per_start_bucket: int = 3
    start_bucket_words: int = 6
    max_verifications_per_chunk: int = 64
    extra_verifications_dense: int = 256
    # Per-hypothesis verification cap to avoid starving mid-ranked candidates.
    max_verifications_per_hypothesis: int = 18
    max_verifications_per_hypothesis_uncertain: int = 30
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
    # Optional partial-ayah detection path.
    detect_partial_ayahs: bool = False
    partial_min_words: int = 3
    partial_min_fragment_words: int = 4
    partial_min_overlap_tokens: int = 2
    partial_min_score: float = 0.45
    partial_length_sigmoid_center: float = 6.0
    partial_length_sigmoid_scale: float = 1.6
    partial_min_distinctiveness: float = 0.22
    partial_min_distinctiveness_short: float = 0.40
    partial_collision_penalty_weight: float = 0.26
    partial_collision_penalty_threshold: float = 0.35
    partial_collision_reject_ratio: float = 0.70
    partial_overlap_suppression_ratio: float = 0.70
    partial_max_emits_per_chunk: int = 4
    partial_short_verse_words: int = 5
    partial_short_min_order_ratio: float = 0.80
    partial_short_require_boundary: bool = True
    partial_relaxed_min_approx_confidence: float = 0.50
    partial_relaxed_extra_start_delta: int = 2
    partial_relaxed_extra_len_delta: int = 2
    # Optional stricter fuzzy gate for very short verses to suppress
    # semantic-near-miss matches that share only common prefixes.
    strict_short_verse_mode: bool = False
    strict_short_verse_words: int = 4
    strict_short_verse_min_overlap: float = 0.66
    strict_short_verse_require_edge_tokens: bool = True
    strict_short_verse_edge_confidence_bypass: float = 0.94
    defer_fuzzy_emit_confidence: float = 0.90
    # Allow immediate emit for strong one-shot fuzzy matches to reduce
    # false negatives when the same verse is unlikely to reappear in later chunks.
    strong_fuzzy_emit_overlap: float = 0.74
    # Long verses are more prone to high-overlap paraphrase/quote-adjacent noise.
    # Require stronger lexical overlap for immediate one-shot fuzzy emits.
    strong_fuzzy_emit_long_verse_words: int = 20
    strong_fuzzy_emit_overlap_long: float = 0.85
    strong_fuzzy_emit_approx_confidence: float = 0.84
    strong_fuzzy_emit_anchor_score: float = 6.0
    strong_fuzzy_emit_anchor_hits: int = 2
    pending_confirmation_hits: int = 2
    pending_max_chunk_gap: int = 2
    end_chunk_adjacent_validations: int = 2

    # Optional bounded rescue scan:
    # run indexed scanner on a short rolling tail when stream path produced no emit.
    rescue_reanchor_enabled: bool = False
    rescue_only_when_no_emit: bool = True
    rescue_window_words: int = 72
    rescue_max_emits_per_scan: int = 2
    rescue_min_chunk_gap: int = 2
    rescue_min_confidence: float = 0.84
    rescue_max_active_hypotheses: int = 40

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
    is_partial: bool = False
    partial_score: float = 0.0
    coverage_ratio: float = 0.0
    purity_ratio: float = 0.0
    order_ratio: float = 0.0
    boundary_ratio: float = 0.0
    distinctiveness_ratio: float = 0.0
    matched_tokens: int = 0
    fragment_chars: int = 0
    collision_ratio: float = 0.0
    partial_label: str = ""


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
    partial_ayahs: list[StreamingQuranHit] = field(default_factory=list)
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
class _AlignedFragment:
    """Best aligned fragment inside a candidate window."""

    start_offset: int
    end_offset: int
    verse_start: int
    verse_end: int
    lcs_count: int
    overlap_count: int
    contiguous_run: int


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


@dataclass
class _PendingEmit:
    """Deferred fuzzy emission candidate waiting for reconfirmation."""

    reference: str
    start_token: int
    end_token: int
    confidence: float
    original_text: str
    correct_text: str
    translations: dict[str, str]
    needs_correction: bool
    created_chunk: int
    last_seen_chunk: int
    seen_count: int
    required_hits: int
    verse: Optional[QuranVerse] = None


class StreamingQuranMatcher:
    """
    Experimental online matcher for Quran verses in ASR chunks.

    Strategy:
    - Build hashed n-gram anchor hits against verse token positions
    - Maintain candidate hypotheses across chunks
    - Verify only a bounded set of candidate spans each chunk
    """

    _SUPPORTED_MODES = ("stream_v2", "main_indexed_v1")
    _DEFAULT_TOKEN_IDF = 1.0
    _HIGH_COLLISION_PARTIAL_PHRASES = (
        "ان في ذلك",
        "ان في ذلك لايات",
        "والله يعلم وانتم لا تعلمون",
        "الى يوم الدين",
        "وان كانوا من قبل",
        "اولئك هم",
        "بسم الله الرحمن الرحيم",
    )
    _HIGH_COLLISION_PARTIAL_FRAGMENTS = {
        "ان شاء الله",
        "الى يوم الدين",
        "والله يعلم وانتم لا تعلمون",
        "والله يعلم وانتم لا",
        "يعلم وانتم لا تعلمون",
        "يعلم وانتم لا",
        "ان في ذلك لايات",
    }

    def __init__(
        self,
        options: Optional[StreamingQuranMatcherOptions] = None,
        translation_provider: Optional['TranslationProvider'] = None,
    ):
        if options is None:
            options = StreamingQuranMatcherOptions()
        self.options = options
        if self.options.matcher_mode not in self._SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported matcher_mode={self.options.matcher_mode!r}. "
                f"Expected one of {self._SUPPORTED_MODES}."
            )

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
        self._partial_emitted_keys: set[tuple[str, int]] = set()
        self._partial_spans_by_ref: dict[str, list[tuple[int, int]]] = {}
        self._pending_by_ref: dict[str, _PendingEmit] = {}
        self._last_rescue_chunk: int = -10_000
        self._token_idf: dict[str, float] = {}
        self._max_token_idf: float = 1.0
        self._build_token_idf()

    def process_chunk(self, text: str) -> StreamingQuranMatcherResult:
        """Process one ASR chunk and emit complete verse matches."""
        self._chunk_index += 1
        chunk_words = text.split()
        if not chunk_words:
            return StreamingQuranMatcherResult(consumed_text=text)
        if self.options.matcher_mode == "main_indexed_v1":
            return self._process_chunk_main_indexed_v1(text=text, chunk_words=chunk_words)

        before_len = len(self._buffer_words)
        new_norm_words = [_normalize_scan_token(w) for w in chunk_words]
        self._buffer_words.extend(chunk_words)
        self._buffer_norm_words.extend(new_norm_words)

        self._seed_hypotheses_from_new_words(before_len, len(self._buffer_words))
        complete_hits, partial_hits, stats = self._verify_hypotheses(flush=False)
        rescue_scans = 0
        rescue_emitted = 0
        if self._should_run_rescue_scan(emitted_count=len(complete_hits)):
            rescue_scans = 1
            refs_this_pass = {h.reference for h in complete_hits if h.reference}
            rescue_hits = self._run_rescue_reanchor_scan(
                chunk_words=chunk_words,
                chunk_start_local=before_len,
                emitted_refs_this_pass=refs_this_pass,
            )
            if rescue_hits:
                rescue_emitted = len(rescue_hits)
                complete_hits.extend(rescue_hits)
        stats["rescue_scans"] = rescue_scans
        stats["rescue_emitted"] = rescue_emitted
        self._prune_hypotheses()
        self._trim_buffer()

        return StreamingQuranMatcherResult(
            complete_verses=complete_hits,
            partial_ayahs=partial_hits,
            partial_hypotheses=len(self._active),
            consumed_text=text,
            stats=stats,
        )

    def flush(self) -> StreamingQuranMatcherResult:
        """Flush any pending hypotheses at end-of-stream."""
        if self.options.matcher_mode == "main_indexed_v1":
            result = StreamingQuranMatcherResult(
                complete_verses=[],
                partial_hypotheses=0,
                consumed_text="",
                stats={
                    "active": 0,
                    "verified": 0,
                    "full_validated": 0,
                    "emitted": 0,
                    "pending": 0,
                    "mode_main_indexed": 1,
                },
            )
            self.reset()
            return result

        complete_hits, partial_hits, stats = self._verify_hypotheses(flush=True)
        result = StreamingQuranMatcherResult(
            complete_verses=complete_hits,
            partial_ayahs=partial_hits,
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
        self._partial_emitted_keys = set()
        self._partial_spans_by_ref = {}
        self._pending_by_ref = {}
        self._last_rescue_chunk = -10_000

    def _process_chunk_main_indexed_v1(
        self,
        *,
        text: str,
        chunk_words: list[str],
    ) -> StreamingQuranMatcherResult:
        """
        Compatibility mode: chunk-local indexed scan (main v1 behavior).

        This path intentionally does not keep cross-chunk hypotheses.
        """
        chunk_start_token = self._stream_token_count
        hits = self.validator.scan_for_verses(
            text,
            min_words=self.options.min_words,
            max_words=self.options.max_words,
            confidence_threshold=self.options.min_confidence,
        )

        complete: list[StreamingQuranHit] = []
        for item in hits:
            ref = str(item.get("reference") or "")
            if not ref:
                continue

            start_char = int(item.get("start_pos", 0) or 0)
            end_char = int(item.get("end_pos", start_char) or start_char)
            start_local = self._word_index_from_char_offset(text, start_char)
            end_local = self._word_index_from_char_offset(text, end_char)
            if end_local <= start_local:
                original = str(item.get("original_text", "") or "")
                fallback_words = len(original.split())
                if fallback_words <= 0:
                    continue
                end_local = start_local + fallback_words
            if end_local > len(chunk_words):
                end_local = len(chunk_words)
            if end_local <= start_local:
                continue

            start_global = chunk_start_token + start_local
            end_global = chunk_start_token + end_local

            verses = item.get("verses") or []
            verse = verses[0] if verses else None
            correct_text = str(item.get("correct_text", "") or "")
            if not correct_text and verse is not None:
                correct_text = verse.text

            complete.append(StreamingQuranHit(
                original_text=" ".join(chunk_words[start_local:end_local]),
                start_token=start_global,
                end_token=end_global,
                correct_text=correct_text,
                reference=ref,
                confidence=float(item.get("confidence", 0.0) or 0.0),
                verses=[verse] if verse else [],
                needs_correction=bool(item.get("needs_correction", False)),
                translations=dict(item.get("translations") or {}),
            ))

        self._stream_token_count += len(chunk_words)
        return StreamingQuranMatcherResult(
            complete_verses=complete,
            partial_hypotheses=0,
            consumed_text=text,
            stats={
                "active": 0,
                "verified": 0,
                "full_validated": len(hits),
                "emitted": len(complete),
                "pending": 0,
                "mode_main_indexed": 1,
            },
        )

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

    def _rank_active_hypotheses(self, limit: int) -> list[ActiveHypothesis]:
        """Rank active hypotheses and enforce diversity caps."""
        if not self._active or limit <= 0:
            return []

        ranked = sorted(
            self._active.values(),
            key=lambda h: (
                -(h.anchor_score + h.best_confidence * 8.0),
                -h.last_anchor_token,
                h.candidate_start_token,
                h.verse_id,
            ),
        )
        return self._select_diverse_hypotheses(ranked, limit=limit)

    def _select_diverse_hypotheses(
        self,
        ranked: list[ActiveHypothesis],
        *,
        limit: int,
    ) -> list[ActiveHypothesis]:
        """Select top hypotheses while capping per-verse and start-region crowding."""
        if not ranked or limit <= 0:
            return []

        per_verse_cap = max(1, int(self.options.max_hypotheses_per_verse))
        per_bucket_cap = max(1, int(self.options.max_hypotheses_per_start_bucket))
        bucket_size = max(1, int(self.options.start_bucket_words))

        selected: list[ActiveHypothesis] = []
        selected_ids: set[tuple[int, int]] = set()
        per_verse: dict[int, int] = {}
        per_bucket: dict[int, int] = {}

        # Seed the beam with one hypothesis per verse first.
        # This prevents dense clusters from starving low-anchor verses that may
        # still validate under ASR noise.
        verse_seed_target = min(limit, max(8, limit // 3))
        for h in ranked:
            if len(selected) >= verse_seed_target:
                break
            if h.verse_id in per_verse:
                continue
            bucket = h.candidate_start_token // bucket_size
            if per_bucket.get(bucket, 0) >= per_bucket_cap:
                continue
            selected.append(h)
            selected_ids.add((h.verse_id, h.candidate_start_token))
            per_verse[h.verse_id] = 1
            per_bucket[bucket] = per_bucket.get(bucket, 0) + 1

        for h in ranked:
            if len(selected) >= limit:
                break
            key = (h.verse_id, h.candidate_start_token)
            if key in selected_ids:
                continue
            if per_verse.get(h.verse_id, 0) >= per_verse_cap:
                continue
            bucket = h.candidate_start_token // bucket_size
            if per_bucket.get(bucket, 0) >= per_bucket_cap:
                continue
            selected.append(h)
            selected_ids.add(key)
            per_verse[h.verse_id] = per_verse.get(h.verse_id, 0) + 1
            per_bucket[bucket] = per_bucket.get(bucket, 0) + 1

        # Backfill without diversity caps if strict caps underfill the beam.
        if len(selected) < limit:
            for h in ranked:
                if len(selected) >= limit:
                    break
                key = (h.verse_id, h.candidate_start_token)
                if key in selected_ids:
                    continue
                selected.append(h)
                selected_ids.add(key)

        return selected

    def _pending_to_hit(self, pending: _PendingEmit) -> StreamingQuranHit:
        """Convert deferred candidate state to emitted hit."""
        verse = pending.verse
        return StreamingQuranHit(
            original_text=pending.original_text,
            start_token=pending.start_token,
            end_token=pending.end_token,
            correct_text=verse.text if verse else pending.correct_text,
            reference=pending.reference,
            confidence=pending.confidence,
            verses=[verse] if verse else [],
            needs_correction=pending.needs_correction,
            translations=dict(pending.translations),
        )

    def _drain_pending_emits(self, *, flush: bool) -> list[StreamingQuranHit]:
        """Expire stale deferred emits and release confirmed ones."""
        if not self._pending_by_ref:
            return []

        emitted: list[StreamingQuranHit] = []
        to_drop: set[str] = set()

        if not flush:
            for ref, pending in self._pending_by_ref.items():
                if (self._chunk_index - pending.last_seen_chunk) > self.options.pending_max_chunk_gap:
                    to_drop.add(ref)

        candidates: list[_PendingEmit] = []
        for ref, pending in self._pending_by_ref.items():
            if ref in to_drop:
                continue
            if flush:
                if pending.confidence >= self.options.min_confidence:
                    candidates.append(pending)
                continue
            if pending.seen_count >= self.options.pending_confirmation_hits:
                if pending.seen_count < max(1, pending.required_hits):
                    continue
                candidates.append(pending)

        candidates.sort(
            key=lambda p: (
                -p.seen_count,
                -p.confidence,
                -p.last_seen_chunk,
                p.start_token,
            )
        )

        for pending in candidates:
            emit_key = (pending.reference, pending.start_token)
            if emit_key in self._emitted_keys:
                to_drop.add(pending.reference)
                continue
            if self._is_duplicate_reference_span(
                pending.reference,
                pending.start_token,
                pending.end_token,
            ):
                to_drop.add(pending.reference)
                continue

            self._emitted_keys.add(emit_key)
            self._record_emitted_reference_span(
                pending.reference,
                pending.start_token,
                pending.end_token,
            )
            emitted.append(self._pending_to_hit(pending))
            to_drop.add(pending.reference)

        for ref in to_drop:
            self._pending_by_ref.pop(ref, None)

        return emitted

    def _update_pending_emit(
        self,
        *,
        reference: str,
        start_token: int,
        end_token: int,
        result,
        original_text: str,
        require_cross_chunk_confirmation: bool = False,
        required_hits: Optional[int] = None,
    ) -> Optional[StreamingQuranHit]:
        """Track low-confidence fuzzy candidate and emit once reconfirmed."""
        if not reference:
            return None
        required_hits_eff = max(
            1,
            int(required_hits or self.options.pending_confirmation_hits),
        )

        verse_obj = result.matched_verse
        existing = self._pending_by_ref.get(reference)
        if existing is None:
            self._pending_by_ref[reference] = _PendingEmit(
                reference=reference,
                start_token=start_token,
                end_token=end_token,
                confidence=result.confidence,
                original_text=original_text,
                correct_text=verse_obj.text if verse_obj else "",
                translations=dict(result.translations),
                needs_correction=(result.match_type != "exact"),
                created_chunk=self._chunk_index,
                last_seen_chunk=self._chunk_index,
                seen_count=1,
                required_hits=required_hits_eff,
                verse=verse_obj,
            )
            return None

        too_old = (self._chunk_index - existing.last_seen_chunk) > self.options.pending_max_chunk_gap
        overlap = min(end_token, existing.end_token) - max(start_token, existing.start_token)
        near_start = abs(start_token - existing.start_token) <= self.options.duplicate_ref_window_tokens
        if too_old or (overlap <= 0 and not near_start):
            existing.start_token = start_token
            existing.end_token = end_token
            existing.confidence = result.confidence
            existing.original_text = original_text
            existing.correct_text = verse_obj.text if verse_obj else ""
            existing.translations = dict(result.translations)
            existing.needs_correction = (result.match_type != "exact")
            existing.created_chunk = self._chunk_index
            existing.last_seen_chunk = self._chunk_index
            existing.seen_count = 1
            existing.required_hits = required_hits_eff
            existing.verse = verse_obj
            return None

        same_chunk_repeat = existing.last_seen_chunk == self._chunk_index
        existing.last_seen_chunk = self._chunk_index
        if not (require_cross_chunk_confirmation and same_chunk_repeat):
            existing.seen_count += 1
        existing.required_hits = max(existing.required_hits, required_hits_eff)
        if result.confidence >= existing.confidence:
            existing.start_token = start_token
            existing.end_token = end_token
            existing.confidence = result.confidence
            existing.original_text = original_text
            existing.correct_text = verse_obj.text if verse_obj else ""
            existing.translations = dict(result.translations)
            existing.needs_correction = (result.match_type != "exact")
            existing.verse = verse_obj

        if existing.seen_count < max(1, existing.required_hits):
            return None

        self._pending_by_ref.pop(reference, None)
        return self._pending_to_hit(existing)

    def _verify_hypotheses(
        self,
        *,
        flush: bool,
    ) -> tuple[list[StreamingQuranHit], list[StreamingQuranHit], dict[str, int]]:
        """Verify top hypotheses against nearby spans and emit complete matches."""
        if not self._active:
            pending_emitted = self._drain_pending_emits(flush=flush)
            return pending_emitted, [], {
                "active": 0,
                "verified": 0,
                "full_validated": 0,
                "emitted": len(pending_emitted),
                "partial_emitted": 0,
                "complete_demoted": 0,
                "pending": len(self._pending_by_ref),
            }

        ranked = self._rank_active_hypotheses(self.options.beam_size)

        emitted: list[StreamingQuranHit] = self._drain_pending_emits(flush=flush)
        partial_emitted: list[StreamingQuranHit] = []
        emitted_refs_this_pass: set[str] = {h.reference for h in emitted if h.reference}
        verified_count = 0
        verification_cap = self.options.max_verifications_per_chunk
        full_validated_count = 0
        complete_demoted = 0
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

            verse_tokens = tuple(self.validator._normalized_verse_words_by_id.get(h.verse_id, ()))
            if not verse_tokens:
                verse_tokens = tuple(_normalize_scan_token(w) for w in verse.text.split())

            candidate_matches: list[_ApproxMatch] = []
            partial_candidates: list[_ApproxMatch] = []
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
            detect_partials = self.options.detect_partial_ayahs
            partial_min_approx = max(
                0.35,
                float(self.options.partial_relaxed_min_approx_confidence),
            )
            scoring_min_approx = local_min_approx
            if detect_partials:
                scoring_min_approx = min(scoring_min_approx, partial_min_approx)
            local_verification_cap = self.options.max_verifications_per_hypothesis
            if uncertain_chain:
                local_verification_cap = self.options.max_verifications_per_hypothesis_uncertain
            if flush:
                local_verification_cap += 6
            local_verified = 0

            start_deltas = range(-start_delta_max, start_delta_max + 1)
            len_deltas = range(-len_delta_max, len_delta_max + 1)

            for start_delta in start_deltas:
                if verified_count >= verification_cap or local_verified >= local_verification_cap:
                    break
                start_global = chain.est_start_global + start_delta
                start_local = start_global - self._buffer_start_token
                if start_local < 0 or start_local >= len(self._buffer_words):
                    continue

                for len_delta in len_deltas:
                    if verified_count >= verification_cap or local_verified >= local_verification_cap:
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
                    local_verified += 1
                    approx = self._score_chain_window(
                        chain,
                        verse_wc=verse_wc,
                        start_global=start_global,
                        end_global=end_global,
                        min_confidence=scoring_min_approx,
                    )
                    if approx is None:
                        continue
                    if approx.approx_confidence >= local_min_approx:
                        candidate_matches.append(approx)
                        continue
                    if detect_partials and approx.approx_confidence >= partial_min_approx:
                        partial_candidates.append(approx)

            h.last_verified_chunk = self._chunk_index
            best_approx_this_h = 0.0
            if candidate_matches:
                best_approx_this_h = max(best_approx_this_h, max(m.approx_confidence for m in candidate_matches))
            if partial_candidates:
                best_approx_this_h = max(best_approx_this_h, max(m.approx_confidence for m in partial_candidates))
            if best_approx_this_h > 0.0:
                h.best_confidence = max(
                    h.best_confidence,
                    best_approx_this_h,
                )

            emitted_complete_for_h = False
            if candidate_matches:
                selected_candidates = self._select_emit_candidates(
                    candidate_matches,
                    limit=self.options.max_emit_window_candidates,
                )

                end_chunk_validations = 0
                for match in selected_candidates:
                    start_local = match.start_local
                    end_local = match.end_local
                    start_global = self._buffer_start_token + start_local
                    end_global = self._buffer_start_token + end_local
                    touches_end = end_local >= len(self._buffer_words)

                    if touches_end and not flush and h.chunk_span < self.options.max_chunk_span:
                        # Validate only a small number of edge candidates before deferring.
                        if end_chunk_validations >= self.options.end_chunk_adjacent_validations:
                            continue
                        end_chunk_validations += 1

                    # Finalize with full validator only for emitted matches to keep
                    # per-candidate verification cheap.
                    if full_validated_count >= full_validation_cap:
                        break

                    window_text = " ".join(self._buffer_words[start_local:end_local])
                    result = self.validator._validate_against_verse(window_text, verse)
                    full_validated_count += 1
                    if not result.is_valid or result.confidence < self.options.min_confidence:
                        continue

                    emit_start_local = start_local
                    emit_end_local = end_local
                    emit_verse_start = 0
                    emit_verse_end = len(verse_tokens)
                    aligned = self._align_fragment_to_verse(
                        tuple(self._buffer_norm_words[start_local:end_local]),
                        verse_tokens,
                        min_match_tokens=max(2, self.options.partial_min_overlap_tokens),
                    )
                    if aligned is not None:
                        emit_start_local = start_local + aligned.start_offset
                        emit_end_local = start_local + aligned.end_offset
                        emit_verse_start = aligned.verse_start
                        emit_verse_end = aligned.verse_end

                    if emit_end_local <= emit_start_local:
                        continue
                    emit_start_global = self._buffer_start_token + emit_start_local
                    emit_end_global = self._buffer_start_token + emit_end_local
                    emit_window_text = " ".join(self._buffer_words[emit_start_local:emit_end_local])
                    emit_window_norm = tuple(self._buffer_norm_words[emit_start_local:emit_end_local])

                    matched_tokens = aligned.overlap_count if aligned is not None else 0
                    fragment_chars = self._fragment_char_count(emit_window_norm)
                    coverage_ratio = (
                        matched_tokens / max(1, len(verse_tokens))
                        if verse_tokens
                        else 0.0
                    )
                    boundary_ratio = 0.0
                    complete_ok = True
                    if detect_partials:
                        complete_ok, matched_tokens, fragment_chars, coverage_ratio, boundary_ratio = (
                            self._complete_passes_quality_gates(
                                aligned=aligned,
                                window_tokens=emit_window_norm,
                                verse_tokens=verse_tokens,
                                confidence=result.confidence,
                            )
                        )
                    if not complete_ok:
                        complete_demoted += 1
                        continue

                    strong_fuzzy_emit = False
                    if result.match_type == "fuzzy":
                        lexical_overlap = self._token_multiset_overlap_ratio(emit_window_norm, verse_tokens)
                        min_overlap = self._min_required_fuzzy_overlap(
                            verse_words=len(verse_tokens),
                            confidence=result.confidence,
                        )
                        if lexical_overlap < min_overlap:
                            continue
                        if (
                            self.options.strict_short_verse_mode
                            and len(verse_tokens) <= self.options.strict_short_verse_words
                        ):
                            if lexical_overlap < self.options.strict_short_verse_min_overlap:
                                continue
                            if (
                                self.options.strict_short_verse_require_edge_tokens
                                and result.confidence < self.options.strict_short_verse_edge_confidence_bypass
                            ):
                                first_tok = verse_tokens[0] if verse_tokens else ""
                                last_tok = verse_tokens[-1] if verse_tokens else ""
                                has_first = first_tok in emit_window_norm if first_tok else True
                                has_last = last_tok in emit_window_norm if last_tok else True
                                if not (has_first and has_last):
                                    continue
                        strong_overlap_threshold = self.options.strong_fuzzy_emit_overlap
                        if len(verse_tokens) >= self.options.strong_fuzzy_emit_long_verse_words:
                            strong_overlap_threshold = max(
                                strong_overlap_threshold,
                                self.options.strong_fuzzy_emit_overlap_long,
                            )
                        strong_fuzzy_emit = (
                            lexical_overlap >= strong_overlap_threshold
                            and match.approx_confidence >= self.options.strong_fuzzy_emit_approx_confidence
                            and h.anchor_score >= self.options.strong_fuzzy_emit_anchor_score
                            and h.anchor_hits >= self.options.strong_fuzzy_emit_anchor_hits
                        )

                    ref = result.reference or ""
                    if not flush and ref and ref in emitted_refs_this_pass:
                        continue

                    if (
                        not flush
                        and ref
                        and result.match_type == "fuzzy"
                        and result.confidence < self.options.defer_fuzzy_emit_confidence
                        and not strong_fuzzy_emit
                    ):
                        pending_hit = self._update_pending_emit(
                            reference=ref,
                            start_token=emit_start_global,
                            end_token=emit_end_global,
                            result=result,
                            original_text=emit_window_text,
                            require_cross_chunk_confirmation=(
                                len(verse_tokens) >= self.options.strong_fuzzy_emit_long_verse_words
                            ),
                            required_hits=(
                                max(self.options.pending_confirmation_hits + 1, 3)
                                if len(verse_tokens) >= self.options.strong_fuzzy_emit_long_verse_words
                                else self.options.pending_confirmation_hits
                            ),
                        )
                        if pending_hit is not None:
                            emit_key = (pending_hit.reference, pending_hit.start_token)
                            if emit_key not in self._emitted_keys and not self._is_duplicate_reference_span(
                                pending_hit.reference,
                                pending_hit.start_token,
                                pending_hit.end_token,
                            ):
                                self._emitted_keys.add(emit_key)
                                self._record_emitted_reference_span(
                                    pending_hit.reference,
                                    pending_hit.start_token,
                                    pending_hit.end_token,
                                )
                                emitted.append(pending_hit)
                                emitted_refs_this_pass.add(pending_hit.reference)
                                self._active.pop((h.verse_id, h.candidate_start_token), None)
                                emitted_complete_for_h = True
                                break
                        continue

                    emit_key = (ref, emit_start_global)
                    if emit_key in self._emitted_keys:
                        continue
                    if self._is_duplicate_reference_span(
                        ref,
                        emit_start_global,
                        emit_end_global,
                    ):
                        continue
                    self._emitted_keys.add(emit_key)
                    self._record_emitted_reference_span(
                        ref,
                        emit_start_global,
                        emit_end_global,
                    )
                    if ref:
                        emitted_refs_this_pass.add(ref)

                    verse_obj = result.matched_verse
                    emitted.append(StreamingQuranHit(
                        original_text=emit_window_text,
                        start_token=emit_start_global,
                        end_token=emit_end_global,
                        correct_text=verse_obj.text if verse_obj else "",
                        reference=ref,
                        confidence=result.confidence,
                        verses=[verse_obj] if verse_obj else [],
                        needs_correction=(result.match_type != "exact"),
                        translations=result.translations,
                        coverage_ratio=coverage_ratio,
                        boundary_ratio=boundary_ratio,
                        matched_tokens=matched_tokens,
                        fragment_chars=fragment_chars,
                    ))

                    # Prevent repeated detections for the same anchored hypothesis.
                    self._active.pop((h.verse_id, h.candidate_start_token), None)
                    self._pending_by_ref.pop(ref, None)
                    emitted_complete_for_h = True
                    break

            if (
                detect_partials
                and not emitted_complete_for_h
                and len(partial_emitted) < self.options.partial_max_emits_per_chunk
            ):
                partial_pool = list(candidate_matches)
                partial_pool.extend(partial_candidates)
                if partial_pool:
                    selected_partial = self._select_emit_candidates(
                        partial_pool,
                        limit=max(
                            self.options.max_emit_window_candidates,
                            self.options.partial_max_emits_per_chunk * 2,
                        ),
                    )
                    selected_partial = self._suppress_overlapping_approx_matches(
                        selected_partial,
                        overlap_ratio=self.options.partial_overlap_suppression_ratio,
                        limit=max(
                            self.options.max_emit_window_candidates,
                            self.options.partial_max_emits_per_chunk * 2,
                        ),
                    )
                    for match in selected_partial:
                        if len(partial_emitted) >= self.options.partial_max_emits_per_chunk:
                            break
                        start_local = match.start_local
                        end_local = match.end_local
                        start_global = self._buffer_start_token + start_local
                        end_global = self._buffer_start_token + end_local
                        partial_hit = self._build_partial_hit(
                            verse=verse,
                            verse_tokens=verse_tokens,
                            start_local=start_local,
                            end_local=end_local,
                            start_global=start_global,
                            end_global=end_global,
                            approx_confidence=match.approx_confidence,
                        )
                        if partial_hit is None:
                            continue
                        if partial_hit.reference:
                            emit_key = (partial_hit.reference, partial_hit.start_token)
                            if emit_key in self._partial_emitted_keys:
                                continue
                            if emit_key in self._emitted_keys:
                                continue
                            if self._is_duplicate_reference_span(
                                partial_hit.reference,
                                partial_hit.start_token,
                                partial_hit.end_token,
                            ):
                                continue
                            if self._is_duplicate_partial_reference_span(
                                partial_hit.reference,
                                partial_hit.start_token,
                                partial_hit.end_token,
                            ):
                                continue
                            self._partial_emitted_keys.add(emit_key)
                            self._record_partial_reference_span(
                                partial_hit.reference,
                                partial_hit.start_token,
                                partial_hit.end_token,
                            )
                        partial_emitted.append(partial_hit)

        return emitted, partial_emitted, {
            "active": len(self._active),
            "verified": verified_count,
            "full_validated": full_validated_count,
            "emitted": len(emitted),
            "partial_emitted": len(partial_emitted),
            "complete_demoted": complete_demoted,
            "pending": len(self._pending_by_ref),
        }

    @staticmethod
    def _word_index_from_char_offset(text: str, char_offset: int) -> int:
        """Map character offset in space-joined text to a token index."""
        if char_offset <= 0:
            return 0
        if char_offset >= len(text):
            return text.count(" ") + 1 if text else 0
        return text[:char_offset].count(" ")

    def _should_run_rescue_scan(self, *, emitted_count: int) -> bool:
        """Gate expensive rescue scan to keep real-time behavior bounded."""
        if not self.options.rescue_reanchor_enabled:
            return False
        if self.options.rescue_only_when_no_emit and emitted_count > 0:
            return False
        if len(self._active) > self.options.rescue_max_active_hypotheses:
            return False
        if (self._chunk_index - self._last_rescue_chunk) < self.options.rescue_min_chunk_gap:
            return False
        return True

    def _run_rescue_reanchor_scan(
        self,
        *,
        chunk_words: list[str],
        chunk_start_local: int,
        emitted_refs_this_pass: set[str],
    ) -> list[StreamingQuranHit]:
        """
        Run a bounded indexed scan on the current chunk to recover missed matches.

        This reuses the validator's indexed scanner (same family as baseline)
        but keeps invocation bounded by gating/cooldown and per-scan emit caps.
        """
        if len(chunk_words) < self.options.min_words:
            return []

        scan_start_local = chunk_start_local
        max_words = max(1, int(self.options.rescue_window_words))
        if len(chunk_words) > max_words:
            cut = len(chunk_words) - max_words
            scan_start_local += cut
            words = chunk_words[cut:]
        else:
            words = chunk_words
        if len(words) < self.options.min_words:
            return []

        chunk_text = " ".join(words)
        hits = self.validator.scan_for_verses(
            chunk_text,
            min_words=self.options.min_words,
            max_words=self.options.max_words,
            confidence_threshold=self.options.rescue_min_confidence,
        )
        self._last_rescue_chunk = self._chunk_index
        if not hits:
            return []

        # Prefer stronger candidates first to minimize extra emissions.
        ranked = sorted(
            hits,
            key=lambda h: (
                -float(h.get("confidence", 0.0)),
                int(h.get("start_pos", 0)),
                int(h.get("end_pos", 0)),
            ),
        )

        emitted: list[StreamingQuranHit] = []
        for item in ranked:
            if len(emitted) >= self.options.rescue_max_emits_per_scan:
                break

            ref = str(item.get("reference", "") or "")
            if not ref:
                continue
            if ref in emitted_refs_this_pass:
                continue

            conf = float(item.get("confidence", 0.0) or 0.0)
            if conf < self.options.min_confidence:
                continue

            start_char = int(item.get("start_pos", 0) or 0)
            end_char = int(item.get("end_pos", start_char) or start_char)
            start_word_local = self._word_index_from_char_offset(chunk_text, start_char)
            end_word_local = self._word_index_from_char_offset(chunk_text, end_char)
            if end_word_local <= start_word_local:
                original = str(item.get("original_text", "") or "")
                span_len_words = len(original.split())
                if span_len_words <= 0:
                    continue
                end_word_local = start_word_local + span_len_words
            if end_word_local > len(words):
                end_word_local = len(words)
            if end_word_local <= start_word_local:
                continue

            start_global = self._buffer_start_token + scan_start_local + start_word_local
            end_global = self._buffer_start_token + scan_start_local + end_word_local
            emit_key = (ref, start_global)
            if emit_key in self._emitted_keys:
                continue
            if self._is_duplicate_reference_span(ref, start_global, end_global):
                continue

            verses = item.get("verses") or []
            verse = verses[0] if verses else None
            original_text = " ".join(words[start_word_local:end_word_local])
            correct_text = str(item.get("correct_text", "") or "")
            if not correct_text and verse is not None:
                correct_text = verse.text

            self._emitted_keys.add(emit_key)
            self._record_emitted_reference_span(ref, start_global, end_global)
            emitted_refs_this_pass.add(ref)
            self._pending_by_ref.pop(ref, None)
            emitted.append(StreamingQuranHit(
                original_text=original_text,
                start_token=start_global,
                end_token=end_global,
                correct_text=correct_text,
                reference=ref,
                confidence=conf,
                verses=[verse] if verse else [],
                needs_correction=bool(item.get("needs_correction", False)),
                translations=dict(item.get("translations") or {}),
            ))

        return emitted

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
        min_confidence: Optional[float] = None,
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

        threshold = self.options.min_approx_confidence
        if min_confidence is not None:
            threshold = min_confidence
        if confidence < threshold:
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

    def _build_token_idf(self) -> None:
        """Build per-token IDF statistics over Quran verses for partial scoring."""
        verse_tokens_iter = self.validator._normalized_verse_words_by_id.values()
        df: Counter[str] = Counter()
        verse_count = 0
        for toks in verse_tokens_iter:
            verse_count += 1
            for tok in set(t for t in toks if t):
                df[tok] += 1

        if verse_count <= 0:
            self._token_idf = {}
            self._max_token_idf = self._DEFAULT_TOKEN_IDF
            return

        token_idf: dict[str, float] = {}
        max_idf = 0.0
        for tok, freq in df.items():
            idf = math.log((1.0 + verse_count) / (1.0 + freq)) + 1.0
            token_idf[tok] = idf
            if idf > max_idf:
                max_idf = idf

        self._token_idf = token_idf
        self._max_token_idf = max(max_idf, self._DEFAULT_TOKEN_IDF)

    def _align_fragment_to_verse(
        self,
        window_tokens: tuple[str, ...],
        verse_tokens: tuple[str, ...],
        *,
        min_match_tokens: int,
    ) -> Optional[_AlignedFragment]:
        """Find a dense aligned token fragment inside a candidate window."""
        if not window_tokens or not verse_tokens:
            return None

        pairs = self._lcs_token_pairs(window_tokens, verse_tokens)
        if len(pairs) < max(1, min_match_tokens):
            return None

        # Pick the densest aligned region (maximize matches, minimize noise).
        best_start = 0
        best_end = len(window_tokens)
        best_score = float("-inf")
        best_match_count = 0
        best_span_len = len(window_tokens)
        k = len(pairs)
        for i in range(k):
            start_i = pairs[i][0]
            for j in range(i, k):
                end_i = pairs[j][0] + 1
                match_count = j - i + 1
                span_len = max(1, end_i - start_i)
                noise = span_len - match_count
                score = (1.15 * match_count) - (0.45 * noise)
                if (
                    score > best_score
                    or (score == best_score and match_count > best_match_count)
                    or (
                        score == best_score
                        and match_count == best_match_count
                        and span_len < best_span_len
                    )
                ):
                    best_score = score
                    best_start = start_i
                    best_end = end_i
                    best_match_count = match_count
                    best_span_len = span_len

        selected_pairs = [(w_i, v_i) for (w_i, v_i) in pairs if best_start <= w_i < best_end]
        if len(selected_pairs) < max(1, min_match_tokens):
            return None

        verse_positions = [v_i for _w_i, v_i in selected_pairs]
        verse_start = min(verse_positions)
        verse_end = max(verse_positions) + 1

        overlap_count = self._token_multiset_overlap_count(
            window_tokens[best_start:best_end],
            verse_tokens,
        )
        contiguous_run = self._longest_monotonic_run(selected_pairs)

        return _AlignedFragment(
            start_offset=best_start,
            end_offset=best_end,
            verse_start=verse_start,
            verse_end=verse_end,
            lcs_count=len(selected_pairs),
            overlap_count=overlap_count,
            contiguous_run=contiguous_run,
        )

    @staticmethod
    def _lcs_token_pairs(
        window_tokens: tuple[str, ...],
        verse_tokens: tuple[str, ...],
    ) -> list[tuple[int, int]]:
        """Return one deterministic LCS alignment as (window_idx, verse_idx) pairs."""
        n = len(window_tokens)
        m = len(verse_tokens)
        if n == 0 or m == 0:
            return []

        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            w_tok = window_tokens[i - 1]
            row = dp[i]
            prev_row = dp[i - 1]
            for j in range(1, m + 1):
                if w_tok == verse_tokens[j - 1]:
                    row[j] = prev_row[j - 1] + 1
                else:
                    left = row[j - 1]
                    up = prev_row[j]
                    row[j] = up if up >= left else left

        pairs_rev: list[tuple[int, int]] = []
        i = n
        j = m
        while i > 0 and j > 0:
            if (
                window_tokens[i - 1] == verse_tokens[j - 1]
                and dp[i][j] == dp[i - 1][j - 1] + 1
            ):
                pairs_rev.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue

            if dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1

        return list(reversed(pairs_rev))

    @staticmethod
    def _longest_monotonic_run(pairs: list[tuple[int, int]]) -> int:
        """Longest contiguous monotonic run in an aligned token pair list."""
        if not pairs:
            return 0
        best = 1
        run = 1
        for idx in range(1, len(pairs)):
            prev_w, prev_v = pairs[idx - 1]
            cur_w, cur_v = pairs[idx]
            if cur_w == (prev_w + 1) and cur_v == (prev_v + 1):
                run += 1
            else:
                run = 1
            if run > best:
                best = run
        return best

    @staticmethod
    def _token_multiset_overlap_ratio(
        input_tokens: tuple[str, ...],
        verse_tokens: tuple[str, ...],
    ) -> float:
        """Multiset token overlap normalized by verse length."""
        if not input_tokens or not verse_tokens:
            return 0.0

        overlap = StreamingQuranMatcher._token_multiset_overlap_count(input_tokens, verse_tokens)
        return overlap / max(1, len(verse_tokens))

    @staticmethod
    def _token_multiset_overlap_count(
        input_tokens: tuple[str, ...],
        verse_tokens: tuple[str, ...],
    ) -> int:
        """Count multiset overlap between input and verse token bags."""
        if not input_tokens or not verse_tokens:
            return 0
        input_counts = Counter(input_tokens)
        verse_counts = Counter(verse_tokens)
        overlap = 0
        for token, v_count in verse_counts.items():
            overlap += min(v_count, input_counts.get(token, 0))
        return overlap

    @staticmethod
    def _ordered_token_match_count(
        input_tokens: tuple[str, ...],
        verse_tokens: tuple[str, ...],
    ) -> int:
        """Greedy monotonic token match count against verse token order."""
        if not input_tokens or not verse_tokens:
            return 0
        positions: dict[str, list[int]] = {}
        for idx, token in enumerate(verse_tokens):
            if token:
                positions.setdefault(token, []).append(idx)

        matched = 0
        last_pos = -1
        for token in input_tokens:
            if not token:
                continue
            pos_list = positions.get(token)
            if not pos_list:
                continue
            k = bisect_left(pos_list, last_pos + 1)
            if k >= len(pos_list):
                continue
            last_pos = pos_list[k]
            matched += 1
            if last_pos >= (len(verse_tokens) - 1):
                break
        return matched

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable logistic helper."""
        if x >= 40.0:
            return 1.0
        if x <= -40.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _fragment_char_count(tokens: tuple[str, ...]) -> int:
        """Count non-space characters in a token fragment."""
        return sum(len(t) for t in tokens if t)

    def _partial_length_factor(self, matched_tokens: int) -> float:
        """Length-aware attenuation so short fragments cannot look perfect."""
        center = float(self.options.partial_length_sigmoid_center)
        scale = max(0.25, float(self.options.partial_length_sigmoid_scale))
        base = self._sigmoid((float(matched_tokens) - center) / scale)
        return 0.40 + (0.60 * base)

    def _collision_phrase_ratio(
        self,
        fragment_tokens: tuple[str, ...],
        *,
        overlap_count: int,
    ) -> float:
        """
        Measure how much overlap mass comes from high-collision phrase templates.
        """
        if not fragment_tokens:
            return 0.0
        marks = [False] * len(fragment_tokens)
        for phrase in self._HIGH_COLLISION_PARTIAL_PHRASES:
            p_tokens = tuple(phrase.split())
            n = len(p_tokens)
            if n <= 0 or n > len(fragment_tokens):
                continue
            limit = len(fragment_tokens) - n + 1
            for i in range(limit):
                if fragment_tokens[i:i + n] == p_tokens:
                    for j in range(i, i + n):
                        marks[j] = True
        collision_tokens = sum(1 for m in marks if m)
        if collision_tokens <= 0:
            return 0.0
        return min(1.0, collision_tokens / max(1, overlap_count))

    @staticmethod
    def _span_overlap_ratio(
        a_start: int,
        a_end: int,
        b_start: int,
        b_end: int,
    ) -> float:
        """Normalized span overlap by shorter span length."""
        overlap = min(a_end, b_end) - max(a_start, b_start)
        if overlap <= 0:
            return 0.0
        a_len = max(1, a_end - a_start)
        b_len = max(1, b_end - b_start)
        return overlap / max(1, min(a_len, b_len))

    def _suppress_overlapping_approx_matches(
        self,
        candidates: list[_ApproxMatch],
        *,
        overlap_ratio: float,
        limit: int,
    ) -> list[_ApproxMatch]:
        """Keep strongest non-overlapping candidate windows for partial emits."""
        if not candidates or limit <= 0:
            return []
        threshold = max(0.0, min(1.0, overlap_ratio))
        ranked = sorted(
            candidates,
            key=lambda m: (
                -m.approx_confidence,
                -m.lexical_overlap,
                -m.anchor_mass_ratio,
                -(m.end_local - m.start_local),
                m.start_local,
            ),
        )
        selected: list[_ApproxMatch] = []
        for cand in ranked:
            if len(selected) >= limit:
                break
            suppress = False
            for prev in selected:
                ov = self._span_overlap_ratio(
                    cand.start_local,
                    cand.end_local,
                    prev.start_local,
                    prev.end_local,
                )
                if ov >= threshold:
                    suppress = True
                    break
            if suppress:
                continue
            selected.append(cand)
        return selected

    def _complete_passes_quality_gates(
        self,
        *,
        aligned: Optional[_AlignedFragment],
        window_tokens: tuple[str, ...],
        verse_tokens: tuple[str, ...],
        confidence: float,
    ) -> tuple[bool, int, int, float, float]:
        """Apply final complete-hit quality gates using aligned-fragment metrics."""
        if not window_tokens or not verse_tokens:
            return False, 0, 0, 0.0, 0.0

        if aligned is not None:
            matched_tokens = int(aligned.overlap_count)
            boundary_left = 1.0 if aligned.verse_start <= 0 else 0.0
            boundary_right = 1.0 if aligned.verse_end >= len(verse_tokens) else 0.0
            boundary_ratio = 0.5 * (boundary_left + boundary_right)
        else:
            matched_tokens = self._token_multiset_overlap_count(window_tokens, verse_tokens)
            if len(verse_tokens) == 1:
                boundary_ratio = 1.0 if verse_tokens[0] in window_tokens else 0.0
            else:
                first_tok = verse_tokens[0]
                last_tok = verse_tokens[-1]
                boundary_ratio = (
                    (1.0 if first_tok in window_tokens else 0.0)
                    + (1.0 if last_tok in window_tokens else 0.0)
                ) / 2.0

        fragment_chars = self._fragment_char_count(window_tokens)
        coverage_ratio = matched_tokens / max(1, len(verse_tokens))

        is_short_verse = len(verse_tokens) <= self.options.complete_short_verse_words
        if is_short_verse:
            ok = (
                confidence >= self.options.complete_short_verse_min_confidence
                and coverage_ratio >= self.options.complete_short_verse_min_coverage
            )
            if self.options.complete_short_verse_require_full_boundary:
                ok = ok and (boundary_ratio >= 1.0)
            return ok, matched_tokens, fragment_chars, coverage_ratio, boundary_ratio

        ok = (
            matched_tokens >= self.options.complete_min_matched_tokens
            and fragment_chars >= self.options.complete_min_fragment_chars
        )
        if not ok:
            return False, matched_tokens, fragment_chars, coverage_ratio, boundary_ratio

        if (
            coverage_ratio >= self.options.complete_min_coverage_ratio
            and boundary_ratio >= self.options.complete_min_boundary_ratio
        ):
            return True, matched_tokens, fragment_chars, coverage_ratio, boundary_ratio

        # Fallback for noisy-ASR complete hits: if the aligned fragment carries
        # enough lexical mass, allow complete emission even with weak boundaries.
        strong_token_mass = matched_tokens >= max(8, int(0.55 * len(verse_tokens)))
        high_confidence = confidence >= max(self.options.min_confidence + 0.03, 0.86)
        ok = strong_token_mass or high_confidence
        return ok, matched_tokens, fragment_chars, coverage_ratio, boundary_ratio

    @staticmethod
    def _partial_label(
        *,
        coverage_ratio: float,
        purity_ratio: float,
        order_ratio: float,
    ) -> str:
        """Bucket partial quality into stable labels for downstream logic."""
        if coverage_ratio >= 0.82 and order_ratio >= 0.82 and purity_ratio >= 0.55:
            return "near_complete"
        if coverage_ratio >= 0.60 and order_ratio >= 0.70:
            return "strong"
        if coverage_ratio >= 0.45 and purity_ratio >= 0.35:
            return "medium"
        return "weak"

    def _build_partial_hit(
        self,
        *,
        verse: QuranVerse,
        verse_tokens: tuple[str, ...],
        start_local: int,
        end_local: int,
        start_global: int,
        end_global: int,
        approx_confidence: float,
    ) -> Optional[StreamingQuranHit]:
        """Build a partial hit candidate from a scored approximate window."""
        if end_local <= start_local:
            return None
        if start_local < 0 or end_local > len(self._buffer_words):
            return None
        if not verse_tokens:
            return None

        full_window_norm = tuple(self._buffer_norm_words[start_local:end_local])
        if not full_window_norm:
            return None
        aligned = self._align_fragment_to_verse(
            full_window_norm,
            verse_tokens,
            min_match_tokens=max(1, self.options.partial_min_overlap_tokens),
        )
        if aligned is None:
            return None

        frag_start_local = start_local + aligned.start_offset
        frag_end_local = start_local + aligned.end_offset
        if frag_end_local <= frag_start_local:
            return None

        window_norm = tuple(self._buffer_norm_words[frag_start_local:frag_end_local])
        if not window_norm:
            return None
        fragment_norm_text = " ".join(window_norm)
        if fragment_norm_text in self._HIGH_COLLISION_PARTIAL_FRAGMENTS:
            return None

        min_fragment_words = max(
            int(self.options.partial_min_words),
            int(self.options.partial_min_fragment_words),
        )
        if len(window_norm) < min_fragment_words:
            return None

        overlap_count = aligned.overlap_count
        if overlap_count < max(1, self.options.partial_min_overlap_tokens):
            return None
        if overlap_count < max(1, self.options.partial_min_words):
            return None

        fragment_chars = self._fragment_char_count(window_norm)
        coverage_ratio = overlap_count / max(1, len(verse_tokens))
        purity_ratio_raw = overlap_count / max(1, len(window_norm))
        ordered_count = self._ordered_token_match_count(window_norm, verse_tokens)
        order_ratio_raw = ordered_count / max(1, overlap_count)
        length_factor = self._partial_length_factor(overlap_count)
        purity_ratio = purity_ratio_raw * length_factor
        order_ratio = order_ratio_raw * length_factor
        collision_ratio = self._collision_phrase_ratio(
            window_norm,
            overlap_count=overlap_count,
        )
        if collision_ratio >= self.options.partial_collision_reject_ratio:
            return None

        matched_idf_mass = 0.0
        input_counts = Counter(window_norm)
        verse_counts = Counter(verse_tokens)
        for token, v_count in verse_counts.items():
            m = min(v_count, input_counts.get(token, 0))
            if m <= 0:
                continue
            matched_idf_mass += self._token_idf.get(token, self._DEFAULT_TOKEN_IDF) * m
        avg_matched_idf = matched_idf_mass / max(1, overlap_count)
        distinctiveness_ratio = avg_matched_idf / max(self._max_token_idf, self._DEFAULT_TOKEN_IDF)
        distinctiveness_ratio = max(0.0, min(1.0, distinctiveness_ratio))

        if len(verse_tokens) == 1:
            boundary_ratio = 1.0 if verse_tokens[0] in window_norm else 0.0
        else:
            first_tok = verse_tokens[0]
            last_tok = verse_tokens[-1]
            boundary_ratio = (
                (1.0 if first_tok in window_norm else 0.0)
                + (1.0 if last_tok in window_norm else 0.0)
            ) / 2.0

        approx_norm = max(0.0, min(1.0, (approx_confidence - 0.40) / 0.60))
        partial_score = (
            0.34 * coverage_ratio
            + 0.22 * purity_ratio
            + 0.18 * order_ratio
            + 0.08 * boundary_ratio
            + 0.08 * approx_norm
            + 0.10 * distinctiveness_ratio
        )
        if collision_ratio > self.options.partial_collision_penalty_threshold:
            span = max(1e-6, 1.0 - self.options.partial_collision_penalty_threshold)
            excess = (collision_ratio - self.options.partial_collision_penalty_threshold) / span
            partial_score -= self.options.partial_collision_penalty_weight * excess
        # Hard floor for very short aligned fragments.
        if (
            overlap_count < self.options.complete_min_matched_tokens
            or fragment_chars < self.options.complete_min_fragment_chars
        ):
            partial_score = min(
                partial_score,
                min(self.options.partial_min_score - 1e-6, 0.44),
            )
        partial_score = max(0.0, min(0.99, partial_score))

        if distinctiveness_ratio < self.options.partial_min_distinctiveness:
            return None
        if len(window_norm) <= max(5, self.options.partial_short_verse_words):
            if distinctiveness_ratio < self.options.partial_min_distinctiveness_short:
                return None

        if partial_score < self.options.partial_min_score:
            return None
        if len(verse_tokens) <= self.options.partial_short_verse_words:
            if order_ratio_raw < self.options.partial_short_min_order_ratio:
                return None
            if self.options.partial_short_require_boundary and boundary_ratio < 0.50:
                return None

        label = self._partial_label(
            coverage_ratio=coverage_ratio,
            purity_ratio=purity_ratio,
            order_ratio=order_ratio,
        )
        original_text = " ".join(self._buffer_words[frag_start_local:frag_end_local])
        verse_words = verse.text.split()
        verse_start = max(0, min(aligned.verse_start, len(verse_words)))
        verse_end = max(verse_start + 1, min(aligned.verse_end, len(verse_words)))
        correct_fragment = " ".join(verse_words[verse_start:verse_end]) if verse_words else verse.text
        translations: dict[str, str] = {}
        if self._translation_provider is not None:
            translations = self._translation_provider.get_translations(
                verse.surah,
                verse.ayah,
            )
        return StreamingQuranHit(
            original_text=original_text,
            start_token=self._buffer_start_token + frag_start_local,
            end_token=self._buffer_start_token + frag_end_local,
            correct_text=correct_fragment,
            reference=f"{verse.surah}:{verse.ayah}",
            confidence=partial_score,
            verses=[verse],
            needs_correction=True,
            translations=translations,
            is_partial=True,
            partial_score=partial_score,
            coverage_ratio=coverage_ratio,
            purity_ratio=purity_ratio,
            order_ratio=order_ratio,
            boundary_ratio=boundary_ratio,
            distinctiveness_ratio=distinctiveness_ratio,
            matched_tokens=overlap_count,
            fragment_chars=fragment_chars,
            collision_ratio=collision_ratio,
            partial_label=label,
        )

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
        return self._is_duplicate_reference_span_from_list(
            prior_spans,
            start_token=start_token,
            end_token=end_token,
            duplicate_ref_window_tokens=self.options.duplicate_ref_window_tokens,
        )

    @staticmethod
    def _is_duplicate_reference_span_from_list(
        prior_spans: Optional[list[tuple[int, int]]],
        *,
        start_token: int,
        end_token: int,
        duplicate_ref_window_tokens: int,
    ) -> bool:
        """Check overlap/near-adjacent duplication against a span list."""
        if not prior_spans:
            return False

        span_len = end_token - start_token
        for prev_start, prev_end in prior_spans:
            prev_len = max(1, prev_end - prev_start)
            overlap = min(end_token, prev_end) - max(start_token, prev_start)
            if overlap <= 0:
                # Also suppress near-adjacent repeats from chunk overlap carryover.
                if abs(start_token - prev_start) <= duplicate_ref_window_tokens:
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

        self._record_reference_span_to_store(
            self._emitted_spans_by_ref,
            reference=reference,
            start_token=start_token,
            end_token=end_token,
            max_keep=48,
        )

    def _is_duplicate_partial_reference_span(
        self,
        reference: str,
        start_token: int,
        end_token: int,
    ) -> bool:
        """Check if this partial span is a duplicate of prior partial emissions."""
        if not reference or end_token <= start_token:
            return False
        prior_spans = self._partial_spans_by_ref.get(reference)
        return self._is_duplicate_reference_span_from_list(
            prior_spans,
            start_token=start_token,
            end_token=end_token,
            duplicate_ref_window_tokens=self.options.duplicate_ref_window_tokens,
        )

    def _record_partial_reference_span(
        self,
        reference: str,
        start_token: int,
        end_token: int,
    ) -> None:
        """Remember emitted partial span for duplicate suppression."""
        if not reference or end_token <= start_token:
            return
        self._record_reference_span_to_store(
            self._partial_spans_by_ref,
            reference=reference,
            start_token=start_token,
            end_token=end_token,
            max_keep=64,
        )

    @staticmethod
    def _record_reference_span_to_store(
        store: dict[str, list[tuple[int, int]]],
        *,
        reference: str,
        start_token: int,
        end_token: int,
        max_keep: int,
    ) -> None:
        """Append span into the provided store while bounding memory."""
        spans = store.setdefault(reference, [])
        spans.append((start_token, end_token))
        # Keep bounded memory for long live streams.
        if len(spans) > max_keep:
            del spans[:-max_keep]

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

        ranked = sorted(
            self._active.values(),
            key=lambda h: (
                -(h.anchor_score + h.best_confidence * 8.0),
                -h.last_anchor_token,
                h.candidate_start_token,
                h.verse_id,
            ),
        )
        selected = self._select_diverse_hypotheses(ranked, limit=self.options.beam_size)
        keep = {(h.verse_id, h.candidate_start_token) for h in selected}
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
