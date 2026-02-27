"""
Legacy approach: original StreamingScanner (validate-per-window loop).

This preserves the initial stateful scanner used before the new streaming
matcher experiments.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ijaza.types import QuranVerse, ValidatorOptions


@dataclass
class LegacyScannedVerse:
    original_text: str
    start_pos: int
    end_pos: int
    correct_text: str
    reference: str
    confidence: float
    verses: list[QuranVerse] = field(default_factory=list)
    needs_correction: bool = False
    translations: dict[str, str] = field(default_factory=dict)


@dataclass
class LegacyPartialVerse:
    candidate_text: str
    candidates: list[tuple[QuranVerse, float]] = field(default_factory=list)
    chunk_count: int = 1
    stream_start_pos: int = 0


@dataclass
class LegacyStreamingResult:
    complete_verses: list[LegacyScannedVerse] = field(default_factory=list)
    partial_verse: Optional[LegacyPartialVerse] = None
    consumed_text: str = ""


@dataclass
class LegacyStreamingScannerOptions:
    overlap_words: int = 10
    min_confidence: float = 0.85
    min_words: int = 3
    max_words: int = 50
    max_chunk_span: int = 3
    asr_tolerant: bool = False


class LegacyStreamingScanner:
    """
    Original streaming scanner that runs validate() on many windows per chunk.
    """

    def __init__(
        self,
        options: Optional[LegacyStreamingScannerOptions] = None,
        translation_provider=None,
    ):
        if options is None:
            options = LegacyStreamingScannerOptions()
        self.options = options

        from ijaza.validator import QuranValidator

        validator_opts = ValidatorOptions(
            fuzzy_threshold=options.min_confidence * 0.9,
            asr_tolerant=options.asr_tolerant,
        )
        self.validator = QuranValidator(
            validator_opts,
            translation_provider=translation_provider,
        )

        self._buffer_words: list[str] = []
        self._partial: Optional[LegacyPartialVerse] = None
        self._stream_offset: int = 0
        self._last_chunk_time: float = 0.0

    def process_chunk(self, text: str) -> LegacyStreamingResult:
        self._last_chunk_time = time.monotonic()

        chunk_words = text.split()
        if not chunk_words:
            return LegacyStreamingResult(consumed_text=text)

        combined_words = self._buffer_words + chunk_words
        complete: list[LegacyScannedVerse] = []
        best_partial: Optional[LegacyPartialVerse] = None
        covered: set[int] = set()

        for start in range(len(combined_words)):
            if start in covered:
                continue

            best_match = None
            best_confidence = 0.0
            best_end = start

            for end in range(
                start + self.options.min_words,
                min(start + self.options.max_words + 1, len(combined_words) + 1),
            ):
                window_text = " ".join(combined_words[start:end])
                result = self.validator.validate(window_text)
                if result.is_valid and result.confidence > best_confidence:
                    best_confidence = result.confidence
                    best_match = result
                    best_end = end

            if best_match and best_confidence >= self.options.min_confidence:
                touches_end = best_end >= len(combined_words)

                if touches_end and best_match.match_type in ("partial", "fuzzy"):
                    if best_partial is None or best_confidence > (
                        best_partial.candidates[0][1] if best_partial.candidates else 0.0
                    ):
                        best_partial = LegacyPartialVerse(
                            candidate_text=" ".join(combined_words[start:]),
                            candidates=(
                                [(best_match.matched_verse, best_confidence)]
                                if best_match.matched_verse
                                else []
                            ),
                            chunk_count=1,
                            stream_start_pos=self._stream_offset + start,
                        )
                else:
                    verse = best_match.matched_verse
                    complete.append(LegacyScannedVerse(
                        original_text=" ".join(combined_words[start:best_end]),
                        start_pos=self._stream_offset + start,
                        end_pos=self._stream_offset + best_end,
                        correct_text=verse.text if verse else "",
                        reference=best_match.reference or "",
                        confidence=best_confidence,
                        verses=[verse] if verse else [],
                        needs_correction=(best_match.match_type != "exact"),
                        translations=best_match.translations,
                    ))
                    for pos in range(start, best_end):
                        covered.add(pos)

        if self._partial and best_partial:
            extended_text = self._partial.candidate_text + " " + best_partial.candidate_text
            result = self.validator.validate(extended_text)
            if result.is_valid and result.confidence >= self.options.min_confidence:
                verse = result.matched_verse
                if verse:
                    complete.append(LegacyScannedVerse(
                        original_text=extended_text,
                        start_pos=self._partial.stream_start_pos,
                        end_pos=self._stream_offset + len(chunk_words),
                        correct_text=verse.text,
                        reference=result.reference or "",
                        confidence=result.confidence,
                        verses=[verse],
                        needs_correction=(result.match_type != "exact"),
                        translations=result.translations,
                    ))
                    best_partial = None

        if best_partial and best_partial.chunk_count > self.options.max_chunk_span:
            best_partial = None

        self._partial = best_partial
        self._buffer_words = chunk_words[-self.options.overlap_words:]
        self._stream_offset += len(chunk_words)

        return LegacyStreamingResult(
            complete_verses=complete,
            partial_verse=self._partial,
            consumed_text=text,
        )

    def flush(self) -> LegacyStreamingResult:
        if not self._partial and not self._buffer_words:
            return LegacyStreamingResult()

        remaining = self._partial.candidate_text if self._partial else " ".join(self._buffer_words)
        result = self.validator.validate(remaining)
        complete: list[LegacyScannedVerse] = []

        if result.is_valid and result.confidence >= self.options.min_confidence:
            verse = result.matched_verse
            if verse:
                complete.append(LegacyScannedVerse(
                    original_text=remaining,
                    start_pos=self._stream_offset,
                    end_pos=self._stream_offset + len(remaining),
                    correct_text=verse.text,
                    reference=result.reference or "",
                    confidence=result.confidence,
                    verses=[verse],
                    needs_correction=(result.match_type != "exact"),
                    translations=result.translations,
                ))

        self.reset()
        return LegacyStreamingResult(complete_verses=complete, consumed_text=remaining)

    def reset(self) -> None:
        self._buffer_words = []
        self._partial = None
        self._stream_offset = 0
        self._last_chunk_time = 0.0
