"""
Streaming scanner for detecting Quranic verses across ASR chunks.

This module provides the StreamingScanner class which maintains state
across text chunks to detect verses that may be split at chunk boundaries.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .normalizer import normalize_arabic
from .types import QuranVerse, ValidatorOptions

if TYPE_CHECKING:
    from .translations import TranslationProvider


@dataclass
class ScannedVerse:
    """A Quranic verse detected in scanned text."""

    # The text as it appears in the input
    original_text: str
    # Start/end character positions in the scanned text
    start_pos: int
    end_pos: int
    # The correct Quranic text (Uthmani)
    correct_text: str
    # Verse reference (e.g., "112:1" or "112:1-4" for ranges)
    reference: str
    # Confidence score (0.0 - 1.0)
    confidence: float
    # The matched verse(s)
    verses: list[QuranVerse] = field(default_factory=list)
    # Whether the original differs from correct
    needs_correction: bool = False
    # Translations (only populated when a TranslationProvider is configured)
    translations: dict[str, str] = field(default_factory=dict)


@dataclass
class PartialVerse:
    """A potential verse that spans a chunk boundary."""

    # Text accumulated so far
    candidate_text: str
    # Best matching verse(s) so far: list of (verse, similarity)
    candidates: list[tuple[QuranVerse, float]] = field(default_factory=list)
    # How many chunks this has spanned
    chunk_count: int = 1
    # Position in the original stream (character offset from stream start)
    stream_start_pos: int = 0


@dataclass
class StreamingResult:
    """Result from processing a single chunk."""

    # Completely detected verses in this chunk
    complete_verses: list[ScannedVerse] = field(default_factory=list)
    # Partial verse that may continue in next chunk
    partial_verse: Optional[PartialVerse] = None
    # The text that was consumed (for position tracking)
    consumed_text: str = ""


@dataclass
class StreamingScannerOptions:
    """Options for the streaming scanner."""

    # Number of words to retain in overlap buffer between chunks
    overlap_words: int = 10
    # Minimum confidence to consider a verse match
    min_confidence: float = 0.85
    # Minimum words to consider as a potential verse fragment
    min_words: int = 3
    # Maximum words in a sliding window
    max_words: int = 50
    # Maximum chunks a partial verse can span before being discarded
    max_chunk_span: int = 3
    # Whether to use ASR-tolerant matching
    asr_tolerant: bool = False


class StreamingScanner:
    """
    Stateful scanner that detects Quranic verses across ASR chunks.

    Maintains an overlap buffer so that verses split across chunk boundaries
    can be reassembled and detected.

    Example:
        >>> scanner = StreamingScanner()
        >>>
        >>> for chunk in asr_stream:
        ...     result = scanner.process_chunk(chunk.text)
        ...     for verse in result.complete_verses:
        ...         print(f"Found: {verse.reference}")
        >>>
        >>> final = scanner.flush()
    """

    def __init__(
        self,
        options: Optional[StreamingScannerOptions] = None,
        translation_provider: Optional[TranslationProvider] = None,
    ):
        if options is None:
            options = StreamingScannerOptions()
        self.options = options

        # Create validator with matching settings
        from .validator import QuranValidator

        validator_opts = ValidatorOptions(
            fuzzy_threshold=options.min_confidence * 0.9,
            asr_tolerant=options.asr_tolerant,
        )
        self.validator = QuranValidator(
            validator_opts,
            translation_provider=translation_provider,
        )
        self._translation_provider = translation_provider

        # State
        self._buffer_words: list[str] = []
        self._partial: Optional[PartialVerse] = None
        self._stream_offset: int = 0
        self._last_chunk_time: float = 0.0

    def process_chunk(self, text: str) -> StreamingResult:
        """
        Process an incoming text chunk.

        The chunk is prepended with the overlap buffer from the previous chunk,
        then scanned with a sliding window. Verses found entirely within the
        non-overlap portion are emitted as complete. Matches that touch the
        end of the chunk are tracked as partial.
        """
        self._last_chunk_time = time.monotonic()

        # Tokenize incoming chunk
        chunk_words = text.split()
        if not chunk_words:
            return StreamingResult(consumed_text=text)

        # Combine buffer + new chunk for scanning
        combined_words = self._buffer_words + chunk_words
        buffer_word_count = len(self._buffer_words)

        # Scan for verses using sliding window
        complete: list[ScannedVerse] = []
        best_partial: Optional[PartialVerse] = None

        # Track which word positions are already covered by a match
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
                window_text = ' '.join(combined_words[start:end])
                result = self.validator.validate(window_text)

                if result.is_valid and result.confidence > best_confidence:
                    best_confidence = result.confidence
                    best_match = result
                    best_end = end

            if best_match and best_confidence >= self.options.min_confidence:
                touches_end = best_end >= len(combined_words)

                if touches_end and best_match.match_type in ('partial', 'fuzzy'):
                    # Potential partial verse at chunk boundary
                    if best_partial is None or best_confidence > (
                        best_partial.candidates[0][1] if best_partial.candidates else 0
                    ):
                        best_partial = PartialVerse(
                            candidate_text=' '.join(combined_words[start:]),
                            candidates=(
                                [(best_match.matched_verse, best_confidence)]
                                if best_match.matched_verse
                                else []
                            ),
                            chunk_count=1,
                            stream_start_pos=self._stream_offset + start,
                        )
                else:
                    # Complete verse found
                    verse = best_match.matched_verse
                    sv = ScannedVerse(
                        original_text=' '.join(combined_words[start:best_end]),
                        start_pos=self._stream_offset + start,
                        end_pos=self._stream_offset + best_end,
                        correct_text=verse.text if verse else '',
                        reference=best_match.reference or '',
                        confidence=best_confidence,
                        verses=[verse] if verse else [],
                        needs_correction=(best_match.match_type != 'exact'),
                        translations=best_match.translations,
                    )
                    complete.append(sv)

                    for pos in range(start, best_end):
                        covered.add(pos)

        # Handle partial verse continuity from previous chunk
        if self._partial and best_partial:
            extended_text = self._partial.candidate_text + ' ' + best_partial.candidate_text
            result = self.validator.validate(extended_text)
            if result.is_valid and result.confidence > best_confidence:
                best_partial = PartialVerse(
                    candidate_text=extended_text,
                    candidates=(
                        [(result.matched_verse, result.confidence)]
                        if result.matched_verse
                        else []
                    ),
                    chunk_count=self._partial.chunk_count + 1,
                    stream_start_pos=self._partial.stream_start_pos,
                )
            elif result.is_valid and result.confidence >= self.options.min_confidence:
                # Previous partial completed with this chunk
                verse = result.matched_verse
                if verse:
                    sv = ScannedVerse(
                        original_text=extended_text,
                        start_pos=self._partial.stream_start_pos,
                        end_pos=self._stream_offset + len(chunk_words),
                        correct_text=verse.text,
                        reference=result.reference or '',
                        confidence=result.confidence,
                        verses=[verse],
                        needs_correction=(result.match_type != 'exact'),
                        translations=result.translations,
                    )
                    complete.append(sv)
                    best_partial = None

        # Discard partials that have spanned too many chunks
        if best_partial and best_partial.chunk_count > self.options.max_chunk_span:
            best_partial = None

        self._partial = best_partial

        # Update overlap buffer: keep last N words from this chunk
        self._buffer_words = chunk_words[-self.options.overlap_words:]
        self._stream_offset += len(chunk_words)

        return StreamingResult(
            complete_verses=complete,
            partial_verse=self._partial,
            consumed_text=text,
        )

    def flush(self) -> StreamingResult:
        """
        Flush remaining buffer at end of stream.

        Attempts to match any remaining partial verse against the database.
        """
        if not self._partial and not self._buffer_words:
            return StreamingResult()

        # Try to match whatever is left
        remaining = (
            self._partial.candidate_text
            if self._partial
            else ' '.join(self._buffer_words)
        )
        result = self.validator.validate(remaining)

        complete: list[ScannedVerse] = []
        if result.is_valid and result.confidence >= self.options.min_confidence:
            verse = result.matched_verse
            if verse:
                complete.append(ScannedVerse(
                    original_text=remaining,
                    start_pos=self._stream_offset,
                    end_pos=self._stream_offset + len(remaining),
                    correct_text=verse.text,
                    reference=result.reference or '',
                    confidence=result.confidence,
                    verses=[verse],
                    needs_correction=(result.match_type != 'exact'),
                    translations=result.translations,
                ))

        self.reset()
        return StreamingResult(complete_verses=complete, consumed_text=remaining)

    def reset(self) -> None:
        """Reset all state for a new stream."""
        self._buffer_words = []
        self._partial = None
        self._stream_offset = 0
        self._last_chunk_time = 0.0
