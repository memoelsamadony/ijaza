"""
Quran Validator - Validate and verify Quranic verses in text.

This module provides the main QuranValidator class for validating
Arabic text against the authentic Quran database.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .translations import TranslationProvider

from .types import (
    QuranVerse,
    QuranSurah,
    ValidationResult,
    DetectionResult,
    DetectionSegment,
    ValidatorOptions,
    MatchType,
    Suggestion,
)
from .normalizer import (
    normalize_arabic,
    contains_arabic,
    extract_arabic_segments,
    calculate_similarity,
    find_differences,
)
from .asr_tolerance import (
    calculate_asr_similarity,
    word_level_similarity,
    preprocess_asr_text,
)


# Default validator options
DEFAULT_OPTIONS = ValidatorOptions(
    fuzzy_threshold=0.8,
    max_suggestions=3,
    include_partial=True,
    min_detection_length=10,
)


def _word_ngrams(tokens: tuple[str, ...], n: int) -> set[str]:
    """Return unique word n-grams for a tokenized string."""
    if n <= 0 or len(tokens) < n:
        return set()
    return {
        ' '.join(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    }


def _iter_word_ngrams_with_pos(tokens: tuple[str, ...], n: int) -> list[tuple[int, str]]:
    """Return (start_index, ngram) pairs for tokenized text."""
    if n <= 0 or len(tokens) < n:
        return []
    return [
        (i, ' '.join(tokens[i:i + n]))
        for i in range(len(tokens) - n + 1)
    ]


_SCAN_TOKEN_EDGE_PUNCT = ".,:;!?،؛؟\"'()[]{}<>«»"


def _normalize_scan_token(token: str) -> str:
    """Normalize a token for n-gram indexing/lookup (keeps alignment stable)."""
    normalized = normalize_arabic(token)
    stripped = normalized.strip(_SCAN_TOKEN_EDGE_PUNCT)
    return stripped or normalized


def _load_json_data(filename: str) -> list:
    """Load bundled JSON data from package resources with a legacy fallback."""
    candidates = [filename, filename.replace('.json', '.min.json')]

    # Primary path: packaged resources (works for installed wheels)
    package_data_dir = resources.files('ijaza').joinpath('data')
    for candidate in candidates:
        resource_file = package_data_dir.joinpath(candidate)
        if resource_file.is_file():
            with resource_file.open('r', encoding='utf-8') as f:
                return json.load(f)

    # Legacy fallback: repository-level data/ directory (editable/dev setups)
    legacy_data_dir = Path(__file__).parent.parent / 'data'
    for candidate in candidates:
        legacy_file = legacy_data_dir / candidate
        if legacy_file.exists():
            with open(legacy_file, 'r', encoding='utf-8') as f:
                return json.load(f)

    raise FileNotFoundError(
        f"Data file not found: {filename}. "
        "Please ensure Quran data JSON files are bundled in 'ijaza/data'."
    )


def _parse_verse(data: dict) -> QuranVerse:
    """Parse a verse dictionary into a QuranVerse object."""
    return QuranVerse(
        id=data['id'],
        surah=data['surah'],
        ayah=data['ayah'],
        text=data['text'],
        text_simple=data.get('textSimple', ''),
        page=data.get('page', 0),
        juz=data.get('juz', 0),
    )


def _parse_surah(data: dict) -> QuranSurah:
    """Parse a surah dictionary into a QuranSurah object."""
    return QuranSurah(
        number=data['number'],
        name=data['name'],
        english_name=data['englishName'],
        verses_count=data['versesCount'],
        revelation_type=data['revelationType'],
    )


class QuranValidator:
    """
    QuranValidator - Validate and verify Quranic verses in text.

    This class provides methods for validating Arabic text against the
    authentic Quran database, detecting Quran quotes in text, and
    searching for verses.

    Example:
        >>> from ijaza import QuranValidator
        >>>
        >>> validator = QuranValidator()
        >>>
        >>> # Validate a specific quote
        >>> result = validator.validate("بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ")
        >>> print(result.is_valid)  # True
        >>> print(result.reference)  # "1:1"
        >>>
        >>> # Detect and validate all Quran quotes in text
        >>> detection = validator.detect_and_validate(llm_output)
        >>> for segment in detection.segments:
        ...     print(segment.text, segment.validation.is_valid if segment.validation else None)
    """

    def __init__(
        self,
        options: Optional[ValidatorOptions] = None,
        translation_provider: Optional['TranslationProvider'] = None,
    ):
        """
        Initialize the QuranValidator.

        Args:
            options: Validator options (uses defaults if not provided)
            translation_provider: Optional provider for verse translations
        """
        if options is None:
            self.options = ValidatorOptions()
        else:
            self.options = options

        self._translation_provider = translation_provider

        # Load verses and surahs from bundled data
        verses_data = _load_json_data('quran-verses.json')
        surahs_data = _load_json_data('quran-surahs.json')

        self.verses: list[QuranVerse] = [_parse_verse(v) for v in verses_data]
        self.surahs: list[QuranSurah] = [_parse_surah(s) for s in surahs_data]

        # Build lookup maps
        self.verse_by_id: dict[int, QuranVerse] = {}
        self.exact_verse_map: dict[str, QuranVerse] = {}
        self.normalized_verse_map: dict[str, list[QuranVerse]] = {}
        self._normalized_verse_text_by_id: dict[int, str] = {}
        self._normalized_verse_words_by_id: dict[int, tuple[str, ...]] = {}
        self._verse_word_count_by_id: dict[int, int] = {}
        self._ngram_index_by_n: dict[int, dict[str, list[int]]] = {
            1: {},
            2: {},
            3: {},
        }
        self._ngram_pos_index_by_n: dict[int, dict[str, list[tuple[int, int]]]] = {
            1: {},
            2: {},
            3: {},
        }

        for verse in self.verses:
            # ID lookup
            self.verse_by_id[verse.id] = verse
            self.exact_verse_map.setdefault(verse.text, verse)

            # Normalized text lookup
            normalized = normalize_arabic(verse.text)
            normalized_words = tuple(_normalize_scan_token(tok) for tok in normalized.split())
            self._normalized_verse_text_by_id[verse.id] = normalized
            self._normalized_verse_words_by_id[verse.id] = normalized_words
            self._verse_word_count_by_id[verse.id] = len(normalized_words)
            if normalized not in self.normalized_verse_map:
                self.normalized_verse_map[normalized] = []
            self.normalized_verse_map[normalized].append(verse)

            # Inverted index for fast fuzzy prefiltering
            for n in (1, 2, 3):
                if len(normalized_words) < n:
                    continue
                for pos, gram in _iter_word_ngrams_with_pos(normalized_words, n):
                    bucket = self._ngram_index_by_n[n].setdefault(gram, [])
                    bucket.append(verse.id)
                    pos_bucket = self._ngram_pos_index_by_n[n].setdefault(gram, [])
                    pos_bucket.append((verse.id, pos))

    def validate(self, text: str) -> ValidationResult:
        """
        Validate a potential Quran quote.

        Args:
            text: The Arabic text to validate

        Returns:
            Validation result with match details

        Example:
            >>> result = validator.validate("بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ")
            >>> if result.is_valid:
            ...     print(f"Found: {result.reference}")  # "1:1"
            ...     print(f"Match type: {result.match_type}")  # "exact"
        """
        trimmed_text = text.strip()

        # Early exit if not Arabic
        if not contains_arabic(trimmed_text):
            return self._no_match()

        # ASR preprocessing (stutter removal, word boundary fixes)
        if self.options.asr_tolerant:
            trimmed_text = preprocess_asr_text(trimmed_text)

        # Step 1: Try exact match (with diacritics)
        exact_match = self._find_exact_match(trimmed_text)
        if exact_match:
            return self._create_result(exact_match, 'exact', 1.0)

        # Step 2: Try normalized match (without diacritics)
        normalized_input = normalize_arabic(trimmed_text)
        normalized_matches = self.normalized_verse_map.get(normalized_input)

        if normalized_matches and len(normalized_matches) > 0:
            # Return first match with suggestions if multiple
            primary = normalized_matches[0]
            result = self._create_result(primary, 'normalized', 0.95)

            # Add differences for correction
            result.differences = find_differences(trimmed_text, primary.text)

            # Add suggestions if multiple matches
            if len(normalized_matches) > 1:
                result.suggestions = [
                    Suggestion(
                        verse=v,
                        confidence=0.95,
                        reference=f"{v.surah}:{v.ayah}",
                    )
                    for v in normalized_matches[:self.options.max_suggestions]
                ]

            return result

        # Step 3: Try partial match (substring)
        if self.options.include_partial:
            partial_match = self._find_partial_match(normalized_input)
            if partial_match:
                result = self._create_result(
                    partial_match['verse'],
                    'partial',
                    partial_match['confidence']
                )
                result.differences = find_differences(trimmed_text, partial_match['verse'].text)
                return result

        # Step 4: Try fuzzy match
        fuzzy_match = self._find_fuzzy_match(normalized_input)
        if fuzzy_match and fuzzy_match['confidence'] >= self.options.fuzzy_threshold:
            result = self._create_result(
                fuzzy_match['verse'],
                'fuzzy',
                fuzzy_match['confidence']
            )
            result.differences = find_differences(trimmed_text, fuzzy_match['verse'].text)
            result.suggestions = fuzzy_match['suggestions']
            return result

        # No match found
        return self._no_match()

    def detect_and_validate(self, text: str) -> DetectionResult:
        """
        Detect and validate all potential Quran quotes in text.

        This is useful for post-processing LLM output to find and verify
        any Quranic content.

        Args:
            text: Text that may contain Quran quotes

        Returns:
            Detection result with validated segments

        Example:
            >>> llm_output = "The verse بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ means..."
            >>> result = validator.detect_and_validate(llm_output)
            >>>
            >>> for segment in result.segments:
            ...     if segment.validation and segment.validation.is_valid:
            ...         print(f"Valid quote: {segment.text}")
            ...     else:
            ...         print(f"Possible misquote: {segment.text}")
        """
        # Extract Arabic segments
        arabic_segments = extract_arabic_segments(text)

        if len(arabic_segments) == 0:
            return DetectionResult(detected=False, segments=[])

        # Filter by minimum length and validate each
        validated_segments: list[DetectionSegment] = []
        for seg in arabic_segments:
            if len(seg.text) >= self.options.min_detection_length:
                validated_segments.append(DetectionSegment(
                    text=seg.text,
                    start_index=seg.start_index,
                    end_index=seg.end_index,
                    validation=self.validate(seg.text),
                ))

        # A detection is positive if we found any Arabic text (even if not Quran)
        detected = any(
            seg.validation and (seg.validation.is_valid or seg.validation.match_type == 'fuzzy')
            for seg in validated_segments
        )

        return DetectionResult(
            detected=detected,
            segments=validated_segments,
        )

    def get_verse(self, surah: int, ayah: int) -> Optional[QuranVerse]:
        """
        Get a verse by reference (surah:ayah).

        Args:
            surah: Surah number (1-114)
            ayah: Ayah number

        Returns:
            The verse or None if not found
        """
        for v in self.verses:
            if v.surah == surah and v.ayah == ayah:
                return v
        return None

    def get_verse_range(
        self,
        surah: int,
        start_ayah: int,
        end_ayah: int
    ) -> Optional[dict]:
        """
        Get a range of verses and concatenate their text.

        Args:
            surah: Surah number (1-114)
            start_ayah: Starting ayah number
            end_ayah: Ending ayah number

        Returns:
            Dict with 'text', 'text_simple', and 'verses' keys, or None if invalid range
        """
        if start_ayah > end_ayah:
            return None

        verses: list[QuranVerse] = []
        for ayah in range(start_ayah, end_ayah + 1):
            verse = self.get_verse(surah, ayah)
            if not verse:
                return None  # Invalid range
            verses.append(verse)

        return {
            'text': ' '.join(v.text for v in verses),
            'text_simple': ' '.join(v.text_simple for v in verses),
            'verses': verses,
        }

    def get_surah_verses(self, surah_number: int) -> list[QuranVerse]:
        """
        Get all verses in a surah.

        Args:
            surah_number: Surah number (1-114)

        Returns:
            List of verses in the surah
        """
        return [v for v in self.verses if v.surah == surah_number]

    def get_surah(self, surah_number: int) -> Optional[QuranSurah]:
        """
        Get surah information.

        Args:
            surah_number: Surah number (1-114)

        Returns:
            Surah info or None
        """
        for s in self.surahs:
            if s.number == surah_number:
                return s
        return None

    def get_all_surahs(self) -> list[QuranSurah]:
        """Get all surahs."""
        return list(self.surahs)

    def search(
        self,
        query: str,
        limit: int = 10
    ) -> list[dict]:
        """
        Search verses by text.

        Args:
            query: Search query (Arabic text)
            limit: Maximum results to return

        Returns:
            List of matching verses with similarity scores
        """
        normalized_query = normalize_arabic(query)

        results = []
        for verse in self.verses:
            similarity = self._calculate_verse_match(normalized_query, verse)
            if similarity > 0.3:
                results.append({
                    'verse': verse,
                    'similarity': similarity,
                })

        # Sort by similarity descending
        results.sort(key=lambda r: r['similarity'], reverse=True)

        return results[:limit]

    def scan_for_verses(
        self,
        text: str,
        min_words: int = 3,
        max_words: int = 50,
        confidence_threshold: float = 0.85,
    ) -> list[dict]:
        """
        Scan continuous Arabic text for Quranic verses using indexed anchors
        plus targeted window verification.

        Unlike validate() which checks if the entire input is a verse,
        this method finds verses embedded within longer Arabic text.

        Args:
            text: Arabic text to scan
            min_words: Minimum word count for a potential verse
            max_words: Maximum word count in the sliding window
            confidence_threshold: Minimum similarity to report

        Returns:
            List of dicts with 'original_text', 'correct_text', 'reference',
            'confidence', 'start_pos', 'end_pos', 'verses', 'needs_correction',
            'translations'
        """
        return self._scan_for_verses_indexed(
            text,
            min_words=min_words,
            max_words=max_words,
            confidence_threshold=confidence_threshold,
        )

    # Private helper methods

    def _scan_for_verses_sliding_window(
        self,
        text: str,
        min_words: int = 3,
        max_words: int = 50,
        confidence_threshold: float = 0.85,
    ) -> list[dict]:
        """
        Original exhaustive sliding-window scanner (kept as fallback/reference).
        """
        words = text.split()
        results: list[dict] = []
        covered: set[int] = set()

        for start in range(len(words)):
            if start in covered:
                continue

            best_match = None
            best_end = start
            best_confidence = 0.0

            for end in range(
                start + min_words,
                min(start + max_words + 1, len(words) + 1),
            ):
                window = ' '.join(words[start:end])
                result = self.validate(window)

                if result.is_valid and result.confidence > best_confidence:
                    best_confidence = result.confidence
                    best_match = result
                    best_end = end

            if best_match and best_confidence >= confidence_threshold:
                verse = best_match.matched_verse
                # Calculate character positions
                start_char = len(' '.join(words[:start])) + (1 if start > 0 else 0)
                end_char = start_char + len(' '.join(words[start:best_end]))

                results.append({
                    'original_text': ' '.join(words[start:best_end]),
                    'correct_text': verse.text if verse else '',
                    'reference': best_match.reference or '',
                    'confidence': best_confidence,
                    'start_pos': start_char,
                    'end_pos': end_char,
                    'verses': [verse] if verse else [],
                    'needs_correction': best_match.match_type != 'exact',
                    'translations': best_match.translations,
                })

                for pos in range(start, best_end):
                    covered.add(pos)

        return results

    def _scan_for_verses_indexed(
        self,
        text: str,
        min_words: int = 3,
        max_words: int = 50,
        confidence_threshold: float = 0.85,
    ) -> list[dict]:
        """
        Hit-driven scanner using word n-gram anchors to avoid exhaustive windows.

        This approximates the "Approach 2" design in docs/sliding-window-scanner.md:
        input n-gram hits propose candidate verse IDs and likely start positions,
        then only a small number of spans are verified.
        """
        words = text.split()
        if len(words) < min_words:
            return []

        normalized_input_words = tuple(_normalize_scan_token(w) for w in words)
        total_words = len(words)

        # Build anchor hypotheses: (candidate_start_word, verse_id) -> score
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
                for verse_id, verse_pos in self._ngram_pos_index_by_n[n].get(gram, ()):
                    verse_wc = self._verse_word_count_by_id.get(verse_id, 0)
                    # Allow some slack because ASR insertions/deletions can shift length.
                    if verse_wc < max(1, min_words - 2) or verse_wc > max_words + 6:
                        continue

                    base_start = input_pos - verse_pos
                    if base_start < 0 or base_start >= total_words:
                        continue

                    # Basic feasibility check for at least one nearby span.
                    min_candidate_len = max(min_words, verse_wc - 3)
                    if base_start + min_candidate_len > total_words:
                        continue

                    key = (base_start, verse_id)
                    anchor_scores[key] = anchor_scores.get(key, 0.0) + weight
                    local_hits += 1

            if local_hits and n >= 2:
                saw_strong_hits = True
                # After collecting tri+bi-gram evidence, skip noisy unigram pass.
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
                item[0][0],  # start position
                abs(self._verse_word_count_by_id.get(item[0][1], 0) - total_words),
                item[0][1],
            ),
        )

        max_anchors = 32 if self.options.asr_tolerant else 20
        ranked_anchors = ranked_anchors[:max_anchors]

        if self.options.asr_tolerant:
            start_deltas = (0, -1, 1, -2, 2)
            len_deltas = (0, -1, 1, -2, 2, -3, 3)
        else:
            start_deltas = (0, -1, 1)
            len_deltas = (0, -1, 1)

        checked_windows: set[tuple[int, int, int]] = set()
        provisional: list[dict] = []

        for (base_start, verse_id), anchor_score in ranked_anchors:
            verse = self.verse_by_id[verse_id]
            verse_wc = self._verse_word_count_by_id.get(verse_id, 0) or len(
                self._normalized_verse_words_by_id.get(verse_id, ())
            )
            if verse_wc <= 0:
                continue

            best_entry: Optional[dict] = None

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

                    window_text = ' '.join(words[start:end])
                    result = self._validate_against_verse(window_text, verse)
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

                    # Exact/normalized match on a top-ranked anchor is already precise.
                    if result.match_type in ('exact', 'normalized') and result.confidence >= 0.95:
                        stop_anchor = True
                        break

            if best_entry is not None:
                provisional.append(best_entry)

        if not provisional:
            return []

        # Greedy non-overlap merge (left-to-right) similar to original scanner.
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
            match_key = (start, end, result.reference or '')
            if match_key in seen_matches:
                continue
            if any(pos in covered for pos in range(start, end)):
                continue

            verse = result.matched_verse
            start_char = len(' '.join(words[:start])) + (1 if start > 0 else 0)
            end_char = start_char + len(' '.join(words[start:end]))

            selected.append({
                'original_text': ' '.join(words[start:end]),
                'correct_text': verse.text if verse else '',
                'reference': result.reference or '',
                'confidence': result.confidence,
                'start_pos': start_char,
                'end_pos': end_char,
                'verses': [verse] if verse else [],
                'needs_correction': result.match_type != 'exact',
                'translations': result.translations,
            })

            seen_matches.add(match_key)
            for pos in range(start, end):
                covered.add(pos)

        return selected

    def _validate_against_verse(self, text: str, verse: QuranVerse) -> ValidationResult:
        """
        Fast validation of a window against a single candidate verse.

        This mirrors the main validation pipeline but skips global corpus search.
        """
        trimmed_text = text.strip()
        if not contains_arabic(trimmed_text):
            return self._no_match()

        if self.options.asr_tolerant:
            trimmed_text = preprocess_asr_text(trimmed_text)

        if verse.text == trimmed_text:
            return self._create_result(verse, 'exact', 1.0)

        normalized_input = normalize_arabic(trimmed_text)
        normalized_verse = self._normalized_verse_text_by_id.get(verse.id, '')
        if normalized_input == normalized_verse:
            return self._create_result(verse, 'normalized', 0.95)

        if self.options.include_partial and normalized_input and normalized_verse:
            if normalized_input in normalized_verse:
                ratio = len(normalized_input) / len(normalized_verse)
                return self._create_result(verse, 'partial', 0.7 + ratio * 0.2)
            if normalized_verse in normalized_input:
                ratio = len(normalized_verse) / len(normalized_input)
                return self._create_result(verse, 'partial', 0.6 + ratio * 0.2)

        similarity = self._calculate_verse_match(
            normalized_input,
            verse,
            input_words=normalized_input.split(),
        )
        if similarity >= self.options.fuzzy_threshold:
            return self._create_result(verse, 'fuzzy', similarity)

        return self._no_match()

    def _candidate_verse_ids(
        self,
        normalized_input: str,
        *,
        for_partial: bool = False,
    ) -> list[int]:
        """
        Fast lexical prefilter for fuzzy/partial matching.

        Uses a word n-gram inverted index to shortlist candidate verses before
        running expensive similarity scoring. If no candidates are found, the
        caller can treat it as a likely non-Quran snippet and skip fuzzy scan.
        """
        tokens = tuple(normalized_input.split())
        if not tokens:
            return []

        input_word_count = len(tokens)
        candidate_scores: dict[int, float] = {}

        # Prefer stronger lexical evidence first (3-grams, then 2-grams).
        # Only fall back to unigram overlap when needed.
        if input_word_count >= 3:
            n_plan = ((3, 4.0), (2, 1.5), (1, 0.35))
        elif input_word_count == 2:
            n_plan = ((2, 2.0), (1, 0.4))
        else:
            n_plan = ((1, 1.0),)

        # Partial matching fragments can differ more in length than full-verse matching.
        if for_partial:
            min_wc = 1
            max_wc = input_word_count + 40
            candidate_limit = 192
        else:
            # Full-verse fuzzy matching should be close in length to the input.
            # A tighter band avoids scoring windows that include lots of extra
            # sermon words before/after an ayah (a major live-scan bottleneck).
            min_wc = max(1, int(input_word_count * 0.75))
            max_wc = max(min_wc, int(input_word_count * 1.40) + 1)
            candidate_limit = 128 if self.options.asr_tolerant else 64

        saw_strong_hits = False
        for n, weight in n_plan:
            if len(tokens) < n:
                continue
            grams = _word_ngrams(tokens, n)
            if not grams:
                continue

            local_hits = 0
            for gram in grams:
                for verse_id in self._ngram_index_by_n[n].get(gram, ()):
                    verse_wc = self._verse_word_count_by_id.get(verse_id, 0)
                    if verse_wc < min_wc or verse_wc > max_wc:
                        continue
                    candidate_scores[verse_id] = candidate_scores.get(verse_id, 0.0) + weight
                    local_hits += 1

            if local_hits and n >= 2:
                saw_strong_hits = True
                # In ASR mode, collect both tri-gram and bi-gram evidence.
                if not self.options.asr_tolerant or n == 2:
                    break

        # No lexical overlap => skip the expensive full-database fuzzy pass.
        # Exact/normalized checks already ran before this method is used.
        if not candidate_scores:
            return []

        # If we have strong n-gram hits, ignore noisy unigram-only candidates.
        if saw_strong_hits:
            filtered_scores: dict[int, float] = {}
            for verse_id, score in candidate_scores.items():
                if score >= 1.0:
                    filtered_scores[verse_id] = score
            if filtered_scores:
                candidate_scores = filtered_scores

        ranked = sorted(
            candidate_scores.items(),
            key=lambda item: (
                -item[1],
                abs(self._verse_word_count_by_id.get(item[0], 0) - input_word_count),
                item[0],
            ),
        )
        return [verse_id for verse_id, _ in ranked[:candidate_limit]]

    def _find_exact_match(self, text: str) -> Optional[QuranVerse]:
        """Find a verse with exact text match."""
        return self.exact_verse_map.get(text)

    def _find_partial_match(
        self,
        normalized_input: str
    ) -> Optional[dict]:
        """Find verses where input is a substring or vice versa."""
        candidate_ids = self._candidate_verse_ids(normalized_input, for_partial=True)
        verse_iter = (
            self.verse_by_id[verse_id] for verse_id in candidate_ids
        ) if candidate_ids else self.verses

        for verse in verse_iter:
            normalized_verse = self._normalized_verse_text_by_id.get(verse.id, '')

            # Input is contained in verse
            if normalized_input in normalized_verse:
                ratio = len(normalized_input) / len(normalized_verse)
                return {'verse': verse, 'confidence': 0.7 + ratio * 0.2}

            # Verse is contained in input
            if normalized_verse in normalized_input:
                ratio = len(normalized_verse) / len(normalized_input)
                return {'verse': verse, 'confidence': 0.6 + ratio * 0.2}

        return None

    def _find_fuzzy_match(self, normalized_input: str) -> Optional[dict]:
        """Find verses using fuzzy matching."""
        matches: list[dict] = []
        input_words = normalized_input.split()
        candidate_ids = self._candidate_verse_ids(normalized_input, for_partial=False)

        # If the lexical prefilter finds nothing, treat as non-Quran text and
        # avoid a full corpus fuzzy scan (too slow for live use).
        if not candidate_ids:
            return None

        for verse_id in candidate_ids:
            verse = self.verse_by_id[verse_id]
            similarity = self._calculate_verse_match(
                normalized_input,
                verse,
                input_words=input_words,
            )

            if similarity >= self.options.fuzzy_threshold * 0.9:
                matches.append({'verse': verse, 'similarity': similarity})

        if len(matches) == 0:
            return None

        # Sort by similarity
        matches.sort(key=lambda m: m['similarity'], reverse=True)

        best = matches[0]
        suggestions = [
            Suggestion(
                verse=m['verse'],
                confidence=m['similarity'],
                reference=f"{m['verse'].surah}:{m['verse'].ayah}",
            )
            for m in matches[:self.options.max_suggestions]
        ]

        return {
            'verse': best['verse'],
            'confidence': best['similarity'],
            'suggestions': suggestions,
        }

    def _calculate_verse_match(
        self,
        normalized_input: str,
        verse: QuranVerse,
        *,
        input_words: Optional[list[str]] = None,
    ) -> float:
        """Calculate similarity between input and verse."""
        normalized_verse = self._normalized_verse_text_by_id.get(verse.id)
        if normalized_verse is None:
            normalized_verse = normalize_arabic(verse.text)

        if self.options.asr_tolerant:
            # Character-level phonetic-aware similarity
            char_sim = calculate_asr_similarity(normalized_input, normalized_verse)

            # Word-level similarity (catches dropped function words)
            if input_words is None:
                input_words = normalized_input.split()
            verse_words = list(self._normalized_verse_words_by_id.get(verse.id, ()))
            if not verse_words:
                verse_words = normalized_verse.split()
            word_sim = word_level_similarity(input_words, verse_words)

            return max(char_sim, word_sim)

        return calculate_similarity(normalized_input, normalized_verse)

    def _create_result(
        self,
        verse: QuranVerse,
        match_type: MatchType,
        confidence: float
    ) -> ValidationResult:
        """Create a successful validation result."""
        translations: dict[str, str] = {}
        if self._translation_provider:
            translations = self._translation_provider.get_translations(
                verse.surah, verse.ayah
            )

        return ValidationResult(
            is_valid=True,
            match_type=match_type,
            confidence=confidence,
            matched_verse=verse,
            reference=f"{verse.surah}:{verse.ayah}",
            translations=translations,
        )

    def _no_match(self) -> ValidationResult:
        """Create a no-match validation result."""
        return ValidationResult(
            is_valid=False,
            match_type='none',
            confidence=0,
        )


def create_validator(options: Optional[ValidatorOptions] = None) -> QuranValidator:
    """
    Create a new QuranValidator instance.

    Args:
        options: Validator options

    Returns:
        QuranValidator instance

    Example:
        >>> from ijaza import create_validator
        >>>
        >>> validator = create_validator(ValidatorOptions(fuzzy_threshold=0.85))
    """
    return QuranValidator(options)
