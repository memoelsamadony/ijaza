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
        self.normalized_verse_map: dict[str, list[QuranVerse]] = {}

        for verse in self.verses:
            # ID lookup
            self.verse_by_id[verse.id] = verse

            # Normalized text lookup
            normalized = normalize_arabic(verse.text)
            if normalized not in self.normalized_verse_map:
                self.normalized_verse_map[normalized] = []
            self.normalized_verse_map[normalized].append(verse)

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
        Scan continuous Arabic text for Quranic verses using a sliding window.

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

    # Private helper methods

    def _find_exact_match(self, text: str) -> Optional[QuranVerse]:
        """Find a verse with exact text match."""
        for v in self.verses:
            if v.text == text:
                return v
        return None

    def _find_partial_match(
        self,
        normalized_input: str
    ) -> Optional[dict]:
        """Find verses where input is a substring or vice versa."""
        for verse in self.verses:
            normalized_verse = normalize_arabic(verse.text)

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

        for verse in self.verses:
            similarity = self._calculate_verse_match(normalized_input, verse)

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
        verse: QuranVerse
    ) -> float:
        """Calculate similarity between input and verse."""
        normalized_verse = normalize_arabic(verse.text)

        if self.options.asr_tolerant:
            # Character-level phonetic-aware similarity
            char_sim = calculate_asr_similarity(normalized_input, normalized_verse)

            # Word-level similarity (catches dropped function words)
            input_words = normalized_input.split()
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
