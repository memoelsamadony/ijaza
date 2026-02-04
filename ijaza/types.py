"""
Type definitions for the Quran Validator package.

This module contains all the dataclasses and type aliases used throughout
the package for validation, detection, and configuration.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


# Type alias for match types
MatchType = Literal['exact', 'normalized', 'partial', 'fuzzy', 'none']


@dataclass
class QuranVerse:
    """Represents a single verse (ayah) from the Quran."""

    # Sequential verse number (1-6236)
    id: int
    # Surah (chapter) number (1-114)
    surah: int
    # Ayah (verse) number within the surah
    ayah: int
    # Full Arabic text with diacritics (Uthmani script)
    text: str
    # Simplified Arabic text without diacritics
    text_simple: str
    # Page number in standard Uthmani mushaf (optional)
    page: int = 0
    # Juz (part) number (1-30) (optional)
    juz: int = 0


@dataclass
class QuranSurah:
    """Represents a Surah (chapter) of the Quran."""

    # Surah number (1-114)
    number: int
    # Arabic name of the surah
    name: str
    # English name of the surah
    english_name: str
    # Number of verses in this surah
    verses_count: int
    # Revelation type: 'Meccan' or 'Medinan'
    revelation_type: Literal['Meccan', 'Medinan']


@dataclass
class Difference:
    """Represents a difference between input and correct text."""

    # What was provided
    input: str
    # What it should be
    correct: str
    # Position in text where difference starts
    position: int


@dataclass
class Suggestion:
    """Represents a suggested verse match."""

    # The suggested verse
    verse: QuranVerse
    # Confidence score for this suggestion
    confidence: float
    # Reference string like "2:255"
    reference: str


@dataclass
class ValidationResult:
    """Result of validating a potential Quran quote."""

    # Whether a valid Quran verse was found
    is_valid: bool
    # Type of match found
    match_type: MatchType
    # Confidence score (0-1), higher is better
    confidence: float
    # The matched verse (if found)
    matched_verse: Optional[QuranVerse] = None
    # Reference string like "2:255"
    reference: Optional[str] = None
    # Specific differences between input and matched verse (for corrections)
    differences: list[Difference] = field(default_factory=list)
    # Suggestions if multiple possible matches exist
    suggestions: list[Suggestion] = field(default_factory=list)


@dataclass
class DetectionSegment:
    """Represents a detected segment in text."""

    # The detected text
    text: str
    # Start position in original text
    start_index: int
    # End position in original text
    end_index: int
    # Validation result for this segment
    validation: Optional[ValidationResult] = None


@dataclass
class DetectionResult:
    """Detection result for finding Quran quotes in text."""

    # Whether potential Quran content was detected
    detected: bool
    # Extracted segments that appear to be Quran quotes
    segments: list[DetectionSegment] = field(default_factory=list)


@dataclass
class ValidatorOptions:
    """Options for the validator."""

    # Minimum confidence threshold for fuzzy matches (default: 0.8)
    fuzzy_threshold: float = 0.8
    # Maximum number of suggestions to return (default: 3)
    max_suggestions: int = 3
    # Whether to include partial matches (default: True)
    include_partial: bool = True
    # Minimum text length to consider for detection (default: 10)
    min_detection_length: int = 10


@dataclass
class NormalizationOptions:
    """Configuration for Arabic text normalization."""

    # Remove diacritics/tashkeel (default: True)
    remove_diacritics: bool = True
    # Normalize alef variants to plain alef (default: True)
    normalize_alef: bool = True
    # Normalize alef maqsura to ya (default: True)
    normalize_alef_maqsura: bool = True
    # Normalize teh marbuta to heh (default: True)
    normalize_teh_marbuta: bool = True
    # Remove tatweel/kashida (default: True)
    remove_tatweel: bool = True
    # Normalize hamza carriers (default: True)
    normalize_hamza: bool = True
    # Normalize whitespace (default: True)
    normalize_whitespace: bool = True
