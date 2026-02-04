"""
Arabic text normalization utilities.

This module provides functions for normalizing Arabic text, including
removing diacritics, normalizing character variants, and computing
text similarity using Levenshtein distance.
"""

import re
from dataclasses import dataclass
from typing import Optional

from .types import NormalizationOptions, Difference


# Arabic Unicode character ranges and patterns
ARABIC_PATTERNS = {
    # Diacritics (tashkeel) - Fatha, Kasra, Damma, Sukun, Shadda, Tanween, etc.
    'diacritics': re.compile(r'[\u064B-\u065F\u0670\u06D6-\u06ED]'),

    # Alef variants: أ إ آ ٱ (not plain alef ا)
    'alef_variants': re.compile(r'[أإآٱ]'),

    # Alef maqsura: ى
    'alef_maqsura': re.compile(r'ى'),

    # Teh marbuta: ة
    'teh_marbuta': re.compile(r'ة'),

    # Tatweel (kashida): ـ
    'tatweel': re.compile(r'ـ'),

    # Waw with hamza above: ؤ
    'waw_hamza': re.compile(r'ؤ'),

    # Ya with hamza above: ئ
    'ya_hamza': re.compile(r'ئ'),

    # Hamza variants (standalone): ء
    'hamza_standalone': re.compile(r'ء'),

    # Multiple whitespace
    'multiple_spaces': re.compile(r'\s+'),

    # Arabic-specific punctuation that might appear in quotes
    'punctuation': re.compile(r'[،؛؟]'),
}

# Arabic Unicode ranges for detection
ARABIC_UNICODE_PATTERN = re.compile(
    r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
)

# Pattern for extracting continuous Arabic text (including spaces between Arabic words)
ARABIC_SEGMENT_PATTERN = re.compile(
    r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
    r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]*'
)


def normalize_arabic(
    text: str,
    options: Optional[NormalizationOptions] = None
) -> str:
    """
    Normalize Arabic text for comparison.

    This function applies various normalization rules to make Arabic text
    comparison more reliable, especially for Quranic verses which may be
    written with different levels of diacritical marks.

    Args:
        text: The Arabic text to normalize
        options: Normalization options (uses defaults if not provided)

    Returns:
        Normalized text

    Example:
        >>> normalize_arabic("بِسْمِ اللَّهِ")
        'بسم الله'
        >>> normalize_arabic("الرَّحْمَٰنِ")
        'الرحمن'
    """
    if options is None:
        options = NormalizationOptions()

    result = text

    # Remove diacritics (tashkeel)
    if options.remove_diacritics:
        result = ARABIC_PATTERNS['diacritics'].sub('', result)

    # Normalize alef variants (أ إ آ ٱ) to plain alef (ا)
    if options.normalize_alef:
        result = ARABIC_PATTERNS['alef_variants'].sub('ا', result)

    # Normalize alef maqsura (ى) to ya (ي)
    if options.normalize_alef_maqsura:
        result = ARABIC_PATTERNS['alef_maqsura'].sub('ي', result)

    # Normalize teh marbuta (ة) to heh (ه)
    if options.normalize_teh_marbuta:
        result = ARABIC_PATTERNS['teh_marbuta'].sub('ه', result)

    # Remove tatweel (kashida)
    if options.remove_tatweel:
        result = ARABIC_PATTERNS['tatweel'].sub('', result)

    # Normalize hamza carriers
    if options.normalize_hamza:
        result = ARABIC_PATTERNS['waw_hamza'].sub('و', result)
        result = ARABIC_PATTERNS['ya_hamza'].sub('ي', result)

    # Normalize whitespace
    if options.normalize_whitespace:
        result = ARABIC_PATTERNS['multiple_spaces'].sub(' ', result).strip()

    return result


def remove_diacritics(text: str) -> str:
    """
    Remove only diacritics (tashkeel) from Arabic text.

    This preserves the base letters but removes vowel marks,
    shadda, sukun, and other diacritical marks.

    Args:
        text: The Arabic text

    Returns:
        Text without diacritics

    Example:
        >>> remove_diacritics("السَّلَامُ عَلَيْكُمُ")
        'السلام عليكم'
    """
    return ARABIC_PATTERNS['diacritics'].sub('', text)


def contains_arabic(text: str) -> bool:
    """
    Check if text contains Arabic characters.

    Args:
        text: The text to check

    Returns:
        True if text contains Arabic characters
    """
    return bool(ARABIC_UNICODE_PATTERN.search(text))


@dataclass
class ArabicSegment:
    """Represents an extracted Arabic text segment."""
    text: str
    start_index: int
    end_index: int


def extract_arabic_segments(text: str) -> list[ArabicSegment]:
    """
    Extract Arabic text segments from mixed text.

    Args:
        text: Text that may contain Arabic and non-Arabic content

    Returns:
        List of Arabic text segments with their positions
    """
    segments: list[ArabicSegment] = []

    for match in ARABIC_SEGMENT_PATTERN.finditer(text):
        segment_text = match.group().strip()
        if segment_text:
            segments.append(ArabicSegment(
                text=segment_text,
                start_index=match.start(),
                end_index=match.end(),
            ))

    return segments


def calculate_similarity(str1: str, str2: str) -> float:
    """
    Calculate similarity between two strings using Levenshtein distance.

    Args:
        str1: First string
        str2: Second string

    Returns:
        Similarity score between 0 and 1 (1 = identical)
    """
    if str1 == str2:
        return 1.0
    if len(str1) == 0 or len(str2) == 0:
        return 0.0

    distance = levenshtein_distance(str1, str2)
    max_length = max(len(str1), len(str2))

    return 1 - distance / max_length


def levenshtein_distance(str1: str, str2: str) -> int:
    """
    Calculate Levenshtein distance between two strings.

    Args:
        str1: First string
        str2: Second string

    Returns:
        Edit distance
    """
    m = len(str1)
    n = len(str2)

    # Create distance matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize first column
    for i in range(m + 1):
        dp[i][0] = i

    # Initialize first row
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j - 1] + 1   # substitution
                )

    return dp[m][n]


def find_differences(input_text: str, correct_text: str) -> list[Difference]:
    """
    Find differences between two strings.

    Args:
        input_text: The input string
        correct_text: The correct string

    Returns:
        List of differences with positions
    """
    differences: list[Difference] = []

    # Use a simple character-by-character comparison
    # This is a basic implementation - could be enhanced with more sophisticated diff algorithms
    min_length = min(len(input_text), len(correct_text))
    diff_start = -1
    input_chunk = ''
    correct_chunk = ''

    for i in range(min_length):
        if input_text[i] != correct_text[i]:
            if diff_start == -1:
                diff_start = i
            input_chunk += input_text[i]
            correct_chunk += correct_text[i]
        elif diff_start != -1:
            differences.append(Difference(
                input=input_chunk,
                correct=correct_chunk,
                position=diff_start,
            ))
            diff_start = -1
            input_chunk = ''
            correct_chunk = ''

    # Handle remaining differences
    if diff_start != -1 or len(input_text) != len(correct_text):
        if diff_start == -1:
            diff_start = min_length

        input_chunk += input_text[diff_start + len(input_chunk):] if diff_start != -1 else input_text[min_length:]
        correct_chunk += correct_text[diff_start + len(correct_chunk):] if diff_start != -1 else correct_text[min_length:]

        if input_chunk or correct_chunk:
            differences.append(Difference(
                input=input_chunk or '(missing)',
                correct=correct_chunk or '(extra)',
                position=diff_start,
            ))

    return differences
