"""
ASR (Automatic Speech Recognition) error tolerance for Arabic Quran matching.

This module provides phonetic-aware similarity computation that handles
common ASR errors in Arabic speech recognition, such as phonetic confusions,
dropped function words, word boundary errors, and repetitions.
"""

import re
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ASROptions:
    """Configuration for ASR error tolerance."""

    # Weight for function word insertions/deletions (0.0 to 1.0)
    function_word_cost: float = 0.2
    # Whether to pre-normalize by removing repeated words
    remove_stutters: bool = True
    # Whether to try fixing word boundaries before matching
    fix_word_boundaries: bool = True


# --- Phonetic confusion pairs ---
# Arabic letter pairs that ASR systems commonly confuse.
# Each tuple is (char_a, char_b, cost) where cost is 0.0-1.0
# (0.0 = free substitution, 1.0 = full penalty like standard Levenshtein)

PHONETIC_CONFUSIONS: list[tuple[str, str, float]] = [
    # Emphatic vs. non-emphatic pairs
    ('ص', 'س', 0.3),   # Sad / Sin
    ('ض', 'د', 0.3),   # Dad / Dal
    ('ط', 'ت', 0.3),   # Ta / Taa
    ('ظ', 'ذ', 0.3),   # Zaa / Dhal
    ('ظ', 'ز', 0.3),   # Zaa / Zay

    # Interdental / sibilant confusions
    ('ث', 'س', 0.3),   # Thaa / Sin
    ('ث', 'ت', 0.4),   # Thaa / Taa
    ('ذ', 'ز', 0.3),   # Dhal / Zay
    ('ذ', 'د', 0.4),   # Dhal / Dal

    # Guttural confusions
    ('ح', 'ه', 0.3),   # Haa / Ha
    ('ع', 'ا', 0.4),   # Ain / Alef (in colloquial speech)
    ('ء', 'ع', 0.4),   # Hamza / Ain
    ('غ', 'خ', 0.4),   # Ghain / Khaa
    ('ق', 'ك', 0.4),   # Qaf / Kaf (dialectal confusion)
    ('ق', 'غ', 0.5),   # Qaf / Ghain

    # Alef variants (already handled by normalizer, but ASR may produce these)
    ('ا', 'أ', 0.0),
    ('ا', 'إ', 0.0),
    ('ا', 'آ', 0.0),
    ('ا', 'ٱ', 0.0),

    # Ya / Alef maqsura
    ('ي', 'ى', 0.0),

    # Ta marbuta / Ha / Ta
    ('ة', 'ه', 0.0),
    ('ة', 'ت', 0.2),   # Ta marbuta / Ta (common ASR)
]

# Build a lookup dict for O(1) access: (char_a, char_b) -> cost
_CONFUSION_MAP: dict[tuple[str, str], float] = {}
for _a, _b, _cost in PHONETIC_CONFUSIONS:
    _CONFUSION_MAP[(_a, _b)] = _cost
    _CONFUSION_MAP[(_b, _a)] = _cost  # symmetric


# Arabic function words that ASR commonly drops or inserts
FUNCTION_WORDS: set[str] = {
    'و', 'في', 'من', 'إلى', 'على', 'عن', 'ما', 'لا', 'أن', 'إن',
    'هو', 'هي', 'لم', 'لن', 'قد', 'بل', 'ثم', 'أو', 'فإن',
    'الى', 'ان',
}


def get_substitution_cost(char_a: str, char_b: str) -> float:
    """
    Get the substitution cost between two Arabic characters.

    Returns a value between 0.0 (free, e.g. alef variants) and 1.0 (unrelated).
    Phonetically similar pairs return a reduced cost.
    """
    if char_a == char_b:
        return 0.0
    return _CONFUSION_MAP.get((char_a, char_b), 1.0)


def weighted_levenshtein(
    str1: str,
    str2: str,
    substitution_cost_fn: Optional[Callable[[str, str], float]] = None,
) -> float:
    """
    Compute weighted Levenshtein distance where substitution cost
    depends on phonetic similarity of the characters.

    Args:
        str1: First string
        str2: Second string
        substitution_cost_fn: Function (char_a, char_b) -> float cost.
            Defaults to get_substitution_cost.

    Returns:
        Weighted edit distance (float)
    """
    if substitution_cost_fn is None:
        substitution_cost_fn = get_substitution_cost

    m, n = len(str1), len(str2)
    dp: list[list[float]] = [[0.0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = float(i)
    for j in range(n + 1):
        dp[0][j] = float(j)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub_cost = substitution_cost_fn(str1[i - 1], str2[j - 1])
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,             # deletion
                dp[i][j - 1] + 1.0,             # insertion
                dp[i - 1][j - 1] + sub_cost,    # substitution
            )

    return dp[m][n]


def calculate_asr_similarity(str1: str, str2: str) -> float:
    """
    Calculate ASR-aware similarity between two Arabic strings.

    Uses weighted Levenshtein with phonetic confusion costs.
    Returns value between 0.0 and 1.0 (1.0 = identical).
    """
    if str1 == str2:
        return 1.0
    if not str1 or not str2:
        return 0.0

    distance = weighted_levenshtein(str1, str2)
    max_length = max(len(str1), len(str2))
    return max(0.0, 1.0 - distance / max_length)


def remove_stutter_repetitions(text: str) -> str:
    """
    Remove likely stutter/repetition artifacts from ASR output.

    Detects consecutive duplicate words and removes the duplicates.
    Example: "قل قل هو الله" -> "قل هو الله"
    """
    words = text.split()
    if len(words) <= 1:
        return text

    cleaned: list[str] = [words[0]]
    for i in range(1, len(words)):
        if words[i] != words[i - 1]:
            cleaned.append(words[i])

    return ' '.join(cleaned)


def normalize_word_boundaries(text: str) -> str:
    """
    Fix common word boundary errors from ASR.

    - Remove zero-width joiners/non-joiners that may cause false splits
    - Collapse multiple spaces
    """
    # Remove zero-width characters
    text = re.sub(r'[\u200B-\u200D\u200E\u200F\uFEFF]', '', text)
    # Collapse spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def word_level_similarity(
    input_words: list[str],
    verse_words: list[str],
    options: Optional[ASROptions] = None,
) -> float:
    """
    Compute word-level edit distance with ASR-specific costs.

    Function words (wa, fi, min, etc.) have a reduced insertion/deletion cost.
    Non-function words are compared using phonetic-weighted char similarity.

    Args:
        input_words: Words from ASR input
        verse_words: Words from the known Quran verse
        options: ASR configuration

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if options is None:
        options = ASROptions()

    m, n = len(input_words), len(verse_words)
    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0

    dp: list[list[float]] = [[0.0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        cost = options.function_word_cost if input_words[i - 1] in FUNCTION_WORDS else 1.0
        dp[i][0] = dp[i - 1][0] + cost
    for j in range(1, n + 1):
        cost = options.function_word_cost if verse_words[j - 1] in FUNCTION_WORDS else 1.0
        dp[0][j] = dp[0][j - 1] + cost

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            input_word = input_words[i - 1]
            verse_word = verse_words[j - 1]

            # Substitution cost based on character-level phonetic similarity
            char_sim = calculate_asr_similarity(input_word, verse_word)
            sub_cost = 1.0 - char_sim

            # Deletion cost (extra word in input)
            ins_cost = options.function_word_cost if input_word in FUNCTION_WORDS else 1.0

            # Insertion cost (word missing from input, present in verse)
            del_cost = options.function_word_cost if verse_word in FUNCTION_WORDS else 1.0

            dp[i][j] = min(
                dp[i - 1][j] + ins_cost,
                dp[i][j - 1] + del_cost,
                dp[i - 1][j - 1] + sub_cost,
            )

    max_length = max(m, n)
    return max(0.0, 1.0 - dp[m][n] / max_length)


def preprocess_asr_text(text: str, options: Optional[ASROptions] = None) -> str:
    """
    Apply all ASR-specific preprocessing to input text.

    Steps:
    1. Normalize word boundaries (remove zero-width chars, collapse spaces)
    2. Remove stutter repetitions

    Args:
        text: Raw ASR output
        options: ASR configuration

    Returns:
        Preprocessed text
    """
    if options is None:
        options = ASROptions()

    result = text

    if options.fix_word_boundaries:
        result = normalize_word_boundaries(result)

    if options.remove_stutters:
        result = remove_stutter_repetitions(result)

    return result
