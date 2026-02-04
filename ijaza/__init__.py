"""
ijaza

Validate and verify Quranic verses in LLM-generated text with high accuracy.

Basic Validation Example:
    >>> from ijaza import QuranValidator
    >>>
    >>> validator = QuranValidator()
    >>>
    >>> # Validate a specific quote
    >>> result = validator.validate("بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ")
    >>> print(result.is_valid)  # True
    >>> print(result.reference)  # "1:1"

LLM Integration Example (Recommended):
    >>> from ijaza import LLMProcessor, SYSTEM_PROMPTS
    >>>
    >>> # 1. Add system prompt to your LLM
    >>> system_prompt = SYSTEM_PROMPTS['xml']
    >>>
    >>> # 2. Process LLM response
    >>> processor = LLMProcessor()
    >>> result = processor.process(llm_response)
    >>>
    >>> # 3. Use corrected text
    >>> print(result.corrected_text)
    >>> print(result.all_valid)  # True if all quotes are authentic
"""

# Main validator
from .validator import QuranValidator, create_validator

# LLM Integration (recommended for processing LLM output)
from .llm_integration import (
    LLMProcessor,
    create_llm_processor,
    quick_validate,
    SYSTEM_PROMPTS,
)

# Normalization utilities
from .normalizer import (
    normalize_arabic,
    remove_diacritics,
    contains_arabic,
    extract_arabic_segments,
    calculate_similarity,
    find_differences,
    ArabicSegment,
)

# Types
from .types import (
    QuranVerse,
    QuranSurah,
    ValidationResult,
    DetectionResult,
    DetectionSegment,
    ValidatorOptions,
    MatchType,
    NormalizationOptions,
    Difference,
    Suggestion,
)

from .llm_integration import (
    ProcessedOutput,
    QuoteAnalysis,
    LLMProcessorOptions,
)

__version__ = "1.0.0"
__author__ = "Yazin Alirhayim"
__license__ = "MIT"

__all__ = [
    # Main validator
    "QuranValidator",
    "create_validator",
    # LLM Integration
    "LLMProcessor",
    "create_llm_processor",
    "quick_validate",
    "SYSTEM_PROMPTS",
    # Normalization utilities
    "normalize_arabic",
    "remove_diacritics",
    "contains_arabic",
    "extract_arabic_segments",
    "calculate_similarity",
    "find_differences",
    "ArabicSegment",
    # Types
    "QuranVerse",
    "QuranSurah",
    "ValidationResult",
    "DetectionResult",
    "DetectionSegment",
    "ValidatorOptions",
    "MatchType",
    "NormalizationOptions",
    "Difference",
    "Suggestion",
    "ProcessedOutput",
    "QuoteAnalysis",
    "LLMProcessorOptions",
]
