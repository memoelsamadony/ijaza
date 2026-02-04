"""
LLM Integration Module.

Provides tools for integrating Quran validation into LLM pipelines:
1. System prompts that instruct LLMs to tag Quran quotes
2. Post-processors that validate and correct tagged quotes
3. Scanners that detect untagged potential Quran content
"""

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from .validator import QuranValidator
from .normalizer import normalize_arabic, calculate_similarity


@dataclass
class QuoteAnalysis:
    """Analysis of a single Quran quote."""

    # Original text from the LLM
    original: str
    # Corrected text (if different)
    corrected: str
    # Whether this was valid
    is_valid: bool
    # Reference if identified (e.g., "2:255")
    reference: Optional[str]
    # Confidence score
    confidence: float
    # How this quote was detected
    detection_method: Literal['tagged', 'contextual', 'fuzzy']
    # Position in original text
    start_index: int
    end_index: int
    # Whether correction was applied
    was_corrected: bool


@dataclass
class ProcessedOutput:
    """Result of processing LLM output for Quran validation."""

    # The corrected output text (with Quran quotes fixed)
    corrected_text: str
    # Whether all Quran quotes were valid
    all_valid: bool
    # Details about each detected quote
    quotes: list[QuoteAnalysis] = field(default_factory=list)
    # Warnings about potential issues
    warnings: list[str] = field(default_factory=list)


@dataclass
class LLMProcessorOptions:
    """Options for the LLM processor."""

    # Auto-correct misquoted verses (default: True)
    auto_correct: bool = True
    # Minimum confidence to consider a fuzzy match valid (default: 0.85)
    min_confidence: float = 0.85
    # Include untagged Arabic text in scan (default: True)
    scan_untagged: bool = True
    # Tag format to look for (default: 'xml')
    tag_format: Literal['xml', 'markdown', 'bracket'] = 'xml'


# Contextual patterns that suggest Quran quotes
QURAN_CONTEXT_PATTERNS = [
    # English patterns
    re.compile(
        r'(?:Allah\s+says?|God\s+says?|the\s+Quran\s+says?|in\s+the\s+Quran|'
        r'Quranic\s+verse|verse\s+states?|ayah|ayat|surah)\s*[:\-]?\s*',
        re.IGNORECASE
    ),
    # Arabic patterns
    re.compile(
        r'(?:قال\s+الله|قال\s+تعالى|يقول\s+الله|في\s+القرآن|الآية|سورة)\s*[:\-]?\s*'
    ),
    # Reference patterns like (2:255) or (2:255-257) or [Al-Baqarah:255]
    re.compile(r'\(?\d{1,3}:\d{1,3}(?:-\d{1,3})?\)?'),
    re.compile(r'\[[\w\-]+:\d+(?:-\d+)?\]'),
]


# System prompts for LLMs to properly format Quran quotes
SYSTEM_PROMPTS = {
    'xml': """When quoting verses from the Quran, you MUST use this exact format:
<quran ref="SURAH:AYAH">ARABIC_TEXT</quran>

For multiple consecutive verses, use a range:
<quran ref="SURAH:START-END">ARABIC_TEXT</quran>

Examples:
<quran ref="1:1">بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ</quran>
<quran ref="112:1-4">قُلْ هُوَ ٱللَّهُ أَحَدٌ ٱللَّهُ ٱلصَّمَدُ لَمْ يَلِدْ وَلَمْ يُولَدْ وَلَمْ يَكُن لَّهُۥ كُفُوًا أَحَدٌۢ</quran>

Rules:
- Always include the reference (surah:ayah or surah:start-end for ranges)
- Use the exact Arabic text with full diacritics if possible
- Never paraphrase or partially quote without indication
- If unsure of exact wording, say "approximately" before the quote""",

    'markdown': """When quoting verses from the Quran, use this format:
```quran ref="SURAH:AYAH"
ARABIC_TEXT
```

For verse ranges, use:
```quran ref="SURAH:START-END"
ARABIC_TEXT
```

Example:
```quran ref="112:1-4"
قُلْ هُوَ ٱللَّهُ أَحَدٌ ٱللَّهُ ٱلصَّمَدُ لَمْ يَلِدْ وَلَمْ يُولَدْ وَلَمْ يَكُن لَّهُۥ كُفُوًا أَحَدٌۢ
```""",

    'bracket': """When quoting Quran verses, use: [[Q:SURAH:AYAH|ARABIC_TEXT]]
For verse ranges: [[Q:SURAH:START-END|ARABIC_TEXT]]

Example: [[Q:1:1|بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ]]
Example range: [[Q:112:1-4|قُلْ هُوَ ٱللَّهُ أَحَدٌ ٱللَّهُ ٱلصَّمَدُ لَمْ يَلِدْ وَلَمْ يُولَدْ وَلَمْ يَكُن لَّهُۥ كُفُوًا أَحَدٌۢ]]""",

    'minimal': """Always cite Quran verses with their reference number in parentheses immediately after, like: "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ (1:1)" or for ranges "... (112:1-4)\"""",
}


# Regex patterns for extracting tagged quotes
# Supports both single verses (1:1) and verse ranges (1:1-7, 107:1-3)
TAG_PATTERNS = {
    'xml': re.compile(
        r'<quran\s+ref=["\'](\d+:\d+(?:-\d+)?)["\']>([\s\S]*?)</quran>',
        re.IGNORECASE
    ),
    'markdown': re.compile(
        r'```quran\s+ref=["\'](\d+:\d+(?:-\d+)?)["\'][\r\n]+([\s\S]*?)[\r\n]+```',
        re.IGNORECASE
    ),
    'bracket': re.compile(
        r'\[\[Q:(\d+:\d+(?:-\d+)?)\|([\s\S]*?)\]\]'
    ),
}

# Also match inline references like "text (1:1)" or "text (1:1-3)"
INLINE_REF_PATTERN = re.compile(
    r'([\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]+)'
    r'\s*\((\d+:\d+(?:-\d+)?)\)'
)

# Pattern for Arabic text segments
ARABIC_SEGMENT_PATTERN = re.compile(
    r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
    r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]*'
)

# Pattern for Arabic text after context triggers
ARABIC_AFTER_CONTEXT_PATTERN = re.compile(
    r'^[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]+'
)


@dataclass
class ParsedReference:
    """Parsed verse reference."""
    surah: int
    start_ayah: int
    end_ayah: Optional[int]
    is_range: bool


def parse_reference(ref: str) -> Optional[ParsedReference]:
    """
    Parse a verse reference, supporting both single verses and ranges.

    Args:
        ref: Reference string like "1:1" or "2:255-257"

    Returns:
        Parsed reference with surah, start_ayah, and optional end_ayah
    """
    match = re.match(r'^(\d+):(\d+)(?:-(\d+))?$', ref)
    if not match:
        return None

    surah = int(match.group(1))
    start_ayah = int(match.group(2))
    end_ayah = int(match.group(3)) if match.group(3) else None

    return ParsedReference(
        surah=surah,
        start_ayah=start_ayah,
        end_ayah=end_ayah,
        is_range=end_ayah is not None,
    )


class LLMProcessor:
    """
    LLM Output Processor.

    Processes LLM-generated text to validate and correct Quran quotes.

    Example:
        >>> from ijaza import LLMProcessor
        >>>
        >>> processor = LLMProcessor()
        >>>
        >>> # Add system prompt to your LLM call
        >>> system_prompt = processor.get_system_prompt()
        >>>
        >>> # Process the LLM response
        >>> result = processor.process(llm_response)
        >>>
        >>> if not result.all_valid:
        ...     print('Corrections needed:', [q for q in result.quotes if q.was_corrected])
        >>>
        >>> # Use the corrected text
        >>> print(result.corrected_text)
    """

    def __init__(self, options: Optional[LLMProcessorOptions] = None):
        """
        Initialize the LLMProcessor.

        Args:
            options: Processor options (uses defaults if not provided)
        """
        self.validator = QuranValidator()

        if options is None:
            self.options = LLMProcessorOptions()
        else:
            self.options = options

    def get_system_prompt(self) -> str:
        """Get the recommended system prompt for the configured tag format."""
        return SYSTEM_PROMPTS[self.options.tag_format]

    def process(self, text: str) -> ProcessedOutput:
        """
        Process LLM output to validate and optionally correct Quran quotes.

        Args:
            text: The LLM-generated text

        Returns:
            Processed output with validation results
        """
        quotes: list[QuoteAnalysis] = []
        warnings: list[str] = []
        corrected_text = text

        # Step 1: Extract and validate tagged quotes
        tagged_quotes = self._extract_tagged_quotes(text)
        for tagged in tagged_quotes:
            analysis = self._analyze_quote(
                tagged['text'],
                tagged['reference'],
                tagged['start_index'],
                tagged['end_index'],
                'tagged'
            )
            quotes.append(analysis)

            if self.options.auto_correct and analysis.was_corrected:
                corrected_text = self._replace_in_text(
                    corrected_text,
                    tagged['full_match'],
                    self._format_corrected_tag(analysis)
                )

        # Step 2: Scan for contextual quotes (preceded by "Allah says", etc.)
        contextual_quotes = self._extract_contextual_quotes(text, tagged_quotes)
        for contextual in contextual_quotes:
            analysis = self._analyze_quote(
                contextual['text'],
                None,
                contextual['start_index'],
                contextual['end_index'],
                'contextual'
            )

            if analysis.is_valid or analysis.confidence >= self.options.min_confidence:
                quotes.append(analysis)

                if self.options.auto_correct and analysis.was_corrected:
                    corrected_text = self._replace_in_text(
                        corrected_text,
                        contextual['text'],
                        analysis.corrected
                    )

        # Step 3: Scan for untagged Arabic that might be Quran (fuzzy)
        if self.options.scan_untagged:
            untagged_quotes = self._scan_untagged_arabic(text, quotes)
            for untagged in untagged_quotes:
                analysis = self._analyze_quote(
                    untagged['text'],
                    None,
                    untagged['start_index'],
                    untagged['end_index'],
                    'fuzzy'
                )

                if analysis.confidence >= self.options.min_confidence:
                    quotes.append(analysis)
                    warnings.append(
                        f'Potential untagged Quran quote detected: "{untagged["text"][:50]}..." '
                        f'(possibly {analysis.reference}, {analysis.confidence * 100:.0f}% confidence)'
                    )

                    if self.options.auto_correct and analysis.was_corrected:
                        corrected_text = self._replace_in_text(
                            corrected_text,
                            untagged['text'],
                            analysis.corrected
                        )

        # Determine overall validity
        all_valid = all(q.is_valid and not q.was_corrected for q in quotes)

        return ProcessedOutput(
            corrected_text=corrected_text,
            all_valid=all_valid,
            quotes=quotes,
            warnings=warnings,
        )

    def validate_quote(
        self,
        text: str,
        expected_ref: Optional[str] = None
    ) -> dict:
        """
        Validate a single quote without full processing.

        Args:
            text: The quote text
            expected_ref: Expected reference (optional)

        Returns:
            Dict with 'is_valid', 'correct_text', and 'actual_ref' keys
        """
        validation = self.validator.validate(text)

        if not validation.is_valid:
            return {'is_valid': False}

        # If expected reference provided, check it matches
        if expected_ref and validation.reference != expected_ref:
            return {
                'is_valid': False,
                'correct_text': validation.matched_verse.text if validation.matched_verse else None,
                'actual_ref': validation.reference,
            }

        # Check if text needs correction
        needs_correction = (
            validation.match_type != 'exact' and validation.matched_verse is not None
        )

        return {
            'is_valid': True,
            'correct_text': validation.matched_verse.text if needs_correction else None,
            'actual_ref': validation.reference,
        }

    # Private methods

    def _extract_tagged_quotes(self, text: str) -> list[dict]:
        """Extract quotes with explicit tags."""
        results: list[dict] = []

        pattern = TAG_PATTERNS[self.options.tag_format]

        for match in pattern.finditer(text):
            results.append({
                'reference': match.group(1),
                'text': match.group(2).strip(),
                'start_index': match.start(),
                'end_index': match.end(),
                'full_match': match.group(0),
            })

        # Also check for inline references
        for match in INLINE_REF_PATTERN.finditer(text):
            # Skip if this overlaps with an already found tagged quote
            overlaps = any(
                match.start() >= r['start_index'] and match.start() < r['end_index']
                for r in results
            )
            if not overlaps:
                results.append({
                    'text': match.group(1).strip(),
                    'reference': match.group(2),
                    'start_index': match.start(),
                    'end_index': match.end(),
                    'full_match': match.group(0),
                })

        return results

    def _extract_contextual_quotes(
        self,
        text: str,
        already_found: list[dict]
    ) -> list[dict]:
        """Extract quotes preceded by context patterns."""
        results: list[dict] = []

        for pattern in QURAN_CONTEXT_PATTERNS:
            for match in pattern.finditer(text):
                # Look for Arabic text following this pattern
                after_match = text[match.end():]
                arabic_match = ARABIC_AFTER_CONTEXT_PATTERN.match(after_match)

                if arabic_match and len(arabic_match.group().strip()) >= 10:
                    start_index = match.end()
                    end_index = start_index + len(arabic_match.group())

                    # Skip if overlaps with already found quotes
                    overlaps = any(
                        (start_index >= r['start_index'] and start_index < r['end_index']) or
                        (end_index > r['start_index'] and end_index <= r['end_index'])
                        for r in already_found
                    )

                    if not overlaps:
                        results.append({
                            'text': arabic_match.group().strip(),
                            'start_index': start_index,
                            'end_index': end_index,
                        })

        return results

    def _scan_untagged_arabic(
        self,
        text: str,
        already_found: list
    ) -> list[dict]:
        """Scan for untagged Arabic text that might be Quran."""
        results: list[dict] = []

        for match in ARABIC_SEGMENT_PATTERN.finditer(text):
            segment = match.group().strip()

            # Skip short segments
            if len(segment) < 15:
                continue

            # Skip if overlaps with already found quotes
            overlaps = any(
                (match.start() >= r.start_index and match.start() < r.end_index) or
                (match.end() > r.start_index and match.end() <= r.end_index)
                for r in already_found
            )

            if not overlaps:
                results.append({
                    'text': segment,
                    'start_index': match.start(),
                    'end_index': match.end(),
                })

        return results

    def _analyze_quote(
        self,
        text: str,
        expected_ref: Optional[str],
        start_index: int,
        end_index: int,
        detection_method: Literal['tagged', 'contextual', 'fuzzy']
    ) -> QuoteAnalysis:
        """Analyze a single quote."""
        # Check if this is a verse range reference
        if expected_ref:
            parsed = parse_reference(expected_ref)
            if parsed and parsed.is_range and parsed.end_ayah:
                return self._analyze_range_quote(
                    text,
                    expected_ref,
                    parsed,
                    start_index,
                    end_index,
                    detection_method
                )

        # Single verse validation
        validation = self.validator.validate(text)

        is_valid = validation.is_valid
        was_corrected = False
        corrected = text

        # Check reference if provided
        if expected_ref and validation.reference and validation.reference != expected_ref:
            # Reference mismatch - this might be an error
            is_valid = False

        # Determine if correction is needed
        if (
            validation.is_valid and
            validation.match_type != 'exact' and
            validation.matched_verse
        ):
            corrected = validation.matched_verse.text
            was_corrected = True

        return QuoteAnalysis(
            original=text,
            corrected=corrected,
            is_valid=is_valid,
            reference=validation.reference or expected_ref,
            confidence=validation.confidence,
            detection_method=detection_method,
            start_index=start_index,
            end_index=end_index,
            was_corrected=was_corrected,
        )

    def _analyze_range_quote(
        self,
        text: str,
        expected_ref: str,
        parsed: ParsedReference,
        start_index: int,
        end_index: int,
        detection_method: Literal['tagged', 'contextual', 'fuzzy']
    ) -> QuoteAnalysis:
        """Analyze a quote that references a verse range (e.g., 107:1-3)."""
        # Get the verse range from validator
        range_result = self.validator.get_verse_range(
            parsed.surah,
            parsed.start_ayah,
            parsed.end_ayah  # type: ignore
        )

        if not range_result:
            # Invalid range - verses don't exist
            return QuoteAnalysis(
                original=text,
                corrected=text,
                is_valid=False,
                reference=expected_ref,
                confidence=0,
                detection_method=detection_method,
                start_index=start_index,
                end_index=end_index,
                was_corrected=False,
            )

        # Compare the quoted text against the concatenated verse range
        normalized_input = normalize_arabic(text)
        normalized_range = normalize_arabic(range_result['text'])

        # Calculate similarity
        similarity = calculate_similarity(normalized_input, normalized_range)

        # Check for exact match
        is_exact = text.strip() == range_result['text']
        is_normalized_match = normalized_input == normalized_range

        is_valid = is_exact or is_normalized_match or similarity >= 0.85
        was_corrected = False
        corrected = text

        if is_exact:
            confidence = 1.0
        elif is_normalized_match:
            confidence = 0.95
        else:
            confidence = similarity

        # Auto-correct if needed
        if is_valid and not is_exact:
            corrected = range_result['text']
            was_corrected = True

        return QuoteAnalysis(
            original=text,
            corrected=corrected,
            is_valid=is_valid,
            reference=expected_ref,
            confidence=confidence,
            detection_method=detection_method,
            start_index=start_index,
            end_index=end_index,
            was_corrected=was_corrected,
        )

    def _format_corrected_tag(self, analysis: QuoteAnalysis) -> str:
        """Format a corrected quote with appropriate tags."""
        if self.options.tag_format == 'xml':
            return f'<quran ref="{analysis.reference}">{analysis.corrected}</quran>'
        elif self.options.tag_format == 'markdown':
            return f'```quran ref="{analysis.reference}"\n{analysis.corrected}\n```'
        elif self.options.tag_format == 'bracket':
            return f'[[Q:{analysis.reference}|{analysis.corrected}]]'
        else:
            return f'{analysis.corrected} ({analysis.reference})'

    def _replace_in_text(
        self,
        text: str,
        original: str,
        replacement: str
    ) -> str:
        """Replace text in string."""
        return text.replace(original, replacement, 1)


def create_llm_processor(
    options: Optional[LLMProcessorOptions] = None
) -> LLMProcessor:
    """
    Create an LLM processor instance.

    Args:
        options: Processor options

    Returns:
        LLMProcessor instance
    """
    return LLMProcessor(options)


def quick_validate(text: str) -> dict:
    """
    Quick validation of a complete LLM response.

    Args:
        text: LLM output to validate

    Returns:
        Dict with 'has_quran_content', 'all_valid', and 'issues' keys
    """
    processor = LLMProcessor(LLMProcessorOptions(auto_correct=False))
    result = processor.process(text)

    issues = []
    for q in result.quotes:
        if not q.is_valid or q.was_corrected:
            status = 'imprecise' if q.is_valid else 'invalid'
            issue = f'Quote "{q.original[:30]}..." is {status}'
            if q.reference:
                issue += f' (should be {q.reference})'
            issues.append(issue)

    issues.extend(result.warnings)

    return {
        'has_quran_content': len(result.quotes) > 0,
        'all_valid': result.all_valid,
        'issues': issues,
    }
