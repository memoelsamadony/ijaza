# Ijaza

**Validate and verify Quranic verses in LLM-generated text with high accuracy.**

Ijaza (Arabic: إجازة, meaning "authorization" or "permission to transmit") is a Python library that ensures the authenticity of Quranic text in AI-generated content. Just as traditional Islamic scholarship requires an *ijaza* to transmit sacred knowledge, this library provides a digital verification layer for Quranic quotes.

## Motivation

Large Language Models (LLMs) frequently misquote Quranic verses — changing words, mixing verses, or even fabricating text that sounds Quranic but isn't. This is a serious concern for:


Ijaza catches these errors automatically, corrects misquotations, and ensures that every Quranic verse in your application is authentic.

## Origin & Credits

This project began as a Python reimplementation of the excellent [quran-validator](https://github.com/yazinsai/quran-validator) npm package by [Yazin Alirhayim](https://github.com/yazinsai). We needed the same functionality for our Python-based projects and decided to port it while adding features specific to our use case.

Ijaza was developed as part of the [PolyKhateeb](https://github.com/memoelsamadony/polykhateeb) project — a real-time transcription and translation system for Islamic sermons (khutbahs). In that context, we needed to:

=======
# ijaza

Validate and verify Quranic verses in LLM-generated text with high accuracy.


## Installation

```bash
pip install ijaza
```

<<<<<<< HEAD
For better fuzzy matching performance (optional):
```bash
pip install ijaza[performance]
```

=======

## Quick Start

### Basic Validation

```python
from ijaza import QuranValidator

validator = QuranValidator()

# Validate a specific quote
result = validator.validate("بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ")
print(result.is_valid)   # True
print(result.reference)  # "1:1"
print(result.match_type) # "exact"
```

### LLM Integration (Recommended)

```python
from ijaza import LLMProcessor, SYSTEM_PROMPTS

# 1. Add system prompt to your LLM call
system_prompt = SYSTEM_PROMPTS['xml']

# 2. Process LLM response
processor = LLMProcessor()
result = processor.process(llm_response)

# 3. Use corrected text
print(result.corrected_text)
print(result.all_valid)  # True if all quotes are authentic

# 4. Check for issues
for quote in result.quotes:
    if quote.was_corrected:
        print(f"Corrected: {quote.original} -> {quote.corrected}")
```

## Features

- **Multi-tier matching**: exact → normalized → partial → fuzzy
- **LLM integration**: System prompts + post-processing validation
- **Arabic normalization**: Handles diacritics, alef variants, hamza, etc.
- **Auto-correction**: Fixes misquoted verses automatically
- **Detection**: Finds untagged Quran quotes in text
- **Full database**: 6,236 verses with Uthmani script
<<<<<<< HEAD
- **Zero dependencies**: Pure Python implementation (optional `rapidfuzz` for performance)
=======
- **Zero dependencies**: Pure Python implementation


## API Reference

### QuranValidator

```python
from ijaza import QuranValidator, ValidatorOptions

# With custom options
validator = QuranValidator(ValidatorOptions(
    fuzzy_threshold=0.85,
    max_suggestions=5,
    include_partial=True,
))

# Validate text
result = validator.validate("Arabic text here")

# Detect and validate all quotes in text
detection = validator.detect_and_validate("Text with Quran quotes...")

# Get specific verse
verse = validator.get_verse(surah=1, ayah=1)

# Get verse range
range_result = validator.get_verse_range(surah=112, start_ayah=1, end_ayah=4)

# Search verses
results = validator.search("search query", limit=10)
```

### LLMProcessor

```python
from ijaza import LLMProcessor, LLMProcessorOptions

processor = LLMProcessor(LLMProcessorOptions(
    auto_correct=True,
    min_confidence=0.85,
    scan_untagged=True,
    tag_format='xml',  # or 'markdown', 'bracket'
))

# Get system prompt for your LLM
prompt = processor.get_system_prompt()

# Process LLM output
result = processor.process(llm_output)
```

### Normalization Utilities

```python
from ijaza import (
    normalize_arabic,
    remove_diacritics,
    contains_arabic,
    extract_arabic_segments,
    calculate_similarity,
)

# Normalize Arabic text
normalized = normalize_arabic("بِسْمِ اللَّهِ")  # "بسم الله"

# Remove only diacritics
clean = remove_diacritics("السَّلَامُ")  # "السلام"

# Check for Arabic content
has_arabic = contains_arabic("Hello مرحبا")  # True

# Extract Arabic segments from mixed text
segments = extract_arabic_segments("The verse بسم الله means...")

# Calculate text similarity
similarity = calculate_similarity("text1", "text2")  # 0.0 - 1.0
```

<<<<<<< HEAD
## Future Work

### Translation Validation
Include major English translations (Sahih International, Pickthall, Yusuf Ali) and validate that translations correspond to the correct Arabic source verse.

### Framework Integrations
- LangChain / LlamaIndex guardrails
- FastAPI middleware
- Streamlit components
- Django/Flask integration

### Performance Optimizations
- N-gram indexing for pre-filtering
- BK-tree for metric-space nearest-neighbor search
- Async/streaming support for real-time pipelines

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT

## Acknowledgments

- [Yazin Alirhayim](https://github.com/yazinsai) for the original [quran-validator](https://github.com/yazinsai/quran-validator) npm package
- [AlQuran.cloud](https://alquran.cloud/) for the Quran API
- The [PolyKhateeb](https://github.com/memoelsamadony/polykhateeb) project team
=======
## License

MIT
