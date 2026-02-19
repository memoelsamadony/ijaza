# Ijaza

**Validate and verify Quranic verses in LLM-generated text with high accuracy.**

Ijaza (Arabic: إجازة, meaning "authorization" or "permission to transmit") is a Python library that ensures the authenticity of Quranic text in AI-generated content. Just as traditional Islamic scholarship requires an *ijaza* to transmit sacred knowledge, this library provides a digital verification layer for Quranic quotes.

## Motivation

Large Language Models (LLMs) frequently misquote Quranic verses — changing words, mixing verses, or even fabricating text that sounds Quranic but isn't. This is a serious concern for:

- **Islamic content creators** who need accurate Quranic citations
- **Educational platforms** teaching Quran and Islamic studies
- **AI applications** serving Muslim communities (chatbots, translation tools, khutbah assistants)
- **Developers** building LLM-powered tools that handle religious text

Ijaza catches these errors automatically, corrects misquotations, and ensures that every Quranic verse in your application is authentic.

## Origin & Credits

This project began as a Python reimplementation of the excellent [quran-validator](https://github.com/yazinsai/quran-validator) npm package by [Yazin Alirhayim](https://github.com/yazinsai). We needed the same functionality for our Python-based projects and decided to port it while adding features specific to our use case.

Ijaza was developed as part of the [PolyKhateeb](https://github.com/memoelsamadony/polykhateeb) project — a real-time transcription and translation system for Islamic sermons (khutbahs). In that context, we needed to:

- Detect Quranic segments in transcribed speech to preserve them verbatim
- Validate LLM-corrected text to catch any misquotations
- Inject system prompts into LLMs to properly tag Quran quotes


## Installation

```bash
pip install ijaza
```

For better fuzzy matching performance (optional):
```bash
pip install ijaza[performance]
```

## Usage

### Basic Validation

```python
from ijaza import QuranValidator

validator = QuranValidator()

# Validate a specific quote
result = validator.validate("بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ")
print(result.is_valid)    # True
print(result.reference)   # "1:1"
print(result.match_type)  # "exact"
print(result.confidence)  # 1.0
```

### Detect Quran Quotes in Text

```python
from ijaza import QuranValidator

validator = QuranValidator()

text = "The Prophet said to recite بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ before eating."
detection = validator.detect_and_validate(text)

for segment in detection.segments:
    if segment.validation and segment.validation.is_valid:
        print(f"Found: {segment.text}")
        print(f"Reference: {segment.validation.reference}")
```

### Look Up Verses

```python
from ijaza import QuranValidator

validator = QuranValidator()

# Get a specific verse
verse = validator.get_verse(surah=112, ayah=1)
print(verse.text)         # Full text with diacritics
print(verse.text_simple)  # Simplified text

# Get a range of verses
result = validator.get_verse_range(surah=112, start_ayah=1, end_ayah=4)
print(result['text'])

# Search for verses
results = validator.search("الرحمن", limit=5)
for r in results:
    print(r)
```

### LLM Integration

```python
from ijaza import LLMProcessor, SYSTEM_PROMPTS

# 1. Add system prompt to your LLM call
system_prompt = SYSTEM_PROMPTS['xml']  # or 'markdown', 'bracket', 'minimal'

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

### Quick Validate (One-liner)

```python
from ijaza import quick_validate

result = quick_validate(llm_response)
print(result['has_quran_content'])  # True/False
print(result['all_valid'])          # True if all quotes are correct
print(result['issues'])             # List of issues found
```

### ASR Error Tolerance

When processing speech-to-text output, Arabic ASR commonly confuses phonetically similar letters (ص/س, ط/ت, ض/د, etc.), drops function words, or produces stutters. Enable `asr_tolerant` mode for phonetic-aware matching:

```python
from ijaza import QuranValidator, ValidatorOptions

validator = QuranValidator(ValidatorOptions(asr_tolerant=True))

# ASR heard "السراط" instead of "الصراط" — phonetic confusion ص/س
# Standard matching would score this lower, ASR mode recognizes
# it as a known phonetic confusion and scores it higher.
result = validator.validate("يا ايها الذين امنوا اتقوا الله حق تقاته ولا تموتن الا وانتم مسلمون")
print(result.is_valid)    # True
print(result.reference)   # "3:102"
```

ASR mode also handles:
- **Stutter removal**: "قل قل هو الله" → "قل هو الله"
- **Function word drops**: Lower penalty when ASR drops و, في, من, etc.
- **Word boundary fixes**: Removes zero-width characters, collapses spaces

### Streaming Scanner (Cross-Chunk Verse Detection)

For real-time ASR pipelines where text arrives in chunks, a Quranic verse may be split across two chunks. The `StreamingScanner` maintains state across chunks to detect these split verses:

```python
from ijaza import StreamingScanner, StreamingScannerOptions
from ijaza.translations import TranslationProvider

provider = TranslationProvider()
scanner = StreamingScanner(
    options=StreamingScannerOptions(
        overlap_words=15,
        min_confidence=0.85,
        asr_tolerant=True,
    ),
    translation_provider=provider,
)

# Process chunks as they arrive from ASR
for chunk in asr_stream:
    result = scanner.process_chunk(chunk.text)

    for verse in result.complete_verses:
        print(f"Found: {verse.reference} — {verse.correct_text}")
        print(f"English: {verse.translations.get('en', '')}")

    if result.partial_verse:
        print("Verse in progress, waiting for next chunk...")

# End of stream — flush remaining
final = scanner.flush()
scanner.reset()
```

For batch processing (non-streaming), use `scan_for_verses()`:

```python
from ijaza import QuranValidator

validator = QuranValidator()

text = "والصلاة والسلام على رسوله قل هو الله احد الله الصمد لم يلد ولم يولد ولم يكن له كفوا احد وهذا يدل على التوحيد"
results = validator.scan_for_verses(text, min_words=3, confidence_threshold=0.85)

for v in results:
    print(f"{v['reference']}: {v['correct_text']}")
```

### Trusted Translations

When a Quranic verse is detected, ijaza can attach authoritative scholarly translations from bundled data — never LLM-generated:

```python
from ijaza import QuranValidator
from ijaza.translations import TranslationProvider

provider = TranslationProvider()  # loads Sahih International + Bubenheim
validator = QuranValidator(translation_provider=provider)

result = validator.validate("بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ")
print(result.translations['en'])  # "In the name of Allah, the Entirely Merciful, the Especially Merciful."
print(result.translations['de'])  # "Im Namen Allahs, des Allerbarmers, des Barmherzigen."
```

Default editions: **Sahih International** (English) and **Bubenheim & Elyas** (German). To use different editions:

```python
from ijaza.translations import TranslationProvider, TranslationConfig

# Use Pickthall for English instead
provider = TranslationProvider(TranslationConfig(
    editions={'en': 'en.pickthall', 'de': 'de.bubenheim'}
))
```

Fetch additional translation editions:

```bash
python scripts/fetch_translations.py --editions en.yusufali de.aburida
python scripts/fetch_translations.py --list-editions  # show all available
```

Available editions: `en.sahih`, `en.pickthall`, `en.yusufali`, `en.asad`, `en.hilali`, `en.itani`, `de.bubenheim`, `de.aburida`, `de.khoury`, `de.zaidan`.

Translations also work with `LLMProcessor` and `StreamingScanner` — pass the `translation_provider` to any of them.

### Arabic Normalization Utilities

```python
from ijaza import normalize_arabic, remove_diacritics, contains_arabic

# Normalize Arabic text for comparison
normalized = normalize_arabic("بِسْمِ اللَّهِ")  # "بسم الله"

# Remove only diacritics
clean = remove_diacritics("السَّلَامُ")  # "السلام"

# Check for Arabic content
has_arabic = contains_arabic("Hello مرحبا")  # True
```

## Features

- **Multi-tier matching**: exact → normalized → partial → fuzzy
- **LLM integration**: System prompts + post-processing validation
- **Arabic normalization**: Handles diacritics, alef variants, hamza, etc.
- **Auto-correction**: Fixes misquoted verses automatically
- **Detection**: Finds untagged Quran quotes in text
- **Full database**: 6,236 verses with Uthmani script
- **ASR error tolerance**: Phonetic-aware matching for speech recognition errors (ص/س, ط/ت, etc.)
- **Streaming scanner**: Cross-chunk verse detection for real-time ASR pipelines
- **Trusted translations**: Bundled English (Sahih International) and German (Bubenheim & Elyas) translations from scholarly sources
- **Zero dependencies**: Pure Python implementation (optional `rapidfuzz` for performance)


## API Reference

### QuranValidator

```python
from ijaza import QuranValidator, ValidatorOptions
from ijaza.translations import TranslationProvider

# With custom options
validator = QuranValidator(
    options=ValidatorOptions(
        fuzzy_threshold=0.85,
        max_suggestions=5,
        include_partial=True,
        asr_tolerant=False,  # set True for ASR input
    ),
    translation_provider=TranslationProvider(),  # optional
)

# Validate text
result = validator.validate("Arabic text here")

# Detect and validate all quotes in text
detection = validator.detect_and_validate("Text with Quran quotes...")

# Scan continuous Arabic text for embedded verses (sliding window)
found = validator.scan_for_verses("long arabic text...", min_words=3, confidence_threshold=0.85)

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
from ijaza.translations import TranslationProvider

processor = LLMProcessor(
    options=LLMProcessorOptions(
        auto_correct=True,
        min_confidence=0.85,
        scan_untagged=True,
        tag_format='xml',  # or 'markdown', 'bracket'
    ),
    translation_provider=TranslationProvider(),  # optional
)

# Get system prompt for your LLM
prompt = processor.get_system_prompt()

# Process LLM output
result = processor.process(llm_output)

# Translations are attached to each detected quote
for quote in result.quotes:
    print(quote.translations)  # {'en': '...', 'de': '...'}
```

### StreamingScanner

```python
from ijaza import StreamingScanner, StreamingScannerOptions
from ijaza.translations import TranslationProvider

scanner = StreamingScanner(
    options=StreamingScannerOptions(
        overlap_words=10,        # words retained between chunks
        min_confidence=0.85,
        min_words=3,
        max_words=50,
        max_chunk_span=3,        # max chunks a partial can span
        asr_tolerant=True,
    ),
    translation_provider=TranslationProvider(),  # optional
)

result = scanner.process_chunk("text chunk...")
# result.complete_verses — fully detected verses
# result.partial_verse — verse in progress at chunk boundary

final = scanner.flush()   # emit remaining at end of stream
scanner.reset()           # reset for new stream
```

### TranslationProvider

```python
from ijaza.translations import TranslationProvider, TranslationConfig, TRUSTED_EDITIONS

# Default: Sahih International (en) + Bubenheim (de)
provider = TranslationProvider()

# Custom editions
provider = TranslationProvider(TranslationConfig(
    editions={'en': 'en.pickthall', 'de': 'de.aburida'}
))

# Look up translations
en = provider.get_translation(surah=1, ayah=1, lang='en')
all_langs = provider.get_translations(surah=1, ayah=1)  # {'en': '...', 'de': '...'}

# Check availability
print(TRUSTED_EDITIONS)  # all known edition identifiers
provider.is_edition_available('en.sahih')  # True
```

### ASR Tolerance Utilities

```python
from ijaza.asr_tolerance import (
    calculate_asr_similarity,    # phonetic-aware string similarity
    preprocess_asr_text,         # stutter removal + boundary fixes
    get_substitution_cost,       # cost for a single char pair
    PHONETIC_CONFUSIONS,         # list of (char_a, char_b, cost) tuples
    FUNCTION_WORDS,              # set of Arabic particles ASR drops
)

# Phonetic-aware similarity (ص and س cost only 0.3 instead of 1.0)
sim = calculate_asr_similarity("الصراط", "السراط")  # ~0.95

# Preprocess ASR output
clean = preprocess_asr_text("قل قل هو  الله")  # "قل هو الله"
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

## Future Work

### Framework Integrations
- LangChain / LlamaIndex guardrails
- FastAPI middleware
- Streamlit components
- Django/Flask integration

### Performance Optimizations
- N-gram indexing for pre-filtering candidates (faster `scan_for_verses`)
- BK-tree for metric-space nearest-neighbor search

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT

## Acknowledgments

- [Yazin Alirhayim](https://github.com/yazinsai) for the original [quran-validator](https://github.com/yazinsai/quran-validator) npm package
- [AlQuran.cloud](https://alquran.cloud/) for the Quran API
- The [PolyKhateeb](https://github.com/memoelsamadony/polykhateeb) project team
