# Quran Validator — Package Structure & PyPI Publishing

## Package Structure

```
quran_validator_py/
├── pyproject.toml              # Package metadata and build config
├── MANIFEST.in                 # Files to include in distribution
├── LICENSE                     # MIT license
├── README.md                   # Package documentation
├── requirements.txt            # Runtime dependencies (none)
├── requirements-dev.txt        # Development dependencies
│
├── data/                       # Quran database (JSON files)
│   ├── quran-verses.json       # 6,236 verses with Uthmani script (2.6 MB)
│   ├── quran-verses.min.json   # Minified version (2.4 MB)
│   ├── quran-surahs.json       # 114 surahs metadata (16 KB)
│   ├── quran-surahs.min.json   # Minified version (12 KB)
│   └── normalized-index.json   # Pre-computed lookup index (770 KB)
│
├── quran_validator/            # Main package
│   ├── __init__.py             # Public API exports
│   ├── py.typed                # PEP 561 type hints marker
│   ├── types.py                # Dataclasses and type definitions
│   ├── normalizer.py           # Arabic text normalization utilities
│   ├── validator.py            # QuranValidator class
│   └── llm_integration.py      # LLMProcessor and system prompts
│
└── docs/                       # Documentation
    └── package-structure-and-publishing.md
```

## Module Overview

| Module | Description |
|--------|-------------|
| `types.py` | Dataclasses: `QuranVerse`, `QuranSurah`, `ValidationResult`, `DetectionResult`, `ValidatorOptions`, `NormalizationOptions` |
| `normalizer.py` | Arabic text processing: `normalize_arabic()`, `remove_diacritics()`, `contains_arabic()`, `extract_arabic_segments()`, `calculate_similarity()`, `find_differences()` |
| `validator.py` | Main validation: `QuranValidator` class with `validate()`, `detect_and_validate()`, `get_verse()`, `get_verse_range()`, `search()` |
| `llm_integration.py` | LLM tools: `LLMProcessor` class, `SYSTEM_PROMPTS`, `quick_validate()` |

## Data Files

| File | Records | Description |
|------|---------|-------------|
| `quran-verses.json` | 6,236 | All verses with `id`, `surah`, `ayah`, `text` (Uthmani), `textSimple` |
| `quran-surahs.json` | 114 | Surah metadata: number, Arabic/English name, revelation type, verse count |
| `normalized-index.json` | ~6,000 | Pre-computed normalized text → verse ID mapping for O(1) lookups |

---

## PyPI Publishing

### Prerequisites

1. **PyPI account**: https://pypi.org/account/register/
2. **API token**: https://pypi.org/manage/account/token/
3. **Build tools**:
   ```bash
   pip install build twine
   ```

### Step 1: Update Version

Edit both files to match:

**pyproject.toml**:
```toml
[project]
version = "1.0.0"
```

**quran_validator/__init__.py**:
```python
__version__ = "1.0.0"
```

### Step 2: Clean Previous Builds

```bash
rm -rf dist/ build/ *.egg-info/
```

### Step 3: Build the Package

```bash
python -m build
```

This creates:
```
dist/
├── quran_validator-1.0.0.tar.gz        # Source distribution
└── quran_validator-1.0.0-py3-none-any.whl  # Wheel (binary)
```

### Step 4: Verify Data Files Included

```bash
# Check source distribution
tar -tzf dist/quran_validator-1.0.0.tar.gz | grep -E "\.json$"

# Expected output:
# quran_validator-1.0.0/data/quran-verses.json
# quran_validator-1.0.0/data/quran-verses.min.json
# quran_validator-1.0.0/data/quran-surahs.json
# quran_validator-1.0.0/data/quran-surahs.min.json
# quran_validator-1.0.0/data/normalized-index.json
```

### Step 5: Test on TestPyPI (Recommended)

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# When prompted:
# Username: __token__
# Password: pypi-YOUR-TESTPYPI-TOKEN

# Test installation
pip install --index-url https://test.pypi.org/simple/ quran-validator

# Verify it works
python -c "from quran_validator import QuranValidator; v = QuranValidator(); print(f'Loaded {len(v.verses)} verses')"
```

### Step 6: Publish to PyPI

```bash
python -m twine upload dist/*

# When prompted:
# Username: __token__
# Password: pypi-YOUR-PYPI-TOKEN
```

### Step 7: Verify Publication

```bash
pip install quran-validator
python -c "from quran_validator import QuranValidator; print('Success!')"
```

---

## Configuration File (Optional)

Create `~/.pypirc` to avoid entering credentials each time:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-XXXXXXXXXXXXXXXXXXXX

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-XXXXXXXXXXXXXXXXXXXX
```

Secure it:
```bash
chmod 600 ~/.pypirc
```

---

## Quick Reference Commands

```bash
# Full publish workflow
cd quran_validator_py
rm -rf dist/ build/ *.egg-info/
python -m build
python -m twine upload --repository testpypi dist/*  # Test first
python -m twine upload dist/*                        # Production

# Check package info
python -m twine check dist/*

# Install locally for testing
pip install -e .

# Run tests (if you have them)
pytest
```

---

## Versioning

Follow [Semantic Versioning](https://semver.org/):

| Change Type | Version Bump | Example |
|-------------|--------------|---------|
| Breaking API changes | MAJOR | 1.0.0 → 2.0.0 |
| New features (backwards compatible) | MINOR | 1.0.0 → 1.1.0 |
| Bug fixes | PATCH | 1.0.0 → 1.0.1 |

---

## Troubleshooting

### "File already exists" error
PyPI doesn't allow overwriting versions. Bump the version number.

### Data files not included
Ensure `MANIFEST.in` exists with:
```
recursive-include data *.json
```

### Package name taken
Check https://pypi.org/project/quran-validator/. If taken, alternatives:
- `quran-text-validator`
- `quran-verse-validator`
- `quranic-validator`

### Import errors after install
Ensure the package structure matches `pyproject.toml`:
```toml
[tool.setuptools]
packages = ["quran_validator"]
include-package-data = true
```
