# Publishing to PyPI

This guide covers how to publish `ijaza` to PyPI.

## Prerequisites

1. **Create PyPI account**: Register at https://pypi.org/account/register/

2. **Create TestPyPI account** (recommended for testing): https://test.pypi.org/account/register/

3. **Generate API tokens**:
   - PyPI: https://pypi.org/manage/account/token/
   - TestPyPI: https://test.pypi.org/manage/account/token/

4. **Install build tools**:
   ```bash
   pip install build twine
   ```

## Pre-publish Checklist

Before publishing, ensure:

- [ ] Version number is updated in `pyproject.toml` and `ijaza/__init__.py`
- [ ] All tests pass
- [ ] README.md is complete and accurate
- [ ] License file exists (create LICENSE with MIT text)
- [ ] Data files are included in the package

## Step-by-Step Publishing

### 1. Update Version

Edit `pyproject.toml`:
```toml
[project]
version = "1.0.0"  # Update this
```

Edit `ijaza/__init__.py`:
```python
__version__ = "1.0.0"  # Update this
```

### 2. Create LICENSE File

```bash
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2024 Yazin Alirhayim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
```

### 3. Clean Previous Builds

```bash
rm -rf dist/ build/ *.egg-info/
```

### 4. Build the Package

```bash
python -m build
```

This creates:
- `dist/ijaza-1.0.0.tar.gz` (source distribution)
- `dist/ijaza-1.0.0-py3-none-any.whl` (wheel)

### 5. Verify the Build

Check the contents:
```bash
tar -tzf dist/ijaza-1.0.0.tar.gz | head -20
unzip -l dist/ijaza-1.0.0-py3-none-any.whl | head -20
```

Ensure data files are included:
```bash
tar -tzf dist/ijaza-1.0.0.tar.gz | grep -E "\.json$"
```

### 6. Test on TestPyPI (Recommended)

Upload to TestPyPI first:
```bash
python -m twine upload --repository testpypi dist/*
```

You'll be prompted for credentials:
- Username: `__token__`
- Password: Your TestPyPI API token (starts with `pypi-`)

Or use a `.pypirc` file (see below).

Test installation:
```bash
pip install --index-url https://test.pypi.org/simple/ ijaza
```

### 7. Publish to PyPI

Once verified on TestPyPI:
```bash
python -m twine upload dist/*
```

Credentials:
- Username: `__token__`
- Password: Your PyPI API token

### 8. Verify Publication

```bash
pip install ijaza
python -c "from ijaza import QuranValidator; print('Success!')"
```

## Configuration File (Optional)

Create `~/.pypirc` for easier uploads:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR-TOKEN-HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR-TESTPYPI-TOKEN-HERE
```

**Important**: Secure this file:
```bash
chmod 600 ~/.pypirc
```

## Updating the Package

For subsequent releases:

1. Update version in `pyproject.toml` and `__init__.py`
2. Update CHANGELOG.md (if you have one)
3. Clean, build, and upload:
   ```bash
   rm -rf dist/ build/
   python -m build
   python -m twine upload dist/*
   ```

## Versioning Guidelines

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR** (1.0.0 → 2.0.0): Breaking API changes
- **MINOR** (1.0.0 → 1.1.0): New features, backwards compatible
- **PATCH** (1.0.0 → 1.0.1): Bug fixes, backwards compatible

## Troubleshooting

### "File already exists" Error
You cannot overwrite an existing version. Bump the version number.

### Data Files Not Included
Ensure `pyproject.toml` has:
```toml
[tool.setuptools]
include-package-data = true

[tool.setuptools.package-data]
"*" = ["*.json"]
```

And add a `MANIFEST.in` if needed:
```
recursive-include data *.json
```

### Package Name Taken
Check https://pypi.org/project/ijaza/ — if taken, choose a different name like:
- `quran-text-validator`
- `quran-verse-validator`
- `quranic-validator`

Update the name in `pyproject.toml`.

## GitHub Actions (CI/CD)

For automated publishing, create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install build twine
      - name: Build
        run: python -m build
      - name: Publish
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

Add your PyPI token as a repository secret named `PYPI_API_TOKEN`.
