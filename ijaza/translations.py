"""
Quran translation support.

Provides access to trusted scholarly translations of Quran verses.
Translations are loaded lazily on first access to keep memory usage low.
"""

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Optional


# Default translations to offer
DEFAULT_EDITIONS: dict[str, str] = {
    'en': 'en.sahih',
    'de': 'de.bubenheim',
}

# All known trusted editions
TRUSTED_EDITIONS: dict[str, str] = {
    'en.sahih': 'Sahih International',
    'en.pickthall': 'Pickthall',
    'en.yusufali': 'Yusuf Ali',
    'en.asad': 'Muhammad Asad',
    'en.hilali': 'Muhsin Khan & Hilali',
    'en.itani': 'Clear Quran (Itani)',
    'de.bubenheim': 'Bubenheim & Elyas',
    'de.aburida': 'Abu Rida',
    'de.khoury': 'Khoury',
    'de.zaidan': 'Zaidan',
}


@dataclass
class TranslationVerse:
    """A single verse translation."""

    id: int
    surah: int
    ayah: int
    text: str
    edition: str


@dataclass
class TranslationConfig:
    """Configuration for which translations to load."""

    # Map of language code -> edition identifier
    # e.g., {'en': 'en.sahih', 'de': 'de.bubenheim'}
    editions: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_EDITIONS))


class TranslationProvider:
    """
    Lazy-loading provider for Quran translations.

    Translations are loaded from bundled JSON files on first access.

    Example:
        >>> provider = TranslationProvider()
        >>> en = provider.get_translation(surah=1, ayah=1, lang='en')
        >>> print(en)  # "In the name of Allah, the Entirely Merciful, ..."
    """

    def __init__(self, config: Optional[TranslationConfig] = None):
        if config is None:
            config = TranslationConfig()
        self.config = config

        # Lazy-loaded caches: edition -> {(surah, ayah) -> TranslationVerse}
        self._loaded: dict[str, dict[tuple[int, int], TranslationVerse]] = {}

    def _load_edition(self, edition: str) -> dict[tuple[int, int], TranslationVerse]:
        """Load a translation edition from bundled JSON data."""
        if edition in self._loaded:
            return self._loaded[edition]

        data = self._load_translation_json(edition)

        translation_map: dict[tuple[int, int], TranslationVerse] = {}
        for entry in data:
            tv = TranslationVerse(
                id=entry['id'],
                surah=entry['surah'],
                ayah=entry['ayah'],
                text=entry['text'],
                edition=edition,
            )
            translation_map[(tv.surah, tv.ayah)] = tv

        self._loaded[edition] = translation_map
        return translation_map

    def _load_translation_json(self, edition: str) -> list:
        """Load translation JSON file."""
        candidates = [f"translations/{edition}.min.json", f"translations/{edition}.json"]

        # Primary path: packaged resources
        package_data_dir = resources.files('ijaza').joinpath('data')
        for candidate in candidates:
            resource_file = package_data_dir.joinpath(candidate)
            if resource_file.is_file():
                with resource_file.open('r', encoding='utf-8') as f:
                    return json.load(f)

        # Legacy fallback: repository-level data/translations/ directory
        legacy_data_dir = Path(__file__).parent.parent / 'data' / 'translations'
        for candidate in [f"{edition}.min.json", f"{edition}.json"]:
            legacy_file = legacy_data_dir / candidate
            if legacy_file.exists():
                with open(legacy_file, 'r', encoding='utf-8') as f:
                    return json.load(f)

        raise FileNotFoundError(
            f"Translation data not found for edition '{edition}'. "
            f"Run: python scripts/fetch_translations.py --editions {edition}"
        )

    def get_translation(
        self,
        surah: int,
        ayah: int,
        lang: str = 'en',
    ) -> Optional[str]:
        """
        Get the translation text for a specific verse.

        Args:
            surah: Surah number (1-114)
            ayah: Ayah number
            lang: Language code ('en', 'de', etc.)

        Returns:
            Translation text, or None if not available
        """
        edition = self.config.editions.get(lang)
        if not edition:
            return None

        try:
            translation_map = self._load_edition(edition)
        except FileNotFoundError:
            return None

        tv = translation_map.get((surah, ayah))
        return tv.text if tv else None

    def get_translations(
        self,
        surah: int,
        ayah: int,
    ) -> dict[str, str]:
        """
        Get all configured translations for a verse.

        Returns:
            Dict of lang_code -> translation text
        """
        result: dict[str, str] = {}
        for lang in self.config.editions:
            text = self.get_translation(surah, ayah, lang)
            if text:
                result[lang] = text
        return result

    def is_edition_available(self, edition: str) -> bool:
        """Check if a translation file is available for an edition."""
        try:
            self._load_edition(edition)
            return True
        except FileNotFoundError:
            return False
