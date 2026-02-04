#!/usr/bin/env python3
"""
Script to fetch Quran data from AlQuran.cloud API.

This script fetches the complete Quran text in Uthmani script from the
AlQuran.cloud API and generates the JSON data files used by ijaza.

Run with: python scripts/fetch_quran_data.py

Output files:
  - data/quran-verses.json      : Full verses with formatting
  - data/quran-verses.min.json  : Minified version
  - data/quran-surahs.json      : Surah metadata with formatting
  - data/quran-surahs.min.json  : Minified version
  - data/normalized-index.json  : Normalized text -> verse IDs mapping
"""

import json
import re
import sys
from pathlib import Path
from typing import TypedDict, List, Dict

try:
    import requests
except ImportError:
    print("Error: requests library not installed. Run: pip install requests")
    sys.exit(1)


class ProcessedVerse(TypedDict):
    id: int
    surah: int
    ayah: int
    text: str
    textSimple: str
    page: int
    juz: int


class ProcessedSurah(TypedDict):
    number: int
    name: str
    englishName: str
    versesCount: int
    revelationType: str


def clean_text(text: str) -> str:
    """Remove BOM and other invisible characters."""
    return re.sub(r'[\uFEFF\u200B-\u200D\uFFFE\uFFFF]', '', text)


def remove_diacritics(text: str) -> str:
    """Remove Arabic diacritics (tashkeel) from text."""
    return re.sub(r'[\u064B-\u065F\u0670\u06D6-\u06ED]', '', text)


def normalize_for_index(text: str) -> str:
    """
    Normalize Arabic text for indexing.

    - Removes diacritics
    - Normalizes alef variants
    - Normalizes ya/alef maqsura
    - Normalizes ta marbuta
    - Normalizes hamza carriers
    - Collapses whitespace
    """
    normalized = text
    # Normalize alef variants
    normalized = re.sub(r'[أإآٱ]', 'ا', normalized)
    # Normalize alef maqsura to ya
    normalized = normalized.replace('ى', 'ي')
    # Normalize ta marbuta to ha
    normalized = normalized.replace('ة', 'ه')
    # Normalize hamza on waw
    normalized = normalized.replace('ؤ', 'و')
    # Normalize hamza on ya
    normalized = normalized.replace('ئ', 'ي')
    # Collapse whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def fetch_quran_data():
    """Fetch and process Quran data from AlQuran.cloud API."""
    print("Fetching Quran data from AlQuran.cloud API...")

    # Fetch Uthmani text (with diacritics)
    url = "https://api.alquran.cloud/v1/quran/quran-uthmani"

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error: Failed to fetch Quran data: {e}")
        sys.exit(1)

    data = response.json()

    if data.get("code") != 200:
        print(f"Error: API error: {data.get('status')}")
        sys.exit(1)

    print("Processing verses...")

    verses: List[ProcessedVerse] = []
    surahs: List[ProcessedSurah] = []

    verse_id = 1

    for surah in data["data"]["surahs"]:
        # Add surah info
        surahs.append({
            "number": surah["number"],
            "name": surah["name"],
            "englishName": surah["englishName"],
            "versesCount": len(surah["ayahs"]),
            "revelationType": "Meccan" if surah["revelationType"] == "Meccan" else "Medinan",
        })

        # Add verses
        for ayah in surah["ayahs"]:
            cleaned_text = clean_text(ayah["text"])
            verses.append({
                "id": verse_id,
                "surah": surah["number"],
                "ayah": ayah["numberInSurah"],
                "text": cleaned_text,
                "textSimple": remove_diacritics(cleaned_text),
                "page": ayah.get("page", 0),
                "juz": ayah.get("juz", 0),
            })
            verse_id += 1

    print(f"Processed {len(verses)} verses from {len(surahs)} surahs")

    # Determine data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Write verses
    verses_path = data_dir / "quran-verses.json"
    with open(verses_path, "w", encoding="utf-8") as f:
        json.dump(verses, f, ensure_ascii=False, indent=2)
    print(f"Wrote verses to {verses_path}")

    # Write surahs
    surahs_path = data_dir / "quran-surahs.json"
    with open(surahs_path, "w", encoding="utf-8") as f:
        json.dump(surahs, f, ensure_ascii=False, indent=2)
    print(f"Wrote surahs to {surahs_path}")

    # Write minified versions for production
    verses_min_path = data_dir / "quran-verses.min.json"
    with open(verses_min_path, "w", encoding="utf-8") as f:
        json.dump(verses, f, ensure_ascii=False, separators=(',', ':'))
    print(f"Wrote minified verses to {verses_min_path}")

    surahs_min_path = data_dir / "quran-surahs.min.json"
    with open(surahs_min_path, "w", encoding="utf-8") as f:
        json.dump(surahs, f, ensure_ascii=False, separators=(',', ':'))
    print(f"Wrote minified surahs to {surahs_min_path}")

    # Create a normalized index for faster lookups
    print("Creating normalized index...")
    normalized_index: Dict[str, List[int]] = {}

    for verse in verses:
        normalized = normalize_for_index(verse["textSimple"])
        if normalized not in normalized_index:
            normalized_index[normalized] = []
        normalized_index[normalized].append(verse["id"])

    index_path = data_dir / "normalized-index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(normalized_index, f, ensure_ascii=False, separators=(',', ':'))
    print(f"Wrote normalized index to {index_path}")

    print("Done!")


if __name__ == "__main__":
    fetch_quran_data()
