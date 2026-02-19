#!/usr/bin/env python3
"""
Fetch Quran translation data from AlQuran.cloud API.

Fetches scholarly translations and generates JSON data files.

Run with: python scripts/fetch_translations.py [--editions en.sahih de.bubenheim ...]

Default editions:
  - en.sahih (Sahih International - English)
  - de.bubenheim (Bubenheim & Elyas - German)

Output:
  - ijaza/data/translations/{edition}.json
  - ijaza/data/translations/{edition}.min.json
"""

import json
import sys
import urllib.request
import urllib.error
from pathlib import Path


# Trusted scholarly translation editions from AlQuran.cloud
TRUSTED_EDITIONS = {
    # English
    'en.sahih': 'Sahih International',
    'en.pickthall': 'Pickthall',
    'en.yusufali': 'Yusuf Ali',
    'en.asad': 'Muhammad Asad',
    'en.hilali': 'Muhsin Khan & Hilali',
    'en.itani': 'Clear Quran (Itani)',
    # German
    'de.bubenheim': 'Bubenheim & Elyas',
    'de.aburida': 'Abu Rida',
    'de.khoury': 'Khoury',
    'de.zaidan': 'Zaidan',
}

DEFAULT_EDITIONS = ['en.sahih', 'de.bubenheim']


def fetch_edition(edition: str) -> list[dict]:
    """Fetch all verses for a single translation edition."""
    url = f"https://api.alquran.cloud/v1/quran/{edition}"

    print(f"  Fetching {edition} ({TRUSTED_EDITIONS.get(edition, 'unknown')})...")
    req = urllib.request.Request(url, headers={'User-Agent': 'ijaza/1.0'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode('utf-8'))

    if data.get("code") != 200:
        raise RuntimeError(f"API error for {edition}: {data.get('status')}")

    verses: list[dict] = []
    verse_id = 1

    for surah in data["data"]["surahs"]:
        for ayah in surah["ayahs"]:
            verses.append({
                "id": verse_id,
                "surah": surah["number"],
                "ayah": ayah["numberInSurah"],
                "text": ayah["text"],
            })
            verse_id += 1

    return verses


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Quran translations")
    parser.add_argument(
        '--editions',
        nargs='+',
        default=DEFAULT_EDITIONS,
        help=f"Edition identifiers. Defaults: {DEFAULT_EDITIONS}",
    )
    parser.add_argument(
        '--list-editions',
        action='store_true',
        help="List all trusted editions and exit",
    )
    args = parser.parse_args()

    if args.list_editions:
        for code, name in sorted(TRUSTED_EDITIONS.items()):
            print(f"  {code:20s}  {name}")
        return

    # Validate editions
    for edition in args.editions:
        if edition not in TRUSTED_EDITIONS:
            print(f"Warning: {edition} is not in the trusted editions list.")

    # Output directory
    script_dir = Path(__file__).parent
    translations_dir = script_dir.parent / "ijaza" / "data" / "translations"
    translations_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {len(args.editions)} translation(s)...")

    for edition in args.editions:
        try:
            verses = fetch_edition(edition)
        except Exception as e:
            print(f"  Error fetching {edition}: {e}")
            continue

        # Write formatted version
        out_path = translations_dir / f"{edition}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(verses, f, ensure_ascii=False, indent=2)

        # Write minified version
        min_path = translations_dir / f"{edition}.min.json"
        with open(min_path, "w", encoding="utf-8") as f:
            json.dump(verses, f, ensure_ascii=False, separators=(',', ':'))

        print(f"  Wrote {len(verses)} verses to {out_path}")

    print("Done!")


if __name__ == "__main__":
    main()
