#!/usr/bin/env python3
"""
Process QUL (Quranic Universal Library) data.

This script converts QUL's JSON format to our internal format,
combining Uthmani (authoritative) and Imlaei simple (for matching).

The QUL data is from Tarteel AI's Quranic Universal Library project.

Run with: python scripts/process_qul_data.py --uthmani path/to/uthmani.json --imlaei path/to/imlaei-simple.json

Output files:
  - data/quran-verses.json      : Full verses with formatting
  - data/quran-verses.min.json  : Minified version
  - data/quran-surahs.json      : Surah metadata with formatting
  - data/quran-surahs.min.json  : Minified version
  - data/normalized-index.json  : Normalized text -> verse IDs mapping
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import TypedDict, List, Dict, Literal


class ProcessedVerse(TypedDict):
    id: int
    surah: int
    ayah: int
    text: str
    textSimple: str


class ProcessedSurah(TypedDict):
    number: int
    name: str
    englishName: str
    versesCount: int
    revelationType: Literal["Meccan", "Medinan"]


# Surah metadata (from Quran.com/AlQuran.cloud)
SURAH_METADATA: List[Dict] = [
    {"number": 1, "name": "الفاتحة", "englishName": "Al-Fatiha", "revelationType": "Meccan"},
    {"number": 2, "name": "البقرة", "englishName": "Al-Baqara", "revelationType": "Medinan"},
    {"number": 3, "name": "آل عمران", "englishName": "Aal-i-Imran", "revelationType": "Medinan"},
    {"number": 4, "name": "النساء", "englishName": "An-Nisa", "revelationType": "Medinan"},
    {"number": 5, "name": "المائدة", "englishName": "Al-Ma'ida", "revelationType": "Medinan"},
    {"number": 6, "name": "الأنعام", "englishName": "Al-An'am", "revelationType": "Meccan"},
    {"number": 7, "name": "الأعراف", "englishName": "Al-A'raf", "revelationType": "Meccan"},
    {"number": 8, "name": "الأنفال", "englishName": "Al-Anfal", "revelationType": "Medinan"},
    {"number": 9, "name": "التوبة", "englishName": "At-Tawba", "revelationType": "Medinan"},
    {"number": 10, "name": "يونس", "englishName": "Yunus", "revelationType": "Meccan"},
    {"number": 11, "name": "هود", "englishName": "Hud", "revelationType": "Meccan"},
    {"number": 12, "name": "يوسف", "englishName": "Yusuf", "revelationType": "Meccan"},
    {"number": 13, "name": "الرعد", "englishName": "Ar-Ra'd", "revelationType": "Medinan"},
    {"number": 14, "name": "إبراهيم", "englishName": "Ibrahim", "revelationType": "Meccan"},
    {"number": 15, "name": "الحجر", "englishName": "Al-Hijr", "revelationType": "Meccan"},
    {"number": 16, "name": "النحل", "englishName": "An-Nahl", "revelationType": "Meccan"},
    {"number": 17, "name": "الإسراء", "englishName": "Al-Isra", "revelationType": "Meccan"},
    {"number": 18, "name": "الكهف", "englishName": "Al-Kahf", "revelationType": "Meccan"},
    {"number": 19, "name": "مريم", "englishName": "Maryam", "revelationType": "Meccan"},
    {"number": 20, "name": "طه", "englishName": "Ta-Ha", "revelationType": "Meccan"},
    {"number": 21, "name": "الأنبياء", "englishName": "Al-Anbiya", "revelationType": "Meccan"},
    {"number": 22, "name": "الحج", "englishName": "Al-Hajj", "revelationType": "Medinan"},
    {"number": 23, "name": "المؤمنون", "englishName": "Al-Mu'minun", "revelationType": "Meccan"},
    {"number": 24, "name": "النور", "englishName": "An-Nur", "revelationType": "Medinan"},
    {"number": 25, "name": "الفرقان", "englishName": "Al-Furqan", "revelationType": "Meccan"},
    {"number": 26, "name": "الشعراء", "englishName": "Ash-Shu'ara", "revelationType": "Meccan"},
    {"number": 27, "name": "النمل", "englishName": "An-Naml", "revelationType": "Meccan"},
    {"number": 28, "name": "القصص", "englishName": "Al-Qasas", "revelationType": "Meccan"},
    {"number": 29, "name": "العنكبوت", "englishName": "Al-Ankabut", "revelationType": "Meccan"},
    {"number": 30, "name": "الروم", "englishName": "Ar-Rum", "revelationType": "Meccan"},
    {"number": 31, "name": "لقمان", "englishName": "Luqman", "revelationType": "Meccan"},
    {"number": 32, "name": "السجدة", "englishName": "As-Sajda", "revelationType": "Meccan"},
    {"number": 33, "name": "الأحزاب", "englishName": "Al-Ahzab", "revelationType": "Medinan"},
    {"number": 34, "name": "سبأ", "englishName": "Saba", "revelationType": "Meccan"},
    {"number": 35, "name": "فاطر", "englishName": "Fatir", "revelationType": "Meccan"},
    {"number": 36, "name": "يس", "englishName": "Ya-Sin", "revelationType": "Meccan"},
    {"number": 37, "name": "الصافات", "englishName": "As-Saffat", "revelationType": "Meccan"},
    {"number": 38, "name": "ص", "englishName": "Sad", "revelationType": "Meccan"},
    {"number": 39, "name": "الزمر", "englishName": "Az-Zumar", "revelationType": "Meccan"},
    {"number": 40, "name": "غافر", "englishName": "Ghafir", "revelationType": "Meccan"},
    {"number": 41, "name": "فصلت", "englishName": "Fussilat", "revelationType": "Meccan"},
    {"number": 42, "name": "الشورى", "englishName": "Ash-Shura", "revelationType": "Meccan"},
    {"number": 43, "name": "الزخرف", "englishName": "Az-Zukhruf", "revelationType": "Meccan"},
    {"number": 44, "name": "الدخان", "englishName": "Ad-Dukhan", "revelationType": "Meccan"},
    {"number": 45, "name": "الجاثية", "englishName": "Al-Jathiya", "revelationType": "Meccan"},
    {"number": 46, "name": "الأحقاف", "englishName": "Al-Ahqaf", "revelationType": "Meccan"},
    {"number": 47, "name": "محمد", "englishName": "Muhammad", "revelationType": "Medinan"},
    {"number": 48, "name": "الفتح", "englishName": "Al-Fath", "revelationType": "Medinan"},
    {"number": 49, "name": "الحجرات", "englishName": "Al-Hujurat", "revelationType": "Medinan"},
    {"number": 50, "name": "ق", "englishName": "Qaf", "revelationType": "Meccan"},
    {"number": 51, "name": "الذاريات", "englishName": "Adh-Dhariyat", "revelationType": "Meccan"},
    {"number": 52, "name": "الطور", "englishName": "At-Tur", "revelationType": "Meccan"},
    {"number": 53, "name": "النجم", "englishName": "An-Najm", "revelationType": "Meccan"},
    {"number": 54, "name": "القمر", "englishName": "Al-Qamar", "revelationType": "Meccan"},
    {"number": 55, "name": "الرحمن", "englishName": "Ar-Rahman", "revelationType": "Medinan"},
    {"number": 56, "name": "الواقعة", "englishName": "Al-Waqi'a", "revelationType": "Meccan"},
    {"number": 57, "name": "الحديد", "englishName": "Al-Hadid", "revelationType": "Medinan"},
    {"number": 58, "name": "المجادلة", "englishName": "Al-Mujadila", "revelationType": "Medinan"},
    {"number": 59, "name": "الحشر", "englishName": "Al-Hashr", "revelationType": "Medinan"},
    {"number": 60, "name": "الممتحنة", "englishName": "Al-Mumtahina", "revelationType": "Medinan"},
    {"number": 61, "name": "الصف", "englishName": "As-Saff", "revelationType": "Medinan"},
    {"number": 62, "name": "الجمعة", "englishName": "Al-Jumu'a", "revelationType": "Medinan"},
    {"number": 63, "name": "المنافقون", "englishName": "Al-Munafiqun", "revelationType": "Medinan"},
    {"number": 64, "name": "التغابن", "englishName": "At-Taghabun", "revelationType": "Medinan"},
    {"number": 65, "name": "الطلاق", "englishName": "At-Talaq", "revelationType": "Medinan"},
    {"number": 66, "name": "التحريم", "englishName": "At-Tahrim", "revelationType": "Medinan"},
    {"number": 67, "name": "الملك", "englishName": "Al-Mulk", "revelationType": "Meccan"},
    {"number": 68, "name": "القلم", "englishName": "Al-Qalam", "revelationType": "Meccan"},
    {"number": 69, "name": "الحاقة", "englishName": "Al-Haqqa", "revelationType": "Meccan"},
    {"number": 70, "name": "المعارج", "englishName": "Al-Ma'arij", "revelationType": "Meccan"},
    {"number": 71, "name": "نوح", "englishName": "Nuh", "revelationType": "Meccan"},
    {"number": 72, "name": "الجن", "englishName": "Al-Jinn", "revelationType": "Meccan"},
    {"number": 73, "name": "المزمل", "englishName": "Al-Muzzammil", "revelationType": "Meccan"},
    {"number": 74, "name": "المدثر", "englishName": "Al-Muddaththir", "revelationType": "Meccan"},
    {"number": 75, "name": "القيامة", "englishName": "Al-Qiyama", "revelationType": "Meccan"},
    {"number": 76, "name": "الإنسان", "englishName": "Al-Insan", "revelationType": "Medinan"},
    {"number": 77, "name": "المرسلات", "englishName": "Al-Mursalat", "revelationType": "Meccan"},
    {"number": 78, "name": "النبأ", "englishName": "An-Naba", "revelationType": "Meccan"},
    {"number": 79, "name": "النازعات", "englishName": "An-Nazi'at", "revelationType": "Meccan"},
    {"number": 80, "name": "عبس", "englishName": "Abasa", "revelationType": "Meccan"},
    {"number": 81, "name": "التكوير", "englishName": "At-Takwir", "revelationType": "Meccan"},
    {"number": 82, "name": "الانفطار", "englishName": "Al-Infitar", "revelationType": "Meccan"},
    {"number": 83, "name": "المطففين", "englishName": "Al-Mutaffifin", "revelationType": "Meccan"},
    {"number": 84, "name": "الانشقاق", "englishName": "Al-Inshiqaq", "revelationType": "Meccan"},
    {"number": 85, "name": "البروج", "englishName": "Al-Buruj", "revelationType": "Meccan"},
    {"number": 86, "name": "الطارق", "englishName": "At-Tariq", "revelationType": "Meccan"},
    {"number": 87, "name": "الأعلى", "englishName": "Al-A'la", "revelationType": "Meccan"},
    {"number": 88, "name": "الغاشية", "englishName": "Al-Ghashiya", "revelationType": "Meccan"},
    {"number": 89, "name": "الفجر", "englishName": "Al-Fajr", "revelationType": "Meccan"},
    {"number": 90, "name": "البلد", "englishName": "Al-Balad", "revelationType": "Meccan"},
    {"number": 91, "name": "الشمس", "englishName": "Ash-Shams", "revelationType": "Meccan"},
    {"number": 92, "name": "الليل", "englishName": "Al-Layl", "revelationType": "Meccan"},
    {"number": 93, "name": "الضحى", "englishName": "Ad-Duha", "revelationType": "Meccan"},
    {"number": 94, "name": "الشرح", "englishName": "Ash-Sharh", "revelationType": "Meccan"},
    {"number": 95, "name": "التين", "englishName": "At-Tin", "revelationType": "Meccan"},
    {"number": 96, "name": "العلق", "englishName": "Al-Alaq", "revelationType": "Meccan"},
    {"number": 97, "name": "القدر", "englishName": "Al-Qadr", "revelationType": "Meccan"},
    {"number": 98, "name": "البينة", "englishName": "Al-Bayyina", "revelationType": "Medinan"},
    {"number": 99, "name": "الزلزلة", "englishName": "Az-Zalzala", "revelationType": "Medinan"},
    {"number": 100, "name": "العاديات", "englishName": "Al-Adiyat", "revelationType": "Meccan"},
    {"number": 101, "name": "القارعة", "englishName": "Al-Qari'a", "revelationType": "Meccan"},
    {"number": 102, "name": "التكاثر", "englishName": "At-Takathur", "revelationType": "Meccan"},
    {"number": 103, "name": "العصر", "englishName": "Al-Asr", "revelationType": "Meccan"},
    {"number": 104, "name": "الهمزة", "englishName": "Al-Humaza", "revelationType": "Meccan"},
    {"number": 105, "name": "الفيل", "englishName": "Al-Fil", "revelationType": "Meccan"},
    {"number": 106, "name": "قريش", "englishName": "Quraysh", "revelationType": "Meccan"},
    {"number": 107, "name": "الماعون", "englishName": "Al-Ma'un", "revelationType": "Meccan"},
    {"number": 108, "name": "الكوثر", "englishName": "Al-Kawthar", "revelationType": "Meccan"},
    {"number": 109, "name": "الكافرون", "englishName": "Al-Kafirun", "revelationType": "Meccan"},
    {"number": 110, "name": "النصر", "englishName": "An-Nasr", "revelationType": "Medinan"},
    {"number": 111, "name": "المسد", "englishName": "Al-Masad", "revelationType": "Meccan"},
    {"number": 112, "name": "الإخلاص", "englishName": "Al-Ikhlas", "revelationType": "Meccan"},
    {"number": 113, "name": "الفلق", "englishName": "Al-Falaq", "revelationType": "Meccan"},
    {"number": 114, "name": "الناس", "englishName": "An-Nas", "revelationType": "Meccan"},
]


def normalize_for_index(text: str) -> str:
    """
    Normalize Arabic text for indexing.

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


def process_qul_data(uthmani_path: str, imlaei_path: str):
    """Process QUL data files and generate ijaza data files."""
    print("Loading QUL data...")

    # Load Uthmani data (keyed by verse_key like "1:1")
    with open(uthmani_path, 'r', encoding='utf-8') as f:
        uthmani_data: Dict = json.load(f)

    # Load Imlaei word data (keyed by location like "1:1:1")
    with open(imlaei_path, 'r', encoding='utf-8') as f:
        imlaei_data: Dict = json.load(f)

    print(f"Loaded {len(uthmani_data)} Uthmani verses")
    print(f"Loaded {len(imlaei_data)} Imlaei words")

    # Aggregate Imlaei words into ayahs
    print("Aggregating Imlaei words into ayahs...")
    imlaei_ayahs: Dict[str, List[str]] = {}

    for location, word in imlaei_data.items():
        verse_key = f"{word['surah']}:{word['ayah']}"
        if verse_key not in imlaei_ayahs:
            imlaei_ayahs[verse_key] = []

        # Store at the correct word index
        word_index = int(word['word']) - 1

        # Extend list if needed
        while len(imlaei_ayahs[verse_key]) <= word_index:
            imlaei_ayahs[verse_key].append('')

        imlaei_ayahs[verse_key][word_index] = word['text']

    # Join words into full ayahs (removing verse number markers like ١٢٣)
    arabic_numerals = re.compile(r'[٠١٢٣٤٥٦٧٨٩]+')
    imlaei_verses: Dict[str, str] = {}

    for verse_key, words in imlaei_ayahs.items():
        full_text = ' '.join(w for w in words if w)
        # Remove Arabic verse numbers that appear at the end
        imlaei_verses[verse_key] = arabic_numerals.sub('', full_text).strip()

    print(f"Aggregated {len(imlaei_verses)} Imlaei ayahs")

    # Process into our format
    print("Processing verses...")
    verses: List[ProcessedVerse] = []
    surah_verse_counts: Dict[int, int] = {}

    for verse_key, uthmani in uthmani_data.items():
        imlaei_text = imlaei_verses.get(verse_key, '')

        verses.append({
            "id": uthmani['id'],
            "surah": uthmani['surah'],
            "ayah": uthmani['ayah'],
            "text": uthmani['text'],
            "textSimple": imlaei_text,
        })

        # Count verses per surah
        surah_num = uthmani['surah']
        surah_verse_counts[surah_num] = surah_verse_counts.get(surah_num, 0) + 1

    # Sort by ID to ensure correct order
    verses.sort(key=lambda v: v['id'])

    # Build surah metadata with verse counts
    surahs: List[ProcessedSurah] = []
    for meta in SURAH_METADATA:
        surahs.append({
            "number": meta["number"],
            "name": meta["name"],
            "englishName": meta["englishName"],
            "versesCount": surah_verse_counts.get(meta["number"], 0),
            "revelationType": meta["revelationType"],
        })

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
        # Use the Imlaei simple text for indexing (already simplified)
        normalized = normalize_for_index(verse["textSimple"])
        if normalized not in normalized_index:
            normalized_index[normalized] = []
        normalized_index[normalized].append(verse["id"])

    index_path = data_dir / "normalized-index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(normalized_index, f, ensure_ascii=False, separators=(',', ':'))
    print(f"Wrote normalized index to {index_path}")

    # Print sample verses for verification
    print("\n--- Sample verses (QUL data) ---")
    if verses:
        print(f"1:1 Uthmani: {verses[0]['text']}")
        print(f"1:1 Imlaei: {verses[0]['textSimple']}")

        # Find Ayat al-Kursi
        ayat_kursi = next((v for v in verses if v['surah'] == 2 and v['ayah'] == 255), None)
        if ayat_kursi:
            print("\n2:255 (Ayat al-Kursi):")
            print(f"Uthmani: {ayat_kursi['text'][:100]}...")
            print(f"Imlaei: {ayat_kursi['textSimple'][:100]}...")

    print("\nDone! QUL data processed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Process QUL (Quranic Universal Library) data files."
    )
    parser.add_argument(
        "--uthmani",
        required=True,
        help="Path to uthmani.json file from QUL"
    )
    parser.add_argument(
        "--imlaei",
        required=True,
        help="Path to imlaei-simple.json file from QUL"
    )

    args = parser.parse_args()

    # Validate paths
    uthmani_path = Path(args.uthmani)
    imlaei_path = Path(args.imlaei)

    if not uthmani_path.exists():
        print(f"Error: Uthmani file not found: {uthmani_path}")
        sys.exit(1)

    if not imlaei_path.exists():
        print(f"Error: Imlaei file not found: {imlaei_path}")
        sys.exit(1)

    process_qul_data(str(uthmani_path), str(imlaei_path))


if __name__ == "__main__":
    main()
