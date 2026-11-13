"""Translate the recognized English text into Hindi and Kannada.

How it works: each recognized word (or phrase) is looked up in translations.csv.
 - Longest phrase first, so "thank you" is one entry and "how you" can become "how are you".
 - Words that are not in the table (for example a name you spelled with letters) stay as typed, or
   are written phonetically if the optional package 'indic-transliteration' is installed
   (pip install indic-transliteration). Spell long vowels with double letters: kumaar, ravii.

The table is a plain CSV (english, hindi, kannada). Open it in Notepad or Excel to fix or add words.
This is a word-by-word translation: it does NOT rearrange words into correct Hindi/Kannada grammar.
"""
import csv
import os

CSV_PATH = "translations.csv"

try:                                    # optional
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
except Exception:                       # not installed: unknown words stay as typed
    sanscript = None
    transliterate = None


def load_translations(path=CSV_PATH):
    """-> {'thank you': {'hindi': '...', 'kannada': '...'}, ...}  (empty if the file is missing)"""
    table = {}
    if not os.path.exists(path):
        return table
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (row.get("english") or "").strip().lower()
            if key:
                table[key] = {"hindi": (row.get("hindi") or "").strip(),
                              "kannada": (row.get("kannada") or "").strip()}
    return table


def _spell_out(word, lang):
    """Phonetic fallback for an unknown word typed in letters (needs indic-transliteration)."""
    if transliterate is None or not word.isascii() or not word.isalpha():
        return word
    target = sanscript.DEVANAGARI if lang == "hindi" else sanscript.KANNADA
    try:
        return transliterate(word, sanscript.ITRANS, target)
    except Exception:
        return word


def translate(text, lang, table, max_phrase=3):
    """lang = 'english', 'hindi' or 'kannada'."""
    if lang == "english" or not text.strip():
        return text
    tokens = text.split()
    out, i = [], 0
    while i < len(tokens):
        for n in range(min(max_phrase, len(tokens) - i), 0, -1):
            hit = table.get(" ".join(tokens[i:i + n]).lower(), {}).get(lang)
            if hit:
                out.append(hit)
                i += n
                break
        else:                           # no entry found for this word
            out.append(_spell_out(tokens[i], lang))
            i += 1
    return " ".join(out)
