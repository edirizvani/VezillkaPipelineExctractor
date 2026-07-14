"""
Vezilka v2 — Shared Text Utilities.

Common text processing functions used across all pipeline phases.
"""

from __future__ import annotations

import re
import unicodedata

# ── Compiled patterns ───────────────────────────────────────────
HYPHEN_PATTERN = re.compile(r"(\w)\s*[-–]\s*([a-zа-ш])")
HEX_HASH_PATTERN = re.compile(r"\b[0-9a-f]{32,}\b", re.IGNORECASE)
MULTI_SPACE = re.compile(r"\s+")
PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")
HEADER_FOOTER = re.compile(
    r"(СЛУЖБЕН\s*ВЕСНИК|GAZETTE\s*ZYRTARE|Стр\.\s*\d+\s*-\s*Бр\.\s*\d+|Faqe\s*\d+)",
    re.IGNORECASE,
)
DATE_HEADER = re.compile(
    r"\d{1,2}\s*(јануари|февруари|март|април|мај|јуни|јули|август|септември|"
    r"октомври|ноември|декември|janar|shkurt|mars|prill|maj|qershor|"
    r"korrik|gusht|shtator|tetor|nëntor|dhjetor)\s*\d{4}",
    re.IGNORECASE,
)
ALBANIAN_SPECIFIC_CHARS = frozenset("ëËçÇ")


def fix_albanian_encoding(text: str) -> str:
    """Fix backtick → ë in Albanian text from older PDF font mappings."""
    return text.replace("`", "ë").replace("\x60", "ë")


def fix_hyphenation(text: str) -> str:
    """Rejoin words broken across PDF lines: 'Ар - мијата' → 'Армијата'."""
    return HYPHEN_PATTERN.sub(r"\1\2", text)


def remove_hex_hashes(text: str) -> str:
    """Remove 32+ char hex strings leaked from PDF internal processing."""
    return HEX_HASH_PATTERN.sub("", text).strip()


def normalise_whitespace(text: str) -> str:
    """Collapse multiple whitespace within lines but preserve newlines."""
    lines = text.split("\n")
    cleaned = [re.sub(r"[^\S\n]+", " ", line).strip() for line in lines]
    return "\n".join(cleaned).strip()


def normalise_unicode(text: str) -> str:
    """Apply NFC unicode normalisation."""
    return unicodedata.normalize("NFC", text)


def clean_text(text: str, is_albanian: bool = False) -> str:
    """Apply all text fixes in the correct order."""
    if not text:
        return ""
    text = normalise_unicode(text)
    if is_albanian:
        text = fix_albanian_encoding(text)
    text = fix_hyphenation(text)
    text = remove_hex_hashes(text)
    # Don't call normalise_whitespace here - let extractors handle it once
    return text


def cyrillic_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are Cyrillic."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if "\u0400" <= c <= "\u04FF") / len(alpha)


def latin_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are Latin."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isascii()) / len(alpha)


def has_albanian_markers(text: str) -> bool:
    """Quick check for Albanian-specific characters (ë, ç)."""
    return bool(ALBANIAN_SPECIFIC_CHARS.intersection(text))


def is_noise_line(line: str) -> bool:
    """Return True if a line is a page number or header/footer."""
    line = line.strip()
    if not line:
        return True
    # Long content lines are never just headers/footers
    if len(line) > 150:
        return False
    if PAGE_NUMBER.match(line):
        return True
    if HEADER_FOOTER.search(line):
        return True
    if DATE_HEADER.search(line):
        return True
    return False


def strip_headers_footers(text: str) -> str:
    """Remove gazette header/footer lines from extracted text."""
    lines = text.split("\n")
    cleaned = [l for l in lines if not is_noise_line(l)]
    return "\n".join(cleaned)


def word_count(text: str) -> int:
    return len(text.split())


def digit_fraction(text: str) -> float:
    if not text:
        return 0.0
    return sum(c.isdigit() for c in text) / len(text)


def number_word_fraction(text: str) -> float:
    words = text.split()
    if not words:
        return 1.0
    return sum(1 for w in words if re.fullmatch(r"[\d.,/%]+", w)) / len(words)
