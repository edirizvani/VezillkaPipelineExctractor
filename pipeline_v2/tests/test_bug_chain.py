"""Confirm the exact bug: normalise_whitespace destroys newlines BEFORE NoiseFilter."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.text_utils import (
    normalise_whitespace, fix_hyphenation, clean_text,
    is_noise_line, strip_headers_footers, HEADER_FOOTER, DATE_HEADER
)
import re

# Simulate what happens in SequentialExtractor._clean()
sample_mk = """Број 6 11 јануари 2021, понеделник година LXXVII
www.slvesnik.com.mk contact@slvesnik.com.mk
СОДРЖИНА
Стр. 69. Решение од Комисија за хартии од вредност
73. Решение за дополнување на Решение
Стр. 2 - Бр. 6 11 јануари 2021
76. Одлука од Совет на јавните обвинители
82. Одлука за изменување и дополнување"""

print("=== ORIGINAL TEXT (with newlines) ===")
print(f"Lines: {len(sample_mk.split(chr(10)))}")
print(f"Has newlines: {chr(10) in sample_mk}")
print()

# Step 1: fix_hyphenation (preserves newlines)
step1 = fix_hyphenation(sample_mk)
print(f"After fix_hyphenation: lines={len(step1.split(chr(10)))}")

# Step 2: normalise_whitespace (THE BUG)
step2 = normalise_whitespace(step1)
print(f"After normalise_whitespace: lines={len(step2.split(chr(10)))}")
print(f"Has newlines: {chr(10) in step2}")
print(f"Text is now ONE LINE: {repr(step2[:120])}...")
print()

# Step 3: clean_text (calls normalise_whitespace again)
step3 = clean_text(step2)
print(f"After clean_text: lines={len(step3.split(chr(10)))}")
print()

# Step 4: NoiseFilter.clean_mk calls strip_headers_footers on this ONE-LINE text
print("=== NOW IN NOISEFILTER ===")
after_strip = strip_headers_footers(step3)
print(f"After strip_headers_footers: '{after_strip[:50]}...' ({len(after_strip)} chars)")
print()

# Demonstrate why: is_noise_line on the giant single line
line = step3
print(f"is_noise_line on the single giant line ({len(line)} chars):")
print(f"  is empty: {not line.strip()}")
page_pat = re.compile(r'^\s*\d{1,4}\s*$')
print(f"  PAGE_NUMBER.match: {bool(page_pat.match(line))}")
print(f"  HEADER_FOOTER.search: {bool(HEADER_FOOTER.search(line))}")
if HEADER_FOOTER.search(line):
    m = HEADER_FOOTER.search(line)
    print(f"    Matched pattern: '{m.group()}' at position {m.start()}")
print(f"  DATE_HEADER.search: {bool(DATE_HEADER.search(line))}")
if DATE_HEADER.search(line):
    m = DATE_HEADER.search(line)
    print(f"    Matched pattern: '{m.group()}' at position {m.start()}")
print(f"  Result: is_noise={is_noise_line(line)}")
print()
print("=== CONCLUSION ===")
print("1. normalise_whitespace() in _clean() replaces ALL \\n with spaces")
print("2. Text becomes ONE GIANT LINE")
print("3. is_noise_line() uses .search() and finds gazette header ANYWHERE in the line")
print("4. The ENTIRE text is classified as noise and removed")
print("5. mk_chars=0, sq_chars=0")
