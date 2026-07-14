#!/usr/bin/env python3
"""
Validate quality of the MK↔SQ sentence pairs produced by build_dataset.py.
Safe to run while the builder is still writing — reads a snapshot of the TSV.

Checks:
  1. Basic stats (total pairs, unique sources, methods)
  2. Cyrillic leak in SQ column (should be 0%)
  3. Latin leak in MK column (should be ~0%)
  4. English contamination in SQ column via lingua
  5. Sentence length ratio outliers (MK vs SQ)
  6. Empty / near-empty cells
  7. Duplicate pairs
  8. Confidence distribution
  9. Random sample of 20 pairs for manual inspection
"""

import csv, re, sys, os, statistics, random
from collections import Counter

TSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "export", "vezilka_mk_sq.tsv",
)

# ── helpers ──────────────────────────────────────────────────────────────────
CYRILLIC = re.compile(r'[\u0400-\u04FF]')
LATIN    = re.compile(r'[A-Za-zÀ-ÖØ-öø-ÿĀ-žḀ-ỿ]')

def cyrillic_ratio(text):
    letters = CYRILLIC.findall(text) 
    total   = len(CYRILLIC.findall(text)) + len(LATIN.findall(text))
    return len(letters) / total if total else 0.0

def latin_ratio(text):
    letters = LATIN.findall(text)
    total   = len(CYRILLIC.findall(text)) + len(letters)
    return len(letters) / total if total else 0.0

def len_ratio(mk, sq):
    a, b = len(mk), len(sq)
    if b == 0: return 999.0
    return a / b

# ── load ─────────────────────────────────────────────────────────────────────
print(f"Reading {TSV} ...")
rows = []
with open(TSV, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        rows.append(r)

N = len(rows)
print(f"\n{'='*60}")
print(f"  TOTAL PAIRS: {N:,}")
print(f"{'='*60}\n")

if N == 0:
    print("No data yet — check back later.")
    sys.exit(0)

# ── 1. Basic stats ──────────────────────────────────────────────────────────
sources = set(r["source"] for r in rows)
methods = Counter(r.get("method", "?") for r in rows)
articles = set(r.get("article_id", "") for r in rows) - {""}

print(f"📁 Unique source PDFs : {len(sources)}")
print(f"📰 Unique article IDs : {len(articles)}")
print(f"🔧 Alignment methods  : {dict(methods)}")
print()

# ── 2. Confidence distribution ──────────────────────────────────────────────
confs = []
for r in rows:
    try:
        confs.append(float(r.get("confidence", 0)))
    except (ValueError, TypeError):
        pass

if confs:
    print(f"📊 Confidence distribution:")
    print(f"   mean   = {statistics.mean(confs):.3f}")
    print(f"   median = {statistics.median(confs):.3f}")
    print(f"   min    = {min(confs):.3f}")
    print(f"   max    = {max(confs):.3f}")
    print(f"   stdev  = {statistics.stdev(confs):.3f}" if len(confs) > 1 else "")
    below_50 = sum(1 for c in confs if c < 0.50)
    print(f"   pairs with confidence < 0.50: {below_50} ({100*below_50/N:.1f}%)")
    print()

# ── 3. Cyrillic in SQ (CRITICAL — should be 0%) ────────────────────────────
cyr_in_sq = 0
cyr_in_sq_examples = []
for r in rows:
    ratio = cyrillic_ratio(r["sq"])
    if ratio > 0.05:
        cyr_in_sq += 1
        if len(cyr_in_sq_examples) < 3:
            cyr_in_sq_examples.append((r["source"], r["sq"][:100], f"{ratio:.0%}"))

pct = 100 * cyr_in_sq / N
flag = "✅" if pct < 1 else "❌"
print(f"{flag} Cyrillic leak in SQ column: {cyr_in_sq}/{N} ({pct:.1f}%)")
for src, txt, ratio in cyr_in_sq_examples:
    print(f"   Example ({ratio}): [{src}] {txt}")
print()

# ── 4. Latin in MK (should be low) ─────────────────────────────────────────
lat_in_mk = 0
lat_in_mk_examples = []
for r in rows:
    ratio = latin_ratio(r["mk"])
    if ratio > 0.50:
        lat_in_mk += 1
        if len(lat_in_mk_examples) < 3:
            lat_in_mk_examples.append((r["source"], r["mk"][:100], f"{ratio:.0%}"))

pct = 100 * lat_in_mk / N
flag = "✅" if pct < 5 else "⚠️"
print(f"{flag} High-Latin in MK column (>50% Latin): {lat_in_mk}/{N} ({pct:.1f}%)")
for src, txt, ratio in lat_in_mk_examples:
    print(f"   Example ({ratio}): [{src}] {txt}")
print()

# ── 5. Empty / near-empty cells ────────────────────────────────────────────
empty_mk = sum(1 for r in rows if len(r["mk"].strip()) < 5)
empty_sq = sum(1 for r in rows if len(r["sq"].strip()) < 5)
flag_mk = "✅" if empty_mk == 0 else "⚠️"
flag_sq = "✅" if empty_sq == 0 else "⚠️"
print(f"{flag_mk} Near-empty MK (<5 chars): {empty_mk}/{N} ({100*empty_mk/N:.1f}%)")
print(f"{flag_sq} Near-empty SQ (<5 chars): {empty_sq}/{N} ({100*empty_sq/N:.1f}%)")
print()

# ── 6. Length ratio outliers ────────────────────────────────────────────────
ratios = [len_ratio(r["mk"], r["sq"]) for r in rows]
extreme = sum(1 for ratio in ratios if ratio > 5.0 or ratio < 0.2)
pct = 100 * extreme / N
flag = "✅" if pct < 5 else "⚠️"
print(f"{flag} Extreme length-ratio outliers (>5x or <0.2x): {extreme}/{N} ({pct:.1f}%)")

if ratios:
    valid_ratios = [r for r in ratios if r < 900]
    if valid_ratios:
        print(f"   Mean MK/SQ length ratio: {statistics.mean(valid_ratios):.2f}")
        print(f"   Median: {statistics.median(valid_ratios):.2f}")
print()

# ── 7. Exact duplicates ────────────────────────────────────────────────────
seen = set()
dupes = 0
for r in rows:
    key = (r["mk"].strip(), r["sq"].strip())
    if key in seen:
        dupes += 1
    seen.add(key)
flag = "✅" if dupes == 0 else "⚠️"
print(f"{flag} Exact duplicate pairs: {dupes}/{N} ({100*dupes/N:.1f}%)")
print()

# ── 8. Lingua validation on SQ (English contamination) ─────────────────────
try:
    from lingua import Language, LanguageDetectorBuilder
    print("🔍 Running lingua validation on SQ column (sampling up to 500 pairs)...")
    detector = LanguageDetectorBuilder.from_languages(
        Language.ALBANIAN, Language.MACEDONIAN, Language.ENGLISH,
        Language.SERBIAN, Language.TURKISH, Language.FRENCH,
        Language.BOSNIAN, Language.CROATIAN
    ).with_minimum_relative_distance(0.05).build()
    
    sample = random.sample(rows, min(500, N))
    eng_count = 0
    non_sq_count = 0
    eng_examples = []
    for r in sample:
        sq = r["sq"].strip()
        if len(sq) < 20:
            continue
        lang = detector.detect_language_of(sq)
        if lang == Language.ENGLISH:
            eng_count += 1
            if len(eng_examples) < 3:
                eng_examples.append(sq[:120])
        elif lang != Language.ALBANIAN:
            non_sq_count += 1
    
    tested = len([r for r in sample if len(r["sq"].strip()) >= 20])
    pct_eng = 100 * eng_count / tested if tested else 0
    pct_non = 100 * non_sq_count / tested if tested else 0
    flag = "✅" if pct_eng < 2 else "❌"
    print(f"{flag} English detected in SQ: {eng_count}/{tested} sampled ({pct_eng:.1f}%)")
    for ex in eng_examples:
        print(f"   Example: {ex}")
    flag2 = "✅" if pct_non < 10 else "⚠️"
    print(f"{flag2} Non-Albanian (non-English) in SQ: {non_sq_count}/{tested} ({pct_non:.1f}%)")
    print()
    
    # Also validate MK column
    print("🔍 Running lingua validation on MK column (sampling up to 500 pairs)...")
    mk_non_mk = 0
    for r in sample:
        mk = r["mk"].strip()
        if len(mk) < 20:
            continue
        lang = detector.detect_language_of(mk)
        if lang not in (Language.MACEDONIAN, Language.SERBIAN, Language.BOSNIAN):
            mk_non_mk += 1
    mk_tested = len([r for r in sample if len(r["mk"].strip()) >= 20])
    pct_mk = 100 * mk_non_mk / mk_tested if mk_tested else 0
    flag = "✅" if pct_mk < 5 else "⚠️"
    print(f"{flag} Non-Macedonian in MK: {mk_non_mk}/{mk_tested} sampled ({pct_mk:.1f}%)")
    print()

except ImportError:
    print("⚠️  lingua not available — skipping language detection checks")
    print()

# ── 9. Random sample for manual inspection ──────────────────────────────────
print(f"{'='*60}")
print(f"  📋 RANDOM SAMPLE (20 pairs for manual inspection)")
print(f"{'='*60}")
sample = random.sample(rows, min(20, N))
for i, r in enumerate(sample, 1):
    mk_short = r["mk"][:90].replace("\n", " ")
    sq_short = r["sq"][:90].replace("\n", " ")
    conf = r.get("confidence", "?")
    art  = r.get("article_id", "?")
    print(f"\n── Pair {i} (article={art}, conf={conf}) ──")
    print(f"  MK: {mk_short}")
    print(f"  SQ: {sq_short}")

print(f"\n{'='*60}")
print(f"  VALIDATION COMPLETE")
print(f"{'='*60}")
