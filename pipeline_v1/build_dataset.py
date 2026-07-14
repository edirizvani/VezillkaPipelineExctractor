#!/usr/bin/env python3
"""
build_dataset.py — Build a MK↔SQ parallel sentence dataset from gazette PDFs.

**Version 2** — uses lingua-language-detector for validation + Gale-Church alignment.

Produces a pandas-friendly TSV file with columns:
    mk              — Macedonian sentence
    sq              — Albanian sentence
    source          — PDF filename (stem)
    article_id      — Matched article number (Член/Neni) or "gc" for Gale-Church
    confidence      — Alignment confidence (0–1)
    method          — Alignment method used

Quality pipeline:
  1. Page-level:   Script-ratio detection (Cyrillic vs Latin)
  2. Page-level:   Lingua verification — reject Latin pages that are English/French/etc.
  3. Paragraph:    Lingua filter — remove non-Albanian paragraphs from SQ text
  4. Article:      StructuralAligner — match articles by number (Член N ↔ Neni N)
  5. Sentence:     GaleChurchAligner — length-based dynamic-programming alignment
  6. Pair-level:   Lingua validation — every output pair is verified

Usage:
    python build_dataset.py                          # process all PDFs
    python build_dataset.py --limit 10               # first 10 PDFs only
    python build_dataset.py --pdf path/to/file.pdf   # single PDF test
    python build_dataset.py --resume                 # resume interrupted batch
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

logger = logging.getLogger("build_dataset")

# ─── Lazy-loaded lingua detector ───────────────────────────────
_DETECTOR = None


def get_detector():
    """Build the lingua detector once (covers the languages we expect in the gazette)."""
    global _DETECTOR
    if _DETECTOR is None:
        from lingua import Language, LanguageDetectorBuilder
        _DETECTOR = (
            LanguageDetectorBuilder
            .from_languages(
                Language.MACEDONIAN,
                Language.ALBANIAN,
                Language.ENGLISH,
                Language.SERBIAN,
                Language.TURKISH,
                Language.FRENCH,
                Language.BOSNIAN,
                Language.CROATIAN,
            )
            .with_minimum_relative_distance(0.05)
            .build()
        )
    return _DETECTOR


# ─── Sentence splitting ────────────────────────────────────────

# Split on sentence-ending punctuation followed by space + uppercase letter
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZА-ЯЀ-Ɀ])")


def split_sentences(text: str, min_chars: int = 15) -> list[str]:
    """Split text into sentences, merging very short fragments."""
    raw = _SENT_RE.split(text)
    result = []
    for sent in raw:
        for subsent in sent.split("\n\n"):
            cleaned = " ".join(subsent.split())  # normalise whitespace
            if len(cleaned) >= min_chars:
                result.append(cleaned)
    return result


# ─── Page header / footer noise removal ────────────────────────

_NOISE_RE = [
    re.compile(r"СЛУЖБЕН\s+ВЕСНИК\s+НА", re.IGNORECASE),
    re.compile(r"РЕПУБЛИКА\s+(СЕВЕРНА\s+)?МАКЕДОНИЈА", re.IGNORECASE),
    re.compile(r"Gazeta\s+Zyrtare", re.IGNORECASE),
    re.compile(r"Gaceta\s+Zyrtare", re.IGNORECASE),
    re.compile(r"^Стр\.\s*\d+\s*-\s*Бр\.\s*\d+", re.IGNORECASE),
    re.compile(r"^Бр\.\s*\d+\s*-\s*Стр\.\s*\d+", re.IGNORECASE),
    re.compile(
        r"^\d{1,2}\s+(јануари|февруари|март|април|мај|јуни|јули|август|"
        r"септември|октомври|ноември|декември)\s+\d{4}",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\d{1,2}\s+(janar|shkurt|mars|prill|maj|qershor|korrik|gusht|"
        r"shtator|tetor|nëntor|dhjetor)\s+\d{4}",
        re.IGNORECASE,
    ),
    re.compile(r"^www\.slvesnik\.com\.mk", re.IGNORECASE),
    re.compile(r"^contact@slvesnik\.com\.mk", re.IGNORECASE),
    re.compile(r"^\d{1,4}$"),  # lone page numbers
]


def clean_text(text: str) -> str:
    """Remove page headers/footers, normalise whitespace."""
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if any(pat.search(stripped) for pat in _NOISE_RE):
            continue
        if stripped:
            lines.append(stripped)
    result = "\n".join(lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


# ─── Paragraph-level lingua filtering ──────────────────────────

def filter_paragraphs_lingua(text: str, expected_lang_name: str) -> str:
    """
    Split *text* into paragraphs and keep only those in the expected language.

    *expected_lang_name* must be ``"MACEDONIAN"`` or ``"ALBANIAN"``.
    Short paragraphs (< 40 chars) are kept unconditionally.
    """
    from lingua import Language
    expected = getattr(Language, expected_lang_name)
    detector = get_detector()

    # Also accept Serbian for MK (very similar languages)
    accept = {expected}
    if expected == Language.MACEDONIAN:
        accept.add(Language.SERBIAN)

    paragraphs = text.split("\n\n")
    kept = []
    dropped = 0

    for para in paragraphs:
        para_stripped = para.strip()
        if len(para_stripped) < 40:
            kept.append(para)
            continue
        detected = detector.detect_language_of(para_stripped)
        if detected is None or detected in accept:
            kept.append(para)
        else:
            dropped += 1
            logger.debug(
                "Dropped paragraph (lingua=%s): %.60s…",
                detected.name if detected else "?", para_stripped,
            )

    if dropped:
        logger.info(
            "Paragraph filter (%s): kept %d, dropped %d",
            expected_lang_name, len(kept), dropped,
        )
    return "\n\n".join(kept)


# ─── Pair-level lingua validation ──────────────────────────────

def validate_pair_lingua(mk_text: str, sq_text: str) -> bool:
    """
    Return True only if lingua confirms MK side is Macedonian (or Serbian)
    and SQ side is Albanian.

    Short texts (< 25 chars) pass automatically (lingua is unreliable on
    very short strings).
    """
    from lingua import Language
    detector = get_detector()

    # Validate MK side
    if len(mk_text) >= 25:
        mk_lang = detector.detect_language_of(mk_text)
        if mk_lang not in (Language.MACEDONIAN, Language.SERBIAN, None):
            return False

    # Validate SQ side
    if len(sq_text) >= 25:
        sq_lang = detector.detect_language_of(sq_text)
        if sq_lang not in (Language.ALBANIAN, None):
            return False

    return True


# ─── Core processing ───────────────────────────────────────────

def process_single_pdf(pdf_path: Path) -> list[dict]:
    """
    Process one PDF and return validated, aligned sentence pairs.

    Pipeline:
      1. LayoutAnalyzer extracts text, classifies pages, lingua-verifies Latin pages
      2. Clean text + filter paragraphs with lingua
      3. StructuralAligner matches articles (Член N ↔ Neni N)
      4. GaleChurchAligner aligns sentences within each article pair
      5. Lingua validates every output pair
    """
    from extractor.layout_analyzer import LayoutAnalyzer
    from aligner.structural_aligner import StructuralAligner
    from aligner.statistical_aligner import GaleChurchAligner

    # ── 1. Extract & classify pages ─────────────────────────────
    analyzer = LayoutAnalyzer(use_lingua=True)
    try:
        doc = analyzer.analyze(pdf_path)
    except Exception as e:
        logger.error("Failed to extract %s: %s", pdf_path.name, e)
        return []

    if not doc.bilingual:
        logger.info(
            "Skipping non-bilingual %s  (MK=%d chars, SQ=%d chars, "
            "sq_pages=%d, other_pages=%d)",
            pdf_path.name, len(doc.macedonian_full),
            len(doc.albanian_full), doc.sq_pages, doc.other_pages,
        )
        return []

    # ── 2. Clean & paragraph-level lingua filter ────────────────
    mk_text = clean_text(doc.macedonian_full)
    sq_text = clean_text(doc.albanian_full)

    mk_text = filter_paragraphs_lingua(mk_text, "MACEDONIAN")
    sq_text = filter_paragraphs_lingua(sq_text, "ALBANIAN")

    if len(mk_text) < 100 or len(sq_text) < 100:
        logger.warning("Not enough text after filtering: %s", pdf_path.name)
        return []

    # ── 3. Article matching ─────────────────────────────────────
    struct_aligner = StructuralAligner()
    mk_segments = struct_aligner.extract_article_segments(mk_text, "mk")
    sq_segments = struct_aligner.extract_article_segments(sq_text, "sq")
    article_pairs = struct_aligner.align_by_articles(mk_segments, sq_segments)

    logger.info(
        "PDF %s: %d MK articles, %d SQ articles → %d matched pairs",
        pdf_path.name, len(mk_segments), len(sq_segments), len(article_pairs),
    )

    # ── 4. Sentence alignment with Gale-Church ──────────────────
    gc = GaleChurchAligner(mean_char_ratio=1.1, variance=6.8)
    all_pairs: list[dict] = []
    gc_rejected = 0

    for pair in article_pairs:
        mk_sents = split_sentences(pair.mk)
        sq_sents = split_sentences(pair.sq)

        if not mk_sents or not sq_sents:
            continue

        gc_pairs = gc.align(mk_sents, sq_sents)

        for gp in gc_pairs:
            confidence = max(0.0, min(1.0, 1.0 - gp.score / 10.0))
            all_pairs.append({
                "mk": gp.mk,
                "sq": gp.sq,
                "source": pdf_path.stem,
                "article_id": pair.article_id,
                "confidence": round(confidence, 3),
                "method": f"structural+gc_{gp.align_type}",
            })

    # ── Also try Gale-Church on article "0" (preamble) if present ──
    # Article "0" means no Член/Neni markers — align the whole block
    # but only if BOTH sides had no articles (to avoid garbage pairing)
    mk_has_articles = any(s.article_id != "0" for s in mk_segments)
    sq_has_articles = any(s.article_id != "0" for s in sq_segments)

    if not mk_has_articles and not sq_has_articles:
        # Both sides are pure text without article markers — try Gale-Church
        mk_sents = split_sentences(mk_text)
        sq_sents = split_sentences(sq_text)
        if mk_sents and sq_sents:
            gc_pairs = gc.align(mk_sents, sq_sents)
            for gp in gc_pairs:
                confidence = max(0.0, min(1.0, 1.0 - gp.score / 10.0))
                # Lower confidence for non-article text
                confidence *= 0.7
                all_pairs.append({
                    "mk": gp.mk,
                    "sq": gp.sq,
                    "source": pdf_path.stem,
                    "article_id": "full_gc",
                    "confidence": round(confidence, 3),
                    "method": f"gc_fulltext_{gp.align_type}",
                })

    # ── 5. Lingua validation on every pair ──────────────────────
    validated = []
    lingua_rejected = 0
    for p in all_pairs:
        if validate_pair_lingua(p["mk"], p["sq"]):
            validated.append(p)
        else:
            lingua_rejected += 1

    logger.info(
        "PDF %s: %d raw pairs → %d after lingua validation "
        "(%d rejected by lingua)",
        pdf_path.name, len(all_pairs), len(validated), lingua_rejected,
    )

    # ── 6. Deduplicate ──────────────────────────────────────────
    seen = set()
    unique = []
    for p in validated:
        key = (p["mk"][:120], p["sq"][:120])
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


# ─── TSV I/O ───────────────────────────────────────────────────

_FIELDNAMES = ["mk", "sq", "source", "article_id", "confidence", "method"]


def save_tsv(pairs: list[dict], output_path: Path) -> None:
    """Write pairs to a fresh TSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES, delimiter="\t",
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(pairs)
    print(f"✓ Saved {len(pairs)} pairs → {output_path}")


def append_tsv(pairs: list[dict], output_path: Path) -> None:
    """Append pairs to an existing TSV (create with header if needed)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with open(output_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES, delimiter="\t",
                                quoting=csv.QUOTE_MINIMAL)
        if write_header:
            writer.writeheader()
        writer.writerows(pairs)


def load_processed_sources(output_path: Path) -> set[str]:
    """Load already-processed PDF stems from existing TSV for resume."""
    sources = set()
    if output_path.exists() and output_path.stat().st_size > 0:
        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                src = row.get("source", "")
                if src:
                    sources.add(src)
    return sources


# ─── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build MK↔SQ parallel sentence dataset from gazette PDFs."
    )
    parser.add_argument("--pdf", type=str, default=None,
                        help="Process a single PDF file.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of PDFs to process.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output TSV file path.")
    parser.add_argument("--min-confidence", type=float, default=0.25,
                        help="Minimum confidence threshold (default: 0.25).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume — skip already-processed PDFs.")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing output and start fresh.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging.")
    args = parser.parse_args()

    # ── Logging setup ───────────────────────────────────────────
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("build_dataset", "extractor", "aligner", "cleaner", "vezilka"):
        logging.getLogger(name).setLevel(level)

    output_path = (
        Path(args.output) if args.output
        else config.EXPORT_DIR / "vezilka_mk_sq.tsv"
    )

    # ── Warm up lingua detector ─────────────────────────────────
    print("🔤 Initialising lingua-language-detector…")
    get_detector()
    print("   ✓ lingua ready\n")

    # ── Single-PDF mode ─────────────────────────────────────────
    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"PDF not found: {pdf_path}")
            sys.exit(1)

        pairs = process_single_pdf(pdf_path)
        if pairs:
            if args.min_confidence > 0:
                before = len(pairs)
                pairs = [p for p in pairs if p["confidence"] >= args.min_confidence]
                if before != len(pairs):
                    print(f"   Confidence filter: {before} → {len(pairs)} "
                          f"(threshold={args.min_confidence})")
            save_tsv(pairs, output_path)

            # Quick quality summary
            print(f"\n── Quality summary ──")
            confs = [p["confidence"] for p in pairs]
            methods = {}
            for p in pairs:
                m = p["method"].split("_")[0]
                methods[m] = methods.get(m, 0) + 1
            print(f"   Pairs: {len(pairs)}")
            print(f"   Confidence: min={min(confs):.2f}  "
                  f"mean={sum(confs)/len(confs):.2f}  max={max(confs):.2f}")
            print(f"   Methods: {methods}")
        else:
            print("⚠  No bilingual sentence pairs found.")
        return

    # ── Batch mode ──────────────────────────────────────────────
    pdf_dir = config.PDF_DIR
    pdf_files = sorted(pdf_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {pdf_dir}")
        sys.exit(1)

    if args.limit:
        pdf_files = pdf_files[:args.limit]

    # Handle resume / fresh
    already_processed: set[str] = set()
    if args.fresh and output_path.exists():
        output_path.unlink()
        print("🗑  Deleted existing output for fresh start.")
    elif args.resume or (not args.fresh and output_path.exists()):
        already_processed = load_processed_sources(output_path)
        if already_processed:
            print(f"📂 Resuming: {len(already_processed)} PDFs already processed.")
    else:
        if output_path.exists():
            output_path.unlink()

    total = len(pdf_files)
    remaining = total - len(already_processed)
    print(f"Processing {total} PDFs ({remaining} remaining)…\n")

    stats = {
        "bilingual": 0, "monolingual": 0, "error": 0,
        "skipped": 0, "total_pairs": 0, "lingua_rejected": 0,
    }
    t0 = time.time()

    for i, pdf_path in enumerate(pdf_files, 1):
        if pdf_path.stem in already_processed:
            stats["skipped"] += 1
            continue

        try:
            pairs = process_single_pdf(pdf_path)

            if pairs:
                stats["bilingual"] += 1
                if args.min_confidence > 0:
                    pairs = [p for p in pairs
                             if p["confidence"] >= args.min_confidence]
                if pairs:
                    append_tsv(pairs, output_path)
                    stats["total_pairs"] += len(pairs)
            else:
                stats["monolingual"] += 1

        except Exception as e:
            stats["error"] += 1
            logger.error("[%d/%d] Error: %s — %s", i, total, pdf_path.name, e)

        # Progress report every 25 PDFs
        processed = i - stats["skipped"]
        if processed % 25 == 0 or i == total:
            elapsed = time.time() - t0
            rate = processed / max(elapsed, 1)
            eta = (total - i) / max(rate, 0.01)
            print(
                f"  [{i}/{total}] "
                f"{stats['bilingual']} bilingual, "
                f"{stats['total_pairs']} pairs, "
                f"{rate:.1f} PDF/s, "
                f"ETA {eta/60:.0f}m"
            )

    # ── Final report ────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"  BATCH COMPLETE  ({elapsed/60:.1f} minutes)")
    print(f"{'='*55}")
    print(f"  Total PDFs scanned:   {total}")
    print(f"  Bilingual PDFs:       {stats['bilingual']}")
    print(f"  Monolingual / empty:  {stats['monolingual']}")
    print(f"  Errors:               {stats['error']}")
    print(f"  Skipped (resumed):    {stats['skipped']}")
    print(f"  Total sentence pairs: {stats['total_pairs']}")
    print(f"  Output: {output_path}")
    if stats["total_pairs"] > 0:
        print(f"\n  Load with pandas:")
        print(f"    import pandas as pd")
        print(f"    df = pd.read_csv('{output_path}', sep='\\t')")
        print(f"    df.head()")
    else:
        print("\n⚠  No bilingual sentence pairs found in any PDF.")


if __name__ == "__main__":
    main()
