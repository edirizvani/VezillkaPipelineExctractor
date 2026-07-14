#!/usr/bin/env python3
"""
pipeline.py — Master orchestrator for the Vezilka corpus pipeline.

Run the full pipeline::

    python pipeline.py --phase all --year-start 2019 --year-end 2025

Run specific phases::

    python pipeline.py --phase scrape --year-start 2023 --year-end 2025
    python pipeline.py --phase extract --input data/pdfs/2023/
    python pipeline.py --phase align   --input data/extracted/
    python pipeline.py --phase export  --input data/aligned/ --format tsv,jsonl,huggingface

Test on a single PDF::

    python pipeline.py --phase test --pdf path/to/file.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

logger = logging.getLogger("vezilka")


# ───────────────────────── Phase implementations ────────────────

def phase_scrape(args: argparse.Namespace) -> None:
    """Scrape the Služben Vesnik website for issue metadata."""
    from scraper.catalog_scraper import CatalogScraper

    scraper = CatalogScraper(
        year_start=args.year_start,
        year_end=args.year_end,
    )
    catalog = scraper.scrape()
    scraper.save_catalog(catalog)
    print(f"✓ Catalog: {len(catalog)} issues.")


def phase_download(args: argparse.Namespace) -> None:
    """Download PDFs listed in the catalog."""
    from scraper.pdf_downloader import PDFDownloader

    dl = PDFDownloader()
    count = dl.download_all(limit=getattr(args, "limit", None))
    print(f"✓ Downloaded {count} PDFs.")


def phase_extract(args: argparse.Namespace) -> None:
    """Extract bilingual text from downloaded PDFs."""
    from extractor.layout_analyzer import LayoutAnalyzer

    analyzer = LayoutAnalyzer()
    input_dir = Path(args.input) if args.input else config.PDF_DIR

    pdf_files = sorted(input_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {input_dir}")
        return

    bilingual_count = 0
    mk_only_count = 0
    error_count = 0

    print(f"Extracting text from {len(pdf_files)} PDFs…")
    for i, pdf_path in enumerate(pdf_files, 1):
        out_path = config.EXTRACTED_DIR / f"{pdf_path.stem}.json"
        if out_path.exists():
            logger.debug("Skipping already-extracted: %s", pdf_path.name)
            continue

        try:
            doc = analyzer.analyze(pdf_path)

            result = {
                "pdf": str(pdf_path),
                "macedonian": doc.macedonian_full,
                "albanian": doc.albanian_full,
                "total_pages": doc.total_pages,
                "mk_pages": doc.mk_pages,
                "sq_pages": doc.sq_pages,
                "bilingual": doc.bilingual,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            if doc.bilingual:
                bilingual_count += 1
                logger.info(
                    "[%d/%d] ✓ %s — %d MK pages, %d SQ pages (bilingual)",
                    i, len(pdf_files), pdf_path.name,
                    doc.mk_pages, doc.sq_pages,
                )
            else:
                mk_only_count += 1
                logger.info(
                    "[%d/%d] ⊘ %s — MK-only (%d pages, %d SQ pages)",
                    i, len(pdf_files), pdf_path.name,
                    doc.mk_pages, doc.sq_pages,
                )
        except Exception as exc:
            error_count += 1
            logger.error("[%d/%d] ✗ %s: %s", i, len(pdf_files), pdf_path.name, exc)

    print(f"\n✓ Extraction complete.")
    print(f"  Bilingual:  {bilingual_count}")
    print(f"  MK-only:    {mk_only_count} (skipped for alignment)")
    print(f"  Errors:     {error_count}")
    print(f"  Output in:  {config.EXTRACTED_DIR}")


def phase_align(args: argparse.Namespace) -> None:
    """Align extracted bilingual text into sentence pairs."""
    from aligner.structural_aligner import StructuralAligner
    from aligner.statistical_aligner import GaleChurchAligner
    from cleaner.text_cleaner import TextCleaner

    struct_aligner = StructuralAligner()
    gc_aligner = GaleChurchAligner()
    cleaner = TextCleaner()

    input_dir = Path(args.input) if args.input else config.EXTRACTED_DIR
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"No extracted JSON files in {input_dir}")
        return

    all_pairs: list[dict] = []
    skipped_mk_only = 0

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Skip documents flagged as MK-only
        if data.get("bilingual") is False:
            skipped_mk_only += 1
            logger.debug("Skipping MK-only: %s", jf.name)
            continue

        mk_text = cleaner.clean_macedonian(data.get("macedonian", ""))
        sq_text = cleaner.clean_albanian(data.get("albanian", ""))

        if not mk_text.strip() or not sq_text.strip():
            logger.warning("No bilingual content in %s", jf.name)
            continue

        source_id = jf.stem

        # 1) Structural alignment (primary)
        article_pairs = struct_aligner.align(mk_text, sq_text)
        sentence_pairs = struct_aligner.align_sentences(article_pairs)

        for sp in sentence_pairs:
            result = cleaner.filter_pair(sp.mk, sp.sq)
            if result.is_valid:
                all_pairs.append({
                    "mk": sp.mk,
                    "sq": sp.sq,
                    "meta": {
                        "source": f"slvesnik_{source_id}",
                        "article": sp.article_id,
                        "confidence": sp.confidence,
                        "method": sp.alignment_method,
                    },
                })

        # 2) Gale-Church fallback for unstructured parts
        #    (text that didn't get article-level matches)
        if not article_pairs:
            import re
            mk_sents = re.split(r"(?<=[.!?])\s+", mk_text)
            sq_sents = re.split(r"(?<=[.!?])\s+", sq_text)
            mk_sents = [s.strip() for s in mk_sents if len(s.strip()) > 20]
            sq_sents = [s.strip() for s in sq_sents if len(s.strip()) > 20]

            gc_pairs = gc_aligner.align(mk_sents, sq_sents)
            for gp in gc_pairs:
                result = cleaner.filter_pair(gp.mk, gp.sq)
                if result.is_valid:
                    all_pairs.append({
                        "mk": gp.mk,
                        "sq": gp.sq,
                        "meta": {
                            "source": f"slvesnik_{source_id}",
                            "article": "gc",
                            "confidence": max(0.0, 1.0 - gp.score / 10.0),
                            "method": f"gale_church_{gp.align_type}",
                        },
                    })

    # Deduplicate
    all_pairs, removed = cleaner.deduplicate(all_pairs)

    # Save
    out_path = config.ALIGNED_DIR / "aligned_pairs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Alignment complete: {len(all_pairs)} pairs ({removed} duplicates removed).")
    print(f"  MK-only skipped: {skipped_mk_only}")
    print(f"  Saved to {out_path}")


def phase_export(args: argparse.Namespace) -> None:
    """Export aligned pairs to TSV / JSONL / HuggingFace."""
    from exporter.dataset_exporter import DatasetExporter

    input_path = Path(args.input) if args.input else config.ALIGNED_DIR / "aligned_pairs.json"
    if not input_path.exists():
        print(f"No aligned pairs file at {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    formats = (args.format or "tsv,jsonl,huggingface").split(",")
    exporter = DatasetExporter()

    if "tsv" in formats:
        exporter.export_tsv(pairs, config.EXPORT_DIR / "vezilka_mk_sq.tsv")
    if "jsonl" in formats:
        exporter.export_jsonl(pairs, config.EXPORT_DIR / "vezilka_mk_sq.jsonl")
    if "huggingface" in formats:
        exporter.export_huggingface(pairs, config.EXPORT_DIR / "huggingface")

    exporter.print_statistics(pairs)
    print(f"\n✓ Export complete.  Output in {config.EXPORT_DIR}")


def phase_test(args: argparse.Namespace) -> None:
    """Test the full pipeline on a single PDF."""
    from extractor.layout_analyzer import LayoutAnalyzer
    from aligner.structural_aligner import StructuralAligner
    from aligner.statistical_aligner import GaleChurchAligner
    from cleaner.text_cleaner import TextCleaner
    from exporter.dataset_exporter import DatasetExporter

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return

    print(f"═══════════════════════════════════════════════")
    print(f"  VEZILKA — Single-PDF Test")
    print(f"  PDF: {pdf_path.name}")
    print(f"═══════════════════════════════════════════════\n")

    # 1. Extract
    print("▶ Phase 1: Extracting text with layout detection…")
    analyzer = LayoutAnalyzer()
    doc = analyzer.analyze(pdf_path)

    print(f"  Total pages: {doc.total_pages}")
    print(f"  MK pages:    {doc.mk_pages}")
    print(f"  SQ pages:    {doc.sq_pages}")
    print(f"  Bilingual:   {doc.bilingual}")
    print(f"  MK chars:    {len(doc.macedonian_full):,}")
    print(f"  SQ chars:    {len(doc.albanian_full):,}")

    if not doc.bilingual:
        print("\n⚠  No bilingual content detected — cannot align.")
        print("   MK preview:", doc.macedonian_full[:200])
        print("   SQ preview:", doc.albanian_full[:200])
        return

    # 2. Clean
    print("\n▶ Phase 2: Cleaning text…")
    cleaner = TextCleaner()
    mk_clean = cleaner.clean_macedonian(doc.macedonian_full)
    sq_clean = cleaner.clean_albanian(doc.albanian_full)
    print(f"  MK cleaned:  {len(mk_clean):,} chars")
    print(f"  SQ cleaned:  {len(sq_clean):,} chars")

    # 3. Align
    print("\n▶ Phase 3: Aligning articles…")
    struct = StructuralAligner()
    article_pairs = struct.align(mk_clean, sq_clean)
    print(f"  Article pairs: {len(article_pairs)}")

    sentence_pairs = struct.align_sentences(article_pairs)
    print(f"  Sentence pairs (structural): {len(sentence_pairs)}")

    # Gale-Church fallback if no structural pairs
    gc_sentence_pairs = []
    if not sentence_pairs:
        print("  → Falling back to Gale-Church…")
        import re
        mk_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", mk_clean) if len(s.strip()) > 20]
        sq_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", sq_clean) if len(s.strip()) > 20]
        gc = GaleChurchAligner()
        gc_pairs = gc.align(mk_sents, sq_sents)
        gc_sentence_pairs = GaleChurchAligner.to_sentence_pairs(gc_pairs)
        print(f"  Sentence pairs (Gale-Church): {len(gc_sentence_pairs)}")

    # 4. Filter
    print("\n▶ Phase 4: Filtering pairs…")
    all_sp = sentence_pairs + gc_sentence_pairs
    valid_pairs: list[dict] = []
    rejected = 0

    for sp in all_sp:
        result = cleaner.filter_pair(sp.mk, sp.sq)
        if result.is_valid:
            valid_pairs.append({
                "mk": sp.mk,
                "sq": sp.sq,
                "meta": {
                    "source": f"test_{pdf_path.stem}",
                    "article": sp.article_id,
                    "confidence": sp.confidence,
                    "method": sp.alignment_method,
                },
            })
        else:
            rejected += 1

    print(f"  Valid pairs:    {len(valid_pairs)}")
    print(f"  Rejected:       {rejected}")

    # 5. Show sample
    print(f"\n{'─' * 80}")
    print(f"  First {min(10, len(valid_pairs))} sentence pairs:")
    print(f"{'─' * 80}")
    for i, p in enumerate(valid_pairs[:10], 1):
        print(f"\n  [{i}] MK: {p['mk'][:120]}")
        print(f"      SQ: {p['sq'][:120]}")
        print(f"      (article={p['meta']['article']}, "
              f"conf={p['meta']['confidence']:.2f}, "
              f"method={p['meta']['method']})")

    # 6. Stats
    print(f"\n{'─' * 80}")
    DatasetExporter.print_statistics(valid_pairs)

    # Save test output
    out = config.EXPORT_DIR / f"test_{pdf_path.stem}.tsv"
    DatasetExporter.export_tsv(valid_pairs, out)
    print(f"\n✓ Test TSV saved to {out}")


# ───────────────────────── CLI entry point ──────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vezilka Corpus Pipeline — MK ↔ SQ parallel corpus builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --phase all --year-start 2019 --year-end 2025
  python pipeline.py --phase scrape --year-start 2023 --year-end 2025
  python pipeline.py --phase extract --input data/pdfs/2023/
  python pipeline.py --phase align --input data/extracted/
  python pipeline.py --phase export --format tsv,jsonl,huggingface
  python pipeline.py --phase test --pdf path/to/file.pdf
        """,
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["all", "scrape", "download", "extract", "align", "export", "test"],
        help="Which pipeline phase to run.",
    )
    parser.add_argument("--year-start", type=int, default=config.MIN_YEAR)
    parser.add_argument("--year-end", type=int, default=2026)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--format", type=str, default="tsv,jsonl,huggingface")
    parser.add_argument("--pdf", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.phase == "all":
        phase_scrape(args)
        phase_download(args)
        phase_extract(args)
        phase_align(args)
        phase_export(args)
    elif args.phase == "scrape":
        phase_scrape(args)
    elif args.phase == "download":
        phase_download(args)
    elif args.phase == "extract":
        phase_extract(args)
    elif args.phase == "align":
        phase_align(args)
    elif args.phase == "export":
        phase_export(args)
    elif args.phase == "test":
        if not args.pdf:
            parser.error("--pdf is required for --phase test")
        phase_test(args)


if __name__ == "__main__":
    main()
