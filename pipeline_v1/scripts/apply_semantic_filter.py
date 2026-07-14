#!/usr/bin/env python3
"""
apply_semantic_filter.py — Apply LASER + LaBSE semantic filtering to existing pairs.

Usage:
    python apply_semantic_filter.py                          # Use default TSV input
    python apply_semantic_filter.py --threshold 0.75         # Custom threshold
    python apply_semantic_filter.py --input custom.tsv       # Custom input file
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from cleaner.semantic_validator import SemanticValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_pairs_from_tsv(path: Path) -> list[dict]:
    """Load pairs from TSV file."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pairs.append({
                "mk": row.get("mk", ""),
                "sq": row.get("sq", ""),
                "meta": {
                    "source": row.get("source", ""),
                    "article": row.get("article_id", ""),
                    "confidence": float(row.get("confidence", 0)) if row.get("confidence") else 0,
                    "method": row.get("method", ""),
                },
            })
    return pairs


def load_pairs_from_json(path: Path) -> list[dict]:
    """Load pairs from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Apply semantic filtering to aligned pairs"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=config.EXPORT_DIR / "vezilka_mk_sq.tsv",
        help="Input TSV or JSON file with pairs",
    )
    parser.add_argument(
        "--output", "-o", 
        type=Path,
        default=None,
        help="Output file (default: adds '_semantic' suffix)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.70,
        help="Minimum similarity threshold (default: 0.70)",
    )
    parser.add_argument(
        "--no-laser",
        action="store_true",
        help="Disable LASER (use only LaBSE)",
    )
    parser.add_argument(
        "--no-labse",
        action="store_true",
        help="Disable LaBSE (use only LASER)",
    )
    parser.add_argument(
        "--export-json",
        action="store_true", 
        help="Also export filtered pairs to JSON",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Process only N pairs (for testing)",
    )
    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Default output path
    if args.output is None:
        args.output = args.input.parent / f"{args.input.stem}_semantic.tsv"

    # Load pairs based on file extension
    logger.info(f"Loading pairs from {args.input}...")
    if args.input.suffix == ".json":
        pairs = load_pairs_from_json(args.input)
    else:
        pairs = load_pairs_from_tsv(args.input)
    
    original_count = len(pairs)
    logger.info(f"  Loaded {original_count:,} pairs")
    
    # Sample for testing
    if args.sample:
        pairs = pairs[:args.sample]
        logger.info(f"  Using sample of {len(pairs):,} pairs")

    # Initialize validator
    logger.info("Initializing semantic validator...")
    logger.info(f"  LASER: {'enabled' if not args.no_laser else 'disabled'}")
    logger.info(f"  LaBSE: {'enabled' if not args.no_labse else 'disabled'}")
    logger.info(f"  Threshold: {args.threshold}")
    
    validator = SemanticValidator(
        min_similarity=args.threshold,
        use_laser=not args.no_laser,
        use_labse=not args.no_labse,
    )

    # Filter pairs
    def progress(current, total):
        logger.info(f"  Progress: {current:,}/{total:,} ({100*current/total:.1f}%)")

    filtered, rejected = validator.filter_pairs(
        pairs, 
        progress_callback=progress,
    )

    # Save filtered pairs to TSV
    logger.info(f"Saving {len(filtered):,} pairs to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("mk\tsq\tlaser_sim\tlabse_sim\tcombined_sim\tsource\tarticle_id\tconfidence\tmethod\n")
        for p in filtered:
            sem = p.get("meta", {}).get("semantic_score", {})
            meta = p.get("meta", {})
            laser = sem.get("laser", "")
            labse = sem.get("labse", "")
            combined = sem.get("combined", "")
            mk = p["mk"].replace("\t", " ").replace("\n", " ")
            sq = p["sq"].replace("\t", " ").replace("\n", " ")
            source = meta.get("source", "")
            article = meta.get("article", "")
            conf = meta.get("confidence", "")
            method = meta.get("method", "")
            f.write(f"{mk}\t{sq}\t{laser}\t{labse}\t{combined}\t{source}\t{article}\t{conf}\t{method}\n")

    # Optional JSON export
    if args.export_json:
        json_path = args.output.with_suffix(".json")
        logger.info(f"Exporting to JSON: {json_path}")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)

    # Print summary
    total = args.sample if args.sample else original_count
    print("\n" + "═" * 60)
    print("  SEMANTIC FILTERING COMPLETE")
    print("═" * 60)
    print(f"  Input:      {total:,} pairs")
    print(f"  Accepted:   {len(filtered):,} pairs ({100*len(filtered)/total:.1f}%)")
    print(f"  Rejected:   {rejected:,} pairs ({100*rejected/total:.1f}%)")
    print(f"  Threshold:  {args.threshold}")
    print(f"  Output:     {args.output}")
    if args.export_json:
        print(f"  JSON:       {json_path}")
    print("═" * 60)


if __name__ == "__main__":
    main()
