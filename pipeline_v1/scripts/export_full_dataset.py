#!/usr/bin/env python3
"""
export_full_dataset.py — Export the complete dataset without train/val/test splits.

Creates multiple output formats:
- TSV (tab-separated)
- CSV (comma-separated)
- JSONL (JSON Lines)
- JSON (full array)

Usage:
    python export_full_dataset.py                           # Use default input
    python export_full_dataset.py --input custom.tsv        # Custom input
    python export_full_dataset.py --name my_corpus          # Custom output name
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
            pair = {
                "mk": row.get("mk", ""),
                "sq": row.get("sq", ""),
            }
            # Add optional metadata if present
            if "source" in row:
                pair["source"] = row["source"]
            if "article_id" in row:
                pair["article_id"] = row["article_id"]
            if "confidence" in row and row["confidence"]:
                try:
                    pair["confidence"] = float(row["confidence"])
                except ValueError:
                    pass
            if "method" in row:
                pair["method"] = row["method"]
            # Add semantic scores if present
            if "laser_sim" in row and row["laser_sim"]:
                try:
                    pair["laser_sim"] = float(row["laser_sim"])
                except ValueError:
                    pass
            if "labse_sim" in row and row["labse_sim"]:
                try:
                    pair["labse_sim"] = float(row["labse_sim"])
                except ValueError:
                    pass
            if "combined_sim" in row and row["combined_sim"]:
                try:
                    pair["combined_sim"] = float(row["combined_sim"])
                except ValueError:
                    pass
            pairs.append(pair)
    return pairs


def export_tsv(pairs: list[dict], path: Path) -> None:
    """Export pairs to TSV format."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        # Determine all keys from first pair
        if pairs:
            fieldnames = list(pairs[0].keys())
        else:
            fieldnames = ["mk", "sq"]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for pair in pairs:
            # Escape tabs and newlines in text
            row = {}
            for k, v in pair.items():
                if isinstance(v, str):
                    row[k] = v.replace("\t", " ").replace("\n", " ")
                else:
                    row[k] = v
            writer.writerow(row)


def export_csv(pairs: list[dict], path: Path) -> None:
    """Export pairs to CSV format."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        if pairs:
            fieldnames = list(pairs[0].keys())
        else:
            fieldnames = ["mk", "sq"]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for pair in pairs:
            row = {}
            for k, v in pair.items():
                if isinstance(v, str):
                    row[k] = v.replace("\n", " ")
                else:
                    row[k] = v
            writer.writerow(row)


def export_jsonl(pairs: list[dict], path: Path) -> None:
    """Export pairs to JSONL format (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def export_json(pairs: list[dict], path: Path) -> None:
    """Export pairs to JSON format (full array)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Export complete dataset without train/val/test splits"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=config.EXPORT_DIR / "vezilka_mk_sq.tsv",
        help="Input TSV file with pairs",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=config.EXPORT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default="vezilka_mk_sq_full",
        help="Base name for output files",
    )
    parser.add_argument(
        "--formats", "-f",
        type=str,
        default="tsv,csv,jsonl,json",
        help="Comma-separated list of formats (tsv,csv,jsonl,json)",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Export only mk and sq columns (no metadata)",
    )
    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load pairs
    logger.info(f"Loading pairs from {args.input}...")
    pairs = load_pairs_from_tsv(args.input)
    logger.info(f"  Loaded {len(pairs):,} pairs")

    # Simplify if requested
    if args.simple:
        pairs = [{"mk": p["mk"], "sq": p["sq"]} for p in pairs]
        logger.info("  Simplified to mk/sq columns only")

    # Export to requested formats
    formats = [f.strip().lower() for f in args.formats.split(",")]
    
    for fmt in formats:
        output_path = args.output_dir / f"{args.name}.{fmt}"
        
        if fmt == "tsv":
            logger.info(f"Exporting TSV to {output_path}...")
            export_tsv(pairs, output_path)
        elif fmt == "csv":
            logger.info(f"Exporting CSV to {output_path}...")
            export_csv(pairs, output_path)
        elif fmt == "jsonl":
            logger.info(f"Exporting JSONL to {output_path}...")
            export_jsonl(pairs, output_path)
        elif fmt == "json":
            logger.info(f"Exporting JSON to {output_path}...")
            export_json(pairs, output_path)
        else:
            logger.warning(f"Unknown format: {fmt}")
            continue
        
        # Get file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"  → {output_path.name} ({size_mb:.2f} MB)")

    # Print summary
    print("\n" + "═" * 60)
    print("  DATASET EXPORT COMPLETE")
    print("═" * 60)
    print(f"  Total pairs:    {len(pairs):,}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Files created:")
    for fmt in formats:
        output_path = args.output_dir / f"{args.name}.{fmt}"
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"    • {output_path.name} ({size_mb:.2f} MB)")
    print("═" * 60)


if __name__ == "__main__":
    main()
