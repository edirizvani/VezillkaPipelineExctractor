#!/usr/bin/env python3
"""
validate.py — Quality checks and spot-checking for the Vezilka corpus.

Runs after the full pipeline to verify alignment quality:
  1. Spot-check:  sample 50 random pairs and print for human review
  2. Script check: verify MK→Cyrillic, SQ→Latin
  3. Length histogram (saved as PNG)
  4. Duplicate rate
  5. Coverage check: % of PDFs that yielded ≥1 valid pair
"""

from __future__ import annotations

import json
import logging
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger("validate")


def _is_cyrillic(ch: str) -> bool:
    return "\u0400" <= ch <= "\u04FF"


def _is_latin(ch: str) -> bool:
    return ("A" <= ch <= "Z") or ("a" <= ch <= "z") or ch in "ëçËÇ"


def load_pairs(path: Path | None = None) -> list[dict]:
    path = path or (config.ALIGNED_DIR / "aligned_pairs.json")
    if not path.exists():
        print(f"No aligned pairs at {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── 1. Spot-check ──────────────────────────────────────────────

def spot_check(pairs: list[dict], n: int = 50) -> None:
    print(f"\n{'═' * 80}")
    print(f"  SPOT-CHECK: {min(n, len(pairs))} random pairs")
    print(f"{'═' * 80}")

    sample = random.sample(pairs, min(n, len(pairs)))
    for i, p in enumerate(sample, 1):
        meta = p.get("meta", {})
        print(f"\n[{i}] article={meta.get('article','-')}, "
              f"conf={meta.get('confidence','-')}, "
              f"method={meta.get('method','-')}")
        print(f"  MK: {p['mk'][:150]}")
        print(f"  SQ: {p['sq'][:150]}")


# ── 2. Script check ────────────────────────────────────────────

def script_check(pairs: list[dict]) -> None:
    print(f"\n{'═' * 80}")
    print("  SCRIPT CHECK")
    print(f"{'═' * 80}")

    mk_ok, mk_fail = 0, 0
    sq_ok, sq_fail = 0, 0

    for p in pairs:
        mk_cyr = sum(1 for c in p["mk"] if _is_cyrillic(c))
        mk_lat = sum(1 for c in p["mk"] if _is_latin(c))
        mk_total = mk_cyr + mk_lat
        if mk_total > 0 and mk_cyr / mk_total > 0.5:
            mk_ok += 1
        else:
            mk_fail += 1

        sq_cyr = sum(1 for c in p["sq"] if _is_cyrillic(c))
        sq_lat = sum(1 for c in p["sq"] if _is_latin(c))
        sq_total = sq_cyr + sq_lat
        if sq_total > 0 and sq_lat / sq_total > 0.5:
            sq_ok += 1
        else:
            sq_fail += 1

    n = len(pairs)
    print(f"  MK Cyrillic:  {mk_ok}/{n} OK  ({mk_fail} failures)")
    print(f"  SQ Latin:     {sq_ok}/{n} OK  ({sq_fail} failures)")

    if mk_fail:
        print(f"  ⚠ {mk_fail} MK sentences have <50% Cyrillic characters")
    if sq_fail:
        print(f"  ⚠ {sq_fail} SQ sentences have <50% Latin characters")


# ── 3. Length distribution ──────────────────────────────────────

def length_histogram(pairs: list[dict]) -> None:
    print(f"\n{'═' * 80}")
    print("  LENGTH DISTRIBUTION")
    print(f"{'═' * 80}")

    mk_lens = [len(p["mk"].split()) for p in pairs]
    sq_lens = [len(p["sq"].split()) for p in pairs]

    print(f"  MK — min={min(mk_lens)}, max={max(mk_lens)}, "
          f"mean={sum(mk_lens)/len(mk_lens):.1f}")
    print(f"  SQ — min={min(sq_lens)}, max={max(sq_lens)}, "
          f"mean={sum(sq_lens)/len(sq_lens):.1f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(mk_lens, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
        axes[0].set_title("MK sentence lengths (words)")
        axes[0].set_xlabel("Words")
        axes[0].set_ylabel("Frequency")

        axes[1].hist(sq_lens, bins=50, alpha=0.7, color="coral", edgecolor="black")
        axes[1].set_title("SQ sentence lengths (words)")
        axes[1].set_xlabel("Words")
        axes[1].set_ylabel("Frequency")

        plt.tight_layout()
        out_path = config.EXPORT_DIR / "length_histogram.png"
        plt.savefig(out_path, dpi=150)
        print(f"  ✓ Histogram saved to {out_path}")
    except ImportError:
        print("  (matplotlib not available — histogram skipped)")


# ── 4. Duplicate rate ──────────────────────────────────────────

def duplicate_rate(pairs: list[dict]) -> None:
    print(f"\n{'═' * 80}")
    print("  DUPLICATE ANALYSIS")
    print(f"{'═' * 80}")

    import hashlib

    hashes: set[str] = set()
    dups = 0
    for p in pairs:
        combined = p["mk"].strip().lower() + " ||| " + p["sq"].strip().lower()
        h = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        if h in hashes:
            dups += 1
        else:
            hashes.add(h)

    pct = 100 * dups / len(pairs) if pairs else 0
    print(f"  Total pairs:     {len(pairs)}")
    print(f"  Exact duplicates: {dups} ({pct:.2f}%)")


# ── 5. Coverage check ──────────────────────────────────────────

def coverage_check(pairs: list[dict]) -> None:
    print(f"\n{'═' * 80}")
    print("  COVERAGE CHECK")
    print(f"{'═' * 80}")

    # Count unique source PDFs that yielded pairs
    sources_with_pairs = {
        p.get("meta", {}).get("source", "") for p in pairs
    }

    # Count total downloaded PDFs
    pdf_count = len(list(config.PDF_DIR.rglob("*.pdf")))

    pct = 100 * len(sources_with_pairs) / pdf_count if pdf_count else 0
    print(f"  Total PDFs downloaded:  {pdf_count}")
    print(f"  PDFs yielding pairs:    {len(sources_with_pairs)}")
    print(f"  Coverage:               {pct:.1f}%")


# ───────────────────────── Main ─────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    import argparse
    parser = argparse.ArgumentParser(description="Validate Vezilka corpus quality")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to aligned_pairs.json")
    parser.add_argument("--sample", type=int, default=50,
                        help="Number of pairs to spot-check")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else None
    pairs = load_pairs(input_path)

    if not pairs:
        print("No pairs to validate.")
        return

    spot_check(pairs, n=args.sample)
    script_check(pairs)
    length_histogram(pairs)
    duplicate_rate(pairs)
    coverage_check(pairs)

    print(f"\n{'═' * 80}")
    print(f"  VALIDATION COMPLETE — {len(pairs)} pairs checked.")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    main()
