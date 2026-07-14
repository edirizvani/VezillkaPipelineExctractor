"""
Dataset Exporter — exports aligned sentence pairs to TSV, JSON Lines,
and HuggingFace ``datasets`` format.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DatasetExporter:
    """
    Exports a list of sentence-pair dicts to multiple formats.

    Each pair dict should have at least::

        {"mk": str, "sq": str,
         "meta": {"source": str, "article": str,
                  "confidence": float, "method": str}}
    """

    # ── TSV ─────────────────────────────────────────────────────

    @staticmethod
    def export_tsv(pairs: list[dict], output_path: str | Path) -> int:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            writer.writerow([
                "mk_sentence", "sq_sentence", "source_issue",
                "article_id", "confidence", "alignment_method",
            ])
            for pair in pairs:
                meta = pair.get("meta", {})
                writer.writerow([
                    pair["mk"],
                    pair["sq"],
                    meta.get("source", ""),
                    meta.get("article", ""),
                    meta.get("confidence", ""),
                    meta.get("method", ""),
                ])

        logger.info("Exported %d pairs to TSV: %s", len(pairs), output_path)
        return len(pairs)

    # ── JSON Lines ──────────────────────────────────────────────

    @staticmethod
    def export_jsonl(pairs: list[dict], output_path: str | Path) -> int:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        logger.info("Exported %d pairs to JSONL: %s", len(pairs), output_path)
        return len(pairs)

    # ── HuggingFace datasets ────────────────────────────────────

    @staticmethod
    def export_huggingface(
        pairs: list[dict],
        output_dir: str | Path,
        train_frac: float = 0.80,
        val_frac: float = 0.10,
        test_frac: float = 0.10,
        seed: int = 42,
    ) -> dict[str, int]:
        try:
            from datasets import Dataset, DatasetDict, Features, Value
        except ImportError:
            logger.error("'datasets' library not installed.")
            return {}

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        flat: list[dict[str, Any]] = []
        for pair in pairs:
            meta = pair.get("meta", {})
            flat.append({
                "mk": pair["mk"],
                "sq": pair["sq"],
                "source": meta.get("source", ""),
                "article_id": meta.get("article", ""),
                "confidence": float(meta.get("confidence", 0.0)),
                "alignment_method": meta.get("method", ""),
            })

        rng = random.Random(seed)
        rng.shuffle(flat)
        n = len(flat)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        splits_data = {
            "train": flat[:n_train],
            "validation": flat[n_train : n_train + n_val],
            "test": flat[n_train + n_val :],
        }

        features = Features({
            "mk": Value("string"),
            "sq": Value("string"),
            "source": Value("string"),
            "article_id": Value("string"),
            "confidence": Value("float32"),
            "alignment_method": Value("string"),
        })

        dataset_dict = DatasetDict({
            name: Dataset.from_list(data, features=features)
            for name, data in splits_data.items()
        })

        for ds in dataset_dict.values():
            ds.info.description = (
                "Macedonian ↔ Albanian parallel corpus extracted from "
                "Služben Vesnik (Official Gazette of North Macedonia). "
                "Part of the Vezilka NLP research project."
            )
            ds.info.license = "public-domain"

        dataset_dict.save_to_disk(str(output_dir))
        sizes = {k: len(v) for k, v in splits_data.items()}
        logger.info("Exported HuggingFace dataset to %s: %s", output_dir, sizes)
        return sizes

    # ── Statistics ──────────────────────────────────────────────

    @staticmethod
    def print_statistics(pairs: list[dict]) -> str:
        if not pairs:
            report = "=== DATASET STATISTICS ===\nNo pairs to report.\n"
            print(report)
            return report

        n = len(pairs)

        method_counts: Counter = Counter()
        year_counts: Counter = Counter()

        for p in pairs:
            meta = p.get("meta", {})
            method_counts[meta.get("method", "unknown")] += 1
            source = meta.get("source", "")
            year = "unknown"
            for part in source.split("_"):
                if part.isdigit() and len(part) == 4:
                    year = part
                    break
            year_counts[year] += 1

        mk_wl = [len(p["mk"].split()) for p in pairs]
        sq_wl = [len(p["sq"].split()) for p in pairs]
        avg_mk = sum(mk_wl) / n
        avg_sq = sum(sq_wl) / n
        ratio = avg_mk / avg_sq if avg_sq else 0

        lines = [
            "=== DATASET STATISTICS ===",
            f"Total pairs:              {n:,}",
            "",
            "By alignment method:",
        ]
        for method, count in method_counts.most_common():
            pct = 100 * count / n
            lines.append(f"  {method:30s} {count:>8,} ({pct:.1f}%)")
        lines.append("")
        lines.append("By year:")
        for year in sorted(year_counts):
            lines.append(f"  {year}: {year_counts[year]:>8,}")
        lines.extend([
            "",
            f"Avg MK sentence length:   {avg_mk:.1f} words",
            f"Avg SQ sentence length:   {avg_sq:.1f} words",
            f"MK/SQ length ratio:       {ratio:.2f}",
        ])

        report = "\n".join(lines)
        print(report)
        return report
