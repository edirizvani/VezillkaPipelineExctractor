"""
Vezilka v2 — Exporter.

Writes the final cleaned corpus in multiple formats:
  • TSV   — tab-separated values
  • JSONL — one JSON object per line with full metadata
  • HuggingFace datasets (optional)

Also produces train / val / test splits.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

import pandas as pd

from config import DEFAULT_CONFIG, VezilkaConfig
from phase4_align.aligner_orchestrator import CandidatePair

logger = logging.getLogger(__name__)


class Exporter:
    """Export validated pairs in multiple formats with splits."""

    def __init__(self, config: VezilkaConfig | None = None):
        self.cfg = config or DEFAULT_CONFIG

    def export(
        self,
        pairs: list[CandidatePair],
        seed: int = 42,
    ) -> dict[str, Path]:
        """Export all pairs and return dict of {format: path}."""
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        exported: dict[str, Path] = {}

        # Convert to dicts
        records = [self._pair_to_dict(p) for p in pairs]
        if not records:
            logger.warning("No pairs to export")
            return exported

        # Full corpus
        if "csv" in self.cfg.export_formats:
            path = self._write_csv(records, "corpus")
            exported["csv"] = path

        if "tsv" in self.cfg.export_formats:
            path = self._write_tsv(records, "corpus")
            exported["tsv"] = path

        if "jsonl" in self.cfg.export_formats:
            path = self._write_jsonl(records, "corpus")
            exported["jsonl"] = path

        # Splits
        random.seed(seed)
        random.shuffle(records)
        n = len(records)
        n_train = int(n * self.cfg.train_fraction)
        n_val = int(n * self.cfg.val_fraction)

        train = records[:n_train]
        val = records[n_train:n_train + n_val]
        test = records[n_train + n_val:]

        for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
            if "csv" in self.cfg.export_formats:
                self._write_csv(split_data, split_name)
            if "tsv" in self.cfg.export_formats:
                self._write_tsv(split_data, split_name)
            if "jsonl" in self.cfg.export_formats:
                self._write_jsonl(split_data, split_name)

        # Optional HuggingFace dataset
        if "huggingface" in self.cfg.export_formats:
            try:
                hf_path = self._write_huggingface(train, val, test)
                exported["huggingface"] = hf_path
            except Exception as e:
                logger.warning("HuggingFace export failed: %s", e)

        logger.info("Exported %d pairs in formats: %s", n, list(exported.keys()))
        return exported

    # ── writers ─────────────────────────────────────────────────

    def _write_csv(self, records: list[dict], name: str) -> Path:
        path = self.cfg.output_dir / f"vezilka_v2_{name}.csv"
        df = pd.DataFrame(records)
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info("Wrote %s (%d rows)", path.name, len(df))
        return path

    def _write_tsv(self, records: list[dict], name: str) -> Path:
        path = self.cfg.output_dir / f"vezilka_v2_{name}.tsv"
        df = pd.DataFrame(records)
        df.to_csv(path, sep="\t", index=False, encoding="utf-8")
        logger.info("Wrote %s (%d rows)", path.name, len(df))
        return path

    def _write_jsonl(self, records: list[dict], name: str) -> Path:
        path = self.cfg.output_dir / f"vezilka_v2_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Wrote %s (%d records)", path.name, len(records))
        return path

    def _write_huggingface(
        self,
        train: list[dict],
        val: list[dict],
        test: list[dict],
    ) -> Path:
        from datasets import Dataset, DatasetDict
        ds = DatasetDict({
            "train": Dataset.from_list(train),
            "validation": Dataset.from_list(val),
            "test": Dataset.from_list(test),
        })
        path = self.cfg.output_dir / "vezilka_v2_hf"
        ds.save_to_disk(str(path))
        logger.info("Wrote HuggingFace dataset to %s", path)
        return path

    # ── helpers ─────────────────────────────────────────────────

    @staticmethod
    def _pair_to_dict(p: CandidatePair) -> dict:
        return {
            "mk": p.mk,
            "sq": p.sq,
            "pdf_id": p.pdf_id,
            "item_number": p.item_number,
            "article_number": p.article_number or 0,
            "alignment_strategy": p.alignment_strategy,
            "layout_type": p.layout_type,
            "labse_score": round(p.labse_score, 4),
            "laser3_score": round(p.laser3_score, 4),
            "comet_qe_score": round(p.comet_qe_score, 4),
            "back_translation_score": round(p.back_translation_score, 4),
            "length_ratio_score": round(p.length_ratio_score, 4),
            "blended_confidence": round(p.blended_confidence, 4),
            "mk_word_count": p.mk_word_count,
            "sq_word_count": p.sq_word_count,
            "tier_reached": p.tier_reached,
        }
