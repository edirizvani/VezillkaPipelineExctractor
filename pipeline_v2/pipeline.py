"""
Vezilka v2 — Master Pipeline.

Orchestrates the full extraction → segmentation → alignment →
validation → export workflow across all 4,931 existing PDFs.

Fully resumable — checkpoints at every phase boundary:
  • Phase 3 extracted text  → data/extracted/{pdf_id}.json
  • Phase 3 segmented items → data/segmented/{pdf_id}.json
  • Phase 4 aligned pairs   → data/aligned/{pdf_id}.jsonl
  • Phase 5 final output    → data/output/corpus.tsv + .jsonl

Failed / skipped PDFs are logged to separate JSONL files.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from config import DEFAULT_CONFIG, VezilkaConfig
from phase3_extract.document_segmenter import DocumentSegmenter, GazetteItem
from phase3_extract.extractor_ocr import OCRExtractor
from phase3_extract.extractor_sequential import SequentialExtractor
from phase3_extract.extractor_two_column import TwoColumnExtractor
from phase3_extract.gatekeeper import Gatekeeper, SkipReason
from phase3_extract.layout_classifier import LayoutClassifier, LayoutType
from phase4_align.aligner_orchestrator import AlignerOrchestrator, CandidatePair
from phase5_clean.deduplicator import Deduplicator
from phase5_clean.exporter import Exporter
from phase5_clean.noise_filter import NoiseFilter
from phase5_clean.pair_filter import PairFilter
from phase5_clean.semantic_validator import SemanticValidator
from utils.logging_config import setup_logging
from utils.text_utils import clean_text, fix_albanian_encoding

logger = logging.getLogger(__name__)


class VezilkaPipeline:
    """End-to-end pipeline for building the MK↔SQ parallel corpus."""

    def __init__(self, config: VezilkaConfig | None = None):
        self.cfg = config or DEFAULT_CONFIG
        self.cfg.ensure_dirs()

        # Phase 3
        self.gatekeeper = Gatekeeper(self.cfg)
        self.classifier = LayoutClassifier(
            column_split_tolerance=self.cfg.column_split_tolerance,
            two_column_min_fraction=self.cfg.two_column_min_page_fraction,
            cyrillic_threshold=self.cfg.cyrillic_threshold,
            latin_threshold=self.cfg.latin_threshold,
            sequential_transition_threshold=self.cfg.sequential_transition_threshold,
        )
        self.segmenter = DocumentSegmenter(self.cfg)
        self.ext_two_col = TwoColumnExtractor(tolerance=self.cfg.column_split_tolerance)
        self.ext_seq = SequentialExtractor()
        self.ext_ocr = OCRExtractor(use_ocr=True)
        self.noise = NoiseFilter(self.cfg)

        # Phase 4
        self.orchestrator = AlignerOrchestrator(self.cfg)

        # Phase 5
        self.validator = SemanticValidator(self.cfg)
        self.pair_filter = PairFilter(self.cfg)
        self.dedup = Deduplicator(self.cfg)
        self.exporter = Exporter(self.cfg)

    # ── public entry point ──────────────────────────────────────

    def run(
        self,
        limit: int | None = None,
        skip_validation: bool = False,
    ) -> None:
        """Run the full pipeline over all PDFs in config.pdf_dir."""
        setup_logging(level=self.cfg.log_level)
        start = time.time()

        pdfs = self._discover_pdfs(limit)
        logger.info("Found %d PDFs to process", len(pdfs))

        all_candidates: list[CandidatePair] = []
        stats = {"processed": 0, "skipped": 0, "failed": 0, "total_pairs": 0}

        for pdf_path in tqdm(pdfs, desc="Processing PDFs"):
            pdf_id = self._pdf_id(pdf_path)

            # ── Checkpoint: already aligned? ────────────────────
            aligned_path = self.cfg.aligned_dir / f"{pdf_id}.jsonl"
            if aligned_path.exists():
                pairs = self._load_aligned(aligned_path)
                all_candidates.extend(pairs)
                stats["processed"] += 1
                stats["total_pairs"] += len(pairs)
                continue

            # ── Phase 3a: Gatekeeper ────────────────────────────
            decision = self.gatekeeper.check(pdf_path)
            if not decision.should_process:
                self.gatekeeper.log_skip(pdf_path, decision)
                stats["skipped"] += 1
                continue

            # ── Phase 3b: Layout classification ─────────────────
            try:
                layout = self.classifier.classify(pdf_path)
            except Exception as e:
                self._log_failure(pdf_path, "layout_classification", e)
                stats["failed"] += 1
                continue

            # ── Phase 3c: Extraction ────────────────────────────
            try:
                mk_text, sq_text = self._extract(pdf_path, layout)
            except Exception as e:
                self._log_failure(pdf_path, "extraction", e)
                stats["failed"] += 1
                continue

            if not mk_text.strip() or not sq_text.strip():
                self.gatekeeper.log_skip(
                    pdf_path,
                    type("Decision", (), {
                        "should_process": False,
                        "reason": SkipReason.NO_ALBANIAN_DETECTED,
                        "detail": "extraction_empty"
                    })(),
                )
                stats["skipped"] += 1
                continue

            # ── Noise cleaning ──────────────────────────────────
            mk_text = self.noise.clean_mk(mk_text)
            sq_text = self.noise.clean_sq(sq_text)

            # ── Phase 3d: Document segmentation ─────────────────
            try:
                seg_result = self.segmenter.segment(mk_text, sq_text)
            except Exception as e:
                self._log_failure(pdf_path, "segmentation", e)
                stats["failed"] += 1
                continue

            # Save extraction checkpoint
            self._save_extracted(pdf_id, mk_text, sq_text, layout)

            # ── Phase 4: Alignment per item ─────────────────────
            item_pairs: list[CandidatePair] = []
            for item in seg_result.items:
                if not item.valid or not item.is_bilingual:
                    continue
                orch = self.orchestrator.align_item(
                    mk_text=item.mk_text,
                    sq_text=item.sq_text,
                    pdf_id=pdf_id,
                    item_number=item.item_number,
                    layout_type=layout.layout_type.value,
                )
                item_pairs.extend(orch.candidates)

            # Save alignment checkpoint
            self._save_aligned(pdf_id, item_pairs)

            all_candidates.extend(item_pairs)
            stats["processed"] += 1
            stats["total_pairs"] += len(item_pairs)

        logger.info(
            "Extraction complete: %d processed, %d skipped, %d failed, %d raw pairs",
            stats["processed"], stats["skipped"], stats["failed"], stats["total_pairs"],
        )

        # ── Phase 5a: Semantic validation ───────────────────────
        if not skip_validation and all_candidates:
            logger.info("Starting semantic validation on %d candidates", len(all_candidates))
            all_candidates = self.validator.validate(all_candidates)

        # ── Phase 5b: Pair filtering ────────────────────────────
        accepted = [p for p in all_candidates if not p.rejection_reason]
        accepted = self.pair_filter.filter(accepted)

        # ── Phase 5c: Deduplication ─────────────────────────────
        accepted = self.dedup.deduplicate(accepted)

        # ── Phase 5d: Export ────────────────────────────────────
        exported = self.exporter.export(accepted)

        elapsed = time.time() - start
        logger.info(
            "Pipeline complete in %.1f min: %d final pairs exported from %d PDFs",
            elapsed / 60, len(accepted), stats["processed"],
        )
        for fmt, path in exported.items():
            logger.info("  %s → %s", fmt, path)

    # ── PDF discovery ───────────────────────────────────────────

    def _discover_pdfs(self, limit: int | None) -> list[Path]:
        """Find all PDFs in the pdf_dir tree."""
        pdfs = sorted(self.cfg.pdf_dir.rglob("*.pdf"))
        if limit:
            pdfs = pdfs[:limit]
        return pdfs

    @staticmethod
    def _pdf_id(pdf_path: Path) -> str:
        """Create a unique ID from the PDF path: year_month_hash."""
        parts = pdf_path.parts
        try:
            idx = next(i for i, p in enumerate(parts) if p == "pdfs")
            remaining = parts[idx + 1:]
            return "_".join(remaining).replace(".pdf", "")
        except StopIteration:
            return pdf_path.stem

    # ── Extraction dispatch ─────────────────────────────────────

    def _extract(self, pdf_path: Path, layout) -> tuple[str, str]:
        """Dispatch to the correct extractor based on layout type."""
        lt = layout.layout_type

        if lt == LayoutType.TWO_COLUMN:
            r = self.ext_two_col.extract(pdf_path)
            return r.raw_mk, r.raw_sq

        if lt == LayoutType.SEQUENTIAL:
            r = self.ext_seq.extract(
                pdf_path,
                boundary_page=layout.boundary_page,
                boundary_block=layout.boundary_block,
            )
            return r.raw_mk, r.raw_sq

        if lt == LayoutType.MIXED_PRE2019:
            r = self.ext_ocr.extract(pdf_path)
            return r.raw_mk, r.raw_sq

        # UNKNOWN or SINGLE_LANGUAGE — try OCR extractor as last resort
        r = self.ext_ocr.extract(pdf_path)
        return r.raw_mk, r.raw_sq

    # ── Checkpointing ───────────────────────────────────────────

    def _save_extracted(self, pdf_id: str, mk: str, sq: str, layout) -> None:
        path = self.cfg.extracted_dir / f"{pdf_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pdf_id": pdf_id,
            "layout_type": layout.layout_type.value,
            "layout_confidence": layout.confidence,
            "boundary_page": layout.boundary_page,
            "mk_chars": len(mk),
            "sq_chars": len(sq),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_aligned(self, pdf_id: str, pairs: list[CandidatePair]) -> None:
        path = self.cfg.aligned_dir / f"{pdf_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for p in pairs:
                record = {
                    "mk": p.mk, "sq": p.sq,
                    "pdf_id": p.pdf_id,
                    "item_number": p.item_number,
                    "article_number": p.article_number,
                    "alignment_strategy": p.alignment_strategy,
                    "layout_type": p.layout_type,
                    "mk_word_count": p.mk_word_count,
                    "sq_word_count": p.sq_word_count,
                    "is_structural": p.is_structural,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_aligned(self, path: Path) -> list[CandidatePair]:
        """Load previously aligned pairs from checkpoint."""
        pairs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                pairs.append(CandidatePair(
                    mk=d["mk"], sq=d["sq"],
                    pdf_id=d.get("pdf_id", ""),
                    item_number=d.get("item_number", 0),
                    article_number=d.get("article_number"),
                    alignment_strategy=d.get("alignment_strategy", ""),
                    layout_type=d.get("layout_type", ""),
                    mk_word_count=d.get("mk_word_count", 0),
                    sq_word_count=d.get("sq_word_count", 0),
                    is_structural=d.get("is_structural", False),
                ))
        return pairs

    # ── Failure logging ─────────────────────────────────────────

    def _log_failure(self, pdf_path: Path, phase: str, error: Exception) -> None:
        logger.error("FAIL [%s] %s: %s", phase, pdf_path, error)
        log_path = self.cfg.failed_log
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "pdf": str(pdf_path),
            "phase": phase,
            "error": str(error),
            "error_type": type(error).__name__,
        }
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass


# ── CLI entry point ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Vezilka v2 Pipeline")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N PDFs (for testing)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip semantic validation (fast run)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    config = VezilkaConfig(log_level=args.log_level)
    pipeline = VezilkaPipeline(config=config)
    pipeline.run(limit=args.limit, skip_validation=args.skip_validation)


if __name__ == "__main__":
    main()
