"""
Vezilka v2 — Configuration Module.

All tuneable parameters in one place.  Uses a dataclass for type safety.

IMPORTANT: pdf_dir points to the EXISTING PDFs in ../pipeline_v1/data/pdfs/
so we don't need to re-scrape or re-download anything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VezilkaConfig:
    """Central configuration for the entire Vezilka v2 pipeline."""

    # ──────────────────── PATHS ─────────────────────────────────
    project_root: Path = Path(__file__).resolve().parent
    pdf_dir: Path = Path(__file__).resolve().parent.parent / "pipeline_v1" / "data" / "pdfs"
    data_dir: Path = Path(__file__).resolve().parent / "data"
    extracted_dir: Path = Path(__file__).resolve().parent / "data" / "extracted"
    segmented_dir: Path = Path(__file__).resolve().parent / "data" / "segmented"
    aligned_dir: Path = Path(__file__).resolve().parent / "data" / "aligned"
    output_dir: Path = Path(__file__).resolve().parent / "data" / "output"
    failed_log: Path = Path(__file__).resolve().parent / "data" / "failed_pdfs.jsonl"
    skipped_log: Path = Path(__file__).resolve().parent / "data" / "skipped_pdfs.jsonl"
    rejected_pairs_log: Path = Path(__file__).resolve().parent / "data" / "rejected_pairs.jsonl"

    # ──────────────────── GATEKEEPER ────────────────────────────
    gatekeeper_min_file_size_kb: int = 50
    gatekeeper_albanian_chars: frozenset = frozenset("ëËçÇ")
    gatekeeper_albanian_keywords: list = field(default_factory=lambda: [
        "Neni", "LIGJ", "ligj", "Maqedonisë", "Veriut", "denarë", "Gazeta",
    ])
    gatekeeper_min_latin_ratio_for_albanian: float = 0.20

    # ──────────────────── DOCUMENT SEGMENTER ────────────────────
    item_boundary_pattern: str = r"^\s*(\d{3,4})\.\s*$"
    segmenter_min_laser_doc_sim: float = 0.80

    # ──────────────────── LAYOUT CLASSIFICATION ─────────────────
    two_column_min_page_fraction: float = 0.50
    column_split_tolerance: float = 0.08
    cyrillic_threshold: float = 0.60
    latin_threshold: float = 0.60
    sequential_transition_threshold: float = 0.70

    albanian_boundary_patterns: list = field(default_factory=lambda: [
        r"^L\s*I\s*G\s*J\b",
        r"^LIGJ\b",
        r"^Neni\s+1\b",
        r"_{5,}",
        r"—{5,}",
    ])

    # ──────────────────── ALIGNMENT ─────────────────────────────
    structural_article_pattern_mk: str = r"Член\s+(\d+)"
    structural_article_pattern_sq: str = r"Neni\s+(\d+)"
    gc_mean_char_ratio: float = 1.1
    gc_variance: float = 6.8
    min_article_text_length: int = 10

    # ──────────────────── TRANSLATION SCORER (MarianMT) ─────────
    translation_model: str = "helsinki"
    translation_batch_size: int = 32
    run_translation_scoring: bool = True

    # ──────────────────── LaBSE ─────────────────────────────────
    labse_model: str = "sentence-transformers/LaBSE"
    labse_batch_size: int = 256
    labse_min_similarity: float = 0.55

    # ──────────────────── LASER3 ────────────────────────────────
    run_laser3: bool = True
    laser3_min_similarity: float = 0.55

    # ──────────────────── DENSE RETRIEVAL ALIGNMENT ─────────────
    dense_retrieval_min_similarity: float = 0.70
    dense_retrieval_use_faiss: bool = True
    dense_retrieval_monotonicity: bool = True
    dense_retrieval_min_pairs_fallback: int = 3
    gc_min_sentences_threshold: int = 10

    # ──────────────────── COMET-QE ──────────────────────────────
    run_comet_qe: bool = True
    comet_qe_model: str = "Unbabel/wmt22-cometkiwi-da"
    comet_qe_min_score: float = 0.65

    # ──────────────────── BACK-TRANSLATION ──────────────────────
    run_back_translation: bool = True
    bt_mk_to_sq_model: str = "Helsinki-NLP/opus-mt-mk-sq"
    bt_sq_to_mk_model: str = "Helsinki-NLP/opus-mt-sq-mk"
    bt_batch_size: int = 16

    # ──────────────────── BLENDED SCORE WEIGHTS ─────────────────
    w_structural_labse: float = 0.50
    w_structural_laser3: float = 0.30
    w_structural_length: float = 0.20

    w_nonstructural_labse: float = 0.25
    w_nonstructural_laser3: float = 0.20
    w_nonstructural_comet_qe: float = 0.25
    w_nonstructural_backtranslation: float = 0.20
    w_nonstructural_length: float = 0.10

    blended_min_score: float = 0.40  # Lowered to be more lenient when signals fail

    # ──────────────────── HARD REJECTION THRESHOLDS ─────────────
    hard_reject_labse: float = 0.45  # Lowered
    hard_reject_comet_qe: float = 0.50  # Lowered
    hard_reject_laser3: float = 0.45  # Lowered
    hard_reject_length_ratio: float = 0.30  # Lowered

    # ──────────────────── FILTERING ─────────────────────────────
    min_words: int = 5
    max_words: int = 200
    min_length_ratio: float = 0.4
    max_length_ratio: float = 2.5
    max_digit_fraction: float = 0.30
    mk_min_cyrillic: float = 0.50
    sq_min_latin: float = 0.50

    # ──────────────────── DEDUPLICATION ──────────────────────────
    minhash_threshold: float = 0.95
    minhash_num_perm: int = 128

    # ──────────────────── EXPORT ────────────────────────────────
    train_fraction: float = 0.80
    val_fraction: float = 0.10
    test_fraction: float = 0.10
    export_formats: list = field(default_factory=lambda: ["csv", "tsv", "jsonl"])

    # ──────────────────── LOGGING ───────────────────────────────
    log_level: str = "INFO"

    def ensure_dirs(self) -> None:
        """Create all data directories if they don't exist."""
        for d in (self.data_dir, self.extracted_dir, self.segmented_dir,
                  self.aligned_dir, self.output_dir):
            d.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = VezilkaConfig()
