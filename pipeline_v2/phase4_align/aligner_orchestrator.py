"""
Vezilka v2 — Aligner Orchestrator.

Chooses the alignment strategy per gazette item and returns raw
candidate pairs tagged with their strategy.  Applies the three-
strategy cascade:

    1. Structural (Член N ↔ Neni N)  — highest reliability
    2. Dense Retrieval (LASER3 mutual-NN + monotonicity) — medium
    3. Gale-Church (character-length DP)  — last resort

For each item in a segmented PDF issue:
  • If article numbers are present → Strategy 1
  • Else if enough sentences → Strategy 2
  • Else → Strategy 3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from config import DEFAULT_CONFIG, VezilkaConfig
from phase4_align.dense_retrieval_aligner import DenseRetrievalAligner
from phase4_align.gale_church import GaleChurchAligner
from phase4_align.structural_aligner import StructuralAligner

logger = logging.getLogger(__name__)


@dataclass
class CandidatePair:
    """A single candidate sentence pair with all metadata."""
    mk: str
    sq: str
    pdf_id: str = ""
    item_number: int = 0
    article_number: Optional[int] = None
    alignment_strategy: str = ""
    layout_type: str = ""
    mk_word_count: int = 0
    sq_word_count: int = 0
    # Individual scores (filled later by semantic validator)
    labse_score: float = 0.0
    laser3_score: float = 0.0
    comet_qe_score: float = 0.0
    back_translation_score: float = 0.0
    length_ratio_score: float = 0.0
    blended_confidence: float = 0.0
    tier_reached: int = 0
    is_structural: bool = False
    rejection_reason: str = ""


@dataclass
class OrchestrationResult:
    candidates: list[CandidatePair] = field(default_factory=list)
    strategy_used: str = ""
    structural_articles: int = 0
    dense_pairs: int = 0
    gc_pairs: int = 0


class AlignerOrchestrator:
    """Three-strategy alignment cascade for a single gazette item."""

    def __init__(self, config: VezilkaConfig | None = None):
        self.cfg = config or DEFAULT_CONFIG
        self.structural = StructuralAligner(config=self.cfg)
        self.dense = DenseRetrievalAligner(config=self.cfg)
        self.gc = GaleChurchAligner(config=self.cfg)

    def align_item(
        self,
        mk_text: str,
        sq_text: str,
        pdf_id: str = "",
        item_number: int = 0,
        layout_type: str = "",
    ) -> OrchestrationResult:
        """Align a single gazette item using the best available strategy."""
        result = OrchestrationResult()

        if not mk_text.strip() or not sq_text.strip():
            return result

        # ── Strategy 1: Structural ──────────────────────────────
        if self.structural.has_articles(mk_text, sq_text):
            s1 = self.structural.align(mk_text, sq_text)
            if s1.pairs:
                result.strategy_used = "structural"
                result.structural_articles = s1.matched_articles
                for p in s1.pairs:
                    result.candidates.append(CandidatePair(
                        mk=p.mk, sq=p.sq,
                        pdf_id=pdf_id, item_number=item_number,
                        article_number=p.article_number,
                        alignment_strategy="structural",
                        layout_type=layout_type,
                        mk_word_count=len(p.mk.split()),
                        sq_word_count=len(p.sq.split()),
                        is_structural=True,
                    ))
                logger.info(
                    "Item %d: structural → %d pairs (%d articles)",
                    item_number, len(s1.pairs), s1.matched_articles,
                )
                return result

        # ── Strategy 2: Dense Retrieval ─────────────────────────
        s2 = self.dense.align(mk_text, sq_text)
        if len(s2.pairs) >= self.cfg.dense_retrieval_min_pairs_fallback:
            result.strategy_used = "dense_retrieval"
            result.dense_pairs = len(s2.pairs)
            for p in s2.pairs:
                result.candidates.append(CandidatePair(
                    mk=p.mk, sq=p.sq,
                    pdf_id=pdf_id, item_number=item_number,
                    alignment_strategy="dense_retrieval",
                    layout_type=layout_type,
                    mk_word_count=len(p.mk.split()),
                    sq_word_count=len(p.sq.split()),
                    laser3_score=p.similarity,
                ))
            logger.info(
                "Item %d: dense retrieval → %d pairs",
                item_number, len(s2.pairs),
            )
            return result

        # ── Strategy 3: Gale-Church ─────────────────────────────
        s3 = self.gc.align(mk_text, sq_text)
        result.strategy_used = "gale_church"
        result.gc_pairs = len(s3.pairs)
        for p in s3.pairs:
            result.candidates.append(CandidatePair(
                mk=p.mk, sq=p.sq,
                pdf_id=pdf_id, item_number=item_number,
                alignment_strategy="gale_church",
                layout_type=layout_type,
                mk_word_count=len(p.mk.split()),
                sq_word_count=len(p.sq.split()),
            ))
        logger.info(
            "Item %d: Gale-Church fallback → %d pairs",
            item_number, len(s3.pairs),
        )
        return result
