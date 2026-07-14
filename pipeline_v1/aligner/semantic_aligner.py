"""
Semantic Aligner — LaBSE-based sentence pair scoring and filtering.

Uses Language-Agnostic BERT Sentence Embeddings (LaBSE) to compute
semantic similarity between Macedonian and Albanian sentence pairs.

This replaces / augments heuristic confidence scores (length-ratio based)
with actual cross-lingual semantic similarity, which is a much stronger
signal for translation equivalence.

LaBSE supports 109 languages including Macedonian and Albanian.

Reference:
    Feng et al. (2022) "Language-agnostic BERT Sentence Embedding"
    https://arxiv.org/abs/2007.01852

Usage::

    from aligner.semantic_aligner import SemanticAligner

    aligner = SemanticAligner()
    scores = aligner.score_pairs(mk_sentences, sq_sentences)
    filtered = aligner.filter_pairs(pairs, threshold=0.6)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Model singleton ─────────────────────────────────────────────
_MODEL = None
_MODEL_NAME = "sentence-transformers/LaBSE"


def _get_model():
    """Load LaBSE model once and cache in module-level singleton."""
    global _MODEL
    if _MODEL is None:
        logger.info("Loading LaBSE model (first call — this takes ~30s)…")
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(_MODEL_NAME)
        logger.info("LaBSE model loaded.")
    return _MODEL


# ── Data classes ────────────────────────────────────────────────

@dataclass
class ScoredPair:
    """A sentence pair with semantic similarity score."""
    mk: str
    sq: str
    semantic_score: float        # cosine similarity from LaBSE
    original_confidence: float   # the old heuristic confidence
    combined_score: float        # blended score
    article_id: str = ""
    source: str = ""
    method: str = ""


# ── Main class ──────────────────────────────────────────────────

class SemanticAligner:
    """
    Scores and filters MK ↔ SQ sentence pairs using LaBSE embeddings.

    The semantic similarity is a cosine similarity in the LaBSE embedding
    space.  Two sentences that are translations of each other will have
    cosine similarity > 0.7 typically, while unrelated pairs will be < 0.4.
    """

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        batch_size: int = 256,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

    # ── Encode sentences ────────────────────────────────────────

    def encode(self, sentences: list[str]) -> np.ndarray:
        """
        Encode a list of sentences into LaBSE embeddings.

        Returns
        -------
        np.ndarray of shape ``(n, 768)``
        """
        model = _get_model()
        embeddings = model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.device,
            normalize_embeddings=True,  # for cosine similarity via dot product
        )
        return embeddings

    # ── Score pairs ─────────────────────────────────────────────

    def score_pairs(
        self,
        mk_sentences: list[str],
        sq_sentences: list[str],
    ) -> np.ndarray:
        """
        Compute semantic similarity for paired sentences.

        Parameters
        ----------
        mk_sentences : list[str]
            Macedonian sentences (N items).
        sq_sentences : list[str]
            Albanian sentences (N items, same order as mk).

        Returns
        -------
        np.ndarray of shape ``(N,)`` with cosine similarities in [-1, 1].
        """
        assert len(mk_sentences) == len(sq_sentences), (
            f"Length mismatch: {len(mk_sentences)} MK vs {len(sq_sentences)} SQ"
        )

        logger.info("Encoding %d MK sentences…", len(mk_sentences))
        mk_emb = self.encode(mk_sentences)

        logger.info("Encoding %d SQ sentences…", len(sq_sentences))
        sq_emb = self.encode(sq_sentences)

        # Cosine similarity (embeddings are already L2-normalized)
        scores = (mk_emb * sq_emb).sum(axis=1)

        logger.info(
            "Semantic scores: mean=%.3f, median=%.3f, min=%.3f, max=%.3f",
            scores.mean(), np.median(scores), scores.min(), scores.max(),
        )
        return scores

    # ── Filter pairs ────────────────────────────────────────────

    def filter_pairs(
        self,
        pairs: list[dict],
        threshold: float = 0.6,
        score_column: str = "semantic_score",
    ) -> tuple[list[dict], list[dict]]:
        """
        Split pairs into kept (above threshold) and rejected (below).

        Parameters
        ----------
        pairs : list[dict]
            Each dict must have 'mk', 'sq', and *score_column*.
        threshold : float
            Minimum semantic similarity to keep.

        Returns
        -------
        (kept, rejected) — two lists of dicts.
        """
        kept = [p for p in pairs if p.get(score_column, 0) >= threshold]
        rejected = [p for p in pairs if p.get(score_column, 0) < threshold]

        logger.info(
            "Semantic filter (threshold=%.2f): %d kept, %d rejected",
            threshold, len(kept), len(rejected),
        )
        return kept, rejected

    # ── Combine scores ──────────────────────────────────────────

    @staticmethod
    def blend_scores(
        semantic: np.ndarray,
        heuristic: np.ndarray,
        semantic_weight: float = 0.7,
    ) -> np.ndarray:
        """
        Blend semantic similarity with old heuristic confidence.

        Default: 70% semantic, 30% heuristic.
        """
        return semantic_weight * semantic + (1 - semantic_weight) * heuristic
