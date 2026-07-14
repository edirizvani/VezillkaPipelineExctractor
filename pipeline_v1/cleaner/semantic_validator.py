"""
Semantic Validator — uses LASER and LaBSE to filter sentence pairs by similarity.

Provides cross-lingual semantic similarity scoring to filter out misaligned pairs.
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ───────────────────────── Result dataclass ─────────────────────


@dataclass
class SemanticScore:
    """Semantic validation score for a sentence pair."""
    laser_sim: Optional[float]  # LASER cosine similarity
    labse_sim: Optional[float]  # LaBSE cosine similarity
    combined: float             # Combined score (average of available)
    is_valid: bool              # Whether pair passes threshold
    reason: str                 # Reason for pass/fail


# ───────────────────────── SemanticValidator ────────────────────


class SemanticValidator:
    """
    Validates sentence pairs using LASER and LaBSE embeddings.

    LASER is loaded for specific languages (Macedonian and Albanian),
    while LaBSE is language-agnostic.

    Usage::

        validator = SemanticValidator()
        score = validator.score_pair("Ова е тест.", "Ky është një test.")
        if score.is_valid:
            ...
    """

    def __init__(
        self,
        min_similarity: float = 0.70,
        use_laser: bool = True,
        use_labse: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize the semantic validator.

        Args:
            min_similarity: Minimum combined similarity score to pass
            use_laser: Whether to use LASER embeddings
            use_labse: Whether to use LaBSE embeddings
            batch_size: Batch size for encoding
        """
        self.min_similarity = min_similarity
        self.batch_size = batch_size
        
        self._laser_mk = None
        self._laser_sq = None
        self._labse = None
        
        self._laser_available = use_laser
        self._labse_available = use_labse

        if use_laser:
            self._init_laser()
        if use_labse:
            self._init_labse()

    def _init_laser(self) -> None:
        """Initialize LASER encoders for Macedonian and Albanian."""
        try:
            from laser_encoders import LaserEncoderPipeline
            
            logger.info("Loading LASER encoder for Macedonian (mkd_Cyrl)...")
            self._laser_mk = LaserEncoderPipeline(lang="mkd_Cyrl")
            
            logger.info("Loading LASER encoder for Albanian (als_Latn)...")
            self._laser_sq = LaserEncoderPipeline(lang="als_Latn")
            
            logger.info("✓ LASER encoders loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load LASER: {e}")
            self._laser_available = False

    def _init_labse(self) -> None:
        """Initialize LaBSE model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info("Loading LaBSE model...")
            self._labse = SentenceTransformer("sentence-transformers/LaBSE")
            logger.info("✓ LaBSE loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load LaBSE: {e}")
            self._labse_available = False

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def score_pair(self, mk: str, sq: str) -> SemanticScore:
        """
        Score a single sentence pair.

        Args:
            mk: Macedonian sentence
            sq: Albanian sentence

        Returns:
            SemanticScore with similarity values and validity
        """
        scores = []
        laser_sim = None
        labse_sim = None

        # LASER similarity
        if self._laser_available and self._laser_mk and self._laser_sq:
            try:
                mk_emb = self._laser_mk.encode_sentences([mk])[0]
                sq_emb = self._laser_sq.encode_sentences([sq])[0]
                laser_sim = self._cosine_similarity(mk_emb, sq_emb)
                scores.append(laser_sim)
            except Exception as e:
                logger.debug(f"LASER scoring failed: {e}")

        # LaBSE similarity
        if self._labse_available and self._labse:
            try:
                embs = self._labse.encode([mk, sq])
                labse_sim = self._cosine_similarity(embs[0], embs[1])
                scores.append(labse_sim)
            except Exception as e:
                logger.debug(f"LaBSE scoring failed: {e}")

        # Combine scores
        if scores:
            combined = float(np.mean(scores))
            is_valid = combined >= self.min_similarity
            reason = "pass" if is_valid else f"low_similarity({combined:.3f})"
        else:
            combined = 0.0
            is_valid = True  # Pass if no models available
            reason = "no_models_available"

        return SemanticScore(
            laser_sim=laser_sim,
            labse_sim=labse_sim,
            combined=combined,
            is_valid=is_valid,
            reason=reason,
        )

    def score_batch(
        self, 
        mk_sentences: list[str], 
        sq_sentences: list[str]
    ) -> list[SemanticScore]:
        """
        Score a batch of sentence pairs efficiently.

        Args:
            mk_sentences: List of Macedonian sentences
            sq_sentences: List of Albanian sentences

        Returns:
            List of SemanticScore objects
        """
        n = len(mk_sentences)
        if n != len(sq_sentences):
            raise ValueError("Number of MK and SQ sentences must match")

        results = []
        laser_sims = [None] * n
        labse_sims = [None] * n

        # LASER batch encoding
        if self._laser_available and self._laser_mk and self._laser_sq:
            try:
                logger.info(f"Computing LASER embeddings for {n} pairs...")
                mk_embs = self._laser_mk.encode_sentences(mk_sentences)
                sq_embs = self._laser_sq.encode_sentences(sq_sentences)
                
                for i in range(n):
                    laser_sims[i] = self._cosine_similarity(mk_embs[i], sq_embs[i])
            except Exception as e:
                logger.warning(f"LASER batch encoding failed: {e}")

        # LaBSE batch encoding
        if self._labse_available and self._labse:
            try:
                logger.info(f"Computing LaBSE embeddings for {n} pairs...")
                mk_embs = self._labse.encode(mk_sentences, batch_size=self.batch_size)
                sq_embs = self._labse.encode(sq_sentences, batch_size=self.batch_size)
                
                for i in range(n):
                    labse_sims[i] = self._cosine_similarity(mk_embs[i], sq_embs[i])
            except Exception as e:
                logger.warning(f"LaBSE batch encoding failed: {e}")

        # Combine scores
        for i in range(n):
            scores = []
            if laser_sims[i] is not None:
                scores.append(laser_sims[i])
            if labse_sims[i] is not None:
                scores.append(labse_sims[i])

            if scores:
                combined = float(np.mean(scores))
                is_valid = combined >= self.min_similarity
                reason = "pass" if is_valid else f"low_similarity({combined:.3f})"
            else:
                combined = 0.0
                is_valid = True
                reason = "no_models_available"

            results.append(SemanticScore(
                laser_sim=laser_sims[i],
                labse_sim=labse_sims[i],
                combined=combined,
                is_valid=is_valid,
                reason=reason,
            ))

        return results

    def filter_pairs(
        self,
        pairs: list[dict],
        mk_key: str = "mk",
        sq_key: str = "sq",
        progress_callback=None,
    ) -> tuple[list[dict], int]:
        """
        Filter a list of pair dictionaries by semantic similarity.

        Args:
            pairs: List of dictionaries with MK and SQ keys
            mk_key: Key for Macedonian text in dict
            sq_key: Key for Albanian text in dict
            progress_callback: Optional callback(current, total)

        Returns:
            Tuple of (filtered_pairs, rejected_count)
        """
        n = len(pairs)
        if n == 0:
            return [], 0

        mk_sentences = [p[mk_key] for p in pairs]
        sq_sentences = [p[sq_key] for p in pairs]

        logger.info(f"Scoring {n} pairs semantically...")
        scores = self.score_batch(mk_sentences, sq_sentences)

        filtered = []
        rejected = 0
        
        for i, (pair, score) in enumerate(zip(pairs, scores)):
            if progress_callback and (i + 1) % 1000 == 0:
                progress_callback(i + 1, n)
            
            if score.is_valid:
                # Add score to metadata
                pair_copy = pair.copy()
                if "meta" not in pair_copy:
                    pair_copy["meta"] = {}
                pair_copy["meta"]["semantic_score"] = {
                    "laser": score.laser_sim,
                    "labse": score.labse_sim,
                    "combined": score.combined,
                }
                filtered.append(pair_copy)
            else:
                rejected += 1
                logger.debug(
                    f"Rejected pair (sim={score.combined:.3f}): "
                    f"MK='{mk_sentences[i][:50]}...' SQ='{sq_sentences[i][:50]}...'"
                )

        logger.info(
            f"Semantic filtering: {len(filtered)} passed, {rejected} rejected "
            f"(threshold={self.min_similarity})"
        )
        return filtered, rejected


# ───────────────────────── Helper function ──────────────────────


def quick_validate(mk: str, sq: str, threshold: float = 0.70) -> bool:
    """
    Quick validation of a single pair.
    
    Loads models lazily and caches them for subsequent calls.
    """
    global _cached_validator
    
    if "_cached_validator" not in globals():
        _cached_validator = SemanticValidator(min_similarity=threshold)
    
    score = _cached_validator.score_pair(mk, sq)
    return score.is_valid
