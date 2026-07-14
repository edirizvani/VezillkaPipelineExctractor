"""
Vezilka v2 — Dense Retrieval Aligner.

Strategy 2 (medium reliability): for unstructured PDFs with no
article numbers.  Uses LASER3 embeddings + FAISS index for mutual
nearest-neighbour matching with a monotonicity constraint.

Algorithm:
  1. Sentence-split both MK and SQ text
  2. Encode MK sentences → LASER3(mkd), SQ sentences → LASER3(sq)
  3. Build FAISS index on SQ embeddings
  4. For each MK sent, find nearest SQ (forward)
  5. Build FAISS index on MK embeddings
  6. For each SQ sent, find nearest MK (backward)
  7. Keep only mutual nearest pairs above threshold
  8. Enforce monotonicity via longest-increasing-subsequence DP
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

from config import DEFAULT_CONFIG, VezilkaConfig

logger = logging.getLogger(__name__)


@dataclass
class DenseAlignedPair:
    mk: str
    sq: str
    mk_index: int = 0
    sq_index: int = 0
    similarity: float = 0.0
    alignment_type: str = "dense_retrieval"


@dataclass
class DenseRetrievalResult:
    pairs: list[DenseAlignedPair] = field(default_factory=list)
    total_mk_sentences: int = 0
    total_sq_sentences: int = 0
    mutual_nn_before_monotone: int = 0
    strategy: str = "dense_retrieval"


class DenseRetrievalAligner:
    """Align MK↔SQ via LASER3 mutual nearest-neighbour + monotonicity."""

    def __init__(self, config: VezilkaConfig | None = None):
        self.cfg = config or DEFAULT_CONFIG
        self._mk_encoder = None
        self._sq_encoder = None

    def align(self, mk_text: str, sq_text: str) -> DenseRetrievalResult:
        """
        Full dense retrieval pipeline.

        Returns list of (mk_index, sq_index, similarity) after mutual-NN
        filtering and monotonicity enforcement.
        """
        result = DenseRetrievalResult()

        mk_sents = self._sentence_split(mk_text)
        sq_sents = self._sentence_split(sq_text)
        result.total_mk_sentences = len(mk_sents)
        result.total_sq_sentences = len(sq_sents)

        if len(mk_sents) < 2 or len(sq_sents) < 2:
            return result

        mk_enc, sq_enc = self._get_encoders()
        if mk_enc is None:
            logger.warning("LASER3 not available — dense retrieval disabled")
            return result

        try:
            mk_embs = mk_enc.encode_sentences(mk_sents)
            sq_embs = sq_enc.encode_sentences(sq_sents)
        except Exception as e:
            logger.error("LASER3 encoding failed: %s", e)
            return result

        mk_embs = np.asarray(mk_embs, dtype=np.float32)
        sq_embs = np.asarray(sq_embs, dtype=np.float32)

        # Normalise for cosine similarity
        mk_embs = mk_embs / (np.linalg.norm(mk_embs, axis=1, keepdims=True) + 1e-12)
        sq_embs = sq_embs / (np.linalg.norm(sq_embs, axis=1, keepdims=True) + 1e-12)

        # Forward: for each MK find nearest SQ
        fwd_idx, fwd_sim = self._nearest_neighbours(mk_embs, sq_embs)
        # Backward: for each SQ find nearest MK
        bwd_idx, bwd_sim = self._nearest_neighbours(sq_embs, mk_embs)

        # Mutual nearest neighbours
        min_sim = self.cfg.dense_retrieval_min_similarity
        candidates: list[tuple[int, int, float]] = []

        for i in range(len(mk_sents)):
            j = int(fwd_idx[i])
            if int(bwd_idx[j]) == i:      # mutual NN
                sim = float(fwd_sim[i])
                if sim >= min_sim:
                    candidates.append((i, j, sim))

        result.mutual_nn_before_monotone = len(candidates)

        # Monotonicity constraint
        if self.cfg.dense_retrieval_monotonicity and candidates:
            candidates = self._enforce_monotonicity(candidates)

        for i, j, sim in candidates:
            result.pairs.append(DenseAlignedPair(
                mk=mk_sents[i], sq=sq_sents[j],
                mk_index=i, sq_index=j,
                similarity=sim,
            ))

        logger.info(
            "Dense retrieval: %d MK + %d SQ → %d mutual NN → %d monotone pairs",
            len(mk_sents), len(sq_sents),
            result.mutual_nn_before_monotone, len(result.pairs),
        )
        return result

    # ── nearest neighbours ──────────────────────────────────────

    def _nearest_neighbours(
        self,
        queries: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, similarities) of nearest target for each query."""
        if self.cfg.dense_retrieval_use_faiss:
            return self._faiss_nn(queries, targets)
        return self._brute_nn(queries, targets)

    @staticmethod
    def _faiss_nn(
        queries: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """FAISS inner-product search (vectors must be L2-normalised)."""
        try:
            import faiss
            dim = targets.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(targets)
            sims, idxs = index.search(queries, 1)
            return idxs.flatten(), sims.flatten()
        except ImportError:
            logger.warning("faiss not installed — using brute-force NN")
            return DenseRetrievalAligner._brute_nn(queries, targets)

    @staticmethod
    def _brute_nn(
        queries: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Brute-force cosine NN via matrix multiplication."""
        sim_matrix = queries @ targets.T          # (Q, T)
        idxs = np.argmax(sim_matrix, axis=1)      # best target per query
        sims = sim_matrix[np.arange(len(queries)), idxs]
        return idxs, sims

    # ── monotonicity enforcement ────────────────────────────────

    @staticmethod
    def _enforce_monotonicity(
        candidates: list[tuple[int, int, float]],
    ) -> list[tuple[int, int, float]]:
        """
        Find the maximum-weight monotone subset.

        A valid alignment is monotone if for pairs (i1,j1) and (i2,j2):
            i1 < i2  ⟹  j1 < j2

        We solve this as a longest-increasing-subsequence (LIS) on j-values,
        weighted by similarity score, using DP.
        """
        if not candidates:
            return []

        # Sort by mk_index first
        candidates.sort(key=lambda t: (t[0], t[1]))

        n = len(candidates)
        # dp[k] = max total similarity for a monotone sequence ending at k
        dp = [0.0] * n
        prev = [-1] * n

        for k in range(n):
            dp[k] = candidates[k][2]   # just this pair
            prev[k] = -1
            for p in range(k):
                if candidates[p][1] < candidates[k][1]:  # monotone j
                    score = dp[p] + candidates[k][2]
                    if score > dp[k]:
                        dp[k] = score
                        prev[k] = p

        # Trace back from best endpoint
        best_end = int(np.argmax(dp))
        chain: list[int] = []
        idx = best_end
        while idx >= 0:
            chain.append(idx)
            idx = prev[idx]
        chain.reverse()

        return [candidates[k] for k in chain]

    # ── encoders ────────────────────────────────────────────────

    def _get_encoders(self):
        if self._mk_encoder is None:
            try:
                from laser_encoders import LaserEncoderPipeline
                self._mk_encoder = LaserEncoderPipeline(lang="mkd")
                self._sq_encoder = LaserEncoderPipeline(lang="sq")
            except ImportError:
                logger.warning("laser-encoders not installed")
                self._mk_encoder = False
                self._sq_encoder = False
            except Exception as e:
                logger.error("LASER3 init failed: %s", e)
                self._mk_encoder = False
                self._sq_encoder = False
        if self._mk_encoder is False:
            return None, None
        return self._mk_encoder, self._sq_encoder

    @staticmethod
    def _sentence_split(text: str) -> list[str]:
        if not text.strip():
            return []
        try:
            from sentence_splitter import SentenceSplitter
            sp = SentenceSplitter(language="en")
            return [s.strip() for s in sp.split(text) if s.strip()]
        except ImportError:
            sents = re.split(r"(?<=[.!?;])\s+", text)
            return [s.strip() for s in sents if s.strip()]
