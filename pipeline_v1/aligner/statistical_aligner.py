"""
Statistical Aligner — Gale-Church (1993) sentence alignment.

Fallback aligner for cases where structural (article-number) alignment
cannot be applied, e.g. preambles, annexes, or free-form text blocks.
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure the project root is on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# ── Gale-Church alignment-type priors ───────────────────────────
_ALIGN_TYPES = {
    "1-1": {"prior": 0.89},
    "1-0": {"prior": 0.005},
    "0-1": {"prior": 0.005},
    "1-2": {"prior": 0.04},
    "2-1": {"prior": 0.04},
    "2-2": {"prior": 0.02},
}


@dataclass
class GaleChurchPair:
    """A pair (or group) produced by Gale-Church alignment."""
    mk: str
    sq: str
    align_type: str      # "1-1", "1-2", "2-1", etc.
    score: float         # distance score (lower = better)


class GaleChurchAligner:
    """
    Gale-Church (1993) sentence alignment using sentence lengths
    (in characters).

    Tuned for Macedonian ↔ Albanian where Albanian is on average
    ~10 % longer than Macedonian.

    Usage::

        aligner = GaleChurchAligner()
        pairs = aligner.align(mk_sentences, sq_sentences)
    """

    def __init__(
        self,
        mean_char_ratio: float = 1.1,
        variance: float = 6.8,
    ):
        self.mean_char_ratio = mean_char_ratio
        self.variance = variance

    # ── Main alignment ──────────────────────────────────────────

    def align(
        self,
        mk_sentences: list[str],
        sq_sentences: list[str],
    ) -> list[GaleChurchPair]:
        n, m = len(mk_sentences), len(sq_sentences)
        if n == 0 or m == 0:
            return []

        mk_lens = [len(s) for s in mk_sentences]
        sq_lens = [len(s) for s in sq_sentences]

        INF = float("inf")
        dp = [[INF] * (m + 1) for _ in range(n + 1)]
        bt = [[None] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0.0

        for i in range(n + 1):
            for j in range(m + 1):
                if dp[i][j] == INF:
                    continue

                # 1-1
                if i < n and j < m:
                    cost = self._distance(mk_lens[i], sq_lens[j])
                    penalty = -math.log(_ALIGN_TYPES["1-1"]["prior"])
                    total = dp[i][j] + cost + penalty
                    if total < dp[i + 1][j + 1]:
                        dp[i + 1][j + 1] = total
                        bt[i + 1][j + 1] = (i, j, "1-1")
                # 1-0
                if i < n:
                    penalty = -math.log(_ALIGN_TYPES["1-0"]["prior"])
                    total = dp[i][j] + penalty
                    if total < dp[i + 1][j]:
                        dp[i + 1][j] = total
                        bt[i + 1][j] = (i, j, "1-0")
                # 0-1
                if j < m:
                    penalty = -math.log(_ALIGN_TYPES["0-1"]["prior"])
                    total = dp[i][j] + penalty
                    if total < dp[i][j + 1]:
                        dp[i][j + 1] = total
                        bt[i][j + 1] = (i, j, "0-1")
                # 1-2
                if i < n and j + 1 < m:
                    cost = self._distance(mk_lens[i], sq_lens[j] + sq_lens[j + 1])
                    penalty = -math.log(_ALIGN_TYPES["1-2"]["prior"])
                    total = dp[i][j] + cost + penalty
                    if total < dp[i + 1][j + 2]:
                        dp[i + 1][j + 2] = total
                        bt[i + 1][j + 2] = (i, j, "1-2")
                # 2-1
                if i + 1 < n and j < m:
                    cost = self._distance(mk_lens[i] + mk_lens[i + 1], sq_lens[j])
                    penalty = -math.log(_ALIGN_TYPES["2-1"]["prior"])
                    total = dp[i][j] + cost + penalty
                    if total < dp[i + 2][j + 1]:
                        dp[i + 2][j + 1] = total
                        bt[i + 2][j + 1] = (i, j, "2-1")
                # 2-2
                if i + 1 < n and j + 1 < m:
                    cost = self._distance(
                        mk_lens[i] + mk_lens[i + 1],
                        sq_lens[j] + sq_lens[j + 1],
                    )
                    penalty = -math.log(_ALIGN_TYPES["2-2"]["prior"])
                    total = dp[i][j] + cost + penalty
                    if total < dp[i + 2][j + 2]:
                        dp[i + 2][j + 2] = total
                        bt[i + 2][j + 2] = (i, j, "2-2")

        # Backtrack
        path: list[tuple[int, int, str]] = []
        ci, cj = n, m
        while ci > 0 or cj > 0:
            if bt[ci][cj] is None:
                break
            pi, pj, atype = bt[ci][cj]
            path.append((pi, pj, atype))
            ci, cj = pi, pj
        path.reverse()

        # Build output pairs (skip 1-0 and 0-1 — they are deletions)
        pairs: list[GaleChurchPair] = []
        for pi, pj, atype in path:
            if atype == "1-1":
                pairs.append(GaleChurchPair(
                    mk=mk_sentences[pi], sq=sq_sentences[pj],
                    align_type="1-1",
                    score=self._distance(mk_lens[pi], sq_lens[pj]),
                ))
            elif atype == "1-2":
                pairs.append(GaleChurchPair(
                    mk=mk_sentences[pi],
                    sq=sq_sentences[pj] + " " + sq_sentences[pj + 1],
                    align_type="1-2",
                    score=self._distance(mk_lens[pi], sq_lens[pj] + sq_lens[pj + 1]),
                ))
            elif atype == "2-1":
                pairs.append(GaleChurchPair(
                    mk=mk_sentences[pi] + " " + mk_sentences[pi + 1],
                    sq=sq_sentences[pj],
                    align_type="2-1",
                    score=self._distance(mk_lens[pi] + mk_lens[pi + 1], sq_lens[pj]),
                ))
            elif atype == "2-2":
                pairs.append(GaleChurchPair(
                    mk=mk_sentences[pi] + " " + mk_sentences[pi + 1],
                    sq=sq_sentences[pj] + " " + sq_sentences[pj + 1],
                    align_type="2-2",
                    score=self._distance(
                        mk_lens[pi] + mk_lens[pi + 1],
                        sq_lens[pj] + sq_lens[pj + 1],
                    ),
                ))

        logger.info(
            "Gale-Church alignment: %d pairs from %d×%d sentences",
            len(pairs), n, m,
        )
        return pairs

    # ── Gale-Church distance ────────────────────────────────────

    def _distance(self, len_mk: int, len_sq: int) -> float:
        if len_mk == 0 and len_sq == 0:
            return 0.0
        mean = (len_mk + len_sq / self.mean_char_ratio) / 2.0
        if mean == 0:
            return float("inf")
        delta = (len_mk - len_sq / self.mean_char_ratio) / math.sqrt(
            mean * self.variance
        )
        return abs(delta) + math.log(2.0)

    # ── Convert to unified SentencePair format ──────────────────

    @staticmethod
    def to_sentence_pairs(
        gc_pairs: list[GaleChurchPair], article_id: str = "gc"
    ) -> list:
        """Convert to the same ``SentencePair`` format used by StructuralAligner."""
        from aligner.structural_aligner import SentencePair
        return [
            SentencePair(
                mk=p.mk, sq=p.sq,
                article_id=article_id,
                confidence=max(0.0, 1.0 - p.score / 10.0),
                alignment_method=f"gale_church_{p.align_type}",
            )
            for p in gc_pairs
        ]
