"""
Vezilka v2 — Gale-Church Statistical Aligner.

Strategy 3 (last resort): character-length DP over full sentence lists.
Used when: < 10 sentences on either side, or dense retrieval produces < 3
mutual nearest-neighbour pairs.

Tuned for MK↔SQ: mean ratio 1.1, variance 6.8.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Optional

from config import DEFAULT_CONFIG, VezilkaConfig

logger = logging.getLogger(__name__)

# Alignment type probabilities (must sum ≈ 1)
ALIGN_PROBS = {
    "1-1": 0.89,
    "1-0": 0.005,
    "0-1": 0.005,
    "1-2": 0.05,
    "2-1": 0.05,
    "2-2": 0.01,
}


@dataclass
class GCPair:
    mk: str
    sq: str
    mk_indices: list[int] = field(default_factory=list)
    sq_indices: list[int] = field(default_factory=list)
    alignment_type: str = "1-1"
    gc_cost: float = 0.0


@dataclass
class GaleChurchResult:
    pairs: list[GCPair] = field(default_factory=list)
    strategy: str = "gale_church"


class GaleChurchAligner:
    """Classic Gale-Church sentence aligner using character lengths."""

    def __init__(self, config: VezilkaConfig | None = None):
        self.cfg = config or DEFAULT_CONFIG
        self.mean_ratio = self.cfg.gc_mean_char_ratio
        self.variance = self.cfg.gc_variance

    def align(self, mk_text: str, sq_text: str) -> GaleChurchResult:
        mk_sents = self._sentence_split(mk_text)
        sq_sents = self._sentence_split(sq_text)
        if not mk_sents or not sq_sents:
            return GaleChurchResult()

        pairs = self._dp_align(mk_sents, sq_sents)
        logger.info("Gale-Church: %d MK + %d SQ → %d pairs",
                     len(mk_sents), len(sq_sents), len(pairs))
        return GaleChurchResult(pairs=pairs)

    # ── DP ──────────────────────────────────────────────────────

    def _dp_align(self, mk: list[str], sq: list[str]) -> list[GCPair]:
        n, m = len(mk), len(sq)
        INF = float("inf")
        cost = [[INF] * (m + 1) for _ in range(n + 1)]
        back = [[None] * (m + 1) for _ in range(n + 1)]
        cost[0][0] = 0.0

        MOVES = [
            (1, 1, "1-1"),
            (1, 0, "1-0"),
            (0, 1, "0-1"),
            (1, 2, "1-2"),
            (2, 1, "2-1"),
            (2, 2, "2-2"),
        ]

        for i in range(n + 1):
            for j in range(m + 1):
                if cost[i][j] == INF:
                    continue
                c0 = cost[i][j]
                for di, dj, atype in MOVES:
                    ni, nj = i + di, j + dj
                    if ni > n or nj > m:
                        continue
                    mk_len = sum(len(mk[i + k]) for k in range(di))
                    sq_len = sum(len(sq[j + k]) for k in range(dj))
                    c = c0 + self._match_cost(mk_len, sq_len, atype)
                    if c < cost[ni][nj]:
                        cost[ni][nj] = c
                        back[ni][nj] = (i, j, atype)

        # Trace back
        raw: list[tuple] = []
        ci, cj = n, m
        while ci > 0 or cj > 0:
            if back[ci][cj] is None:
                break
            pi, pj, atype = back[ci][cj]
            raw.append((pi, pj, ci, cj, atype))
            ci, cj = pi, pj

        pairs: list[GCPair] = []
        for pi, pj, ci, cj, atype in reversed(raw):
            di = ci - pi
            dj = cj - pj
            mk_text_merged = " ".join(mk[pi:ci]) if di > 0 else ""
            sq_text_merged = " ".join(sq[pj:cj]) if dj > 0 else ""
            mk_idx = list(range(pi, ci))
            sq_idx = list(range(pj, cj))

            if mk_text_merged.strip() and sq_text_merged.strip():
                pairs.append(GCPair(
                    mk=mk_text_merged, sq=sq_text_merged,
                    mk_indices=mk_idx, sq_indices=sq_idx,
                    alignment_type=atype,
                    gc_cost=cost[ci][cj],
                ))

        return pairs

    def _match_cost(self, mk_len: int, sq_len: int, atype: str) -> float:
        """Cost = -log(prior) + penalty(length ratio)."""
        prior = ALIGN_PROBS.get(atype, 0.001)
        penalty = self._length_penalty(mk_len, sq_len)
        return -math.log(prior + 1e-12) + penalty

    def _length_penalty(self, mk_len: int, sq_len: int) -> float:
        if sq_len == 0 and mk_len == 0:
            return 0.0
        if sq_len == 0:
            return 100.0
        r = mk_len / sq_len
        if r <= 0:
            return 100.0
        return (math.log(r / self.mean_ratio) ** 2) / self.variance

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
