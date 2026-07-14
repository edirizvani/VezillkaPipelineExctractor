"""
Vezilka v2 — Structural Aligner.

Strategy 1 (highest reliability): matches Член N ↔ Neni N by number,
then aligns sentences within each matched article pair using
dynamic-programming (1:1, 1:2, 2:1 merges).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Optional

from config import DEFAULT_CONFIG, VezilkaConfig

logger = logging.getLogger(__name__)


@dataclass
class AlignedPair:
    mk: str
    sq: str
    mk_indices: list[int] = field(default_factory=list)
    sq_indices: list[int] = field(default_factory=list)
    alignment_type: str = "1-1"
    article_number: Optional[int] = None
    dp_cost: float = 0.0


@dataclass
class StructuralAlignmentResult:
    pairs: list[AlignedPair] = field(default_factory=list)
    matched_articles: int = 0
    unmatched_mk_articles: list[int] = field(default_factory=list)
    unmatched_sq_articles: list[int] = field(default_factory=list)
    strategy: str = "structural"


class StructuralAligner:
    """Aligns MK↔SQ text via article-number matching + sentence DP."""

    def __init__(self, config: VezilkaConfig | None = None):
        self.cfg = config or DEFAULT_CONFIG
        self._mk_pat = re.compile(self.cfg.structural_article_pattern_mk, re.MULTILINE)
        self._sq_pat = re.compile(self.cfg.structural_article_pattern_sq, re.MULTILINE)

    def align(self, mk_text: str, sq_text: str) -> StructuralAlignmentResult:
        """Structural alignment: match articles, then sentence-DP inside each."""
        result = StructuralAlignmentResult()

        mk_articles = self._extract_articles(mk_text, self._mk_pat)
        sq_articles = self._extract_articles(sq_text, self._sq_pat)

        if not mk_articles or not sq_articles:
            return result   # no article numbers → caller should try next strategy

        mk_nums = set(mk_articles.keys())
        sq_nums = set(sq_articles.keys())
        matched = sorted(mk_nums & sq_nums)
        result.unmatched_mk_articles = sorted(mk_nums - sq_nums)
        result.unmatched_sq_articles = sorted(sq_nums - mk_nums)
        result.matched_articles = len(matched)

        for num in matched:
            mk_sents = self._sentence_split(mk_articles[num])
            sq_sents = self._sentence_split(sq_articles[num])
            if not mk_sents or not sq_sents:
                continue
            pairs = self._dp_align(mk_sents, sq_sents, article_num=num)
            result.pairs.extend(pairs)

        logger.info("Structural: %d articles matched → %d sentence pairs",
                     result.matched_articles, len(result.pairs))
        return result

    def has_articles(self, mk_text: str, sq_text: str) -> bool:
        """Quick check whether structural alignment is viable."""
        return bool(self._mk_pat.search(mk_text) and self._sq_pat.search(sq_text))

    # ── article extraction ──────────────────────────────────────

    @staticmethod
    def _extract_articles(text: str, pattern: re.Pattern) -> dict[int, str]:
        matches = list(pattern.finditer(text))
        if not matches:
            return {}
        result: dict[int, str] = {}
        for i, m in enumerate(matches):
            num = int(m.group(1))
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if len(body) >= 5:
                result[num] = body
        return result

    # ── sentence splitting ──────────────────────────────────────

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

    # ── DP alignment (Gale-Church-style within an article) ──────

    def _dp_align(
        self,
        mk_sents: list[str],
        sq_sents: list[str],
        article_num: int,
    ) -> list[AlignedPair]:
        """DP alignment with 1:1, 1:2, 2:1 merge support."""
        n, m = len(mk_sents), len(sq_sents)
        INF = float("inf")

        # cost[i][j] = min cost to align mk[:i] with sq[:j]
        cost = [[INF] * (m + 1) for _ in range(n + 1)]
        back = [[None] * (m + 1) for _ in range(n + 1)]
        cost[0][0] = 0.0

        mean_r = self.cfg.gc_mean_char_ratio
        var = self.cfg.gc_variance

        def _cost(mk_len: int, sq_len: int) -> float:
            if sq_len == 0:
                return INF if mk_len > 0 else 0.0
            r = mk_len / sq_len
            return (math.log(r / mean_r) ** 2) / var if r > 0 else INF

        for i in range(n + 1):
            for j in range(m + 1):
                if cost[i][j] == INF:
                    continue
                c0 = cost[i][j]

                # 1:1
                if i < n and j < m:
                    c = c0 + _cost(len(mk_sents[i]), len(sq_sents[j]))
                    if c < cost[i + 1][j + 1]:
                        cost[i + 1][j + 1] = c
                        back[i + 1][j + 1] = (i, j, "1-1")

                # 1:2  (one MK → two SQ)
                if i < n and j + 1 < m:
                    merged_sq = len(sq_sents[j]) + len(sq_sents[j + 1])
                    c = c0 + _cost(len(mk_sents[i]), merged_sq)
                    if c < cost[i + 1][j + 2]:
                        cost[i + 1][j + 2] = c
                        back[i + 1][j + 2] = (i, j, "1-2")

                # 2:1  (two MK → one SQ)
                if i + 1 < n and j < m:
                    merged_mk = len(mk_sents[i]) + len(mk_sents[i + 1])
                    c = c0 + _cost(merged_mk, len(sq_sents[j]))
                    if c < cost[i + 2][j + 1]:
                        cost[i + 2][j + 1] = c
                        back[i + 2][j + 1] = (i, j, "2-1")

        # Trace back
        pairs: list[AlignedPair] = []
        ci, cj = n, m
        raw_pairs: list[tuple] = []
        while ci > 0 or cj > 0:
            if back[ci][cj] is None:
                break
            pi, pj, atype = back[ci][cj]
            raw_pairs.append((pi, pj, ci, cj, atype))
            ci, cj = pi, pj

        for pi, pj, ci, cj, atype in reversed(raw_pairs):
            if atype == "1-1":
                mk_t = mk_sents[pi]
                sq_t = sq_sents[pj]
                mk_idx = [pi]
                sq_idx = [pj]
            elif atype == "1-2":
                mk_t = mk_sents[pi]
                sq_t = sq_sents[pj] + " " + sq_sents[pj + 1]
                mk_idx = [pi]
                sq_idx = [pj, pj + 1]
            elif atype == "2-1":
                mk_t = mk_sents[pi] + " " + mk_sents[pi + 1]
                sq_t = sq_sents[pj]
                mk_idx = [pi, pi + 1]
                sq_idx = [pj]
            else:
                continue

            pairs.append(AlignedPair(
                mk=mk_t, sq=sq_t,
                mk_indices=mk_idx, sq_indices=sq_idx,
                alignment_type=atype,
                article_number=article_num,
                dp_cost=cost[ci][cj],
            ))

        return pairs
