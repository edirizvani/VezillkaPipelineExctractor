"""
Structural Aligner — aligns Macedonian ↔ Albanian text using legal article numbers.

Primary alignment strategy.  Legal texts in the Služben Vesnik always use
numbered articles (Член / Neni) identical in both languages.  We exploit this
to produce high-confidence parallel segments, then refine to sentence level.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ───────────────────────── Data classes ─────────────────────────

@dataclass
class ArticleSegment:
    """One numbered article (or sub-section) in a single language."""
    article_id: str          # "1", "24a", etc.
    text: str
    start: int               # Character offset in source text
    end: int
    heading: str = ""        # Original heading text (e.g. "Член 1")


@dataclass
class AlignedPair:
    """A matched pair of MK ↔ SQ text segments."""
    mk: str
    sq: str
    article_id: str
    confidence: float = 1.0
    alignment_method: str = "structural"


@dataclass
class SentencePair:
    """A single sentence pair."""
    mk: str
    sq: str
    article_id: str
    confidence: float = 1.0
    alignment_method: str = "structural"


# ───────────────────────── Regex patterns ───────────────────────

# Macedonian article patterns
MK_ARTICLE_PATTERNS = [
    r"Член\s+(\d+[а-яА-Яa-zA-Z]?)",      # Член 1, Член 24а
    r"Чл\.\s*(\d+)",                        # Чл. 5
    r"ЧЛЕН\s+(\d+)",                         # ЧЛЕН 1 (uppercase)
]

MK_SUBSECTION_PATTERNS = [
    r"точка\s+(\d+)",                        # точка 5
    r"Ставот?\s+(\d+)",                      # Ставот 2
]

# Albanian article patterns
SQ_ARTICLE_PATTERNS = [
    r"Neni\s+(\d+[a-zA-Zа-яА-Я]?)",        # Neni 1, Neni 24a
    r"NENI\s+(\d+)",                         # NENI 1 (uppercase)
]

SQ_SUBSECTION_PATTERNS = [
    r"Pika\s+(\d+)",                         # Pika 5 (point)
    r"Paragrafi\s+(\d+)",                    # Paragrafi 2
    r"Shkronja\s+([a-zA-Z])",               # Shkronja a
]

# Pre-compiled combined patterns
_MK_ARTICLE_RE = re.compile("|".join(f"(?:{p})" for p in MK_ARTICLE_PATTERNS))
_SQ_ARTICLE_RE = re.compile("|".join(f"(?:{p})" for p in SQ_ARTICLE_PATTERNS))


# ───────────────────────── Structural Aligner ───────────────────

class StructuralAligner:
    """
    Aligns Macedonian ↔ Albanian text by matching article numbers.

    Usage::

        aligner = StructuralAligner()
        pairs = aligner.align(mk_text, sq_text)
        sentences = aligner.align_sentences(pairs)
    """

    def __init__(
        self,
        min_article_len: int = 10,
        min_sentence_words: int = 5,
        max_sentence_words: int = 200,
    ):
        self.min_article_len = min_article_len
        self.min_sentence_words = min_sentence_words
        self.max_sentence_words = max_sentence_words

    # ── Article-level extraction ────────────────────────────────

    def extract_article_segments(
        self, text: str, language: str
    ) -> list[ArticleSegment]:
        """
        Split *text* into segments at article boundaries.

        Parameters
        ----------
        text : str
            Full-document text in one language.
        language : str
            ``"mk"`` or ``"sq"``.
        """
        if language == "mk":
            pattern = _MK_ARTICLE_RE
        elif language == "sq":
            pattern = _SQ_ARTICLE_RE
        else:
            raise ValueError(f"Unsupported language: {language}")

        segments: list[ArticleSegment] = []
        matches = list(pattern.finditer(text))

        if not matches:
            if text.strip():
                segments.append(
                    ArticleSegment(
                        article_id="0", text=text.strip(),
                        start=0, end=len(text), heading="",
                    )
                )
            return segments

        for i, match in enumerate(matches):
            article_id = next(
                (g for g in match.groups() if g is not None), "0"
            ).strip()

            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            heading = match.group(0)
            body = text[match.end():end].strip()

            if len(body) >= self.min_article_len:
                segments.append(
                    ArticleSegment(
                        article_id=article_id,
                        text=body,
                        start=start,
                        end=end,
                        heading=heading,
                    )
                )

        logger.debug("Extracted %d %s article segments", len(segments), language)
        return segments

    # ── Article-level alignment ─────────────────────────────────

    def align_by_articles(
        self,
        mk_segments: list[ArticleSegment],
        sq_segments: list[ArticleSegment],
    ) -> list[AlignedPair]:
        """Match MK and SQ segments that share the same ``article_id``."""
        mk_map = {seg.article_id: seg for seg in mk_segments}
        sq_map = {seg.article_id: seg for seg in sq_segments}

        all_ids = sorted(
            set(mk_map) | set(sq_map), key=lambda x: self._sort_key(x)
        )

        pairs: list[AlignedPair] = []
        unmatched_mk: list[str] = []
        unmatched_sq: list[str] = []

        for aid in all_ids:
            mk_seg = mk_map.get(aid)
            sq_seg = sq_map.get(aid)
            if mk_seg and sq_seg:
                pairs.append(
                    AlignedPair(
                        mk=mk_seg.text, sq=sq_seg.text,
                        article_id=aid, confidence=1.0,
                        alignment_method="structural",
                    )
                )
            elif mk_seg:
                unmatched_mk.append(aid)
            elif sq_seg:
                unmatched_sq.append(aid)

        if unmatched_mk:
            logger.warning("MK articles without SQ match: %s", unmatched_mk)
        if unmatched_sq:
            logger.warning("SQ articles without MK match: %s", unmatched_sq)

        logger.info(
            "Article alignment: %d pairs, %d unmatched MK, %d unmatched SQ",
            len(pairs), len(unmatched_mk), len(unmatched_sq),
        )
        return pairs

    # ── High-level convenience ──────────────────────────────────

    def align(self, mk_text: str, sq_text: str) -> list[AlignedPair]:
        """End-to-end article-level alignment of MK and SQ text."""
        mk_segments = self.extract_article_segments(mk_text, "mk")
        sq_segments = self.extract_article_segments(sq_text, "sq")
        return self.align_by_articles(mk_segments, sq_segments)

    # ── Sentence-level alignment ────────────────────────────────

    def align_sentences(
        self, article_pairs: list[AlignedPair]
    ) -> list[SentencePair]:
        """
        For each article-level pair, split into sentences and align
        using length-ratio heuristics.
        """
        all_pairs: list[SentencePair] = []
        for pair in article_pairs:
            mk_sents = self._split_sentences(pair.mk, "mk")
            sq_sents = self._split_sentences(pair.sq, "sq")
            aligned = self._align_sentence_lists(
                mk_sents, sq_sents, pair.article_id
            )
            all_pairs.extend(aligned)

        logger.info(
            "Sentence alignment: %d pairs from %d articles",
            len(all_pairs), len(article_pairs),
        )
        return all_pairs

    # ── Sentence splitting ──────────────────────────────────────

    def _split_sentences(self, text: str, lang: str) -> list[str]:
        """Split text into sentences, filtering by word count."""
        try:
            from sentence_splitter import SentenceSplitter
            # No native MK/SQ tokeniser — use English rules as a fallback
            splitter = SentenceSplitter(language="en")
            sents = splitter.split(text)
        except Exception:
            sents = re.split(r"(?<=[.!?])\s+", text)

        return [
            s.strip() for s in sents
            if s.strip() and len(s.strip().split()) >= self.min_sentence_words
        ]

    def _align_sentence_lists(
        self,
        mk_sents: list[str],
        sq_sents: list[str],
        article_id: str,
    ) -> list[SentencePair]:
        if not mk_sents or not sq_sents:
            return []

        pairs: list[SentencePair] = []
        n_mk, n_sq = len(mk_sents), len(sq_sents)

        if n_mk == n_sq:
            # Perfect 1:1
            for ms, ss in zip(mk_sents, sq_sents):
                pairs.append(
                    SentencePair(
                        mk=ms, sq=ss, article_id=article_id,
                        confidence=0.9, alignment_method="structural_1to1",
                    )
                )
        elif abs(n_mk - n_sq) <= 2:
            pairs = self._dp_align(mk_sents, sq_sents, article_id)
        else:
            # Counts differ too much — align what we can
            for i in range(min(n_mk, n_sq)):
                pairs.append(
                    SentencePair(
                        mk=mk_sents[i], sq=sq_sents[i],
                        article_id=article_id,
                        confidence=0.5,
                        alignment_method="structural_truncated",
                    )
                )

        return pairs

    # ── DP alignment for near-equal sentence counts ─────────────

    def _dp_align(
        self,
        mk_sents: list[str],
        sq_sents: list[str],
        article_id: str,
    ) -> list[SentencePair]:
        """DP alignment allowing 1:1, 1:2, and 2:1 merges."""
        n, m = len(mk_sents), len(sq_sents)
        INF = float("inf")

        dp = [[INF] * (m + 1) for _ in range(n + 1)]
        bt = [[None] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0.0

        for i in range(n + 1):
            for j in range(m + 1):
                if dp[i][j] == INF:
                    continue
                # 1:1
                if i < n and j < m:
                    cost = self._len_ratio_cost(mk_sents[i], sq_sents[j])
                    if dp[i][j] + cost < dp[i + 1][j + 1]:
                        dp[i + 1][j + 1] = dp[i][j] + cost
                        bt[i + 1][j + 1] = (i, j, "1:1")
                # 1:2
                if i < n and j + 1 < m:
                    merged = sq_sents[j] + " " + sq_sents[j + 1]
                    cost = self._len_ratio_cost(mk_sents[i], merged)
                    if dp[i][j] + cost < dp[i + 1][j + 2]:
                        dp[i + 1][j + 2] = dp[i][j] + cost
                        bt[i + 1][j + 2] = (i, j, "1:2")
                # 2:1
                if i + 1 < n and j < m:
                    merged = mk_sents[i] + " " + mk_sents[i + 1]
                    cost = self._len_ratio_cost(merged, sq_sents[j])
                    if dp[i][j] + cost < dp[i + 2][j + 1]:
                        dp[i + 2][j + 1] = dp[i][j] + cost
                        bt[i + 2][j + 1] = (i, j, "2:1")

        # Backtrack
        alignments: list[tuple[int, int, str]] = []
        ci, cj = n, m
        while ci > 0 or cj > 0:
            if bt[ci][cj] is None:
                break
            pi, pj, atype = bt[ci][cj]
            alignments.append((pi, pj, atype))
            ci, cj = pi, pj
        alignments.reverse()

        pairs: list[SentencePair] = []
        for pi, pj, atype in alignments:
            if atype == "1:1":
                pairs.append(SentencePair(
                    mk=mk_sents[pi], sq=sq_sents[pj],
                    article_id=article_id, confidence=0.85,
                    alignment_method="structural_dp_1to1",
                ))
            elif atype == "1:2":
                pairs.append(SentencePair(
                    mk=mk_sents[pi],
                    sq=sq_sents[pj] + " " + sq_sents[pj + 1],
                    article_id=article_id, confidence=0.75,
                    alignment_method="structural_dp_1to2",
                ))
            elif atype == "2:1":
                pairs.append(SentencePair(
                    mk=mk_sents[pi] + " " + mk_sents[pi + 1],
                    sq=sq_sents[pj],
                    article_id=article_id, confidence=0.75,
                    alignment_method="structural_dp_2to1",
                ))
        return pairs

    @staticmethod
    def _len_ratio_cost(s1: str, s2: str) -> float:
        l1, l2 = max(len(s1), 1), max(len(s2), 1)
        return (math.log(l1 / l2)) ** 2

    @staticmethod
    def _sort_key(article_id: str) -> tuple:
        num = re.match(r"(\d+)(.*)", article_id)
        if num:
            return (int(num.group(1)), num.group(2))
        return (999999, article_id)
