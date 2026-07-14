"""
Vezilka v2 — Pair Filter.

Post-alignment heuristic filters applied to candidate pairs AFTER
semantic validation.  Catches length outliers, digit-heavy pairs,
and script contamination at the pair level.
"""

from __future__ import annotations

import logging

from config import DEFAULT_CONFIG, VezilkaConfig
from phase4_align.aligner_orchestrator import CandidatePair
from utils.text_utils import cyrillic_ratio, digit_fraction, latin_ratio, word_count

logger = logging.getLogger(__name__)


class PairFilter:
    """Heuristic post-validation filter for candidate pairs."""

    def __init__(self, config: VezilkaConfig | None = None):
        self.cfg = config or DEFAULT_CONFIG

    def filter(self, pairs: list[CandidatePair]) -> list[CandidatePair]:
        """Return only pairs that pass all heuristic checks."""
        kept, dropped = [], 0
        for p in pairs:
            reason = self._check(p)
            if reason:
                p.rejection_reason = reason
                dropped += 1
            else:
                kept.append(p)
        logger.info("PairFilter: %d kept, %d dropped", len(kept), dropped)
        return kept

    def _check(self, p: CandidatePair) -> str:
        # Already rejected upstream
        if p.rejection_reason:
            return p.rejection_reason

        # Word count bounds
        mw = word_count(p.mk)
        sw = word_count(p.sq)
        if mw < self.cfg.min_words or sw < self.cfg.min_words:
            return f"too_short_mk{mw}_sq{sw}"
        if mw > self.cfg.max_words or sw > self.cfg.max_words:
            return f"too_long_mk{mw}_sq{sw}"

        # Length ratio
        ratio = mw / max(sw, 1)
        if ratio < self.cfg.min_length_ratio or ratio > self.cfg.max_length_ratio:
            return f"length_ratio_{ratio:.2f}"

        # Digit fraction
        if digit_fraction(p.mk) > self.cfg.max_digit_fraction:
            return f"digit_heavy_mk_{digit_fraction(p.mk):.2f}"
        if digit_fraction(p.sq) > self.cfg.max_digit_fraction:
            return f"digit_heavy_sq_{digit_fraction(p.sq):.2f}"

        # Script purity
        if cyrillic_ratio(p.mk) < self.cfg.mk_min_cyrillic and len(p.mk) > 20:
            return f"mk_low_cyrillic_{cyrillic_ratio(p.mk):.2f}"
        if latin_ratio(p.sq) < self.cfg.sq_min_latin and len(p.sq) > 20:
            return f"sq_low_latin_{latin_ratio(p.sq):.2f}"

        return ""
