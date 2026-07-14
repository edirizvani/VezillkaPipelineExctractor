"""
Vezilka v2 — Noise Filter.

Rule-based filters applied to raw extracted text BEFORE alignment.
Removes gazette boiler-plate, page numbers, header/footer lines,
and ensures script purity (MK = Cyrillic, SQ = Latin).
"""

from __future__ import annotations

import logging
import re

from config import DEFAULT_CONFIG, VezilkaConfig
from utils.text_utils import (
    cyrillic_ratio,
    digit_fraction,
    has_albanian_markers,
    is_noise_line,
    latin_ratio,
    strip_headers_footers,
    word_count,
)

logger = logging.getLogger(__name__)


class NoiseFilter:
    """Pre-alignment noise removal for raw text blocks."""

    def __init__(self, config: VezilkaConfig | None = None):
        self.cfg = config or DEFAULT_CONFIG

    def clean_mk(self, text: str) -> str:
        """Clean and validate a Macedonian text block."""
        text = strip_headers_footers(text)
        lines = text.split("\n")
        cleaned = [l for l in lines if not is_noise_line(l)]
        text = "\n".join(cleaned)

        # Script contamination check
        cyr = cyrillic_ratio(text)
        if cyr < self.cfg.mk_min_cyrillic and len(text.strip()) > 20:
            logger.debug("MK block has low Cyrillic ratio %.2f — discarding", cyr)
            return ""
        return text.strip()

    def clean_sq(self, text: str) -> str:
        """Clean and validate an Albanian text block."""
        text = strip_headers_footers(text)
        lines = text.split("\n")
        cleaned = [l for l in lines if not is_noise_line(l)]
        text = "\n".join(cleaned)

        lat = latin_ratio(text)
        if lat < self.cfg.sq_min_latin and len(text.strip()) > 20:
            logger.debug("SQ block has low Latin ratio %.2f — discarding", lat)
            return ""
        return text.strip()

    def is_content_line(self, line: str) -> bool:
        """Return True if the line has enough actual content."""
        line = line.strip()
        if not line:
            return False
        if len(line) < 3:
            return False
        if digit_fraction(line) > self.cfg.max_digit_fraction:
            return False
        if is_noise_line(line):
            return False
        return True
