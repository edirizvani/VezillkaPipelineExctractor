"""
Vezilka v2 — Language Detector.

Three-layer detection: Unicode script → lingua ML → langdetect fallback.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class DetectedLanguage(Enum):
    MACEDONIAN = "mk"
    ALBANIAN = "sq"
    SERBIAN = "sr"
    UNKNOWN = "unknown"


class LanguageDetector:
    """Multi-strategy language detector for MK and SQ text."""

    def __init__(self, cyrillic_threshold: float = 0.60, latin_threshold: float = 0.60):
        self.cyrillic_threshold = cyrillic_threshold
        self.latin_threshold = latin_threshold
        self._lingua = None

    def _get_lingua(self):
        if self._lingua is None:
            try:
                from lingua import Language, LanguageDetectorBuilder
                self._lingua = (
                    LanguageDetectorBuilder
                    .from_languages(Language.MACEDONIAN, Language.ALBANIAN, Language.SERBIAN)
                    .with_minimum_relative_distance(0.25)
                    .build()
                )
            except ImportError:
                logger.warning("lingua not installed; script-only detection")
                self._lingua = False
        return self._lingua if self._lingua is not False else None

    def detect(self, text: str) -> DetectedLanguage:
        if not text or len(text.strip()) < 3:
            return DetectedLanguage.UNKNOWN

        # Layer 1: Script analysis
        alpha = [c for c in text if c.isalpha()]
        if len(alpha) >= 5:
            cyr = sum(1 for c in alpha if "\u0400" <= c <= "\u04FF") / len(alpha)
            if cyr >= self.cyrillic_threshold:
                return DetectedLanguage.MACEDONIAN
            lat = sum(1 for c in alpha if c.isascii()) / len(alpha)
            if lat >= self.latin_threshold:
                return DetectedLanguage.ALBANIAN

        # Layer 2: lingua
        det = self._get_lingua()
        if det:
            try:
                from lingua import Language
                r = det.detect_language_of(text)
                if r == Language.MACEDONIAN:
                    return DetectedLanguage.MACEDONIAN
                if r == Language.ALBANIAN:
                    return DetectedLanguage.ALBANIAN
                if r == Language.SERBIAN:
                    return DetectedLanguage.SERBIAN
            except Exception:
                pass

        # Layer 3: langdetect
        try:
            from langdetect import detect
            code = detect(text)
            mapping = {"mk": DetectedLanguage.MACEDONIAN, "sq": DetectedLanguage.ALBANIAN,
                       "sr": DetectedLanguage.SERBIAN}
            if code in mapping:
                return mapping[code]
        except Exception:
            pass

        return DetectedLanguage.UNKNOWN
