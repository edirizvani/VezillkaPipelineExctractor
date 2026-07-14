"""
Language Detector — identifies Macedonian vs Albanian text blocks.

Uses a combination of Unicode script analysis (fast) and the
``lingua-language-detector`` library (accurate even for short texts).
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Lazy singleton for lingua detector ──────────────────────────
_lingua_detector = None


def _get_lingua_detector():
    """Build the lingua detector once and cache it."""
    global _lingua_detector
    if _lingua_detector is None:
        try:
            from lingua import Language, LanguageDetectorBuilder
            _lingua_detector = (
                LanguageDetectorBuilder.from_languages(
                    Language.MACEDONIAN,
                    Language.ALBANIAN,
                    Language.SERBIAN,
                )
                .with_preloaded_language_models()
                .build()
            )
        except ImportError:
            logger.warning(
                "lingua-language-detector not installed; falling back to langdetect"
            )
    return _lingua_detector


# ───────────────────────── Data classes ─────────────────────────

@dataclass
class LanguageResult:
    language: str          # "mk" | "sq" | "sr" | "unknown"
    confidence: float      # 0.0 – 1.0


@dataclass
class TextSegment:
    text: str
    language: str          # "mk" | "sq" | "unknown"
    start_char: int
    end_char: int


# ───────────────────────── Main class ───────────────────────────

class LanguageDetector:
    """
    Multi-strategy language detector for Macedonian / Albanian text.

    Priority order:
      1. Script-based (Cyrillic vs Latin) — very fast, any length
      2. ``lingua-language-detector`` — accurate for short texts
      3. ``langdetect`` fallback — broad language coverage
    """

    # Serbian-specific Cyrillic letters (not in the Macedonian alphabet)
    SERBIAN_INDICATORS = set("ЂђЋћ")

    # ── Script detection (Unicode ranges) ───────────────────────

    @staticmethod
    def _is_cyrillic(ch: str) -> bool:
        return "\u0400" <= ch <= "\u04FF"

    @staticmethod
    def _is_latin(ch: str) -> bool:
        return (
            ("A" <= ch <= "Z")
            or ("a" <= ch <= "z")
            or ch in "ëçËÇàáâãäåèéêìíîïòóôùúûüý"
        )

    def detect_script(self, text: str) -> str:
        """
        Fast script classification.

        Returns
        -------
        ``"macedonian"`` — predominantly Cyrillic
        ``"albanian_or_latin"`` — predominantly Latin
        ``"mixed"`` — no clear majority
        """
        cyr = sum(1 for c in text if self._is_cyrillic(c))
        lat = sum(1 for c in text if self._is_latin(c))
        total = cyr + lat

        if total == 0:
            return "mixed"
        if cyr / total > 0.60:
            return "macedonian"
        if lat / total > 0.60:
            return "albanian_or_latin"
        return "mixed"

    # ── Language detection (multi-strategy) ─────────────────────

    def detect_language(self, text: str) -> LanguageResult:
        """Detect whether *text* is Macedonian, Albanian, Serbian, or unknown."""
        if not text or not text.strip():
            return LanguageResult(language="unknown", confidence=0.0)

        text = unicodedata.normalize("NFC", text)

        # Very short text → rely on script only
        if len(text.strip()) < 20:
            return self._from_script(text)

        # Check for Serbian indicators before calling lingua
        if any(c in self.SERBIAN_INDICATORS for c in text):
            return LanguageResult(language="sr", confidence=0.85)

        # Try lingua first
        result = self._detect_lingua(text)
        if result is not None and result.confidence > 0.5:
            return result

        # Fallback to langdetect
        result = self._detect_langdetect(text)
        if result is not None:
            return result

        # Last resort: script detection
        return self._from_script(text)

    # ── Private helpers ─────────────────────────────────────────

    def _from_script(self, text: str) -> LanguageResult:
        script = self.detect_script(text)
        if script == "macedonian":
            return LanguageResult(language="mk", confidence=0.70)
        if script == "albanian_or_latin":
            return LanguageResult(language="sq", confidence=0.65)
        return LanguageResult(language="unknown", confidence=0.0)

    @staticmethod
    def _detect_lingua(text: str) -> Optional[LanguageResult]:
        detector = _get_lingua_detector()
        if detector is None:
            return None
        try:
            from lingua import Language
            confidence_values = detector.compute_language_confidence_values(text)
            if not confidence_values:
                return None
            best = confidence_values[0]
            lang_map = {
                Language.MACEDONIAN: "mk",
                Language.ALBANIAN: "sq",
                Language.SERBIAN: "sr",
            }
            code = lang_map.get(best.language, "unknown")
            return LanguageResult(language=code, confidence=round(best.value, 4))
        except Exception as exc:
            logger.debug("lingua detection failed: %s", exc)
            return None

    @staticmethod
    def _detect_langdetect(text: str) -> Optional[LanguageResult]:
        try:
            from langdetect import detect_langs
            results = detect_langs(text)
            if not results:
                return None
            best = results[0]
            lang_map = {"mk": "mk", "sq": "sq", "sr": "sr"}
            code = lang_map.get(best.lang, "unknown")
            return LanguageResult(language=code, confidence=round(best.prob, 4))
        except Exception as exc:
            logger.debug("langdetect failed: %s", exc)
            return None

    # ── Block-level splitting ───────────────────────────────────

    def split_by_language(self, text: str) -> list[TextSegment]:
        """
        Split *text* into contiguous segments of the same language,
        processing paragraph by paragraph.
        """
        if not text:
            return []

        paragraphs = text.split("\n")
        segments: list[TextSegment] = []
        offset = 0

        for para in paragraphs:
            end = offset + len(para)
            stripped = para.strip()
            if not stripped:
                offset = end + 1      # +1 for the newline
                continue

            lang_result = self.detect_language(stripped)
            lang = (
                lang_result.language
                if lang_result.language in ("mk", "sq")
                else "unknown"
            )

            # Merge with previous segment if same language
            if segments and segments[-1].language == lang:
                segments[-1].text += "\n" + stripped
                segments[-1].end_char = end
            else:
                segments.append(
                    TextSegment(
                        text=stripped,
                        language=lang,
                        start_char=offset,
                        end_char=end,
                    )
                )

            offset = end + 1

        return segments
