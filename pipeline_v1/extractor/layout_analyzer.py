"""
Layout Analyzer — page-level analysis of PDF structure.

The Official Gazette (Služben Vesnik) layout:
  - Every page uses two newspaper-style columns (both columns are the SAME language)
  - Bilingual issues: first half = MK pages (Cyrillic), second half = SQ pages (Latin)
  - MK-only issues: all pages are Cyrillic

This module classifies each page by language and collects the full
MK and SQ texts separately for downstream alignment.

**v2**: Uses lingua-language-detector to verify that Latin-script pages
are actually Albanian (not English, French, etc.).
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extractor.pdf_extractor import PDFExtractor, PageData

logger = logging.getLogger(__name__)

# ── Lazy-initialised lingua detector ───────────────────────────
_LINGUA_DETECTOR = None


def _get_lingua_detector():
    """Build a lingua detector (once) covering the languages we expect."""
    global _LINGUA_DETECTOR
    if _LINGUA_DETECTOR is None:
        from lingua import Language, LanguageDetectorBuilder
        _LINGUA_DETECTOR = (
            LanguageDetectorBuilder
            .from_languages(
                Language.MACEDONIAN,
                Language.ALBANIAN,
                Language.ENGLISH,
                Language.SERBIAN,
                Language.TURKISH,
                Language.FRENCH,
                Language.BOSNIAN,
                Language.CROATIAN,
            )
            .with_minimum_relative_distance(0.05)
            .build()
        )
    return _LINGUA_DETECTOR


@dataclass
class DocumentContent:
    """Full bilingual content extracted from a PDF document."""
    pdf_path: str
    total_pages: int = 0
    mk_pages: int = 0                 # count of Macedonian pages
    sq_pages: int = 0                 # count of Albanian pages
    mixed_pages: int = 0
    empty_pages: int = 0
    other_pages: int = 0              # Latin pages that are NOT Albanian
    macedonian_full: str = ""         # All MK text concatenated
    albanian_full: str = ""           # All SQ text concatenated
    bilingual: bool = False           # True if significant SQ text exists
    page_languages: list[str] = field(default_factory=list)  # per-page lang labels


class LayoutAnalyzer:
    """
    Analyses a full PDF: classifies every page by language, extracts
    text in reading order (merging both newspaper columns), and
    collects MK + SQ sections separately.

    When *use_lingua* is True (default), Latin-script pages are verified
    with lingua-language-detector to ensure they are actually Albanian
    (not English, French, etc.).

    Usage::

        analyzer = LayoutAnalyzer()
        doc = analyzer.analyze("path/to/issue.pdf")
        print(f"Bilingual: {doc.bilingual}")
        print(f"MK chars: {len(doc.macedonian_full)}")
        print(f"SQ chars: {len(doc.albanian_full)}")
    """

    def __init__(self, use_lingua: bool = True):
        self.extractor = PDFExtractor()
        self.use_lingua = use_lingua

    def analyze(self, pdf_path: str | Path) -> DocumentContent:
        """
        Full analysis pipeline for a single PDF.
        Returns a ``DocumentContent`` with separated MK and SQ text.
        """
        pdf_path = Path(pdf_path)
        pages_data = self.extractor.extract_with_layout(str(pdf_path))

        doc = DocumentContent(pdf_path=str(pdf_path), total_pages=len(pages_data))
        mk_parts: list[str] = []
        sq_parts: list[str] = []

        for page_data in pages_data:
            lang = page_data.language

            # Extract full page text (both columns merged in reading order)
            page_text = self.extractor.extract_page_text(page_data)

            if not page_text.strip():
                doc.page_languages.append("empty")
                doc.empty_pages += 1
                continue

            if lang == "mk":
                doc.page_languages.append("mk")
                doc.mk_pages += 1
                mk_parts.append(page_text)
            elif lang == "sq":
                # ── lingua verification: is this really Albanian? ──
                verified = self._verify_latin_page(page_text)
                if verified == "sq":
                    doc.page_languages.append("sq")
                    doc.sq_pages += 1
                    sq_parts.append(page_text)
                else:
                    doc.page_languages.append(f"other:{verified}")
                    doc.other_pages += 1
                    logger.debug(
                        "Page %d: Latin script but lingua detected '%s' — excluded",
                        page_data.page_number, verified,
                    )
            elif lang == "mixed":
                doc.page_languages.append("mixed")
                doc.mixed_pages += 1
                mk_lines, sq_lines = self._split_mixed_page(page_text)
                if mk_lines:
                    mk_parts.append(mk_lines)
                if sq_lines:
                    # Also verify the Latin portion with lingua
                    sq_verified = self._verify_latin_page(sq_lines) if len(sq_lines) > 50 else "sq"
                    if sq_verified == "sq":
                        sq_parts.append(sq_lines)
                    else:
                        logger.debug(
                            "Page %d mixed Latin portion is '%s' — excluded",
                            page_data.page_number, sq_verified,
                        )
            else:
                doc.page_languages.append("empty")
                doc.empty_pages += 1

        # Merge cross-page text
        doc.macedonian_full = PDFExtractor.merge_cross_page_articles(mk_parts)
        doc.albanian_full = PDFExtractor.merge_cross_page_articles(sq_parts)

        # Determine bilingual status: SQ section must have real content
        sq_len = len(doc.albanian_full.strip())
        doc.bilingual = sq_len >= 200 and doc.sq_pages >= 3

        logger.info(
            "Analyzed %s: %d pages (%d MK, %d SQ, %d other, %d mixed, %d empty), "
            "MK=%d chars, SQ=%d chars, bilingual=%s",
            pdf_path.name,
            doc.total_pages, doc.mk_pages, doc.sq_pages, doc.other_pages,
            doc.mixed_pages, doc.empty_pages,
            len(doc.macedonian_full), len(doc.albanian_full),
            doc.bilingual,
        )
        return doc

    # ── Lingua verification ─────────────────────────────────────

    def _verify_latin_page(self, text: str) -> str:
        """
        Use lingua to verify Latin-script text is Albanian.

        Returns ``"sq"`` if Albanian, otherwise a short language tag
        (``"en"``, ``"sr"``, ``"fr"``, etc.).
        """
        if not self.use_lingua:
            return "sq"

        sample = text.strip()
        if len(sample) < 40:
            return "sq"  # Too short — trust script detection

        # Take a representative middle sample for long pages
        if len(sample) > 1500:
            mid = len(sample) // 2
            sample = sample[max(0, mid - 750):mid + 750]

        try:
            from lingua import Language
            detector = _get_lingua_detector()
            detected = detector.detect_language_of(sample)

            if detected is None:
                return "sq"  # Uncertain → trust script heuristic
            if detected == Language.ALBANIAN:
                return "sq"
            if detected == Language.MACEDONIAN:
                return "mk"
            if detected == Language.ENGLISH:
                return "en"
            if detected == Language.SERBIAN:
                return "sr"
            return detected.name.lower()[:2]
        except Exception as e:
            logger.warning("Lingua verification failed: %s", e)
            return "sq"

    # ── Mixed-page handling ─────────────────────────────────────

    @staticmethod
    def _is_cyrillic(ch: str) -> bool:
        return "\u0400" <= ch <= "\u04FF"

    @staticmethod
    def _is_latin(ch: str) -> bool:
        return ("A" <= ch <= "Z") or ("a" <= ch <= "z") or ch in "ëçËÇ"

    def _split_mixed_page(self, text: str) -> tuple[str, str]:
        """
        For pages that are a mix of Cyrillic and Latin (transition pages),
        split lines into MK and SQ groups.
        """
        mk_lines = []
        sq_lines = []

        for line in text.split("\n"):
            cyr = sum(1 for c in line if self._is_cyrillic(c))
            lat = sum(1 for c in line if self._is_latin(c))
            total = cyr + lat
            if total == 0:
                continue
            if cyr / total > 0.6:
                mk_lines.append(line)
            elif lat / total > 0.6:
                sq_lines.append(line)
            # Skip ambiguous lines

        return "\n".join(mk_lines), "\n".join(sq_lines)
