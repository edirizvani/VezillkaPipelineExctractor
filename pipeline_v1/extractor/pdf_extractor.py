"""
PDF Text Extractor — extracts text with layout information from Služben Vesnik PDFs.

Uses pdfplumber for coordinate-aware text extraction.

Official Gazette layout (confirmed by empirical analysis):
  - Each page uses a two-column NEWSPAPER layout
    (text flows left-col top→bottom, then right-col top→bottom, BOTH columns same language)
  - Bilingual issues: MK section first (Cyrillic pages), then SQ section (Latin pages)
  - MK-only issues: all pages are Cyrillic
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber

logger = logging.getLogger(__name__)


# ───────────────────────── Data classes ─────────────────────────

@dataclass
class WordInfo:
    """A single word with its bounding box."""
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page: int = 0

    @property
    def mid_x(self) -> float:
        return (self.x0 + self.x1) / 2.0


@dataclass
class PageData:
    """All extracted data for a single PDF page."""
    page_number: int
    width: float
    height: float
    words: list[WordInfo] = field(default_factory=list)
    raw_text: str = ""
    layout: str = "unknown"            # "two_column" | "sequential"
    split_x: Optional[float] = None
    language: str = "unknown"          # "mk" | "sq" | "mixed" | "empty"


@dataclass
class ColumnTexts:
    """Text extracted from a two-column layout."""
    left_column: str
    right_column: str


# ───────────────────────── Main extractor ───────────────────────

class PDFExtractor:
    """
    Extracts text with layout-awareness from Služben Vesnik PDFs.

    Usage::

        extractor = PDFExtractor()
        pages = extractor.extract_with_layout("path/to/issue.pdf")
        for page in pages:
            print(page.language, page.raw_text[:100])
    """

    @staticmethod
    def _is_cyrillic(ch: str) -> bool:
        return "\u0400" <= ch <= "\u04FF"

    @staticmethod
    def _is_latin(ch: str) -> bool:
        return ("A" <= ch <= "Z") or ("a" <= ch <= "z") or ch in "ëçËÇ"

    # ── Core extraction ─────────────────────────────────────────

    def extract_with_layout(self, pdf_path: str | Path) -> list[PageData]:
        """
        Open *pdf_path* and return a ``PageData`` for every page,
        with words, layout type, page language, and raw text.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages: list[PageData] = []

        with pdfplumber.open(str(pdf_path)) as pdf:
            for idx, page in enumerate(pdf.pages):
                page_number = idx + 1
                width = float(page.width)
                height = float(page.height)

                raw_words = page.extract_words(
                    keep_blank_chars=False,
                    x_tolerance=3,
                    y_tolerance=3,
                    extra_attrs=["fontname", "size"],
                )
                words = [
                    WordInfo(
                        text=w["text"],
                        x0=w["x0"],
                        y0=w["top"],
                        x1=w["x1"],
                        y1=w["bottom"],
                        page=page_number,
                    )
                    for w in raw_words
                ]

                raw_text = page.extract_text() or ""

                pd = PageData(
                    page_number=page_number,
                    width=width,
                    height=height,
                    words=words,
                    raw_text=raw_text,
                )

                # Detect two-column newspaper layout
                layout_info = self._detect_columns(pd)
                pd.layout = layout_info["layout"]
                pd.split_x = layout_info.get("split_x")

                # Classify page language by script ratio
                pd.language = self._detect_page_language(raw_text)

                pages.append(pd)
                logger.debug(
                    "Page %d: %d words, layout=%s, lang=%s",
                    page_number, len(words), pd.layout, pd.language,
                )

        logger.info("Extracted %d pages from %s", len(pages), pdf_path.name)
        return pages

    # ── Page-level language detection ───────────────────────────

    def _detect_page_language(self, text: str) -> str:
        """
        Classify page as 'mk' (Cyrillic-dominant), 'sq' (Latin-dominant),
        'mixed', or 'empty'.
        """
        if not text or len(text.strip()) < 20:
            return "empty"

        cyr = sum(1 for c in text if self._is_cyrillic(c))
        lat = sum(1 for c in text if self._is_latin(c))
        total = cyr + lat
        if total == 0:
            return "empty"

        if cyr / total > 0.70:
            return "mk"
        elif lat / total > 0.70:
            return "sq"
        else:
            return "mixed"

    # ── Column / layout detection ───────────────────────────────

    def _detect_columns(self, page_data: PageData) -> dict:
        """
        Detect if the page is two-column (newspaper) or single-column.
        """
        words = page_data.words
        if len(words) < 10:
            return {"layout": "sequential", "split_x": None}

        width = page_data.width
        mid = width / 2.0
        tolerance = width * 0.08

        left_words = [w for w in words if w.mid_x < mid - tolerance]
        right_words = [w for w in words if w.mid_x > mid + tolerance]
        total = len(words)

        left_frac = len(left_words) / total if total else 0
        right_frac = len(right_words) / total if total else 0

        if (
            left_frac > 0.25
            and right_frac > 0.25
            and len(left_words) >= 15
            and len(right_words) >= 15
        ):
            left_xs = sorted(w.x1 for w in left_words)
            right_xs = sorted(w.x0 for w in right_words)
            split_x = (left_xs[-1] + right_xs[0]) / 2.0 if right_xs else mid
            return {"layout": "two_column", "split_x": split_x}

        return {"layout": "sequential", "split_x": None}

    # ── Full-page text extraction (reading order) ───────────────

    def extract_page_text(self, page_data: PageData) -> str:
        """
        Extract full page text in reading order.
        Two-column: left col top→bottom, then right col top→bottom.
        Sequential: top→bottom.
        """
        if not page_data.words:
            return ""

        if page_data.layout == "two_column" and page_data.split_x:
            split_x = page_data.split_x
            left_words = [w for w in page_data.words if w.mid_x < split_x]
            right_words = [w for w in page_data.words if w.mid_x >= split_x]
            left_text = self._reconstruct_text(left_words)
            right_text = self._reconstruct_text(right_words)
            return left_text + "\n" + right_text
        else:
            return self._reconstruct_text(page_data.words)

    # ── Two-column extraction (for backward compat) ─────────────

    def extract_columns(
        self, page_data: PageData, split_x: Optional[float] = None
    ) -> ColumnTexts:
        if split_x is None:
            split_x = page_data.width / 2.0

        left_words = [w for w in page_data.words if w.mid_x < split_x]
        right_words = [w for w in page_data.words if w.mid_x >= split_x]

        return ColumnTexts(
            left_column=self._reconstruct_text(left_words),
            right_column=self._reconstruct_text(right_words),
        )

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _reconstruct_text(words: list[WordInfo]) -> str:
        """
        Reconstruct running text from words,
        keeping vertical (line) and horizontal (word) order.
        """
        if not words:
            return ""

        words = sorted(words, key=lambda w: (w.y0, w.x0))

        lines: list[list[str]] = []
        current_line: list[str] = [words[0].text]
        prev_y = words[0].y0

        for w in words[1:]:
            word_height = w.y1 - w.y0
            if abs(w.y0 - prev_y) > max(word_height * 0.5, 3):
                lines.append(current_line)
                current_line = []
            current_line.append(w.text)
            prev_y = w.y0

        if current_line:
            lines.append(current_line)

        return "\n".join(" ".join(line) for line in lines)

    @staticmethod
    def merge_cross_page_articles(pages_text: list[str]) -> str:
        """
        Merge text from consecutive pages. Join mid-sentence page breaks.
        """
        if not pages_text:
            return ""

        merged = pages_text[0]
        for page_text in pages_text[1:]:
            stripped = merged.rstrip()
            if stripped and stripped[-1] not in ".!?;:":
                merged = stripped + " " + page_text.lstrip()
            else:
                merged = stripped + "\n" + page_text
        return merged
