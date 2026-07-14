"""
Vezilka v2 — Two-Column Extractor (Layout Type A).

For PDFs where MK is in the left column and SQ in the right column
on the same pages.  We split at the page midpoint, group words
per column, then sentence-split each language.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber

from phase3_extract.language_detector import DetectedLanguage, LanguageDetector
from utils.text_utils import (
    clean_text,
    cyrillic_ratio,
    fix_albanian_encoding,
    fix_hyphenation,
    latin_ratio,
    normalise_whitespace,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractedArticle:
    """A single article / legal-act pair from the PDF."""
    article_id: str = ""
    mk_text: str = ""
    sq_text: str = ""
    mk_sentences: list[str] = field(default_factory=list)
    sq_sentences: list[str] = field(default_factory=list)
    source_pdf: str = ""
    pages: list[int] = field(default_factory=list)


@dataclass
class ExtractionResult:
    articles: list[ExtractedArticle] = field(default_factory=list)
    raw_mk: str = ""
    raw_sq: str = ""
    source_pdf: str = ""


class TwoColumnExtractor:
    """Extracts MK / SQ text from a two-column bilingual PDF."""

    def __init__(self, tolerance: float = 0.08):
        self.tolerance = tolerance
        self._lang = LanguageDetector()

    def extract(self, pdf_path: Path) -> ExtractionResult:
        result = ExtractionResult(source_pdf=str(pdf_path))
        all_mk: list[str] = []
        all_sq: list[str] = []

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages:
                    mk_lines, sq_lines = self._split_page(page)
                    all_mk.extend(mk_lines)
                    all_sq.extend(sq_lines)
        except Exception as e:
            logger.error("Two-column extraction failed for %s: %s", pdf_path, e)
            return result

        result.raw_mk = self._join_lines(all_mk, "mk")
        result.raw_sq = self._join_lines(all_sq, "sq")

        # Build articles by splitting on article number patterns
        articles = self._split_into_articles(result.raw_mk, result.raw_sq, pdf_path)
        if not articles:
            # Whole-document fallback: treat entire text as one article
            art = ExtractedArticle(
                article_id="full",
                mk_text=result.raw_mk,
                sq_text=result.raw_sq,
                mk_sentences=self._sentence_split(result.raw_mk),
                sq_sentences=self._sentence_split(result.raw_sq),
                source_pdf=str(pdf_path),
            )
            if art.mk_text.strip() and art.sq_text.strip():
                articles.append(art)

        result.articles = articles
        return result

    def _split_page(self, page) -> tuple[list[str], list[str]]:
        """Split a page into left (MK) and right (SQ) columns."""
        words = page.extract_words(use_text_flow=True) or []
        if not words:
            return [], []

        width = float(page.width) if page.width else 612.0
        mid = width / 2.0
        tol = width * self.tolerance

        left, right = [], []
        for w in words:
            x0 = float(w.get("x0", 0))
            if x0 < mid - tol:
                left.append(w)
            elif x0 >= mid + tol:
                right.append(w)
            else:
                # In the gap — assign by script
                txt = w.get("text", "")
                if cyrillic_ratio(txt) > 0.5:
                    left.append(w)
                else:
                    right.append(w)

        mk_lines = self._words_to_lines(left)
        sq_lines = self._words_to_lines(right)
        return mk_lines, sq_lines

    @staticmethod
    def _words_to_lines(words: list[dict]) -> list[str]:
        """Group words into text lines by vertical position."""
        if not words:
            return []
        words = sorted(words, key=lambda w: (round(float(w.get("top", 0)), 1), float(w.get("x0", 0))))
        lines: list[str] = []
        current: list[str] = []
        prev_top: Optional[float] = None
        for w in words:
            top = round(float(w.get("top", 0)), 1)
            if prev_top is not None and abs(top - prev_top) > 3.0:
                if current:
                    lines.append(" ".join(current))
                current = []
            current.append(w.get("text", ""))
            prev_top = top
        if current:
            lines.append(" ".join(current))
        return lines

    def _join_lines(self, lines: list[str], lang: str) -> str:
        """Join raw lines, fix hyphenation, clean text."""
        text = "\n".join(lines)
        text = fix_hyphenation(text)
        text = normalise_whitespace(text)
        if lang == "sq":
            text = fix_albanian_encoding(text)
        return clean_text(text)

    def _split_into_articles(self, mk: str, sq: str, pdf_path: Path) -> list[ExtractedArticle]:
        """Split parallel MK / SQ texts into articles based on Член/Neni numbering."""
        mk_pattern = re.compile(r"(?:^|\n)\s*Член\s+(\d+)", re.MULTILINE)
        sq_pattern = re.compile(r"(?:^|\n)\s*Neni\s+(\d+)", re.MULTILINE)

        mk_articles = self._split_by_pattern(mk, mk_pattern)
        sq_articles = self._split_by_pattern(sq, sq_pattern)

        if not mk_articles or not sq_articles:
            return []

        articles = []
        matched_sq = set()
        for mk_id, mk_text in mk_articles.items():
            sq_text = sq_articles.get(mk_id, "")
            if sq_text:
                matched_sq.add(mk_id)
            art = ExtractedArticle(
                article_id=f"article_{mk_id}",
                mk_text=mk_text,
                sq_text=sq_text,
                mk_sentences=self._sentence_split(mk_text),
                sq_sentences=self._sentence_split(sq_text),
                source_pdf=str(pdf_path),
            )
            if art.mk_text.strip() and art.sq_text.strip():
                articles.append(art)

        return articles

    @staticmethod
    def _split_by_pattern(text: str, pattern: re.Pattern) -> dict[str, str]:
        matches = list(pattern.finditer(text))
        if not matches:
            return {}
        result: dict[str, str] = {}
        for i, m in enumerate(matches):
            num = m.group(1)
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            result[num] = text[start:end].strip()
        return result

    @staticmethod
    def _sentence_split(text: str) -> list[str]:
        """Robust sentence splitter handling MK and SQ punctuation."""
        if not text.strip():
            return []
        try:
            from sentence_splitter import SentenceSplitter
            splitter = SentenceSplitter(language="en")
            return [s.strip() for s in splitter.split(text) if s.strip()]
        except ImportError:
            pass
        # Fallback: split on sentence boundaries
        sents = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sents if s.strip()]
