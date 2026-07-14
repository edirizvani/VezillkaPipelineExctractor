"""
Vezilka v2 — Sequential Extractor (Layout Type B).

For PDFs where the Macedonian text comes first in full, followed
by the Albanian text.  The boundary page (or block) is found by
the LayoutClassifier.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber

from phase3_extract.language_detector import LanguageDetector
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


class SequentialExtractor:
    """Extracts MK / SQ text from sequential-block bilingual PDFs."""

    def __init__(self):
        self._lang = LanguageDetector()

    def extract(
        self,
        pdf_path: Path,
        boundary_page: Optional[int] = None,
        boundary_block: Optional[int] = None,
    ) -> ExtractionResult:
        result = ExtractionResult(source_pdf=str(pdf_path))

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                if boundary_page is not None:
                    mk_text, sq_text = self._extract_by_page_boundary(pdf, boundary_page)
                else:
                    mk_text, sq_text = self._extract_by_language_detection(pdf)
        except Exception as e:
            logger.error("Sequential extraction failed for %s: %s", pdf_path, e)
            return result

        result.raw_mk = self._clean(mk_text, "mk")
        result.raw_sq = self._clean(sq_text, "sq")

        articles = self._split_articles(result.raw_mk, result.raw_sq, pdf_path)
        if not articles:
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

    def _extract_by_page_boundary(self, pdf, boundary_page: int) -> tuple[str, str]:
        mk_pages, sq_pages = [], []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if i < boundary_page:
                mk_pages.append(text)
            else:
                sq_pages.append(text)
        return "\n".join(mk_pages), "\n".join(sq_pages)

    def _extract_by_language_detection(self, pdf) -> tuple[str, str]:
        """No boundary known — detect language per page and split."""
        mk_pages, sq_pages = [], []
        last_lang = "mk"
        found_transition = False

        for page in pdf.pages:
            text = page.extract_text() or ""
            cyr = cyrillic_ratio(text)
            lat = latin_ratio(text)

            if not found_transition:
                if lat > 0.5 and cyr < 0.3:
                    found_transition = True
                    sq_pages.append(text)
                    last_lang = "sq"
                else:
                    mk_pages.append(text)
            else:
                sq_pages.append(text)

        return "\n".join(mk_pages), "\n".join(sq_pages)

    @staticmethod
    def _clean(text: str, lang: str) -> str:
        text = fix_hyphenation(text)
        text = normalise_whitespace(text)
        if lang == "sq":
            text = fix_albanian_encoding(text)
        return clean_text(text)

    def _split_articles(self, mk: str, sq: str, pdf_path: Path) -> list[ExtractedArticle]:
        mk_pattern = re.compile(r"(?:^|\n)\s*Член\s+(\d+)", re.MULTILINE)
        sq_pattern = re.compile(r"(?:^|\n)\s*Neni\s+(\d+)", re.MULTILINE)

        mk_arts = self._split_by_pattern(mk, mk_pattern)
        sq_arts = self._split_by_pattern(sq, sq_pattern)

        if not mk_arts or not sq_arts:
            return []

        articles = []
        for num, mk_text in mk_arts.items():
            sq_text = sq_arts.get(num, "")
            if not sq_text:
                continue
            art = ExtractedArticle(
                article_id=f"article_{num}",
                mk_text=mk_text,
                sq_text=sq_text,
                mk_sentences=self._sentence_split(mk_text),
                sq_sentences=self._sentence_split(sq_text),
                source_pdf=str(pdf_path),
            )
            articles.append(art)
        return articles

    @staticmethod
    def _split_by_pattern(text: str, pattern: re.Pattern) -> dict[str, str]:
        matches = list(pattern.finditer(text))
        if not matches:
            return {}
        result = {}
        for i, m in enumerate(matches):
            num = m.group(1)
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            result[num] = text[start:end].strip()
        return result

    @staticmethod
    def _sentence_split(text: str) -> list[str]:
        if not text.strip():
            return []
        try:
            from sentence_splitter import SentenceSplitter
            splitter = SentenceSplitter(language="en")
            return [s.strip() for s in splitter.split(text) if s.strip()]
        except ImportError:
            sents = re.split(r"(?<=[.!?])\s+", text)
            return [s.strip() for s in sents if s.strip()]
