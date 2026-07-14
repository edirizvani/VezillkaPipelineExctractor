"""
Vezilka v2 — OCR / Mixed Extractor (Layout Type C).

For pre-2019 PDFs with inconsistent or interleaved layout.
Uses PyMuPDF (fitz) for raw text plus optional EasyOCR fallback
when text layer quality is too low.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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


class OCRExtractor:
    """Extracts bilingual text from mixed/pre-2019 PDFs with fallback OCR."""

    OCR_MIN_CHARS_PER_PAGE = 50
    MIN_TEXT_QUALITY = 0.3      # fraction of alpha chars

    def __init__(self, use_ocr: bool = True):
        self.use_ocr = use_ocr
        self._lang = LanguageDetector()

    def extract(self, pdf_path: Path) -> ExtractionResult:
        result = ExtractionResult(source_pdf=str(pdf_path))

        # Try PyMuPDF first for text extraction
        raw_text = self._extract_with_fitz(pdf_path)
        if not raw_text or self._is_low_quality(raw_text):
            if self.use_ocr:
                raw_text = self._extract_with_ocr(pdf_path)

        if not raw_text:
            logger.warning("No text extracted from %s", pdf_path)
            return result

        # Split the mixed text into MK and SQ segments
        mk_lines, sq_lines = self._segment_by_language(raw_text)

        result.raw_mk = self._clean("\n".join(mk_lines), "mk")
        result.raw_sq = self._clean("\n".join(sq_lines), "sq")

        # Try to align articles
        articles = self._build_articles(result.raw_mk, result.raw_sq, pdf_path)
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

    def _extract_with_fitz(self, pdf_path: Path) -> str:
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            pages = []
            for page in doc:
                pages.append(page.get_text("text"))
            doc.close()
            return "\n\n".join(pages)
        except ImportError:
            logger.warning("PyMuPDF not installed, using pdfplumber for OCR fallback")
            return self._extract_with_pdfplumber(pdf_path)
        except Exception as e:
            logger.error("Fitz extraction error for %s: %s", pdf_path, e)
            return ""

    @staticmethod
    def _extract_with_pdfplumber(pdf_path: Path) -> str:
        try:
            import pdfplumber
            with pdfplumber.open(str(pdf_path)) as pdf:
                pages = []
                for page in pdf.pages:
                    pages.append(page.extract_text() or "")
                return "\n\n".join(pages)
        except Exception as e:
            logger.error("pdfplumber extraction error for %s: %s", pdf_path, e)
            return ""

    def _extract_with_ocr(self, pdf_path: Path) -> str:
        """OCR fallback using pdf2image + easyocr."""
        try:
            from pdf2image import convert_from_path
            import easyocr
        except ImportError:
            logger.warning("OCR libs not available (pdf2image / easyocr)")
            return ""

        try:
            images = convert_from_path(str(pdf_path), dpi=300)
            reader = easyocr.Reader(["mk", "sq", "en"], gpu=False)
            all_text = []
            for img in images:
                import numpy as np
                img_array = np.array(img)
                results = reader.readtext(img_array, detail=0)
                all_text.append("\n".join(results))
            return "\n\n".join(all_text)
        except Exception as e:
            logger.error("OCR failed for %s: %s", pdf_path, e)
            return ""

    def _segment_by_language(self, text: str) -> tuple[list[str], list[str]]:
        """Split mixed text into MK and SQ lines by language."""
        mk_lines, sq_lines = [], []
        for line in text.split("\n"):
            line = line.strip()
            if not line or len(line) < 3:
                continue
            lang = self._lang.detect(line)
            if lang == DetectedLanguage.MACEDONIAN:
                mk_lines.append(line)
            elif lang == DetectedLanguage.ALBANIAN:
                sq_lines.append(line)
            else:
                # Heuristic: check script
                if cyrillic_ratio(line) > 0.5:
                    mk_lines.append(line)
                elif latin_ratio(line) > 0.5:
                    sq_lines.append(line)
        return mk_lines, sq_lines

    @staticmethod
    def _is_low_quality(text: str) -> bool:
        if not text:
            return True
        alpha = sum(c.isalpha() for c in text)
        return alpha / max(len(text), 1) < 0.3

    @staticmethod
    def _clean(text: str, lang: str) -> str:
        text = fix_hyphenation(text)
        text = normalise_whitespace(text)
        if lang == "sq":
            text = fix_albanian_encoding(text)
        return clean_text(text)

    def _build_articles(self, mk: str, sq: str, pdf_path: Path) -> list[ExtractedArticle]:
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
            articles.append(ExtractedArticle(
                article_id=f"article_{num}",
                mk_text=mk_text, sq_text=sq_text,
                mk_sentences=self._sentence_split(mk_text),
                sq_sentences=self._sentence_split(sq_text),
                source_pdf=str(pdf_path),
            ))
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
