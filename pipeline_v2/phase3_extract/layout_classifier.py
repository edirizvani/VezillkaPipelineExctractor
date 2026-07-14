"""
Vezilka v2 — Layout Classifier.

Classifies each PDF into a layout type BEFORE extraction:
  A) TWO_COLUMN:    MK left | SQ right on same pages
  B) SEQUENTIAL:    Full MK block then full SQ block
  C) MIXED_PRE2019: Interleaved, inconsistent
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import pdfplumber

from utils.text_utils import cyrillic_ratio, latin_ratio, has_albanian_markers

logger = logging.getLogger(__name__)


class LayoutType(Enum):
    TWO_COLUMN = "two_column"
    SEQUENTIAL = "sequential"
    MIXED_PRE2019 = "mixed"
    SINGLE_LANGUAGE = "single"
    UNKNOWN = "unknown"


@dataclass
class LayoutClassificationResult:
    layout_type: LayoutType
    confidence: float
    has_albanian: bool
    boundary_page: Optional[int] = None
    boundary_block: Optional[int] = None
    total_pages: int = 0
    detail: str = ""


ALBANIAN_BOUNDARY_PATTERNS = [
    re.compile(r"^L\s*I\s*G\s*J\b", re.MULTILINE),
    re.compile(r"^LIGJ\b", re.MULTILINE),
    re.compile(r"^Neni\s+1\b", re.MULTILINE),
]
SEPARATOR_PATTERN = re.compile(r"_{5,}|—{5,}")


class LayoutClassifier:
    """Classifies a PDF's bilingual layout type before extraction."""

    def __init__(
        self,
        column_split_tolerance: float = 0.08,
        two_column_min_fraction: float = 0.50,
        cyrillic_threshold: float = 0.60,
        latin_threshold: float = 0.60,
        sequential_transition_threshold: float = 0.70,
    ):
        self.column_split_tolerance = column_split_tolerance
        self.two_column_min_fraction = two_column_min_fraction
        self.cyrillic_threshold = cyrillic_threshold
        self.latin_threshold = latin_threshold
        self.sequential_transition_threshold = sequential_transition_threshold

    def classify(self, pdf_path: Path) -> LayoutClassificationResult:
        """Classify a PDF into one of the layout types."""
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                total = len(pdf.pages)
                if total == 0:
                    return LayoutClassificationResult(LayoutType.UNKNOWN, 0.0, False, detail="Empty PDF")

                page_data = self._analyse_pages(pdf)
                has_alb = any(pd["has_latin"] for pd in page_data)

                if not has_alb:
                    return LayoutClassificationResult(
                        LayoutType.SINGLE_LANGUAGE, 0.9, False, total_pages=total,
                        detail="No Albanian content detected",
                    )

                # Try Type A
                result = self._check_two_column(page_data, total)
                if result:
                    return result

                # Try Type B
                result = self._check_sequential(pdf, page_data, total)
                if result:
                    return result

                # Fallback Type C
                return LayoutClassificationResult(
                    LayoutType.MIXED_PRE2019, 0.5, has_alb, total_pages=total,
                    detail="No clear two-column or sequential structure",
                )
        except Exception as e:
            logger.error("Layout classification failed for %s: %s", pdf_path, e)
            return LayoutClassificationResult(LayoutType.UNKNOWN, 0.0, False, detail=str(e))

    def _analyse_pages(self, pdf: pdfplumber.PDF) -> list[dict]:
        page_data = []
        for i, page in enumerate(pdf.pages):
            words = page.extract_words(use_text_flow=True) or []
            full_text = " ".join(w.get("text", "") for w in words)
            cyr_r = cyrillic_ratio(full_text)
            lat_r = latin_ratio(full_text)

            page_width = float(page.width) if page.width else 612.0
            mid = page_width / 2.0
            tol = page_width * self.column_split_tolerance

            left_words = [w for w in words if float(w.get("x0", 0)) < mid - tol]
            right_words = [w for w in words if float(w.get("x0", 0)) >= mid + tol]
            left_text = " ".join(w.get("text", "") for w in left_words)
            right_text = " ".join(w.get("text", "") for w in right_words)

            is_bilingual = (
                len(left_words) > 10 and len(right_words) > 10
                and cyrillic_ratio(left_text) > self.cyrillic_threshold
                and latin_ratio(right_text) > self.latin_threshold
            )

            page_data.append({
                "page_index": i, "full_text": full_text,
                "cyrillic_ratio": cyr_r, "latin_ratio": lat_r,
                "has_latin": lat_r > 0.20, "has_cyrillic": cyr_r > 0.20,
                "is_bilingual_columns": is_bilingual,
                "word_count": len(words),
                "left_text": left_text, "right_text": right_text,
            })
        return page_data

    def _check_two_column(self, page_data: list[dict], total: int) -> Optional[LayoutClassificationResult]:
        bi_count = sum(1 for pd in page_data if pd["is_bilingual_columns"])
        frac = bi_count / max(total, 1)
        if frac >= self.two_column_min_fraction:
            return LayoutClassificationResult(
                LayoutType.TWO_COLUMN, min(frac + 0.1, 1.0), True, total_pages=total,
                detail=f"{bi_count}/{total} pages bilingual two-column ({frac:.0%})",
            )
        return None

    def _check_sequential(self, pdf, page_data, total) -> Optional[LayoutClassificationResult]:
        if total < 2:
            return None

        # Strategy 1: language transition
        transition = self._find_transition(page_data)
        # Strategy 2: Albanian boundary markers
        marker_page = self._find_boundary_markers(page_data)

        boundary = marker_page if marker_page is not None else transition
        if boundary is None:
            return None

        before = page_data[:boundary]
        after = page_data[boundary:]
        mk_cyr = sum(1 for p in before if p["cyrillic_ratio"] > 0.5) / max(len(before), 1)
        sq_lat = sum(1 for p in after if p["latin_ratio"] > 0.3) / max(len(after), 1)

        if mk_cyr > 0.5 and sq_lat > 0.3:
            return LayoutClassificationResult(
                LayoutType.SEQUENTIAL, (mk_cyr + sq_lat) / 2, True,
                boundary_page=boundary, boundary_block=0, total_pages=total,
                detail=f"MK pages 0–{boundary - 1}, SQ pages {boundary}–{total - 1}",
            )
        return None

    def _find_transition(self, page_data: list[dict]) -> Optional[int]:
        for i in range(1, len(page_data)):
            if (page_data[i - 1]["cyrillic_ratio"] > self.sequential_transition_threshold
                    and page_data[i]["latin_ratio"] > self.sequential_transition_threshold):
                return i
        # Softer: rolling window
        for i in range(2, len(page_data)):
            before_avg = sum(p["cyrillic_ratio"] for p in page_data[max(0, i - 3):i]) / min(3, i)
            remaining = len(page_data) - i
            if remaining > 0:
                after_avg = sum(p["latin_ratio"] for p in page_data[i:min(len(page_data), i + 3)]) / min(3, remaining)
                if before_avg > 0.6 and after_avg > 0.5:
                    return i
        return None

    def _find_boundary_markers(self, page_data: list[dict]) -> Optional[int]:
        for pd in page_data:
            if pd["cyrillic_ratio"] > 0.8 and pd["page_index"] < len(page_data) * 0.3:
                continue
            for pattern in ALBANIAN_BOUNDARY_PATTERNS:
                if pattern.search(pd["full_text"]):
                    logger.info("Albanian boundary on page %d", pd["page_index"])
                    return pd["page_index"]
            if SEPARATOR_PATTERN.search(pd["full_text"]) and pd["latin_ratio"] > 0.3:
                return pd["page_index"]
        return None
