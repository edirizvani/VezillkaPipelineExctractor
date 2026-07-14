"""
Vezilka v2 — Gatekeeper.

Runs BEFORE any extraction.  Cheaply decides whether a PDF is worth
processing at all.  Catches:
    • Too-small files (< 50 KB)
    • No Albanian content detected (monolingual MK)
    • Scanned image-only PDFs with no text layer

All skipped PDFs are logged to data/skipped_pdfs.jsonl.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pdfplumber

from config import DEFAULT_CONFIG, VezilkaConfig
from utils.text_utils import cyrillic_ratio, has_albanian_markers, latin_ratio

logger = logging.getLogger(__name__)


class SkipReason(Enum):
    NONE = "none"
    FILE_TOO_SMALL = "file_too_small"
    NO_ALBANIAN_DETECTED = "no_albanian_detected"
    SCANNED_NO_TEXT_LAYER = "scanned_no_text_layer"
    CORRUPT_OR_UNREADABLE = "corrupt_or_unreadable"


@dataclass
class ProcessingDecision:
    """Result of the gatekeeper pre-check."""
    should_process: bool
    reason: SkipReason
    detail: str = ""
    is_scanned: bool = False


class Gatekeeper:
    """Cheap pre-filter that decides whether to process a PDF."""

    def __init__(self, config: VezilkaConfig | None = None):
        self.cfg = config or DEFAULT_CONFIG

    # ── public API ──────────────────────────────────────────────

    def check(self, pdf_path: Path) -> ProcessingDecision:
        """Run all gatekeeper checks on *pdf_path*."""

        # Check 1 — file size
        try:
            size_kb = pdf_path.stat().st_size / 1024
        except OSError as e:
            return self._skip(SkipReason.CORRUPT_OR_UNREADABLE, str(e))

        if size_kb < self.cfg.gatekeeper_min_file_size_kb:
            return self._skip(SkipReason.FILE_TOO_SMALL,
                              f"{size_kb:.0f} KB < {self.cfg.gatekeeper_min_file_size_kb} KB")

        # Check 2 + 3 — text layer existence + Albanian presence
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                total_pages = len(pdf.pages)
                if total_pages == 0:
                    return self._skip(SkipReason.CORRUPT_OR_UNREADABLE, "0 pages")

                # Check 3 first (text layer) on a sample of pages
                total_chars = 0
                sample_indices = self._sample_pages(total_pages)
                page_texts: dict[int, str] = {}

                for idx in sample_indices:
                    text = pdf.pages[idx].extract_text() or ""
                    page_texts[idx] = text
                    total_chars += len(text.strip())

                if total_chars < 20:
                    # Very little text — scan all pages to be sure
                    for idx in range(total_pages):
                        if idx not in page_texts:
                            text = pdf.pages[idx].extract_text() or ""
                            total_chars += len(text.strip())
                            if total_chars >= 20:
                                break
                    if total_chars < 20:
                        return self._skip(SkipReason.SCANNED_NO_TEXT_LAYER,
                                          f"Only {total_chars} chars across {total_pages} pages")

                # Check 2 — Albanian presence in sampled pages
                if self._has_albanian_in_texts(page_texts.values()):
                    return ProcessingDecision(True, SkipReason.NONE)

                # Full scan — stop at first hit
                for idx in range(total_pages):
                    if idx in page_texts:
                        continue
                    text = pdf.pages[idx].extract_text() or ""
                    if self._text_has_albanian(text):
                        return ProcessingDecision(True, SkipReason.NONE)

                return self._skip(SkipReason.NO_ALBANIAN_DETECTED,
                                  f"No Albanian signals in {total_pages} pages")

        except Exception as e:
            logger.error("Gatekeeper error for %s: %s", pdf_path, e)
            return self._skip(SkipReason.CORRUPT_OR_UNREADABLE, str(e))

    def log_skip(self, pdf_path: Path, decision: ProcessingDecision) -> None:
        """Append a skip record to skipped_pdfs.jsonl."""
        log_path = self.cfg.skipped_log
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "pdf": str(pdf_path),
            "reason": decision.reason.value,
            "detail": decision.detail,
        }
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error("Could not write skip log: %s", e)

    # ── internals ───────────────────────────────────────────────

    @staticmethod
    def _sample_pages(total: int) -> list[int]:
        """Return indices for first, middle, last pages."""
        if total == 1:
            return [0]
        if total == 2:
            return [0, 1]
        return [0, total // 2, total - 1]

    def _has_albanian_in_texts(self, texts) -> bool:
        return any(self._text_has_albanian(t) for t in texts)

    def _text_has_albanian(self, text: str) -> bool:
        """Check if *text* contains Albanian signals."""
        if not text:
            return False
        # Albanian-specific characters
        if any(c in text for c in self.cfg.gatekeeper_albanian_chars):
            return True
        # Albanian keywords
        for kw in self.cfg.gatekeeper_albanian_keywords:
            if kw in text:
                return True
        # High Latin ratio in a mainly Cyrillic document
        if latin_ratio(text) > self.cfg.gatekeeper_min_latin_ratio_for_albanian:
            return True
        return False

    @staticmethod
    def _skip(reason: SkipReason, detail: str = "") -> ProcessingDecision:
        return ProcessingDecision(
            should_process=False,
            reason=reason,
            detail=detail,
            is_scanned=(reason == SkipReason.SCANNED_NO_TEXT_LAYER),
        )
