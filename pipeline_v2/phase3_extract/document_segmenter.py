"""
Vezilka v2 — Document Segmenter.

A single PDF issue contains multiple independent gazette items
(laws, decisions, regulations) separated by bold item numbers like
``262.``, ``263.``, ``264.``.

This module:
  1) Splits the extracted full text into individual items.
  2) Matches each MK item to its SQ counterpart by item number.
  3) Validates matches using document-level LASER3 similarity.

Alignment must ALWAYS be scoped to a single item.  This module
runs after extraction but before alignment.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import DEFAULT_CONFIG, VezilkaConfig

logger = logging.getLogger(__name__)

ITEM_BOUNDARY = re.compile(r"^\s*(\d{3,4})\.\s*$", re.MULTILINE)


@dataclass
class GazetteItem:
    """A single gazette item (law/decision) within a PDF issue."""
    item_number: int
    mk_text: str = ""
    sq_text: str = ""
    is_bilingual: bool = False
    laser_doc_sim: float = 0.0
    valid: bool = True
    skip_reason: str = ""


@dataclass
class SegmentationResult:
    items: list[GazetteItem] = field(default_factory=list)
    total_items_mk: int = 0
    total_items_sq: int = 0
    matched_items: int = 0
    unmatched_mk: list[int] = field(default_factory=list)
    unmatched_sq: list[int] = field(default_factory=list)


class DocumentSegmenter:
    """Splits extracted text into individual gazette items and matches MK↔SQ."""

    def __init__(self, config: VezilkaConfig | None = None):
        self.cfg = config or DEFAULT_CONFIG
        self._laser_mk = None
        self._laser_sq = None

    def segment(self, mk_text: str, sq_text: str) -> SegmentationResult:
        """Segment MK and SQ text and return matched items."""
        mk_items = self._split_items(mk_text)
        sq_items = self._split_items(sq_text)

        result = SegmentationResult(
            total_items_mk=len(mk_items),
            total_items_sq=len(sq_items),
        )

        if not mk_items and not sq_items:
            # No item boundaries found → treat whole text as one item
            item = GazetteItem(
                item_number=0,
                mk_text=mk_text.strip(),
                sq_text=sq_text.strip(),
                is_bilingual=bool(mk_text.strip() and sq_text.strip()),
            )
            if item.is_bilingual:
                result.items.append(item)
                result.matched_items = 1
            return result

        # If only one side has items, still treat the whole text as a single block
        if not mk_items:
            mk_items = {0: mk_text.strip()}
        if not sq_items:
            sq_items = {0: sq_text.strip()}

        # Match items by number
        mk_nums = set(mk_items.keys())
        sq_nums = set(sq_items.keys())
        matched_nums = mk_nums & sq_nums
        result.unmatched_mk = sorted(mk_nums - sq_nums)
        result.unmatched_sq = sorted(sq_nums - mk_nums)

        for num in sorted(matched_nums):
            item = GazetteItem(
                item_number=num,
                mk_text=mk_items[num],
                sq_text=sq_items[num],
                is_bilingual=True,
            )
            result.items.append(item)

        result.matched_items = len(result.items)

        if result.unmatched_mk:
            logger.info("Unmatched MK items: %s", result.unmatched_mk)
        if result.unmatched_sq:
            logger.info("Unmatched SQ items: %s", result.unmatched_sq)

        return result

    def validate_with_laser(self, result: SegmentationResult) -> SegmentationResult:
        """Validate matched items using document-level LASER3 similarity.

        Items below threshold are flagged and skipped to avoid bad pairs.
        """
        if not result.items:
            return result

        mk_enc, sq_enc = self._get_laser_encoders()
        if mk_enc is None:
            logger.warning("LASER3 not available — skipping doc-level validation")
            return result

        validated = []
        for item in result.items:
            if not item.mk_text.strip() or not item.sq_text.strip():
                item.valid = False
                item.skip_reason = "empty_text"
                validated.append(item)
                continue

            try:
                mk_emb = mk_enc.encode_sentences([item.mk_text])
                sq_emb = sq_enc.encode_sentences([item.sq_text])
                sim = float(self._cosine_sim(mk_emb[0], sq_emb[0]))
                item.laser_doc_sim = sim

                if sim < self.cfg.segmenter_min_laser_doc_sim:
                    item.valid = False
                    item.skip_reason = f"low_doc_sim_{sim:.3f}"
                    logger.info(
                        "Item %d: doc-level LASER3 %.3f < %.2f — skipping",
                        item.item_number, sim, self.cfg.segmenter_min_laser_doc_sim,
                    )
            except Exception as e:
                logger.warning("LASER3 validation failed for item %d: %s",
                               item.item_number, e)
                # Keep item but flag
                item.laser_doc_sim = 0.0

            validated.append(item)

        result.items = validated
        return result

    # ── internals ───────────────────────────────────────────────

    @staticmethod
    def _split_items(text: str) -> dict[int, str]:
        """Split text into gazette items by item number boundaries."""
        if not text:
            return {}

        matches = list(ITEM_BOUNDARY.finditer(text))
        if not matches:
            return {}

        items: dict[int, str] = {}
        for i, m in enumerate(matches):
            num = int(m.group(1))
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                items[num] = body

        return items

    def _get_laser_encoders(self):
        """Lazy-load LASER3 encoders."""
        if self._laser_mk is None:
            try:
                from laser_encoders import LaserEncoderPipeline
                self._laser_mk = LaserEncoderPipeline(lang="mkd")
                self._laser_sq = LaserEncoderPipeline(lang="sq")
            except ImportError:
                logger.warning("laser-encoders not installed")
                self._laser_mk = False
                self._laser_sq = False
            except Exception as e:
                logger.error("Failed to load LASER3: %s", e)
                self._laser_mk = False
                self._laser_sq = False

        if self._laser_mk is False:
            return None, None
        return self._laser_mk, self._laser_sq

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32).flatten()
        b = np.asarray(b, dtype=np.float32).flatten()
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm < 1e-12:
            return 0.0
        return float(dot / norm)
