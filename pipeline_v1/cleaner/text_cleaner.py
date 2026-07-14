"""
Text Cleaner — normalization, filtering, and deduplication of sentence pairs.

Handles Macedonian Cyrillic and Albanian Latin text, including common
OCR artifacts, header/footer removal, and near-duplicate detection.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ───────────────────────── Noise patterns ───────────────────────

# Macedonian page headers / footers to strip
_MK_NOISE_PATTERNS = [
    re.compile(r"Службен\s+весник\s+на\s+Р[СC]?М?.*", re.IGNORECASE),
    re.compile(
        r"Службен\s+весник\s+на\s+Република\s+(Северна\s+)?Македонија.*",
        re.IGNORECASE,
    ),
    re.compile(r"бр\.\s*\d+.*"),
    re.compile(r"стр\.\s*\d+.*"),
    re.compile(r"^\d+\s*$"),
    re.compile(r"^[-–—]+\s*\d*\s*[-–—]*$"),
]

# Albanian page headers / footers to strip
_SQ_NOISE_PATTERNS = [
    re.compile(r"Gazeta\s+Zyrtare\s+e\s+R[SM]?.*", re.IGNORECASE),
    re.compile(r"Gazeta\s+Zyrtare\s+nr\.?\s*\d*.*", re.IGNORECASE),
    re.compile(r"Gaceta\s+Zyrtare.*", re.IGNORECASE),
    re.compile(r"nr\.\s*\d+.*", re.IGNORECASE),
    re.compile(r"fq\.\s*\d+.*", re.IGNORECASE),
    re.compile(r"^\d+\s*$"),
    re.compile(r"^[-–—]+\s*\d*\s*[-–—]*$"),
]

# Common Cyrillic OCR confusions
_MK_OCR_FIXES = {
    "ё": "е",   # Macedonian doesn't use ё
    "й": "и",   # rare confusion
}


# ───────────────────────── Filter result ────────────────────────

@dataclass
class FilterResult:
    is_valid: bool
    reason: str


# ───────────────────────── TextCleaner ──────────────────────────

class TextCleaner:
    """
    Cleans and filters Macedonian ↔ Albanian sentence pairs.

    Usage::

        cleaner = TextCleaner()
        mk = cleaner.clean_macedonian(raw_mk)
        sq = cleaner.clean_albanian(raw_sq)
        result = cleaner.filter_pair(mk, sq)
        if result.is_valid:
            ...
    """

    def __init__(
        self,
        min_words: int = 5,
        max_words: int = 200,
        min_length_ratio: float = 0.4,
        max_length_ratio: float = 2.5,
        max_number_fraction: float = 0.30,
        max_consecutive_upper: int = 5,
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.min_length_ratio = min_length_ratio
        self.max_length_ratio = max_length_ratio
        self.max_number_fraction = max_number_fraction
        self.max_consecutive_upper = max_consecutive_upper
        self._seen_hashes: set[str] = set()

    # ── Macedonian cleaning ─────────────────────────────────────

    def clean_macedonian(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        for old, new in _MK_OCR_FIXES.items():
            text = text.replace(old, new)

        lines = text.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if any(p.fullmatch(stripped) for p in _MK_NOISE_PATTERNS):
                continue
            cleaned.append(stripped)

        return self._normalize_whitespace("\n".join(cleaned))

    # ── Albanian cleaning ───────────────────────────────────────

    def clean_albanian(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)

        lines = text.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if any(p.fullmatch(stripped) for p in _SQ_NOISE_PATTERNS):
                continue
            cleaned.append(stripped)

        return self._normalize_whitespace("\n".join(cleaned))

    # ── Pair filtering ──────────────────────────────────────────

    def filter_pair(self, mk_sentence: str, sq_sentence: str) -> FilterResult:
        mk, sq = mk_sentence.strip(), sq_sentence.strip()

        if not mk or not sq:
            return FilterResult(False, "empty_sentence")

        mk_words, sq_words = mk.split(), sq.split()
        n_mk, n_sq = len(mk_words), len(sq_words)

        if n_mk < self.min_words or n_sq < self.min_words:
            return FilterResult(False, f"too_short(mk={n_mk},sq={n_sq})")
        if n_mk > self.max_words or n_sq > self.max_words:
            return FilterResult(False, f"too_long(mk={n_mk},sq={n_sq})")

        ratio = n_mk / n_sq if n_sq > 0 else 999
        if ratio < self.min_length_ratio or ratio > self.max_length_ratio:
            return FilterResult(False, f"bad_length_ratio({ratio:.2f})")

        mk_nums = sum(1 for w in mk_words if re.fullmatch(r"[\d.,]+", w))
        sq_nums = sum(1 for w in sq_words if re.fullmatch(r"[\d.,]+", w))
        if n_mk and mk_nums / n_mk > self.max_number_fraction:
            return FilterResult(False, "mk_too_many_numbers")
        if n_sq and sq_nums / n_sq > self.max_number_fraction:
            return FilterResult(False, "sq_too_many_numbers")

        if self._has_consecutive_upper(mk_words, self.max_consecutive_upper):
            return FilterResult(False, "mk_consecutive_uppercase")
        if self._has_consecutive_upper(sq_words, self.max_consecutive_upper):
            return FilterResult(False, "sq_consecutive_uppercase")

        pair_hash = self._hash_pair(mk, sq)
        if pair_hash in self._seen_hashes:
            return FilterResult(False, "exact_duplicate")
        self._seen_hashes.add(pair_hash)

        return FilterResult(True, "ok")

    # ── Deduplication ───────────────────────────────────────────

    def deduplicate(self, pairs: list[dict]) -> tuple[list[dict], int]:
        """
        Remove exact and near-duplicate pairs.
        Each dict must have ``"mk"`` and ``"sq"`` keys.
        Returns ``(deduplicated_pairs, num_removed)``.
        """
        # Pass 1: exact dedup
        seen: set[str] = set()
        exact_deduped: list[dict] = []
        exact_removed = 0
        for pair in pairs:
            h = self._hash_pair(pair["mk"], pair["sq"])
            if h not in seen:
                seen.add(h)
                exact_deduped.append(pair)
            else:
                exact_removed += 1

        # Pass 2: near-dup via MinHash
        near_removed = 0
        try:
            final = self._minhash_dedup(exact_deduped)
            near_removed = len(exact_deduped) - len(final)
        except ImportError:
            logger.warning("datasketch not installed — skipping near-dup removal")
            final = exact_deduped

        total_removed = exact_removed + near_removed
        logger.info(
            "Dedup: %d exact + %d near removed, %d remaining",
            exact_removed, near_removed, len(final),
        )
        return final, total_removed

    def _minhash_dedup(
        self, pairs: list[dict], threshold: float = 0.95, num_perm: int = 128
    ) -> list[dict]:
        from datasketch import MinHash, MinHashLSH

        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        duplicates: set[int] = set()

        for idx, pair in enumerate(pairs):
            combined = pair["mk"] + " ||| " + pair["sq"]
            mh = MinHash(num_perm=num_perm)
            for i in range(len(combined) - 2):
                mh.update(combined[i : i + 3].encode("utf-8"))

            key = f"pair_{idx}"
            result = lsh.query(mh)
            if result:
                duplicates.add(idx)
            else:
                lsh.insert(key, mh)

        return [p for i, p in enumerate(pairs) if i not in duplicates]

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        lines = text.split("\n")
        return "\n".join(
            re.sub(r"[ \t]+", " ", line).strip()
            for line in lines if line.strip()
        )

    @staticmethod
    def _has_consecutive_upper(words: list[str], max_run: int) -> bool:
        run = 0
        for w in words:
            if w.isupper() and len(w) > 1:
                run += 1
                if run > max_run:
                    return True
            else:
                run = 0
        return False

    @staticmethod
    def _hash_pair(mk: str, sq: str) -> str:
        combined = mk.strip().lower() + " ||| " + sq.strip().lower()
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()
