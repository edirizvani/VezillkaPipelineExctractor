"""
Vezilka v2 — Deduplicator.

Uses MinHash LSH (datasketch) to detect near-duplicate pairs.
Two pairs are duplicates if their MK *or* SQ side exceeds the
minhash similarity threshold.
"""

from __future__ import annotations

import logging
from typing import Optional

from config import DEFAULT_CONFIG, VezilkaConfig
from phase4_align.aligner_orchestrator import CandidatePair

logger = logging.getLogger(__name__)


class Deduplicator:
    """MinHash-LSH near-duplicate removal for sentence pairs."""

    def __init__(self, config: VezilkaConfig | None = None):
        self.cfg = config or DEFAULT_CONFIG

    def deduplicate(self, pairs: list[CandidatePair]) -> list[CandidatePair]:
        """Remove near-duplicate pairs by MK-side and SQ-side MinHash."""
        if not pairs:
            return pairs

        try:
            from datasketch import MinHash, MinHashLSH
        except ImportError:
            logger.warning("datasketch not installed — skipping deduplication")
            return pairs

        threshold = self.cfg.minhash_threshold
        num_perm = self.cfg.minhash_num_perm

        # Build LSH index on MK side
        mk_lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        sq_lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        mk_hashes: dict[int, MinHash] = {}
        sq_hashes: dict[int, MinHash] = {}

        for i, p in enumerate(pairs):
            mk_mh = self._minhash(p.mk, num_perm)
            sq_mh = self._minhash(p.sq, num_perm)
            mk_hashes[i] = mk_mh
            sq_hashes[i] = sq_mh

        # Insert into LSH and find duplicates
        seen_mk: set[int] = set()
        seen_sq: set[int] = set()
        keep_indices: list[int] = []

        for i, p in enumerate(pairs):
            # Check MK duplicates
            mk_key = f"mk_{i}"
            mk_dupes = set()
            try:
                mk_dupes = set(mk_lsh.query(mk_hashes[i]))
            except Exception:
                pass

            sq_key = f"sq_{i}"
            sq_dupes = set()
            try:
                sq_dupes = set(sq_lsh.query(sq_hashes[i]))
            except Exception:
                pass

            if mk_dupes or sq_dupes:
                # Duplicate found — skip this pair
                continue

            # Not a duplicate — keep and insert
            try:
                mk_lsh.insert(mk_key, mk_hashes[i])
            except ValueError:
                pass  # already inserted
            try:
                sq_lsh.insert(sq_key, sq_hashes[i])
            except ValueError:
                pass
            keep_indices.append(i)

        removed = len(pairs) - len(keep_indices)
        logger.info("Deduplication: %d → %d pairs (removed %d near-duplicates)",
                     len(pairs), len(keep_indices), removed)
        return [pairs[i] for i in keep_indices]

    @staticmethod
    def _minhash(text: str, num_perm: int = 128):
        from datasketch import MinHash
        mh = MinHash(num_perm=num_perm)
        # Character 3-gram shingles
        text = text.lower().strip()
        for i in range(len(text) - 2):
            mh.update(text[i:i + 3].encode("utf-8"))
        return mh
