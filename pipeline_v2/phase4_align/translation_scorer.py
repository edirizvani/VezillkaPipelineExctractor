"""
Vezilka v2 — Translation Scorer (MarianMT / chrF++).

Provides a quick forward-translation quality signal by translating
MK → SQ with MarianMT and computing chrF++ between the MT output
and the actual SQ candidate.  Used inside the orchestrator as an
additional scoring dimension.
"""

from __future__ import annotations

import logging
from typing import Optional

from config import DEFAULT_CONFIG, VezilkaConfig

logger = logging.getLogger(__name__)


class TranslationScorer:
    """Score (mk, sq) pairs by forward-translating MK → SQ + chrF++."""

    def __init__(self, config: VezilkaConfig | None = None):
        self.cfg = config or DEFAULT_CONFIG
        self._translator = None
        self._chrf = None

    def score_pairs(
        self,
        mk_sentences: list[str],
        sq_sentences: list[str],
        batch_size: int | None = None,
    ) -> list[float]:
        """Return chrF++ scores for each (mk, sq) pair."""
        if not mk_sentences:
            return []

        bs = batch_size or self.cfg.translation_batch_size
        translator = self._get_translator()
        chrf = self._get_chrf()

        if translator is None or chrf is None:
            logger.warning("Translation scorer unavailable — returning 0.0")
            return [0.0] * len(mk_sentences)

        scores: list[float] = []
        for start in range(0, len(mk_sentences), bs):
            batch_mk = mk_sentences[start:start + bs]
            batch_sq = sq_sentences[start:start + bs]
            try:
                outputs = translator(batch_mk, max_length=512, batch_size=len(batch_mk))
                mt_sq = [o["translation_text"] for o in outputs]
            except Exception as e:
                logger.error("MarianMT translation failed: %s", e)
                scores.extend([0.0] * len(batch_mk))
                continue

            for mt_sent, ref_sent in zip(mt_sq, batch_sq):
                try:
                    s = chrf.sentence_score(mt_sent, [ref_sent]).score / 100.0
                except Exception:
                    s = 0.0
                scores.append(s)

        return scores

    # ── lazy loaders ────────────────────────────────────────────

    def _get_translator(self):
        if self._translator is None:
            try:
                from transformers import pipeline as hf_pipeline
                self._translator = hf_pipeline(
                    "translation",
                    model=self.cfg.bt_mk_to_sq_model,
                    device=-1,
                )
            except Exception as e:
                logger.error("Could not load MarianMT: %s", e)
                self._translator = False
        return self._translator if self._translator is not False else None

    def _get_chrf(self):
        if self._chrf is None:
            try:
                from sacrebleu.metrics import CHRF
                self._chrf = CHRF(word_order=2)
            except ImportError:
                logger.error("sacrebleu not installed")
                self._chrf = False
        return self._chrf if self._chrf is not False else None
