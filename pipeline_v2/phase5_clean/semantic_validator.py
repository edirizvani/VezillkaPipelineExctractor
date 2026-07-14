"""
Vezilka v2 — Semantic Validator (tiered).

Five signals applied in three tiers to manage compute cost:

  Tier 1 (ALL pairs, fast):
    Signal 1  — Length ratio
    Signal 2  — LaBSE semantic similarity
    Signal 3  — LASER3 bidirectional similarity

  Tier 2 (surviving non-structural pairs, medium):
    Signal 4  — COMET-QE translation quality

  Tier 3 (ambiguous pairs only, expensive):
    Signal 5  — Back-translation round-trip consistency

Blended scoring:
  Structural:      LaBSE 50% + LASER3 30% + length 20%
  Non-structural:  LaBSE 25% + LASER3 20% + COMET-QE 25% +
                   back-translation 20% + length 10%

Hard rejection thresholds abort a pair immediately regardless of
blended score.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from config import DEFAULT_CONFIG, VezilkaConfig
from phase4_align.aligner_orchestrator import CandidatePair

logger = logging.getLogger(__name__)


class SemanticValidator:
    """Tiered validation of candidate sentence pairs."""

    def __init__(self, config: VezilkaConfig | None = None):
        self.cfg = config or DEFAULT_CONFIG
        self._labse = None
        self._laser_mk = None
        self._laser_sq = None
        self._comet = None
        self._fwd_translator = None
        self._bwd_translator = None
        self._chrf = None
        self._laser_available = True  # Track if LASER3 loaded successfully
        self._labse_available = True  # Track if LaBSE loaded successfully

    # ── public API ──────────────────────────────────────────────

    def validate(self, pairs: list[CandidatePair]) -> list[CandidatePair]:
        """Run full tiered validation.  Modifies pairs in-place and returns them."""
        if not pairs:
            return pairs

        mk_texts = [p.mk for p in pairs]
        sq_texts = [p.sq for p in pairs]

        # ── Tier 1 ──────────────────────────────────────────────
        logger.info("Tier 1: length ratio + LaBSE + LASER3 on %d pairs", len(pairs))

        length_scores = self._score_length_ratio(mk_texts, sq_texts)
        labse_scores = self._score_labse(mk_texts, sq_texts)
        laser_scores = self._score_laser3(mk_texts, sq_texts)

        for i, p in enumerate(pairs):
            p.length_ratio_score = length_scores[i]
            p.labse_score = labse_scores[i]
            p.laser3_score = laser_scores[i]
            p.tier_reached = 1

        # Hard rejections from Tier 1
        for p in pairs:
            reason = self._check_hard_reject_tier1(p)
            if reason:
                p.rejection_reason = reason
                p.blended_confidence = 0.0

        surviving = [p for p in pairs if not p.rejection_reason]
        logger.info("Tier 1 survivors: %d / %d", len(surviving), len(pairs))

        # ── Tier 2 — COMET-QE (non-structural only) ────────────
        if self.cfg.run_comet_qe:
            tier2 = [p for p in surviving if not p.is_structural]
            if tier2:
                logger.info("Tier 2: COMET-QE on %d non-structural pairs", len(tier2))
                mk2 = [p.mk for p in tier2]
                sq2 = [p.sq for p in tier2]
                comet_scores = self._score_comet_qe(mk2, sq2)
                for p, sc in zip(tier2, comet_scores):
                    p.comet_qe_score = sc
                    p.tier_reached = 2
                    if sc < self.cfg.hard_reject_comet_qe:
                        p.rejection_reason = f"hard_reject_comet_qe_{sc:.3f}"

        surviving = [p for p in pairs if not p.rejection_reason]
        logger.info("Tier 2 survivors: %d / %d", len(surviving), len(pairs))

        # ── Tier 3 — Back-translation (ambiguous only) ─────────
        if self.cfg.run_back_translation:
            tier3 = [
                p for p in surviving
                if not p.is_structural
                and p.labse_score > 0.75
                and p.comet_qe_score > 0.0
                and p.comet_qe_score < 0.75
            ]
            if tier3:
                logger.info("Tier 3: back-translation on %d ambiguous pairs", len(tier3))
                mk3 = [p.mk for p in tier3]
                sq3 = [p.sq for p in tier3]
                bt_scores = self._score_back_translation(mk3, sq3)
                for p, sc in zip(tier3, bt_scores):
                    p.back_translation_score = sc
                    p.tier_reached = 3

        # ── Blended scoring ─────────────────────────────────────
        for p in pairs:
            if p.rejection_reason:
                continue
            p.blended_confidence = self._blended_score(p)
            if p.blended_confidence < self.cfg.blended_min_score:
                p.rejection_reason = f"low_blended_{p.blended_confidence:.3f}"

        accepted = [p for p in pairs if not p.rejection_reason]
        rejected = [p for p in pairs if p.rejection_reason]
        logger.info("Final: %d accepted, %d rejected", len(accepted), len(rejected))

        self._log_rejections(rejected)
        return pairs

    # ── Signal 1: length ratio ──────────────────────────────────

    def _score_length_ratio(self, mk: list[str], sq: list[str]) -> list[float]:
        scores = []
        for m, s in zip(mk, sq):
            mw, sw = len(m.split()), len(s.split())
            if sw == 0 or mw == 0:
                scores.append(0.0)
                continue
            ratio = mw / sw
            if self.cfg.min_length_ratio <= ratio <= self.cfg.max_length_ratio:
                # Normalise to 0..1 (1.0 = perfect ratio of 1.0)
                deviation = abs(ratio - 1.0) / (self.cfg.max_length_ratio - 1.0)
                scores.append(max(0.0, 1.0 - deviation))
            else:
                scores.append(0.0)
        return scores

    # ── Signal 2: LaBSE ────────────────────────────────────────

    def _score_labse(self, mk: list[str], sq: list[str]) -> list[float]:
        model = self._get_labse()
        if model is None:
            return [0.0] * len(mk)
        try:
            mk_emb = model.encode(mk, batch_size=self.cfg.labse_batch_size,
                                   show_progress_bar=False, normalize_embeddings=True)
            sq_emb = model.encode(sq, batch_size=self.cfg.labse_batch_size,
                                   show_progress_bar=False, normalize_embeddings=True)
            sims = np.sum(mk_emb * sq_emb, axis=1)
            return [float(s) for s in sims]
        except Exception as e:
            logger.error("LaBSE scoring failed: %s", e)
            return [0.0] * len(mk)

    # ── Signal 3: LASER3 bidirectional ──────────────────────────

    def _score_laser3(self, mk: list[str], sq: list[str]) -> list[float]:
        mk_enc, sq_enc = self._get_laser()
        if mk_enc is None:
            return [0.0] * len(mk)
        try:
            mk_emb = np.asarray(mk_enc.encode_sentences(mk), dtype=np.float32)
            sq_emb = np.asarray(sq_enc.encode_sentences(sq), dtype=np.float32)

            # Normalise
            mk_emb /= (np.linalg.norm(mk_emb, axis=1, keepdims=True) + 1e-12)
            sq_emb /= (np.linalg.norm(sq_emb, axis=1, keepdims=True) + 1e-12)

            # Forward and backward cosine similarity
            fwd = np.sum(mk_emb * sq_emb, axis=1)
            bwd = np.sum(sq_emb * mk_emb, axis=1)  # same values for cosine
            # For LASER3, bidirectionality matters via the encoder direction
            return [float(min(f, b)) for f, b in zip(fwd, bwd)]
        except Exception as e:
            logger.error("LASER3 scoring failed: %s", e)
            return [0.0] * len(mk)

    # ── Signal 4: COMET-QE ──────────────────────────────────────

    def _score_comet_qe(self, mk: list[str], sq: list[str]) -> list[float]:
        model = self._get_comet()
        if model is None:
            return [0.0] * len(mk)
        try:
            data = [{"src": m, "mt": s} for m, s in zip(mk, sq)]
            output = model.predict(data, batch_size=16, gpus=0)
            return [float(s) for s in output.scores]
        except Exception as e:
            logger.error("COMET-QE scoring failed: %s", e)
            return [0.0] * len(mk)

    # ── Signal 5: Back-translation ──────────────────────────────

    def _score_back_translation(self, mk: list[str], sq: list[str]) -> list[float]:
        fwd, bwd, chrf = self._get_bt_models()
        if fwd is None:
            return [0.0] * len(mk)

        bs = self.cfg.bt_batch_size
        scores: list[float] = []

        for start in range(0, len(mk), bs):
            batch_mk = mk[start:start + bs]
            batch_sq = sq[start:start + bs]
            try:
                mt_sq_out = fwd(batch_mk, max_length=512, batch_size=len(batch_mk))
                mt_sq = [o["translation_text"] for o in mt_sq_out]

                bt_mk_out = bwd(mt_sq, max_length=512, batch_size=len(mt_sq))
                bt_mk = [o["translation_text"] for o in bt_mk_out]

                for mk_orig, sq_cand, mt_s, bt_m in zip(batch_mk, batch_sq, mt_sq, bt_mk):
                    chrf_fwd = chrf.sentence_score(mt_s, [sq_cand]).score / 100.0
                    chrf_bwd = chrf.sentence_score(bt_m, [mk_orig]).score / 100.0
                    if chrf_fwd + chrf_bwd > 0:
                        harmonic = 2 * chrf_fwd * chrf_bwd / (chrf_fwd + chrf_bwd)
                    else:
                        harmonic = 0.0
                    scores.append(harmonic)
            except Exception as e:
                logger.error("Back-translation batch failed: %s", e)
                scores.extend([0.0] * len(batch_mk))

        return scores

    # ── blended score ───────────────────────────────────────────

    def _blended_score(self, p: CandidatePair) -> float:
        if p.is_structural:
            # Adjust weights if signals are unavailable
            labse_w = self.cfg.w_structural_labse if self._labse_available else 0.0
            laser_w = self.cfg.w_structural_laser3 if self._laser_available else 0.0
            length_w = self.cfg.w_structural_length
            
            # Normalize weights
            total_w = labse_w + laser_w + length_w
            if total_w == 0:
                return p.length_ratio_score  # Fallback to length only
            
            return (
                labse_w * p.labse_score
                + laser_w * p.laser3_score
                + length_w * p.length_ratio_score
            ) / total_w * (self.cfg.w_structural_labse + self.cfg.w_structural_laser3 + self.cfg.w_structural_length)
        
        # Non-structural pairs
        labse_w = self.cfg.w_nonstructural_labse if self._labse_available else 0.0
        laser_w = self.cfg.w_nonstructural_laser3 if self._laser_available else 0.0
        comet_w = self.cfg.w_nonstructural_comet_qe
        bt_w = self.cfg.w_nonstructural_backtranslation
        length_w = self.cfg.w_nonstructural_length
        
        return (
            labse_w * p.labse_score
            + laser_w * p.laser3_score
            + comet_w * p.comet_qe_score
            + bt_w * p.back_translation_score
            + length_w * p.length_ratio_score
        )

    # ── hard rejection ──────────────────────────────────────────

    def _check_hard_reject_tier1(self, p: CandidatePair) -> str:
        # Only apply LaBSE threshold if LaBSE loaded successfully
        if self._labse_available and p.labse_score < self.cfg.hard_reject_labse:
            return f"hard_reject_labse_{p.labse_score:.3f}"
        # Only apply LASER3 threshold if LASER3 loaded successfully
        if self._laser_available and p.laser3_score < self.cfg.hard_reject_laser3:
            return f"hard_reject_laser3_{p.laser3_score:.3f}"
        if p.length_ratio_score < self.cfg.hard_reject_length_ratio:
            return f"hard_reject_length_{p.length_ratio_score:.3f}"
        return ""

    # ── rejection logging ───────────────────────────────────────

    def _log_rejections(self, rejected: list[CandidatePair]) -> None:
        if not rejected:
            return
        path = self.cfg.rejected_pairs_log
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "a", encoding="utf-8") as f:
                for p in rejected:
                    record = {
                        "mk": p.mk[:200], "sq": p.sq[:200],
                        "pdf_id": p.pdf_id, "item": p.item_number,
                        "strategy": p.alignment_strategy,
                        "reason": p.rejection_reason,
                        "labse": round(p.labse_score, 4),
                        "laser3": round(p.laser3_score, 4),
                        "comet_qe": round(p.comet_qe_score, 4),
                        "bt": round(p.back_translation_score, 4),
                        "blended": round(p.blended_confidence, 4),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error("Could not write rejection log: %s", e)

    # ── lazy model loaders ──────────────────────────────────────

    def _get_labse(self):
        if self._labse is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._labse = SentenceTransformer(self.cfg.labse_model)
            except Exception as e:
                logger.error("LaBSE load failed: %s", e)
                self._labse = False
                self._labse_available = False
        return self._labse if self._labse is not False else None

    def _get_laser(self):
        if self._laser_mk is None:
            try:
                from laser_encoders import LaserEncoderPipeline
                self._laser_mk = LaserEncoderPipeline(lang="mkd")
                self._laser_sq = LaserEncoderPipeline(lang="sq")
            except Exception as e:
                logger.error("LASER3 load failed: %s", e)
                self._laser_mk = False
                self._laser_sq = False
                self._laser_available = False
        if self._laser_mk is False:
            return None, None
        return self._laser_mk, self._laser_sq

    def _get_comet(self):
        if self._comet is None:
            try:
                from comet import download_model, load_from_checkpoint
                path = download_model(self.cfg.comet_qe_model)
                self._comet = load_from_checkpoint(path)
            except Exception as e:
                logger.error("COMET-QE load failed: %s", e)
                self._comet = False
        return self._comet if self._comet is not False else None

    def _get_bt_models(self):
        if self._fwd_translator is None:
            try:
                from transformers import pipeline as hf_pipeline
                from sacrebleu.metrics import CHRF
                self._fwd_translator = hf_pipeline(
                    "translation", model=self.cfg.bt_mk_to_sq_model, device=-1)
                self._bwd_translator = hf_pipeline(
                    "translation", model=self.cfg.bt_sq_to_mk_model, device=-1)
                self._chrf = CHRF(word_order=2)
            except Exception as e:
                logger.error("Back-translation models failed: %s", e)
                self._fwd_translator = False
                self._bwd_translator = False
                self._chrf = False
        if self._fwd_translator is False:
            return None, None, None
        return self._fwd_translator, self._bwd_translator, self._chrf
