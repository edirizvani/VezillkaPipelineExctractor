# Vezilka v2 — MK ↔ SQ Parallel Corpus Pipeline

A complete rewrite of the Vezilka pipeline that creates a sentence-level
Macedonian ↔ Albanian parallel corpus from bilingual PDF documents published
by the Službен Весник (Official Gazette of North Macedonia).

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        PIPELINE.PY                               │
│                   Master Orchestrator                             │
└─────────┬────────────┬────────────┬────────────┬─────────────────┘
          │            │            │            │
  ┌───────▼───────┐   │   ┌────────▼────────┐   │
  │  PHASE 3      │   │   │  PHASE 4        │   │
  │  EXTRACT      │   │   │  ALIGN          │   │
  │               │   │   │                 │   │
  │ Gatekeeper    │   │   │ Structural      │   │
  │ LayoutClassif │   │   │ Dense Retrieval │   │
  │ DocSegmenter  │   │   │ Gale-Church     │   │
  │ TwoColumn ────┼───┘   │ Orchestrator    │   │
  │ Sequential    │       └────────┬────────┘   │
  │ OCR/Mixed     │                │            │
  └───────────────┘       ┌────────▼────────┐   │
                          │  PHASE 5        │   │
          ┌───────────┐   │  CLEAN          │   │
          │  UTILS    │   │                 │   │
          │           │   │ SemanticValid.  │───┘
          │ config.py │   │ NoiseFilter     │
          │ text.py   │   │ PairFilter      │
          │ logging   │   │ Deduplicator    │
          └───────────┘   │ Exporter        │
                          └─────────────────┘
```

## Three PDF Layout Types

| Type | Name | Description | Detection |
|------|------|-------------|-----------|
| A | **Two-Column** | MK left column, SQ right column on same page | >50% pages have bilingual column split |
| B | **Sequential** | Full MK block then full SQ block | Cyrillic→Latin transition + Albanian boundary markers |
| C | **Mixed/Pre-2019** | Interleaved MK/SQ, inconsistent | Fallback when A and B don't match |

## Three Alignment Strategies

| Strategy | Reliability | Trigger | Method |
|----------|------------|---------|--------|
| **Structural** | Highest | Член N / Neni N present | Article-number matching + sentence DP |
| **Dense Retrieval** | Medium | No articles, enough text | LASER3 mutual-NN + FAISS + monotonicity |
| **Gale-Church** | Lowest | <10 sents or dense fails | Character-length DP |

## Five Validation Signals (Tiered)

### Tier 1 — All pairs (fast)
1. **Length ratio** — word count ratio between 0.4–2.5
2. **LaBSE** — semantic similarity (hard reject < 0.55)
3. **LASER3** — bidirectional translation similarity (hard reject < 0.55)

### Tier 2 — Non-structural pairs (medium cost)
4. **COMET-QE** — translation quality estimation (hard reject < 0.65)

### Tier 3 — Ambiguous pairs only (expensive)
5. **Back-translation** — round-trip MK→SQ→MK consistency via chrF++

### Blended Scoring

**Structural pairs:** LaBSE 50% + LASER3 30% + length 20%

**Non-structural pairs:** LaBSE 25% + LASER3 20% + COMET-QE 25% + back-translation 20% + length 10%

Final threshold: blended score ≥ 0.70

## Quick Start

```bash
# Install dependencies
cd pipeline_v2
pip install -r requirements.txt

# Run full pipeline (all 4,931 PDFs)
python pipeline.py

# Test with a small batch first
python pipeline.py --limit 10

# Fast run without expensive semantic validation
python pipeline.py --limit 50 --skip-validation

# Debug mode
python pipeline.py --limit 5 --log-level DEBUG
```

## Configuration

All parameters live in `config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pdf_dir` | `../pipeline_v1/data/pdfs/` | Path to existing PDFs |
| `gatekeeper_min_file_size_kb` | 50 | Skip PDFs smaller than this |
| `dense_retrieval_min_similarity` | 0.70 | LASER3 mutual-NN threshold |
| `blended_min_score` | 0.70 | Final acceptance threshold |
| `hard_reject_labse` | 0.55 | Absolute floor for LaBSE |
| `hard_reject_comet_qe` | 0.65 | Absolute floor for COMET-QE |
| `minhash_threshold` | 0.95 | Near-duplicate detection |

## Output Schema

Each pair in the final output:

```json
{
  "mk": "Македонска реченица...",
  "sq": "Fjali shqipe...",
  "pdf_id": "2021_01_abc123",
  "item_number": 262,
  "article_number": 5,
  "alignment_strategy": "structural",
  "layout_type": "sequential",
  "labse_score": 0.91,
  "laser3_score": 0.88,
  "comet_qe_score": 0.79,
  "back_translation_score": 0.71,
  "length_ratio_score": 0.95,
  "blended_confidence": 0.87,
  "mk_word_count": 23,
  "sq_word_count": 25,
  "tier_reached": 2
}
```

## Checkpointing

The pipeline is fully resumable:

| Phase | Checkpoint Location | Format |
|-------|-------------------|--------|
| Extract | `data/extracted/{pdf_id}.json` | JSON metadata |
| Segment | `data/segmented/{pdf_id}.json` | Segmented items |
| Align | `data/aligned/{pdf_id}.jsonl` | Raw candidate pairs |
| Output | `data/output/corpus.tsv` | Final cleaned corpus |

Re-running `pipeline.py` will skip already-processed PDFs.

## Audit Logs

| Log File | Contents |
|----------|----------|
| `data/skipped_pdfs.jsonl` | PDFs skipped by gatekeeper (too small, no Albanian, scanned) |
| `data/failed_pdfs.jsonl` | PDFs that failed during processing (with error + phase) |
| `data/rejected_pairs.jsonl` | Pairs rejected by semantic validation (with reason + scores) |

## Known Limitations

- **Pre-2019 PDFs**: Lower extraction quality due to inconsistent layouts
- **OCR fallback**: Requires `easyocr` + `pdf2image` (not installed by default)
- **COMET-QE + back-translation**: Slow on CPU; recommend GPU for full corpus run
- **MK↔SQ translation models**: Helsinki-NLP opus-mt models have limited quality for this pair
- **LASER3**: Model download required on first run (~2 GB)
