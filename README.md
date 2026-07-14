# Vezilka — Macedonian ↔ Albanian Parallel Corpus Pipeline

Builds a sentence-level **Macedonian ↔ Albanian** parallel corpus from the bilingual PDFs published
by *Службен весник на Република Северна Македонија* (the Official Gazette of North Macedonia).

Each gazette issue contains the same legal text in both languages, which makes the archive a large
source of naturally parallel MK/SQ text — a pair of languages with very little public parallel data.
The pipeline scrapes the archive, extracts the two language streams out of each PDF, aligns them at
the sentence level, and filters the result down to a usable training corpus.

**Current output:** ~81,000 aligned sentence pairs from ~4,900 gazette issues.

## Two pipelines

The repo contains two generations of the pipeline. **`pipeline_v2/` is the current one**; `v1` is
kept because its scraper/downloader still supplies the PDFs and its notebooks document the corpus
analysis.

| | `pipeline_v1/` | `pipeline_v2/` |
|---|---|---|
| Role | Scraping, downloading, first working corpus | Full rewrite of extraction → alignment → cleaning |
| Stages | `scraper/` → `extractor/` → `aligner/` → `cleaner/` → `exporter/` | `phase3_extract/` → `phase4_align/` → `phase5_clean/` |
| Alignment | Structural + statistical + semantic | Structural, Gale-Church, and dense retrieval (LaBSE), combined by an orchestrator |
| Still used for | The scraper and the PDF archive it produced | Everything downstream |

`pipeline_v2` reads its input PDFs from `../pipeline_v1/data/pdfs/` (see `pdf_dir` in
`pipeline_v2/config.py`).

## Architecture (v2)

```
                            pipeline.py  (orchestrator)
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
  PHASE 3 — EXTRACT           PHASE 4 — ALIGN            PHASE 5 — CLEAN
  Gatekeeper                  Structural aligner         Semantic validator
  LayoutClassifier            Gale-Church                NoiseFilter
  DocumentSegmenter           Dense retrieval (LaBSE)    PairFilter
  TwoColumn / Sequential      Translation scorer         Deduplicator
  OCR / Mixed extractors      Orchestrator               Exporter
```

A gatekeeper first decides whether an issue is genuinely bilingual; a layout classifier then picks
the right extractor, since issues vary between two-column and sequential (MK section followed by SQ
section) layouts.

## Setup

Each pipeline has its own dependencies:

```bash
cd pipeline_v2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python pipeline.py
```

Some v2 stages (LaBSE embeddings, COMET scoring, back-translation) want a GPU — see
`pipeline_v2/notebooks/` for the Colab runners.

## The corpus

**The released corpora are in this repo** — see **[`DATASET.md`](DATASET.md)** for schema and stats.

- `pipeline_v2/data/output/*.tsv` — the current corpus: **81,351 pairs**, with train/val/test splits
  and per-pair quality scores (LaBSE, LASER3, COMET-QE, back-translation)
- `pipeline_v1/data/export/*.tsv` — the earlier v1 corpus (100,898 pairs), via **Git LFS**

```bash
git lfs install && git lfs pull   # needed only for the v1 files
```

The *source* data is not in the repo: 31 GB of gazette PDFs plus intermediate artifacts, ~34 GB in
all, deliberately gitignored. The scraped catalogs (`pipeline_v1/data/catalog*.json`) and two sample
PDFs are committed so you can re-download and re-run without re-scraping the site. See
`pipeline_v1/data/README.md` and `pipeline_v2/data/README.md` for the expected layout.

## Credentials

The GCS upload scripts read a service-account key path from `GOOGLE_APPLICATION_CREDENTIALS`.
Copy `.env.example` to `.env` and set it. Key files (`vezilka-*.json`) are gitignored — never
commit one.

## Layout

```
├── docs/                  # presentation, design notes
├── pipeline_v1/
│   ├── scraper/ extractor/ aligner/ cleaner/ exporter/
│   ├── notebooks/         # corpus analysis + cleaning notebooks
│   ├── scripts/           # one-off diagnostics, validation, GCS upload
│   └── pipeline.py  config.py  build_dataset.py
└── pipeline_v2/
    ├── phase3_extract/ phase4_align/ phase5_clean/ utils/
    ├── notebooks/         # Colab runners
    ├── tests/
    └── pipeline.py  config.py
```
