# Vezilka — Macedonian ↔ Albanian Parallel Corpus Builder

**Vezilka** (Везилка — "embroidery needle" in Macedonian) is a research pipeline that
extracts, aligns and exports sentence-level parallel data from the
**Služben Vesnik** (Official Gazette of North Macedonia) PDFs.

The gazette is published bilingually in Macedonian (Cyrillic) and Albanian (Latin),
making it a unique source of aligned legal text for low-resource NLP.

---

## Quick Start

```bash
# 1. Clone / enter the project
cd pipeline_v1

# 2. Create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (2019+ issues — best bilingual coverage)
python pipeline.py --phase all --year-start 2019 --year-end 2025
```

---

## Project Structure

```
pipeline_v1/
├── config.py                 # Central configuration (paths, URLs, thresholds)
├── pipeline.py               # Master CLI orchestrator
├── validate.py               # Post-pipeline quality checks
├── requirements.txt          # Python dependencies
│
├── scraper/
│   ├── catalog_scraper.py    # Scrapes issue metadata from slvesnik.com.mk
│   └── pdf_downloader.py     # Downloads PDFs with rate limiting & retries
│
├── extractor/
│   ├── pdf_extractor.py      # Coordinate-aware PDF text extraction (pdfplumber)
│   ├── language_detector.py  # Script + ML language detection (MK/SQ)
│   └── layout_analyzer.py    # High-level bilingual extraction orchestrator
│
├── aligner/
│   ├── structural_aligner.py # Article-number matching (Член N / Neni N)
│   └── statistical_aligner.py# Gale-Church sentence alignment fallback
│
├── cleaner/
│   └── text_cleaner.py       # Noise removal, filtering, deduplication
│
├── exporter/
│   └── dataset_exporter.py   # Export to TSV, JSONL, HuggingFace datasets
│
└── data/                     # Created at runtime
    ├── pdfs/                 # Downloaded gazette PDFs (by year)
    ├── extracted/            # Per-PDF extracted JSON
    ├── aligned/              # Aligned sentence pairs JSON
    └── export/               # Final TSV / JSONL / HuggingFace output
```

---

## Pipeline Phases

| Phase      | Command                                                 | Description                                      |
| ---------- | ------------------------------------------------------- | ------------------------------------------------ |
| `scrape`   | `python pipeline.py --phase scrape --year-start 2019`   | Scrape issue URLs from the website               |
| `download` | `python pipeline.py --phase download --limit 50`        | Download PDFs with polite rate limiting           |
| `extract`  | `python pipeline.py --phase extract`                    | Extract bilingual text from PDFs (layout-aware)  |
| `align`    | `python pipeline.py --phase align`                      | Align MK ↔ SQ at article + sentence level       |
| `export`   | `python pipeline.py --phase export --format tsv,jsonl`  | Export to TSV / JSONL / HuggingFace datasets     |
| `all`      | `python pipeline.py --phase all`                        | Run all phases end-to-end                        |
| `test`     | `python pipeline.py --phase test --pdf path/to/file.pdf`| Quick test on a single PDF                       |

---

## Testing on a Single PDF

```bash
python pipeline.py --phase test --pdf data/pdfs/2023/someissue.pdf
```

This runs extract → clean → align → filter → print sample on one file,
ideal for debugging without downloading the full catalog.

---

## Validation

After the pipeline completes:

```bash
python validate.py --sample 50
```

This runs five quality checks:

1. **Spot-check** — prints 50 random pairs for human review
2. **Script check** — verifies MK text is Cyrillic, SQ text is Latin
3. **Length histogram** — word-count distributions (PNG saved to `data/export/`)
4. **Duplicate rate** — exact-duplicate percentage
5. **Coverage** — what % of downloaded PDFs produced ≥1 valid pair

---

## Output Formats

### TSV (`vezilka_mk_sq.tsv`)
```
mk_sentence\tsq_sentence\tsource\tarticle\tconfidence\tmethod
Законот стапува на сила...\tLigji hyn në fuqi...\tslvesnik_abc123\t42\t0.95\tstructural_1:1
```

### JSONL (`vezilka_mk_sq.jsonl`)
```json
{"mk": "...", "sq": "...", "meta": {"source": "...", "article": "42", "confidence": 0.95, "method": "structural_1:1"}}
```

### HuggingFace (`huggingface/`)
Standard `datasets.DatasetDict` with train / validation / test splits (80 / 10 / 10).

```python
from datasets import load_from_disk
ds = load_from_disk("data/export/huggingface")
print(ds["train"][0])
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **pdfplumber** for extraction | Coordinate-aware word boxes enable column detection |
| **Two-column detection** via midpoint split | Post-2019 layout places MK left, SQ right |
| **Sequential-block detection** via script transitions | Pre-2019 layout interleaves MK/SQ blocks |
| **Article-number alignment** as primary method | Articles numbered identically (Член N = Neni N) |
| **Gale-Church** as fallback | Handles preambles and non-article text |
| **MinHash deduplication** | Catches near-duplicates (threshold 0.95) beyond exact matching |
| **Aggressive noise filtering** | Removes headers, footers, page numbers, OCR artefacts |

---

## Domain Notes

- **Albanian content** appears from **2001** onward (Ohrid Framework Agreement).
- **Systematic bilingual publication** became standard from **2019** (Law on Use of Languages).
- Best yield is from **2019–present** issues (two-column layout, consistent formatting).
- Pre-2019 issues may have scanned/OCR'd pages with lower quality.

---

## Configuration

All thresholds are in [`config.py`](config.py):

| Setting | Default | Description |
|---------|---------|-------------|
| `MIN_YEAR` | 2001 | Earliest year to scrape |
| `COLUMN_SPLIT_TOLERANCE` | 0.08 | Midpoint tolerance for column detection |
| `CYRILLIC_THRESHOLD` | 0.60 | Cyrillic ratio to classify as Macedonian |
| `MIN_SENTENCE_WORDS` | 5 | Minimum words per sentence |
| `GALE_CHURCH_MEAN_RATIO` | 1.1 | MK/SQ character-length mean ratio |
| `NEAR_DUPLICATE_THRESHOLD` | 0.95 | MinHash Jaccard threshold |

---

## Requirements

- Python ≥ 3.10
- ~2 GB disk for full PDF corpus (2019–2025)
- Internet access for scraping / downloading

See [`requirements.txt`](requirements.txt) for all dependencies.

---

## License

Academic research use.  The gazette text is published by the Government of
North Macedonia as public law.

---

## Citation

If you use this corpus in your research, please cite:

```bibtex
@misc{vezilka2025,
  title   = {Vezilka: A Macedonian–Albanian Parallel Corpus from the Official Gazette},
  year    = {2025},
  note    = {Built from Služben Vesnik (slvesnik.com.mk)},
}
```
