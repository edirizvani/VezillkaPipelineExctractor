# 🧵 Vezilka — Macedonian ↔ Albanian Parallel Corpus Builder

### A Research Pipeline for Low-Resource NLP

---

## 📌 Project Overview

**Vezilka** (Везилка — "embroidery needle" in Macedonian) is an end-to-end automated pipeline that builds a **sentence-level parallel corpus** for the **Macedonian–Albanian** language pair — one of the most under-resourced pairs in NLP.

The data source is the **Služben Vesnik** (Official Gazette of North Macedonia), which has been published bilingually since the 2001 Ohrid Framework Agreement and in a standardised two-column layout since 2019.

> **Goal:** Extract, align, clean, and export high-quality MK ↔ SQ sentence pairs suitable for training machine translation models.

---

## 🏛️ Data Source

| Detail | Value |
|--------|-------|
| **Source** | Služben Vesnik (slvesnik.com.mk) — Official Gazette of North Macedonia |
| **Format** | PDF (legally published, public domain government text) |
| **Languages** | Macedonian (Cyrillic) & Albanian (Latin) |
| **Year range** | 2001 – 2025 (best coverage from 2019 onward) |
| **Domain** | Legal / legislative text |

---

## ⚙️ Pipeline Architecture

The system is built as a **5-phase modular pipeline**, orchestrated by a single CLI entry point:

```
┌──────────┐    ┌────────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐
│  SCRAPE  │───▶│  DOWNLOAD  │───▶│  EXTRACT  │───▶│  ALIGN   │───▶│  EXPORT  │
│ Catalog  │    │   PDFs     │    │ Text+Lang │    │ Sentence │    │ TSV/JSONL│
│ metadata │    │ from web   │    │ detection │    │  pairs   │    │ HF Dataset│
└──────────┘    └────────────┘    └───────────┘    └──────────┘    └──────────┘
```

### Phase 1 — Scrape
- Crawls the Služben Vesnik website for PDF issue metadata
- Handles a **JavaScript-driven (AngularJS)** year selector via headless browser
- Builds a resumable JSON catalog of all gazette issues

### Phase 2 — Download
- Downloads PDFs with polite rate-limiting and jitter
- **20 concurrent threads** for throughput
- Retry logic with exponential backoff (max 3 retries)

### Phase 3 — Extract
- **Coordinate-aware text extraction** from PDF pages
- Detects two-column newspaper layout via word-box midpoint analysis
- Classifies each page by language (Macedonian / Albanian / other)
- Concatenates MK and SQ text sections separately

### Phase 4 — Align
- **Primary method:** Structural alignment by article numbers (`Член N` = `Neni N`)
- **Fallback method:** Gale-Church (1993) statistical sentence alignment
- Cleaning, filtering, and MinHash near-duplicate removal

### Phase 5 — Export
- Outputs to **TSV**, **JSON Lines**, and **HuggingFace `datasets`** format
- Automatic 80/10/10 train / validation / test splits

---

## 🛠️ Technology Stack

### Language & Runtime
| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Core language for the entire pipeline |

### Web Scraping
| Library | Purpose |
|---------|---------|
| **Playwright** | Headless Chromium browser for scraping JavaScript-rendered pages |
| **BeautifulSoup4** | HTML parsing for extracting PDF links and metadata |
| **lxml** | Fast XML/HTML parser backend |
| **Requests** | HTTP client for downloading PDFs |

### PDF Processing
| Library | Purpose |
|---------|---------|
| **pdfplumber** | Coordinate-aware text extraction (word bounding boxes for column detection) |
| **PyMuPDF (fitz)** | Complementary PDF parsing and rendering |

### Language Detection
| Library | Purpose |
|---------|---------|
| **lingua-language-detector** | High-accuracy ML-based detection for Macedonian, Albanian, Serbian |
| **langdetect** | Fallback broad-coverage language identification |
| Unicode script analysis | Fast Cyrillic vs. Latin classification via Unicode ranges |

### Text Alignment
| Library / Algorithm | Purpose |
|---------------------|---------|
| **Structural alignment** (custom) | Article-number matching (`Член N` → `Neni N`) for legal text |
| **Gale-Church (1993)** (custom) | Statistical sentence alignment by character-length ratios |
| **sacremoses** | Moses tokenizer for sentence segmentation |
| **sentence-splitter** | Rule-based sentence boundary detection |
| **regex** | Advanced regex engine for complex pattern matching |

### Text Cleaning & Deduplication
| Library | Purpose |
|---------|---------|
| **datasketch (MinHash LSH)** | Near-duplicate detection with Jaccard similarity (threshold 0.95) |
| **unicodedata** | Unicode normalization (NFC) for consistent text |
| Custom noise filters | Header/footer removal, OCR artifact correction |

### Data Export & Analysis
| Library | Purpose |
|---------|---------|
| **HuggingFace `datasets`** | Export to standard NLP dataset format with train/val/test splits |
| **pandas** | Data manipulation and statistics |
| **matplotlib** | Visualization of length distributions and quality metrics |

### Utilities
| Library | Purpose |
|---------|---------|
| **tqdm** | Progress bars for long-running operations |
| **python-dotenv** | Environment variable management |

---

## 📁 Project Structure

```
pipeline_v1/
├── pipeline.py               # Master CLI orchestrator
├── config.py                 # Central configuration & thresholds
├── validate.py               # Post-pipeline quality checks
│
├── scraper/
│   ├── catalog_scraper.py    # Playwright-based web scraper
│   └── pdf_downloader.py     # Concurrent PDF downloader
│
├── extractor/
│   ├── pdf_extractor.py      # Coordinate-aware PDF → text
│   ├── language_detector.py  # Multi-strategy language detection
│   └── layout_analyzer.py    # Page classification & text separation
│
├── aligner/
│   ├── structural_aligner.py # Article-number matching (primary)
│   └── statistical_aligner.py# Gale-Church alignment (fallback)
│
├── cleaner/
│   └── text_cleaner.py       # Normalization, filtering, dedup
│
├── exporter/
│   └── dataset_exporter.py   # TSV / JSONL / HuggingFace export
│
└── data/                     # Runtime data (PDFs, JSON, exports)
    ├── catalog.json
    ├── pdfs/       (2001–2025, organized by year/month)
    ├── extracted/  (per-PDF JSON with MK + SQ text)
    ├── aligned/    (aligned sentence pairs JSON)
    └── export/     (final TSV, JSONL, HuggingFace dataset)
```

---

## 🔬 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **pdfplumber** over plain text extraction | Word bounding boxes enable precise two-column detection |
| **Midpoint-based column splitting** | Post-2019 gazette uses MK (left) / SQ (right) columns |
| **Sequential block detection** via script transitions | Pre-2019 issues interleave MK/SQ blocks without columns |
| **Article-number alignment** as primary strategy | Legal articles are identically numbered (`Член N` = `Neni N`) |
| **Gale-Church** as statistical fallback | Handles preambles, annexes, and non-article content |
| **MinHash deduplication** | Near-duplicate removal at Jaccard threshold 0.95 |
| **lingua** over langdetect for primary detection | Much higher accuracy on short Macedonian/Albanian fragments |
| **Playwright** for scraping | The gazette site uses AngularJS postbacks (not crawlable with HTTP alone) |

---

## 📊 Output Formats

### TSV
```
mk_sentence    sq_sentence    source_issue    article_id    confidence    alignment_method
Законот стапува...    Ligji hyn në fuqi...    slvesnik_abc123    42    0.95    structural_1:1
```

### JSON Lines
```json
{"mk": "...", "sq": "...", "meta": {"source": "...", "article": "42", "confidence": 0.95, "method": "structural_1:1"}}
```

### HuggingFace Dataset
```python
from datasets import load_from_disk
ds = load_from_disk("data/export/huggingface")
print(ds["train"][0])  # {'mk': '...', 'sq': '...', 'confidence': 0.95, ...}
```

---

## ✅ Quality Assurance

The pipeline includes a dedicated **validation suite** (`validate.py`) with 5 automated checks:

1. **Spot-check** — random sample of pairs for human review
2. **Script check** — verifies MK text is Cyrillic, SQ text is Latin
3. **Length histogram** — word-count distributions (saved as PNG)
4. **Duplicate rate** — exact-duplicate percentage measurement
5. **Coverage** — % of downloaded PDFs that produced ≥ 1 valid pair

### Filtering Criteria
| Filter | Threshold |
|--------|-----------|
| Minimum sentence length | 5 words |
| Maximum sentence length | 200 words |
| Length ratio (MK/SQ) | 0.4 – 2.5 |
| Maximum number fraction | 30% |
| Near-duplicate Jaccard threshold | 0.95 |

---

## 🌍 Domain Context

- **Albanian content** in the gazette begins from **2001** (Ohrid Framework Agreement)
- **Systematic bilingual publication** became standard from **2019** (Law on Use of Languages)
- Best data yield comes from **2019–present** (consistent two-column layout)
- Pre-2019 issues may contain scanned/OCR'd pages with lower extraction quality
- The corpus covers **legal/legislative** domain text — ideal for domain-specific MT

---

## 🚀 Usage

```bash
# Full pipeline (2019–2025)
python pipeline.py --phase all --year-start 2019 --year-end 2025

# Individual phases
python pipeline.py --phase scrape --year-start 2023 --year-end 2025
python pipeline.py --phase download --limit 50
python pipeline.py --phase extract
python pipeline.py --phase align
python pipeline.py --phase export --format tsv,jsonl,huggingface

# Quick test on a single PDF
python pipeline.py --phase test --pdf data/pdfs/2023/someissue.pdf

# Validate output quality
python validate.py --sample 50
```

---

## 📈 Summary

| Metric | Value |
|--------|-------|
| **Language pair** | Macedonian (mk) ↔ Albanian (sq) |
| **Data source** | Služben Vesnik (Official Gazette, 2001–2025) |
| **Pipeline phases** | 5 (scrape → download → extract → align → export) |
| **Alignment methods** | 2 (structural + Gale-Church fallback) |
| **Export formats** | 3 (TSV, JSONL, HuggingFace) |
| **Python libraries** | 17 |
| **Modules** | 10 source files across 5 packages |
| **Domain** | Legal / legislative |
| **License** | Academic research (public government text) |

---

*Built with ❤️ for low-resource NLP research.*
