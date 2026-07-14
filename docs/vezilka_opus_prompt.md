# Prompt for Claude Opus — Vezilka v2: Redesigned Pipeline Architecture

> Paste this entire prompt into a new Claude Opus conversation. It is self-contained and includes all context needed.

---

## Project: Vezilka v2 — Macedonian ↔ Albanian Parallel Corpus Builder (Redesigned)

You are building **Vezilka v2**, a complete rewrite of an NLP pipeline that creates a sentence-level parallel corpus for the **Macedonian (mk) ↔ Albanian (sq)** language pair from bilingual PDF documents published by the **Službен Весник** (Official Gazette of North Macedonia) at slvesnik.com.mk.

Create everything inside a new folder called **`vezilka_v2/`** with a clean, modular architecture. Each phase must be its own Python module. Write production-quality code with full error handling, logging, and docstrings throughout.

---

## Critical Context: What the PDFs Actually Look Like

The PDFs come in **three distinct layout types** that require different extraction strategies:

### Layout Type A — Two-Column Bilingual (post-2019, most common)
- MK text in the LEFT column, SQ text in the RIGHT column
- Both languages on the same page simultaneously
- Columns split at approximately the horizontal midpoint of the page

### Layout Type B — Sequential Block (also post-2019, very common)
- The FULL Macedonian version of a law appears first (pages 1–N)
- Then the FULL Albanian version follows on pages N+1 to end
- A separator line `__________` or the Albanian title header marks the boundary
- **Example:** The Census Law 2021 (Službен Весник №19, 25 January 2021) is entirely this layout
- The Albanian section always starts with a header matching `^L\s*I\s*G\s*J\b` or `^LIGJ\b`

### Layout Type C — Pre-2019 Mixed / Scanned
- Inconsistent interleaving of MK and SQ text blocks
- Some issues are scanned images (requires OCR)
- Lower quality, handle separately

The pipeline **must detect the layout type first** before attempting extraction. This is the single most important architectural improvement over v1.

---

## Folder Structure to Create

```
vezilka_v2/
├── README.md
├── requirements.txt
├── config.py                    # All tuneable parameters in one place
├── pipeline.py                  # Master orchestrator — runs all phases
│
├── phase1_scrape/
│   ├── __init__.py
│   └── scraper.py               # Playwright-based scraper (resumable JSON catalog)
│
├── phase2_download/
│   ├── __init__.py
│   └── downloader.py            # Concurrent PDF downloader with retry logic
│
├── phase3_extract/
│   ├── __init__.py
│   ├── layout_classifier.py     # *** NEW *** Detects layout type A/B/C per PDF
│   ├── extractor_two_column.py  # Handles Layout Type A (column split)
│   ├── extractor_sequential.py  # *** NEW *** Handles Layout Type B (sequential blocks)
│   ├── extractor_ocr.py         # Handles Layout Type C (OCR fallback via Surya/Tesseract)
│   └── language_detector.py    # 3-layer language detection (Unicode + lingua + langdetect)
│
├── phase4_align/
│   ├── __init__.py
│   ├── structural_aligner.py    # Article-number matching (Член N = Neni N)
│   ├── gale_church.py           # Gale-Church statistical alignment (fallback)
│   ├── translation_scorer.py    # *** NEW *** MT-based alignment scoring (MarianMT/NLLB)
│   └── aligner_orchestrator.py  # Chooses alignment strategy, blends scores
│
├── phase5_clean/
│   ├── __init__.py
│   ├── noise_filter.py          # Regex-based noise removal
│   ├── pair_filter.py           # Length ratio, digit ratio, script validation
│   ├── semantic_filter.py       # LaBSE cosine similarity scoring
│   ├── deduplicator.py          # MinHash LSH deduplication
│   └── exporter.py              # TSV / JSONL / HuggingFace export
│
└── utils/
    ├── __init__.py
    ├── logging_config.py
    └── text_utils.py            # Shared text helpers
```

---

## Phase 3: Layout Classifier (Most Critical Module)

### `phase3_extract/layout_classifier.py`

This module must classify each PDF **before** any text extraction occurs. Implement the following logic:

```python
class LayoutType(Enum):
    TWO_COLUMN = "two_column"      # Type A: MK left | SQ right on same pages
    SEQUENTIAL = "sequential"      # Type B: Full MK block, then full SQ block
    MIXED_PRE2019 = "mixed"        # Type C: Interleaved, inconsistent
    SINGLE_LANGUAGE = "single"     # No Albanian content found
    UNKNOWN = "unknown"

class LayoutClassifier:
    def classify(self, pdf_path: Path) -> LayoutClassificationResult:
        """
        Returns layout type + metadata:
        - layout_type: LayoutType enum value
        - boundary_page: int (for SEQUENTIAL type — page where Albanian starts)
        - boundary_block: int (for SEQUENTIAL type — block index on boundary page)
        - confidence: float (0-1)
        - has_albanian: bool
        """
```

**Detection algorithm for each type:**

**Type A (Two-Column):**
- Open PDF with pdfplumber
- On each page, check if bounding boxes of text blocks fall in BOTH left half (x < midpoint) AND right half (x > midpoint)
- Check that left-half blocks are predominantly Cyrillic AND right-half blocks are predominantly Latin+Albanian chars
- If >50% of pages satisfy this → TWO_COLUMN

**Type B (Sequential):**
- Scan ALL pages in order tracking a rolling Cyrillic ratio
- Look for a **strong language transition**: a page or block where the text shifts from >70% Cyrillic to >70% Latin
- Also search for these Albanian boundary markers (in order of reliability):
  1. Regex `^L\s*I\s*G\s*J\b` (Albanian word for "Law" typeset with spaces)
  2. Regex `^LIGJ\b`
  3. Regex `^Neni\s+1\b` (Albanian "Article 1" — restart of numbering signals new language version)
  4. Horizontal separator line matching `_{5,}` or `—{5,}` followed by Latin text
- Record the exact (page_index, block_index) of the boundary
- If a clear transition is found → SEQUENTIAL

**Type C (Mixed/Pre-2019):**
- No clear two-column structure AND no clean sequential transition
- Language alternates within pages at block level
- → MIXED_PRE2019

**Albanian Character Detection Helper:**
```python
ALBANIAN_SPECIFIC_CHARS = set('ëËçÇ')  # chars common in Albanian but rare in other Latin scripts

def has_albanian_markers(text: str) -> bool:
    """Quick check for Albanian-specific characters."""
    return bool(ALBANIAN_SPECIFIC_CHARS.intersection(text))

def cyrillic_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if '\u0400' <= c <= '\u04FF') / len(alpha)

def latin_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isascii()) / len(alpha)
```

---

## Phase 3: Sequential Block Extractor (New Module)

### `phase3_extract/extractor_sequential.py`

For Layout Type B PDFs (like the Census Law example), extract MK and SQ text blocks separately using the boundary detected by the classifier.

```python
class SequentialExtractor:
    def extract(
        self,
        pdf_path: Path,
        boundary_page: int,
        boundary_block: int
    ) -> tuple[list[str], list[str]]:
        """
        Returns (mk_sentences, sq_sentences) extracted from a sequential-block PDF.
        
        mk_sentences: all text from page 0 to (boundary_page, boundary_block - 1)
        sq_sentences: all text from (boundary_page, boundary_block) to end of PDF
        """
```

**Implementation details:**
- Use pdfplumber for coordinate-aware text extraction
- Preserve reading order using `page.extract_words(use_text_flow=True)`
- Apply the same hyphenation fix: `(\w)\s*-\s*([a-zа-ш])` → `\1\2`
- Apply backtick-to-ë fix for Albanian: `` ` `` → `ë`
- After extraction, run each block through the language detector to validate (MK blocks must be >60% Cyrillic, SQ blocks must be >60% Latin with Albanian markers)
- Log any blocks that fail validation with their page number and text preview

---

## Phase 4: Translation-Based Alignment Scorer (New Module)

### `phase4_align/translation_scorer.py`

This module adds a **machine translation signal** to alignment scoring. The insight is that translating MK→SQ and comparing the MT output to the actual SQ candidate catches false positives that fool length-ratio and LaBSE scoring.

```python
class TranslationAlignmentScorer:
    """
    Uses MarianMT (fast) or NLLB-200 (higher quality) to score MK-SQ sentence pairs
    by translating MK→SQ and computing chrF++ against the candidate SQ sentence.
    
    This catches false positives where two sentences have similar LENGTH and 
    EMBEDDING but are not actually translations of each other (common in 
    legal boilerplate with repeated structures).
    """
    
    def __init__(self, model: str = "helsinki"):
        """
        model options:
        - "helsinki": Helsinki-NLP/opus-mt-mk-sq (fast, ~300MB, good for alignment)
        - "nllb": facebook/nllb-200-distilled-600M (slower, higher quality)
        - "nllb-large": facebook/nllb-200-distilled-1.3B (best quality, needs GPU)
        """
    
    def score_pair(self, mk_sentence: str, sq_sentence: str) -> float:
        """
        Returns chrF++ score (0.0 to 1.0) between MT(mk) and sq_sentence.
        Higher = more likely to be a true translation pair.
        """
    
    def score_batch(
        self, 
        mk_sentences: list[str], 
        sq_sentences: list[str],
        batch_size: int = 32
    ) -> list[float]:
        """Batch scoring for efficiency. Pairs are (mk[i], sq[i])."""
```

**Implementation requirements:**
- Use `sacrebleu.metrics.CHRF` for chrF++ computation (install sacrebleu)
- Implement batched translation using the HuggingFace `pipeline` with `batch_size` parameter
- Cache translations to avoid re-translating the same MK sentence multiple times (use `functools.lru_cache` or a simple dict cache keyed on hash of MK sentence)
- Add a `warmup()` method that loads the model once and keeps it in memory
- Handle CUDA/CPU automatically: use GPU if `torch.cuda.is_available()`, else CPU with a warning
- For the Helsinki model, source lang code is `mk`, target is `sq`
- For NLLB, source lang code is `mkd_Cyrl`, target is `sqi_Latn`

---

## Phase 4: Aligner Orchestrator (Updated)

### `phase4_align/aligner_orchestrator.py`

This orchestrates all alignment signals and produces a **blended confidence score**:

```python
SCORE_WEIGHTS = {
    "structural": 0.0,    # Binary: either structurally matched or not
    "labse": 0.45,        # Semantic similarity (your existing LaBSE code)
    "translation": 0.30,  # NEW: MT-based chrF++ score
    "length_ratio": 0.25, # Original Gale-Church length signal (normalized)
}

def blend_scores(
    labse_score: float,
    translation_score: float,
    length_ratio_score: float,
    is_structural: bool
) -> float:
    """
    Structural matches get a bonus: they bypass the translation scorer
    (too slow for the 82% of pairs that are structurally aligned) and
    receive a fixed structural_confidence boosted by LaBSE.
    
    Non-structural pairs get the full blended score.
    """
    if is_structural:
        # Structural + LaBSE is sufficient; skip expensive MT scoring
        return 0.15 + (0.85 * labse_score)
    else:
        return (
            SCORE_WEIGHTS["labse"] * labse_score +
            SCORE_WEIGHTS["translation"] * translation_score +
            SCORE_WEIGHTS["length_ratio"] * length_ratio_score
        )
```

**Important:** Only run `TranslationAlignmentScorer` on **non-structural pairs** (the ~18% that come from Gale-Church). Structural pairs are already matched by article number, so MT scoring is wasteful there.

---

## config.py — All Parameters in One Place

```python
# vezilka_v2/config.py

from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class VezilkaConfig:
    # Paths
    data_dir: Path = Path("data")
    catalog_path: Path = Path("data/catalog.json")
    pdf_dir: Path = Path("data/pdfs")
    output_dir: Path = Path("data/output")
    
    # Scraping
    base_url: str = "https://www.slvesnik.com.mk"
    scrape_delay_seconds: float = 0.1
    scrape_jitter_seconds: float = 0.05
    
    # Downloading
    download_workers: int = 20
    download_max_retries: int = 3
    
    # Layout classification
    two_column_min_page_fraction: float = 0.5
    cyrillic_threshold: float = 0.60
    latin_threshold: float = 0.60
    column_split_tolerance: float = 0.08  # 8% tolerance on midpoint
    
    # Albanian detection
    albanian_chars: str = "ëËçÇ"
    albanian_boundary_patterns: list = field(default_factory=lambda: [
        r"^L\s*I\s*G\s*J\b",
        r"^LIGJ\b",
        r"^Neni\s+1\b",
        r"_{5,}",
    ])
    
    # Alignment
    structural_article_pattern_mk: str = r"Член\s+(\d+)"
    structural_article_pattern_sq: str = r"Neni\s+(\d+)"
    gc_mean_char_ratio: float = 1.1
    gc_variance: float = 6.8
    
    # Translation scoring
    translation_model: str = "helsinki"  # or "nllb", "nllb-large"
    translation_batch_size: int = 32
    run_translation_scoring: bool = True  # Set False to skip (faster but lower quality)
    
    # LaBSE
    labse_model: str = "sentence-transformers/LaBSE"
    labse_min_similarity: float = 0.60
    
    # Blended score weights (must sum to 1.0 for non-structural pairs)
    weight_labse: float = 0.45
    weight_translation: float = 0.30
    weight_length_ratio: float = 0.25
    
    # Filtering
    min_words: int = 5
    max_words: int = 200
    min_length_ratio: float = 0.4
    max_length_ratio: float = 2.5
    max_digit_fraction: float = 0.30
    mk_min_cyrillic: float = 0.50
    sq_min_latin: float = 0.50
    
    # Deduplication
    minhash_threshold: float = 0.95
    minhash_num_perm: int = 128
    
    # Export
    train_fraction: float = 0.80
    val_fraction: float = 0.10
    test_fraction: float = 0.10
    export_formats: list = field(default_factory=lambda: ["tsv", "jsonl", "huggingface"])
```

---

## requirements.txt

Generate a complete `requirements.txt` that includes:

```
# Web scraping
playwright>=1.40.0
beautifulsoup4>=4.12.0
requests>=2.31.0
lxml>=5.0.0

# PDF extraction
pdfplumber>=0.10.0
pymupdf>=1.23.0

# OCR (optional, for pre-2019 scanned PDFs)
pytesseract>=0.3.10
# surya-ocr>=0.4.0  # Uncomment for best OCR quality (requires GPU recommended)

# Language detection
lingua-language-detector>=2.0.0
langdetect>=1.0.9

# Sentence splitting
sacremoses>=0.1.1
sentence-splitter>=1.4.0

# Alignment
# (Gale-Church: implemented from scratch, no external dep needed)

# Translation scoring
transformers>=4.35.0
torch>=2.1.0
sacrebleu>=2.3.0

# Semantic scoring
sentence-transformers>=2.3.0

# Deduplication
datasketch>=1.6.0

# Data processing
pandas>=2.1.0
datasets>=2.15.0  # HuggingFace datasets

# Utilities
tqdm>=4.66.0
python-dotenv>=1.0.0
matplotlib>=3.8.0
```

---

## README.md — Must Include

Write a comprehensive README that covers:

1. **Project overview** — what Vezilka v2 does and why MK↔SQ is important
2. **Architecture diagram** (ASCII art) showing all 5 phases and data flow
3. **Layout type explanations** with examples of what each looks like
4. **Quick start** — how to run the full pipeline in one command
5. **Phase-by-phase usage** — how to run individual phases
6. **Configuration** — how to tune `config.py`
7. **Score interpretation** — what the blended confidence score means
8. **Known limitations** — pre-2019 quality, translation model speed

---

## Additional Implementation Requirements

### Error Handling
- Every module must use Python `logging` (not `print`)
- All file I/O must be wrapped in try/except with informative error messages
- Failed PDFs must be logged to `data/failed_pdfs.jsonl` with reason and stack trace
- The pipeline must be **resumable** — if interrupted, it picks up where it left off using checkpointing

### Checkpointing
- Phase 3 (extract): Save extracted text blocks to `data/extracted/{pdf_id}.json` — skip if file exists
- Phase 4 (align): Save aligned pairs to `data/aligned/{pdf_id}.jsonl` — skip if file exists
- Phase 5 (clean): Final cleaned pairs written to `data/output/corpus.tsv`

### Logging Schema
Each aligned pair in the output must include these fields:
```json
{
  "mk": "Македонска реченица...",
  "sq": "Fjali shqipe...",
  "pdf_id": "2021_019",
  "article_number": 5,
  "alignment_type": "structural",
  "layout_type": "sequential",
  "labse_score": 0.91,
  "translation_score": 0.73,
  "length_ratio_score": 0.95,
  "blended_confidence": 0.87,
  "mk_word_count": 23,
  "sq_word_count": 25
}
```

### Testing
Create `tests/` folder with:
- `test_layout_classifier.py` — unit tests for each layout type detection
- `test_translation_scorer.py` — tests with known MK/SQ sentence pairs
- `test_sequential_extractor.py` — test against the Census Law 2021 PDF structure
- `test_noise_filter.py` — test all regex patterns
- Use `pytest` as the test runner

---

## Specific Implementation Notes

### Albanian `ë` Fix (must be in `utils/text_utils.py`)
```python
def fix_albanian_encoding(text: str) -> str:
    """Fix common font encoding issues in Albanian text from PDF extraction."""
    # Backtick used instead of ë in older PDFs
    text = text.replace('`', 'ë')
    # Also handle hex variant
    text = text.replace('\x60', 'ë')
    return text
```

### Hyphenation Fix
```python
import re

HYPHEN_PATTERN = re.compile(r'(\w)\s*[-–]\s*([a-zа-ш])')

def fix_hyphenation(text: str) -> str:
    """Rejoin words broken across PDF lines by hyphenation."""
    return HYPHEN_PATTERN.sub(r'\1\2', text)
```

### Hex Hash Removal
```python
HEX_HASH_PATTERN = re.compile(r'\b[0-9a-f]{32,}\b', re.IGNORECASE)

def remove_hex_hashes(text: str) -> str:
    return HEX_HASH_PATTERN.sub('', text)
```

---

---

## Phase 4: Deep Semantic Validation (New Module — Critical)

### `phase4_align/semantic_validator.py`

This is the most important new module beyond the layout classifier. The fundamental problem with v1 is that **every signal used (length ratio, LaBSE, Gale-Church) only asks "are these sentences similar?" — none of them ask "is this sentence actually a translation of the other?"**

These are different questions. Two legal sentences can be semantically similar (both talk about penalties, both mention 200-300 euros, both reference the same article number) without being mutual translations. LaBSE will score them highly. Length ratio will pass them. They will end up in your corpus as false pairs.

The modules below fix this by using models that are explicitly trained on **cross-lingual translation equivalence**, not just semantic similarity.

---

### Validator 1: COMET-QE (Reference-Free Quality Estimation)

COMET-QE (Rei et al., 2021 — `Unbabel/wmt22-cometkiwi-da`) is a neural model trained specifically to judge whether a translation is correct — **without needing a reference translation**. It takes a source sentence and a hypothesis translation and outputs a quality score. This is exactly what you need.

```python
from comet import download_model, load_from_checkpoint

class COMETQEValidator:
    """
    Uses COMET-QE (wmt22-cometkiwi-da) to score whether sq_sentence
    is a valid translation of mk_sentence.
    
    This model was trained on human translation quality judgments from
    WMT shared tasks. It directly models translation adequacy — 
    meaning it penalizes pairs where the content doesn't match even
    if the sentences are topically similar.
    
    Score range: approximately 0.0 to 1.0
    Threshold: reject pairs below 0.75 (tune on your data)
    """
    
    MODEL_NAME = "Unbabel/wmt22-cometkiwi-da"
    
    def __init__(self, device: str = "auto"):
        model_path = download_model(self.MODEL_NAME)
        self.model = load_from_checkpoint(model_path)
        self.device = device
    
    def score_pairs(
        self, 
        mk_sentences: list[str], 
        sq_sentences: list[str],
        batch_size: int = 16
    ) -> list[float]:
        """
        Score a list of (mk, sq) pairs for translation quality.
        Returns list of floats, one per pair.
        
        The model is called with:
          - src: Macedonian sentence
          - mt: Albanian sentence (treated as the MT hypothesis)
        
        Note: COMET-QE does not require a reference — it scores
        the (source, hypothesis) pair directly.
        """
        data = [{"src": mk, "mt": sq} for mk, sq in zip(mk_sentences, sq_sentences)]
        scores = self.model.predict(data, batch_size=batch_size, gpus=1 if self.device == "cuda" else 0)
        return scores.scores
    
    def filter_pairs(
        self,
        pairs: list[dict],
        threshold: float = 0.75
    ) -> tuple[list[dict], list[dict]]:
        """
        Split pairs into (kept, rejected) based on COMET-QE threshold.
        Rejected pairs are logged for inspection, not silently discarded.
        """
```

**Why COMET-QE specifically:**
- Trained on human adequacy judgments from professional translators
- Explicitly models whether the *meaning* transferred correctly
- Unlike LaBSE (which measures embedding proximity), COMET-QE was built to catch subtle translation errors: omissions, additions, wrong numbers, wrong names
- It will catch pairs like: `Член 48` (Article 48 penalty clause in MK) wrongly aligned with `Neni 51` (Article 51 in SQ) even though both discuss fines of 200-300 euros and have similar length and LaBSE score

**Install:** `pip install unbabel-comet`

---

### Validator 2: LASER3 Bidirectional Similarity

LASER3 (Heffernan et al., 2022 — Meta AI) produces language-agnostic sentence embeddings trained on 200+ languages specifically for **parallel corpus mining**. Unlike LaBSE which was trained for retrieval, LASER3 was trained to be discriminative — it maximizes similarity for true translation pairs and pushes apart non-translation pairs from the same domain.

```python
from laser_encoders import LaserEncoderPipeline

class LASER3Validator:
    """
    Uses LASER3 embeddings to compute bidirectional translation similarity.
    
    Key difference from LaBSE:
    - LaBSE: trained for multilingual retrieval, measures semantic similarity
    - LASER3: trained for parallel corpus mining, measures translation equivalence
    
    We use BIDIRECTIONAL scoring:
    - Forward:  embed MK as source, embed SQ as target, compute cosine sim
    - Backward: embed SQ as source, embed MK as target, compute cosine sim
    - Final score: min(forward, backward)
    
    Using the MINIMUM of both directions catches asymmetric false pairs —
    cases where the MK sentence entails the SQ content but not vice versa
    (or where one sentence is much more general than the other).
    
    Threshold: reject pairs below 0.65 (LASER3 scores are better calibrated
    than LaBSE for translation equivalence at this threshold)
    """
    
    def __init__(self):
        # LASER3 has language-specific encoders
        self.mk_encoder = LaserEncoderPipeline(lang="mkd")  # Macedonian
        self.sq_encoder = LaserEncoderPipeline(lang="sqi")  # Albanian
    
    def score_bidirectional(
        self,
        mk_sentences: list[str],
        sq_sentences: list[str]
    ) -> list[float]:
        """
        Returns min(forward_sim, backward_sim) for each pair.
        This is more conservative than unidirectional LaBSE scoring.
        """
        mk_embeddings = self.mk_encoder.encode_sentences(mk_sentences)
        sq_embeddings = self.sq_encoder.encode_sentences(sq_sentences)
        
        # Forward: cosine similarity using MK→SQ direction
        forward_scores = cosine_similarity(mk_embeddings, sq_embeddings)
        
        # Backward: cosine similarity using SQ→MK direction  
        # (swap the encoder roles to get the other directional embedding)
        sq_as_src = self.sq_encoder.encode_sentences(sq_sentences)
        mk_as_tgt = self.mk_encoder.encode_sentences(mk_sentences)
        backward_scores = cosine_similarity(sq_as_src, mk_as_tgt)
        
        # Conservative: take the minimum
        return [min(f, b) for f, b in zip(forward_scores, backward_scores)]
```

**Install:** `pip install laser-encoders`

---

### Validator 3: Back-Translation Consistency Check

This is the most direct test of whether two sentences are mutual translations. The logic:

1. Translate MK → SQ using MarianMT/NLLB (forward pass)
2. Translate that output SQ → MK (back-translation)
3. Compute chrF++ between the original MK and the back-translated MK
4. If the round-trip preserves meaning, the alignment is valid

```python
class BackTranslationValidator:
    """
    Tests translation pair validity via round-trip consistency.
    
    Algorithm:
    1. mt_sq = translate(mk)          # MK → SQ
    2. bt_mk = translate(mt_sq)       # SQ → MK (back-translation)  
    3. chrf_forward = chrF++(mt_sq, sq_candidate)   # Does MT match actual SQ?
    4. chrf_backward = chrF++(bt_mk, mk_original)   # Does back-translation match original MK?
    5. consistency_score = harmonic_mean(chrf_forward, chrf_backward)
    
    A high consistency score means:
    - The actual SQ sentence looks like what a translator would produce from the MK
    - Round-tripping back to MK recovers the original meaning
    - The pair is a genuine translation, not just topically related
    
    This catches a specific failure mode: two sentences about the same legal topic
    (e.g., both mention "State Statistical Office" and "census") that are NOT
    the same article. They pass LaBSE and length ratio but fail back-translation.
    """
    
    def __init__(self, mk_to_sq_model: str = "Helsinki-NLP/opus-mt-mk-sq",
                       sq_to_mk_model: str = "Helsinki-NLP/opus-mt-sq-mk"):
        self.forward_translator = pipeline("translation", model=mk_to_sq_model)
        self.backward_translator = pipeline("translation", model=sq_to_mk_model)
        self.chrf = CHRF(word_order=2)  # chrF++ (word_order=2)
    
    def score_pairs(
        self,
        mk_sentences: list[str],
        sq_sentences: list[str],
        batch_size: int = 16
    ) -> list[float]:
        """Returns harmonic mean of forward and backward chrF++ scores."""
        
        # Forward: MK → SQ
        mt_sq_outputs = self.forward_translator(
            mk_sentences, batch_size=batch_size, max_length=512
        )
        mt_sq = [o['translation_text'] for o in mt_sq_outputs]
        
        # Backward: MT(SQ) → MK
        bt_mk_outputs = self.backward_translator(
            mt_sq, batch_size=batch_size, max_length=512
        )
        bt_mk = [o['translation_text'] for o in bt_mk_outputs]
        
        scores = []
        for mk_orig, sq_cand, mt_sq_sent, bt_mk_sent in zip(
            mk_sentences, sq_sentences, mt_sq, bt_mk
        ):
            # Forward: does MT output match actual SQ?
            chrf_fwd = self.chrf.sentence_score(mt_sq_sent, [sq_cand]).score / 100.0
            
            # Backward: does back-translation match original MK?
            chrf_bwd = self.chrf.sentence_score(bt_mk_sent, [mk_orig]).score / 100.0
            
            # Harmonic mean (punishes if either direction fails)
            if chrf_fwd + chrf_bwd > 0:
                harmonic = 2 * chrf_fwd * chrf_bwd / (chrf_fwd + chrf_bwd)
            else:
                harmonic = 0.0
            scores.append(harmonic)
        
        return scores
```

---

### Updated Blended Scoring — Full Signal Stack

Update `phase4_align/aligner_orchestrator.py` to incorporate all signals:

```python
SCORE_WEIGHTS_STRUCTURAL = {
    # For article-number matched pairs (fast path — skip expensive MT validators)
    "labse": 0.50,
    "laser3": 0.30,
    "length_ratio": 0.20,
}

SCORE_WEIGHTS_NON_STRUCTURAL = {
    # For Gale-Church pairs (slow path — run all validators)
    "labse": 0.25,
    "laser3": 0.20,
    "comet_qe": 0.25,        # Translation adequacy judge
    "back_translation": 0.20, # Round-trip consistency
    "length_ratio": 0.10,
}

# Hard rejection thresholds — pair is REJECTED if ANY of these fail
# regardless of blended score
HARD_REJECTION_THRESHOLDS = {
    "labse": 0.55,           # Absolute floor on semantic similarity
    "comet_qe": 0.65,        # Absolute floor on translation quality
    "laser3": 0.55,          # Absolute floor on translation equivalence
    "length_ratio": 0.35,    # Absolute floor on length compatibility
}

def validate_pair(signals: dict) -> tuple[float, bool, str]:
    """
    Returns (blended_score, is_accepted, rejection_reason).
    
    A pair is rejected if:
    1. Any signal falls below its hard rejection threshold, OR
    2. The blended score is below the overall threshold (0.70)
    
    rejection_reason is a string describing which signal failed,
    useful for debugging and dataset analysis.
    """
    # Check hard thresholds first
    for signal, threshold in HARD_REJECTION_THRESHOLDS.items():
        if signal in signals and signals[signal] < threshold:
            return 0.0, False, f"hard_reject_{signal}_{signals[signal]:.3f}"
    
    # Compute blended score
    weights = (SCORE_WEIGHTS_STRUCTURAL 
               if signals.get("is_structural") 
               else SCORE_WEIGHTS_NON_STRUCTURAL)
    
    blended = sum(
        weights.get(k, 0) * v 
        for k, v in signals.items() 
        if k in weights
    )
    
    accepted = blended >= 0.70
    reason = "accepted" if accepted else f"low_blended_{blended:.3f}"
    return blended, accepted, reason
```

---

### Processing Strategy: Tiered Validation

Running COMET-QE and back-translation on all 100K+ pairs is expensive. Use a **tiered approach**:

```python
def tiered_validation_strategy(pairs: list[dict]) -> list[dict]:
    """
    Tier 1 (all pairs): LaBSE + LASER3 + length ratio
        → Fast, catches obvious misalignments
        → Reject if blended score < 0.60
    
    Tier 2 (surviving pairs): + COMET-QE
        → Medium cost (~2 hours on GPU for 100K pairs)
        → Reject if COMET-QE < 0.65
    
    Tier 3 (high-value ambiguous pairs only): + Back-Translation
        → Expensive — only run on pairs where Tier 1 and Tier 2 disagree
        → i.e., pairs where LaBSE > 0.75 but COMET-QE < 0.75
        → These are the exactly the "looks similar but wrong" false positives
    
    This means:
    - ~100% of pairs go through Tier 1 (fast)
    - ~70% of pairs go through Tier 2 (medium)  
    - ~10-15% of pairs go through Tier 3 (expensive)
    - Total compute: manageable on a single GPU in one day
    """
```

---

### Why Each Model Catches Different Errors

| Failure Mode | LaBSE | LASER3 | COMET-QE | Back-Translation |
|---|---|---|---|---|
| Wrong article (same topic) | ❌ misses | ✅ catches | ✅ catches | ✅ catches |
| Boilerplate near-duplicate | ❌ misses | ❌ misses | ✅ catches | ✅ catches |
| Correct meaning, wrong numbers | ❌ misses | ❌ misses | ✅ catches | ✅ catches |
| Wrong language in column | ✅ catches | ✅ catches | ✅ catches | ✅ catches |
| 1:2 merge needed | ❌ misses | ❌ misses | ✅ catches | ✅ catches |
| OCR garbled text | ✅ catches | ✅ catches | ✅ catches | ✅ catches |

The critical column is **"Correct meaning, wrong numbers"** — this is endemic in legal text where multiple articles discuss the same fine amounts, same procedures, same ministries. LaBSE and LASER3 both fail here because the embeddings are genuinely similar. COMET-QE catches it because it was trained on human judgments of whether *specific content* transferred correctly.

---



Build in this order so the architecture is testable incrementally:

1. `config.py` + `utils/` (foundation)
2. `phase3_extract/layout_classifier.py` (most critical new module)
3. `phase3_extract/extractor_sequential.py` (handles the majority of post-2019 PDFs)
4. `phase3_extract/extractor_two_column.py` (port from v1 with improvements)
5. `phase3_extract/language_detector.py` (port from v1)
6. `phase4_align/structural_aligner.py` (port from v1)
7. `phase4_align/translation_scorer.py` (new quality signal)
8. `phase4_align/aligner_orchestrator.py` (integrates all signals)
9. `phase5_clean/` (port from v1 with updated blended scoring)
10. `phase1_scrape/` + `phase2_download/` (port from v1, minimal changes)
11. `pipeline.py` (master orchestrator)
12. `README.md` + `tests/`

---

## Final Instruction

Write **complete, runnable Python code** for every module listed above. Do not write pseudocode or stubs — implement the full logic. Use type hints throughout. Follow PEP 8. Each file should be independently importable and testable.

Start with `config.py`, then `utils/`, then `phase3_extract/layout_classifier.py` as the first and most important new module.
