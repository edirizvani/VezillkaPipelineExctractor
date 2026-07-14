"""
Vezilka Corpus Builder — Configuration
All configuration constants, paths, and thresholds.
"""

import os
from pathlib import Path

# ──────────────────── BASE PATHS ────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
EXTRACTED_DIR = DATA_DIR / "extracted"
ALIGNED_DIR = DATA_DIR / "aligned"
EXPORT_DIR = DATA_DIR / "export"
CATALOG_PATH = DATA_DIR / "catalog.json"

# Create directories on import
for _d in (DATA_DIR, PDF_DIR, EXTRACTED_DIR, ALIGNED_DIR, EXPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ──────────────────── WEBSITE / SCRAPER ─────────────
BASE_URL = "https://slvesnik.com.mk"
CATALOG_URL = f"{BASE_URL}/besplaten-pristap-do-izdanija.nspx"
PDF_URL_TEMPLATE = f"{BASE_URL}/Issues/{{hash}}.pdf"

USER_AGENT = "VezilkaResearchBot/1.0 (Academic NLP Research, North Macedonia)"
RESPECT_ROBOTS = True

# Scraper rate-limiting
REQUEST_DELAY_SECONDS = 0.1
REQUEST_DELAY_JITTER = 0.05
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2.0

# Concurrent download threads
DOWNLOAD_WORKERS = 20

# Only scrape issues from 2001 onwards (Ohrid Framework Agreement)
MIN_YEAR = 2001

# ──────────────────── PDF EXTRACTION ────────────────
COLUMN_SPLIT_TOLERANCE = 0.08          # 8% of page width
MIN_WORDS_PER_COLUMN = 20
THREE_COLUMN_SKIP = True

# ──────────────────── LANGUAGE DETECTION ────────────
CYRILLIC_THRESHOLD = 0.60
LATIN_THRESHOLD = 0.60
SHORT_TEXT_CHAR_LIMIT = 20

# Languages to consider (lingua)
LINGUA_LANGUAGES = ["MACEDONIAN", "ALBANIAN", "SERBIAN"]

# ──────────────────── ALIGNMENT ─────────────────────
MIN_ARTICLE_TEXT_LENGTH = 10
MIN_SENTENCE_WORDS = 5
MAX_SENTENCE_WORDS = 200

# Length ratio filtering
MIN_LENGTH_RATIO = 0.4
MAX_LENGTH_RATIO = 2.5

# Gale-Church parameters (tuned for MK-SQ)
GALE_CHURCH_MEAN_RATIO = 1.1
GALE_CHURCH_VARIANCE = 6.8

# ──────────────────── CLEANING ──────────────────────
MAX_NUMBER_FRACTION = 0.30
MAX_CONSECUTIVE_UPPER = 5
NEAR_DUPLICATE_THRESHOLD = 0.95

# ──────────────────── EXPORT ────────────────────────
TRAIN_SPLIT = 0.80
VALIDATION_SPLIT = 0.10
TEST_SPLIT = 0.10

# ──────────────────── LOGGING ───────────────────────
LOG_LEVEL = os.getenv("VEZILKA_LOG_LEVEL", "INFO")
