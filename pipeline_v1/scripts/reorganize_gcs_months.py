#!/usr/bin/env python3
"""
Reorganize GCS PDFs into correct month folders by extracting dates from PDF content.

This script:
1. Lists all PDFs in GCS bucket
2. Downloads first page of each PDF
3. Extracts the publication date from the content
4. Moves the PDF to the correct month folder
"""

import os
import re
import io
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google.cloud import storage
from google.oauth2 import service_account
import fitz  # PyMuPDF
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# GCS Configuration
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
GCS_BUCKET_NAME = "vezilka-pdfs-2026"
GCS_CREDENTIALS_PATH = Path(
    os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS",
        PIPELINE_ROOT / "vezilka-13d3f9c31528.json",
    )
)

# Macedonian month names (Cyrillic)
MK_MONTHS = {
    'јануари': 1, 'јануар': 1,
    'февруари': 2, 'фебруар': 2,
    'март': 3,
    'април': 4,
    'мај': 5,
    'јуни': 6, 'јун': 6,
    'јули': 7, 'јул': 7,
    'август': 8,
    'септември': 9, 'септембар': 9,
    'октомври': 10, 'октобар': 10,
    'ноември': 11, 'новембар': 11,
    'декември': 12, 'децембар': 12,
}

# Serbian/Croatian month names (Latin)
SR_MONTHS = {
    'januar': 1, 'januara': 1,
    'februar': 2, 'februara': 2,
    'mart': 3, 'marta': 3,
    'april': 4, 'aprila': 4,
    'maj': 5, 'maja': 5,
    'jun': 6, 'juna': 6, 'juni': 6,
    'jul': 7, 'jula': 7, 'juli': 7,
    'avgust': 8, 'avgusta': 8,
    'septembar': 9, 'septembra': 9,
    'oktobar': 10, 'oktobra': 10,
    'novembar': 11, 'novembra': 11,
    'decembar': 12, 'decembra': 12,
}

# Roman numerals
ROMAN_MONTHS = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6,
    'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10, 'XI': 11, 'XII': 12,
}


def extract_month_from_text(text: str, expected_year: int) -> int | None:
    """Extract month from PDF text content."""
    text_lower = text.lower()
    
    # Pattern 1: Day + Macedonian month name + year (e.g., "15 јануари 1947")
    for month_name, month_num in MK_MONTHS.items():
        pattern = rf'\b(\d{{1,2}})\s*\.?\s*{month_name}\s*\.?\s*(\d{{4}})'
        match = re.search(pattern, text_lower)
        if match and int(match.group(2)) == expected_year:
            return month_num
    
    # Pattern 2: Serbian/Croatian month names
    for month_name, month_num in SR_MONTHS.items():
        pattern = rf'\b(\d{{1,2}})\s*\.?\s*{month_name}\s*\.?\s*(\d{{4}})'
        match = re.search(pattern, text_lower)
        if match and int(match.group(2)) == expected_year:
            return month_num
    
    # Pattern 3: Roman numerals (e.g., "15. III 1947" or "15 III 1947")
    for roman, month_num in sorted(ROMAN_MONTHS.items(), key=lambda x: -len(x[0])):
        pattern = rf'\b(\d{{1,2}})\s*\.?\s*{roman}\s*\.?\s*(\d{{4}})'
        match = re.search(pattern, text)  # Case sensitive for roman
        if match and int(match.group(2)) == expected_year:
            return month_num
    
    # Pattern 4: Numeric date format (DD.MM.YYYY or DD/MM/YYYY)
    pattern = rf'\b(\d{{1,2}})[./](\d{{1,2}})[./]({expected_year})\b'
    match = re.search(pattern, text)
    if match:
        month = int(match.group(2))
        if 1 <= month <= 12:
            return month
    
    # Pattern 5: Look for standalone month + year
    for month_name, month_num in {**MK_MONTHS, **SR_MONTHS}.items():
        pattern = rf'\b{month_name}\s*\.?\s*(\d{{4}})'
        match = re.search(pattern, text_lower)
        if match and int(match.group(1)) == expected_year:
            return month_num
    
    return None


def extract_month_from_pdf(pdf_bytes: bytes, expected_year: int) -> int | None:
    """Extract month from PDF first page."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(doc) == 0:
            return None
        
        # Get text from first page (and second if needed)
        text = ""
        for page_num in range(min(2, len(doc))):
            text += doc[page_num].get_text() + "\n"
        doc.close()
        
        return extract_month_from_text(text, expected_year)
    except Exception as e:
        logger.debug(f"PDF parse error: {e}")
        return None


class GCSReorganizer:
    """Reorganize PDFs in GCS by actual publication month."""
    
    def __init__(self, bucket_name: str = GCS_BUCKET_NAME, credentials_path: Path = GCS_CREDENTIALS_PATH, workers: int = 20):
        self.workers = workers
        
        credentials = service_account.Credentials.from_service_account_file(str(credentials_path))
        self.client = storage.Client(credentials=credentials, project=credentials.project_id)
        self.bucket = self.client.bucket(bucket_name)
        
        self.moved_count = 0
        self.failed_count = 0
        self.unchanged_count = 0
    
    def reorganize_year(self, year: int, dry_run: bool = False) -> dict:
        """Reorganize all PDFs for a given year."""
        prefix = f"pdfs/{year}/"
        blobs = list(self.bucket.list_blobs(prefix=prefix))
        
        logger.info(f"Found {len(blobs)} PDFs for year {year}")
        
        results = {"moved": 0, "unchanged": 0, "failed": 0, "details": []}
        
        pbar = tqdm(total=len(blobs), desc=f"Reorganizing {year}", unit="pdf")
        
        def process_blob(blob):
            try:
                # Parse current path
                parts = blob.name.split("/")
                if len(parts) < 4:
                    return ("failed", blob.name, None, "invalid path")
                
                current_month = int(parts[2])
                filename = parts[3]
                
                # Download PDF
                pdf_bytes = blob.download_as_bytes()
                
                # Extract actual month
                actual_month = extract_month_from_pdf(pdf_bytes, year)
                
                if actual_month is None:
                    return ("failed", blob.name, current_month, "could not extract month")
                
                if actual_month == current_month:
                    return ("unchanged", blob.name, current_month, None)
                
                # Need to move
                new_path = f"pdfs/{year}/{actual_month:02d}/{filename}"
                
                if not dry_run:
                    # Copy to new location
                    new_blob = self.bucket.blob(new_path)
                    new_blob.upload_from_string(pdf_bytes, content_type="application/pdf")
                    # Delete old
                    blob.delete()
                
                return ("moved", blob.name, actual_month, new_path)
                
            except Exception as e:
                return ("failed", blob.name, None, str(e)[:50])
        
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(process_blob, blob): blob for blob in blobs}
            
            for fut in as_completed(futures):
                status, name, month, info = fut.result()
                
                if status == "moved":
                    results["moved"] += 1
                    results["details"].append({"old": name, "new": info, "month": month})
                elif status == "unchanged":
                    results["unchanged"] += 1
                else:
                    results["failed"] += 1
                    logger.debug(f"Failed {name}: {info}")
                
                pbar.update(1)
        
        pbar.close()
        return results
    
    def reorganize_all(self, year_start: int, year_end: int, dry_run: bool = False):
        """Reorganize all years."""
        total_moved = 0
        total_unchanged = 0
        total_failed = 0
        
        for year in range(year_start, year_end + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing year {year}")
            logger.info(f"{'='*50}")
            
            results = self.reorganize_year(year, dry_run=dry_run)
            
            logger.info(f"Year {year}: Moved {results['moved']}, Unchanged {results['unchanged']}, Failed {results['failed']}")
            
            total_moved += results["moved"]
            total_unchanged += results["unchanged"]
            total_failed += results["failed"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPLETE: Moved {total_moved}, Unchanged {total_unchanged}, Failed {total_failed}")
        logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Reorganize GCS PDFs by publication month")
    parser.add_argument("--year-start", type=int, default=1945)
    parser.add_argument("--year-end", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="Don't actually move files")
    parser.add_argument("--year", type=int, help="Process single year only")
    args = parser.parse_args()
    
    reorganizer = GCSReorganizer(workers=args.workers)
    
    if args.year:
        results = reorganizer.reorganize_year(args.year, dry_run=args.dry_run)
        logger.info(f"Results: Moved {results['moved']}, Unchanged {results['unchanged']}, Failed {results['failed']}")
    else:
        reorganizer.reorganize_all(args.year_start, args.year_end, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
