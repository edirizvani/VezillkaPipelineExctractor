#!/usr/bin/env python3
"""
Fast PDF Scraper & GCS Uploader for Služben Vesnik (1944-2025)

Downloads PDFs directly to Google Cloud Storage using concurrent workers.
Extracts month from Angular scope data on the website.

Usage:
    python scrape_to_gcs.py --year-start 1944 --year-end 2025
    python scrape_to_gcs.py --fresh  # Clear existing and start fresh
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# Suppress SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Google Cloud Storage
from google.cloud import storage
from google.oauth2 import service_account

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────
GCS_BUCKET_NAME = "vezilka-pdfs-2026"
GCS_CREDENTIALS_PATH = Path(
    os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS",
        PIPELINE_ROOT / "vezilka-13d3f9c31528.json",
    )
)
PDF_HASH_RE = re.compile(r"([a-f0-9]{32})", re.IGNORECASE)

# Month name to number mapping
MONTH_MAP = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12,
}


def make_session(workers: int = 30) -> requests.Session:
    """Build a fast requests session with connection pooling."""
    s = requests.Session()
    s.headers["User-Agent"] = config.USER_AGENT
    s.verify = False
    adapter = HTTPAdapter(
        pool_connections=workers,
        pool_maxsize=workers * 2,
        max_retries=Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        ),
    )
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


class FastGCSScraper:
    """
    Scrapes Služben Vesnik PDFs and uploads directly to GCS.
    Gets month from Angular scope data on the website.
    """

    def __init__(
        self,
        year_start: int = 1944,
        year_end: int = 2000,
        workers: int = 30,
        bucket_name: str = GCS_BUCKET_NAME,
        credentials_path: Path = GCS_CREDENTIALS_PATH,
    ):
        self.year_start = year_start
        self.year_end = year_end
        self.workers = workers
        self.bucket_name = bucket_name
        
        # Initialize GCS client
        credentials = service_account.Credentials.from_service_account_file(
            str(credentials_path)
        )
        self.gcs_client = storage.Client(credentials=credentials, project=credentials.project_id)
        self.bucket = self.gcs_client.bucket(bucket_name)
        
        # Track progress
        self.uploaded_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self._lock = threading.Lock()
        
        # Catalog for tracking
        self.catalog: list[dict] = []
        self.catalog_path = PIPELINE_ROOT / "data" / "catalog_1944_2000.json"

    def clear_gcs_folder(self, prefix: str = "pdfs/") -> int:
        """Delete all blobs with given prefix from GCS bucket."""
        logger.info(f"Clearing gs://{self.bucket_name}/{prefix}...")
        blobs = list(self.bucket.list_blobs(prefix=prefix))
        count = len(blobs)
        
        if count == 0:
            logger.info("No existing files to clear.")
            return 0
        
        logger.info(f"Deleting {count} existing files...")
        deleted = 0
        for blob in tqdm(blobs, desc="Deleting old files", unit="file"):
            try:
                blob.delete()
                deleted += 1
            except Exception as e:
                logger.debug(f"Delete error: {e}")
        
        logger.info(f"Deleted {deleted} files from GCS.")
        return deleted

    def scrape_and_upload(self, fresh: bool = False) -> int:
        """Main entry: scrape all years and upload PDFs to GCS."""
        from playwright.sync_api import sync_playwright

        logger.info(f"Starting scrape for years {self.year_start}–{self.year_end}")
        logger.info(f"Target bucket: gs://{self.bucket_name}/")
        logger.info(f"Workers: {self.workers}")

        # Clear existing if fresh start requested
        if fresh:
            self.clear_gcs_folder(f"pdfs/")
            self.catalog = []
            if self.catalog_path.exists():
                self.catalog_path.unlink()
            logger.info("Fresh start - cleared all existing data.")
        else:
            self._load_existing_catalog()
        
        existing_ids = {e["issue_id"] for e in self.catalog}

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=config.USER_AGENT,
                locale="mk-MK",
            )
            page = context.new_page()

            logger.info(f"Loading {config.CATALOG_URL}...")
            page.goto(config.CATALOG_URL, wait_until="networkidle", timeout=60000)
            time.sleep(2)
            self._dismiss_cookies(page)

            # Collect all issues by year with month from Angular scope
            all_issues = []
            for year in range(self.year_start, self.year_end + 1):
                logger.info(f"── Scanning year {year} ──")
                
                if not self._select_year(page, year):
                    logger.warning(f"Could not select year {year}")
                    continue

                # Extract issues with month data from Angular scope
                issues = self._extract_issues_with_months(page, year)
                new_issues = [i for i in issues if i["issue_id"] not in existing_ids]
                all_issues.extend(new_issues)
                
                # Count by month
                month_counts = {}
                for i in issues:
                    m = i.get("month", 0)
                    month_counts[m] = month_counts.get(m, 0) + 1
                
                logger.info(f"  Year {year}: {len(issues)} issues, {len(new_issues)} new")
                logger.info(f"    Months: {dict(sorted(month_counts.items()))}")
                time.sleep(0.5)

            browser.close()

        logger.info(f"\nTotal issues to download: {len(all_issues)}")
        
        if not all_issues:
            logger.info("No new issues to download.")
            return 0

        # Download and upload concurrently
        self._download_and_upload_batch(all_issues)
        
        # Save final catalog
        self._save_catalog()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPLETE: Uploaded {self.uploaded_count}, Failed {self.failed_count}, Skipped {self.skipped_count}")
        logger.info(f"{'='*60}")
        
        return self.uploaded_count

    def _extract_issues_with_months(self, page, year: int) -> list[dict]:
        """Extract issues with month data from Angular scope."""
        # Get all issues data from Angular scope
        # The scope.issue contains { MonthName: "JANUARY", Issues: [...] }
        scope_data = page.evaluate("""() => {
            const results = [];
            const seen = new Set();
            const links = document.querySelectorAll('a[href*="Issues"]');
            
            for (const link of links) {
                try {
                    const scope = angular.element(link).scope();
                    if (scope && scope.issue && scope.issue.MonthName && scope.issue.Issues) {
                        // scope.issue is actually the month object
                        const monthName = scope.issue.MonthName;
                        const issues = scope.issue.Issues;
                        
                        for (const issue of issues) {
                            const key = issue.FileName;
                            if (key && !seen.has(key)) {
                                seen.add(key);
                                results.push({
                                    fileName: issue.FileName || '',
                                    monthName: monthName,
                                    month: issue.Month || monthName, 
                                    issueNumber: issue.IssueNumber || 0,
                                    title: issue.Title || ''
                                });
                            }
                        }
                    }
                } catch(e) {}
            }
            return results;
        }""")
        
        issues = []
        for item in scope_data:
            file_name = item.get("fileName", "")
            if not file_name or len(file_name) != 32:
                continue
            
            # Convert month name to number (use monthName from the parent object)
            month_name = item.get("monthName", "").lower()
            month_num = MONTH_MAP.get(month_name, 1)
            
            issues.append({
                "issue_id": file_name.upper(),
                "year": year,
                "month": month_num,
                "issue_number": item.get("issueNumber", 0),
                "title": item.get("title", ""),
                "url": f"https://slvesnik.com.mk/Issues/{file_name}.pdf",
            })
        
        return issues

    def _download_and_upload_batch(self, issues: list[dict]) -> None:
        """Download PDFs and upload to GCS with correct month folder."""
        session = make_session(self.workers)
        
        pbar = tqdm(total=len(issues), desc="Downloading & Uploading", unit="pdf")
        
        def job(issue: dict) -> tuple[dict, str]:
            """Download PDF and upload to GCS. Returns (issue, status)."""
            try:
                url = issue["url"]
                year = issue.get("year", 0)
                month = issue.get("month", 1)
                issue_id = issue["issue_id"]
                
                # Check if already in GCS
                blob_path = f"pdfs/{year}/{month:02d}/{issue_id}.pdf"
                blob = self.bucket.blob(blob_path)
                
                if blob.exists():
                    return issue, "skipped"
                
                # Download PDF to memory
                resp = session.get(url, timeout=60)
                if resp.status_code != 200:
                    return issue, f"http_{resp.status_code}"
                
                content = resp.content
                if len(content) < 1000:
                    return issue, "too_small"
                
                # Upload to GCS
                blob.upload_from_string(content, content_type="application/pdf")
                
                return issue, "ok"
                
            except Exception as e:
                return issue, f"error:{str(e)[:50]}"

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(job, issue): issue for issue in issues}
            
            for fut in as_completed(futures):
                issue, status = fut.result()
                
                with self._lock:
                    if status == "ok":
                        self.uploaded_count += 1
                        issue["uploaded"] = True
                        issue["gcs_path"] = f"gs://{self.bucket_name}/pdfs/{issue['year']}/{issue.get('month', 1):02d}/{issue['issue_id']}.pdf"
                        self.catalog.append(issue)
                    elif status == "skipped":
                        self.skipped_count += 1
                        issue["uploaded"] = True
                        self.catalog.append(issue)
                    else:
                        self.failed_count += 1
                        logger.debug(f"Failed {issue['issue_id']}: {status}")
                    
                    # Save catalog every 100 uploads
                    if (self.uploaded_count + self.skipped_count) % 100 == 0:
                        self._save_catalog()
                
                pbar.update(1)
        
        pbar.close()

    def _select_year(self, page, year: int) -> bool:
        """Select year in the AngularJS dropdown."""
        year_str = str(year)
        try:
            result = page.evaluate(f"""() => {{
                const spans = document.querySelectorAll('span[ng-click*="getIssueByYear"]');
                for (const sp of spans) {{
                    if (sp.textContent.trim() === '{year_str}') {{
                        try {{
                            const scope = angular.element(sp).scope();
                            if (scope && scope.getIssueByYear) {{
                                scope.getIssueByYear('{year_str}');
                                scope.$apply();
                                return 'scope';
                            }}
                        }} catch (e) {{}}
                        sp.click();
                        return 'click';
                    }}
                }}
                return null;
            }}""")
            
            if result:
                page.wait_for_load_state("networkidle", timeout=10000)
                time.sleep(1)
                return True
            return False
            
        except Exception as e:
            logger.debug(f"Year select error: {e}")
            return False

    def _dismiss_cookies(self, page) -> None:
        """Try to dismiss cookie consent."""
        try:
            for selector in ["#accept-cookies", ".cookie-accept", "[data-accept]"]:
                btn = page.query_selector(selector)
                if btn:
                    btn.click()
                    time.sleep(0.5)
                    break
        except:
            pass

    def _load_existing_catalog(self) -> None:
        """Load existing catalog for resume support."""
        if self.catalog_path.exists():
            try:
                with open(self.catalog_path, "r", encoding="utf-8") as f:
                    self.catalog = json.load(f)
                logger.info(f"Loaded existing catalog: {len(self.catalog)} entries")
            except:
                self.catalog = []

    def _save_catalog(self) -> None:
        """Save catalog to disk."""
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.catalog_path, "w", encoding="utf-8") as f:
            json.dump(self.catalog, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Fast PDF scraper to GCS with month from website")
    parser.add_argument("--year-start", type=int, default=1944)
    parser.add_argument("--year-end", type=int, default=2025)
    parser.add_argument("--workers", type=int, default=30, help="Concurrent downloads")
    parser.add_argument("--bucket", type=str, default=GCS_BUCKET_NAME)
    parser.add_argument("--fresh", action="store_true", help="Clear existing GCS files and start fresh")
    args = parser.parse_args()

    scraper = FastGCSScraper(
        year_start=args.year_start,
        year_end=args.year_end,
        workers=args.workers,
        bucket_name=args.bucket,
    )
    
    count = scraper.scrape_and_upload(fresh=args.fresh)
    print(f"\n✓ Uploaded {count} PDFs to gs://{args.bucket}/")


if __name__ == "__main__":
    main()
