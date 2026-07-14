"""
PDF Downloader — concurrent gazette PDF downloader.

Reads the catalog produced by ``catalog_scraper.py`` and downloads
all PDFs that haven't been marked ``downloaded: true`` yet,
using a thread pool for parallel downloads.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# Suppress InsecureRequestWarning (the site's cert is untrusted)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


def _make_session() -> requests.Session:
    """Build a requests session with connection pooling and auto-retries."""
    s = requests.Session()
    s.headers["User-Agent"] = config.USER_AGENT
    s.verify = False
    adapter = HTTPAdapter(
        pool_connections=config.DOWNLOAD_WORKERS,
        pool_maxsize=config.DOWNLOAD_WORKERS,
        max_retries=Retry(
            total=config.MAX_RETRIES,
            backoff_factor=config.RETRY_BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
        ),
    )
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


class PDFDownloader:
    """
    Downloads Služben Vesnik PDFs from the catalog.

    Features:
      - Skips already-downloaded issues (resumable)
      - Concurrent thread-pool downloads (default 10 workers)
      - Automatic retries with backoff (urllib3)
      - tqdm progress bar
      - Thread-safe catalog persistence
    """

    def __init__(
        self,
        catalog_path: Path = config.CATALOG_PATH,
        pdf_dir: Path = config.PDF_DIR,
        workers: int = config.DOWNLOAD_WORKERS,
    ):
        self.catalog_path = catalog_path
        self.pdf_dir = pdf_dir
        self.workers = workers
        self._catalog_lock = threading.Lock()

    def download_all(self, limit: int | None = None) -> int:
        catalog = self._load_catalog()
        pending = [e for e in catalog if not e.get("downloaded")]

        if limit is not None:
            pending = pending[:limit]
        if not pending:
            logger.info("No pending downloads.")
            return 0

        logger.info(
            "Downloading %d PDFs with %d workers…", len(pending), self.workers
        )
        ok_count = 0
        session = _make_session()
        pbar = tqdm(total=len(pending), desc="Downloading PDFs", unit="pdf")

        # Periodically flush catalog (every N successes)
        SAVE_EVERY = 25
        unsaved = 0

        def _job(entry: dict) -> tuple[dict, bool]:
            return entry, self._download_one(entry, session)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(_job, e): e for e in pending}
            for fut in as_completed(futures):
                entry, success = fut.result()
                if success:
                    entry["downloaded"] = True
                    ok_count += 1
                    unsaved += 1
                pbar.update(1)

                if unsaved >= SAVE_EVERY:
                    with self._catalog_lock:
                        self._save_catalog(catalog)
                    unsaved = 0

        pbar.close()

        # Final save
        self._save_catalog(catalog)
        logger.info("Downloaded %d / %d PDFs", ok_count, len(pending))
        return ok_count

    # ── Download a single PDF ───────────────────────────────────

    def _download_one(self, entry: dict, session: requests.Session) -> bool:
        url = entry["url"]
        year = entry.get("year", 0)
        month = entry.get("month", 0)
        issue_id = entry["issue_id"]

        # Organise as  data/pdfs/<year>/<MM>/<hash>.pdf
        if month:
            dest_dir = self.pdf_dir / str(year) / f"{month:02d}"
        else:
            dest_dir = self.pdf_dir / str(year) / "00"
        dest_dir.mkdir(parents=True, exist_ok=True)
        output_path = dest_dir / f"{issue_id}.pdf"

        # Also check the old flat year-only location and relocate if found
        old_path = self.pdf_dir / str(year) / f"{issue_id}.pdf"
        if not output_path.exists() and old_path.exists() and old_path.stat().st_size > 1000:
            old_path.rename(output_path)
            logger.debug("Relocated %s → %s", old_path, output_path)

        if output_path.exists() and output_path.stat().st_size > 1000:
            logger.debug("Already exists: %s", output_path)
            return True

        try:
            resp = session.get(url, timeout=60, stream=True)
            resp.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)

            size = output_path.stat().st_size
            if size < 1000:
                logger.warning("Tiny PDF (%d B): %s", size, url)
                output_path.unlink(missing_ok=True)
                return False

            logger.debug("OK: %s (%d B)", output_path.name, size)
            return True

        except requests.RequestException as exc:
            logger.warning("FAILED %s: %s", url, exc)
            output_path.unlink(missing_ok=True)
            return False

    # ── Catalog I/O ─────────────────────────────────────────────

    def _load_catalog(self) -> list[dict]:
        if not self.catalog_path.exists():
            raise FileNotFoundError(
                f"Catalog not found: {self.catalog_path}\n"
                "Run catalog_scraper.py first."
            )
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_catalog(self, catalog: list[dict]) -> None:
        with open(self.catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, ensure_ascii=False, indent=2)


# ───────────────────────── CLI ──────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download Služben Vesnik PDFs"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--catalog", type=str, default=str(config.CATALOG_PATH)
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format="%(levelname)s %(name)s: %(message)s",
    )

    dl = PDFDownloader(catalog_path=Path(args.catalog))
    count = dl.download_all(limit=args.limit)
    print(f"✓ Downloaded {count} PDFs.")


if __name__ == "__main__":
    main()
