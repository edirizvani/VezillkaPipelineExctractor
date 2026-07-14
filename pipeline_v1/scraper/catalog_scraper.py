"""
Catalog Scraper — scrapes all Služben Vesnik PDF links using Playwright.

The website at https://slvesnik.com.mk/besplaten-pristap-do-izdanija.nspx
lists gazette issues per-year, but the year selector is JavaScript-driven
(ASP.NET postback).  We use Playwright to click each year tab and extract
every PDF link.

Usage::

    python -m scraper.catalog_scraper --year-start 2001 --year-end 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

# ── PDF URL regex (href uses backslash on the AngularJS site) ───
PDF_HASH_RE = re.compile(r"Issues[/\\]([a-f0-9]{32})\.pdf", re.IGNORECASE)
ISSUE_NUM_RE = re.compile(r"(\d+)\s*$")


class CatalogScraper:
    """
    Scrapes the Služben Vesnik free-access page for every gazette PDF
    URL from *year_start* to *year_end* inclusive.

    Uses Playwright (headless Chromium) because the year selector
    is a JavaScript-driven dropdown that can't be crawled with plain
    HTTP requests.
    """

    def __init__(
        self,
        year_start: int = config.MIN_YEAR,
        year_end: int = 2026,
        catalog_path: Path = config.CATALOG_PATH,
        headless: bool = True,
    ):
        self.year_start = year_start
        self.year_end = year_end
        self.catalog_path = catalog_path
        self.headless = headless

    # ── Main entry point ────────────────────────────────────────

    def scrape(self) -> list[dict]:
        """
        Scrape all years and return a combined catalog list.
        Merges with any existing catalog on disk (resumable).
        """
        from playwright.sync_api import sync_playwright

        existing = self._load_existing()
        existing_ids = {e["issue_id"] for e in existing}
        all_issues: list[dict] = list(existing)

        logger.info(
            "Scraping years %d–%d  (existing catalog: %d entries)",
            self.year_start, self.year_end, len(existing),
        )

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=self.headless)
            context = browser.new_context(
                user_agent=config.USER_AGENT,
                locale="mk-MK",
            )
            page = context.new_page()

            # Navigate to the free-access page
            logger.info("Loading %s …", config.CATALOG_URL)
            page.goto(config.CATALOG_URL, wait_until="networkidle", timeout=60000)
            time.sleep(2)

            # Dismiss cookie banner if present
            self._dismiss_cookies(page)

            for year in range(self.year_start, self.year_end + 1):
                logger.info("── Year %d ──", year)

                if not self._select_year(page, year):
                    logger.warning("Could not select year %d — skipping", year)
                    continue

                issues = self._extract_issues_from_page(page, year)
                new_count = 0
                for issue in issues:
                    if issue["issue_id"] not in existing_ids:
                        all_issues.append(issue)
                        existing_ids.add(issue["issue_id"])
                        new_count += 1

                logger.info(
                    "  Year %d: %d issues found, %d new",
                    year, len(issues), new_count,
                )

                # Save incrementally after each year
                self._save_catalog(all_issues)
                time.sleep(1)

            browser.close()

        logger.info("Scraping complete: %d total entries", len(all_issues))
        return all_issues

    # ── Year selection ──────────────────────────────────────────

    def _select_year(self, page, year: int) -> bool:
        """Click the year tab on the AngularJS page. Returns True on success.

        DOM structure per year::

            <span class="year ng-scope" ng-class="{'active': active==year.Year}">
                <span ng-click="getIssueByYear(year.Year)" class="ng-binding">
                    2020
                </span>
            </span>

        The **inner** span has the ng-click — we must click that one,
        or call the AngularJS scope function directly.
        """
        year_str = str(year)
        try:
            # Strategy 1: call AngularJS scope function directly (most reliable)
            clicked = page.evaluate(f"""() => {{
                // Find the inner <span ng-click="getIssueByYear(...)">
                const inner = document.querySelectorAll('span[ng-click*="getIssueByYear"]');
                for (const sp of inner) {{
                    if (sp.textContent.trim() === '{year_str}') {{
                        // Trigger via AngularJS scope
                        try {{
                            const scope = angular.element(sp).scope();
                            if (scope && scope.getIssueByYear) {{
                                scope.getIssueByYear('{year_str}');
                                scope.$apply();
                                return 'scope';
                            }}
                        }} catch(e) {{}}
                        // Fallback: plain click
                        sp.click();
                        return 'click';
                    }}
                }}
                return false;
            }}""")

            if not clicked:
                logger.warning("No clickable element found for year %d", year)
                return False

            logger.debug("Year %d triggered via %s", year, clicked)

            # Wait for AngularJS to update the DOM
            time.sleep(2)
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass
            time.sleep(1)

            # Verify the year is now active
            is_active = page.evaluate(f"""() => {{
                const spans = document.querySelectorAll('span.year.active');
                for (const sp of spans) {{
                    if (sp.textContent.trim() === '{year_str}') return true;
                }}
                return false;
            }}""")
            if not is_active:
                logger.debug(
                    "Year %d not marked active — content may still have loaded",
                    year,
                )

            return True

        except Exception as exc:
            logger.error("Error selecting year %d: %s", year, exc)
            return False

    # ── Extract PDF links from current page state ───────────────

    # Month label-for → month number mapping
    MONTH_MAP = {
        "JANUARY": 1, "FEBRUARY": 2, "MARCH": 3, "APRIL": 4,
        "MAY": 5, "JUNE": 6, "JULY": 7, "AUGUST": 8,
        "SEPTEMBER": 9, "OCTOBER": 10, "NOVEMBER": 11, "DECEMBER": 12,
    }

    def _extract_issues_from_page(self, page, year: int) -> list[dict]:
        """Parse all PDF links visible on the current page.

        The AngularJS DOM groups issues by month::

            <div ng-repeat="issue in issues">
                <h2 label-for="JANUARY">Јануари</h2>
                <div class="documents">
                    <a href="Issues\\hash.pdf">…</a>
                </div>
            </div>

        We walk each month group to tag every PDF with its month.
        """
        issues: list[dict] = []

        links = page.evaluate("""() => {
            const results = [];
            // Walk each month-group container
            const groups = document.querySelectorAll('div[ng-repeat*="issue"]');
            for (const grp of groups) {
                // The <h2 label-for="JANUARY"> tells us the month
                const h2 = grp.querySelector('h2[label-for]');
                const monthLabel = h2 ? h2.getAttribute('label-for') : '';
                const anchors = grp.querySelectorAll('a');
                for (const a of anchors) {
                    const href = a.getAttribute('href') || '';
                    if (href.includes('Issues') && href.endsWith('.pdf')) {
                        results.push({
                            href: href,
                            text: a.textContent.trim(),
                            month: monthLabel
                        });
                    }
                }
            }
            // Fallback: if no month groups found, scan all links
            if (results.length === 0) {
                const anchors = document.querySelectorAll('a');
                for (const a of anchors) {
                    const href = a.getAttribute('href') || '';
                    if (href.includes('Issues') && href.endsWith('.pdf')) {
                        results.push({
                            href: href,
                            text: a.textContent.trim(),
                            month: ''
                        });
                    }
                }
            }
            return results;
        }""")

        seen = set()
        for link_data in links:
            href = link_data["href"]
            text = link_data["text"]
            month_label = link_data.get("month", "")

            m = PDF_HASH_RE.search(href)
            if not m:
                continue

            issue_id = m.group(1)
            if issue_id in seen:
                continue
            seen.add(issue_id)

            # Build a proper download URL with forward slashes
            pdf_url = f"{config.BASE_URL}/Issues/{issue_id}.pdf"

            # Extract issue number from text
            # e.g. "СЛУЖБЕН ВЕСНИК НА РСМ 42" → "42"
            num_match = ISSUE_NUM_RE.search(text)
            issue_number = num_match.group(1) if num_match else ""

            # Map month label to number (1-12)
            month_num = self.MONTH_MAP.get(month_label.upper(), 0)

            issues.append({
                "issue_id": issue_id,
                "url": pdf_url,
                "year": year,
                "month": month_num,
                "issue_number": issue_number,
                "label": text,
                "downloaded": False,
                "processed": False,
                "bilingual": None,   # determined at extraction time
            })

        return issues

    # ── Cookie banner ───────────────────────────────────────────

    def _dismiss_cookies(self, page) -> None:
        """Try to dismiss cookie consent banner."""
        try:
            for btn_text in [
                "ПРИФАТИ ГИ СИТЕ КОЛАЧИЊА",
                "Прифати",
                "Accept All",
                "Accept",
            ]:
                try:
                    btn = page.locator(f"button:has-text('{btn_text}')").first
                    if btn.is_visible(timeout=3000):
                        btn.click()
                        time.sleep(1)
                        logger.debug("Dismissed cookie banner: %s", btn_text)
                        return
                except Exception:
                    continue
        except Exception:
            pass

    # ── Catalog I/O ─────────────────────────────────────────────

    def save_catalog(self, catalog: list[dict]) -> None:
        self._save_catalog(catalog)

    def _save_catalog(self, catalog: list[dict]) -> None:
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, ensure_ascii=False, indent=2)
        logger.debug("Saved %d entries → %s", len(catalog), self.catalog_path)

    def _load_existing(self) -> list[dict]:
        if self.catalog_path.exists():
            try:
                with open(self.catalog_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info("Loaded existing catalog: %d entries", len(data))
                return data
            except (json.JSONDecodeError, IOError):
                logger.warning("Bad catalog file — starting fresh")
        return []


# ───────────────────────── CLI ──────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scrape Služben Vesnik catalog (all PDF links per year)"
    )
    parser.add_argument("--year-start", type=int, default=config.MIN_YEAR,
                        help="First year to scrape (default: 2001)")
    parser.add_argument("--year-end", type=int, default=2026,
                        help="Last year to scrape (default: 2026)")
    parser.add_argument("--output", type=str, default=str(config.CATALOG_PATH),
                        help="Path for catalog JSON output")
    parser.add_argument("--no-headless", action="store_true",
                        help="Show browser window (debug mode)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    scraper = CatalogScraper(
        year_start=args.year_start,
        year_end=args.year_end,
        catalog_path=Path(args.output),
        headless=not args.no_headless,
    )
    catalog = scraper.scrape()
    scraper.save_catalog(catalog)

    # Summary
    years = sorted(set(e["year"] for e in catalog))
    print(f"\n✓ Catalog complete: {len(catalog)} issues across {len(years)} years")
    for y in years:
        count = sum(1 for e in catalog if e["year"] == y)
        print(f"  {y}: {count} issues")


if __name__ == "__main__":
    main()
