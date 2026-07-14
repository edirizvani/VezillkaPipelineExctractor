#!/usr/bin/env python3
"""Check website structure for month selectors."""
from playwright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://slvesnik.com.mk/besplaten-pristap-do-izdanija.nspx', timeout=60000)
    time.sleep(3)
    
    # Select year 1947
    js_code = """() => {
        const spans = document.querySelectorAll('span[ng-click*="getIssueByYear"]');
        for (const sp of spans) {
            if (sp.textContent.trim() === '1947') {
                try {
                    const scope = angular.element(sp).scope();
                    if (scope && scope.getIssueByYear) {
                        scope.getIssueByYear('1947');
                        scope.$apply();
                        return 'scope';
                    }
                } catch (e) {}
                sp.click();
                return 'click';
            }
        }
        return null;
    }"""
    result = page.evaluate(js_code)
    print(f'Year selection result: {result}')
    time.sleep(3)
    
    # Now check for month selector or all PDF links
    all_ng_clicks = page.query_selector_all('[ng-click]')
    print(f'\nTotal ng-click elements after year select: {len(all_ng_clicks)}')
    
    # Look for month-related
    print('\nMonth-related ng-click elements:')
    for el in all_ng_clicks:
        ng = el.get_attribute('ng-click')
        if 'month' in ng.lower() or 'Month' in ng:
            text = el.inner_text()[:40] if el.inner_text() else ''
            print(f'  MONTH: {ng} -> "{text}"')
    
    # Count PDFs visible
    links = page.query_selector_all("a[href*='Issues']")
    print(f'\nPDF links visible: {len(links)}')
    
    # Print first few PDF texts
    print('\nFirst 10 PDFs:')
    for link in links[:10]:
        text = link.inner_text()[:50] if link.inner_text() else ''
        print(f'  PDF: "{text}"')
    
    # Print last few PDF texts
    print('\nLast 10 PDFs:')
    for link in links[-10:]:
        text = link.inner_text()[:50] if link.inner_text() else ''
        print(f'  PDF: "{text}"')
    
    browser.close()
