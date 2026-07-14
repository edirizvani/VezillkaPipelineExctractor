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
    page.evaluate("""() => {
        const spans = document.querySelectorAll('span[ng-click*="getIssueByYear"]');
        for (const sp of spans) {
            if (sp.textContent.trim() === '1947') {
                const scope = angular.element(sp).scope();
                scope.getIssueByYear('1947');
                scope.$apply();
                break;
            }
        }
    }""")
    time.sleep(2)
    
    # Look for month selector elements
    all_ng_clicks = page.query_selector_all('[ng-click]')
    print("Looking for month-related elements after selecting year 1947:")
    
    month_elements = []
    for el in all_ng_clicks:
        ng = el.get_attribute('ng-click') or ''
        text = (el.inner_text() or '').strip()[:30]
        if 'month' in ng.lower():
            month_elements.append((text, ng))
            print(f"  MONTH: {ng} -> '{text}'")
    
    # Look for numbered spans (1-12) with ng-click
    print("\nSpans with numbers 1-12 and ng-click:")
    spans = page.query_selector_all('span[ng-click]')
    for sp in spans:
        text = (sp.inner_text() or '').strip()
        ng = sp.get_attribute('ng-click') or ''
        if text.isdigit() and 1 <= int(text) <= 12:
            print(f"  {text}: {ng}")
    
    # Try clicking month 11 if exists
    print("\nTrying to select month 11...")
    result = page.evaluate("""() => {
        // Look for month selector
        const monthSpans = document.querySelectorAll('span[ng-click*="getIssueByMonth"]');
        if (monthSpans.length > 0) {
            console.log('Found getIssueByMonth spans:', monthSpans.length);
            for (const sp of monthSpans) {
                if (sp.textContent.trim() === '11') {
                    const scope = angular.element(sp).scope();
                    if (scope && scope.getIssueByMonth) {
                        scope.getIssueByMonth('11');
                        scope.$apply();
                        return 'selected month 11 via scope';
                    }
                    sp.click();
                    return 'clicked month 11';
                }
            }
        }
        
        // Maybe months are shown differently
        const allSpans = document.querySelectorAll('span[ng-click]');
        for (const sp of allSpans) {
            const ng = sp.getAttribute('ng-click') || '';
            if (ng.includes('Month') || ng.includes('month')) {
                return 'found: ' + ng;
            }
        }
        return 'no month selector found';
    }""")
    print(f"Result: {result}")
    
    time.sleep(2)
    
    # Count PDFs after selecting month
    links = page.query_selector_all("a[href*='Issues']")
    print(f"\nPDF links visible after month select: {len(links)}")
    
    browser.close()
