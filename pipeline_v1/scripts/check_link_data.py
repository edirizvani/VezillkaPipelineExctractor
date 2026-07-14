#!/usr/bin/env python3
"""Check if PDF links or surrounding HTML contain date info."""
from playwright.sync_api import sync_playwright
import time
import re

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
    
    # Get all PDF links with their surrounding context
    print("Checking PDF links and their HTML context for date info...\n")
    
    links = page.query_selector_all("a[href*='Issues']")
    
    # Check first 10 links
    for i, link in enumerate(links[:10]):
        href = link.get_attribute("href") or ""
        text = (link.inner_text() or "").strip()
        
        # Get parent element for more context
        parent = link.evaluate("el => el.parentElement ? el.parentElement.outerHTML.slice(0, 500) : ''")
        
        # Get any data attributes
        all_attrs = link.evaluate("el => Object.fromEntries([...el.attributes].map(a => [a.name, a.value]))")
        
        print(f"Link {i+1}: {text}")
        print(f"  href: {href[-50:]}")
        print(f"  attrs: {all_attrs}")
        # Look for date patterns in parent
        date_patterns = re.findall(r'\d{1,2}[./]\d{1,2}[./]\d{4}|\d{1,2}\s+\w+\s+\d{4}', parent)
        if date_patterns:
            print(f"  dates in parent: {date_patterns}")
        print()
    
    # Check Angular scope data
    print("\nChecking Angular scope for issue data...")
    scope_data = page.evaluate("""() => {
        const link = document.querySelector('a[href*="Issues"]');
        if (!link) return null;
        try {
            const scope = angular.element(link).scope();
            // Get the issue object if available
            if (scope && scope.issue) {
                return JSON.stringify(scope.issue);
            }
            // Check for issues array
            if (scope && scope.issues && scope.issues.length > 0) {
                return JSON.stringify(scope.issues.slice(0, 3));
            }
            // Get all scope keys
            const keys = Object.keys(scope || {}).filter(k => !k.startsWith('$'));
            return 'Scope keys: ' + keys.join(', ');
        } catch(e) {
            return 'Error: ' + e.message;
        }
    }""")
    print(f"Scope data: {scope_data[:1000] if scope_data else 'None'}")
    
    browser.close()
