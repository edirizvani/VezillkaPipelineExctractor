#!/usr/bin/env python3
"""Debug Angular scope structure after selecting a year."""
from playwright.sync_api import sync_playwright
import time
import json

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True)
    page = browser.new_page()
    
    page.goto("https://slvesnik.com.mk/besplaten-pristap-do-izdanija.nspx", wait_until="networkidle", timeout=60000)
    time.sleep(2)
    
    # Select 1947
    page.evaluate("""() => {
        const spans = document.querySelectorAll('span[ng-click*="getIssueByYear"]');
        for (const sp of spans) {
            if (sp.textContent.trim() === '1947') {
                sp.click();
                break;
            }
        }
    }""")
    time.sleep(2)
    
    # Check if scope.issue exists per link
    print("=== Check scope.issue on each link ===")
    data = page.evaluate("""() => {
        const links = document.querySelectorAll('a[href*="Issues"]');
        const results = [];
        for (let i = 0; i < Math.min(5, links.length); i++) {
            const link = links[i];
            let issueData = null;
            try {
                const scope = angular.element(link).scope();
                if (scope && scope.issue) {
                    issueData = JSON.parse(JSON.stringify(scope.issue));
                }
            } catch(e) {
                issueData = 'error: ' + e.toString();
            }
            results.push({
                href: link.href,
                text: link.textContent.trim(),
                issueData: issueData
            });
        }
        return results;
    }""")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    
    # Also check parent elements
    print("\n=== Check parent container scope ===")
    data2 = page.evaluate("""() => {
        const link = document.querySelector('a[href*="Issues"]');
        if (!link) return 'no link found';
        
        // Walk up parents checking for scope with month/issues
        let el = link;
        const results = [];
        for (let i = 0; i < 10 && el; i++) {
            try {
                const scope = angular.element(el).scope();
                if (scope) {
                    const keys = Object.keys(scope).filter(k => !k.startsWith('$') && !k.startsWith('_'));
                    results.push({
                        level: i,
                        tag: el.tagName,
                        keys: keys.slice(0, 15),
                        hasIssue: !!scope.issue,
                        hasMonth: !!scope.month,
                        hasMonthName: !!scope.monthName,
                    });
                }
            } catch(e) {}
            el = el.parentElement;
        }
        return results;
    }""")
    print(json.dumps(data2, indent=2, ensure_ascii=False))
    
    # Check for month in ng-repeat context
    print("\n=== Check ng-repeat context for month ===")
    data3 = page.evaluate("""() => {
        const ngRepeats = document.querySelectorAll('[ng-repeat*="month"], [ng-repeat*="Month"]');
        if (ngRepeats.length === 0) {
            // Try all ng-repeats
            const all = document.querySelectorAll('[ng-repeat]');
            return {
                monthRepeatCount: 0,
                allNgRepeats: Array.from(all).slice(0, 10).map(e => e.getAttribute('ng-repeat'))
            };
        }
        return {
            monthRepeatCount: ngRepeats.length,
            sample: ngRepeats[0]?.getAttribute('ng-repeat')
        };
    }""")
    print(json.dumps(data3, indent=2, ensure_ascii=False))
    
    browser.close()
