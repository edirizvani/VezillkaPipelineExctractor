#!/usr/bin/env python3
"""Check page-by-page layout of a known bilingual PDF (2008)."""
import pdfplumber
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def script_stats(text):
    cyr = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    lat = sum(1 for c in text if ('A' <= c <= 'Z') or ('a' <= c <= 'z'))
    return cyr, lat

# This PDF was 46.9% Latin - clearly bilingual
pdf_path = os.path.join(BASE, 'data/pdfs/2008/03/3B7481999CB57F4095B31D5726000D7F.pdf')
print(f"=== Bilingual PDF page-by-page: {os.path.basename(pdf_path)} ===\n")

with pdfplumber.open(pdf_path) as pdf:
    total = len(pdf.pages)
    print(f"Total pages: {total}\n")
    
    for i, page in enumerate(pdf.pages[:20]):  # first 20 pages
        w = page.width
        mid = w / 2.0
        words = page.extract_words()
        if not words:
            print(f"Page {i+1}: NO WORDS")
            continue
        
        left = [wd for wd in words if (wd['x0'] + wd['x1']) / 2 < mid]
        right = [wd for wd in words if (wd['x0'] + wd['x1']) / 2 >= mid]
        
        left_text = ' '.join(wd['text'] for wd in sorted(left, key=lambda x: (x['top'], x['x0'])))
        right_text = ' '.join(wd['text'] for wd in sorted(right, key=lambda x: (x['top'], x['x0'])))
        
        lc, ll = script_stats(left_text)
        rc, rl = script_stats(right_text)
        
        l_label = "CYR" if lc > ll*3 else ("LAT" if ll > lc*3 else "MIX")
        r_label = "CYR" if rc > rl*3 else ("LAT" if rl > rc*3 else "MIX")
        
        bilin = "MK|SQ" if l_label == "CYR" and r_label == "LAT" else \
                "SQ|MK" if l_label == "LAT" and r_label == "CYR" else \
                "SAME" if l_label == r_label else "OTHER"
        
        print(f"Page {i+1:3d}: L={l_label}(c={lc},l={ll}) R={r_label}(c={rc},l={rl})  => {bilin}")
        
        # Show first few chars for interesting pages
        if i < 5 or bilin != "SAME":
            print(f"    L: {left_text[:80]}")
            print(f"    R: {right_text[:80]}")
