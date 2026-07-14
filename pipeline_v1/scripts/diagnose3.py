#!/usr/bin/env python3
"""Check WHERE Albanian text appears in the 'bilingual' 2008 PDF."""
import pdfplumber
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def script_stats(text):
    cyr = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    lat = sum(1 for c in text if ('A' <= c <= 'Z') or ('a' <= c <= 'z'))
    return cyr, lat

pdf_path = os.path.join(BASE, 'data/pdfs/2008/03/3B7481999CB57F4095B31D5726000D7F.pdf')
print(f"=== ALL PAGES of {os.path.basename(pdf_path)} (120 pages) ===\n")

with pdfplumber.open(pdf_path) as pdf:
    prev_label = None
    for i, page in enumerate(pdf.pages):
        full_text = page.extract_text() or ""
        c, l = script_stats(full_text)
        total = c + l
        if total == 0:
            label = "EMPTY"
        elif l > c:
            label = "LATIN-dominant"
        elif c > l * 3:
            label = "CYRILLIC"
        else:
            label = "MIXED"
        
        # Only print when language changes or at boundaries
        if label != prev_label:
            print(f"Page {i+1:3d}: {label} (cyr={c}, lat={l}, lat%={l/max(total,1)*100:.0f}%)")
            if label == "LATIN-dominant" or (label == "MIXED" and prev_label != "MIXED"):
                print(f"    First 150 chars: {full_text[:150]}")
            prev_label = label
    
    print(f"\nTotal pages: {len(pdf.pages)}")
    
    # Show the transition page in detail
    print("\n=== Looking for MK->SQ transition ===")
    for i, page in enumerate(pdf.pages):
        full_text = page.extract_text() or ""
        c, l = script_stats(full_text)
        total = c + l
        if total > 0 and l > c:
            print(f"\nFirst Latin-dominant page: {i+1}")
            print(f"Text[:300]: {full_text[:300]}")
            break
