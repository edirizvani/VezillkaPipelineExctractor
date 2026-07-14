#!/usr/bin/env python3
"""Diagnose column languages in a PDF."""
import pdfplumber
import os, glob

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def script_stats(text):
    cyr = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    lat = sum(1 for c in text if ('A' <= c <= 'Z') or ('a' <= c <= 'z'))
    return cyr, lat

# Check the problematic PDF
pdf_path = os.path.join(BASE, 'data/pdfs/2001/02/EB48E5392E54461FB15074CA772AFF64.pdf')
print(f"=== Checking {pdf_path} ===\n")

with pdfplumber.open(pdf_path) as pdf:
    total_pages = len(pdf.pages)
    print(f"Total pages: {total_pages}\n")
    
    for i, page in enumerate(pdf.pages):
        w = page.width
        mid = w / 2.0
        words = page.extract_words()
        if not words:
            print(f"Page {i+1}: NO WORDS (scanned?)")
            continue
        
        left = [wd for wd in words if (wd['x0'] + wd['x1']) / 2 < mid]
        right = [wd for wd in words if (wd['x0'] + wd['x1']) / 2 >= mid]
        
        left_text = ' '.join(wd['text'] for wd in sorted(left, key=lambda x: (x['top'], x['x0'])))
        right_text = ' '.join(wd['text'] for wd in sorted(right, key=lambda x: (x['top'], x['x0'])))
        
        lc, ll = script_stats(left_text)
        rc, rl = script_stats(right_text)
        
        l_label = "CYR" if lc > ll else ("LAT" if ll > lc else "MIX")
        r_label = "CYR" if rc > rl else ("LAT" if rl > rc else "MIX")
        
        print(f"Page {i+1:3d}: LEFT={l_label}(cyr={lc},lat={ll})  RIGHT={r_label}(cyr={rc},lat={rl})")
        
        # Print first 80 chars of each column for first 3 pages + any page with Latin
        if i < 3 or r_label == "LAT" or l_label == "LAT":
            print(f"    L: {left_text[:80]}")
            print(f"    R: {right_text[:80]}")

print("\n\n=== Now checking a few other PDFs for comparison ===\n")

# Check a couple more PDFs to find bilingual ones
pdf_dir = os.path.join(BASE, 'data/pdfs')
checked = 0
for root, dirs, files in os.walk(pdf_dir):
    for f in sorted(files):
        if not f.endswith('.pdf'):
            continue
        if checked >= 5:
            break
        fpath = os.path.join(root, f)
        if 'EB48E5392E54461FB15074CA772AFF64' in fpath:
            continue  # skip the one we already checked
        
        try:
            with pdfplumber.open(fpath) as pdf2:
                total_cyr = 0
                total_lat = 0
                for page in pdf2.pages:
                    full_text = page.extract_text() or ""
                    c, l = script_stats(full_text)
                    total_cyr += c
                    total_lat += l
                
                ratio = total_lat / max(total_cyr + total_lat, 1) * 100
                label = "BILINGUAL?" if ratio > 20 else "MK-only"
                print(f"{fpath}: {label} (Cyrillic={total_cyr}, Latin={total_lat}, Latin%={ratio:.1f}%)")
                
                # If bilingual, show first page columns
                if ratio > 20:
                    page = pdf2.pages[0]
                    w = page.width
                    mid = w / 2.0
                    words = page.extract_words()
                    left = [wd for wd in words if (wd['x0'] + wd['x1']) / 2 < mid]
                    right = [wd for wd in words if (wd['x0'] + wd['x1']) / 2 >= mid]
                    left_text = ' '.join(wd['text'] for wd in sorted(left, key=lambda x: (x['top'], x['x0'])))
                    right_text = ' '.join(wd['text'] for wd in sorted(right, key=lambda x: (x['top'], x['x0'])))
                    print(f"  P1 LEFT[:80]:  {left_text[:80]}")
                    print(f"  P1 RIGHT[:80]: {right_text[:80]}")
        except Exception as e:
            print(f"{fpath}: ERROR - {e}")
        
        checked += 1
    if checked >= 5:
        break

# Also scan ALL downloaded PDFs to find ratio
print("\n\n=== Overall bilingual scan of ALL downloaded PDFs ===\n")
bilingual_count = 0
mk_only_count = 0

for root, dirs, files in os.walk(pdf_dir):
    for f in sorted(files):
        if not f.endswith('.pdf'):
            continue
        fpath = os.path.join(root, f)
        try:
            with pdfplumber.open(fpath) as pdf3:
                total_cyr = 0
                total_lat = 0
                for page in pdf3.pages[:10]:  # check first 10 pages
                    full_text = page.extract_text() or ""
                    c, l = script_stats(full_text)
                    total_cyr += c
                    total_lat += l
                ratio = total_lat / max(total_cyr + total_lat, 1) * 100
                if ratio > 20:
                    bilingual_count += 1
                else:
                    mk_only_count += 1
        except:
            pass

print(f"Bilingual (>20% Latin): {bilingual_count}")
print(f"MK-only: {mk_only_count}")
print(f"Total checked: {bilingual_count + mk_only_count}")
