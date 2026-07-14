"""Test script to trace the pipeline flow for a specific PDF."""
import sys
from pathlib import Path

PIPELINE_V2 = Path(__file__).resolve().parent.parent
SAMPLES = PIPELINE_V2.parent / "pipeline_v1" / "data" / "samples"
sys.path.insert(0, str(PIPELINE_V2))

import pdfplumber
from phase3_extract.extractor_sequential import SequentialExtractor
from phase3_extract.layout_classifier import LayoutClassifier
from phase5_clean.noise_filter import NoiseFilter
from config import DEFAULT_CONFIG

pdf_path = SAMPLES / '37b63d88c83044869283315a643e3779.pdf'

# Step 1: Classify
classifier = LayoutClassifier()
layout = classifier.classify(pdf_path)
print("=== LAYOUT CLASSIFICATION ===")
print(f"  type: {layout.layout_type}")
print(f"  boundary_page: {layout.boundary_page}")
print(f"  confidence: {layout.confidence:.2f}")
print(f"  has_albanian: {layout.has_albanian}")
print(f"  detail: {layout.detail}")
print()

# Step 2: Extract with SequentialExtractor
ext = SequentialExtractor()
result = ext.extract(pdf_path, boundary_page=layout.boundary_page, boundary_block=layout.boundary_block)
print("=== SEQUENTIAL EXTRACTION ===")
print(f"  raw_mk chars: {len(result.raw_mk)}")
print(f"  raw_sq chars: {len(result.raw_sq)}")
print(f"  raw_mk first 200: {repr(result.raw_mk[:200])}")
print(f"  raw_sq first 200: {repr(result.raw_sq[:200])}")
print(f"  articles: {len(result.articles)}")
print()

# Step 3: Check pipeline empty guard
mk_text = result.raw_mk
sq_text = result.raw_sq
print("=== PIPELINE EMPTY CHECK ===")
print(f"  mk_text.strip() is empty: {not mk_text.strip()}")
print(f"  sq_text.strip() is empty: {not sq_text.strip()}")
if not mk_text.strip() or not sq_text.strip():
    print("  >>> WOULD SKIP (extraction_empty)")
else:
    print("  >>> WOULD PROCEED")
print()

# Step 4: Noise filter
noise = NoiseFilter(DEFAULT_CONFIG)
mk_clean = noise.clean_mk(mk_text)
sq_clean = noise.clean_sq(sq_text)
print("=== NOISE FILTER ===")
print(f"  mk_clean chars: {len(mk_clean)}")
print(f"  sq_clean chars: {len(sq_clean)}")
print(f"  mk_clean first 200: {repr(mk_clean[:200])}")
print()

# Step 5: Segmenter
from phase3_extract.document_segmenter import DocumentSegmenter
seg = DocumentSegmenter(DEFAULT_CONFIG)
seg_result = seg.segment(mk_clean, sq_clean)
print("=== SEGMENTATION ===")
print(f"  total_items_mk: {seg_result.total_items_mk}")
print(f"  total_items_sq: {seg_result.total_items_sq}")
print(f"  matched_items: {seg_result.matched_items}")
print(f"  items with is_bilingual: {sum(1 for i in seg_result.items if i.is_bilingual)}")
for item in seg_result.items[:3]:
    print(f"    item {item.item_number}: mk={len(item.mk_text)} sq={len(item.sq_text)} bilingual={item.is_bilingual}")
