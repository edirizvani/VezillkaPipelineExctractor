"""Deep dive into NoiseFilter behavior."""
import sys
from pathlib import Path

PIPELINE_V2 = Path(__file__).resolve().parent.parent
SAMPLES = PIPELINE_V2.parent / "pipeline_v1" / "data" / "samples"
sys.path.insert(0, str(PIPELINE_V2))

from phase3_extract.extractor_sequential import SequentialExtractor
from phase3_extract.layout_classifier import LayoutClassifier
from phase5_clean.noise_filter import NoiseFilter
from config import DEFAULT_CONFIG
from utils.text_utils import (
    cyrillic_ratio, latin_ratio, is_noise_line, strip_headers_footers
)

pdf_path = SAMPLES / '37b63d88c83044869283315a643e3779.pdf'

# Get raw text
classifier = LayoutClassifier()
layout = classifier.classify(pdf_path)
ext = SequentialExtractor()
result = ext.extract(pdf_path, boundary_page=layout.boundary_page, boundary_block=layout.boundary_block)

print("=== RAW MK TEXT ANALYSIS (boundary_page=2, pages 0-1) ===")
mk = result.raw_mk
print(f"Total chars: {len(mk)}")
print(f"Cyrillic ratio (raw): {cyrillic_ratio(mk):.3f}")
print(f"Latin ratio (raw): {latin_ratio(mk):.3f}")

# Step-by-step NoiseFilter.clean_mk
mk_after_strip = strip_headers_footers(mk)
print(f"\nAfter strip_headers_footers: {len(mk_after_strip)} chars")

lines = mk_after_strip.split("\n")
noise_lines = [(i, l) for i, l in enumerate(lines) if is_noise_line(l)]
content_lines = [(i, l) for i, l in enumerate(lines) if not is_noise_line(l)]
print(f"Total lines: {len(lines)}")
print(f"Noise lines removed: {len(noise_lines)}")
print(f"Content lines kept: {len(content_lines)}")

remaining = "\n".join(l for _, l in content_lines)
cyr = cyrillic_ratio(remaining)
print(f"\nRemaining text cyrillic_ratio: {cyr:.3f}")
print(f"Remaining text length: {len(remaining.strip())}")
print(f"mk_min_cyrillic threshold: {DEFAULT_CONFIG.mk_min_cyrillic}")
print(f"Would discard: {cyr < DEFAULT_CONFIG.mk_min_cyrillic and len(remaining.strip()) > 20}")

if content_lines:
    print(f"\nFirst 5 content lines:")
    for i, l in content_lines[:5]:
        print(f"  [{i}] {repr(l[:100])}")
    print(f"\nLast 5 content lines:")
    for i, l in content_lines[-5:]:
        print(f"  [{i}] {repr(l[:100])}")

print("\n\n=== RAW SQ TEXT ANALYSIS (pages 2-43) ===")
sq = result.raw_sq
print(f"Total chars: {len(sq)}")
print(f"Cyrillic ratio (raw): {cyrillic_ratio(sq):.3f}")
print(f"Latin ratio (raw): {latin_ratio(sq):.3f}")

sq_after_strip = strip_headers_footers(sq)
sq_lines = sq_after_strip.split("\n")
sq_noise = [(i, l) for i, l in enumerate(sq_lines) if is_noise_line(l)]
sq_content = [(i, l) for i, l in enumerate(sq_lines) if not is_noise_line(l)]
remaining_sq = "\n".join(l for _, l in sq_content)
lat = latin_ratio(remaining_sq)
print(f"\nAfter noise removal:")
print(f"  Lines kept: {len(sq_content)}")
print(f"  Latin ratio: {lat:.3f}")
print(f"  sq_min_latin threshold: {DEFAULT_CONFIG.sq_min_latin}")
print(f"  Would discard: {lat < DEFAULT_CONFIG.sq_min_latin and len(remaining_sq.strip()) > 20}")

# Test with a DIFFERENT PDF (boundary_page=18, should be more reasonable)
print("\n\n=== TEST PDF #2 (boundary_page=18) ===")
pdf2 = SAMPLES / '4fd75fa9d93541a68e68a7f602c2530d.pdf'
layout2 = classifier.classify(pdf2)
print(f"Layout: {layout2.layout_type}, boundary={layout2.boundary_page}, total_pages={layout2.total_pages}")
result2 = ext.extract(pdf2, boundary_page=layout2.boundary_page, boundary_block=layout2.boundary_block)
print(f"raw_mk chars: {len(result2.raw_mk)}")
print(f"raw_sq chars: {len(result2.raw_sq)}")
print(f"raw_mk cyrillic_ratio: {cyrillic_ratio(result2.raw_mk):.3f}")
print(f"raw_sq latin_ratio: {latin_ratio(result2.raw_sq):.3f}")

noise = NoiseFilter(DEFAULT_CONFIG)
mk2_clean = noise.clean_mk(result2.raw_mk)
sq2_clean = noise.clean_sq(result2.raw_sq)
print(f"After NoiseFilter: mk={len(mk2_clean)}, sq={len(sq2_clean)}")
if not mk2_clean:
    mk2_stripped = strip_headers_footers(result2.raw_mk)
    mk2_lines = [l for l in mk2_stripped.split("\n") if not is_noise_line(l)]
    mk2_remaining = "\n".join(mk2_lines)
    print(f"  mk after noise removal: {len(mk2_remaining)} chars, cyrillic_ratio={cyrillic_ratio(mk2_remaining):.3f}")
if not sq2_clean:
    sq2_stripped = strip_headers_footers(result2.raw_sq)
    sq2_lines = [l for l in sq2_stripped.split("\n") if not is_noise_line(l)]
    sq2_remaining = "\n".join(sq2_lines)
    print(f"  sq after noise removal: {len(sq2_remaining)} chars, latin_ratio={latin_ratio(sq2_remaining):.3f}")
