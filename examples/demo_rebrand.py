"""Demo: re-theme a matplotlib chart (viridis colormap + DejaVu font) to a brand style
(#00407A / #52BDEC + Liberation Sans) from the rendered PNG alone.

Run: python examples/demo_rebrand.py
Requires: pip install "figsurgeon[rebrand]"  AND  apt-get install tesseract-ocr
          (pytesseract is a wrapper around the tesseract binary, which pip cannot install)

Box coordinates below are hand-measured for tests/data/rebrand_source.png specifically --
this script demonstrates the pipeline, not automatic region detection. A different chart
needs its own boxes, the same way remove_object and clean_transparent_box already require
an explicit region rather than guessing one.
"""
import os

from PIL import Image
from figsurgeon import rebrand as R

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'tests', 'data', 'rebrand_source.png')
OUT = os.path.join(HERE, 'output')
os.makedirs(OUT, exist_ok=True)
BRAND_COLOURS = ('#00407A', '#52BDEC')

im = Image.open(SRC).convert('RGB')

# Step 1: remap the viridis-based data cloud and its colorbar to the brand gradient.
# Both use the same source colormap and the same detected alpha (0.76), found by
# grid-search: matching against the pure LUT without this correction matches only 4.3%
# of pixels.
step1, frac_data, alpha = R.remap_colormap(im, box=(103, 4, 735, 252),
                                           new_colours=BRAND_COLOURS,
                                           match_tolerance=30, alpha=0.76)
step2, frac_cbar, _ = R.remap_colormap(step1, box=(776, 5, 788, 252),
                                       new_colours=BRAND_COLOURS,
                                       match_tolerance=30, alpha=0.76)
print(f'colormap alpha detected: {alpha}')
print(f'data area matched: {frac_data:.0%}  |  colorbar matched: {frac_cbar:.0%}')

# Step 2: swap every text element to Liberation Sans (the standard free Arial-metric
# substitute -- literal Arial is proprietary and typically unavailable).
# The two rotated labels (y-axis title, colorbar title) need explicit regions: plain OCR
# reads rotated text as garbage at deceptively high confidence (77-89), so confidence
# filtering alone cannot separate it from real content.
final, log = R.replace_font(
    step2, min_confidence=55,
    rotated_regions=[((15, 0, 56, 260), 90), ((826, 15, 896, 240), 90)])
print(f'{len(log)} text elements redrawn in Liberation Sans')

final.save(os.path.join(OUT, 'rebrand_final.png'))
im.save(os.path.join(OUT, 'rebrand_before.png'))
print('saved examples/output/rebrand_before.png / rebrand_final.png')
