"""Regression tests for advanced.clean_transparent_box.

  1. Single-column legend: a bleed-through blob near the border can cross the min-channel
     protection threshold. Protected blobs must be positionally consistent.

  2. Multi-column legend: the min-channel "deviation" metric is hue-biased, scoring a pale
     colour lower even at full opacity than a saturated one blended halfway to white.
     Saturation (max-min spread) avoids this, since it scales with blend toward white for
     any hue.
"""
import os

import numpy as np
from PIL import Image
from scipy import ndimage

from figsurgeon import advanced as A

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def _blob_sizes(crop, thresh=20):
    a = np.array(crop).astype(int)
    mx, mn = a.max(axis=2), a.min(axis=2)
    tinted = (mx - mn) > thresh
    labels, n = ndimage.label(tinted, structure=np.ones((3, 3)))
    return sorted((int((labels == i).sum()) for i in range(1, n + 1)
                  if (labels == i).sum() >= 3), reverse=True)


def _inside(img, box, inset=8):
    """The legend's own interior, not the caller's box, which overshoots the legend frame
    by a few pixels into the chart's own data and would otherwise read as extra glyph blobs.
    """
    x0, y0, x1, y1 = box
    return img.crop((x0 + inset, y0 + inset, x1 - inset, y1 - inset))


def run():
    checks = []

    # Case 1: single-column legend -- exactly 3 real glyphs, no stray bleed kept. The image
    # is a private figure, not in the public repository; the case is skipped without it.
    path1 = os.path.join(DATA, 'legend_source.png')
    if os.path.exists(path1):
        im1 = Image.open(path1).convert('RGB')
        box1 = (258, 288, 698, 537)
        full1, _ = A.clean_transparent_box(im1, box=box1)
        sizes1 = _blob_sizes(_inside(full1, box1))
        checks.append(('single-column: exactly 3 glyph blobs survive', len(sizes1) == 3))
        # Sizes shrank when the interior tint threshold was halved: each swatch loses its
        # faint anti-aliased ring to the fill. See evals/legend_clean.py for the trade.
        checks.append(('single-column: sizes match known-good values',
                       sizes1 == sorted([264, 239, 131], reverse=True)))
    else:
        print('  SKIP  single-column: tests/data/legend_source.png is not in this checkout')

    # Case 2: multi-column legend (2x2) -- exactly 4 real glyphs, INCLUDING the pale pink
    # one that the earlier min-channel metric silently dropped
    im2 = Image.open(os.path.join(DATA, 'moderate_multicol_source.png')).convert('RGB')
    box2 = (240, 390, 460, 460)
    full2, _ = A.clean_transparent_box(im2, box=box2)
    sizes2 = [s for s in _blob_sizes(_inside(full2, box2, inset=4)) if s >= 20]
    checks.append(('multi-column: exactly 4 glyph blobs survive (was 2, then 3)',
                   len(sizes2) == 4))
    checks.append(('multi-column: all 4 blobs are similar size (one real swatch type)',
                   len(sizes2) == 4 and max(sizes2) - min(sizes2) < 15))

    ok = True
    for label, passed in checks:
        print(f'  {"PASS" if passed else "FAIL"}  {label}')
        ok &= passed
    return ok


if __name__ == '__main__':
    import sys
    sys.exit(0 if run() else 1)


def test_transparent_box_checks():
    assert run(), 'see printed output for which legend-cleaning check failed'
