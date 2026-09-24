"""A candidate PASS signal for `check_object_erased`, measured but not shipped.

    .venv/bin/python evals/erase_structure.py

`check_object_erased` can only say FAIL or abstain. This measures `edge_ratio`: Canny edge
density inside the erased box vs. a ring of real pixels just outside it, on `fill='lama'`
only -- TELEA's smoother interpolation does not separate on this measure at all.

A threshold of 0.75 separates the real low-band set (4 clean, 7 ghost) except two cases:
`sailboat_mast` reads low but is recovered by a `solidity` shape gate (the before-edit
extent's own area over its convex hull); `red_shoes_pair` reads low and is not recoverable --
a smooth shoe-shaped mass against a busier background beats the edge measure regardless.

Not shipped: `verify_photo.check_object_erased` is unchanged. Kept so this false-PASS result
is not rediscovered by guessing the target set again.
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from evals.erase_field import CASES, JUDGEMENTS_LAMA, JUDGEMENTS_TELEA, load_working  # noqa: E402
from figsurgeon import objects as O                                                  # noqa: E402
from figsurgeon import verify_photo as V                                             # noqa: E402

# The 4 clean LaMa erases and 7 ghosts that read in the low band (<= V._ERASE_GONE) under
# fill='lama' -- see the module docstring.
CLEAN_LAMA = {'roadworks_cone', 'car_red_street', 'street_person', 'red_door'}
GHOST_LAMA = {'sailboat_mast', 'striped_bollard', 'white_horse', 'cow_at_fence',
              'motorcycle', 'red_shoes_pair', 'guitarist'}


def edge_ratio(after, box, band_frac=0.15, min_band=6):
    """Canny edge density inside `box` divided by the density in a ring of real, untouched
    pixels just outside it -- both on `after` only. >1 means the fill has more structure
    than its own surround; <1 means less. Neither direction is safe to threshold alone (the
    sailboat misses low, the shoes miss high) -- see the module docstring.
    """
    a = np.asarray(after.convert('RGB'))
    H, W = a.shape[:2]
    x0, y0, x1, y1 = (max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3]))
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    e_box = float((cv2.Canny(gray[y0:y1, x0:x1], 50, 150) > 0).mean())

    band = max(min_band, int(band_frac * min(x1 - x0, y1 - y0)))
    ox0, oy0 = max(0, x0 - band), max(0, y0 - band)
    ox1, oy1 = min(W, x1 + band), min(H, y1 + band)
    ring = np.zeros((H, W), bool)
    ring[oy0:oy1, ox0:ox1] = True
    ring[y0:y1, x0:x1] = False
    e_ring = float((cv2.Canny(gray, 50, 150).astype(bool) & ring).sum()) / float(ring.sum())
    return e_box / (e_ring + 1e-6)


def solidity(before, box, subject):
    """The before-edit phrase extent's own area over its convex hull -- a property of the
    object's shape, resolved before the edit, sharing no evidence with the fill. Low for a
    thin/skeletal object (a mast and rigging); high for a compact one.
    """
    asked, _position = V._drop_position(subject)
    prob = V._phrase_probability(before.convert('RGB'), asked, box)
    x0, y0, x1, y1 = box
    sub = (prob[y0:y1, x0:x1] > V._PHRASE_THRESHOLD).astype(np.uint8)
    contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    hull = cv2.convexHull(np.concatenate(contours))
    hull_area = cv2.contourArea(hull)
    return float(sub.sum()) / hull_area if hull_area > 0 else 0.0


def _run(fill, targets, judgements):
    print(f"\n--- fill={fill!r} ---")
    rows = []
    for name, rel, box, subject, category, note in CASES:
        if name not in targets:
            continue
        before = load_working(rel)
        out, _frame_frac, _warning = O.erase_object(before, box=box, fill=fill)
        ratio = edge_ratio(out, box)
        label = judgements.get(name, ('?', ''))[0]
        rows.append((name, label, ratio))
        print(f'{name:18s} {label:6s} edge_ratio={ratio:.3f}')
    return rows


def true_lama_low_band():
    """The real 4 clean + 7 ghost cases -- see CLEAN_LAMA / GHOST_LAMA above."""
    return _run('lama', CLEAN_LAMA | GHOST_LAMA, JUDGEMENTS_LAMA)


def telea_edge_ratios():
    """The same measure against every TELEA case, to show it does not transfer."""
    return _run('telea', set(JUDGEMENTS_TELEA), JUDGEMENTS_TELEA)


if __name__ == '__main__':
    rows = true_lama_low_band()
    clean_max = max(r for n, l, r in rows if l == 'clean')
    false_pass = [n for n, l, r in rows if l == 'ghost' and r <= clean_max]
    n_ghost = sum(1 for n, l, r in rows if l == 'ghost')
    print(f'\nclean cases all read <= {clean_max:.3f}; of {n_ghost} ghosts, '
          f'{len(false_pass)} would read as a false PASS at that cut: {false_pass}')
    telea_edge_ratios()
