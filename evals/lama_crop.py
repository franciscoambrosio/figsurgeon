"""Measurement: does feeding LaMa a padded crop (or a downscaled frame) instead of the whole
frame stop it from regenerating large erased objects?

    flock /tmp/figsurgeon-model.lock OMP_NUM_THREADS=3 \\
        .venv/bin/python evals/lama_crop.py [out_dir] [subset|full]

`erase_field.JUDGEMENTS_LAMA` found whole-frame LaMa routinely regenerates the erased object
(18 ghost of 24). Each variant calls the public `inpaint.lama` on a smaller array and pastes
the result back, so `inpaint.py` is untouched:

  'baseline'   -- whole frame, whole mask.
  'crop_pX'    -- crop to the mask's bounding box padded by `X` * the box's size per side.
  'scale_X'    -- downscale frame and mask to `X`, fill, upsample and composite.

`SUBSET` is the four clean cases plus the worst regenerations; `full` runs every case in
`erase_field.CASES`. Sheets are saved as `<name>_<variant>.jpg` (before | after), and
`JUDGEMENTS` below holds the labels set by looking at each at 1:1.
"""
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from figsurgeon import objects as O                                     # noqa: E402
from figsurgeon import inpaint as I                                     # noqa: E402
from erase_field import CASES, load_working                              # noqa: E402

SHEETS = os.path.join(HERE, 'out_lama_crop')

SUBSET = [
    'roadworks_cone', 'car_red_street', 'street_person', 'red_door',      # must stay clean
    'red_backpack', 'studio_umbrella', 'kingfisher', 'white_horse',       # named regenerations
    'motorcycle', 'cow_at_fence', 'sailboat_mast', 'taxi_foreground',     # more spread
]


def _segmented_binary(img, box):
    """Same segment+dilate `erase_object` does, exposed so every variant fills the SAME mask."""
    mask, coverage = O.segment_object(img, box, iterations=3, feather=2,
                                      backend='auto', follow_object=True)
    binary = (mask > 0.5).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3 * 2 + 1, 3 * 2 + 1))
    binary = cv2.dilate(binary, kernel)
    frame_frac = float((binary > 0).mean())
    return binary, frame_frac


def fill_baseline(rgb, binary):
    return I.lama(rgb, binary)


def fill_crop(rgb, binary, pad):
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return rgb.copy()
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    bw, bh = x1 - x0, y1 - y0
    mx, my = int(bw * pad), int(bh * pad)
    H, W = rgb.shape[:2]
    cx0, cy0 = max(0, x0 - mx), max(0, y0 - my)
    cx1, cy1 = min(W, x1 + mx), min(H, y1 + my)
    rgb_crop = rgb[cy0:cy1, cx0:cx1].copy()
    mask_crop = binary[cy0:cy1, cx0:cx1]
    filled_crop = I.lama(rgb_crop, mask_crop)
    result = rgb.copy()
    result[cy0:cy1, cx0:cx1] = filled_crop
    return result


def fill_scale(rgb, binary, scale):
    H, W = rgb.shape[:2]
    nh, nw = max(8, int(round(H * scale))), max(8, int(round(W * scale)))
    small_rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(binary, (nw, nh), interpolation=cv2.INTER_NEAREST)
    filled_small = I.lama(small_rgb, small_mask)
    filled_up = cv2.resize(filled_small, (W, H), interpolation=cv2.INTER_LINEAR)
    result = rgb.copy()
    m = binary > 0
    result[m] = filled_up[m]
    return result


VARIANTS = {
    'baseline': lambda rgb, binary: fill_baseline(rgb, binary),
    'crop_p0.3': lambda rgb, binary: fill_crop(rgb, binary, 0.3),
    'crop_p0.8': lambda rgb, binary: fill_crop(rgb, binary, 0.8),
    'scale_0.5': lambda rgb, binary: fill_scale(rgb, binary, 0.5),
    # Large boxes clip to the frame at pad 0.3/0.8, so only this one tests a truly tight crop.
    'crop_p0.05': lambda rgb, binary: fill_crop(rgb, binary, 0.05),
}


def save_sheet(before, after, box, out_dir, name, variant):
    x0, y0, x1, y1 = box
    mx, my = int((x1 - x0) * 0.4) + 20, int((y1 - y0) * 0.4) + 20
    W, H = before.size
    cx0, cy0 = max(0, x0 - mx), max(0, y0 - my)
    cx1, cy1 = min(W, x1 + mx), min(H, y1 + my)
    crop_before = before.crop((cx0, cy0, cx1, cy1))
    crop_after = after.crop((cx0, cy0, cx1, cy1))
    cw, ch = crop_before.size
    compare = Image.new('RGB', (cw * 2 + 10, ch), (255, 0, 0))
    compare.paste(crop_before, (0, 0))
    compare.paste(crop_after, (cw + 10, 0))
    path = os.path.join(out_dir, f'{name}_{variant}.jpg')
    compare.save(path, quality=92)
    return path


def run(out_dir=SHEETS, which='subset', variants=None):
    os.makedirs(out_dir, exist_ok=True)
    variants = variants or list(VARIANTS)
    cases = CASES if which == 'full' else [c for c in CASES if c[0] in SUBSET]
    rows = []
    for name, rel, box, subject, category, note in cases:
        before = load_working(rel)
        rgb = np.array(before.convert('RGB'))
        try:
            binary, frame_frac = _segmented_binary(before, box)
        except Exception as e:
            print(f'{name:22s} SEGMENT FAILED  {type(e).__name__}: {e}')
            continue
        for variant in variants:
            t0 = time.time()
            try:
                out = VARIANTS[variant](rgb, binary)
            except Exception as e:
                print(f'{name:22s} {variant:12s} FILL FAILED  {type(e).__name__}: {e}')
                continue
            dt = time.time() - t0
            after = Image.fromarray(out)
            # Outside-mask bytes must match the input exactly, for every variant.
            m = binary > 0
            outside_ok = np.array_equal(rgb[~m], out[~m])
            path = save_sheet(before, after, box, out_dir, name, variant)
            rows.append({'name': name, 'variant': variant, 'frame_frac': frame_frac,
                        'seconds': dt, 'outside_ok': outside_ok, 'sheet': path})
            flag = '' if outside_ok else '  COMPOSITE BROKEN'
            print(f'{name:22s} {variant:12s} frame_frac={frame_frac:5.1%}  '
                  f'{dt:5.1f}s{flag}')
    print(f'\n{len(rows)} (case, variant) sheets written to {out_dir}')
    return rows


# Labels by eye at 1:1, same definitions as `erase_field.JUDGEMENTS`.
#
# Result: no crop padding (0.05, 0.3, 0.8) changed a single label from baseline, on the
# 12-case subset or the full 24 -- the regenerations are not caused by whole-frame context.
# Only crop_p0.05 is recorded; the other paddings give the same table, checked by eye.
JUDGEMENTS = {
    'crop_p0.05': {
        'roadworks_cone':       ('clean', 'unchanged from baseline'),
        'car_red_street':       ('clean', 'unchanged from baseline'),
        'street_person':        ('clean', 'unchanged from baseline'),
        'red_door':              ('clean', 'unchanged from baseline'),

        'shepherd_dog':          ('faint', 'unchanged from baseline -- dog-shaped darker patch survives'),
        'blue_chair':            ('faint', 'unchanged from baseline -- ghost-transparent chair/table legs'),

        'taxi_foreground':       ('ghost', 'unchanged from baseline'),
        'sailboat_mast':         ('ghost', 'unchanged from baseline'),
        'night_tower':           ('ghost', 'unchanged from baseline -- spire still regenerated'),
        'striped_bollard':       ('ghost', 'unchanged from baseline -- dark stub still left'),
        'kingfisher':            ('ghost', 'unchanged from baseline -- still a translucent green bird'),
        'white_horse':           ('ghost', 'unchanged from baseline -- head/neck still an invented hedge'),
        'cow_at_fence':          ('ghost', 'unchanged from baseline -- translucent body-tent still there'),
        'cat_face':              ('ghost', 'unchanged from baseline'),
        'motorcycle':            ('ghost', 'unchanged from baseline -- mirror still floats, shadow stays'),
        'red_backpack':          ('ghost', 'unchanged from baseline -- still a faded backpack, handles and all'),
        'white_shoe':            ('ghost', 'unchanged from baseline -- still a ghost-white shoe'),
        'red_shoes_pair':        ('ghost', 'unchanged from baseline -- dark red shoe-shaped mass remains'),
        'studio_umbrella':       ('ghost', 'unchanged from baseline -- umbrella still redrawn, cleaner than before'),
        'guitarist':             ('ghost', 'unchanged from baseline -- translucent figure-shaped slab'),
        'flower_petal':          ('ghost', 'unchanged from baseline -- petal still fully there'),
        'backpack_straps_only':  ('ghost', 'unchanged from baseline'),
        'dog_head_only':         ('ghost', 'unchanged from baseline'),
        'second_car_sliver':     ('ghost', 'unchanged from baseline'),
    },
    # Subset only. Same non-result, plus a regression on `red_door`: downscaling blends the
    # door's red into what LaMa sees as background, and the fill inherits the tint.
    'scale_0.5': {
        'roadworks_cone':       ('clean', 'unchanged from baseline'),
        'car_red_street':       ('clean', 'unchanged from baseline'),
        'street_person':        ('clean', 'unchanged from baseline'),
        'red_door':              ('faint', 'REGRESSION from clean: a pink/red colour cast '
                                  'bleeds across the fill from downscaling the door alongside '
                                  'its own mask before the model runs'),
        'red_backpack':          ('ghost', 'unchanged from baseline, but more plaid/glitchy '
                                  'texture than a recognisable backpack silhouette'),
        'studio_umbrella':       ('ghost', 'unchanged from baseline'),
        'kingfisher':            ('ghost', 'unchanged from baseline, marginally softer/more '
                                  'washed out but still a clear translucent bird'),
        'white_horse':           ('ghost', 'unchanged from baseline -- hedge still invented, '
                                  'slightly blurrier'),
        'motorcycle':            ('ghost', 'unchanged from baseline'),
        'cow_at_fence':          ('ghost', 'unchanged from baseline'),
        'sailboat_mast':         ('ghost', 'unchanged from baseline'),
        'taxi_foreground':       ('ghost', 'unchanged from baseline'),
    },
}


if __name__ == '__main__':
    out_dir = sys.argv[1] if len(sys.argv) > 1 else SHEETS
    which = sys.argv[2] if len(sys.argv) > 2 else 'subset'
    run(out_dir, which)
