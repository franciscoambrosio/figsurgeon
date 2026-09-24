"""Does `erase_object` leave the object's cast shadow behind, and does
`extend_mask_with_shadow` fix it without eating background that was never in shadow?

    flock /tmp/figsurgeon-model.lock .venv/bin/python evals/erase_shadows.py [out_dir] [telea|lama]

17 cases (8 with a real cast shadow, 9 without), cut from `evals/erase_field.py`'s corpus and
labelled by eye against the original photographs (`SHADOW_JUDGEMENTS` below). Per case:
whether `detect_shadow` finds the shadow, how much of the before/after darkness gap survives
erase with and without the hook (1.0 untouched, 0.0 fully filled), and how many extra frame
pixels the hook adds on no-shadow cases. Sheets are before | baseline-after | hook-after,
saved under `out_dir`. `fill='telea'` by default; `fill='lama'` is for final sheets only.

Headline: the detector found 1/8 true shadows and false-fired on 3/9 no-shadow cases (one
adding 354% of the object's own mask) -- not usable as shipped.
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import cv2                                                                # noqa: E402

from figsurgeon import objects as O                                     # noqa: E402
from scipy import ndimage

from figsurgeon.composite import _arr
from figsurgeon.photo import _to_hsv, _feather

# Tuned against both directions of the 17-case corpus below, not a single photograph.
RING_FRAC = 0.6        # how far out to search, as a fraction of the object's own bbox size
RING_MIN, RING_MAX = 20, 220
REF_FRAC = 0.4         # how much further out, beyond the ring, to sample "clean" background
REF_MIN, REF_MAX = 20, 160
HUE_TOL = 0.07          # max circular hue distance (of 0.5) from the reference to count
SAT_TOL = 0.16          # max |saturation - reference| to count
VALUE_DROP_MIN = 0.05   # must be at least this much darker (0..1 value) than the reference
VALUE_DROP_MAX = 0.55   # ... but not so much darker it's plausibly a different, solid object
MIN_COMPONENT_FRAC = 0.03   # drop candidate blobs smaller than this fraction of the object


def _hue_dist(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 1.0 - d)


def _circular_mean_hue(h):
    if h.size == 0:
        return 0.0
    ang = np.angle(np.mean(np.exp(2j * np.pi * h)))
    return float((ang / (2 * np.pi)) % 1.0)


def _ring_and_reference(binary):
    """The search ring (just outside the mask) and the reference annulus (further out
    still) that `detect_shadow` compares against. Split out so `evals/erase_shadows.py` can
    measure the same geometry independently of the hue/value test built on top of it."""
    ys, xs = np.nonzero(binary)
    if ys.size == 0:
        return np.zeros_like(binary), np.zeros_like(binary), 0, 0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    scale = max(y1 - y0, x1 - x0)
    ring_px = int(np.clip(scale * RING_FRAC, RING_MIN, RING_MAX))
    ref_px = int(np.clip(scale * REF_FRAC, REF_MIN, REF_MAX))
    struct = np.ones((3, 3), dtype=bool)
    search = ndimage.binary_dilation(binary, structure=struct, iterations=ring_px)
    ring = search & ~binary
    wider = ndimage.binary_dilation(search, structure=struct, iterations=ref_px)
    reference = wider & ~search
    return ring, reference, ring_px, ref_px


def detect_shadow(img, mask):
    """Find the cast-shadow pixels adjacent to a segmented object's mask.

    `img` is the original photograph; `mask` is the soft (0..1) mask `segment_object`
    returned for it. Returns `(shadow_binary, info)`: `shadow_binary` is True where a cast
    shadow was found (all False, not an error, when none was); `info` carries `found`,
    `shadow_px`, `object_px`, `ref_hsv`, or `reason` when detection couldn't run at all.
    """
    binary = np.asarray(mask) > 0.5
    H, W = binary.shape
    if not binary.any():
        return np.zeros_like(binary), {'found': False, 'reason': 'empty mask'}

    ring, reference, ring_px, ref_px = _ring_and_reference(binary)
    struct = np.ones((3, 3), dtype=bool)
    object_px = int(binary.sum())
    if not ring.any() or not reference.any():
        return np.zeros_like(binary), {'found': False, 'reason': 'no room to sample',
                                        'object_px': object_px}

    a = _arr(img)
    h, s, v = _to_hsv(a)

    ref_h = _circular_mean_hue(h[reference])
    ref_s = float(np.median(s[reference]))
    ref_v = float(np.median(v[reference]))

    darker = (v <= ref_v - VALUE_DROP_MIN) & (v >= ref_v - VALUE_DROP_MAX)
    same_hue = _hue_dist(h, ref_h) <= HUE_TOL
    same_sat = np.abs(s - ref_s) <= SAT_TOL
    candidate = ring & darker & same_hue & same_sat

    labels, n = ndimage.label(candidate, structure=struct)
    shadow = np.zeros_like(binary)
    min_size = max(1, int(object_px * MIN_COMPONENT_FRAC))
    if n > 0:
        touches_object = ndimage.binary_dilation(binary, structure=struct, iterations=1)
        for i in range(1, n + 1):
            comp = labels == i
            if comp.sum() < min_size:
                continue
            if not (comp & touches_object).any():
                continue
            shadow |= comp

    return shadow, {
        'found': bool(shadow.any()), 'shadow_px': int(shadow.sum()), 'object_px': object_px,
        'ref_hsv': (ref_h, ref_s, ref_v),
    }


def extend_mask_with_shadow(img, mask, feather=2):
    """`mask` unioned with any detected cast shadow, feathered the same way `segment_object`
    feathers its own edge. Returns `(extended_mask, info)` -- `info` is `detect_shadow`'s.

    When nothing is found, returns `mask` unchanged (not a copy-then-noop dance -- the
    caller can compare `info['found']` if it needs to know whether anything changed).
    """
    shadow, info = detect_shadow(img, mask)
    if not info.get('found'):
        return mask, info
    soft_shadow = _feather(shadow.astype(float), feather)
    extended = np.clip(np.maximum(np.asarray(mask, dtype=float), soft_shadow), 0, 1)
    return extended, info

# ---------------------------------------------------------------------------------------
# A detector kept here (not in the package) because this eval shows it doesn't work: it
# breaks on real shadows (which shift hue, e.g. bluer under skylight, not just darken) and
# false-fires on any smooth, slightly darker region (cloud, bokeh, wet tarmac).
# ---------------------------------------------------------------------------------------



# The measurement below calls the candidate as an imported module; `S` is simply this file.
import sys as _sys                                                        # noqa: E402
S = _sys.modules[__name__]

from figsurgeon import inpaint as I                                     # noqa: E402
import erase_field as EF                                                 # noqa: E402

SHEETS = os.path.join(HERE, 'out_erase_shadows')

# name -> (box under EF.CASES already thumbnails to WORKING=1200, same as erase_field.py)
_BY_NAME = {c[0]: c for c in EF.CASES}

CASE_NAMES = [
    # --- has a real cast shadow, confirmed by eye against evals/_corpus ---
    'motorcycle', 'striped_bollard', 'blue_chair', 'red_backpack', 'shepherd_dog',
    'cow_at_fence', 'studio_umbrella', 'white_shoe',
    # --- no meaningful cast shadow, confirmed by eye ---
    'kingfisher', 'sailboat_mast', 'night_tower', 'guitarist', 'cat_face', 'red_door',
    'street_person', 'roadworks_cone', 'red_shoes_pair',
]

# 'shadow': a real cast shadow, visible in the original photo, distinct from the object's
# own dark colouring. 'no_shadow': otherwise (flat light, no ground plane, or too tight a
# frame). Set by eye against evals/_corpus, not by name or by this script's numbers.
SHADOW_JUDGEMENTS = {
    'motorcycle':      ('shadow', 'long dark shadow under the whole bike on plain tarmac'),
    'striped_bollard': ('shadow', 'sharp diagonal shadow cast onto cobblestone, unmistakable'),
    'blue_chair':      ('shadow', 'soft shadow under the chair legs on the tiled floor'),
    'red_backpack':    ('shadow', 'soft studio contact shadow at the base, on the white sweep'),
    'shepherd_dog':    ('shadow', 'shadow along the dog\'s back/flank on the grass -- busy bg'),
    'cow_at_fence':    ('shadow', 'shadow under the head and belly on the grass -- busy bg'),
    'studio_umbrella': ('shadow', 'soft shadow of the umbrella on the grey studio backdrop'),
    'white_shoe':      ('shadow', 'small contact shadow under the sole, on the carpet'),

    'kingfisher':      ('no_shadow', 'only a tiny shadow of the FEET on the branch itself; '
                                      'the frame is dominated by bokeh with nothing to '
                                      'shadow onto'),
    'sailboat_mast':   ('no_shadow', 'water and sky, no ground plane at all'),
    'night_tower':     ('no_shadow', 'silhouetted against open sky'),
    'guitarist':       ('no_shadow', 'dark stage backdrop; the only shadow-like shape is the '
                                      'guitar cable\'s own cast shadow on his shirt, which is '
                                      'not a ground shadow of the guitarist'),
    'cat_face':        ('no_shadow', 'macro fill of fur and face; no ground visible'),
    'red_door':        ('no_shadow', 'flat facade, no ground plane in frame'),
    'street_person':   ('no_shadow', 'overcast light, under a shop awning; no visible cast '
                                      'shadow'),
    'roadworks_cone':  ('no_shadow', 'flat overcast lighting, no shadow on the forecourt'),
    'red_shoes_pair':  ('no_shadow', 'black background and a densely printed paper surface '
                                      'with no discernible cast shadow'),
}


def _inpaint(a, binary_mask, fill, radius=6):
    """The dilate + fill half of `objects.erase_object`, reimplemented here so this script
    can hand it an ARBITRARY mask (the extended one) rather than only the one
    `segment_object` finds. Deliberately mirrors `erase_object`'s own grow=3 default."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3 * 2 + 1, 3 * 2 + 1))
    grown = cv2.dilate(binary_mask, kernel)
    if fill == 'lama':
        out = I.lama(a, grown)
    else:
        out = cv2.inpaint(a, grown, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    return out


def _persistence(before_a, after_a, ring, reference):
    """How much of the ring/reference VALUE gap survives from `before_a` to `after_a`: 1.0
    untouched, 0.0 fully filled to background, negative if overshot. `None` when there was
    no meaningful gap to begin with.
    """
    from figsurgeon.photo import _to_hsv
    _, _, v_before = _to_hsv(before_a)
    _, _, v_after = _to_hsv(after_a)
    ref_v = float(np.median(v_before[reference]))
    ring_v_before = float(np.median(v_before[ring]))
    ring_v_after = float(np.median(v_after[ring]))
    gap_before = ref_v - ring_v_before
    if gap_before <= 0.02:            # no real darkness there to begin with
        return None, gap_before
    gap_after = ref_v - ring_v_after
    return gap_after / gap_before, gap_before


def run(out_dir=SHEETS, fill='telea'):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for name in CASE_NAMES:
        _, rel, box, subject, category, note = _BY_NAME[name]
        before = EF.load_working(rel)
        a = np.array(before.convert('RGB'))

        mask, coverage = O.segment_object(before, box=box, backend='auto', follow_object=True)
        binary = (mask > 0.5)
        binary_u8 = binary.astype(np.uint8) * 255

        shadow, info = S.detect_shadow(before, mask)
        extended, _ = S.extend_mask_with_shadow(before, mask)
        extended_u8 = (extended > 0.5).astype(np.uint8) * 255

        baseline_out = Image.fromarray(_inpaint(a, binary_u8, fill))
        hook_out = Image.fromarray(_inpaint(a, extended_u8, fill))

        ring, reference, ring_px, ref_px = S._ring_and_reference(binary)
        baseline_a = np.array(baseline_out)
        hook_a = np.array(hook_out)
        persist_base, gap = _persistence(a.astype(float), baseline_a.astype(float), ring, reference)
        persist_hook, _ = _persistence(a.astype(float), hook_a.astype(float), ring, reference)

        added_px = int(((extended > 0.5) & ~binary).sum())
        added_frac_of_object = added_px / max(1, int(binary.sum()))

        # Sheet: before | baseline-after | hook-after, cropped with generous margin.
        x0, y0, x1, y1 = box
        mx, my = int((x1 - x0) * 0.4) + 30, int((y1 - y0) * 0.4) + 30
        W, H = before.size
        cx0, cy0 = max(0, x0 - mx), max(0, y0 - my)
        cx1, cy1 = min(W, x1 + mx), min(H, y1 + my)
        crops = [im.crop((cx0, cy0, cx1, cy1)) for im in (before, baseline_out, hook_out)]
        cw, ch = crops[0].size
        sheet = Image.new('RGB', (cw * 3 + 20, ch), (255, 0, 0))
        for i, c in enumerate(crops):
            sheet.paste(c, (i * (cw + 10), 0))
        sheet.save(os.path.join(out_dir, f'{name}_before_baseline_hook.jpg'), quality=92)

        label, why = SHADOW_JUDGEMENTS.get(name, ('?', ''))
        row = {
            'name': name, 'label': label, 'found': info.get('found', False),
            'shadow_px': info.get('shadow_px', 0), 'object_px': info.get('object_px', 0),
            'added_frac_of_object': added_frac_of_object,
            'persist_baseline': persist_base, 'persist_hook': persist_hook, 'gap': gap,
        }
        rows.append(row)

        def fmt(x):
            return 'n/a ' if x is None else f'{x:+.2f}'
        print(f'{name:18s} label={label:9s} found={"Y" if row["found"] else "n"}  '
              f'gap={gap:5.2f}  persist base={fmt(persist_base)} hook={fmt(persist_hook)}  '
              f'added={added_frac_of_object:5.1%} of object  {why}')

    print(f'\n{len(rows)} cases; sheets in {out_dir}')
    tally(rows)
    return rows


def tally(rows):
    shadow_rows = [r for r in rows if r['label'] == 'shadow']
    no_shadow_rows = [r for r in rows if r['label'] == 'no_shadow']

    found_on_shadow = sum(1 for r in shadow_rows if r['found'])
    fired_on_no_shadow = [r['name'] for r in no_shadow_rows if r['found']]

    print(f'\nDETECTION: found a shadow on {found_on_shadow}/{len(shadow_rows)} true-shadow '
          f'cases; fired on {len(fired_on_no_shadow)}/{len(no_shadow_rows)} no-shadow cases'
          + (f' -- {fired_on_no_shadow}' if fired_on_no_shadow else ''))

    scored = [r for r in shadow_rows if r['persist_baseline'] is not None]
    if scored:
        base_med = np.median([r['persist_baseline'] for r in scored])
        hook_med = np.median([r['persist_hook'] for r in scored])
        print(f'\nPERSISTENCE on the {len(scored)} shadow cases with a measurable gap '
              f'(median darkness remaining after erase, 1.0=untouched, 0.0=fully filled):')
        print(f'  baseline (no hook): median {base_med:+.2f}')
        print(f'  with shadow hook:   median {hook_med:+.2f}')
        for r in scored:
            print(f'    {r["name"]:18s} base={r["persist_baseline"]:+.2f}  '
                  f'hook={r["persist_hook"]:+.2f}')

    if no_shadow_rows:
        added = [r['added_frac_of_object'] for r in no_shadow_rows]
        print(f'\nFALSE-POSITIVE COST on the {len(no_shadow_rows)} no-shadow cases: pixels '
              f'added beyond the object mask, as a fraction of the object\'s own mask size '
              f'-- max {max(added):.1%}, median {np.median(added):.1%}')


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else SHEETS,
        sys.argv[2] if len(sys.argv) > 2 else 'telea')
