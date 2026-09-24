"""Occlusion: does the named object's completeness check survive something in front of it?

    .venv/bin/python evals/occlusion_phrases.py [out_dir]

The named object is real, whole, and correctly recoloured, but something else -- a wire
fence, a bollard, a post, a neighbouring bicycle -- sits between it and the camera and
covers part of it. Two failure modes: (1) the mask swallows the occluder along with the
object; (2) the completeness check is confused by the gap and reads a genuinely occluded
object as incomplete.

Ten cases on real photographs (sheep behind a wire fence, a cow behind barbed wire, bikes,
a man behind a post, cars behind bollards) -- this eval needs its own copies of those
images; they are not in the repo. Both directions, as elsewhere in this repo:

  full   the shipped mask, wholly recoloured  -- must PASS
  half   the same mask cut in two             -- must FAIL

Every mask is looked at before its row is kept (`LOOKED_AT` below): with an occluder in the
box, a segmenter can also pick the right object AND the occluder, or lose the object behind it.
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from figsurgeon import objects as O                                     # noqa: E402
from figsurgeon import locate                                           # noqa: E402
from figsurgeon import verify_photo as V                                # noqa: E402

CORPUS = os.path.join(HERE, '_corpus')
SHEETS = os.path.join(HERE, 'out_occlusion_phrases')
WORKING = 1200
TARGET = (40, 90, 200)

# (case, path under _corpus/, box -- read off the original image at its native resolution,
# the caller's words, what occludes it)
CASES = [
    ('sheep_left', 'audit_phrase/sheep_fence.jpg', (190, 245, 295, 350),
     'the sheep', 'wire fence (3 strands + post nearby)'),
    ('sheep_mid', 'audit_phrase/sheep_fence.jpg', (420, 245, 560, 355),
     'the sheep', 'wire fence (3 strands cross the body)'),
    ('sheep_right', 'audit_phrase/sheep_fence.jpg', (680, 245, 795, 345),
     'the sheep', 'wire fence (2 strands cross the legs)'),
    ('cow_fence', 'occlusion/cow_fence.jpg', (300, 0, 1180, 780),
     'the cow', 'barbed wire fence + post, 4 strands cross the body'),
    ('green_bike', 'occlusion/bikes_leiden2.jpg', (490, 335, 665, 525),
     'the green bicycle', 'a black bicycle frame in front of its lower half'),
    ('man_post', 'street_people.jpg', (2200, 1140, 2540, 2561),
     'the man', 'a black metal post crossing his torso and leg'),
    ('red_car_bollard1', 'occlusion/sainsburys_bollards.jpg', (0, 240, 215, 425),
     'the car', 'a black-and-yellow bollard in front of the wheel'),
    ('silver_car_bollard1', 'occlusion/sainsburys_bollards.jpg', (555, 325, 640, 465),
     'the car', 'a black-and-yellow bollard in front of the wheel'),
    ('red_car_bollard2', 'occlusion/tesco_bollards.jpg', (325, 305, 427, 400),
     'the car', 'a black bollard in front of the bumper'),
    ('black_car_bollard2', 'occlusion/tesco_bollards.jpg', (325, 385, 427, 470),
     'the car', 'a black bollard in front of the wheel'),
]

# What each mask was judged to be by eye (SlimSAM, all ten). Cases marked EXCLUDED are not
# the named object and are kept in the table but dropped from must-PASS scoring.
LOOKED_AT = {
    'sheep_left': 'that sheep, all four legs; the wire fence in front not taken',
    'sheep_mid': 'that sheep, head down to the grass; the wire fence in front not taken',
    'sheep_right': 'that sheep only; wire fence in front not taken',
    'cow_fence': 'the cow, head to hooves; barbed wire and post excluded (a few stray '
                 'green flecks along the far strand, negligible area)',
    'green_bike': 'EXCLUDED -- the mask is NOT the green bicycle: it swallowed the black '
                  'bike behind it, a red basket, and part of a third bike -- a crowded-rack '
                  'occlusion failure, not a clean case',
    'man_post': 'the man, cap to sandals, phone included; the black post beside his leg '
                'excluded',
    'red_car_bollard1': 'EXCLUDED -- the box (drawn too wide) put two cars inside it, and '
                        'the mask is BOTH the silver car and the red car, not "the car"',
    'silver_car_bollard1': 'the silver car\'s front bumper and wheel arch only; the '
                           'bollard excluded',
    'red_car_bollard2': 'the red car; the bollard in front of the bumper excluded',
    'black_car_bollard2': 'EXCLUDED -- coverage 0.04: the mask found almost nothing, a '
                          'sliver of bollard highlight, not the car',
}


def image_and_scale(rel):
    """Load a corpus image, thumbnailed like every other eval here -- and report the scale
    so a box read off the original (native-resolution) picture can be converted to match.
    `Image.thumbnail` never enlarges, so images already under WORKING keep scale 1.0.
    """
    orig = Image.open(os.path.join(CORPUS, rel)).convert('RGB')
    img = orig.copy()
    img.thumbnail((WORKING, WORKING), Image.LANCZOS)
    return img, img.size[0] / orig.size[0]


def recolour(img, mask, to_rgb=TARGET):
    """`objects.recolour_object`'s maths, driven by a mask given from outside."""
    a = np.array(img).astype(float)
    target = np.array(to_rgb, dtype=float)
    lum = a.mean(axis=2) / 255.0
    scale = np.clip(lum / max(target.mean() / 255.0, 1e-6), 0.35, 1.9)
    hit = np.clip(target[None, None, :] * scale[:, :, None], 0, 255)
    m = mask[:, :, None]
    return Image.fromarray((a * (1 - m) + hit * m).astype(np.uint8))


def half(mask):
    """One half of the object, split across its longer axis."""
    m = mask > 0.5
    ys, xs = np.nonzero(m)
    out = m.copy()
    if np.ptp(ys) >= np.ptp(xs):
        out[(ys.min() + ys.max()) // 2:, :] = False
    else:
        out[:, (xs.min() + xs.max()) // 2:] = False
    return out.astype(float)


def verdict(before, after, box, subject):
    ok, note = V.check_object_recoloured(before, after, box, subject=subject)
    return {True: 'PASS', False: 'FAIL', None: 'none'}[ok], note


def run(out_dir=SHEETS):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for case, rel, box0, subject, occluder in CASES:
        img, scale = image_and_scale(rel)
        box = tuple(round(v * scale) for v in box0)
        mask, cov = O.segment_object(img, box)
        full = recolour(img, mask)
        halved = recolour(img, half(mask))
        x0, y0, x1, y1 = box
        pad = 40
        crop = (max(0, x0 - pad), max(0, y0 - pad),
                min(img.size[0], x1 + pad), min(img.size[1], y1 + pad))
        locate.mask_overlay(img, mask).crop(crop).save(
            os.path.join(out_dir, f'{case}_mask.jpg'), quality=85)
        full.crop(crop).save(os.path.join(out_dir, f'{case}_full.jpg'), quality=85)
        row = {'case': case, 'subject': subject, 'occluder': occluder, 'coverage': cov}
        row['full'] = verdict(img, full, box, subject)
        row['half'] = verdict(img, halved, box, subject)
        rows.append(row)
        print(f'{case:20s} cov={cov:4.2f}  full: {row["full"][0]:4s}   half: '
              f'{row["half"][0]:4s}   occluder: {occluder}')
    print(f'\nsheets in {out_dir} -- LOOK at every *_mask.jpg before trusting a row, then '
          f'fill LOOKED_AT and re-run to see it echoed above.')
    summarise(rows)
    return rows


def summarise(rows):
    n = len(rows)
    scored = [r for r in rows if not LOOKED_AT.get(r['case'], '').startswith('EXCLUDED')]
    print(f'\n{n} cases, {len(scored)} scored (excluding wrong-mask rows)')
    full_pass = sum(1 for r in scored if r['full'][0] == 'PASS')
    half_fail = sum(1 for r in scored if r['half'][0] == 'FAIL')
    print(f'full: {full_pass}/{len(scored)} PASS (want all)')
    print(f'half: {half_fail}/{len(scored)} FAIL (want all)')
    for r in scored:
        if r['full'][0] not in ('PASS', 'none'):
            print(f'\nCORRECT WORK FAILED: {r["case"]}: {r["full"][1]}')
        if r['half'][0] == 'PASS':
            print(f'\nWRONG WORK VERIFIED: {r["case"]}: {r["half"][1]}')


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else SHEETS)
