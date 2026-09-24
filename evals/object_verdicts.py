"""isolate_object and extract_object had no way to say WHICH thing they acted on.

    .venv/bin/python evals/object_verdicts.py [out_dir]

Three mask variants per case: full (shipped mask, must PASS), half (mask cut in two, must
FAIL), inverse (the BACKGROUND inside the box kept, must FAIL). Inverse is what a segmenter
that took the sky instead of the rocket produces, and it passed the old checks (colour kept
inside the box, lost outside) with full marks.

Both tools are now graded against the region the caller's `subject` words resolve to,
narrowed to pixels that HAD colour (a colour-pop can't keep colour where there was none) --
the same evidence that fixed the recolour verdict (`object_completeness.py`).
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from figsurgeon import locate                                           # noqa: E402
from figsurgeon import objects as O                                     # noqa: E402
from figsurgeon import verify_photo as V                                # noqa: E402

CORPUS = os.path.join(HERE, '_corpus')
SHEETS = os.path.join(HERE, 'out_object_verdicts')
WORKING = 1200

# (case, image under _corpus/, box, the caller's words)
CASES = [
    ('sheep_mid', 'audit_phrase/sheep_fence.jpg', (420, 245, 560, 355), 'the middle sheep'),
    ('saddle_front', 'instances/bicycles.jpg', (0, 505, 460, 800), 'the saddle at the front'),
    ('person_left', 'street_people.jpg', (390, 370, 500, 720), 'the person on the left'),
    ('person_mid', 'street_people.jpg', (505, 395, 625, 730), 'the man in the middle'),
    ('chair_right', 'eight_boxes/chairs_blue.jpg', (705, 430, 975, 700), 'the chair on the right'),
    ('backpack', 'eight_boxes/backpack_red.jpg', (110, 25, 880, 1160), 'the backpack'),
    ('dog', 'eight_boxes/dog_shepherd.jpg', (150, 30, 1060, 740), 'the dog'),
    ('kingfisher', 'eight_boxes/kingfisher.jpg', (480, 130, 1010, 600), 'the bird'),
    ('car_red', 'car_red.jpg', None, 'the red car'),
    ('pizza', 'food_pizza.jpg', None, 'the pizza'),
]
# Boxes for the two `real_corpus` photographs, read off the working-size image like the rest.
EXTRA_BOXES = {'car_red': (515, 790, 795, 895), 'pizza': (240, 120, 1120, 900)}
# A box on the shopfront window two floors above the car, instead of the car itself, is what
# the wrong-box branch exists to catch: wrong box, or words that don't describe the edit.

# What each mask was judged to be by eye (SlimSAM, all ten). Every one is the named object,
# which makes `full` rows must-PASS and `inverse` rows a fair test: the inverse of a correct
# mask is exactly the background kept instead of the object.
LOOKED_AT = {
    'sheep_mid': 'that sheep, head down; neither neighbour touched',
    'saddle_front': 'the near saddle, down to the clamp under its nose',
    'person_left': 'the woman, bag strap and papers included, hair to shoes',
    'person_mid': 'the man in black, cap to shoes, the folder included',
    'chair_right': 'the right chair, white frame and blue seat, arms and legs',
    'backpack': 'the bag, both handles, the background between them resolved',
    'dog': 'the dog, tail to front paws',
    'kingfisher': 'the bird, open beak and both wings; the branch excluded',
    'car_red': 'the whole car, wheels and windows; the shopfront behind it excluded',
    'pizza': 'the pizza itself, basil leaves included; the plate rim excluded',
}


def image(rel):
    img = Image.open(os.path.join(CORPUS, rel)).convert('RGB')
    img.thumbnail((WORKING, WORKING), Image.LANCZOS)
    return img


def isolate(img, mask, flatten=0.5):
    """`objects.isolate_object`'s maths, driven by a mask given from outside."""
    a = np.array(img).astype(float)
    lum = a.mean(axis=2, keepdims=True)
    grey = np.repeat(lum, 3, axis=2) * (1 - flatten) + 128.0 * flatten
    m = np.asarray(mask, dtype=float)[:, :, None]
    return Image.fromarray(np.clip(a * m + grey * (1 - m), 0, 255).astype(np.uint8))


def cutout(img, mask):
    """`objects.extract_object`'s maths, driven by a mask given from outside."""
    a = np.array(img).astype(np.uint8)
    alpha = np.clip(np.asarray(mask, float) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([a, alpha]))


def half(mask):
    m = np.asarray(mask) > 0.5
    ys, xs = np.nonzero(m)
    out = m.copy()
    if np.ptp(ys) >= np.ptp(xs):
        out[(ys.min() + ys.max()) // 2:, :] = False
    else:
        out[:, (xs.min() + xs.max()) // 2:] = False
    return out.astype(float)


def inverse(mask, box, shape):
    """The background inside the box -- what a segmenter that took the sky produces."""
    x0, y0, x1, y1 = box
    win = np.zeros(shape, bool)
    win[y0:y1, x0:x1] = True
    return (win & (np.asarray(mask) < 0.5)).astype(float)


def run(out_dir=SHEETS):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for case, rel, box, subject in CASES:
        img = image(rel)
        box = box or EXTRA_BOXES[case]
        mask, _cov = O.segment_object(img, box)
        variants = {'full': np.asarray(mask, dtype=float),
                    'half': half(mask),
                    'inverse': inverse(mask, box, np.asarray(mask).shape)}
        x0, y0, x1, y1 = box
        pad = 40
        crop = (max(0, x0 - pad), max(0, y0 - pad),
                min(img.size[0], x1 + pad), min(img.size[1], y1 + pad))
        locate.mask_overlay(img, mask).crop(crop).save(
            os.path.join(out_dir, f'{case}_mask.jpg'), quality=85)
        row = {'case': case, 'subject': subject}
        for label, m in variants.items():
            popped = isolate(img, m)
            popped.crop(crop).save(os.path.join(out_dir, f'{case}_{label}.jpg'), quality=85)
            row['pop_' + label] = (V.check_isolate_object(img, popped, [list(box)]),
                                   V.check_isolate_object(img, popped, [list(box)],
                                                          subject=subject))
            cut = cutout(img, m)
            cut.crop(crop).save(os.path.join(out_dir, f'{case}_{label}_cutout.png'))
            row['cut_' + label] = (V.check_object_extracted(img, cut, box),
                                   V.check_object_extracted(img, cut, box, subject=subject))
        rows.append(row)
        print(f'{case:13s} colour-pop ' + ' '.join(
            f'{label}: {_v(row["pop_" + label][0][0])}->{_v(row["pop_" + label][1][0])}'
            for label in ('full', 'half', 'inverse')) + '  | cutout ' + ' '.join(
            f'{label}: {_v(row["cut_" + label][0][0])}->{_v(row["cut_" + label][1][0])}'
            for label in ('full', 'half', 'inverse')))
    summarise(rows)
    print(f'\nsheets in {out_dir} -- LOOK at them; the numbers cannot tell you whether the '
          f'thing that survived is the object.')
    return rows


def _v(ok):
    return {True: 'PASS', False: 'FAIL', None: 'none'}[ok]


def summarise(rows):
    fmt = lambda v: f'{v.count(True)}/{v.count(None)}/{v.count(False)}'         # noqa: E731
    for op, prefix in (('isolate_object', 'pop_'), ('extract_object', 'cut_')):
        print(f'\n{op:16s} {"want":5s} {"blind: P/n/F":>16s} {"with subject: P/n/F":>22s}')
        for label, want in (('full', 'PASS'), ('half', 'FAIL'), ('inverse', 'FAIL')):
            key = prefix + label
            print(f'{label:16s} {want:5s} '
                  f'{fmt([r[key][0][0] for r in rows]):>16s} '
                  f'{fmt([r[key][1][0] for r in rows]):>22s}')
    print()
    for r in rows:
        for prefix in ('pop_', 'cut_'):
            for label in ('half', 'inverse'):
                if r[prefix + label][1][0] is True:
                    print(f'WRONG WORK VERIFIED: {r["case"]} {prefix}{label} -- '
                          f'{r[prefix + label][1][1]}')
            if r[prefix + 'full'][1][0] is False:
                print(f'CORRECT WORK FAILED: {r["case"]} {prefix}full -- '
                      f'{r[prefix + "full"][1][1]}')


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else SHEETS)
