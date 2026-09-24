"""One of several identical things: does the positional phrase still grade the edit?

    .venv/bin/python evals/instance_phrases.py [out_dir]

With no object detection, the caller draws a box and describes it positionally ("the middle
sheep") since nothing else distinguishes identical instances. The check resolves the phrase
on the box's own crop, not the frame, so a positional qualifier can lose its referent: a crop
holding one sheep makes "on the left" only readable as the left part of it.

  full   the shipped mask, wholly recoloured   -- must PASS (or at worst abstain)
  half   the same mask cut in two              -- must FAIL

Each edit is run once with the positional phrase, once with the bare noun; masks are looked
at before a row is kept.
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
SHEETS = os.path.join(HERE, 'out_instance_phrases')
WORKING = 1200
TARGET = (40, 90, 200)

# (case, path under _corpus/, box, the caller's positional words, the bare noun)
CASES = [
    ('sheep_left',   'audit_phrase/sheep_fence.jpg', (190, 245, 295, 350),
     'the sheep on the left', 'the sheep'),
    ('sheep_mid',    'audit_phrase/sheep_fence.jpg', (420, 245, 560, 355),
     'the middle sheep', 'the sheep'),
    ('sheep_right',  'audit_phrase/sheep_fence.jpg', (680, 245, 795, 345),
     'the sheep on the right', 'the sheep'),
    ('saddle_front', 'instances/bicycles.jpg', (0, 505, 460, 800),
     'the saddle at the front', 'the saddle'),
    ('saddle_2nd',   'instances/bicycles.jpg', (180, 225, 500, 430),
     'the second saddle from the front', 'the saddle'),
    ('saddle_3rd',   'instances/bicycles.jpg', (350, 110, 545, 215),
     'the third saddle from the front', 'the saddle'),
    ('person_left',  'street_people.jpg', (390, 370, 500, 720),
     'the person on the left', 'the person'),
    ('person_mid',   'street_people.jpg', (505, 395, 625, 730),
     'the man in the middle', 'the man'),
    ('person_right', 'street_people.jpg', (660, 350, 790, 790),
     'the man on the right', 'the man'),
    ('chair_left',   'eight_boxes/chairs_blue.jpg', (380, 455, 640, 700),
     'the chair on the left', 'the chair'),
    ('chair_right',  'eight_boxes/chairs_blue.jpg', (705, 430, 975, 700),
     'the chair on the right', 'the chair'),
]

# What each mask was judged to be by eye (SlimSAM, all eleven). Every one is the named
# thing, which is what makes the `full` rows usable as must-PASS cases.
LOOKED_AT = {
    'sheep_left': 'that sheep, all four legs; the wire in front of it not taken',
    'sheep_mid': 'that sheep, head down to the grass; neither neighbour touched',
    'sheep_right': 'that sheep only',
    'saddle_front': 'the near saddle, down to the clamp under its nose',
    'saddle_2nd': 'the second saddle; the frame below it excluded',
    'saddle_3rd': 'the third saddle, blurred -- the mask follows it anyway',
    'person_left': 'the woman, bag strap and papers included, from hair to shoes',
    'person_mid': 'the man in black, cap to shoes, the folder he carries included',
    'person_right': 'the man leaning on the wall, phone and sandals included',
    'chair_left': 'the left chair, frame and blue seat; the table leg behind it excluded',
    'chair_right': 'the right chair, arms and legs',
}


def image(rel):
    img = Image.open(os.path.join(CORPUS, rel)).convert('RGB')
    img.thumbnail((WORKING, WORKING), Image.LANCZOS)
    return img


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
    """One half of the object, split across its longer axis -- the leg-left-behind case."""
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
    for case, rel, box, positional, bare in CASES:
        img = image(rel)
        mask, _cov = O.segment_object(img, box)
        full = recolour(img, mask)
        halved = recolour(img, half(mask))
        x0, y0, x1, y1 = box
        pad = 40
        crop = (max(0, x0 - pad), max(0, y0 - pad),
                min(img.size[0], x1 + pad), min(img.size[1], y1 + pad))
        locate.mask_overlay(img, mask).crop(crop).save(
            os.path.join(out_dir, f'{case}_mask.jpg'), quality=85)
        full.crop(crop).save(os.path.join(out_dir, f'{case}_full.jpg'), quality=85)
        row = {'case': case, 'phrase': positional, 'bare': bare}
        for label, edit in (('full', full), ('half', halved)):
            for kind, phrase in (('pos', positional), ('bare', bare)):
                row[label + '_' + kind] = verdict(img, edit, box, phrase)
        rows.append(row)
        print(f'{case:13s} full: {row["full_pos"][0]:4s} (positional) '
              f'{row["full_bare"][0]:4s} (bare)   '
              f'half: {row["half_pos"][0]:4s} {row["half_bare"][0]:4s}')
    summarise(rows)
    print(f'\nsheets in {out_dir} -- LOOK at the masks before trusting any row.')
    return rows


def summarise(rows):
    def count(key, want):
        return sum(1 for r in rows if r[key][0] == want)
    n = len(rows)
    print(f'\n{"":18s} {"PASS":>5s} {"none":>5s} {"FAIL":>5s}   of {n}')
    for key, want in (('full_pos', 'PASS'), ('full_bare', 'PASS'),
                      ('half_pos', 'FAIL'), ('half_bare', 'FAIL')):
        print(f'{key:18s} {count(key, "PASS"):5d} {count(key, "none"):5d} '
              f'{count(key, "FAIL"):5d}   want {want}')
    lost = [r['case'] for r in rows
            if r['full_bare'][0] == 'PASS' and r['full_pos'][0] != 'PASS']
    if lost:
        print(f'\nCorrect work whose verdict the positional wording COST: {", ".join(lost)}')
    for r in rows:
        if r['full_pos'][0] != r['full_bare'][0]:
            print(f'\n{r["case"]}\n  {r["phrase"]!r}: {r["full_pos"][1]}'
                  f'\n  {r["bare"]!r}: {r["full_bare"][1]}')


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else SHEETS)
