"""Field survey: how often does `check_object_erased`'s verdict match what a person sees?

    .venv/bin/python evals/erase_field.py [out_dir] [telea|lama]

`evals/erase_verdicts.py` validates the check on hand-built and composited cases; this runs
it across a spread of real erase operations (`objects.erase_object`, no synthetic masks) on
real photographs and measures how often the verdict agrees with a human looking at the
before/after crop. Photographs are thumbnailed to at most 1200px before any box is drawn.

The human judgement lives in this file, `JUDGEMENTS` below, set by looking at each case's
`_before_after.jpg` at 1:1 -- not at this script's numbers. Labelled separately per fill
backend, since TELEA and LaMa fail differently.
"""
import os
import re
import sys

from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from figsurgeon import objects as O                                     # noqa: E402
from figsurgeon import verify_photo as V                                # noqa: E402

CORPUS = os.path.join(HERE, '_corpus')
SHEETS = os.path.join(HERE, 'out_erase_field')
WORKING = 1200

# (name, path under _corpus/, box in the WORKING-thumbnailed frame, subject phrase,
#  category, note)
CASES = [
    # --- small isolated objects: expected to erase cleanly ---
    ('roadworks_cone', 'agent_loop/roadworks.jpg', (1030, 520, 1200, 710),
     'the traffic cone', 'small/isolated', 'foreground cone at a petrol station forecourt'),
    ('car_red_street', 'car_red.jpg', (540, 795, 810, 900),
     'the red car', 'small/isolated on busy bg', 'Tesla, small in a wide street facade shot'),
    ('taxi_foreground', 'agent_loop/taxi.jpg', (390, 460, 500, 545),
     'the yellow taxi in the foreground', 'small/isolated on busy bg',
     'tilt-shift traffic scene, many other cars nearby'),

    # --- thin / skeletal objects ---
    ('sailboat_mast', 'eight_boxes/boat_sunset.jpg', (20, 400, 340, 840),
     'the sailboat', 'thin/skeletal, flat bg', 'thin mast and rigging against gradient sky'),
    ('night_tower', 'night_city.jpg', (790, 15, 955, 345),
     'the illuminated tower', 'thin/skeletal, flat bg', 'narrow spire against night sky'),
    ('striped_bollard', 'occlusion/bollards_car.jpg', (300, 440, 405, 730),
     'the striped bollard', 'thin/skeletal on busy bg', 'thin pole, night street scene'),

    # --- objects on busy backgrounds ---
    ('kingfisher', 'eight_boxes/kingfisher.jpg', (470, 140, 1040, 610),
     'the kingfisher', 'busy background', 'wings spread, textured branch + foliage bg'),
    ('white_horse', 'eight_boxes/horse_white.jpg', (215, 120, 660, 1060),
     'the white horse', 'large + busy background', 'trees and fence behind it'),
    ('shepherd_dog', 'eight_boxes/dog_shepherd.jpg', (110, 15, 1105, 760),
     'the dog', 'large + busy background', 'fills most of the frame, grass bg'),
    ('cow_at_fence', 'occlusion/cow_fence.jpg', (260, 10, 1080, 650),
     'the cow', 'large + busy background', 'barbed-wire fence crosses the body'),
    ('street_person', 'street_people.jpg', (505, 395, 625, 730),
     'the man in the black shirt', 'medium on busy bg', 'street scene, several other people'),
    ('cat_face', 'audit_phrase/cat_grass.jpg', (0, 0, 460, 800),
     'the cat', 'large + busy background', 'fur detail against grass'),
    ('motorcycle', 'eight_boxes/motorcycle.jpg', (160, 150, 1080, 770),
     'the motorcycle', 'very large + busy background', 'fence and park behind it'),

    # --- objects on flat backgrounds ---
    ('red_backpack', 'eight_boxes/backpack_red.jpg', (150, 170, 900, 1150),
     'the red backpack', 'large + flat background', 'studio product shot, plain grey wall'),
    ('white_shoe', 'product_white.jpg', (580, 110, 990, 700),
     'the shoe', 'large + flat background', 'known past limitation case (see docstring)'),
    ('red_shoes_pair', 'product_shoe.jpg', (430, 15, 900, 390),
     'the red shoes', 'medium-large + flat background', 'black background, high contrast'),
    ('studio_umbrella', 'portrait_studio.jpg', (0, 0, 798, 700),
     'the blue umbrella', 'very large + flat background', 'thin ribs, plain backdrop'),
    ('blue_chair', 'eight_boxes/chairs_blue.jpg', (380, 455, 640, 700),
     'the chair on the left', 'medium + flat background', 'tiled floor, two identical chairs'),
    ('guitarist', 'eight_boxes/guitarist.jpg', (100, 150, 750, 1200),
     'the guitarist', 'large + flat background', 'stage lighting, dark background'),
    ('red_door', 'agent_loop/door.jpg', (455, 0, 1200, 799), 'the red door',
     'large + flat background', 'flat saturated wall either side'),
    ('flower_petal', 'flower_macro.jpg', (0, 0, 260, 380), 'the petal',
     'small/thin on busy bg', 'macro shot, other petals crowd the frame'),

    # --- deliberately WRONG / partial boxes (FAIL direction) ---
    ('backpack_straps_only', 'eight_boxes/backpack_red.jpg', (250, 60, 650, 260),
     'the red backpack', 'WRONG BOX: partial', 'box covers only the carry handles, not '
     'the bag body -- most of "the backpack" sits outside it'),
    ('dog_head_only', 'eight_boxes/dog_shepherd.jpg', (620, 20, 1010, 320),
     'the dog', 'WRONG BOX: partial', 'box covers only the head -- the body is untouched'),
    ('second_car_sliver', 'agent_loop/two_cars.jpg', (790, 380, 1080, 560),
     'the car', 'WRONG BOX: partial', 'box mostly grass + a sliver of the rear car, not '
     'either car whole'),
]


# What a person sees in the before/after crop, one label per case:
#   'clean'  -- the object is gone and what replaced it reads as background
#   'ghost'  -- the object is still visible, or a smear in its shape is
#   'faint'  -- neither: mostly gone, with residue visible at 1:1 (not scored)
#
# Labelled per fill backend (TELEA and LaMa fail differently). TELEA's finding: 22 of 24
# are 'ghost' -- it propagates surrounding colour and can't invent structure behind a large
# object, leaving too few clean cases to calibrate a PASS band against.
JUDGEMENTS_TELEA = {
    'roadworks_cone':       ('clean', 'soft patch of asphalt where the cone was; reads as road'),
    'kingfisher':           ('faint', 'bird gone and the branch continues, but a translucent '
                                      'wing shape and a warm smudge survive in the bokeh'),

    'car_red_street':       ('ghost', 'dark car-shaped blob across the pavement'),
    'taxi_foreground':      ('ghost', 'dark bowtie-shaped blob in the traffic lane'),
    'sailboat_mast':        ('ghost', 'white smear on the water; mast and rigging still legible'),
    'night_tower':          ('ghost', 'orange plume smeared up out of the roofline'),
    'striped_bollard':      ('ghost', 'pale vertical smear in the bollard shape'),
    'white_horse':          ('ghost', 'horse-shaped green smear, legs and head still readable'),
    'shepherd_dog':         ('ghost', 'dog-shaped green smear filling most of the box'),
    'cow_at_fence':         ('ghost', 'body smeared to green; the head is untouched'),
    'street_person':        ('ghost', 'blue and white smear standing where the man was'),
    'cat_face':             ('ghost', 'grey wedges over the fur; the face is untouched'),
    'motorcycle':           ('ghost', 'green and grey fan across the kerb; mirror still there'),
    'red_backpack':         ('ghost', 'pink polygons in the bag shape; the straps survive'),
    'white_shoe':           ('ghost', 'grey and cream wedges; half the shoe is still there'),
    'red_shoes_pair':       ('ghost', 'red and black smear over the printed sheet'),
    'studio_umbrella':      ('ghost', 'the umbrella is restyled into blue facets, not removed'),
    'blue_chair':           ('ghost', 'grey smear where the chair was; its shadow stays'),
    'guitarist':            ('ghost', 'pale figure-shaped smear; the guitar neck survives'),
    'red_door':             ('ghost', 'pink and white smear over the whole door'),
    'flower_petal':         ('ghost', 'the petal is still there -- nothing visible was removed'),
    'backpack_straps_only': ('ghost', 'WRONG BOX: handle stubs left floating in the air'),
    'dog_head_only':        ('ghost', 'WRONG BOX: the dog is still there, muzzle smeared'),
    'second_car_sliver':    ('ghost', 'WRONG BOX: dark smear over the hedge, car untouched'),
}

# Filled the same way, by looking at every sheet from a `fill='lama'` run at 1:1.
#
# 4 clean, 2 faint, 18 ghost -- better than TELEA's 1 clean, but a different kind of ghost:
# TELEA leaves a smear, LaMa regenerates the object, since the most plausible continuation
# of a masked-out subject is often the subject itself.
JUDGEMENTS_LAMA = {
    'roadworks_cone':       ('clean', 'cone gone, asphalt and kerb line continue through'),
    'car_red_street':       ('clean', 'car gone; pavement, crossing and shopfront rebuilt'),
    'street_person':        ('clean', 'man gone; pavement, awning and shopfront rebuilt'),
    'red_door':             ('clean', 'door gone, replaced by plausible rendered wall'),

    'shepherd_dog':         ('faint', 'dog gone and the grass is grass, but a dog-shaped '
                                      'darker patch survives where it lay'),
    'blue_chair':           ('faint', 'chair gone and the tiles continue, but a white leg '
                                      'stub and a faint haze are left'),

    'taxi_foreground':      ('ghost', 'REGENERATED: a yellow-brown van-shaped body in its place'),
    'sailboat_mast':        ('ghost', 'REGENERATED: hull gone, mast and rigging redrawn as '
                                      'fine lines against the sky'),
    'night_tower':          ('ghost', 'REGENERATED: a new grey spire built on the roofline'),
    'striped_bollard':      ('ghost', 'a dark stub of post is left at the base; shadow stays'),
    'kingfisher':           ('ghost', 'REGENERATED: the bird comes back in translucent green'),
    'white_horse':          ('ghost', 'INVENTED: a hedge and bush where the head and neck were'),
    'cow_at_fence':         ('ghost', 'REGENERATED: a translucent body-shaped tent; head stays'),
    'cat_face':             ('ghost', 'the face is untouched and the body became smooth fur-grey'),
    'motorcycle':           ('ghost', 'fill is convincing, but the mirror is left floating in '
                                      'mid-air and the bike-shaped shadow stays'),
    'red_backpack':         ('ghost', 'REGENERATED: a faded backpack, handles and all'),
    'white_shoe':           ('ghost', 'REGENERATED: a ghost-white shoe in the same pose'),
    'red_shoes_pair':       ('ghost', 'a dark red shoe-shaped mass remains'),
    'studio_umbrella':      ('ghost', 'REGENERATED: the umbrella redrawn, cleaner than before'),
    'guitarist':            ('ghost', 'REGENERATED: a figure-shaped translucent slab'),
    'flower_petal':         ('ghost', 'the petal is still there -- nothing visible was removed'),
    'backpack_straps_only': ('ghost', 'WRONG BOX: orange debris where the handle was'),
    'dog_head_only':        ('ghost', 'WRONG BOX: the dog is still there, muzzle smudged'),
    'second_car_sliver':    ('ghost', 'WRONG BOX: the grass is rebuilt well; the car, which '
                                      'is what was asked for, is untouched'),
}

JUDGEMENTS = {'telea': JUDGEMENTS_TELEA, 'lama': JUDGEMENTS_LAMA}


def load_working(rel):
    img = Image.open(os.path.join(CORPUS, rel))
    img = ImageOps.exif_transpose(img).convert('RGB')
    img.thumbnail((WORKING, WORKING), Image.LANCZOS)
    return img


def _v(ok):
    return {True: 'PASS', False: 'FAIL', None: 'undecided/none'}[ok]


def run(out_dir=SHEETS, fill='telea'):
    os.makedirs(out_dir, exist_ok=True)
    labels = JUDGEMENTS[fill]
    rows = []
    for name, rel, box, subject, category, note in CASES:
        before = load_working(rel)
        try:
            out, frame_frac, warning = O.erase_object(before, box=box, fill=fill)
        except Exception as e:
            print(f'{name:22s} ERASE FAILED  {type(e).__name__}: {e}')
            rows.append({'name': name, 'category': category, 'error': str(e)})
            continue
        after = out

        # A generous crop around the box (margin = 40% of the box's own size) so the
        # sheet shows enough surrounding context to judge smear/containment by eye.
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
        compare.save(os.path.join(out_dir, f'{name}_before_after.jpg'), quality=92)

        ok, detail = V.check_object_erased(before, after, box, subject=subject)

        warn_fired = warning is not None
        row = {
            'name': name, 'category': category, 'photo': rel, 'box': box,
            'subject': subject, 'frame_frac': frame_frac, 'smear_warning': warn_fired,
            'ok': ok, 'detail': detail, 'note': note,
        }
        rows.append(row)
        label = labels.get(name, ('?', ''))[0]
        print(f'{name:22s} frame_frac={frame_frac:5.1%}  smear_warn={"Y" if warn_fired else "n":1s}  '
              f'eye={label:5s}  verdict={_v(ok):14s}  {detail}')

    print(f'\n{len(rows)} cases with fill={fill!r}; sheets (before|after crops) in {out_dir}')
    tally(rows, fill)
    return rows


_READING = re.compile(r'(\d+)% of the original extent')


def _reading(detail):
    """The `remaining` fraction out of the detail line, or None if the check abstained."""
    m = _READING.search(detail or '')
    return None if m is None else int(m.group(1)) / 100.0


def tally(rows, fill='telea'):
    """Score the readings against `JUDGEMENTS`. Counts readings, not verdicts: there's no
    PASS verdict to key on, so what matters is how often a low reading lands on a picture a
    person calls a ghost -- the error any candidate PASS band would have to beat.
    """
    labels = JUDGEMENTS[fill]
    if not labels:
        print(f'\nno labels recorded for fill={fill!r} yet -- look at the sheets and fill '
              f'in JUDGEMENTS_{fill.upper()} before reading anything into the numbers')
        return
    low, false_fail, scored = [], [], 0
    for r in rows:
        label = labels.get(r['name'], ('?', ''))[0]
        x = _reading(r.get('detail'))
        if label in ('clean', 'ghost'):
            scored += 1
        if x is not None and x <= V._ERASE_GONE and label in ('clean', 'ghost'):
            low.append((r['name'], x, label))
        if r.get('ok') is False and label == 'clean':
            false_fail.append(r['name'])
    wrong = [t for t in low if t[2] == 'ghost']
    print(f'\n{scored} of {len(rows)} cases carry a clean/ghost label '
          f'({sum(1 for n in labels.values() if n[0] == "faint")} faint, not scored)')
    print(f'{len(low)} readings sit in the low band (<= {V._ERASE_GONE:.2f}), '
          f'of which {len(wrong)} are on a picture the eye calls a ghost:')
    for name, x, label in low:
        mark = 'WOULD HAVE PASSED A GHOST' if label == 'ghost' else 'correct'
        print(f'    {name:22s} {x:4.0%}  {label:5s}  {mark}')
    print(f'false FAILs (verdict FAIL on a clean erase): {len(false_fail)}'
          + (f' -- {false_fail}' if false_fail else ''))


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else SHEETS,
        sys.argv[2] if len(sys.argv) > 2 else 'telea')
