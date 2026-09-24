"""Does `check_object_erased`, the verdict for `erase_object`, actually work?

    .venv/bin/python evals/erase_verdicts.py [out_dir]

`erase_object` can't be graded like the other object checks (mask vs. changed pixels): if it
worked, the object is gone from the output. `check_object_erased` instead asks CLIPSeg the
caller's phrase after the edit, at the pixels the same phrase pointed to before it.

Three kinds of evidence: (1) calibration against a pixel-exact rendered ground truth --
eight real objects composited onto real photographs, then a known fraction of each reverted
to the true background (`composite_case`/`partial_erase`); (2) the real tool on real photos,
both a clean must-pass erase and must-fail cases (partial erase, no-op, wrong box); (3) a
documented known limitation, where a large smeared erase fools CLIPSeg into a wrong PASS.
"""
import os
import re
import sys

import numpy as np
from PIL import Image
from skimage import data as skdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from figsurgeon import grounding                                        # noqa: E402
from figsurgeon import objects as O                                     # noqa: E402
from figsurgeon import verify_photo as V                                # noqa: E402
from evals import real_corpus                                            # noqa: E402

SHEETS = os.path.join(HERE, 'out_erase_verdicts')
CORPUS = os.path.join(HERE, '_corpus')
WORKING = 1200      # same working size `object_verdicts.py` and its neighbours downscale to


def _v(ok):
    return {True: 'PASS', False: 'FAIL', None: 'none'}[ok]


# --------------------------------------------------------------------- 1. calibration

def object_cutout(source_img, box, target_size):
    """A tight crop of `source_img` around `box`, resized to `target_size`, plus its own
    segmentation mask at that size -- the thing that gets composited elsewhere."""
    mask, _cov = O.segment_object(source_img, box)
    mask = np.asarray(mask, dtype=float)
    ys, xs = np.nonzero(mask > 0.5)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = source_img.crop((int(x0), int(y0), int(x1) + 1, int(y1) + 1))
    m_crop = Image.fromarray((mask[y0:y1 + 1, x0:x1 + 1] * 255).astype(np.uint8))
    crop_r = np.array(crop.resize(target_size, Image.LANCZOS), dtype=float)
    mask_r = np.asarray(m_crop.resize(target_size, Image.LANCZOS), dtype=float) / 255.0
    return crop_r, mask_r


def composite_case(background_img, obj_rgb, obj_mask, top_left):
    """Paste `obj_rgb` (masked by `obj_mask`) onto `background_img` at `top_left`. Returns
    (composite image, box, the background pixels under the paste -- the real, unedited
    ground truth a correct erase must recover)."""
    px, py = top_left
    h, w = obj_mask.shape
    base = np.array(background_img.convert('RGB')).astype(float)
    region_bg = base[py:py + h, px:px + w].copy()
    m3 = obj_mask[:, :, None]
    base[py:py + h, px:px + w] = region_bg * (1 - m3) + obj_rgb * m3
    return Image.fromarray(base.astype(np.uint8)), [px, py, px + w, py + h], region_bg


def partial_erase(composite_img, obj_mask, region_bg, top_left, fraction):
    """Revert `fraction` of the object's own AREA (top rows first) back to the real
    background it was pasted over -- a pixel-exact "X % erased" image, independent of
    whether any inpainter could actually do this well. `fraction=1.0` recovers the true
    background exactly; `fraction=0.0` is the composite unchanged (a genuine no-op)."""
    px, py = top_left
    h, w = obj_mask.shape
    present = obj_mask > 0.5
    ys, xs = np.nonzero(present)
    order = np.argsort(ys, kind='stable')          # top-to-bottom, deterministic
    n_revert = int(round(len(ys) * fraction))
    remaining = present.copy()
    if n_revert:
        remaining[ys[order[:n_revert]], xs[order[:n_revert]]] = False
    a = np.array(composite_img).astype(float)
    obj_rgb = a[py:py + h, px:px + w]
    m3 = remaining.astype(float)[:, :, None]
    a[py:py + h, px:px + w] = region_bg * (1 - m3) + obj_rgb * m3
    return Image.fromarray(a.astype(np.uint8))


FRACTIONS = [0.0, 0.30, 0.50, 0.70, 0.85, 1.0]     # fraction of the object's area erased


def _corpus_image(rel):
    img = Image.open(os.path.join(CORPUS, rel)).convert('RGB')
    img.thumbnail((WORKING, WORKING), Image.LANCZOS)
    return img


# (name, category, source image/box, subject phrase, cutout size, background image, paste
# top-left). Eight unrelated objects, each pasted onto a different real photograph, covering
# thin/skeletal, low-contrast, large, busy-background, textured, and human subjects.
CASES = [
    ('dog_on_floor', 'textured (fur)',
     'eight_boxes/dog_shepherd.jpg', (150, 30, 1060, 740), 'the dog',
     (380, 300), 'product_white.jpg', (20, 440)),
    ('bird_on_flowers', 'thin/skeletal (open beak, spread wing feathers, thin legs)',
     'eight_boxes/kingfisher.jpg', (480, 130, 1010, 600), 'the bird',
     (260, 220), 'flower_macro.jpg', (700, 80)),
    ('backpack_on_street', 'textured, busy background',
     'eight_boxes/backpack_red.jpg', (110, 25, 880, 1160), 'the backpack',
     (220, 320), 'street_people.jpg', (850, 380)),
    ('chair_on_skyline', 'manmade, thin frame',
     'eight_boxes/chairs_blue.jpg', (705, 430, 975, 700), 'the chair',
     (240, 240), 'city_wide.jpg', (280, 650)),
    ('motorcycle_on_floor', 'large object',
     'eight_boxes/motorcycle.jpg', (140, 75, 1095, 765), 'the motorcycle',
     (560, 395), 'product_white.jpg', (180, 320)),
    ('boat_on_night_city', 'thin/skeletal (mast and rigging), busy background',
     'eight_boxes/boat_sunset.jpg', (10, 415, 340, 845), 'the sailboat',
     (230, 335), 'night_city.jpg', (700, 300)),
    ('horse_on_haze', 'low-contrast (pale coat pasted onto a hazy, similarly pale sky)',
     'eight_boxes/horse_white.jpg', (170, 130, 660, 1090), 'the horse',
     (120, 470), 'city_wide.jpg', (15, 15)),
    ('person_on_street', 'a person',
     'street_people.jpg', (390, 370, 500, 720), 'the person',
     (140, 350), 'car_red.jpg', (950, 500)),
]


def run_calibration(out_dir):
    rows = []
    for name, category, src_rel, box, subject, size, bg_rel, top_left in CASES:
        src = _corpus_image(src_rel)
        obj_rgb, obj_mask = object_cutout(src, box, size)
        bg = _corpus_image(bg_rel)
        before, comp_box, region_bg = composite_case(bg, obj_rgb, obj_mask, top_left)
        before.save(os.path.join(out_dir, f'{name}_full.jpg'), quality=90)
        for frac in FRACTIONS:
            after = partial_erase(before, obj_mask, region_bg, top_left, frac)
            ok, detail = V.check_object_erased(before, after, comp_box, subject=subject)
            true_remaining = 1.0 - frac
            rows.append({'case': name, 'category': category, 'erased_frac': frac,
                        'true_remaining': true_remaining, 'ok': ok, 'detail': detail})
            print(f'{name:20s} erased {frac:4.0%} (true remaining {true_remaining:4.0%}) '
                 f'-> {_v(ok):4s}  {detail}')
    return rows


# --------------------------------------------------------------- 2 & 3. the real tool

def run_real_tool(out_dir):
    rows = []

    # MUST PASS: a small, isolated object, erased cleanly against a plain background --
    # exactly the case `erase_object`'s own docstring claims works ("convincing at ~1% of
    # frame, a thin mast against sky").
    astronaut = Image.fromarray(skdata.astronaut())
    box = [368, 0, 500, 195]
    out, frame_frac, warning = O.erase_object(astronaut, box=box)
    out.crop((box[0] - 20, box[1], box[2] + 20, box[3] + 40)).save(
        os.path.join(out_dir, 'shuttle_erased.jpg'), quality=90)
    ok, detail = V.check_object_erased(astronaut, out, box, subject='the rocket')
    # `check_object_erased` has no PASS verdict (see its docstring), so what a clean erase
    # must not do is fail -- abstaining is the good outcome here.
    rows.append({'case': 'shuttle_model (real tool)', 'want': 'NOT FAIL', 'ok': ok,
                'detail': f'frame_frac={frame_frac:.1%}; ' + detail})
    print(f'MUST NOT FAIL  shuttle_model  frame_frac={frame_frac:.1%} -> {_v(ok)}  {detail}')

    # MUST FAIL: the same tool, aimed at a box that only covers PART of a small object (a
    # spoon's handle, not its bowl) -- the object is not gone, most of it just sits outside
    # what got inpainted.
    coffee = Image.fromarray(skdata.coffee())
    box = [330, 130, 460, 260]
    out, frame_frac, warning = O.erase_object(coffee, box=box)
    out.crop((box[0] - 40, box[1] - 40, box[2] + 40, box[3] + 40)).save(
        os.path.join(out_dir, 'spoon_erased.jpg'), quality=90)
    ok, detail = V.check_object_erased(coffee, out, box, subject='the spoon')
    rows.append({'case': 'spoon_partial (real tool)', 'want': 'FAIL', 'ok': ok,
                'detail': f'frame_frac={frame_frac:.1%}; ' + detail})
    print(f'MUST FAIL  spoon_partial  frame_frac={frame_frac:.1%} -> {_v(ok)}  {detail}')

    # MUST FAIL: a no-op. The call ran, segmentation found nothing worth inpainting (or the
    # caller's box sat over featureless background), and nothing changed.
    ok, detail = V.check_object_erased(astronaut, astronaut.copy(), [60, 180, 420, 512],
                                       subject='the suit')
    rows.append({'case': 'silent_noop', 'want': 'FAIL', 'ok': ok, 'detail': detail})
    print(f'MUST FAIL  silent_noop -> {_v(ok)}  {detail}')

    # MUST FAIL: a box drawn next to the object rather than on it, constructed rather than
    # found (a real call rarely lands squarely half-on an object). Half the wrong box is
    # background that visibly changed; the other half is the true, untouched object.
    suit_rgb, suit_mask = object_cutout(Image.fromarray(skdata.astronaut()),
                                        [60, 180, 420, 512], (140, 138))
    before, _box, _bg = composite_case(
        Image.fromarray(skdata.rocket()).convert('RGB'), suit_rgb, suit_mask, (385, 10))
    a = np.array(before).astype(float)
    box_wrong = [425, 10, 565, 148]         # the suit is at [385, 10, 525, 148]
    changed = a.copy()
    changed[10:148, 525:565] = np.clip(a[10:148, 525:565] + 40, 0, 255)
    after = Image.fromarray(changed.astype(np.uint8))
    ok, detail = V.check_object_erased(before, after, box_wrong, subject='the orange suit')
    rows.append({'case': 'wrong_box', 'want': 'FAIL', 'ok': ok, 'detail': detail})
    print(f'MUST FAIL  wrong_box -> {_v(ok)}  {detail}')

    # WHAT DID NOT WORK: a large object, erased past the tool's own smear warning, left a
    # warped but human-recognisable ghost -- and CLIPSeg did not recognise it either. Kept
    # here as a documented failure, not fixed by moving a threshold.
    shoe = real_corpus.load('product_white')
    box = [480, 100, 990, 700]
    out, frame_frac, warning = O.erase_object(shoe, box=box)
    out.crop((box[0] - 40, box[1] - 40, box[2] + 40, box[3] + 40)).save(
        os.path.join(out_dir, 'shoe_smeared.jpg'), quality=90)
    ok, detail = V.check_object_erased(shoe, out, box, subject='the shoe')
    frame_prob_after = grounding._probability(out, 'the shoe')
    peak_whole_frame = float(frame_prob_after[box[1]:box[3], box[0]:box[2]].max())
    rows.append({'case': 'shoe_smeared (KNOWN LIMITATION)', 'want': 'FAIL or no-verdict',
                'ok': ok, 'detail': f'frame_frac={frame_frac:.1%}; {detail}; '
                                    f'whole-frame peak for "the shoe" after: '
                                    f'{peak_whole_frame:.2f}'})
    print(f'KNOWN LIMIT  shoe_smeared  frame_frac={frame_frac:.1%} -> {_v(ok)} '
         f'(whole-frame peak {peak_whole_frame:.2f})  {detail}')
    return rows


def _read_remaining(detail):
    """What the check actually read, recovered from its own sentence -- the same parse the
    error statistics below already rely on. None where no number was reported (a no-op)."""
    m = re.search(r'(\d+)% of the original extent', detail)
    return int(m.group(1)) / 100.0 if m else None


def summarise(cal_rows, real_rows):
    print('\n--- calibration: true fraction still present vs. what the check reads ---')
    for case in sorted(set(r['case'] for r in cal_rows)):
        crows = [r for r in cal_rows if r['case'] == case]
        print(f'{case}:')
        for r in crows:
            read = r['detail']
            print(f"  true {r['true_remaining']:4.0%}  ok={_v(r['ok']):4s}  {read}")
    errs = []
    for r in cal_rows:
        read = _read_remaining(r['detail'])
        if read is not None:
            errs.append(abs(read - r['true_remaining']))
    if errs:
        errs.sort()
        median = errs[len(errs) // 2] if len(errs) % 2 else (
            errs[len(errs) // 2 - 1] + errs[len(errs) // 2]) / 2
        print(f'\n{len(errs)} numeric points; median error {median:.2f}; '
             f'worst {max(errs):.2f}')

    # Both error directions are counted, not just one: a false FAIL is as bad as a false
    # PASS. Since there's no PASS verdict, unsafe_pass is read from the low-band sentence
    # (not the rounded percentage, which would misclassify borderline values).
    unsafe_pass = [r for r in cal_rows
                   if 'NOT a verdict that it is gone' in (r['detail'] or '')
                   and r['true_remaining'] >= 0.50]
    unsafe_fail = [r for r in cal_rows if r['ok'] is False and r['true_remaining'] <= 0.15]
    print(f'\n{len(unsafe_pass)} point(s) reading inside the low band with half or more of '
         f'the object\'s true area still present (no longer passed, but this is why):')
    for r in unsafe_pass:
        print(f"  {r['case']:20s} true {r['true_remaining']:4.0%}  {r['detail']}")
    print(f'{len(unsafe_fail)} FAIL verdict(s) with the object 85 % or more gone (correct '
         f'work called incomplete).')

    print('\n--- real tool, both directions ---')
    wrong = 0
    for r in real_rows:
        flag = ''
        want = r['want']
        if want == 'NOT FAIL' and r['ok'] is False:
            flag = '  <-- WANTED ANYTHING BUT FAIL'
            wrong += 1
        if want == 'FAIL' and r['ok'] is not False:
            flag = '  <-- WANTED FAIL'
            wrong += 1
        print(f"{r['case']:34s} want {want:18s} got {_v(r['ok']):4s}{flag}")
    print(f'\n{wrong} unexpected verdict(s) among the cases with a firm expectation '
         f'(the smeared-shoe row is reported, not scored -- see the module docstring).')


def run(out_dir=SHEETS):
    os.makedirs(out_dir, exist_ok=True)
    cal_rows = run_calibration(out_dir)
    real_rows = run_real_tool(out_dir)
    summarise(cal_rows, real_rows)
    print(f'\nsheets in {out_dir}')
    return cal_rows, real_rows


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else SHEETS)
