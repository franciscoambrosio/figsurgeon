"""Regression tests for objects.py, from real measurements against skimage sample images.

Guards a README-documented pattern (a full-frame box starves GrabCut's background model;
a large erased region smears because inpainting has no information about what was behind
it) and the flag-stays-red case: a hue-based recolour of "the suit's colour" would also
catch the flag's red stripes, which fall inside the same bounding box, but GrabCut's
spatial segmentation does not, since the flag is not part of the object found in the box.
"""
import numpy as np
from PIL import Image
from skimage import data

from figsurgeon import objects
from figsurgeon.tools import dispatch

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@check('full-frame box raises ValueError on GrabCut, not an opaque cv2 assertion')
def t1():
    img = Image.fromarray(data.astronaut())
    W, H = img.size
    try:
        objects.segment_object(img, (0, 0, W, H), backend='grabcut')
        return False, 'no exception raised'
    except ValueError as e:
        return True, str(e)[:60]
    except Exception as e:
        return False, f'wrong exception type: {type(e).__name__}'


@check('a 98%-of-frame box also raises on GrabCut (not just exactly 100%)')
def t2():
    img = Image.fromarray(data.astronaut())
    W, H = img.size
    try:
        objects.segment_object(img, (2, 2, W - 2, H - 2), backend='grabcut')
        return False, 'no exception raised'
    except ValueError:
        return True, 'raised as expected'


@check('SlimSAM has no background model, so a full-frame box is answered, not refused')
def t2b():
    """GrabCut's full-frame guard does not apply to SlimSAM, which has no background model:
    a full-frame box is a legitimate way to ask for the salient object. Skipped without
    torch."""
    if not objects.available():
        return True, 'skipped: no torch'
    img = Image.fromarray(data.astronaut())
    W, H = img.size
    mask, coverage = objects.segment_object(img, (0, 0, W, H), backend='slimsam')
    return 0.05 < coverage < 0.60, f'coverage={coverage:.0%} of the whole frame'


@check('recolour_object: astronaut suit turns blue, flag stripe inside the same box stays red')
def t3():
    img = Image.fromarray(data.astronaut())
    box = (60, 180, 420, 512)
    out, coverage = objects.recolour_object(img, box, to_rgb=(30, 60, 180))
    before, after = np.array(img), np.array(out)

    flag_pt = (180, 69)   # (y, x): a red flag-stripe pixel that falls INSIDE the box
    suit_pt = (300, 150)  # (y, x): a suit pixel

    flag_unchanged = np.abs(after[flag_pt].astype(int) - before[flag_pt].astype(int)).max() < 5
    suit_b_before = int(before[suit_pt][2])
    suit_r_after, suit_b_after = int(after[suit_pt][0]), int(after[suit_pt][2])
    suit_turned_blue = suit_b_after > suit_r_after and suit_b_after > suit_b_before
    return (flag_unchanged and suit_turned_blue,
            f'flag {before[flag_pt]}->{after[flag_pt]}, suit {before[suit_pt]}->{after[suit_pt]}, '
            f'coverage={coverage:.0%}')


@check('recolour_object dispatch no longer carries the coverage > 0.95 give-up warning')
def t4():
    """No coverage-based give-up warning exists: no signal tested separates a genuine give-up
    from a correct object that legitimately fills its box (a railing, a door). Forced through
    GrabCut, the backend most likely to reach high coverage."""
    img = Image.fromarray(data.chelsea())
    original = objects.choose_backend
    objects.choose_backend = lambda im: 'grabcut'
    try:
        out, note = dispatch('recolour_object',
                             {'box': [0, 0, 300, 300], 'to_rgb': [30, 60, 180]}, img)
    finally:
        objects.choose_backend = original
    return 'WARNING' not in note, note[:80]


@check('the note always states how much of the box was recoloured, high coverage or not')
def t4b():
    """The plain coverage number is reported unconditionally, with no interpretive warning."""
    img = Image.fromarray(data.chelsea())
    out, note = dispatch('recolour_object',
                         {'box': [0, 0, 300, 300], 'to_rgb': [30, 60, 180]}, img)
    return '% of box' in note, note[:80]


@check('isolate_object: chelsea cat isolated from same-hue background')
def t5():
    img = Image.fromarray(data.chelsea())
    box = (60, 20, 400, 290)
    out, coverage = objects.isolate_object(img, box)
    return 0.2 < coverage < 0.8, f'coverage={coverage:.0%}'


@check('erase_object: rocket mast (~1% of frame) erases cleanly, no size warning')
def t6():
    img = Image.fromarray(data.rocket())
    box = (150, 80, 230, 340)
    out, frame_frac, warning = objects.erase_object(img, box)
    return (frame_frac < 0.05 and warning is None,
            f'frame_frac={frame_frac:.1%}, warning={warning!r}')


@check('erase_object: coffee cup (~33% of frame) triggers the size warning')
def t7():
    img = Image.fromarray(data.coffee())
    box = (80, 30, 420, 340)
    out, frame_frac, warning = objects.erase_object(img, box)
    return (frame_frac > 0.08 and warning is not None,
            f'frame_frac={frame_frac:.1%}, warning={"set" if warning else None}')


@check('extract_object: returns RGBA with meaningful alpha coverage')
def t8():
    img = Image.fromarray(data.chelsea())
    box = (60, 20, 400, 290)
    out = objects.extract_object(img, box)
    alpha = np.array(out)[:, :, 3]
    frac_opaque = float((alpha > 128).mean())
    return out.mode == 'RGBA' and 0.05 < frac_opaque < 0.95, f'frac_opaque={frac_opaque:.0%}'


def run():
    ok = 0
    for name, fn in CHECKS:
        try:
            passed, detail = fn()
        except Exception as e:
            passed, detail = False, f'{type(e).__name__}: {e}'
        print(f'  {"PASS" if passed else "FAIL"}  {name}  [{detail}]')
        ok += bool(passed)
    print(f'\n{ok}/{len(CHECKS)} object-editing checks passed')
    return ok == len(CHECKS)


if __name__ == '__main__':
    import sys
    sys.exit(0 if run() else 1)


def test_object_editing_checks():
    assert run(), 'see printed output for which object-editing check failed'
