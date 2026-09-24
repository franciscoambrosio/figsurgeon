"""`space='oklab'` is wired through, and behaves differently enough to be worth having.

These checks go through the same doors a caller uses (`photo.*` and `tools.dispatch`), not
`perceptual.colour_mask` directly, since the wiring itself is the fragile part.

Thresholds are loose enough not to be a snapshot of evals/colour_spaces.py's magnitudes, and
tight enough that silently reverting to the hue window would fail.
"""
import numpy as np
from PIL import Image
from skimage import data

from figsurgeon import photo
from figsurgeon.tools import dispatch

CHECKS = []
SUIT = (194, 90, 60)          # sampled from the astronaut's orange suit


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def astronaut():
    return Image.fromarray(data.astronaut())


@check('oklab selects far less of a photograph than the hue window, in far fewer pieces')
def t1():
    from scipy import ndimage
    img = astronaut()
    _, hsv = photo.isolate_colour(img, SUIT)
    _, ok = photo.isolate_colour(img, SUIT, space='oklab')
    fh, fo = (hsv > 0.5).mean(), (ok > 0.5).mean()
    ch = ndimage.label(hsv > 0.5)[1]
    co = ndimage.label(ok > 0.5)[1]
    return (fo < 0.5 * fh and co < 0.5 * ch,
            f'hsv {fh:.1%} in {ch} components, oklab {fo:.1%} in {co}')


@check('the oklab mask is soft by distance, not a hard cut: partial weights exist')
def t2():
    _, ok = photo.isolate_colour(astronaut(), SUIT, space='oklab')
    partial = float(((ok > 0.02) & (ok < 0.98)).mean())
    return partial > 0.01, f'{partial:.1%} of pixels carry a partial weight'


@check('replace_colour honours space and still recolours something')
def t3():
    img = astronaut()
    out, mask = photo.replace_colour(img, SUIT, (60, 120, 200), space='oklab')
    changed = int((np.array(out) != np.array(img)).any(axis=2).sum())
    _, hsv_mask = photo.replace_colour(img, SUIT, (60, 120, 200))
    return (changed > 1000 and mask.mean() < hsv_mask.mean(),
            f'{changed} px changed, mask mean {mask.mean():.3f} vs hsv {hsv_mask.mean():.3f}')


@check("a bare hue with space='oklab' is refused with a usable message, not silently reused")
def t4():
    try:
        photo.isolate_colour(astronaut(), 0.05, space='oklab')
        return False, 'no exception -- a hue was accepted as a colour'
    except ValueError as e:
        return 'sample_colour' in str(e), str(e)[:70]


@check('an unknown space raises rather than falling back to hsv')
def t5():
    for fn, args in ((photo.isolate_colour, (astronaut(), SUIT)),
                     (photo.replace_colour, (astronaut(), SUIT, (0, 0, 255)))):
        try:
            fn(*args, space='cielab')
            return False, f'{fn.__name__} accepted an unknown space'
        except ValueError:
            pass
    return True, 'both refused'


@check('dispatch passes space through to isolate_colour, replace_colour and the preview')
def t6():
    img = astronaut()
    notes = {}
    for space in ('hsv', 'oklab'):
        notes[space] = dispatch('preview_colour_mask',
                                {'target_rgb': list(SUIT), 'space': space}, img).note
        for tool, args in (('isolate_colour', {'target_rgb': list(SUIT)}),
                           ('replace_colour', {'from_rgb': list(SUIT),
                                               'to_rgb': [60, 120, 200]})):
            note = dispatch(tool, dict(args, space=space), img)[1]
            if space not in note:
                return False, f'{tool} note does not say which space was used: {note}'
    return notes['hsv'] != notes['oklab'], 'both spaces reachable, notes differ'


@check('the preview tells a model to try oklab when a hue window takes half the photo')
def t7():
    note = dispatch('preview_colour_mask', {'target_rgb': list(SUIT)}, astronaut()).note
    return "space='oklab'" in note, note[-70:]


@check('a neutral target is selectable in oklab and unservable by a hue window')
def t8():
    # White, grey and black have no meaningful hue, so a hue window cannot separate them at
    # any tolerance; the saturation floor throws them out first.
    a = np.full((200, 300, 3), 120.0)
    a[40:160, 20:100] = 245                       # the white target
    a[40:160, 110:190] = 150                      # a light grey distractor
    a[40:160, 200:280] = 25                       # a near-black distractor
    img = Image.fromarray(a.astype(np.uint8))
    truth = np.zeros((200, 300), bool)
    truth[40:160, 20:100] = True

    def iou(mask):
        return float((mask & truth).sum() / max((mask | truth).sum(), 1))

    _, ok_mask = photo.isolate_colour(img, (245, 245, 245), space='oklab', oklab_tol=0.02)
    best_hsv = max(iou(photo.isolate_colour(img, (245, 245, 245), hue_tol=t)[1] > 0.5)
                   for t in (0.02, 0.07, 0.14, 0.28))
    return (iou(ok_mask > 0.5) > 0.95 and best_hsv < 0.05,
            f'oklab IoU {iou(ok_mask > 0.5):.2f}, best hsv over four tolerances {best_hsv:.2f}')


def run():
    ok = 0
    for name, fn in CHECKS:
        try:
            passed, detail = fn()
        except Exception as e:
            passed, detail = False, f'{type(e).__name__}: {e}'
        print(f'  {"PASS" if passed else "FAIL"}  {name}  [{detail}]')
        ok += bool(passed)
    print(f'\n{ok}/{len(CHECKS)} colour-space checks passed')
    return ok == len(CHECKS)


if __name__ == '__main__':
    import sys
    sys.exit(0 if run() else 1)


def test_colour_space_checks():
    assert run(), 'see printed output for which colour-space check failed'
