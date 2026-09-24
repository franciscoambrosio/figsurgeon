"""An adjustment is graded against the factor that was asked for, not a fixed 5 % bar,
which fails correct small edits and passes edits that under-deliver on a large one.

Each check below is paired: a correct edit that must pass, and a broken one on the same
image that must fail, so a check that only ever fails cannot pass unnoticed.
"""
import numpy as np
from PIL import Image
from skimage import data

from figsurgeon import photo
from figsurgeon.verify_photo import _predict_adjust, _ADJUST_STAT
from figsurgeon.workspace import verify

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def astro():
    return Image.fromarray(data.astronaut())


def white_ish():
    """A near-white frame: almost no headroom to get brighter, plenty to get darker."""
    a = np.full((200, 200, 3), 240, np.uint8)
    a[50:150, 50:150] = 200
    return Image.fromarray(a)


def v(img, args, out):
    return verify('adjust', args, img, out)


@check('a correct 2 % lift passes -- the old fixed 5 % bar failed it')
def t_small_true():
    img = astro()
    ok, detail = v(img, {'brightness': 1.02}, photo.adjust(img, brightness=1.02))
    return ok is True, detail


@check('a correct 2 % dim passes -- 0.98 could never clear the old bar in either direction')
def t_small_down():
    img = astro()
    ok, detail = v(img, {'brightness': 0.98}, photo.adjust(img, brightness=0.98))
    return ok is True, detail


@check('a no-op fails even though the request was small')
def t_small_noop():
    img = astro()
    ok, detail = v(img, {'brightness': 1.02}, img.copy())
    return ok is False, detail


@check('asking x1.5 and delivering x1.05 fails')
def t_under_delivers():
    img = astro()
    ok, detail = v(img, {'brightness': 1.5}, photo.adjust(img, brightness=1.05))
    return ok is False, detail


@check('an edit in the wrong direction fails')
def t_wrong_way():
    img = astro()
    ok, detail = v(img, {'contrast': 1.5}, photo.adjust(img, contrast=0.7))
    return ok is False, detail


@check('an unreachable factor passes, and says how much was reachable')
def t_headroom():
    """x4 on a near-white frame cannot brighten by 4x, so this passes with the reachable
    headroom stated rather than failing against the literal target."""
    img = white_ish()
    ok, detail = v(img, {'brightness': 4.0}, photo.adjust(img, brightness=4.0))
    return ok is True and 'reachable before clipping' in detail, detail


@check('a request below the 8-bit noise floor is inconclusive, not failed')
def t_noise_floor():
    """A change smaller than 8-bit rounding noise is unmeasurable, so the verdict must be
    inconclusive rather than a failure."""
    rng = np.random.default_rng(0)
    a = np.clip(128 + rng.normal(0, 25, (200, 200, 1)) + rng.normal(0, 1.5, (200, 200, 3)),
                0, 255).astype(np.uint8)
    img = Image.fromarray(a)
    ok, detail = v(img, {'saturation': 1.02}, photo.adjust(img, saturation=1.02))
    return ok is None and 'too small to verify' in detail, detail


@check('an impossible request fails, and is told apart from an unmeasurable one')
def t_impossible():
    """Saturating a greyscale photograph returns it untouched. This must fail rather than
    read as inconclusive, so the check pushes the operation to its extreme and asks whether
    the image can move at all, rather than telling the caller to retry with a bigger number."""
    from skimage import data
    grey = Image.fromarray(np.stack([data.camera()] * 3, axis=-1))
    ok, detail = v(grey, {'saturation': 1.8}, photo.adjust(grey, saturation=1.8))
    return ok is False and 'greyscale' in detail, detail


@check('a fully clipped image cannot be brightened further, and says so')
def t_clipped():
    img = Image.fromarray(np.full((80, 80, 3), 255, np.uint8))
    ok, detail = v(img, {'brightness': 2.0}, photo.adjust(img, brightness=2.0))
    return ok is False and 'already at the limit' in detail, detail


@check('an unchecked adjustment is named, not denied')
def t_sharpness_message():
    """`sharpness` has no check behind it, so the note must name it rather than say "no
    measurable adjustment requested" about a call that changed the image."""
    img = astro()
    ok, detail = v(img, {'sharpness': 1.8}, photo.adjust(img, sharpness=1.8))
    return ok is None and 'sharpness was adjusted' in detail, detail


@check('the prediction of what a factor can achieve matches PIL to within rounding')
def t_prediction():
    """If this drifts, every verdict above is being measured against the wrong target."""
    img = astro()
    worst = 0.0
    for key in ('brightness', 'contrast', 'saturation'):
        for f in (0.5, 0.98, 1.02, 1.5, 4.0):
            real = np.array(photo.adjust(img, **{key: f})).astype(float)
            pred = _predict_adjust(np.array(img).astype(float), key, f)
            worst = max(worst, float(np.abs(real - pred).max()))
    return worst <= 2.0, f'largest disagreement {worst:.2f}/255 over 15 factor/key pairs'


@check('every statistic is monotonic in the factor, so a fraction of it means something')
def t_monotonic():
    img = astro()
    a0 = np.array(img).astype(float)
    bad = []
    for key, stat in _ADJUST_STAT.items():
        vals = [stat(_predict_adjust(a0, key, f)) for f in (0.5, 0.8, 1.0, 1.3, 1.8)]
        if any(b < a for a, b in zip(vals, vals[1:])):
            bad.append(f'{key}: {[round(x, 2) for x in vals]}')
    return not bad, '; '.join(bad) or 'brightness, contrast and saturation all monotonic'


def run():
    ok = 0
    for name, fn in CHECKS:
        try:
            passed, detail = fn()
        except Exception as e:
            passed, detail = False, f'{type(e).__name__}: {e}'
        print(f'  {"PASS" if passed else "FAIL"}  {name}  [{detail}]')
        ok += bool(passed)
    print(f'\n{ok}/{len(CHECKS)} adjust-threshold checks passed')
    return ok == len(CHECKS)


if __name__ == '__main__':
    import sys
    sys.exit(0 if run() else 1)


def test_adjust_threshold_checks():
    assert run(), 'see printed output for which adjust-threshold check failed'
