"""A PNG with transparency survives an edit -- it is ordinary input, not an edge case.

Every pixel-wise operation starts with `convert('RGB')` for the maths, so restoring alpha
alone is not enough: the colour hidden under transparent pixels must not become visible.
The fixture below is that case exactly: a blue mark on a transparent field whose hidden RGB
is red, which reappears if an operation drops the alpha.
"""
import numpy as np
from PIL import Image

from figsurgeon import photo

CHECKS = []
BLUE = (30, 90, 200)


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def logo():
    a = np.zeros((120, 120, 4), np.uint8)
    a[..., 0], a[..., 1], a[..., 2] = 220, 30, 30    # hidden red under the transparency
    a[30:90, 30:90] = [30, 90, 200, 255]             # the visible mark
    return Image.fromarray(a, 'RGBA')


OPERATIONS = [
    ('adjust', lambda i: photo.adjust(i, brightness=1.3)),
    ('grayscale', photo.grayscale),
    ('sepia', photo.sepia),
    ('sketch', photo.sketch),
    ('vignette', photo.vignette),
    ('isolate_colour hsv', lambda i: photo.isolate_colour(i, BLUE)[0]),
    ('isolate_colour oklab', lambda i: photo.isolate_colour(i, BLUE, space='oklab')[0]),
    ('replace_colour hsv', lambda i: photo.replace_colour(i, BLUE, (20, 180, 60))[0]),
    ('replace_colour oklab',
     lambda i: photo.replace_colour(i, BLUE, (20, 180, 60), space='oklab')[0]),
]


@check('every pixel-wise operation keeps the transparency it was given')
def t_kept():
    img = logo()
    bad = []
    for name, op in OPERATIONS:
        out = op(img)
        if out.mode != 'RGBA' or np.array(out)[0, 0, 3] != 0:
            bad.append(name)
    return not bad, ', '.join(bad) or f'{len(OPERATIONS)} operations kept the alpha channel'


@check('the edit still happens -- transparency is kept, not used to skip the work')
def t_not_vacuous():
    img = logo()
    before = np.array(img.convert('RGB'))[60, 60]
    after = np.array(photo.grayscale(img).convert('RGB'))[60, 60]
    return not np.array_equal(before, after), f'visible mark {tuple(before)} -> {tuple(after)}'


@check('an RGB photograph is not turned into RGBA on the way through')
def t_rgb_untouched():
    rgb = Image.fromarray(np.random.default_rng(0).integers(0, 255, (60, 60, 3), dtype=np.uint8))
    modes = {name: op(rgb).mode for name, op in OPERATIONS}
    wrong = [n for n, m in modes.items() if m != 'RGB']
    return not wrong, ', '.join(wrong) or 'all nine stayed RGB'


@check('rotating a transparent image gives transparent corners, not white ones')
def t_rotate_fill():
    """`fillcolor=(255,255,255)` on an RGBA image fills alpha 255 too, which would box a
    rotated logo in opaque white instead of transparency."""
    corner = np.array(photo.crop_and_rotate(logo(), angle=15))[0, 0]
    rgb = Image.fromarray(np.full((60, 60, 3), 128, np.uint8))
    rgb_corner = tuple(np.array(photo.crop_and_rotate(rgb, angle=15))[0, 0])
    return (corner[3] == 0 and rgb_corner == (255, 255, 255),
            f'RGBA corner alpha {corner[3]}, RGB corner {rgb_corner}')


def run():
    ok = 0
    for name, fn in CHECKS:
        try:
            passed, detail = fn()
        except Exception as e:
            passed, detail = False, f'{type(e).__name__}: {e}'
        print(f'  {"PASS" if passed else "FAIL"}  {name}  [{detail}]')
        ok += bool(passed)
    print(f'\n{ok}/{len(CHECKS)} alpha checks passed')
    return ok == len(CHECKS)


if __name__ == '__main__':
    import sys
    sys.exit(0 if run() else 1)


def test_alpha_is_preserved():
    assert run(), 'see printed output for which alpha check failed'
