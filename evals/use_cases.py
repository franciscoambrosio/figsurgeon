"""Whole workflows, not single operations: what people actually ask an image editor for.

`edit_quality.py` tests one operation at a time; real requests are three to six steps deep,
and the failures that matter most only show up in sequence: state carried wrongly between
steps, an operation that's fine alone and wrong in combination, or a mistake several steps
back that must be undone without losing the good steps after it.

Every workflow writes a filmstrip (original + every intermediate state, captioned) since a
single before/after can't show which step went wrong.

Run:  python evals/use_cases.py [outdir]
"""
import os
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figsurgeon import ImageWorkspace                                   # noqa: E402
from figsurgeon.locate import _font                                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- images
def _sk(name):
    from skimage import data
    a = getattr(data, name)()
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    return Image.fromarray(a[:, :, :3].astype('uint8'))


def _low_light():
    """A dark, noisy frame -- the single most common "please fix this" photo. Built with
    known noise/brightness so the rescue's actual recovery can be checked.
    """
    rng = np.random.default_rng(0)
    a = np.array(_sk('coffee')).astype(float) * 0.28
    a += rng.normal(0, 12, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype('uint8'))


def _warm_cast():
    """A scene under tungsten light: a real colour cast over a neutral-ish subject."""
    a = np.array(_sk('chelsea')).astype(float)
    a *= np.array([1.25, 1.0, 0.72])
    return Image.fromarray(np.clip(a, 0, 255).astype('uint8'))


def _scanned_page():
    """A document photographed off-angle: rotated, low contrast, uneven lighting."""
    page = _sk('page').resize((700, 520))
    a = np.array(page).astype(float)
    yy = np.linspace(0.75, 1.15, a.shape[0])[:, None, None]     # uneven lighting
    a = np.clip(a * yy * 0.85 + 40, 0, 255)
    img = Image.fromarray(a.astype('uint8'))
    return img.rotate(-4.5, expand=True, fillcolor=(120, 118, 112))


def _ui_screenshot():
    """Flat UI: sharp edges, solid fills, small text -- nothing photographic to segment."""
    img = Image.new('RGB', (720, 460), (245, 246, 248))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 720, 56], fill=(28, 40, 66))
    d.text((18, 18), 'Dashboard', font=_font(21), fill=(255, 255, 255))
    for i, (label, value) in enumerate([('Revenue', '$41.2k'), ('Users', '1,284'),
                                        ('Churn', '2.1%')]):
        x = 24 + i * 228
        d.rectangle([x, 84, x + 204, 196], fill=(255, 255, 255), outline=(220, 224, 230))
        d.text((x + 16, 104), label, font=_font(15), fill=(120, 128, 140))
        d.text((x + 16, 132), value, font=_font(30), fill=(28, 40, 66))
    d.rectangle([24, 224, 696, 436], fill=(255, 255, 255), outline=(220, 224, 230))
    for i in range(12):                                          # a bar chart
        h = 30 + (i * 37) % 140
        d.rectangle([56 + i * 52, 404 - h, 92 + i * 52, 404], fill=(64, 132, 214))
    return img


IMAGES = {
    'portrait': lambda: _sk('astronaut'),
    'pet': lambda: _sk('chelsea'),
    'product': lambda: _sk('coffee'),
    'scene': lambda: _sk('rocket'),
    'low_light': _low_light,
    'warm_cast': _warm_cast,
    'scanned_page': _scanned_page,
    'ui_screenshot': _ui_screenshot,
    'flat_graphic': lambda: _sk('logo'),
    'texture': lambda: _sk('brick'),
}


# --------------------------------------------------------------------------- assertions
def _has_alpha(img):
    if img.mode != 'RGBA':
        return False, f'mode is {img.mode}, not RGBA -- transparency was lost'
    a = np.array(img)[..., 3]
    frac = float((a < 250).mean())
    return frac > 0.05, f'{frac:.0%} of pixels are transparent'


def _brighter_and_cleaner(before, after):
    """A low-light rescue must add light AND not amplify the grain while doing it."""
    from scipy.ndimage import gaussian_filter
    b = np.array(before.convert('RGB')).astype(float).mean(axis=2)
    a = np.array(after.convert('RGB')).astype(float).mean(axis=2)
    noise_b = float(np.abs(b - gaussian_filter(b, 2)).mean())
    noise_a = float(np.abs(a - gaussian_filter(a, 2)).mean())
    ok = a.mean() > b.mean() * 1.4 and noise_a < noise_b * 1.3
    return ok, (f'brightness {b.mean():.0f} -> {a.mean():.0f}, '
                f'grain {noise_b:.2f} -> {noise_a:.2f}')


def _neutral_greys(before, after):
    """White balance worked if what should be neutral got closer to neutral."""
    def cast(img):
        a = np.array(img.convert('RGB')).astype(float).reshape(-1, 3)
        bright = a[a.mean(axis=1) > np.percentile(a.mean(axis=1), 70)]
        return float(np.abs(bright.mean(axis=0) - bright.mean()).max())
    c0, c1 = cast(before), cast(after)
    return c1 < c0, f'colour cast in the bright tones {c0:.1f} -> {c1:.1f}'


def _text_is_legible(before, after):
    """A document clean-up must raise contrast without crushing the text away."""
    a0 = np.array(before.convert('L')).astype(float)
    a1 = np.array(after.convert('L')).astype(float)
    dark0 = float((a0 < 100).mean())
    dark1 = float((a1 < 100).mean())
    ok = a1.std() > a0.std() * 1.1 and dark1 > dark0 * 0.5
    return ok, (f'contrast {a0.std():.0f} -> {a1.std():.0f}, '
                f'dark (ink) pixels {dark0:.1%} -> {dark1:.1%}')


def _size_is(w, h):
    def check(before, after):
        ok = after.size == (w, h)
        return ok, f'final size {after.size[0]}x{after.size[1]} (wanted {w}x{h})'
    return check


def _unchanged(before, after):
    same = np.array_equal(np.array(before), np.array(after))
    return same, 'byte-identical to the original' if same else 'differs from the original'


# --------------------------------------------------------------------------- workflows
# (id, image, what a user would say, [steps], assertion)
WORKFLOWS = [
    ('ecommerce_cutout', 'product',
     'cut out the cup, put it on white, crop it square and size it for the site',
     [('remove_background', {}),
      ('replace_background', {'colour_rgb': [255, 255, 255]}),
      ('crop', {'box': [100, 0, 500, 400]}),
      ('resize', {'width': 600})],
     _size_is(600, 600)),

    ('cutout_keeps_alpha', 'pet',
     'cut the cat out and brighten it -- the transparency must survive the brighten',
     [('remove_background', {}),
      ('adjust', {'brightness': 1.25}),
      ('stylise', {'effect': 'vignette', 'strength': 0.3})],
     lambda b, a: _has_alpha(a)),

    ('profile_photo', 'portrait',
     'blur the background, brighten her a little, crop to a square headshot',
     [('blur_background', {'focus': 'subject', 'radius': 14}),
      ('adjust', {'brightness': 1.12, 'contrast': 1.08}),
      ('crop', {'box': [150, 40, 420, 310]})],
     _size_is(270, 270)),

    # Order is the scenario: brightening first multiplies grain along with signal, and
    # denoising after can't undo it.
    ('low_light_rescue', 'low_light',
     'this came out dark and grainy -- fix it',
     [('denoise', {'strength': 18}),
      ('adjust', {'brightness': 2.2}),
      ('adjust', {'contrast': 1.15})],
     _brighter_and_cleaner),

    ('remove_the_cast', 'warm_cast',
     'this was shot under a yellow bulb -- neutralise it',
     [('auto_white_balance', {})],
     _neutral_greys),

    ('document_cleanup', 'scanned_page',
     'straighten this scan and make the text crisp',
     [('rotate', {'angle': -4.5, 'expand': False}),
      ('adjust', {'contrast': 1.9, 'brightness': 1.05}),
      ('stylise', {'effect': 'grayscale'})],
     _text_is_legible),

    ('social_post', 'scene',
     'crop it wide, warm it up, and put a caption on it',
     [('crop', {'box': [0, 60, 640, 420]}),
      ('adjust', {'saturation': 1.3, 'contrast': 1.1}),
      ('add_text', {'text': 'LAUNCH DAY', 'position': [24, 24], 'size': 34,
                    'colour_rgb': [255, 255, 255]})],
     _size_is(640, 360)),

    # Undo carrying the whole point: a product-variant workflow where the first colour is
    # rejected. The steps AFTER the undo must build on the good state, not on the reject.
    ('colour_variant_retry', 'portrait',
     'show the suit in green -- no, too dark, make it blue instead',
     [('recolour_object', {'box': [60, 180, 420, 512], 'to_rgb': [20, 70, 30]}),
      ('undo', {}),
      ('recolour_object', {'box': [60, 180, 420, 512], 'to_rgb': [40, 90, 220]})],
     lambda b, a: _suit_is_blue(b, a)),

    # The box must land somewhere GrabCut actually finds something to segment, or
    # erase_object changes zero pixels and reports "object erased (0.0% of frame) |
    # verified" -- a silent no-op passing as success.
    ('cleanup_then_finish', 'scene',
     'get rid of the mast on the right, then add a subtle vignette',
     [('erase_object', {'box': [150, 80, 230, 340]}),
      ('stylise', {'effect': 'vignette', 'strength': 0.35})],
     lambda b, a: _mast_is_gone(b, a)),

    ('poster_treatment', 'pet',
     'grey everything except the eyes and push the contrast',
     [('isolate_object', {'boxes': [[141, 85, 211, 145], [288, 100, 352, 158]],
                          'flatten': 0.6}),
      ('adjust', {'contrast': 1.25})],
     lambda b, a: _eyes_kept_colour(a)),

    # A flat UI has nothing photographic in it. The neural subject mask has no subject to
    # find, so this is where "the tool ran and produced nonsense" would show up.
    ('screenshot_crop', 'ui_screenshot',
     'crop out just the stat cards and double the size for a slide',
     [('crop', {'box': [24, 84, 696, 196]}),
      ('resize', {'scale': 2.0})],
     _size_is(1344, 224)),

    ('flat_graphic_recolour', 'flat_graphic',
     'recolour the logo to brand blue',
     [('replace_colour', {'from_rgb': [220, 30, 30], 'to_rgb': [30, 80, 200]})],
     None),

    ('reset_after_a_mess', 'texture',
     'several edits, then start over -- reset must restore the original exactly',
     [('stylise', {'effect': 'sepia'}),
      ('adjust', {'brightness': 1.8}),
      ('stylise', {'effect': 'vignette'}),
      ('reset', {})],
     _unchanged),
]


def _suit_is_blue(before, after):
    """After the retry, the suit must be BLUE -- not the rejected green, not the original."""
    px = np.array(after.convert('RGB'))[400:460, 150:250].reshape(-1, 3).mean(axis=0)
    ok = px[2] > px[1] + 25 and px[2] > px[0] + 25
    return ok, f'suit RGB {tuple(int(v) for v in px)} (blue channel must dominate)'


def _mast_is_gone(before, after):
    """Measured as structure removed, not variation reduced: local horizontal roughness
    drops sharply when the mast is erased, while overall std dev barely moves since the
    region is mostly smooth sky already.
    """
    def roughness(img):
        a = np.array(img.convert('L')).astype(float)[80:340, 150:230]
        return float(np.abs(np.diff(a, axis=1)).mean())
    r0, r1 = roughness(before), roughness(after)
    return r1 < r0 * 0.5, f'local structure in the erased region {r0:.2f} -> {r1:.2f}'


def _eyes_kept_colour(after):
    def sat(box):
        a = np.array(after.convert('RGB')).astype(float)[box[1]:box[3], box[0]:box[2]]
        mx, mn = a.max(axis=2), a.min(axis=2)
        return float(((mx - mn) / np.maximum(mx, 1e-6)).mean())
    left, right = sat([141, 85, 211, 145]), sat([288, 100, 352, 158])
    ok = left > 0.10 and right > 0.10
    return ok, f'saturation left eye {left:.2f}, right eye {right:.2f}'


# --------------------------------------------------------------------------- running
def filmstrip(states, title, verdict, cell=260):
    """The original plus every intermediate state, captioned. One before/after cannot show
    which step of a six-step workflow went wrong; this can."""
    thumbs = []
    for label, img in states:
        if img.mode == 'RGBA':
            # Composite onto a checkerboard, not black: convert('RGB') would flatten alpha
            # onto black, which looks like the very background-not-removed bug this checks.
            board = Image.new('RGB', img.size, (255, 255, 255))
            d0 = ImageDraw.Draw(board)
            for yy in range(0, img.size[1], 16):
                for xx in range(0, img.size[0], 16):
                    if (xx // 16 + yy // 16) % 2:
                        d0.rectangle([xx, yy, xx + 15, yy + 15], fill=(214, 216, 220))
            board.paste(img, (0, 0), img)
            t = board
        else:
            t = img.convert('RGB').copy()
        t = t.copy()
        t.thumbnail((cell, cell), Image.LANCZOS)
        thumbs.append((label, t))
    W = cell * len(thumbs)
    H = max(t.size[1] for _, t in thumbs) + 96
    sheet = Image.new('RGB', (W, H), (246, 246, 248))
    d = ImageDraw.Draw(sheet)
    for i, (label, t) in enumerate(thumbs):
        x = i * cell
        sheet.paste(t, (x + (cell - t.size[0]) // 2, 62))
        d.text((x + 6, 44), label[:38], font=_font(12), fill=(70, 70, 70))
        if i:
            d.text((x - 8, 62 + t.size[1] // 2), '>', font=_font(20), fill=(150, 150, 150))
    d.text((8, 8), title[:120], font=_font(17), fill=(10, 10, 10))
    colour = (0, 120, 0) if verdict.startswith('OK') else (190, 0, 0)
    d.text((8, 28), verdict[:160], font=_font(13), fill=colour)
    return sheet


def run(outdir):
    os.makedirs(outdir, exist_ok=True)
    cache, rows = {}, []

    for wid, image_name, intent, steps, assertion in WORKFLOWS:
        if image_name not in cache:
            cache[image_name] = IMAGES[image_name]()
        img = cache[image_name]

        ws = ImageWorkspace(img)
        states = [('original', img)]
        problems, t0 = [], time.time()
        try:
            for name, args in steps:
                result = ws.apply(name, args)
                label = f'{name}'
                if 'CHECK FAILED' in result.note:
                    problems.append(f'{name}: {result.note.split("CHECK FAILED:")[1][:70]}')
                    label += ' [FAILED]'
                elif result.note.startswith('error'):
                    problems.append(f'{name}: {result.note[:80]}')
                    label += ' [ERROR]'
                states.append((label, ws.image))
        except Exception as e:
            problems.append(f'RAISED {type(e).__name__}: {e}')
            traceback.print_exc()
        elapsed = time.time() - t0

        detail = ''
        if assertion is not None and not any(p.startswith('RAISED') for p in problems):
            ok, detail = assertion(img, ws.image)
            if not ok:
                problems.append(f'outcome: {detail}')

        status = 'OK' if not problems else 'PROBLEM'
        verdict = f'{status}: {detail}' if not problems else f'PROBLEM: {"; ".join(problems)}'
        filmstrip(states, f'{wid} -- {intent}', verdict).save(
            os.path.join(outdir, f'{wid}.png'))
        rows.append((wid, status, verdict, elapsed))

        print(f'{status:8s} {wid:24s} {len(steps)} steps {elapsed:5.1f}s  {detail[:64]}')
        for p in problems:
            print(f'{"":8s} {"":24s}          -> {p[:100]}')

    bad = [r for r in rows if r[1] != 'OK']
    print(f'\n{len(rows) - len(bad)}/{len(rows)} workflows behaved; strips written to {outdir}')
    return rows


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'eval_out_workflows'))
