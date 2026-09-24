"""Which colour space should select the pixels: HSV's hue window, or OKLab distance?

`photo.isolate_colour`/`replace_colour` take `space='hsv' | 'oklab'`. This eval tests whether
either wins everywhere -- it doesn't, which is why HSV stays the default and OKLab is opt-in.

Ground truth is constructed, never hand-drawn: for the chart, the same figure rendered twice
with one series recoloured (the diff is the series); for the shaded object, a synthesised
image with a known mask. Each space is given its best tolerance, found by sweeping (a fixed
default pair would decide the result in advance). On real photographs there's no ground
truth, so each space is tuned to the same selection size and the result is matched fraction,
fragmentation, and mask overlays -- the numbers rank, only the pictures judge.

Run:  python evals/colour_spaces.py [outdir]
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figsurgeon import perceptual, photo                                # noqa: E402
from figsurgeon.locate import mask_overlay                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_corpus')

HSV_TOLS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.14, 0.20, 0.28]
OKLAB_TOLS = [0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.22, 0.30]


# --------------------------------------------------------------------------- masks
def select(img, colour, space, tol):
    """The mask each space would use, as the editors use it -- not a reimplementation."""
    if space == 'hsv':
        _, soft = photo.isolate_colour(img, tuple(colour), hue_tol=tol)
    else:
        _, soft = photo.isolate_colour(img, tuple(colour), space='oklab', oklab_tol=tol)
    return soft > 0.5


def components(mask):
    from scipy import ndimage
    return int(ndimage.label(mask)[1])


def iou(mask, truth):
    union = (mask | truth).sum()
    return float((mask & truth).sum() / union) if union else 0.0


def best_by_iou(img, colour, space, truth):
    """The space's best shot: the tolerance in its sweep with the highest IoU."""
    tols = HSV_TOLS if space == 'hsv' else OKLAB_TOLS
    rows = []
    for tol in tols:
        m = select(img, colour, space, tol)
        inter = (m & truth).sum()
        rows.append({'tol': tol, 'iou': iou(m, truth),
                     'recall': float(inter / max(truth.sum(), 1)),
                     'precision': float(inter / max(m.sum(), 1)),
                     'mask': m})
    return max(rows, key=lambda r: r['iou'])


def at_fraction(img, colour, space, target_frac):
    """The tolerance whose selection is closest in size to `target_frac` of the frame:
    removes the trivial way one space can look better by simply selecting less of the image.
    """
    tols = HSV_TOLS if space == 'hsv' else OKLAB_TOLS
    best = None
    for tol in tols:
        m = select(img, colour, space, tol)
        gap = abs(m.mean() - target_frac)
        if best is None or gap < best[0]:
            best = (gap, tol, m)
    return {'tol': best[1], 'mask': best[2], 'fraction': float(best[2].mean()),
            'components': components(best[2]),
            # A budget the sweep never reaches is itself the result -- "HSV cannot select
            # this object at ANY tolerance" is a stronger statement than a fragmentation
            # count, and silently comparing a 24 % mask against a 12 % one would hide it.
            'reached': bool(abs(best[2].mean() - target_frac) <= 0.25 * target_frac),
            'edge': best[1] in (tols[0], tols[-1])}


# --------------------------------------------------------------- constructed cases
def chart_case():
    """A matplotlib chart rendered twice; the diff IS the series, exactly."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def render(series_colour):
        rng = np.random.default_rng(7)
        x = np.arange(40)
        fig, ax = plt.subplots(figsize=(5, 3), dpi=110)
        ax.plot(x, np.cumsum(rng.normal(size=40)) + 12, color=series_colour, lw=2,
                label='treated')
        ax.plot(x, np.cumsum(rng.normal(size=40)) + 4, color='#4c72b0', lw=2, label='control')
        ax.plot(x, np.cumsum(rng.normal(size=40)) - 4, color='#55a868', lw=2, label='sham')
        ax.set_title('outcome by arm')
        ax.legend(loc='upper left')
        fig.tight_layout()
        fig.canvas.draw()
        a = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return a

    target = (0xC4, 0x4E, 0x52)          # a flat, saturated series red
    a = render('#c44e52')
    b = render('#7f3fbf')                # same figure, that one series in violet
    truth = (a != b).any(axis=2)
    return Image.fromarray(a), target, truth, 'chart: one flat series among three'


def shaded_object_case():
    """A shaded object, plus a same-hue distractor that a hue window cannot reject: the disc
    carries a lightness ramp (shading), and the distractor shares its hue exactly while
    being darker and less saturated -- a different colour, invisible to `dh < hue_tol`.
    """
    H, W = 260, 360
    bg = np.full((H, W, 3), (208, 212, 216), float)
    yy, xx = np.mgrid[0:H, 0:W]
    disc = (xx - 120) ** 2 / 78 ** 2 + (yy - 130) ** 2 / 95 ** 2 <= 1.0
    base = np.array([194, 90, 60], float)
    ramp = np.clip(0.45 + 0.70 * (xx - 42) / 156.0, 0.45, 1.15)
    bg[disc] = np.clip(base * ramp[disc][:, None], 0, 255)

    import colorsys
    h, s, v = colorsys.rgb_to_hsv(*(base / 255.0))
    dark = np.array(colorsys.hsv_to_rgb(h, s * 0.45, v * 0.32)) * 255
    bar = (xx > 250) & (xx < 330) & (yy > 60) & (yy < 200)
    bg[bar] = dark
    return (Image.fromarray(bg.astype(np.uint8)), tuple(int(v) for v in base), disc,
            'synthetic: shaded disc + same-hue dark distractor')


def neutral_case():
    """Three neutral patches: a white target, a light grey and a near-black distractor.

    The counter-example to `lightness_weight`: at weight 0 (which wins every other case
    here), white and black are the SAME colour, so selecting white takes the whole frame.
    """
    H, W = 200, 300
    a = np.full((H, W, 3), 120.0)
    a[40:160, 20:100] = 245
    a[40:160, 110:190] = 150
    a[40:160, 200:280] = 25
    truth = np.zeros((H, W), bool)
    truth[40:160, 20:100] = True
    return (Image.fromarray(a.astype(np.uint8)), (245, 245, 245), truth,
            'neutral: white target, grey and black distractors')


def lightness_sweep(cases, photos):
    """Set `lightness_weight` from measurement instead of from assertion: each weight gets
    its best tolerance on each case, as everywhere else here.
    """
    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    print('LIGHTNESS WEIGHT  (best IoU over the tolerance sweep, per constructed case)\n')
    print('    ' + ' '.join(f'lw={w:<4}' for w in weights))
    totals = {w: 0.0 for w in weights}
    for img, colour, truth, label in cases:
        row = []
        for w in weights:
            best = max(iou(perceptual.colour_mask(img, colour, tolerance=t,
                                                  lightness_weight=w) > 0.5, truth)
                       for t in OKLAB_TOLS)
            totals[w] += best
            row.append(f'{best:.2f}')
        print('    ' + ' '.join(f'{v:<7}' for v in row) + f' {label}')
    print('    ' + ' '.join(f'{totals[w] / len(cases):<7.2f}' for w in weights) + ' MEAN')
    print()
    print('    the same weights on the photographs, tuned to each budget '
          '(% of frame / mask components)\n')
    for img, colour, budget, label in photos:
        row = []
        for w in weights:
            tol = min(((abs((perceptual.colour_mask(img, colour, tolerance=t,
                                                    lightness_weight=w) > 0.5).mean() - budget), t)
                       for t in OKLAB_TOLS))[1]
            m = perceptual.colour_mask(img, colour, tolerance=tol, lightness_weight=w) > 0.5
            row.append(f'{m.mean():.0%}/{components(m)}')
        print('    ' + ' '.join(f'{v:<7}' for v in row) + f' {label}')
    print()


# --------------------------------------------------------------------------- report
def photo_cases():
    """(image, target colour, budget, label) -- real images, no ground truth."""
    out = []
    from skimage import data
    astro = Image.fromarray(data.astronaut())
    out.append((astro, photo.sample_colour(astro, (200, 300, 320, 420)), 0.12,
                'astronaut: orange suit, against warm face/hair/flag'))
    # Boxes are read off the 900 px thumbnail BY LOOKING at it, and they sample the target
    # colour only -- they are not the edit region and nothing is scored against them. A
    # centre-of-frame default was tried first and is worthless: on the flower it sampled the
    # near-black disc floret (26, 17, 14), so both spaces were faithfully compared on a
    # colour nobody would ask for. `budget` is likewise an eyeballed size for the object
    # meant, not a measurement.
    for name, box, budget, label in [
            ('flower_macro.jpg', (140, 400, 200, 450), 0.45,
             'flower macro: red petals, gradient into orange and yellow'),
            ('car_red.jpg', (460, 620, 540, 640), 0.02,
             'red car, small in a busy street scene'),
            ('product_shoe.jpg', (370, 190, 410, 230), 0.05,
             'red shoes on black, with white and a printed sheet')]:
        path = os.path.join(CORPUS, name)
        if not os.path.exists(path):
            continue
        img = Image.open(path).convert('RGB')
        img.thumbnail((900, 900))
        out.append((img, photo.sample_colour(img, box), budget, label))
    return out


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    wins = {'hsv': 0, 'oklab': 0}

    cases = (chart_case(), shaded_object_case(), neutral_case())
    photos = photo_cases()

    print('GROUND TRUTH CASES  (best tolerance per space, by IoU)\n')
    for img, colour, truth, label in cases:
        print(f'  {label}   target rgb {tuple(colour)}, truth {truth.mean():.1%} of frame')
        res = {}
        for space in ('hsv', 'oklab'):
            r = best_by_iou(img, colour, space, truth)
            res[space] = r
            print(f'    {space:<6} IoU {r["iou"]:.3f}  precision {r["precision"]:.3f}  '
                  f'recall {r["recall"]:.3f}  at tol {r["tol"]}')
            mask_overlay(img, r['mask'].astype(float)).save(
                os.path.join(outdir, f'{label.split(":")[0]}_{space}.png'))
        better = max(res, key=lambda s: res[s]['iou'])
        wins[better] += 1
        margin = res[better]['iou'] - res['hsv' if better == 'oklab' else 'oklab']['iou']
        print(f'    -> {better} by {margin:.3f} IoU\n')

    lightness_sweep(cases, photos)

    print('REAL PHOTOGRAPHS  (no ground truth; each space tuned to the same selection size)\n')
    for img, colour, budget, label in photos:
        print(f'  {label}   target rgb {tuple(colour)}, budget {budget:.0%}')
        for space in ('hsv', 'oklab'):
            r = at_fraction(img, colour, space, budget)
            flag = '' if r['reached'] else '  <- BUDGET UNREACHABLE at any tolerance swept'
            if r['edge'] and not r['reached']:
                flag += ' (pinned to the end of the sweep)'
            print(f'    {space:<6} took {r["fraction"]:.1%} in {r["components"]:>5} '
                  f'components at tol {r["tol"]}{flag}')
            mask_overlay(img, r['mask'].astype(float)).save(
                os.path.join(outdir, f'{label.split(":")[0]}_{space}.png'))
        print()

    print(f'ground-truth cases won: hsv {wins["hsv"]}, oklab {wins["oklab"]}')
    print(f'\noverlays written to {outdir}/ -- fragmentation counts rank the masks, but '
          f'only looking at them says whether the selection followed the object.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else
                  os.path.join(ROOT, 'eval_out_colour_spaces')))
