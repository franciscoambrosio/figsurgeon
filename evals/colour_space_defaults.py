"""Should `space='oklab'` become the default for isolate_colour / replace_colour / preview?

`evals/colour_spaces.py` answers a different question: given each space its best tolerance,
which can select the object at all (OKLab wins 3-0)? That's the wrong way to choose a
default, since a default is used at its own default tolerance (HSV 0.07 hue fraction, OKLab
0.06 perceptual distance) -- flipping `space` flips both at once.

This eval measures the flip as a caller would experience it, and looks for the case that
blocks it: a translucent chart confidence band (swept over four opacities) and a grey chart
series, alongside real charts (exact-RGB ground truth) and real photographs (no ground
truth -- matched fraction, fragmentation, and side-by-side pictures).

Run:  python evals/colour_space_defaults.py [outdir]
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import colour_spaces as CS                                              # noqa: E402
from figsurgeon import photo                                           # noqa: E402
from figsurgeon.locate import mask_overlay                             # noqa: E402

CORPUS = os.path.join(HERE, '_corpus')

# The two numbers the flip actually changes, from the signatures in photo.py.
HSV_DEFAULT = 0.07
OKLAB_DEFAULT = 0.06


def side_by_side(images, labels, width=520):
    """One sheet, because the verdict here is what the edit LOOKS like."""
    from PIL import ImageDraw
    scaled = []
    for im in images:
        im = im.convert('RGB').copy()
        im.thumbnail((width, width))
        scaled.append(im)
    h = max(im.height for im in scaled)
    sheet = Image.new('RGB', (sum(im.width for im in scaled), h + 22), 'white')
    x = 0
    draw = ImageDraw.Draw(sheet)
    for im, label in zip(scaled, labels):
        sheet.paste(im, (x, 22))
        draw.text((x + 4, 6), label, fill='black')
        x += im.width
    return sheet


# --------------------------------------------------------------- new constructed cases
def chart_band_case(alpha=0.4):
    """A series drawn as a line PLUS its translucent confidence band, rendered twice.

    A reader sees one series in one colour; a hue window keeps the band along with the line,
    while a perceptual distance -- correctly, by its own definition -- calls it a different
    colour. `alpha` is swept in `band_alpha_sweep` rather than trusted at one value.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def render(series_colour):
        rng = np.random.default_rng(7)
        x = np.arange(40)
        y = np.cumsum(rng.normal(size=40)) + 12
        fig, ax = plt.subplots(figsize=(5, 3), dpi=110)
        ax.plot(x, y, color=series_colour, lw=2, label='treated')
        ax.fill_between(x, y - 2.5, y + 2.5, color=series_colour, alpha=alpha, lw=0)
        ax.plot(x, np.cumsum(rng.normal(size=40)) + 4, color='#4c72b0', lw=2, label='control')
        ax.set_title('outcome with 95 % band')
        ax.legend(loc='upper left')
        fig.tight_layout()
        fig.canvas.draw()
        a = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return a

    a, b = render('#c44e52'), render('#7f3fbf')
    return (Image.fromarray(a), (0xC4, 0x4E, 0x52), (a != b).any(axis=2),
            'chart band: one series as a line plus its translucent fill')


def band_alpha_sweep():
    """How much of the translucent-band case is real vs. an artifact of the one alpha
    chosen.

    HSV's automatic saturation floor is set from the image's own distribution, so a band
    fainter than that floor is dropped by both spaces regardless. Prints the floor and the
    band's own median saturation next to the verdict so which regime each row is in is
    visible.
    """
    from figsurgeon.photo import _to_hsv
    print('THE BLOCKING CASE, SWEPT  (translucent band, by how translucent it is)\n')
    print('    alpha  hsv    oklab  auto sat_floor  band median sat')
    for alpha in (0.15, 0.25, 0.4, 0.6):
        img, colour, truth, _ = chart_band_case(alpha)
        _, s, _ = _to_hsv(np.array(img).astype(float))
        nontrivial = s[s > 0.02]
        floor = float(np.clip(np.percentile(nontrivial, 15), 0.03, 0.18))
        ious = [CS.iou(CS.select(img, colour, space, tol), truth)
                for space, tol in (('hsv', HSV_DEFAULT), ('oklab', OKLAB_DEFAULT))]
        print(f'    {alpha:<6} {ious[0]:<6.2f} {ious[1]:<6.2f} {floor:<15.3f} '
              f'{float(np.median(s[truth])):.3f}')
    print()


def chart_grey_series_case():
    """The reverse counter-example: a chart whose series is grey. No hue window can address
    a neutral at any tolerance, and real charts do have grey series (`chart_health.png`).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def render(series_colour):
        rng = np.random.default_rng(3)
        x = np.arange(6)
        fig, ax = plt.subplots(figsize=(5, 3), dpi=110)
        ax.bar(x - 0.22, rng.integers(20, 90, 6), width=0.2, color='#4c72b0', label='ley')
        ax.bar(x, rng.integers(20, 90, 6), width=0.2, color=series_colour, label='fr')
        ax.bar(x + 0.22, rng.integers(20, 90, 6), width=0.2, color='#dd8452', label='u5im')
        ax.set_title('indicators by year')
        ax.legend(loc='upper right')
        fig.tight_layout()
        fig.canvas.draw()
        a = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return a

    a, b = render('#d9d9d9'), render('#7f3fbf')
    return (Image.fromarray(a), (0xD9, 0xD9, 0xD9), (a != b).any(axis=2),
            'chart grey: a neutral series among coloured ones')


# ------------------------------------------------------------------ real chart truth
REAL_CHARTS = [
    ('chart_bar.png', (102, 127, 170), 'the blue "Industrial" series',
     [(153, 78, 86), (111, 147, 96), (188, 82, 43)]),
    ('chart_bar.png', (153, 78, 86), 'the red "Artisanal" series',
     [(102, 127, 170), (111, 147, 96), (188, 82, 43)]),
    ('chart_health.png', (217, 217, 217), 'the GREY "FR" series',
     [(237, 125, 49), (68, 114, 196), (255, 192, 0), (91, 155, 213)]),
]


def exact(a, rgb):
    return (a == np.array(rgb, dtype=a.dtype)).all(axis=-1)


def named_colour_path(outdir):
    """"colour pop the red" -- the instruction typed at the plain-language front-end.

    `photo_describe` resolves a colour word to one fixed RGB, which a hue window matches
    coarsely but a tight perceptual distance (0.06) mostly misses in real photographs --
    under a flipped default this class of instruction becomes a near no-op.
    """
    from figsurgeon.photo_describe import COLOUR_WORDS
    target = COLOUR_WORDS['red']
    print(f'\nA COLOUR NAME AS THE TARGET  ("colour pop the red" -> rgb {target})\n')
    print('    % of frame matched, by space, on each corpus photograph\n')
    for name in sorted(os.listdir(CORPUS)) if os.path.isdir(CORPUS) else []:
        if not name.lower().endswith(('.jpg', '.png')):
            continue
        img = Image.open(os.path.join(CORPUS, name)).convert('RGB')
        img.thumbnail((900, 900))
        fracs = [CS.select(img, target, space, tol).mean()
                 for space, tol in (('hsv', HSV_DEFAULT), ('oklab', OKLAB_DEFAULT))]
        flag = '   <- oklab selects essentially nothing' if fracs[1] < 0.001 else ''
        print(f'    {name:<22} hsv {fracs[0]:6.2%}   oklab {fracs[1]:6.2%}{flag}')
    path = os.path.join(CORPUS, 'car_red.jpg')
    if os.path.exists(path):
        img = Image.open(path).convert('RGB')
        img.thumbnail((900, 900))
        edits = [img] + [photo.isolate_colour(img, target, space=s, hue_tol=HSV_DEFAULT,
                                              oklab_tol=OKLAB_DEFAULT)[0]
                         for s in ('hsv', 'oklab')]
        side_by_side(edits, ['original', 'hsv (default now)',
                             'oklab (proposed)']).save(
            os.path.join(outdir, 'edit_named_red_car_red.png'))
    print()


# --------------------------------------------------------------------------- report
def main(outdir):
    os.makedirs(outdir, exist_ok=True)

    cases = (CS.chart_case(), chart_band_case(), chart_grey_series_case(),
             CS.shaded_object_case(), CS.neutral_case())

    print('AT THE DEFAULTS  (the call a caller makes when they pass no tolerance)\n')
    print(f'    hsv@{HSV_DEFAULT} vs oklab@{OKLAB_DEFAULT}, IoU against constructed truth; '
          f'"best" is each space tuned\n')
    for img, colour, truth, label in cases:
        row = {}
        for space, tol in (('hsv', HSV_DEFAULT), ('oklab', OKLAB_DEFAULT)):
            m = CS.select(img, colour, space, tol)
            row[space] = (CS.iou(m, truth),
                          float((m & truth).sum() / max(truth.sum(), 1)),
                          float((m & truth).sum() / max(m.sum(), 1)))
            mask_overlay(img, m.astype(float)).save(
                os.path.join(outdir, f'default_{label.split(":")[0].replace(" ", "_")}'
                                     f'_{space}.png'))
        best = {s: CS.best_by_iou(img, colour, s, truth) for s in ('hsv', 'oklab')}
        winner = 'oklab' if row['oklab'][0] > row['hsv'][0] else 'hsv'
        print(f'  {label}')
        for space in ('hsv', 'oklab'):
            iou_, rec, prec = row[space]
            print(f'    {space:<6} IoU {iou_:.3f}  recall {rec:.2f}  precision {prec:.2f}'
                  f'   (best {best[space]["iou"]:.3f} at tol {best[space]["tol"]})')
        print(f'    -> at the defaults: {winner} by '
              f'{abs(row["oklab"][0] - row["hsv"][0]):.3f} IoU\n')

    band_alpha_sweep()

    print('IS THERE ONE OKLAB TOLERANCE THAT SERVES THEM ALL?  (IoU per case)\n')
    tols = [0.03, 0.04, 0.06, 0.08, 0.10, 0.14]
    print('    ' + ' '.join(f'{t:<6}' for t in tols))
    for img, colour, truth, label in cases:
        row = [CS.iou(CS.select(img, colour, 'oklab', t), truth) for t in tols]
        print('    ' + ' '.join(f'{v:<6.2f}' for v in row) + f' {label}')
    print('    ' + ' '.join(f'{min(CS.iou(CS.select(img, colour, "oklab", t), truth) for img, colour, truth, _ in cases):<6.2f}' for t in tols) + ' WORST CASE')
    print()

    print('REAL CHARTS  (recall on the series\' exact colour; leak into the other series)\n')
    for name, target, label, others in REAL_CHARTS:
        path = os.path.join(CORPUS, name)
        if not os.path.exists(path):
            print(f'  {name} missing -- run evals/real_world.py once to fetch the corpus')
            continue
        img = Image.open(path).convert('RGB')
        a = np.array(img)
        core = exact(a, target)
        distractors = np.zeros(core.shape, bool)
        for o in others:
            distractors |= exact(a, o)
        print(f'  {name}: {label}, {core.mean():.1%} of the frame is exactly {target}')
        for space, tol in (('hsv', HSV_DEFAULT), ('oklab', OKLAB_DEFAULT)):
            m = CS.select(img, target, space, tol)
            recall = float((m & core).sum() / max(core.sum(), 1))
            leak = float((m & distractors).sum() / max(distractors.sum(), 1))
            print(f'    {space:<6} took {m.mean():.1%} of the frame, kept {recall:.1%} of '
                  f'the series, and {leak:.1%} of the OTHER series')
            mask_overlay(img, m.astype(float)).save(
                os.path.join(outdir, f'realchart_{name.split(".")[0]}_'
                                     f'{target[0]}_{space}.png'))
        # The edit itself, both ways, because a chart with half a series recoloured is
        # obvious in a picture and invisible in a recall number.
        edits = [img] + [photo.replace_colour(img, target, (60, 60, 60), space=s,
                                              hue_tol=HSV_DEFAULT,
                                              oklab_tol=OKLAB_DEFAULT)[0]
                         for s in ('hsv', 'oklab')]
        side_by_side(edits, ['original', 'hsv (default now)', 'oklab (proposed)']).save(
            os.path.join(outdir, f'edit_{name.split(".")[0]}_{target[0]}.png'))
        print()

    print('REAL PHOTOGRAPHS  (no ground truth: the numbers rank, the pictures judge)\n')
    for img, colour, budget, label in CS.photo_cases():
        print(f'  {label}   target rgb {tuple(colour)}')
        for space, tol in (('hsv', HSV_DEFAULT), ('oklab', OKLAB_DEFAULT)):
            m = CS.select(img, colour, space, tol)
            print(f'    {space:<6} took {m.mean():.1%} of the frame in '
                  f'{CS.components(m):>5} components')
        edits = [img] + [photo.isolate_colour(img, colour, space=s, hue_tol=HSV_DEFAULT,
                                              oklab_tol=OKLAB_DEFAULT)[0]
                         for s in ('hsv', 'oklab')]
        side_by_side(edits, ['original', 'hsv (default now)', 'oklab (proposed)']).save(
            os.path.join(outdir, f'edit_{label.split(":")[0].replace(" ", "_")}.png'))
        print()

    named_colour_path(outdir)

    print(f'sheets written to {outdir}/ -- edit_*.png are the three-up comparisons.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else
                  os.path.join(ROOT, 'eval_out_colour_space_defaults')))
