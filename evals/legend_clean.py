"""Cleaning a legend's bleed-through, scored against ground truth.

    python evals/legend_clean.py          # needs matplotlib; fetches 2 figures once

Ground truth: the same legend rendered twice over the same figure, at `framealpha=0.75` and
at `1.0`. The opaque render is what a correct clean should produce pixel for pixel, so "did
it work" is an RGB error, not a matter of opinion.

Mean |RGB| error inside the box against the opaque render:

    figure     uncleaned   as shipped    now
    exo           3.7         2.2        1.8
    gcd          51.1        14.3        5.2
    ackley       43.1       158.0        8.3      <- "cleaning" made it 3.7x worse
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from evals import chart_remap                                            # noqa: E402
from figsurgeon import advanced, verify_photo                           # noqa: E402

SHEETS = os.path.join(HERE, 'out_legend_clean')

# The figure each legend is drawn over, and the Commons title to fetch it by. `chart_remap`
# already caches two of these; the scatter is fetched the same way.
BASES = [
    ('gcd', 'Heatmap of GCD Matrix.png', 1200, 'a viridis heatmap: hard bleed, dark cells'),
    ('ackley', 'Ackley 2d.png', 1200, 'a viridis contour plot: the case that went black'),
    ('exo', 'Exoplanet distance mass relation by planet pairs inner to outer 1.png', 1200,
     'a scatter chart: light behind the legend, where the code always worked'),
]


def rendered(base_path, alpha, out_path):
    """The same legend over the same figure at `alpha`. Returns its box in image pixels.

    matplotlib is used as a renderer here, not as the thing under test: it draws a legend
    with the bleed-through a real chart has, and at alpha=1.0 it draws the answer.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    img = np.asarray(Image.open(base_path).convert('RGB'))
    h, w = img.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_axis_off()
    handles = [Line2D([0], [0], color=c, lw=3, label=l) for c, l in
               (('#1f77b4', 'Observed'), ('#d62728', 'Model'), ('#2ca02c', 'Residual'))]
    leg = ax.legend(handles=handles, loc='center', framealpha=alpha, fontsize=14,
                    facecolor='white')
    fig.canvas.draw()
    bb = leg.get_window_extent()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return (int(bb.x0), int(h - bb.y1), int(np.ceil(bb.x1)), int(np.ceil(h - bb.y0)))


def error(got, truth, box):
    """Mean and share-off-by-more-than-12 inside the box, against the opaque render."""
    x0, y0, x1, y1 = box
    d = np.abs(np.asarray(got).astype(int)[y0:y1, x0:x1]
               - np.asarray(truth).astype(int)[y0:y1, x0:x1]).max(axis=2)
    return d.mean(), (d > 12).mean()


def swatch_survival(got, truth, box):
    """Share of the legend's own swatch pixels that came through unharmed.

    The swatches are read off the opaque render -- the one place they are unambiguous --
    which is what makes this measurable at all.
    """
    x0, y0, x1, y1 = box
    T = np.asarray(truth).astype(int)[y0:y1, x0:x1]
    G = np.asarray(got).astype(int)[y0:y1, x0:x1]
    sw = (T.max(axis=2) - T.min(axis=2)) > 70
    if not sw.any():
        return float('nan'), 0
    return float((np.abs(G[sw] - T[sw]).max(axis=1) < 40).mean()), int(sw.sum())


def run():
    os.makedirs(SHEETS, exist_ok=True)
    for name, title, width, what in BASES:
        base = chart_remap.image(name if name != 'exo' else 'exo_scatter', title, width)
        base_path = os.path.join(SHEETS, f'{name}_base.png')
        base.save(base_path)
        dirty_path = os.path.join(SHEETS, f'{name}_bleeding.png')
        truth_path = os.path.join(SHEETS, f'{name}_opaque.png')
        box = rendered(base_path, 0.75, dirty_path)
        rendered(base_path, 1.0, truth_path)
        dirty = Image.open(dirty_path).convert('RGB')
        truth = Image.open(truth_path).convert('RGB')

        cleaned, _ = advanced.clean_transparent_box(dirty, box)
        cleaned.save(os.path.join(SHEETS, f'{name}_cleaned.png'))
        mb, fb = error(dirty, truth, box)
        ma, fa = error(cleaned, truth, box)
        kept, n = swatch_survival(cleaned, truth, box)
        ok, detail = verify_photo.check_transparent_box_cleaned(dirty, cleaned, box)
        verdict = {True: 'verified', False: 'FAILED', None: 'no verdict'}[ok]

        print(f'{name:7s} {what}')
        print(f'{"":7s} box {box}; error vs the opaque render: uncleaned mean {mb:5.1f} '
              f'({fb:5.1%} of px off by >12) -> cleaned mean {ma:5.1f} ({fa:5.1%})')
        print(f'{"":7s} swatch pixels kept {kept:.1%} of {n};  {verdict}: {detail}\n')

        # bleeding | cleaned | opaque, at 2x, cropped to the legend with a margin
        x0, y0, x1, y1 = box
        p = 14
        crop = (x0 - p, y0 - p, x1 + p, y1 + p)
        w, h = crop[2] - crop[0], crop[3] - crop[1]
        sheet = Image.new('RGB', (w * 3 + 24, h), 'white')
        for i, im in enumerate((dirty, cleaned, truth)):
            sheet.paste(im.crop(crop), (i * (w + 12), 0))
        sheet.resize(((w * 3 + 24) * 2, h * 2), Image.NEAREST).save(
            os.path.join(SHEETS, f'{name}_triple.png'))

    print(f'sheets in {SHEETS}: each _triple is bleeding | cleaned | opaque at 2x. The '
          f'middle panel should be the right one.')


if __name__ == '__main__':
    run()
