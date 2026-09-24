"""Box -> mask on flat-colour figures (charts), not photographs.

    python evals/segmentation_models/synthetic.py        # needs figsurgeon[grounding]

RESULTS.md's photograph comparison favoured SlimSAM; on figures it does not agree: on a
loose box over a flat-colour element, SlimSAM regularly returns the background inside the
box instead of the element, while GrabCut is nearly perfect. Ground truth is constructed
(each figure rendered twice with the target element in two colours; differing pixels are
the element). The box is deliberately loose (~20% pad), matching what a caller draws and
what `locate.refine_box` builds before segmenting.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cases as C                                                        # noqa: E402
import compare                                                           # noqa: E402

from figsurgeon import objects as O                                     # noqa: E402

A, B = '#cc6677', '#117733'          # the two colours each target is rendered in


def _bar(colour):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.bar([0, 1, 2], [3, 5, 2], color=['#4477aa', colour, '#228833'])
    fig.canvas.draw()
    out = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return out


def _pie(colour):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.pie([3, 5, 2, 4], colors=['#4477aa', colour, '#228833', '#ddcc77'])
    fig.canvas.draw()
    out = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return out


def _legend_swatch(colour):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    x = np.linspace(0, 6, 40)
    ax.plot(x, np.sin(x), color='#4477aa', label='alpha')
    ax.plot(x, np.cos(x), color='#888888', label='beta')
    leg = ax.legend(loc='upper right')
    leg.legend_handles[1].set_color(colour)
    fig.canvas.draw()
    out = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return out


def _disc(colour):
    img = Image.new('RGB', (300, 220), (30, 40, 60))
    ImageDraw.Draw(img).ellipse([100, 60, 200, 160], fill=colour)
    return img


CASES = [('disc_on_flat_field', _disc, 'the disc'),
         ('bar_middle', _bar, 'the middle bar'),
         ('pie_wedge', _pie, 'one wedge'),
         ('legend_swatch', _legend_swatch, 'the legend line for beta')]


def loose_box(truth, pad=0.20, shape=None):
    ys, xs = np.nonzero(truth)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    px, py = int((x1 - x0) * pad) + 4, int((y1 - y0) * pad) + 4
    H, W = shape
    return (max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py))


def run():
    out_dir = os.path.join(C.HERE, 'sheets_synthetic')
    os.makedirs(out_dir, exist_ok=True)
    print(f'{"case":22} {"pad":>5} {"GrabCut IoU":>12} {"SlimSAM IoU":>12}')
    for case_id, draw, what in CASES:
        img = draw(A)
        truth = np.any(np.asarray(img) != np.asarray(draw(B)), axis=2)
        for pad in (0.20, 0.80):
            box = loose_box(truth, pad=pad, shape=truth.shape)
            panels = [('the figure', img)]
            scores = {}
            for backend in ('grabcut', 'slimsam'):
                try:
                    mask = O.segment_object(img, box, feather=0, backend=backend)[0] > 0.5
                except ValueError:       # GrabCut refuses a near-full-frame box
                    scores[backend] = float('nan')
                    panels.append((f'{backend}: refused', img))
                    continue
                iou = float((mask & truth).sum()) / max(float((mask | truth).sum()), 1.0)
                scores[backend] = iou
                panels.append((f'{backend} IoU {iou:.2f}', compare.overlay(img, mask, box)))
            panels.append(('what was asked for (constructed truth)',
                           compare.overlay(img, truth, box)))
            print(f'{case_id:22} {pad:5.0%} {scores["grabcut"]:12.2f} '
                  f'{scores["slimsam"]:12.2f}')
            compare.sheet(img, box, panels,
                          f'{case_id} -- {what}, from a box padded {pad:.0%}').save(
                os.path.join(out_dir, f'{case_id}_pad{int(pad * 100)}.png'))
    print('sheets in', out_dir)


if __name__ == '__main__':
    run()
