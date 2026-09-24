"""Does recolouring a chart series move its data? Read back in data units, against known truth.

    .venv/bin/python evals/chart_data_drift.py [model ... | --free]

Seven synthetic charts with seeded, known data (four simple, three dense). One series in each is recoloured blue -> red,
by figsurgeon (`replace_colour`) and by hosted image models given the whole chart. Every
version is then READ BACK: the axes frame is found in that image's own pixels and mapped to
the known axis limits, so a resized or shifted return is calibrated on its own terms and
only a moved mark counts. The original is read back the same way, as the measurement floor.

Returns are cached under the out dir, so re-running re-reads without paying again.

Result (2026-09-24, google/gemini-3-pro-image, one call per chart, $0.14 each): no data moved.
On all seven charts the red series reads back within the original's own measurement floor
(bars within 0.24 units, points within 0.6, on a 0-100 axis), and the two untouched series
stay put. The noisy line reads 1.68 against a 1.34 floor only because the model drew it
slightly thicker: overlaid in frame coordinates it covers 99.4 % of the original's pixels,
spike for spike. The one displacement seen so far is elsewhere: `frontier_editors.py`'s
five-series chart, where the recoloured line was redrawn 2.5 px higher.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from figsurgeon.tools import dispatch                                   # noqa: E402
import openrouter_edit                                                  # noqa: E402

OUT = os.path.join(HERE, 'out_chart_data_drift')
MODELS = ['google/gemini-3-pro-image']
BLUE, RED, ORANGE, GREEN = (31, 119, 180), (214, 39, 40), (255, 127, 14), (44, 160, 44)
XLIM, YLIM = (0, 10), (0, 100)
INSTRUCTION = ('Change the colour of the {what} from blue to red. Keep everything else in '
               'the chart exactly as it is.')


def _axes(figsize=(9, 6)):
    fig = plt.figure(figsize=figsize, dpi=100)
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.82])
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    for side in ax.spines.values():
        side.set_color('black')
        side.set_linewidth(1.5)
    return fig, ax


def _png(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100)
    plt.close(fig)
    return Image.open(buf).convert('RGB')


def charts():
    """(name, what the instruction calls the series, image, truth) for each chart."""
    rng = np.random.default_rng(5)
    out = []

    heights = np.round(rng.uniform(15, 90, 8), 1)
    fig, ax = _axes()
    xs = np.arange(8) + 1.0
    ax.bar(xs, heights, width=0.6, color=np.array(BLUE) / 255)
    ax.set_title('Monthly output')
    out.append(('bars', 'bars', _png(fig), {'kind': 'bars', 'x': xs, 'y': heights}))

    x = np.linspace(0.5, 9.5, 200)
    y = 50 + 25 * np.sin(x * 0.9) + 8 * np.cos(x * 2.3)
    fig, ax = _axes()
    ax.plot(x, y, color=np.array(BLUE) / 255, lw=2.5)
    ax.set_title('Sensor reading')
    out.append(('line', 'line', _png(fig), {'kind': 'line', 'x': x, 'y': y}))

    fig, ax = _axes()
    y_orange, y_green = 30 + 10 * np.sin(x * 0.7), 75 + 8 * np.cos(x * 0.5)
    ax.plot(x, y_orange, color=np.array(ORANGE) / 255, lw=2.5)
    ax.plot(x, y, color=np.array(BLUE) / 255, lw=2.5)
    ax.plot(x, y_green, color=np.array(GREEN) / 255, lw=2.5)
    ax.set_title('Three series')
    out.append(('three_lines', 'blue line', _png(fig),
                {'kind': 'line', 'x': x, 'y': y,
                 'others': [('orange (untouched)', ORANGE, y_orange),
                            ('green (untouched)', GREEN, y_green)]}))

    pts = np.column_stack([rng.uniform(1, 9, 30), rng.uniform(10, 90, 30)])
    fig, ax = _axes()
    ax.scatter(pts[:, 0], pts[:, 1], s=60, color=np.array(BLUE) / 255)
    ax.set_title('Measurements')
    out.append(('scatter', 'points', _png(fig), {'kind': 'scatter', 'pts': pts}))

    # Denser versions of the same three kinds, closer to a real figure's detail.
    thin = np.round(rng.uniform(10, 95, 30), 1)
    xs30 = np.linspace(0.4, 9.6, 30)
    fig, ax = _axes()
    ax.bar(xs30, thin, width=0.15, color=np.array(BLUE) / 255)
    ax.set_title('Daily output')
    out.append(('bars_dense', 'bars', _png(fig), {'kind': 'bars', 'x': xs30, 'y': thin}))

    xn = np.linspace(0.5, 9.5, 900)
    yn = 50 + 20 * np.sin(xn * 0.8) + rng.normal(0, 4, xn.size)
    fig, ax = _axes()
    ax.plot(xn, yn, color=np.array(BLUE) / 255, lw=1)
    ax.set_title('Raw signal')
    out.append(('line_noisy', 'line', _png(fig), {'kind': 'line', 'x': xn, 'y': yn}))

    many = np.column_stack([rng.uniform(1, 9, 150), rng.uniform(10, 90, 150)])
    fig, ax = _axes()
    ax.scatter(many[:, 0], many[:, 1], s=14, color=np.array(BLUE) / 255)
    ax.set_title('All measurements')
    out.append(('scatter_dense', 'points', _png(fig), {'kind': 'scatter', 'pts': many}))
    return out


def frame(img):
    """The axes frame in this image's own pixels: (left, right, top, bottom) spine centres."""
    a = np.asarray(img.convert('L')) < 90
    h, w = a.shape
    rows = np.flatnonzero(a.sum(axis=1) > 0.6 * w)
    cols = np.flatnonzero(a.sum(axis=0) > 0.6 * h)

    def ends(idx):
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        return runs[0].mean(), runs[-1].mean()
    top, bottom = ends(rows)
    left, right = ends(cols)
    return left, right, top, bottom


def to_data(img, px, py):
    left, right, top, bottom = frame(img)
    x = XLIM[0] + (np.asarray(px) - left) / (right - left) * (XLIM[1] - XLIM[0])
    y = YLIM[0] + (bottom - np.asarray(py)) / (bottom - top) * (YLIM[1] - YLIM[0])
    return x, y


PALETTE = np.array([BLUE, RED, ORANGE, GREEN, (0, 0, 0), (255, 255, 255)])


def near(img, colour, tol=90):
    """Pixels whose nearest palette colour is `colour`. A fixed tolerance alone is not
    enough: a hue-shifted red keeps the blue's darkness and lands near orange."""
    a = np.asarray(img.convert('RGB')).astype(int)
    d = np.abs(a[:, :, None, :] - PALETTE[None, None]).max(axis=3)
    i = int(np.flatnonzero((PALETTE == colour).all(axis=1))[0])
    return (d.argmin(axis=2) == i) & (d[:, :, i] < tol)


def read(img, truth, colour):
    """The series' values as drawn in `img`, in data units, next to the truth."""
    m = near(img, colour)
    left, right, top, bottom = frame(img)
    m[:, :int(left) + 3] = m[:, int(right) - 2:] = False
    m[:int(top) + 3] = m[int(bottom) - 2:] = False
    if truth['kind'] == 'bars':
        got = []
        for x0 in truth['x']:
            px = left + (x0 - XLIM[0]) / (XLIM[1] - XLIM[0]) * (right - left)
            col = m[:, int(round(px))]
            got.append(to_data(img, px, np.flatnonzero(col).min())[1] if col.any() else np.nan)
        return np.array(got), truth['y']
    if truth['kind'] == 'line':
        cols = np.flatnonzero(m.any(axis=0))
        centres = np.array([np.flatnonzero(m[:, c]).mean() for c in cols])
        gx, gy = to_data(img, cols, centres)
        keep = (gx > truth['x'][0] + 0.1) & (gx < truth['x'][-1] - 0.1)
        return gy[keep], np.interp(gx[keep], truth['x'], truth['y'])
    labels, n = ndimage.label(m)
    cy, cx = np.array(ndimage.center_of_mass(m, labels, range(1, n + 1))).T
    gx, gy = to_data(img, cx, cy)
    got = np.column_stack([gx, gy])
    pts = truth['pts']
    nearest = np.array([pts[np.argmin(np.hypot(*(pts - g).T))] for g in got])
    return got[:, 1], nearest[:, 1]


def summary(got, want):
    err = np.asarray(got, float) - np.asarray(want, float)
    err = err[np.isfinite(err)]
    return {'n': int(err.size), 'mean_abs': round(float(np.abs(err).mean()), 2),
            'max_abs': round(float(np.abs(err).max()), 2),
            'mean_signed': round(float(err.mean()), 2)}


def run(models):
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name, what, img, truth in charts():
        img.save(os.path.join(OUT, f'{name}__original.png'))
        instruction = INSTRUCTION.format(what=what)
        versions = [('original', img, BLUE),
                    ('figsurgeon', dispatch('replace_colour', {'from_rgb': list(BLUE),
                                                               'to_rgb': list(RED)}, img)[0], RED)]
        for model in models:
            path = os.path.join(OUT, f'{name}__{model.replace("/", "_")}.png')
            if not os.path.exists(path):
                got, meta = openrouter_edit.edit(img, instruction, model, fmt='PNG')
                if got is None:
                    print(f'{name}: {model} returned no image -- {str(meta)[:200]}')
                    continue
                got.save(path)
                json.dump(meta, open(path.replace('.png', '.json'), 'w'), indent=2, default=str)
            versions.append((model, Image.open(path).convert('RGB'), RED))
        print(f'\n=== {name}: "{instruction}"')
        for arm, im, colour in versions:
            s = summary(*read(im, truth, colour))
            rows.append({'chart': name, 'arm': arm, 'size': list(im.size), **s})
            print(f'    {arm:28s} {im.size[0]}x{im.size[1]}  error in data units: '
                  f'mean {s["mean_abs"]}, max {s["max_abs"]}, bias {s["mean_signed"]:+}  '
                  f'(n={s["n"]}, y axis 0-100)')
            for label, other, y_other in truth.get('others', []):
                o = summary(*read(im, {'kind': 'line', 'x': truth['x'], 'y': y_other}, other))
                rows.append({'chart': name, 'arm': arm, 'series': label, **o})
                print(f'        {label:24s} mean {o["mean_abs"]}, max {o["max_abs"]}, '
                      f'bias {o["mean_signed"]:+}')
    json.dump(rows, open(os.path.join(OUT, 'rows.json'), 'w'), indent=2)
    return rows


if __name__ == '__main__':
    args = sys.argv[1:]
    run([] if args == ['--free'] else args or MODELS)
