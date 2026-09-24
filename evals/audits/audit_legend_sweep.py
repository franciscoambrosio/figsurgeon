"""Broad ground-truth sweep for advanced.clean_transparent_box / verify_photo.
check_transparent_box_cleaned. Ground truth: render the same legend over a real
background twice, at framealpha<1 (dirty) and framealpha=1.0 (truth); the opaque
render is what a correct clean should produce, so error is measurable in RGB.

Run:  <repo>/.venv/bin/python evals/audits/audit_legend_sweep.py
Writes: results.csv, sheets/*.png (worst cases), and prints a summary.
"""
import csv
import os
import sys
import time
import traceback

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)  # evals modules assume repo-root-relative cache paths

from figsurgeon import advanced, verify_photo  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SHEETS = os.path.join(HERE, 'sheets')
os.makedirs(SHEETS, exist_ok=True)

CORPUS_CHARTS = os.path.join(REPO, 'evals', '_corpus', 'charts')
CORPUS_PHOTOS = os.path.join(REPO, 'evals', '_corpus')

BACKGROUNDS = {
    # name: (path, description)
    'heatmap_dark':  (os.path.join(CORPUS_CHARTS, 'gcd.png'),
                       'viridis heatmap, dark saturated cells + white gridlines'),
    'contour_dark':  (os.path.join(CORPUS_CHARTS, 'ackley.png'),
                       'viridis contour plot, dark, white contour lines'),
    'scatter_light': (os.path.join(CORPUS_CHARTS, 'exo_scatter.png'),
                       'light scatter chart background'),
    'midgrey_floor': (os.path.join(CORPUS_PHOTOS, 'product_white.jpg'),
                       'white sneakers on mid-grey textured floor'),
    'busy_street':   (os.path.join(CORPUS_PHOTOS, 'street_people.jpg'),
                       'busy photographic street scene'),
    'busy_pizza':    (os.path.join(CORPUS_PHOTOS, 'food_pizza.jpg'),
                       'busy warm-toned food photo'),
    'no_light_black':(os.path.join(CORPUS_PHOTOS, 'product_shoe.jpg'),
                       'red shoes on solid black -- no light content behind the legend'),
    'night_dark':    (os.path.join(CORPUS_PHOTOS, 'night_city.jpg'),
                       'night photograph, deep shadows, point lights'),
    'ui_flat':       (os.path.join(CORPUS_PHOTOS, 'ui_screenshot.png'),
                       'flat light UI screenshot'),
}

SIZES = {  # crop window (w,h), fontsize, handlelength
    'small':  (380, 260, 9, 1.2),
    'medium': (520, 340, 14, 2.0),
    'large':  (760, 480, 20, 3.0),
}

FACECOLORS = {
    'white':      'white',
    'whitesmoke': 'whitesmoke',
    'lightgrey09':(0.9, 0.9, 0.9),
    'cream':      '#F5F0DC',
    'paleyellow': '#FFF9C4',
}

TEXTCOLORS = {
    'black':    'black',
    'darkgrey': '#3f3f3f',
    'darkblue': '#00008B',
}

HUE_SETS = {
    'default3': ['#1f77b4', '#d62728', '#2ca02c'],
    'full6':    ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#ff69b4'],
    'navy1':    ['#000080'],
    'pale3':    ['#ffb6c1', '#000080', '#87ceeb'],   # pink, navy, sky -- the spec's two hard hues
}

_bg_cache = {}
_region_cache = {}


def load_bg(name):
    if name not in _bg_cache:
        path, _ = BACKGROUNDS[name]
        img = Image.open(path).convert('RGB')
        if max(img.size) > 1600:
            img.thumbnail((1600, 1600), Image.LANCZOS)
        _bg_cache[name] = img
    return _bg_cache[name]


def pick_regions(img, win_w, win_h, step=48):
    """Coarse grid search for the lowest- and highest-std window of this size."""
    key = (id(img), win_w, win_h)
    if key in _region_cache:
        return _region_cache[key]
    a = np.asarray(img.convert('L')).astype(float)
    H, W = a.shape
    if win_w >= W or win_h >= H:
        box = (0, 0, min(win_w, W), min(win_h, H))
        _region_cache[key] = (box, box)
        return box, box
    best_lo, best_lo_v = None, 1e18
    best_hi, best_hi_v = None, -1
    for y in range(0, H - win_h, step):
        for x in range(0, W - win_w, step):
            win = a[y:y + win_h, x:x + win_w]
            v = win.std()
            if v < best_lo_v:
                best_lo_v, best_lo = v, (x, y, x + win_w, y + win_h)
            if v > best_hi_v:
                best_hi_v, best_hi = v, (x, y, x + win_w, y + win_h)
    _region_cache[key] = (best_hi, best_lo)  # (busy, plain)
    return best_hi, best_lo


def render(base_img, alpha, facecolor, textcolor, hues, fontsize, handlelength, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    img = np.asarray(base_img)
    h, w = img.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_axis_off()
    handles = [Line2D([0], [0], color=c, lw=4, label=f'S{i}') for i, c in enumerate(hues)]
    leg = ax.legend(handles=handles, loc='center', framealpha=alpha, fontsize=fontsize,
                     facecolor=facecolor, handlelength=handlelength)
    for t in leg.get_texts():
        t.set_color(textcolor)
    fig.canvas.draw()
    bb = leg.get_window_extent()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return (int(bb.x0), int(h - bb.y1), int(np.ceil(bb.x1)), int(np.ceil(h - bb.y0)))


def error(got, truth, box):
    x0, y0, x1, y1 = box
    d = np.abs(np.asarray(got).astype(int)[y0:y1, x0:x1]
               - np.asarray(truth).astype(int)[y0:y1, x0:x1]).max(axis=2)
    if d.size == 0:
        return float('nan'), float('nan')
    return float(d.mean()), float((d > 12).mean())


def swatch_survival(got, truth, box):
    x0, y0, x1, y1 = box
    T = np.asarray(truth).astype(int)[y0:y1, x0:x1]
    G = np.asarray(got).astype(int)[y0:y1, x0:x1]
    sw = (T.max(axis=2) - T.min(axis=2)) > 70
    if not sw.any():
        return float('nan'), 0
    return float((np.abs(G[sw] - T[sw]).max(axis=1) < 40).mean()), int(sw.sum())


def truth_fill_lightness(truth, box):
    """Max-channel of the box's modal calm-light pixel in the opaque render, using
    clean_transparent_box's own light/calm logic."""
    x0, y0, x1, y1 = box
    T = np.asarray(truth).astype(int)[y0:y1, x0:x1]
    mx, mn = T.max(axis=2), T.min(axis=2)
    light = (mx >= 210) & ((mx - mn) <= 70)
    calm = light & ((mx - mn) < 6)
    if calm.sum() < 5:
        # fall back: modal colour of the least-tinted 10% of pixels
        tint = mx - mn
        thresh = np.percentile(tint, 10)
        calm = tint <= thresh
    if not calm.any():
        return float(mx.mean())
    return float(mx[calm].mean())


def build_cases():
    cases = []

    def add(name, bg, size='medium', pos='busy', facecolor='white', alpha=0.75,
            textcolor='black', hueset='default3'):
        cases.append(dict(name=name, bg=bg, size=size, pos=pos, facecolor=facecolor,
                           alpha=alpha, textcolor=textcolor, hueset=hueset))

    # background x framealpha grid: where does the 210 rule break?
    for bg in BACKGROUNDS:
        for alpha in (0.5, 0.65, 0.75, 0.85, 0.95):
            add(f'grid_{bg}_a{alpha}', bg, alpha=alpha)

    for fc in FACECOLORS:
        add(f'facecolor_{fc}_on_heatmap', 'heatmap_dark', facecolor=fc)
        add(f'facecolor_{fc}_on_light', 'scatter_light', facecolor=fc)

    for tc in TEXTCOLORS:
        add(f'textcolor_{tc}_on_heatmap', 'heatmap_dark', textcolor=tc)

    for hs in HUE_SETS:
        add(f'hues_{hs}_on_heatmap', 'heatmap_dark', hueset=hs)
        add(f'hues_{hs}_on_pizza', 'busy_pizza', hueset=hs)

    for size in SIZES:
        for bg in ('heatmap_dark', 'busy_street', 'scatter_light'):
            add(f'size_{size}_on_{bg}', bg, size=size)

    for bg in ('heatmap_dark', 'busy_street', 'midgrey_floor', 'no_light_black', 'night_dark'):
        for pos in ('busy', 'plain'):
            add(f'pos_{pos}_on_{bg}', bg, pos=pos)

    # compounding stress cases
    add('stress_low_alpha_navy_large_night', 'night_dark', alpha=0.5, hueset='navy1',
        size='large', textcolor='darkblue')
    add('stress_low_alpha_pale_heatmap', 'heatmap_dark', alpha=0.5, hueset='pale3')
    add('stress_high_alpha_cream_street', 'busy_street', alpha=0.95, facecolor='cream')
    add('stress_small_dark_navy_street', 'busy_street', size='small', hueset='navy1', alpha=0.6)
    add('stress_no_light_pale_black', 'no_light_black', pos='plain', hueset='pale3', alpha=0.6)
    add('stress_grey_on_grey_floor', 'midgrey_floor', facecolor='lightgrey09', alpha=0.6)

    return cases


def run():
    cases = build_cases()
    print(f'{len(cases)} cases')
    rows = []
    worst = []  # (error_after or inf, name, dirty, cleaned, truth, box)

    for i, cfg in enumerate(cases):
        name = cfg['name']
        try:
            bg_full = load_bg(cfg['bg'])
            win_w, win_h, fontsize, handlelength = SIZES[cfg['size']]
            busy_box, plain_box = pick_regions(bg_full, win_w, win_h)
            box_sel = busy_box if cfg['pos'] == 'busy' else plain_box
            crop = bg_full.crop(box_sel)

            hues = HUE_SETS[cfg['hueset']]
            facecolor = FACECOLORS[cfg['facecolor']] if cfg['facecolor'] in FACECOLORS else cfg['facecolor']
            textcolor = TEXTCOLORS[cfg['textcolor']] if cfg['textcolor'] in TEXTCOLORS else cfg['textcolor']

            dirty_path = os.path.join(SHEETS, '_tmp_dirty.png')
            truth_path = os.path.join(SHEETS, '_tmp_truth.png')
            box = render(crop, cfg['alpha'], facecolor, textcolor, hues, fontsize,
                         handlelength, dirty_path)
            render(crop, 1.0, facecolor, textcolor, hues, fontsize, handlelength, truth_path)
            dirty = Image.open(dirty_path).convert('RGB')
            truth = Image.open(truth_path).convert('RGB')

            mb, fb = error(dirty, truth, box)
            n_sw_truth = swatch_survival(truth, truth, box)[1]
            tfill = truth_fill_lightness(truth, box)

            refused, refuse_msg = False, ''
            cleaned = None
            try:
                cleaned, _ = advanced.clean_transparent_box(dirty, box)
            except ValueError as e:
                refused, refuse_msg = True, str(e)

            if cleaned is not None:
                ma, fa = error(cleaned, truth, box)
                kept, _ = swatch_survival(cleaned, truth, box)
                ok, detail = verify_photo.check_transparent_box_cleaned(dirty, cleaned, box)
            else:
                ma, fa, kept = float('nan'), float('nan'), float('nan')
                ok, detail = 'REFUSED', refuse_msg

            row = dict(name=name, bg=cfg['bg'], pos=cfg['pos'], size=cfg['size'],
                       facecolor=cfg['facecolor'], alpha=cfg['alpha'],
                       textcolor=cfg['textcolor'], hueset=cfg['hueset'], box=box,
                       n_swatch_px=n_sw_truth, truth_fill_maxch=round(tfill, 1),
                       err_before=round(mb, 2), frac_off12_before=round(fb, 4),
                       refused=refused, err_after=(round(ma, 2) if ma == ma else ''),
                       frac_off12_after=(round(fa, 4) if fa == fa else ''),
                       swatch_kept=(round(kept, 4) if kept == kept else ''),
                       verify_ok=str(ok), verify_detail=detail)
            rows.append(row)

            if cleaned is not None:
                key = ma if ma == ma else -1
                worst.append((key, name, dirty.copy(), cleaned.copy(), truth.copy(), box))
            else:
                worst.append((mb, name + '__REFUSED', dirty.copy(), dirty.copy(), truth.copy(), box))

            print(f'[{i+1}/{len(cases)}] {name:45s} before={mb:6.2f} '
                  f'after={"REFUSED" if refused else f"{ma:6.2f}"} '
                  f'kept={"" if cleaned is None else f"{kept:.0%}"} verify={ok}')
        except Exception as e:
            print(f'[{i+1}/{len(cases)}] {name}: CRASHED {type(e).__name__}: {e}')
            traceback.print_exc()
            rows.append(dict(name=name, bg=cfg['bg'], pos=cfg['pos'], size=cfg['size'],
                              facecolor=cfg['facecolor'], alpha=cfg['alpha'],
                              textcolor=cfg['textcolor'], hueset=cfg['hueset'], box='',
                              n_swatch_px='', truth_fill_maxch='', err_before='',
                              frac_off12_before='', refused='CRASH', err_after='',
                              frac_off12_after='', swatch_kept='', verify_ok='CRASH',
                              verify_detail=f'{type(e).__name__}: {e}'))

    csv_path = os.path.join(HERE, 'results.csv')
    fields = ['name', 'bg', 'pos', 'size', 'facecolor', 'alpha', 'textcolor', 'hueset', 'box',
              'n_swatch_px', 'truth_fill_maxch', 'err_before', 'frac_off12_before', 'refused',
              'err_after', 'frac_off12_after', 'swatch_kept', 'verify_ok', 'verify_detail']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {csv_path} ({len(rows)} rows)')

    # top 14 by error_after (refusals ranked by error_before instead)
    worst_sorted = sorted(worst, key=lambda t: -(t[0] if t[0] == t[0] else -1))
    for key, name, dirty, cleaned, truth, box in worst_sorted[:14]:
        x0, y0, x1, y1 = box
        p = 14
        cr = (max(0, x0 - p), max(0, y0 - p), x1 + p, y1 + p)
        w_, h_ = cr[2] - cr[0], cr[3] - cr[1]
        sheet = Image.new('RGB', (w_ * 3 + 24, h_), 'white')
        for i, im in enumerate((dirty, cleaned, truth)):
            sheet.paste(im.crop(cr), (i * (w_ + 12), 0))
        safe = name.replace('/', '_')
        sheet.resize((sheet.width * 2, sheet.height * 2), Image.NEAREST).save(
            os.path.join(SHEETS, f'{safe}_triple.png'))
    print(f'wrote {min(14, len(worst_sorted))} worst-case sheets to {SHEETS}')


if __name__ == '__main__':
    t0 = time.time()
    run()
    print(f'done in {time.time()-t0:.0f}s')
