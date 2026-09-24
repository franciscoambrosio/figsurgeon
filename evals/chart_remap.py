"""Re-theming a published figure: what a colormap remap does to charts nobody here made.

    python evals/chart_remap.py          # needs matplotlib; fetches 3 figures once

Figures come from Wikimedia Commons, not from this package, to measure the remap on charts
it wasn't tuned against: a viridis heatmap, a viridis contour plot, a jet field with
annotation curves.

Two defects, invisible in the operation's own report but obvious at 1:1: an anti-aliased
edge along gridlines/contours can keep the old palette (fixed); a band of values on the jet
figure doesn't match at all, since that figure's jet isn't matplotlib's -- not fixable
without recolouring untouched content, so the check fails it and names the band instead.

Sheets are before | after crops where the defect lives. Look at them.
"""
import os
import sys
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from evals import real_corpus                                            # noqa: E402
from figsurgeon import rebrand, verify_photo                            # noqa: E402

CACHE = os.path.join(HERE, '_corpus', 'charts')
SHEETS = os.path.join(HERE, 'out_chart_remap')
BRAND = ('#00407A', '#52BDEC')          # a dark and a light brand blue
BRAND_WARM = ('#3B0F70', '#FDE725')

# Commons title, render width, boxes, colormap, and the 1:1 crop where the defect shows.
# `boxes=[None]` is the shipped default (whole figure, colorbar included); the heatmap also
# runs the boxed path, where a caller can forget the key and the check has to say so.
CASES = [
    ('gcd_heatmap', 'Heatmap of GCD Matrix.png', 1200, 'viridis', BRAND,
     [None], (430, 380, 580, 520),
     'a viridis heatmap with white gridlines, and its colorbar, in ONE call'),
    ('gcd_cells_only', 'Heatmap of GCD Matrix.png', 1200, 'viridis', BRAND,
     [(134, 66, 634, 565)], (600, 40, 800, 590),
     'the same heatmap with a box around the cells: the key is left behind, and said so'),
    ('ackley_contour', 'Ackley 2d.png', 1200, 'viridis', BRAND,
     [None], (300, 60, 560, 240),
     'a viridis contour plot: white contour lines over the field'),
    ('photon_jet', 'Photon Cross Sections.png', 1200, 'jet', BRAND_WARM,
     [None], (600, 300, 1060, 700),
     "a jet field whose jet is not matplotlib's -- the band case"),
]


def image(name, title, width):
    """The figure, cached under evals/_corpus/charts/ (gitignored, credited on fetch)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name + '.png')
    if not (os.path.exists(path) and os.path.getsize(path) > 1000):
        info = real_corpus._api({'action': 'query', 'format': 'json',
                                 'titles': f'File:{title}', 'prop': 'imageinfo',
                                 'iiprop': 'url|extmetadata', 'iiurlwidth': str(width)})
        page = next(iter(info['query']['pages'].values()))
        if 'imageinfo' not in page:
            raise LookupError(f'Commons has no file named {title!r}')
        ii = page['imageinfo'][0]
        req = urllib.request.Request(ii.get('thumburl') or ii['url'],
                                     headers={'User-Agent': real_corpus.UA})
        with urllib.request.urlopen(req, timeout=120) as fh:
            data = fh.read()
        with open(path, 'wb') as out:
            out.write(data)
        meta = ii.get('extmetadata', {})
        with open(os.path.join(CACHE, 'CREDITS.txt'), 'a') as fh:
            fh.write(f'{name}: File:{title}\n'
                     f'    artist:  {real_corpus._strip_html(meta.get("Artist", {}).get("value"))}\n'
                     f'    licence: {real_corpus._strip_html(meta.get("LicenseShortName", {}).get("value"))}\n'
                     f'    source:  https://commons.wikimedia.org/wiki/File:'
                     f'{urllib.parse.quote(title.replace(" ", "_"))}\n')
    return Image.open(path).convert('RGB')


def old_scale_left(before, after, box, cmap):
    """Pixels in the box still wearing the old colour scale: unchanged, colourful, and
    close to the source colormap. The measure the check is built on, reported here per
    figure so the sheet and the number can be compared."""
    a = np.asarray(before).astype(float)
    c = np.asarray(after).astype(float)
    x0, y0, x1, y1 = box
    A, C = a[y0:y1, x0:x1], c[y0:y1, x0:x1]
    changed = np.abs(A - C).max(axis=2) > 2
    _, d = rebrand._nearest_in_lut(A.reshape(-1, 3), rebrand._cmap_lut(cmap), key=cmap)
    d = d.reshape(A.shape[:2])
    colourful = (A.max(axis=2) - A.min(axis=2)) > 30
    left = (~changed) & colourful & (d < 60)
    content = changed | left
    return int(left.sum()), int(content.sum())


def pair(before, after, crop, path, scale=3):
    """before | after at 1:1 (or larger). The defects are invisible at figure scale, which is
    why this crops in to 1:1."""
    x0, y0, x1, y1 = crop
    w, h = x1 - x0, y1 - y0
    sheet = Image.new('RGB', (w * 2 + 16, h), 'white')
    sheet.paste(before.crop(crop), (0, 0))
    sheet.paste(after.crop(crop), (w + 16, 0))
    sheet.resize(((w * 2 + 16) * scale, h * scale), Image.NEAREST).save(path)


def run():
    os.makedirs(SHEETS, exist_ok=True)
    for name, title, width, cmap, new, boxes, crop, what in CASES:
        img = image(name.replace('_cells_only', ''), title, width)
        out = img
        fracs = []
        for box in boxes:
            out, frac, alpha = rebrand.remap_colormap(out, box, source_cmap=cmap,
                                                      new_colours=new)
            fracs.append(frac)                    # per box: a single number across two boxes
                                                  # would hide that the colorbar is the 100 % one
        whole = (0, 0, img.size[0], img.size[1])
        left, content = old_scale_left(img, out, boxes[0] or whole, cmap)
        ok, detail = verify_photo.check_colormap_remapped(img, out, boxes[0],
                                                          source_cmap=cmap, new_colours=new)
        verdict = {True: 'verified', False: 'FAILED', None: 'no verdict'}[ok]
        print(f'{name:16s} {img.size[0]}x{img.size[1]}  {what}')
        where = 'whole figure' if boxes[0] is None else f'{len(boxes)} box(es)'
        print(f'{"":16s} {where}; matched {", ".join(f"{f:.1%}" for f in fracs)}, alpha {alpha:.2f}; '
              f'{left:,} of {content:,} px left on the old scale ({left / max(content, 1):.2%})')
        print(f'{"":16s} {verdict}: {detail}\n')
        out.save(os.path.join(SHEETS, f'{name}_after.png'))
        pair(img, out, crop, os.path.join(SHEETS, f'{name}_pair.png'))
    print(f'sheets in {SHEETS} -- the _pair files are 1:1 crops. LOOK AT THEM: both defects '
          f'found here are invisible at figure scale.')


if __name__ == '__main__':
    run()
