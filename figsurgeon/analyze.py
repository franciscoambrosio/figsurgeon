"""Measure a figure before editing it.

Nothing here mutates pixels.  These are the checks that turn "the plot looks like X" into
numbers, and the assumption tests that say when the engine's model does not apply.
"""
import numpy as np
from collections import Counter
from PIL import Image


def load(path):
    return np.array(Image.open(path).convert('RGB'))


def census(a, min_count=200):
    """Dominant colours, split into chromatic and neutral.  Run this first, always."""
    rgb = a.astype(int)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    out = {}
    for label, mask in (('chromatic', sat > 40), ('neutral', sat <= 40)):
        c = Counter(map(tuple, rgb[mask]))
        out[label] = [(k, n) for k, n in c.most_common(40) if n >= min_count]
    return out


def count_near(a, colour, tol=30):
    d = np.sqrt(((a.astype(float) - np.array(colour, float)) ** 2).sum(axis=2))
    return int((d <= tol).sum())


def background_columns(a, interior, rows=None, min_frac=0.05):
    """Per-column background, assumed constant down each column.

    Rule: among colours occupying at least `min_frac` of the column, take the LIGHTEST.
    Both simpler rules fail on real figures:
      * plain modal colour returns the LINE for any column dominated by a full-height spike
        or a dash-dot marker;
      * modal-among-the-lightest returns a faint anti-aliased line when that line covers
        more of the column than the bare background does.
    Requiring frequency rejects sparse light outliers; taking the lightest rejects lines.
    """
    x0, x1, y0, y1 = interior
    if rows is None:
        rows = np.arange(y0, y1 + 1)
    w = np.array([0.299, 0.587, 0.114])
    need = max(1, int(min_frac * len(rows)))
    bg = np.zeros(a.shape, float)
    picks = {}
    ai = a.astype(int)
    for x in range(x0, x1 + 1):
        counts = Counter(map(tuple, ai[rows, x]))
        common = [c for c, k in counts.items() if k >= need]
        if not common:
            common = [counts.most_common(1)[0][0]]
        picks[x] = max(common, key=lambda c: np.dot(c, w))

    # A column whose pick disagrees with both neighbours while THEY agree is an isolated
    # contamination (a line's anti-aliasing sneaking above the frequency floor).  A genuine
    # edge of a shaded span has neighbours that disagree with each other, so it is left be.
    for x in range(x0 + 1, x1):
        L, C, R = picks[x - 1], picks[x], picks[x + 1]
        if L == R and C != L:
            picks[x] = L
    for x in range(x0, x1 + 1):
        bg[:, x] = picks[x]
    return bg


def collinear_groups(spec):
    """Series that share a base colour are collinear over any background.

    A 40 %-alpha black line renders at 0.6 * background -- the same ray as an opaque black
    line at 60 % coverage.  No tolerance can separate them; the engine splits them
    spatially instead.  This reports the groups so the caller knows it is happening.
    """
    groups = {}
    for s in spec.series:
        groups.setdefault(tuple(s.colour), []).append(s)
    return {k: v for k, v in groups.items() if len(v) > 1}


def check_band_assumptions(a, spec, bg):
    """Test the three things the band model assumes, against the actual pixels.

    full_height  -- a span reaching both the first and last interior row.  Partial-height
                    spans would leave the engine neutralising a background that is not there.
    behind_lines -- opaque series colours survive inside band columns, so the spans are
                    UNDER the artists.  If a span were drawn on top with alpha, no pure
                    series colour would appear inside it and re-compositing would be wrong.
    vertical     -- tinted columns are tinted for their whole height, not in patches.
    """
    if spec.bands is None:
        return {'applicable': False}
    x0, x1, y0, y1 = spec.interior
    ai = a.astype(int)
    rows = np.array([y for y in range(y0, y1 + 1)
                     if not (spec.legend and spec.legend[2] - 4 <= y <= spec.legend[3] + 4)])

    def tinted(px):
        r, g, b = px[..., 0], px[..., 1], px[..., 2]
        return (np.abs(r - b) <= 6) & (g < r - 6)

    ref = bg[rows[0]]
    band_cols = [x for x in range(x0, x1 + 1) if tinted(ref[x])]
    if not band_cols:
        return {'applicable': True, 'band_columns': 0}

    cols = np.array(band_cols)
    ends = tinted(ai[y0, cols]) & tinted(ai[y1, cols])
    col_tint_frac = tinted(ai[rows][:, cols]).mean(axis=0)

    opaque = [s for s in spec.series if s.alpha >= 1.0 and any(s.colour)]
    inside = ai[rows][:, cols].reshape(-1, 3)
    survive = sum(int((np.sqrt(((inside - np.array(c.colour, float)) ** 2).sum(axis=1)) < 12).sum())
                  for c in opaque)

    return {
        'applicable': True,
        'band_columns': len(cols),
        'full_height_frac': round(float(ends.mean()), 3),
        'vertical_median_tint_frac': round(float(np.median(col_tint_frac)), 3),
        'behind_lines_px': survive,
    }


def probe_interior(a, min_frac=0.9):
    """Suggest the axes bbox from full-height vertical spans and full-width horizontal lines.

    A suggestion only.  Spines are often absent, so confirm the numbers before feeding them
    to a spec.
    """
    ai = a.astype(int)
    nonwhite = ai.min(axis=2) < 245
    H, W = nonwhite.shape

    def longest(v):
        best = cur = start = bstart = 0
        for i, x in enumerate(v):
            if x:
                if cur == 0:
                    start = i
                cur += 1
                if cur > best:
                    best, bstart = cur, start
            else:
                cur = 0
        return best, bstart

    rows = [(y, *longest(nonwhite[y])) for y in range(H)]
    cols = [(x, *longest(nonwhite[:, x])) for x in range(W)]
    rmax = max(r[1] for r in rows)
    cmax = max(c[1] for c in cols)
    yr = [c[0] for c in cols if c[1] >= cmax * min_frac]
    xr = [r[0] for r in rows if r[1] >= rmax * min_frac]
    span_col = [c for c in cols if c[1] >= cmax * min_frac]
    return {
        'suggested_x': (min(x for _, l, x in [(c[0], c[1], c[2]) for c in cols if c[1] >= cmax * min_frac]) if span_col else None,
                        None),
        'longest_row_run': rmax,
        'longest_col_run': cmax,
        'rows_with_full_run': xr[:6],
        'cols_with_full_run': yr[:6],
    }
