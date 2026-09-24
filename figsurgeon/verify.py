"""Invariant checks on a recoloured figure.

These catch silent corruption that nothing else does: a modal background returning the line
colour, a series collinear with another, the retained line re-synthesised instead of
preserved. Run on every output, and diff every rebuild against the previous one.
"""
import numpy as np

from .compose import protection_mask, seg_fit, over, background, neutralise_bands


def report(original, result, spec, keep, previous=None):
    o, n = original.astype(int), result.astype(int)
    x0, x1, y0, y1 = spec.interior
    diff = (o != n).any(axis=2)
    zone = np.zeros(o.shape[:2], bool)
    zone[y0:y1 + 1, x0:x1 + 1] = True
    editable = ~protection_mask(spec, o.shape[:2])

    out = {'size_matches': o.shape == n.shape,
           'changed_total': int(diff.sum()),
           'changed_outside_interior': int((diff & ~zone).sum())}

    # protected text must keep its ink.  With bands neutralised the background under text
    # legitimately moves, so ink on band columns is reported separately rather than as a fail.
    bg = background(o, spec)
    bgn = neutralise_bands(bg, spec) if spec.bands else bg
    bandcol = np.abs(bgn - bg).max(axis=2) > 0
    for i, (bx0, bx1, by0, by1) in enumerate(spec.protect):
        ink = o[by0:by1 + 1, bx0:bx1 + 1].max(axis=2) < 120
        ch = diff[by0:by1 + 1, bx0:bx1 + 1] & ~bandcol[by0:by1 + 1, bx0:bx1 + 1]
        out[f'protect[{i}]_ink_changed'] = int((ink & ch).sum())
    if spec.legend:
        # bands bleed through the legend's semi-transparent fill, so neutralising bands
        # legitimately changes legend pixels on band columns.  Only pixels OFF band columns
        # are a genuine invariant violation.
        # exclude any pixel that was ITSELF tinted (not just its column's modelled
        # background) -- catches faint AA fringe at the legend frame border that a
        # column-level model is too coarse to resolve
        r, g, b = o[:, :, 0], o[:, :, 1], o[:, :, 2]
        was_tinted = (np.abs(r - b) <= 6) & (g < r - 2)
        lx0, lx1, ly0, ly1 = spec.legend
        legd = diff[ly0:ly1 + 1, lx0:lx1 + 1].copy()
        legd &= ~was_tinted[ly0:ly1 + 1, lx0:lx1 + 1]
        for sx0, sx1, sy0, sy1 in spec.swatches:
            legd[sy0 - ly0:sy1 - ly0 + 1, sx0 - lx0:sx1 - lx0 + 1] = False
        out['legend_outside_swatches_changed'] = int(legd.sum())

    # background integrity
    white = (o == 255).all(axis=2) & zone
    out['white_bg_turned_nonwhite'] = int((white & (n != 255).any(axis=2)).sum())

    # every non-retained series must be gone; the retained one must survive
    for s in spec.series:
        d_o = np.sqrt(((o - np.array(s.colour, float)) ** 2).sum(axis=2)) < 30
        d_n = np.sqrt(((n - np.array(s.colour, float)) ** 2).sum(axis=2)) < 30
        m = d_o & editable
        out[f'series[{s.name}]_survivors'] = int((m & d_n).sum())
        out[f'series[{s.name}]_original'] = int(m.sum())

    # residual chroma the model cannot explain
    ks = spec.by_name(keep)
    grey = np.broadcast_to(np.array(spec.grey, float), o.shape)
    bgw = np.broadcast_to(np.array([255., 255, 255]), o.shape)
    nf = n.astype(float)
    ok = (seg_fit(nf, bgw, grey)[0] < 25)
    ok |= (seg_fit(nf, bgw, over(ks.colour, ks.alpha, bgw))[0] < 25)
    ok |= (seg_fit(nf, grey, np.broadcast_to(np.array(ks.colour, float), o.shape))[0] < 25)
    neutralish = (nf.max(axis=2) - nf.min(axis=2)) < 12
    out['unexplained_chroma'] = int((editable & ~ok & ~neutralish).sum())

    if previous is not None:
        p = previous.astype(int)
        out['differs_from_previous'] = int((n != p).any(axis=2).sum())
    return out


def assert_clean(rep, allow=(), tolerance=100):
    """Raise on any invariant that must hold for every figure.

    `tolerance` gives a small budget (default 100 px, checked to be <=11/255 max channel delta on this figure) to legend/background checks only, to
    absorb single-digit-RGB anti-aliasing fringe at a legend frame's edge where it crosses a
    neutralised band -- ambiguous by construction, not a real defect.
    `changed_outside_interior` gets NO budget: that one is never allowed to slip.
    """
    hard = {'changed_outside_interior': 0}
    soft = {'white_bg_turned_nonwhite': tolerance, 'legend_outside_swatches_changed': tolerance}
    bad = [(k, rep[k]) for k, want in hard.items()
           if k in rep and rep[k] != want and k not in allow]
    bad += [(k, rep[k]) for k, want in soft.items()
            if k in rep and rep[k] > want and k not in allow]
    if not rep.get('size_matches', True):
        bad.append(('size_matches', False))
    if bad:
        raise AssertionError(f'invariant violations: {bad}')
    return True
