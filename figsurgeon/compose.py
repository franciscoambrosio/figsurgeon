"""Recolouring engine: every pixel is `background + t * (series_colour_over_background -
background)`, t being anti-aliasing coverage. Recolouring re-composites at the same t
against a grey series colour, rather than flattening matched pixels to a constant."""
import numpy as np
from scipy import ndimage

from .analyze import background_columns, collinear_groups


def seg_fit(P, c0, c1):
    """Distance from P to the segment [c0, c1], and the position t along it."""
    d = c1 - c0
    den = np.maximum((d * d).sum(axis=2), 1e-6)
    t = np.clip(((P - c0) * d).sum(axis=2) / den, 0.0, 1.0)
    return np.sqrt(((P - (c0 + t[..., None] * d)) ** 2).sum(axis=2)), t


def over(colour, alpha, bg):
    """What `colour` drawn at `alpha` looks like at full coverage over `bg`."""
    base = np.broadcast_to(np.array(colour, float), bg.shape)
    return alpha * base + (1 - alpha) * bg


def protection_mask(spec, shape):
    """Pixels the engine must never touch."""
    x0, x1, y0, y1 = spec.interior
    prot = np.ones(shape, bool)
    prot[y0:y1 + 1, x0:x1 + 1] = False
    for bx0, bx1, by0, by1 in spec.protect:
        prot[by0:by1 + 1, bx0:bx1 + 1] = True
    if spec.legend:
        lx0, lx1, ly0, ly1 = spec.legend
        prot[ly0:ly1 + 1, lx0:lx1 + 1] = True
    for sx0, sx1, sy0, sy1 in spec.swatches:
        prot[sy0:sy1 + 1, sx0:sx1 + 1] = False
    return prot


def plot_rows(spec):
    """Interior rows for estimating background colour: excludes the legend and any full-width protected band."""
    x0, x1, y0, y1 = spec.interior
    width = x1 - x0 + 1
    skip = np.zeros(y1 + 1 - y0, bool)
    if spec.legend:
        lx0, lx1, ly0, ly1 = spec.legend
        skip[max(0, ly0 - 4 - y0):ly1 + 5 - y0] = True
    for bx0, bx1, by0, by1 in spec.protect:
        if (bx1 - bx0 + 1) >= 0.8 * width:
            skip[max(0, by0 - y0):by1 + 1 - y0] = True
    return np.arange(y0, y1 + 1)[~skip]


def background(a, spec):
    bg = background_columns(a, spec.interior, rows=plot_rows(spec))
    if spec.legend:
        lx0, lx1, ly0, ly1 = spec.legend
        sw = [s for s in spec.swatches]
        rows = np.array([y for y in range(ly0 + 4, ly1 - 3)
                         if not any(r[2] <= y <= r[3] for r in sw)])
        lbg = background_columns(a, (lx0, lx1, ly0, ly1), rows=rows)
        bg[ly0:ly1 + 1, lx0:lx1 + 1] = lbg[ly0:ly1 + 1, lx0:lx1 + 1]
    return bg


def classify(a, bg, spec):
    """Assign each editable pixel to a series, with its coverage t."""
    P = a.astype(float)
    editable = ~protection_mask(spec, a.shape[:2])
    collinear = collinear_groups(spec)

    # one segment per distinct base colour; collinear siblings share it
    groups = {}
    for s in spec.series:
        groups.setdefault(tuple(s.colour), []).append(s)

    fits = {}
    for base, members in groups.items():
        d, t = seg_fit(P, bg, over(base, 1.0, bg))
        fits[base] = (d, t)

    keys = list(groups)
    stack = np.stack([np.where((fits[k][0] < spec.tol) & (fits[k][1] > 0.02),
                               fits[k][0], np.inf) for k in keys])
    win = np.argmin(stack, axis=0)
    hit = np.isfinite(stack.min(axis=0)) & editable

    masks, tmap = {}, {}
    for i, base in enumerate(keys):
        members = groups[base]
        owned = hit & (win == i)
        d, t = fits[base]
        if len(members) == 1:
            s = members[0]
            masks[s.name] = owned
            tmap[s.name] = np.clip(t / s.alpha, 0, 1)
            continue
        # seed the opaque sibling from coverage the translucent one can't reach, then grow
        members = sorted(members, key=lambda s: -s.alpha)
        deep, shallow = members[0], members[-1]
        seed = owned & (t > shallow.alpha + spec.seed_margin)
        grown = ndimage.binary_dilation(seed, np.ones((2 * spec.halo + 1,) * 2))
        masks[deep.name] = owned & grown
        tmap[deep.name] = np.clip(t / deep.alpha, 0, 1)
        masks[shallow.name] = owned & ~grown
        tmap[shallow.name] = np.clip(t / shallow.alpha, 0, 1)
    return masks, tmap, collinear


def blend_pass(a, bg, spec, masks, keep):
    """Pixels that are blends of TWO series sit on no [background -> series] segment."""
    P = a.astype(float)
    editable = ~protection_mask(spec, a.shape[:2])
    taken = np.zeros(a.shape[:2], bool)
    for m in masks.values():
        taken |= m
    todo = editable & ~taken & (np.abs(P - bg).max(axis=2) > 12)
    idx = np.argwhere(todo)
    if not len(idx):
        return np.zeros(a.shape[:2], bool), np.zeros(a.shape, float)

    p, b = P[todo], bg[todo]
    opaque = [s for s in spec.series if s.alpha >= 1.0]
    translucent = [s for s in spec.series if s.alpha < 1.0]
    layers = {s.name: np.broadcast_to(np.array(s.colour, float), b.shape) for s in opaque}
    for s in translucent:                       # a translucent series over each other layer
        for base_name, base in list(layers.items()) + [('bg', b)]:
            layers[f'{s.name}/{base_name}'] = s.alpha * np.array(s.colour, float) + (1 - s.alpha) * base

    names = list(layers)
    best = np.full(len(p), np.inf)
    ok = np.zeros(len(p), bool)
    newp = p.copy()
    grey = np.array(spec.grey, float)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni, nj = names[i].split('/')[0], names[j].split('/')[0]
            if ni == nj:
                continue
            ci, cj = layers[names[i]], layers[names[j]]
            d = cj - ci
            den = np.maximum((d * d).sum(axis=1), 1e-6)
            t = np.clip(((p - ci) * d).sum(axis=1) / den, 0, 1)
            dist = np.sqrt(((p - (ci + t[:, None] * d)) ** 2).sum(axis=1))
            hitm = (dist < spec.tol) & (dist < best)
            best = np.where(hitm, dist, best)
            if keep in (ni, nj):
                kc = ci if ni == keep else cj
                tk = t if ni == keep else (1 - t)
                cand = kc + tk[:, None] * (grey - kc)
            else:
                cand = np.broadcast_to(grey, p.shape)
            newp = np.where(hitm[:, None], cand, newp)
            ok |= hitm
    m = np.zeros(a.shape[:2], bool)
    sel = idx[ok]
    m[sel[:, 0], sel[:, 1]] = True
    out = np.zeros(a.shape, float)
    out[sel[:, 0], sel[:, 1]] = newp[ok]
    return m, out


def neutralise_bands(bg, spec):
    """Replace shaded-span backgrounds with neutral ones of the same weight, without disturbing anything already darkening the column."""
    if spec.bands is None:
        return bg
    r, g, b = bg[:, :, 0], bg[:, :, 1], bg[:, :, 2]
    is_band = (np.abs(r - b) <= 6) & (g < r - 6)
    v = r - spec.bands.darken * (r - g)
    out = bg.copy()
    out[is_band] = np.repeat(v[is_band][:, None], 3, axis=1)
    return out


def recolour(a, spec, keep):
    """Return the recoloured image, plus the masks used (for verification)."""
    bg = background(a, spec)
    masks, tmap, collinear = classify(a, bg, spec)
    blend_m, blend_vals = blend_pass(a, bg, spec, masks, keep)

    for pr in spec.protect_runs:
        m = masks[pr.series]
        prot = np.zeros(a.shape[:2], bool)
        for x in range(pr.x0, pr.x1 + 1):
            col, y = m[:, x], 0
            while y < len(col):
                if col[y]:
                    y2 = y
                    while y2 < len(col) and col[y2]:
                        y2 += 1
                    if y2 - y >= pr.min_run:
                        prot[y:y2, x] = True
                    y = y2
                else:
                    y += 1
        masks[pr.series] = m & ~prot

    if spec.swatches:                 # inside swatch boxes only touch real ink
        sw = np.zeros(a.shape[:2], bool)
        for sx0, sx1, sy0, sy1 in spec.swatches:
            sw[sy0:sy1 + 1, sx0:sx1 + 1] = True
        for nm in list(masks):
            masks[nm] = masks[nm] & (~sw | (tmap[nm] >= spec.swatch_ink))
        blend_m &= ~sw

    bgn = neutralise_bands(bg, spec)
    moved = np.abs(bgn - bg).max(axis=2) > 0
    grey = np.array(spec.grey, float)

    out = a.astype(float).copy()
    handled = blend_m.copy()
    out[blend_m] = blend_vals[blend_m]

    for nm, m in masks.items():
        s = spec.by_name(nm)
        t = tmap[nm][..., None]
        if nm == keep:
            here = m & moved     # only where the background under it changed
            out[here] = (bgn + t * (over(s.colour, s.alpha, bgn) - bgn))[here]
            handled |= m
            continue
        out[m] = (bgn + t * (grey - bgn))[m]
        handled |= m

    if spec.bands is not None:
        # unowned pixels are usually black-over-background (bgn/bg rescales exactly); ones whose channel ratios disagree (e.g. a grey legend frame) are handled below.
        x0, x1, y0, y1 = spec.interior
        zone = np.zeros(a.shape[:2], bool)
        zone[y0:y1 + 1, x0:x1 + 1] = True
        todo = zone & ~handled
        r, g, b = out[:, :, 0], out[:, :, 1], out[:, :, 2]
        fam = todo & (np.abs(r - b) <= 6) & (g < r - 6)
        v = r - spec.bands.darken * (r - g)
        out[fam] = np.repeat(v[fam][:, None], 3, axis=1)

        ratio = np.divide(out, np.maximum(bg, 1e-6))
        flat = (ratio.max(axis=2) - ratio.min(axis=2)) < 0.06
        scale = todo & ~fam & flat & moved
        out[scale] = np.clip(out * (bgn / np.maximum(bg, 1e-6)), 0, 255)[scale]

    # three-way blends sit on no two-endpoint segment: strip chroma, keep luminance
    editable = ~protection_mask(spec, a.shape[:2])
    ks = spec.by_name(keep)
    d_keep, _ = seg_fit(out, bgn, over(ks.colour, ks.alpha, bgn))
    sat = out.max(axis=2) - out.min(axis=2)
    tinted = (np.abs(out[:, :, 0] - out[:, :, 2]) < 6) & (out[:, :, 1] < out[:, :, 0] - 6)
    leftover = editable & ~handled & (sat > 18) & ~tinted & (d_keep > 25)
    lum = out @ np.array([0.299, 0.587, 0.114])
    out[leftover] = np.repeat(lum[leftover][:, None], 3, axis=1)

    return np.clip(np.rint(out), 0, 255).astype(np.uint8), {'masks': masks, 'tmap': tmap,
                                                            'bg': bg, 'bgn': bgn,
                                                            'collinear': collinear}
