"""High-level edits, expressed the way a person describes them.

    highlight(spec, keep=['mlp'])                       "highlight MLP, grey the rest"
    highlight(spec, keep=['mlp','offline'])              "keep MLP and Offline, grey the rest"
    recolour(spec, target='mlp', colour=(200,0,0))       "make the MLP line red"
    thicken(spec, target='offline', factor=2.0)          "make the offline line thicker"

Each returns a new FigureSpec plus the composited image, so a chain of edits can be
inspected and verified at every step rather than trusting one big transform.
"""
import numpy as np
from scipy import ndimage

from .compose import recolour as _recolour, background, classify
from .analyze import load


def highlight(image_or_path, spec, keep):
    """Grey out every series except those named in `keep` (str or list of str)."""
    keep = [keep] if isinstance(keep, str) else list(keep)
    a = load(image_or_path) if isinstance(image_or_path, str) else image_or_path
    if len(keep) == 1:
        return _recolour(a, spec, keep[0])
    # multi-keep: run the single-keep engine for the first survivor, then patch in every
    # other survivor's ORIGINAL pixels from the shared classification, so none of them gets
    # recomposited against a background that isn't theirs
    bg = background(a, spec)
    masks, tmap, _ = classify(a, bg, spec)
    base, info = _recolour(a, spec, keep[0])
    out = base.copy()
    for s in spec.series:
        if s.name in keep[1:]:
            m = info['masks'].get(s.name)
            if m is not None:
                out[m] = a[m]
    return out, info


def recolour_series(image_or_path, spec, target, colour):
    """Change one series' own colour, before compositing (e.g. MLP -> red)."""
    a = load(image_or_path) if isinstance(image_or_path, str) else image_or_path
    # composite the target from the engine's own masks, so protected runs and legend
    # swatches stay untouched
    out, info = _recolour(a, spec, target)
    bg = info['bg']
    m, t = info['masks'][target], info['tmap'][target][..., None]
    from .compose import over
    out = out.astype(float)
    out[m] = (bg + t * (over(colour, spec.by_name(target).alpha, bg) - bg))[m]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), info


def thicken(image_or_path, spec, target, factor=2.0, keep_others=True):
    """Grow a series' line by dilating its recovered mask before compositing.

    This is the raster ceiling: there is no stroke width to change, only pixels already on
    the page.  Dilation approximates a thicker stroke well for factor <= ~2; beyond that the
    line starts eating into neighbouring series and the result should be inspected closely.
    """
    a = load(image_or_path) if isinstance(image_or_path, str) else image_or_path
    bg = background(a, spec)
    masks, tmap, _ = classify(a, bg, spec)
    r = max(1, round((factor - 1) * 1.5))
    grown = ndimage.binary_dilation(masks[target], np.ones((2 * r + 1, 2 * r + 1)))

    out = a.astype(float).copy()
    from .compose import protection_mask, over
    editable = ~protection_mask(spec, a.shape[:2])
    new_ink = grown & editable & ~masks[target]
    s = spec.by_name(target)
    out[new_ink] = over(s.colour, s.alpha, bg)[new_ink]

    if keep_others:
        others, info = highlight(a, spec, target) if len(spec.series) > 1 else (out, {})
        out = np.where(new_ink[..., None] | masks[target][..., None], out, others)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), {'grown_px': int(new_ink.sum())}
