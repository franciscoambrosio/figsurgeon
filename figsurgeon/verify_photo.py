"""Verify that an edit did what was asked, measured on the output pixels.

A change can run without error and still not be the requested change -- e.g. "blur the
background" blurring the whole frame, subject included. Each check encodes the *intent* of
an operation as a measurable property and returns (passed, detail).

Not an image-quality model: these test whether the requested change happened and whether
untouched things stayed untouched. Whether the result looks good is not checked.
"""
import re

import numpy as np

from .composite import _arr, _clamp, _padded


def _b(x):
    """Coerce to a real Python bool: numpy.bool_ satisfies neither `is True` nor `is False`,
    which callers rely on to tell a failure from a skip."""
    return bool(x)


def _sat(a):
    mx, mn = a.max(axis=2), a.min(axis=2)
    return (mx - mn) / np.maximum(mx, 1e-6)


# Minimum off-hue fraction before "the rest was greyed out" is evidence rather than a fixed
# pixel floor, which is too low on a real image (a near-complete colour-pop can leave as
# little as 0.02% off-hue).
_EVIDENCE_FRACTION = 0.005


def check_isolate_colour(before, after, target_rgb, hue_tol=0.09, space='hsv',
                         oklab_tol=0.06):
    """The pixels the operation was asked to KEEP hold their saturation; the rest lose it.

    `space` must select the target pixels the same way the operation selected them, or the
    check grades a different edit than the one that ran: a hue-window in HSV includes pixels
    an OKLab isolation correctly excludes, so grading an OKLab edit with an HSV window reads
    a tight, correct isolation as a failure (`evals/space_aware_checks.py`).
    """
    from .photo import _to_hsv
    import colorsys
    a0, a1 = _arr(before), _arr(after)
    h, s, _ = _to_hsv(a0)
    if space == 'oklab':
        from .perceptual import colour_mask
        m = colour_mask(before.convert('RGB'), tuple(int(c) for c in target_rgb),
                        tolerance=oklab_tol)
        on_hue = m > 0.5
        # Far from the target perceptually and saturated enough for greying to be visible.
        off_hue = (m < 0.1) & (s > 0.20)
    else:
        target_h = colorsys.rgb_to_hsv(*(np.array(target_rgb) / 255.0))[0]
        dh = np.minimum(np.abs(h - target_h), 1 - np.abs(h - target_h))
        on_hue = (dh < hue_tol) & (s > 0.12)
        off_hue = (dh > hue_tol * 2.5) & (s > 0.20)
    if on_hue.sum() < 50:
        return None, 'target colour barely present in source; check skipped'
    if off_hue.sum() < max(50, _EVIDENCE_FRACTION * h.size):
        # No off-hue pixels means nothing for the operation to grey out, so there is no
        # evidence either way -- not a pass.
        return None, (f'only {off_hue.mean():.2%} of the frame is both saturated and off '
                      f'the target hue, which is not enough to tell whether anything was '
                      f'greyed out. Either the image has nothing to separate -- colour '
                      f'isolation cannot help there, use the box-based isolate_object -- or '
                      f'an earlier pass already greyed it, in which case this one had '
                      f'nothing left to do')
    s0, s1 = _sat(a0), _sat(a1)
    kept = float(s1[on_hue].mean() / max(s0[on_hue].mean(), 1e-6))
    dropped = float(s1[off_hue].mean() / max(s0[off_hue].mean(), 1e-6))
    ok = _b(kept > 0.55 and dropped < 0.55)
    what = 'target hue' if space != 'oklab' else f'the colour within {oklab_tol:.2f} of the target'
    return ok, (f'{what} retained {kept:.0%} of saturation, the rest of the frame retained '
                f'{dropped:.0%}')


def check_background_blur(before, after, subject_mask=None, session=None):
    """The subject must stay sharp while the background loses detail.

    Without this, blurring the whole image passes as success: a naive diff rewards any
    large change. When no mask is given, one is derived from the same subject segmentation
    the operation uses, rather than assuming the corners are background -- a subject that
    reaches into the corner would otherwise fail correct work.
    """
    from scipy.ndimage import gaussian_filter
    a0, a1 = _arr(before).mean(axis=2), _arr(after).mean(axis=2)

    def detail(x, m):
        if m.sum() < 100:
            return None
        d = np.abs(x - gaussian_filter(x, 3))
        return float(d[m].mean())

    if subject_mask is None:
        try:
            from . import advanced
            cut = advanced.remove_background(before, session=session)
            subject_mask = np.array(cut.split()[-1]).astype(float) / 255.0
        except Exception as e:
            return None, f'could not derive subject mask ({type(e).__name__}); check skipped'

    sub = subject_mask > 0.6
    bg = subject_mask < 0.2
    d0s, d1s = detail(a0, sub), detail(a1, sub)
    d0b, d1b = detail(a0, bg), detail(a1, bg)
    if None in (d0s, d1s, d0b, d1b):
        return None, 'subject or background region too small to measure; check skipped'
    r_sub = d1s / max(d0s, 1e-6)
    r_bg = d1b / max(d0b, 1e-6)
    ok = _b(r_sub > 0.6 and r_bg < 0.6)
    return ok, f'subject detail retained {r_sub:.0%}, background detail retained {r_bg:.0%}'


def check_grayscale(before, after):
    a1 = _arr(after)
    residual = float(_sat(a1).mean())
    return _b(residual < 0.02), f'mean residual saturation {residual:.4f}'


def _luma(a):
    """PIL's RGB -> L weights; the grey each enhancer blends toward."""
    return a[:, :, 0] * .299 + a[:, :, 1] * .587 + a[:, :, 2] * .114


def _predict_adjust(a0, key, factor):
    """What `factor` can actually achieve on this image, clipping included.

    Models the intent, not the implementation: "brightness x1.5" means scale luminance by
    1.5 and clip at white. Written against the meaning of the request, so an implementation
    swap is still graded against what the caller asked for.
    """
    if key == 'brightness':
        out = a0 * factor
    elif key == 'saturation':
        g = _luma(a0)[:, :, None]
        out = g + (a0 - g) * factor
    elif key == 'contrast':
        grey = float(np.round(_luma(a0).mean()))
        out = grey + (a0 - grey) * factor
    else:
        raise ValueError(key)
    return np.round(np.clip(out, 0, 255))


_ADJUST_STAT = {
    'brightness': lambda a: float(a.mean()),
    'contrast': lambda a: float(a.std()),
    # Mean distance from grey, not (max-min)/max: the enhancer blends linearly toward grey,
    # so this statistic is linear in the factor and the achievable change is predictable.
    'saturation': lambda a: float(np.abs(a - _luma(a)[:, :, None]).mean()),
}

# Below this multiple of the quantisation noise, the requested change is smaller than 8-bit
# rounding can resolve and no verdict is honest.
_NOISE_GATE = 3.0
_ACHIEVED_BAND = (0.5, 2.5)


def _quantisation_noise(a0, key, factor, draws=4):
    """How much this statistic wobbles under sub-level (<0.5) changes in the input.

    An 8-bit image only records where each channel landed after rounding, so the achievable
    change carries an uncertainty that has nothing to do with the edit -- a fixed +/-5% band
    would call a correct edit a failure.
    """
    rng = np.random.default_rng(0)
    stat = _ADJUST_STAT[key]
    deltas = []
    for _ in range(draws):
        jittered = np.clip(a0 + rng.uniform(-0.5, 0.5, a0.shape), 0, 255)
        deltas.append(stat(_predict_adjust(jittered, key, factor)) - stat(np.round(jittered)))
    achievable = stat(_predict_adjust(a0, key, factor)) - stat(a0)
    return float(np.std(deltas) + abs(float(np.mean(deltas)) - achievable))


def _subsample(a, limit=400_000):
    """A strided view with at least `limit` pixels, for statistics that are global anyway.
    Verification must not cost more than the edit it checks. The stride avoids multiples of
    8 so it doesn't land on the same position in every JPEG block."""
    h, w = a.shape[:2]
    if h * w <= limit:
        return a
    step = int(np.ceil(np.sqrt(h * w / limit)))
    if step % 8 == 0:
        step += 1
    return a[::step, ::step]


def _check_adjust(before, after, key, direction='up', factor=None):
    """Grade an adjustment against what the requested factor can achieve on this image, not
    against a fixed threshold: a request the image has no headroom for is a pass, with the
    headroom noted, because the operation did everything it could. With no factor, falls
    back to a direction-only test (moved the right way by 5%)."""
    stat = _ADJUST_STAT[key]
    a0, a1 = _arr(before), _arr(after)
    if a0.shape != a1.shape:
        return None, 'image was resized; cannot compare an adjustment'
    # Same sampling for both, so before/after stay comparable pixel for pixel.
    a0, a1 = _subsample(a0), _subsample(a1)
    m0, m1 = stat(a0), stat(a1)
    if factor is None or factor == 1.0:
        ok = _b(m1 > m0 * 1.05 if direction == 'up' else m1 < m0 * 0.95)
        return ok, f'{key} {m0:.3f} -> {m1:.3f}'

    predicted = _predict_adjust(a0, key, factor)
    mp = stat(predicted)
    achievable, observed = mp - m0, m1 - m0
    noise = _quantisation_noise(a0, key, factor)
    # Two things look alike and must not share a verdict: a request that cannot move this
    # image at all (fail) vs. one too small to measure (not a failure, rounding is louder).
    tiny = 1e-6 * max(abs(m0), 1.0)
    if abs(achievable) <= tiny:
        # Push the same operation to its extreme: if that still can't move the image, the
        # request is impossible here; if it can, this factor just rounds away.
        extreme = 8.0 if factor > 1 else 0.0
        headroom = stat(_predict_adjust(a0, key, extreme)) - m0
        if abs(headroom) <= tiny:
            why = ('the image is greyscale, so there is no colour to scale'
                   if key == 'saturation' else
                   'every channel is already at the limit this would push it toward')
            return _b(False), (f'{key} x{factor:g} cannot change this image at all -- {why} '
                               f'({key} stays at {m0:.3f}). No factor would help; this '
                               f'request needs a different operation.')
    if abs(achievable) < _NOISE_GATE * noise:
        return None, (f'{key} x{factor:g} can move this image by at most {achievable:+.3f} '
                      f'({m0:.3f} -> {mp:.3f}), which 8-bit rounding alone moves by '
                      f'{noise:.3f}; the request is too small to verify on this image')
    achieved = observed / achievable
    lo, hi = _ACHIEVED_BAND
    ok = _b(lo <= achieved <= hi)
    detail = (f'{key} {m0:.3f} -> {m1:.3f}; x{factor:g} can reach {mp:.3f} on this image, '
              f'so {achieved:.0%} of the requested change happened')
    # Worth reporting the gap between what was asked and what the image can take: a bare
    # "verified" on a x4 that could only ever reach x1.25 is true but misleading.
    achievable_ratio = mp / m0 if m0 > 1e-9 else float('inf')
    if abs(achievable_ratio - factor) > 0.15 * abs(factor - 1) + 0.02:
        clipped = float((predicted >= 255).mean() if factor > 1 else (predicted <= 0).mean())
        detail += (f' -- note that x{factor:g} was asked for but only x{achievable_ratio:.2f} '
                   f'is reachable before clipping ({clipped:.0%} of channels at the limit)')
    return ok, detail


def check_brightness(before, after, direction='up', factor=None):
    return _check_adjust(before, after, 'brightness', direction, factor)


def check_contrast(before, after, direction='up', factor=None):
    return _check_adjust(before, after, 'contrast', direction, factor)


def check_saturation(before, after, direction='up', factor=None):
    return _check_adjust(before, after, 'saturation', direction, factor)


def check_vignette(before, after):
    """Corners must darken substantially more than the centre.

    Skips images too small to have a distinguishable corner: at 16x12 the corner sample
    `[:h//8, :w//8]` is an empty slice, whose mean is nan, and comparing against nan
    silently returns False -- a failure verdict manufactured from no data.
    """
    a0, a1 = _arr(before).mean(axis=2), _arr(after).mean(axis=2)
    h, w = a0.shape
    if h < 16 or w < 16:
        return None, f'image is {w}x{h}, too small to measure a corner against a centre'
    cen = (slice(int(h * .4), int(h * .6)), slice(int(w * .4), int(w * .6)))
    corner_before = float(a0[:h // 8, :w // 8].mean())
    if corner_before < 2.0:
        # Already-black corner can't darken further; a floor-divided ratio would score a
        # no-op as "0% brightness kept" and pass it.
        return None, (f'the corner is already black ({corner_before:.1f}/255), so darkening '
                      f'it further cannot be measured; look at the image instead')
    corner_ratio = float(a1[:h // 8, :w // 8].mean() / corner_before)
    centre_ratio = float(a1[cen].mean() / max(a0[cen].mean(), 1e-6))
    ok = _b(corner_ratio < centre_ratio * 0.85)
    return ok, f'corner kept {corner_ratio:.0%} brightness, centre kept {centre_ratio:.0%}'


def check_background_removed(before, after_rgba):
    """A real cutout: meaningful transparency, and the subject largely opaque."""
    if after_rgba.mode != 'RGBA':
        return False, f'expected RGBA output, got {after_rgba.mode}'
    alpha = np.array(after_rgba.split()[-1]).astype(float) / 255.0
    transparent = float((alpha < 0.1).mean())
    opaque = float((alpha > 0.9).mean())
    ok = _b(0.05 < transparent < 0.95 and opaque > 0.05)
    return ok, f'{transparent:.0%} transparent, {opaque:.0%} opaque'


# 0.85 is the loosest bar that still passes every correct edit in the corpus, while a no-op
# leaves 100%. The floor separates "no noise was removed" from "there was no noise to remove".
_DENOISE_BAND = 0.85
_FLAT_NOISE_FLOOR = 0.05


def check_denoise(before, after):
    """Noise must drop where noise lives, and the image must not be destroyed.

    Measured in the flat half of the picture, not the whole frame: denoising promises to
    smooth flat areas while keeping edges, so whole-frame energy grades it partly on detail
    it is meant to preserve, marking down a detailed photograph for doing its job.

    Guards below give no verdict (rather than a spurious one from arithmetic on no data)
    when there is no high-frequency detail to remove, or the flat regions are already
    noise-free, or the image is a constant colour (`corrcoef` on it is nan).
    """
    from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude
    a0, a1 = _arr(before).mean(axis=2), _arr(after).mean(axis=2)
    n0 = float(np.abs(a0 - gaussian_filter(a0, 2)).mean())
    if n0 < 0.05:
        return None, (f'the image has essentially no high-frequency detail ({n0:.3f}) and '
                      f'so nothing to denoise; nothing to verify')
    if a0.std() < 1e-6 or a1.std() < 1e-6:
        return None, 'image is a constant colour; structural correlation is undefined'
    if a0.shape != a1.shape:
        return False, f'denoise changed the image size, {a0.shape} -> {a1.shape}'
    grad = gaussian_gradient_magnitude(a0, 2)
    # `<=`, not `<`: on a synthetic figure more than half the gradients are exactly zero, so
    # a strict comparison selects no pixels and the mean of an empty slice is nan.
    flat = grad <= np.percentile(grad, 50)
    f0 = float(np.abs(a0 - gaussian_filter(a0, 2))[flat].mean()) if flat.any() else 0.0
    f1 = float(np.abs(a1 - gaussian_filter(a1, 2))[flat].mean()) if flat.any() else 0.0
    if not np.isfinite(f0) or not np.isfinite(f1) or f0 < _FLAT_NOISE_FLOOR:
        return None, (f'the flat regions are already noise-free ({f0:.3f}); there is '
                      f'nothing there to remove, so the request cannot be graded')
    structure = float(np.corrcoef(a0.ravel(), a1.ravel())[0, 1])
    ok = _b(f1 < f0 * _DENOISE_BAND and structure > 0.9)
    return ok, (f'noise in flat regions {f0:.2f} -> {f1:.2f} ({f1 / f0:.0%} of it left), '
                f'structural correlation {structure:.3f}')


def check_replace_colour(before, after, from_rgb, to_rgb, hue_tol=0.09, space='hsv',
                         oklab_tol=0.06):
    """The region the operation selected must shift measurably toward the target hue, while
    its own brightness (shading) is preserved.

    `space` selects the region for the same reason as `check_isolate_colour`: measuring in
    the wrong space grades a region the caller didn't ask for. Brightness is checked
    alongside the hue shift to distinguish a real hue-shift from a flat recolour that
    destroys folds and shadows.

    Compares distance-to-target before vs. after, not absolute proximity after: the latter
    would pass an unchanged image whenever `from_rgb` and `to_rgb` are already close in hue.
    That case (source and target already close) is reported as skipped instead.
    """
    from .photo import _to_hsv
    import colorsys
    a0, a1 = _arr(before), _arr(after)
    h0, s0, v0 = _to_hsv(a0)
    h1, s1, v1 = _to_hsv(a1)
    from_h = colorsys.rgb_to_hsv(*(np.array(from_rgb) / 255.0))[0]
    to_h = colorsys.rgb_to_hsv(*(np.array(to_rgb) / 255.0))[0]
    if space == 'oklab':
        from .perceptual import colour_mask
        region = colour_mask(before.convert('RGB'), tuple(int(c) for c in from_rgb),
                             tolerance=oklab_tol) > 0.5
    else:
        dh0 = np.minimum(np.abs(h0 - from_h), 1 - np.abs(h0 - from_h))
        region = (dh0 < hue_tol) & (s0 > 0.15)
    if region.sum() < 50:
        return None, 'the source colour is barely present; check skipped'

    def dist_to(hue_arr, target):
        return np.minimum(np.abs(hue_arr - target), 1 - np.abs(hue_arr - target))

    before_gap = float(np.median(dist_to(h0[region], to_h)))
    after_gap = float(np.median(dist_to(h1[region], to_h)))
    if before_gap < 0.03:
        return None, f'source and target hues already {before_gap:.3f} apart; movement not verifiable'
    improvement = 1 - after_gap / before_gap
    brightness_preserved = float(1 - np.abs(v1[region] - v0[region]).mean())
    # Share of selected pixels that moved at least halfway to the target, not just the
    # median: the median alone lets a half-recoloured edit pass (`evals/space_aware_checks.py`).
    moved = float((dist_to(h1[region], to_h) < dist_to(h0[region], to_h) * 0.5).mean())
    ok = _b(improvement > 0.6 and brightness_preserved > 0.85 and moved > 0.80)
    return ok, (f'hue gap to target {before_gap:.3f} -> {after_gap:.3f} '
                f'({improvement:.0%} closed), {moved:.0%} of the selected pixels moved, '
                f'brightness preserved {brightness_preserved:.0%}')


def check_unchanged_outside(before, after, region):
    """Everything outside `region` must be byte-identical -- for localised edits.

    The clamping is not defensive boilerplate: a negative coordinate silently inverts the
    meaning of the slice (`d[-8:188]` wraps instead of meaning "from 8px above the top"),
    which would count every changed pixel as a violation. Regions reaching past the image
    edge are normal input here, since callers pad boxes.
    """
    a0, a1 = np.array(before.convert('RGB')), np.array(after.convert('RGB'))
    if a0.shape != a1.shape:
        return _b(False), (f'image size changed {a0.shape[1]}x{a0.shape[0]} -> '
                           f'{a1.shape[1]}x{a1.shape[0]}; a localised edit must not resize')
    H, W = a0.shape[:2]
    x0, y0, x1, y1 = _clamp(region, W, H)
    d = (a0 != a1).any(axis=2)
    if x1 > x0 and y1 > y0:
        d[y0:y1, x0:x1] = False
    n = int(d.sum())
    return _b(n == 0), f'{n} px changed outside the target region'


# The same measurement, said in the operation's own words -- so a colour-pop's kept-colour
# pixels aren't reported with a recolour's phrasing.
_RECOLOURED = ('was recoloured', 'changed', 'kept its original colour')
_KEPT_COLOUR = ('kept its colour', 'kept its colour', 'was greyed out with the background')


def check_isolate_object(before, after, box, subject=None):
    """See below; `box` may be a single [x0,y0,x1,y1] or a list of them."""
    """Spatial colour-pop: some coherent part of the box keeps its colour, the rest of the
    frame loses it.

    `isolate_object` was the one flagship operation with no check at all, because it does not
    fit either shape the others have: it is not localised (desaturating everything OUTSIDE
    the box is its whole purpose, so `check_unchanged_outside` reports a correct edit as a
    131,100-pixel violation) and it is not hue-based (`check_isolate_colour` has no target
    hue to test against). Both properties still need testing, just as a pair.

    Measured as the FRACTION of coloured pixels in the box that kept their colour, not the
    mean retention across the box. The mean is what this checked first, and it fails correct
    work by construction: a box always contains background margin around the object, that
    margin is meant to be greyed, and it drags the mean down in proportion to how tight the
    box was. Measured on a synthetic disc filling 46 % of its box -- a visually perfect
    isolation -- the mean read "45 % retained" and was marked FAILED.

    What the fraction cannot check is whether the RIGHT thing inside the box kept its
    colour. That needs evidence the edit did not produce -- and `subject`, the caller's own
    words, is the one source of it here, exactly as on the recolour path: the pixels that
    kept their colour are graded against the region those words resolve to
    (`_completeness`). Without it the two detectable failures are still caught -- nothing
    kept its colour (the segmentation found no object) and nothing lost it (the
    desaturation never happened) -- and the note says the rest was not checked.

    The named region is narrowed to the pixels that HAD colour before the edit. A colour-pop
    cannot keep colour where there was none, and grading it against the whole phrase region
    would fail correct work on any object with a white or black part -- measured on the deck
    chair, whose white frame is half of what "the chair" resolves to.
    """
    a0, a1 = _arr(before), _arr(after)
    if a0.shape != a1.shape:
        return None, 'image was resized; cannot compare regions'
    H, W = a0.shape[:2]
    # One box or several: `isolate_object` accepts a list of regions ("the eyes").
    regions = box if box and isinstance(box[0], (list, tuple)) else [box]
    inside = np.zeros((H, W), bool)
    for region in regions:
        x0, y0, x1, y1 = _clamp(region, W, H)
        if x1 > x0 and y1 > y0:
            inside[y0:y1, x0:x1] = True
    if not inside.any():
        return None, f'{box} is empty or outside the image'
    s0, s1 = _sat(a0), _sat(a1)
    coloured = s0 > 0.12
    in_sel, out_sel = inside & coloured, (~inside) & coloured
    if in_sel.sum() < 50:
        return None, 'too little colour inside the box to measure; check skipped'
    if out_sel.sum() < 50:
        return None, 'too little colour outside the box to measure; check skipped'

    retention = s1[in_sel] / np.maximum(s0[in_sel], 1e-6)
    kept_frac = float((retention > 0.6).mean())
    dropped = float(s1[out_sel].mean() / max(s0[out_sel].mean(), 1e-6))
    ok = _b(kept_frac > 0.15 and dropped < 0.4)
    note = (f'{kept_frac:.0%} of the coloured pixels in the box kept their colour, '
            f'outside the box retained {dropped:.0%}')
    if not subject or ok is False:
        return ok, note
    kept = np.zeros((H, W), bool)
    kept[in_sel] = retention > 0.6
    union = (int(min(r[0] for r in regions)), int(min(r[1] for r in regions)),
             int(max(r[2] for r in regions)), int(max(r[3] for r in regions)))
    if len(regions) > 1:
        named, detail = _completeness_per_box(before, kept, regions, union, subject, note,
                                              eligible=coloured, words=_KEPT_COLOUR)
    else:
        named, detail = _completeness(before, kept, union, subject, note, eligible=coloured,
                                      words=_KEPT_COLOUR)
    if named is False:
        return _b(False), detail
    return (_b(True) if (ok is True and named is True) else None), detail


_EXTRACTED = ('survived the cutout', 'survived', 'was cut away with the background')


def check_object_extracted(before, after_rgba, box, subject=None):
    """The cutout is not empty, it is not the whole frame -- and, with words, it is the
    named thing that survived.

    Two failures are detectable without words: an empty cutout, and one that kept
    everything (segmentation gave up and returned the box). A third needs `subject`: an
    extraction of the background inside the box looks like a correct cutout from the
    outside. The alpha channel is graded against the region the words resolve to
    (`_completeness`), with no `eligible` narrowing -- unlike a colour-pop, a cutout can
    keep a white pixel as easily as a red one.
    """
    if after_rgba.mode != 'RGBA':
        return None, f'the result is {after_rgba.mode}, not RGBA; there is no cutout to check'
    a1 = np.asarray(after_rgba)
    if a1.shape[:2] != (before.size[1], before.size[0]):
        return None, 'the image was resized; the cutout cannot be compared with the original'
    kept = a1[:, :, 3] > 127
    frac = float(kept.mean())
    if not kept.any():
        return _b(False), 'the cutout is empty: every pixel is transparent'
    if frac > 0.98:
        return _b(False), (f'{frac:.0%} of the frame is opaque -- nothing was cut out, which '
                           f'is what a segmentation that gave up and kept its box produces')
    note = f'{int(kept.sum()):,} px survived the cutout, {frac:.0%} of the frame'
    if not subject:
        return None, (note + '; WHAT survived is not checked -- pass `subject` (your own '
                             'words for the thing) and it will be, or call '
                             'preview_object_mask and look')
    x0, y0, x1, y1 = (int(v) for v in box)
    return _completeness(before, kept, (x0, y0, x1, y1), subject, note, words=_EXTRACTED)


def check_background_replaced(before, after, colour_rgb, session=None):
    """The background became the requested colour, and the subject survived.

    Both halves are needed. Checking only "is the background the new colour" passes an
    output that is ENTIRELY the new colour -- a segmentation that found no subject and
    flood-filled the frame. Checking only "did the subject survive" passes an image that
    was not changed at all.
    """
    a0, a1 = _arr(before), _arr(after)
    if a0.shape != a1.shape:
        return None, 'image was resized; cannot compare regions'
    try:
        from . import advanced
        cut = advanced.remove_background(before, session=session)
        mask = np.array(cut.split()[-1]).astype(float) / 255.0
    except Exception as e:
        return None, f'could not derive subject mask ({type(e).__name__}); check skipped'

    bg, sub = mask < 0.2, mask > 0.8
    if bg.sum() < 100 or sub.sum() < 100:
        return None, 'subject or background too small to measure; check skipped'

    target = np.array(colour_rgb, dtype=float)
    bg_gap = float(np.abs(a1[bg] - target).max(axis=1).mean())
    subject_moved = float(np.abs(a1[sub] - a0[sub]).max(axis=1).mean())
    ok = _b(bg_gap < 12 and subject_moved < 12)
    return ok, (f'background is {bg_gap:.0f}/255 from the requested colour, '
                f'subject moved {subject_moved:.0f}/255')


def check_crop(before, after, box):
    """The output is exactly the requested region of the input -- not a resized whole image.

    Cheap, and it catches the mistake that actually happens with crops: coordinates given in
    the wrong convention (x/y swapped, or width/height instead of x1/y1) produce a plausible
    picture of the wrong part of the image, which nothing else here would notice.
    """
    a0 = np.array(before.convert('RGB'))
    a1 = np.array(after.convert('RGB'))
    H, W = a0.shape[:2]
    x0, y0, x1, y1 = _clamp(box, W, H)
    if x1 <= x0 or y1 <= y0:
        return None, f'box {tuple(box)} is empty or outside the image'
    expected = a0[y0:y1, x0:x1]
    if a1.shape != expected.shape:
        return False, (f'output is {a1.shape[1]}x{a1.shape[0]}, but the box asks for '
                       f'{expected.shape[1]}x{expected.shape[0]}')
    ok = _b(np.array_equal(a1, expected))
    return ok, (f'cropped to {expected.shape[1]}x{expected.shape[0]}'
                if ok else 'output does not match the requested region of the input')


# What counts as a recoloured pixel. 8/255 is the same order as the noise gate the adjust
# checks use: below it, JPEG rounding alone moves a pixel.
_CHANGE_GATE = 8

# Flattening: `recolour_object(preserve_shading=False)` writes one colour into the mask, so
# a dog's fur or a car's reflections come back as a silhouette, and every other check here
# passes it since 100% of the object did change colour. Measured as the median ratio of
# local standard deviation, after vs. before, over edited pixels that had texture to lose --
# flattened objects read ~0.00, correct recolours 0.71-1.26 (`evals/recolour_vs_generative.py`).
# Below 0.5, over half the object's texture is gone. Cannot see a replaced object (redrawn as
# something else): that keeps its own texture and reads 0.08-1.23.
_TEXTURE_WIN = 7
_TEXTURE_FLOOR = 4.0       # local sigma, /255: below this the ORIGINAL had nothing to lose
_TEXTURE_SHARE = 0.10      # of the edited pixels, or the check abstains
_TEXTURE_MIN_PX = 400
_TEXTURE_FAIL = 0.5


def _local_sigma(a, win=_TEXTURE_WIN):
    from scipy.ndimage import uniform_filter
    mu = uniform_filter(a, win)
    return np.sqrt(np.maximum(uniform_filter(a * a, win) - mu * mu, 0))


def texture_kept(before, after, changed):
    """How much of the edited region's texture survived: (ratio, textured px), or
    (None, textured px) when the original had too little texture for the question to mean
    anything -- a flat painted door, a chart bar, a clear sky."""
    la = _arr(before).astype(float).mean(axis=2)
    lb = _arr(after).astype(float).mean(axis=2)
    sa = _local_sigma(la)
    textured = changed & (sa > _TEXTURE_FLOOR)
    n = int(textured.sum())
    if n < _TEXTURE_MIN_PX or n < _TEXTURE_SHARE * max(int(changed.sum()), 1):
        return None, n
    sb = _local_sigma(lb)
    ratio = np.clip(sb[textured] / np.maximum(sa[textured], 1e-6), 0, 2)
    return float(np.median(ratio)), n


# `grounding.mask_for`'s own threshold and its own "found nothing" floor, the same numbers
# deliberately: this is the same region that tool would hand a caller, so a subject this
# check refuses to grade against is one `mask_for` would refuse to act on.
_PHRASE_THRESHOLD = 0.5
_NEEDS_EXTRA = ('without the grounding extra (pip install "figsurgeon[grounding]"), '
                'which is what turns those words into a region')


def check_object_recoloured(before, after, box, subject=None, boxes=None):
    """Did the recolour happen at all, how much of the box did it take -- and, when the
    caller says what they meant, how much of THAT changed?

    Not a completeness check on its own: `check_unchanged_outside` catches leakage out of
    the box, but the common failure with a box-driven segmenter is the opposite one (e.g.
    it finds only part of the object and containment still passes). `subject` -- the
    caller's own words, resolved to a region by `grounding.py` -- estimates the object's
    extent from different evidence than the box, so it survives a generous box; other ways
    of inferring the extent from the edit itself were tried and dropped for grading the box
    rather than the edit (`evals/object_completeness.py`).

    Abstains rather than guesses when the words find nothing in the box, cover the whole
    region, or disagree with the edit about which thing this is (a wrong-box symptom).
    Without `subject`, or without the grounding extra, reports only whether anything
    changed and how much of the box, with no completeness verdict.
    """
    a0, a1 = _arr(before), _arr(after)
    if a0.shape != a1.shape:
        return None, 'image was resized; cannot compare regions'
    H, W = a0.shape[:2]
    x0, y0, x1, y1 = _clamp(box, W, H)
    if x1 <= x0 or y1 <= y0:
        return None, f'box {tuple(box)} is empty or outside the image'
    d = np.abs(a0[y0:y1, x0:x1] - a1[y0:y1, x0:x1]).max(axis=2)
    n = int((d > _CHANGE_GATE).sum())
    if n == 0:
        # A no-op passes containment perfectly -- zero pixels changed outside a region zero
        # pixels changed inside. That is a true statement about a recolour that did not
        # happen, so it needs its own explicit failure here.
        return _b(False), ('nothing inside the box changed at all; either the '
                           'segmentation found no object to recolour there, or the colour '
                           'asked for is the one it already had')
    frac = n / float((x1 - x0) * (y1 - y0))
    took = (f'{n:,} px recoloured, {frac:.0%} of the edited region'
            + (' -- nearly the whole rectangle, which is what a segmenter returns '
               'when it gives up and keeps everything' if frac > 0.95 else ''))
    changed = np.zeros((H, W), bool)
    changed[y0:y1, x0:x1] = d > _CHANGE_GATE

    # Before the `subject` branch on purpose: flattening is measured on the pixels alone
    # and needs no phrase, so the friendliest call -- a box, a colour, no words -- gets
    # this verdict too.
    kept, textured = texture_kept(before, after, changed)
    if kept is not None:
        took += f'; texture kept {kept:.2f} where the object had any'
        if kept < _TEXTURE_FAIL:
            # Reported as a failure rather than folded into completeness: the object did
            # change colour, completely, and is ruined. See the note above `texture_kept`.
            return _b(False), (
                f'{took} -- the recolour FLATTENED the object: {textured:,} px of it '
                f'carried texture and {1 - kept:.0%} of that is gone, so it is a '
                f'silhouette of one colour rather than the same object in a new one. '
                f'preserve_shading=False does exactly this; if it was not set, the mask '
                f'may have swallowed a flat background')

    if not subject:
        return None, took
    if boxes and len(boxes) > 1:
        return _completeness_per_box(before, changed, boxes, (x0, y0, x1, y1), subject, took)
    return _completeness(before, changed, (x0, y0, x1, y1), subject, took)


# The verdict band: fraction of the named thing, inside the region the edit could touch,
# that changed. The estimate is out by a median of 0.05, and the gap below is where no
# threshold could honestly decide (`evals/object_completeness.py`).
_COMPLETE = 0.85            # essentially all of it changed
_INCOMPLETE = 0.60          # two fifths of it left behind -- a leg, a sleeve, a wheel

# The extent has to be corroborated by the edit before it can be used to grade it.
_CORROBORATED = 0.60
_DEGENERATE_EXTENT = 0.95   # the words cover the whole region; nothing left to separate


_EXTENT_PAD = 0.3      # context kept around the box when the phrase is resolved


def _phrase_probability(rgb, subject, region, pad=_EXTENT_PAD):
    """CLIPSeg's map for `subject`, resolved on the box plus `pad` of context rather than
    the whole frame. The model sees a fixed 352x352 input, so a small or thin object read
    on the whole frame comes back as a filled silhouette; resolving on the box puts it at a
    size the model can actually read."""
    import numpy as np
    from . import grounding
    W, H = rgb.size
    cx0, cy0, cx1, cy1 = _padded(region, (W, H), pad)
    if cx1 - cx0 < 8 or cy1 - cy0 < 8:
        return grounding._probability(rgb, subject)
    crop = grounding._probability(rgb.crop((cx0, cy0, cx1, cy1)), subject)
    # Outside the crop the phrase was not asked, so report zero rather than guess.
    full = np.zeros((H, W), dtype=float)
    full[cy0:cy1, cx0:cx1] = crop
    return full


def _completeness_per_box(before, changed, boxes, union, subject, took, eligible=None,
                          words=_RECOLOURED):
    """One verdict per box, not one for their union: a pooled average hides one box finding
    nothing behind another finding everything. The phrase is still resolved once, on the
    union plus context, so a plural phrase still works."""
    import numpy as np
    verdicts, details = [], []
    for i, b in enumerate(boxes):
        x0, y0, x1, y1 = (int(v) for v in b)
        window = np.zeros(changed.shape, bool)
        window[y0:y1, x0:x1] = True
        if not (changed & window).any():
            verdicts.append(False)
            details.append(f'box {i + 1}: nothing changed inside it')
            continue
        ok, detail = _completeness(before, changed & window, (x0, y0, x1, y1), subject,
                                   f'box {i + 1}', union=union, eligible=eligible,
                                   words=words)
        verdicts.append(ok)
        details.append(detail)
    joined = f'{took}; ' + ' | '.join(details)
    if any(v is False for v in verdicts):
        return _b(False), joined
    if all(v is True for v in verdicts):
        return _b(True), joined
    return None, joined


# Words that say which one rather than what it is. Stripped before the phrase is resolved,
# because the phrase is resolved on the box (see `_phrase_probability`), and inside one box
# there is nothing left for them to point at: "the sheep on the left" asked about a crop
# holding one sheep is answered with the left half of it (`evals/instance_phrases.py`).
_POSITION = re.compile(
    r"""(?ix)
    \s*\b(?:                       # ... on the left, at the front, in the background
        (?:on|at|in|to|from|toward|towards)\s+the\s+
        (?:far\s+|extreme\s+|very\s+)?
        (?:left|right|top|bottom|front|back|rear|middle|centre|center|
           foreground|background|near|far)
        (?:\s+hand)?(?:\s+side)?
      | (?:nearest|closest|next)\s+(?:to\s+)?(?:the\s+)?(?:camera|viewer|front)
      | in\s+front(?!\s+of)
    )\b\s*""")
_POSITION_ADJ = re.compile(
    r"""(?ix)
    ^(the|this|that)\s+(?:               # the middle sheep, the second saddle, the far chair
        first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|next|
        middle|centre|center|central|left|right|leftmost|rightmost|topmost|bottommost|
        upper|lower|top|bottom|front|back|rear|near|nearest|far|farthest|furthest|closest
    )\s+""")


def _drop_position(subject):
    """(what the thing is, the positional words removed) -- '' when there were none.
    Only locators go: an identifying attribute like "the red car" is untouched, and a
    phrase that is nothing BUT a position is also untouched (no object left to ask about)."""
    bare = re.sub(r"\s+", " ", _POSITION.sub(" ", subject)).strip()
    bare = _POSITION_ADJ.sub(lambda m: m.group(1) + " ", bare).strip()
    if bare == subject.strip():
        return subject, ""
    if not re.sub(r"(?i)^(the|this|that|a|an)\s*(one|ones|thing)?\b", "", bare).strip():
        return subject, ""           # "the one on the left" -- nothing left to ask about
    return bare, subject.strip()


def _completeness(before, changed, region, subject, took, union=None, eligible=None,
                  words=_RECOLOURED):
    """Grade the edit against the caller's words, minus any that only say WHICH one (see
    `_POSITION`). The full phrase is kept for the whole-frame probe inside
    `_completeness_of_phrase`, where a position IS answerable."""
    asked, position = _drop_position(subject)
    ok, note = _completeness_of_phrase(before, changed, region, subject, asked, took, union,
                                       eligible, words)
    if position:
        note += (f" (asked as {asked!r} -- inside the box there is nothing left for the "
                 f"rest of {position!r} to point at, so the box is what chose WHICH one "
                 f"and that part is not checked)")
    return ok, note


def _completeness_of_phrase(before, changed, region, spoken, subject, took, union=None,
                            eligible=None, words=_RECOLOURED):
    """Grade `changed` against the region the caller's WORDS point to.

    `union` is where to RESOLVE the phrase when it differs from where grading happens (the
    multi-box case, where asking about one box's crop alone matches nothing). `eligible`
    narrows the named region to pixels the operation COULD have acted on -- a colour-pop
    can only keep colour where there was colour, so grading it against the whole phrase
    region would fail correct work on an object with a white or black part.
    """
    from . import grounding
    did, acted, leftover = words
    if not grounding.available():
        return None, took + f'; cannot check that all of {subject!r} {did} ' + _NEEDS_EXTRA
    x0, y0, x1, y1 = region
    try:
        prob = _phrase_probability(before.convert('RGB'), subject, union or region)
    except Exception as e:                     # a download that failed, a model that moved
        return None, f'{took}; could not resolve {subject!r} ({type(e).__name__})'
    whole = prob > _PHRASE_THRESHOLD
    extent = np.zeros(prob.shape, bool)
    extent[y0:y1, x0:x1] = whole[y0:y1, x0:x1]
    if eligible is not None:
        extent &= eligible
    n_extent, n_whole = int(extent.sum()), int(whole.sum())
    floor = grounding.MIN_COVERAGE * prob.size
    if n_whole < floor:
        # "Here" is the box plus context, not the picture, so the two silences (nothing in
        # the box vs. nothing anywhere) must be told apart before either is stated; probe
        # the whole frame, once, with the caller's own words (position and all).
        frame = grounding._probability(before.convert('RGB'), spoken)
        if int((frame > _PHRASE_THRESHOLD).sum()) >= grounding.MIN_COVERAGE * frame.size:
            return None, (f'{took}; NO completeness verdict, and look at this edit: '
                          f'{subject!r} IS in this picture, but essentially none of it is '
                          f'in the region that {acted}. Either the box is in the wrong '
                          f'place or the words do not describe what you meant to edit')
        return None, (f'{took}; no completeness verdict: {subject!r} matches nothing in '
                      f'this picture (peak confidence {max(prob.max(), frame.max()):.2f}), '
                      f'so there is no independent idea of the object to grade the edit '
                      f'against')
    if n_extent < floor:
        # The words match something -- just not where the edit happened: the wrong-box
        # failure, distinct from "matches nothing".
        return None, (f'{took}; NO completeness verdict, and look at this edit: '
                      f'{subject!r} IS in this picture, but essentially none of it is in '
                      f'the region that {acted}. Either the box is in the wrong place or '
                      f'the words do not describe what you meant to edit')
    if n_extent > _DEGENERATE_EXTENT * (x1 - x0) * (y1 - y0):
        return None, (f'{took}; no completeness verdict: {subject!r} covers the whole '
                      f'region, so "how much of it {acted}" would just be the fraction '
                      f'above again')
    hit = int((extent & changed).sum())
    corroboration = hit / float(changed.sum())
    if corroboration < _CORROBORATED:
        return None, (f'{took}; NO completeness verdict, and look at this edit: only '
                      f'{corroboration:.0%} of what {acted} is inside {subject!r}. The box '
                      f'and the words disagree about which thing this is -- one of them is '
                      f'wrong, and nothing here can say which')
    k = hit / float(n_extent)
    detail = (f'{took}; {k:.0%} of {subject!r} {did} '
              f'(+/- 5 %, from the words rather than from the box)')
    if k >= _COMPLETE:
        return _b(True), detail
    if k <= _INCOMPLETE:
        return _b(False), detail + f' -- {1 - k:.0%} of it {leftover}'
    return None, (detail + ' -- too much to call complete, too little to call a failure; '
                           'look at it')


# The band `check_object_erased` reads its verdict from: the fraction of the named thing's
# own extent (established before the edit) that still reads as that thing after it.
# Calibrated against a rendered ground truth (`evals/erase_verdicts.py`, 8 objects x 6
# points): median error 0.03, but on 4 of the 8 objects (busy background, thin/skeletal, low
# contrast, or a person) the check can read 0% remaining while half the object is still
# there. A field survey of 24 real erases found the same failure with TELEA's fill (a
# faceted colour-block artifact CLIPSeg stops reading as the object) and, at a much lower
# rate, with LaMa's cleaner fills (`evals/erase_field.py`).
#
# So a low `remaining` returns None (look at it), not True: it means the words no longer
# find the thing, which is a different claim from "the thing is gone". The fail band is
# kept -- 0 false fails in calibration and the field survey -- and catches a partial box
# (e.g. only the handle erased) that containment alone calls clean.
_ERASE_GONE = 0.20         # below this the words no longer find it -- not a pass, see above
_ERASE_STILL_THERE = 0.55  # unmistakably still there -- the erase did not happen

# A structural pass signal (edge density inside the box vs. a ring just outside it) was
# tried and rejected -- kept as `evals/erase_structure.py`, not here -- because it produces
# a false pass on the real corpus: a true ghost can read as less structured than its own
# surround, the same signature the measure uses for a clean fill.


def _seam_distance(before, after, inner, outer):
    """Mean OKLab distance between `inner` (new pixels) and `outer` (original background).
    Shared tail of `fill_seam` and `fill_seam_mask`, which differ only in how the two sets
    are built."""
    if not inner.any() or not outer.any():
        return None
    a0, a1 = _arr(before), _arr(after)
    from . import perceptual
    return round(float(perceptual.delta_e(a1[inner].mean(0).astype(np.uint8),
                                          a0[outer].mean(0).astype(np.uint8))), 4)


def fill_seam(before, after, box, band=4):
    """How far a redrawn region's edge sits from the background it is meant to continue, as
    a mean OKLab distance between two thin bands either side of the box boundary. None when
    the box is too small to have two distinct bands. A number, not a verdict: see
    `check_generative_fill`."""
    x0, y0, x1, y1 = (int(v) for v in box)
    H, W = np.asarray(after).shape[:2]
    inner = np.zeros((H, W), bool)
    inner[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
    outer = np.zeros((H, W), bool)
    outer[max(0, y0 - band):min(H, y1 + band), max(0, x0 - band):min(W, x1 + band)] = True
    outer &= ~inner
    shrunk = np.zeros((H, W), bool)
    shrunk[max(0, y0 + band):min(H, y1 - band), max(0, x0 + band):min(W, x1 - band)] = True
    rim = inner & ~shrunk
    return _seam_distance(before, after, rim, outer)


def fill_seam_mask(before, after, mask, band=4):
    """`fill_seam`, generalised from a rectangular box to an arbitrary mask -- for
    `generative_fill`'s `mask_to_object` path, where only the segmented object's own pixels
    changed. None when the mask is empty or too thin to have two distinct rings.
    """
    from scipy.ndimage import binary_dilation, binary_erosion
    binary = np.asarray(mask) > 0.5
    if not binary.any():
        return None
    outer = binary_dilation(binary, iterations=band) & ~binary
    inner = binary & ~binary_erosion(binary, iterations=band)
    return _seam_distance(before, after, inner, outer)


def check_generative_fill(before, after, box, mask=None):
    """Always (None, detail). Containment holds by construction here (the paste guarantees
    nothing outside the box/mask changed), so reporting it as a pass would be a green
    verdict about the one property that could not have failed, while whether the invented
    content is right goes unmentioned. The seam distance is reported as a number instead,
    with the verdict left open.
    """
    scope = 'the segmented object inside the box' if mask is not None else 'the box'
    seam = fill_seam_mask(before, after, mask) if mask is not None \
        else fill_seam(before, after, box)
    where = '' if seam is None else f'; seam {seam:.3f} in OKLab (~0.02 is invisible, 0.1 is a clearly visible join)'
    return None, (f'a hosted image model invented this region. Nothing outside {scope} '
                  'changed -- that is guaranteed by how the paste works, not something '
                  'this check established, so it is not offered as a pass' + where +
                  '. Whether the fill is PLAUSIBLE is not checked: zoom in on the edges '
                  'and look')


def check_object_erased(before, after, box, subject=None):
    """Did `erase_object` remove the named thing, or does the box just look different now?

    Unlike the other object checks, the thing being graded is GONE if the edit worked, so
    grading against what changed doesn't work; containment already catches a no-op. What's
    left is a segmentation that found and inpainted only part of the object. CLIPSeg is
    asked the caller's words on the image before AND after the edit, at the same pixels
    (never against the inpainter's own mask -- see `check_object_recoloured`): if the object
    is really gone, nothing there should still look like it.

    Abstains, as the recolour check does, if the words match nothing before the edit.
    NO PASS VERDICT, deliberately: a low reading means the words no longer find the thing,
    which is a different claim from "the thing is gone" -- CLIPSeg stops recognising an
    object replaced by an inpainting smear just as readily as one genuinely absent. See the
    comment above `_ERASE_GONE` for the numbers behind this.
    """
    a0, a1 = _arr(before), _arr(after)
    if a0.shape != a1.shape:
        return None, 'image was resized; cannot compare regions'
    H, W = a0.shape[:2]
    x0, y0, x1, y1 = _clamp(box, W, H)
    if x1 <= x0 or y1 <= y0:
        return None, f'box {tuple(box)} is empty or outside the image'
    d = np.abs(a0[y0:y1, x0:x1] - a1[y0:y1, x0:x1]).max(axis=2)
    n = int((d > _CHANGE_GATE).sum())
    if n == 0:
        # A no-op passes containment perfectly -- zero pixels changed outside a region zero
        # pixels changed inside. See `check_object_recoloured` for the same trap.
        return _b(False), ('nothing inside the box changed at all; either the segmentation '
                           'found no object to erase there, or the erase was a no-op')
    took = f'{n:,} px changed, {n / float((x1 - x0) * (y1 - y0)):.0%} of the box'
    if not subject:
        return None, (took + '; WHAT was erased is not checked -- pass `subject` (your own '
                             'words for the thing) and it will be, or call '
                             'preview_object_mask and look')
    from . import grounding
    if not grounding.available():
        return None, took + f'; cannot check whether {subject!r} is gone ' + _NEEDS_EXTRA
    asked, position = _drop_position(subject)
    try:
        prob_before = _phrase_probability(before.convert('RGB'), asked, (x0, y0, x1, y1))
    except Exception as e:                     # a download that failed, a model that moved
        return None, f'{took}; could not resolve {subject!r} ({type(e).__name__})'
    extent = np.zeros((H, W), bool)
    extent[y0:y1, x0:x1] = prob_before[y0:y1, x0:x1] > _PHRASE_THRESHOLD
    floor = grounding.MIN_COVERAGE * prob_before.size
    if extent.sum() < floor:
        return None, (f'{took}; no completeness verdict: {asked!r} matches nothing in this '
                      f'box before the edit (peak {prob_before[y0:y1, x0:x1].max():.2f}), so '
                      f'there is no independent idea of the object to check the erase '
                      f'against')
    try:
        prob_after = _phrase_probability(after.convert('RGB'), asked, (x0, y0, x1, y1))
    except Exception as e:
        return None, (f'{took}; could not resolve {subject!r} after the edit '
                      f'({type(e).__name__})')
    remaining = float((prob_after[extent] > _PHRASE_THRESHOLD).mean())
    detail = (f'{took}; {remaining:.0%} of the original extent of {asked!r} still reads as '
              f'{asked!r} after the erase (+/- 5 %, from the words rather than from the box)')
    if position:
        detail += (f" (asked as {asked!r} -- inside the box there is nothing left for the "
                   f"rest of {position!r} to point at, so the box is what chose WHICH one "
                   f"and that part is not checked)")
    if remaining >= _ERASE_STILL_THERE:
        return _b(False), detail + ' -- the erase did not remove it'
    if remaining <= _ERASE_GONE:
        # NOT a pass. Two studies say a low reading here is not evidence the thing is gone
        # -- see the comment above the bands. The words no longer find it; that is all this
        # can honestly say, and it is the caller's cue to look, not a verdict.
        return None, (detail + ' -- the words no longer find it there, but that is NOT a '
                               'verdict that it is gone. Both inpainters can read this way '
                               'without removing anything: TELEA leaves a smear in its '
                               'shape, and LaMa can regenerate the object itself (a faded '
                               'backpack, a redrawn umbrella) or invent scenery that was '
                               'never there. Look at the picture')
    return None, (detail + ' -- too much left to call it gone, too little to call it still '
                           'there; look at it')


def check_object_scaled(before, after, region, scale):
    """Did the object's own extent change by roughly `scale`? Reports the number, not a verdict.

    A resize doesn't change colour anywhere, so the pixel-diff trick the other object checks
    lean on can't see it. `segment_object` is instead run independently on `before` and
    `after` over the same `region`, and the two mask areas compared (`scale**2`, since area
    scales with the square of a linear factor).

    NO PASS/FAIL BAND: re-segmenting can legitimately disagree by more than a real scale
    error would move the number, and there is no labelled set to calibrate a cut against.
    The measured ratio and the ratio `scale` implies are both reported instead. The one case
    this CAN decide: the object was found in `before` but nothing is left in `region` in
    `after` -- no `scale` a caller would sensibly ask for produces that, so it is a genuine
    failure.
    """
    from . import objects as O
    try:
        seg0 = O.segment_object(before, tuple(int(v) for v in region))
        seg1 = O.segment_object(after, tuple(int(v) for v in region))
    except ValueError as e:
        return None, f'could not re-segment {tuple(region)} to measure the scale ({e})'
    area0 = float((seg0[0] > 0.5).sum())
    area1 = float((seg1[0] > 0.5).sum())
    if area0 == 0:
        return None, (f'could not find the object inside {tuple(region)} in the ORIGINAL '
                      f'image to measure against')
    if area1 == 0:
        return _b(False), (f'no object found inside {tuple(region)} after scaling by '
                           f'{scale}x -- not what any scale factor should produce; check '
                           f'with preview_object_mask')
    ratio = area1 / area0
    expected = scale ** 2
    return None, (f'object area {area0:.0f} -> {area1:.0f} px ({ratio:.2f}x by area; '
                  f'scale={scale} implies {expected:.2f}x). Not graded -- re-segmenting '
                  f'before and after can disagree by more than a real scale error would '
                  f'move this number. Compare it to what you asked for and look at the '
                  f'picture.')


# The quantity is the share of the box's colormap content still on the OLD colour scale
# after the remap. Measured over 27 published figures (`evals/_corpus/audit_remap`): true
# survivors read 1.0-3.0%, clean re-themes 0.00-0.27%, so 1.5% separates them.
_OLD_SCALE_BAD = 0.015
_OLD_SCALE_OK = 0.005
_VALUE_DRIFT = 0.02       # LUT quantisation alone is 1/128 = 0.008; see the eval


def check_colormap_remapped(before, after, box=None, source_cmap='viridis',
                            new_colours=('#00407A', '#52BDEC'), alpha=None):
    """Did the remap replace the WHOLE colour scale, and does each colour still mean what
    it meant?

    Containment alone misses two real failure modes: a band of values that sits just
    outside `match_tolerance` and is skipped (mixing old and new scales), and the value
    encoding shifting (each changed pixel's LUT position compared before vs. after).

    Both sides are matched in the space the figure is actually drawn in: `remap_colormap`
    re-blends at the alpha it found, so pixels sit on the new gradient blended over white,
    not the pure gradient -- matching against the pure one biases every position toward the
    middle of the scale. The alpha is detected from `before` when not given.

    "Colormap content" is every pixel that either changed or is colourful and still close to
    the source colormap, so a remap that skipped a band is still seen. Returns False when a
    band survived or the values moved, True when neither did, None in between.
    """
    import numpy as np
    from scipy.ndimage import binary_erosion, label as cclabel
    from .rebrand import _cmap_lut, build_gradient, _nearest_in_lut
    a = np.asarray(before.convert('RGB')).astype(float)
    c = np.asarray(after.convert('RGB')).astype(float)
    if a.shape != c.shape:
        return None, 'the image was resized; a colormap remap cannot be checked against it'
    if box is None:
        box = (0, 0, a.shape[1], a.shape[0])
    x0, y0, x1, y1 = (int(v) for v in box)
    A, C = a[y0:y1, x0:x1], c[y0:y1, x0:x1]
    if A.size == 0:
        return None, f'box {tuple(box)} selects no pixels'

    changed = np.abs(A - C).max(axis=2) > 2
    if alpha is None:
        from .rebrand import detect_colormap_alpha
        alpha, _err = detect_colormap_alpha(before, (x0, y0, x1, y1), source_cmap)
    src_pure, new_pure = _cmap_lut(source_cmap), build_gradient(new_colours)
    src = _blend_lut(src_pure, alpha)
    new = _blend_lut(new_pure, alpha)
    # `_nearest_in_lut`'s cache is keyed by name; the alpha is part of what this LUT is, so
    # it must be part of the key or a blended LUT would silently answer from the pure one.
    src_key = f'{source_cmap}@{alpha:.2f}'
    idx_before, dist_before = _nearest_in_lut(A.reshape(-1, 3), src, key=src_key)
    dist_before = dist_before.reshape(A.shape[:2])
    if alpha < 1.0:
        # Not every part of a figure is drawn at the same alpha (e.g. a blended plot next
        # to an opaque colorbar), so a pixel counts as old-scale if it matches the source
        # at EITHER rendering; only the drift measurement stays in blended space.
        _, dist_pure = _nearest_in_lut(A.reshape(-1, 3), src_pure, key=f'{source_cmap}@1.00')
        dist_before = np.minimum(dist_before, dist_pure.reshape(A.shape[:2]))
    colourful = (A.max(axis=2) - A.min(axis=2)) > 30
    # 60 rather than the operation's own 15: has to be wider than the window it matched
    # with, to catch what it missed. Neutral pixels excluded: axis lines and gridlines are
    # legitimately untouched.
    left = (~changed) & colourful & (dist_before < 60)
    # Erode by a pixel first: anti-aliased contour lines are colourful, unchanged, and near
    # the source colormap, so without this the mask traces line-work instead of a surviving
    # band. A line trace doesn't survive erosion; a band does.
    left = binary_erosion(left, structure=np.ones((3, 3)), iterations=1)
    content = changed | left
    n_content = int(content.sum())
    if n_content == 0:
        return False, (f'nothing in {tuple(box)} matches {source_cmap!r}, and nothing '
                       f'changed: either the box misses the colormap or the source '
                       f'colormap is not the one in this figure')
    if not changed.any():
        return False, (f'no pixel changed, though {n_content:,} in the box look like '
                       f'{source_cmap!r}')

    share = left.sum() / float(n_content)
    idx_after, _ = _nearest_in_lut(C[changed], new)
    pos_before = idx_before.reshape(A.shape[:2])[changed] / (len(src) - 1)
    pos_after = idx_after / (len(new) - 1)
    drift = float(np.median(np.abs(pos_before - pos_after)))

    detail = (f'{int(changed.sum()):,} px remapped, {share:.1%} of the colormap content '
              f'left on the old scale; the value each colour encodes moved by {drift:.3f} '
              f'of the scale (median)')

    # The colorbar is usually outside the box, and a key still in the old colours is wrong
    # in the way that matters most (cells say one thing, scale says another) but invisible
    # to every measurement above. Reported as a FACT, not a verdict: remapping the plot and
    # the colorbar are two calls, and failing the first would be crying wolf mid-edit.
    outside = np.ones(a.shape[:2], bool)
    outside[y0:y1, x0:x1] = False
    rest = a[outside].reshape(-1, 3)
    idx_src, dist_src = _nearest_in_lut(rest, src, key=src_key)
    if alpha < 1.0:
        idx_pure, dist_pure = _nearest_in_lut(rest, src_pure, key=f'{source_cmap}@1.00')
        closer = dist_pure < dist_src
        idx_src = np.where(closer, idx_pure, idx_src)
        dist_src = np.minimum(dist_src, dist_pure)
    _, dist_new = _nearest_in_lut(rest, new)
    # Closer to the source scale than to the new one, not merely near it: the two overlap,
    # and a brand gradient of two blues lies within tolerance of viridis's blue end.
    still = (dist_src < 15) & (dist_src < dist_new) & ((rest.max(axis=1) - rest.min(axis=1)) > 30)
    n_still = int(still.sum())
    # Has to be a SCALE, not a colour: a surviving colorbar or plot area spans the RANGE of
    # the old scale, while an artifact (e.g. an anti-aliased gridline fringe) is one colour.
    spread, biggest = 0, 0.0
    if n_still:
        bins = np.bincount((idx_src[still] * 20 // len(src)).astype(int), minlength=20)
        spread = int((bins > 0.02 * n_still).sum())
        # And has to be ONE thing: a colorbar or panel is a single connected region, while
        # coincidental matches on an unrelated scale are scattered over the figure.
        region = np.zeros(a.shape[:2], bool)
        region[outside] = still
        labels, _ = cclabel(region, structure=np.ones((3, 3)))
        sizes = np.bincount(labels.ravel())[1:]
        biggest = float(sizes.max() / n_still) if len(sizes) else 0.0
    if (n_still > 0.0005 * a.shape[0] * a.shape[1] and spread >= 5 and biggest >= 0.5):
        ys, xs = np.where(outside)
        sy, sx = ys[still], xs[still]
        detail += (f'. NOTE: {n_still:,} px OUTSIDE the box still carry {source_cmap!r} '
                   f'across {spread}/20 of its scale, {biggest:.0%} of them in one region '
                   f'(around x {sx.min()}-{sx.max()}, y {sy.min()}-{sy.max()}) -- if that is '
                   f'the colorbar, the figure now carries two colour scales and its key no '
                   f'longer matches its cells. Remap it too')
    if drift > _VALUE_DRIFT:
        return _b(False), (detail + ' -- the remap did NOT preserve the value encoding, '
                           'which is the one thing it promises')
    if share >= _OLD_SCALE_BAD:
        lo, hi = np.percentile(idx_before.reshape(A.shape[:2])[left] / (len(src) - 1),
                               [10, 90])
        if share > 0.5:
            # Not a band: almost nothing matched, so the colormap named is probably wrong.
            return _b(False), (detail + f' -- almost none of it was remapped. {source_cmap!r} '
                               f'is probably not the colormap in this figure')
        return _b(False), (detail + f' -- a BAND of values survived in the old colours '
                           f'(around {lo:.2f}-{hi:.2f} of the scale). The figure now '
                           f'carries two colour scales at once; a reader cannot tell which '
                           f'one a cell belongs to')
    if share <= _OLD_SCALE_OK:
        return _b(True), detail
    return None, (detail + ' -- too much of the old scale left to call it clean, too little '
                  'to call it a failure; look at the figure at 1:1')


def _blend_lut(lut, alpha, background=255.0):
    """The LUT as it appears ON SCREEN at this alpha -- `rebrand._unblend`'s inverse, and
    the space both sides of the drift measurement have to be compared in."""
    return lut * alpha + background * (1.0 - alpha)


_SWATCH_FLOOR = 70        # the same glyph/bleed separation `clean_transparent_box` uses
_GHOST_BAND = (8, 70)     # tinted, but too faint to be a drawn glyph: bleed-through


def check_transparent_box_cleaned(before, after, box, glyph_saturation_floor=_SWATCH_FLOOR):
    """Did the legend lose its bleed-through -- and did it keep its key?

    Containment alone misses two real failures (`evals/legend_clean.py`): a swatch merged
    with adjacent bleed-through and got erased as a ghost, or the sampled fill colour came
    from bleeding text instead of the box. So two things are measured instead: the count of
    drawn glyphs (swatches) must not FALL, and the ghost-band tint (bleed-through) must go.

    Returns False when a swatch disappeared or the box got darker; True when ghosts dropped
    and every swatch survived; None when nothing measurable happened.
    """
    import numpy as np
    from scipy import ndimage
    a = np.asarray(before.convert('RGB')).astype(int)
    c = np.asarray(after.convert('RGB')).astype(int)
    if a.shape != c.shape:
        return None, 'the image was resized; the legend box cannot be compared'
    x0, y0, x1, y1 = (int(v) for v in box)
    A, C = a[y0:y1, x0:x1], c[y0:y1, x0:x1]
    if A.size == 0:
        return None, f'box {tuple(box)} selects no pixels'

    def swatch_area(region, scale):
        """Area in blobs big enough to BE a swatch (at least half the largest one), and the
        size of the largest. Counting blobs directly cries wolf: small anti-aliased text
        specks come and go with any cleaning and are a different order of size than a
        swatch."""
        sat = region.max(axis=2) - region.min(axis=2)
        labels, n = ndimage.label(sat > glyph_saturation_floor, structure=np.ones((3, 3)))
        sizes = np.array([(labels == i).sum() for i in range(1, n + 1)]) if n else np.array([])
        if not len(sizes):
            return 0, 0
        cut = scale if scale else max(4, sizes.max() / 2)
        return int(sizes[sizes >= cut].sum()), int(sizes.max())

    def ghost_share(region, fill=None):
        """Share of the box that is neither fill nor glyph -- by CHROMA or by LIGHTNESS.
        Chroma alone is blind to bleed-through with little chroma but much darker than the
        fill (grey pavement or black clothing behind a white legend)."""
        sat = region.max(axis=2) - region.min(axis=2)
        ghost = (sat > _GHOST_BAND[0]) & (sat <= _GHOST_BAND[1])
        if fill is not None:
            off = np.abs(region - fill).max(axis=2)
            ghost |= (off > _GHOST_BAND[0]) & (sat <= _GHOST_BAND[1])
        return float(ghost.mean())

    def fill_of(region):
        """The box's own fill: the modal light, low-chroma colour, as the cleaner reads it."""
        sat = region.max(axis=2) - region.min(axis=2)
        pool = (region.max(axis=2) >= 210) & (sat < 6)
        if pool.sum() < 20:
            return None
        values, counts = np.unique(region[pool].reshape(-1, 3), axis=0, return_counts=True)
        return values[int(np.argmax(counts))].astype(float)

    _, biggest = swatch_area(A, 0)
    scale = max(4, biggest / 2)
    area_before, _ = swatch_area(A, scale)
    area_after, _ = swatch_area(C, scale)
    kept = area_after / float(area_before) if area_before else 1.0
    fill = fill_of(C) if fill_of(C) is not None else fill_of(A)
    ghost_before, ghost_after = ghost_share(A, fill), ghost_share(C, fill)
    light_before = float(A.max(axis=2).mean())
    light_after = float(C.max(axis=2).mean())

    detail = (f'{kept:.0%} of the legend\'s drawn glyphs kept; '
              f'bleed-through {ghost_before:.1%} of the box -> {ghost_after:.1%}')

    if np.array_equal(A, C):
        return None, 'nothing inside the box changed'
    if light_after < light_before - 20:
        return _b(False), (detail + f'; the box got DARKER (mean lightness {light_before:.0f} '
                           f'-> {light_after:.0f}). The fill colour was probably sampled '
                           f'from the text rather than the box')
    # 0.85: losing ONE of three swatches costs a third of this area, and the specks that
    # come and go cost under 2 % -- measured on three legends over real figures, where the
    # good outputs keep 99-100 % and the two broken ones keep 33 % and 0 %.
    if kept < 0.85:
        return _b(False), (detail + ' -- the legend LOST a swatch. Cleaning removed a drawn '
                           'glyph along with the bleed-through, so the key no longer says '
                           'what the colours mean')
    if ghost_after > ghost_before * 0.5:
        return None, (detail + ' -- most of the bleed-through is still there; the box may be '
                      'off the legend, or the tint too faint to separate from the fill')
    return _b(True), detail


# Measured over sixteen photographs, phrase against segmenter (`evals/grounding.py`):
# 0.79-0.95 where the phrase mask is a usable object mask, 0.22-0.65 on three where it
# is not (the phrase region swallowed extra context, e.g. a boat's rigging or a chair's floor).
_CORROBORATED_SHAPE = 0.75
_DISAGREES = 0.55


def check_subject_recoloured(before, after, subject, to_rgb=None):
    """Did the thing the WORDS picked out have the shape of an object?

    `grounding.recolour_subject` paints a phrase-resolved region with no box, so a bad match
    (e.g. "recolour the guitar" bleeding onto the player's arms) has nothing to check it
    against. What can honestly be checked is not whether the words found the right object
    (that would share evidence with the mask), but whether a SECOND model, given no words,
    agrees about the shape: the painted region is handed to `objects.segment_object` as a
    box and compared. Agreement is NOT confirmation; disagreement is the real signal, and
    is what the failing band reports. Costs one extra segmentation call (~3s on CPU).
    """
    import re
    import numpy as np
    from scipy.ndimage import label as cclabel
    from skimage.morphology import convex_hull_image
    from . import objects
    a = np.asarray(before.convert('RGB')).astype(int)
    c = np.asarray(after.convert('RGB')).astype(int)
    if a.shape != c.shape:
        return None, 'the image was resized; the edited region cannot be compared'
    changed = np.abs(a - c).max(axis=2) > 8
    n = int(changed.sum())
    if n < 50:
        return _b(False), (f'nothing changed: {subject!r} either matched nothing or the '
                           f'recolour had no effect')
    took = f'{n:,} px recoloured ({changed.mean():.1%} of the frame)'
    if not objects.available():
        return None, took + '; ' + _NEEDS_EXTRA + ' to corroborate the shape'

    ys, xs = np.where(changed)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    if (box[2] - box[0]) < 8 or (box[3] - box[1]) < 8:
        return None, took + '; the edited region is too small to corroborate'
    try:
        mask = np.asarray(objects.segment_object(before, box)[0]) > 0.5
    except Exception as exc:                       # a segmenter failure is not a verdict
        return None, f'{took}; could not corroborate the shape ({type(exc).__name__})'

    iou = float((changed & mask).sum() / max((changed | mask).sum(), 1))
    detail = (f'{took}; a segmenter given only the edited region, and none of the words, '
              f'agrees with its shape at IoU {iou:.2f}')

    # Three kinds of request this comparison cannot judge, so it says so rather than
    # forcing a verdict: several separate regions (a segmenter given one box can't
    # reproduce that shape), a part of a thing (segments whole objects), and a skeletal
    # object (scores low against any blob-shaped second opinion regardless of correctness).
    pieces, n_pieces = cclabel(changed, structure=np.ones((3, 3)))
    sizes = np.bincount(pieces.ravel())[1:]
    substantial = int((sizes > 0.15 * n).sum()) if len(sizes) else 0
    # Skeletal, measured as area against the region's own convex hull: compact objects run
    # much higher than something mostly rigging or wire.
    hull = convex_hull_image(changed) if n > 40 else None
    solidity = float(n / max(hull.sum(), 1)) if hull is not None else 1.0
    part_of = bool(re.search(r"\b\w+'s\s+\w+|\b(?:part|edge|tip|corner|handle|wheel|"
                             r"sleeve|collar|ear|tail|leg|arm|roof|lid)\b",
                             (subject or '').lower()))
    if substantial > 1:
        return None, (detail + f' -- but the edit is {substantial} separate regions, and a '
                      f'segmenter prompted with one box around all of them cannot reproduce '
                      f'that shape. No verdict on {subject!r}: look at it')
    if part_of:
        return None, (detail + f' -- but {subject!r} names PART of a thing, and the second '
                      f'opinion segments whole objects, so a low score here is expected and '
                      f'means nothing. No verdict: look at it')
    if solidity < 0.40:
        return None, (detail + f' -- but the edited region is skeletal (it fills '
                      f'{solidity:.0%} of its own convex hull), and a blob-shaped second '
                      f'opinion cannot match that either way. No verdict: look at it')
    if iou >= _CORROBORATED_SHAPE:
        return _b(True), (detail + ' -- an object-shaped region, corroborated by a model '
                          'that cannot see the phrase. That is not the same as the RIGHT '
                          'object: look at it if the phrase was ambiguous')
    if iou <= _DISAGREES:
        return _b(False), (detail + f' -- the two disagree about what {subject!r} is. The '
                           f'painted region is probably not that object: look at it before '
                           f'keeping this edit')
    return None, (detail + ' -- too close to call: the region is roughly object-shaped but '
                  'the two models differ about its edges')
