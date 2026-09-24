"""General-purpose photo edits, for real photographs rather than flat-colour figures.

Unlike compose.py's exact-colour engine (built for a chart's small palette), this module
selects by perceptual proximity in HSV space, tolerating a photo's gradients and noise.
"""
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import colorsys
import numbers

from .composite import _arr, _blend, _clamp


def _to_hsv(a):
    """Vectorised RGB -> HSV (0..1), without a Python-level loop over pixels."""
    r, g, b = a[..., 0] / 255.0, a[..., 1] / 255.0, a[..., 2] / 255.0
    mx, mn = np.max(a, axis=-1) / 255.0, np.min(a, axis=-1) / 255.0
    v = mx
    d = mx - mn
    s = np.where(mx == 0, 0, d / np.where(mx == 0, 1, mx))
    h = np.zeros_like(mx)
    mask = d != 0
    rc, gc, bc = np.zeros_like(mx), np.zeros_like(mx), np.zeros_like(mx)
    dd = np.where(mask, d, 1)
    rc = (mx - r) / dd
    gc = (mx - g) / dd
    bc = (mx - b) / dd
    h = np.select(
        [mask & (mx == r), mask & (mx == g), mask & (mx == b)],
        [(bc - gc), 2.0 + (rc - bc), 4.0 + (gc - rc)],
        default=0.0)
    h = (h / 6.0) % 1.0
    return h, s, v


def _feather(mask, sigma):
    """Soften a mask's edge outward without weakening thin selected regions.

    A plain gaussian blur of the mask spreads its weight outward everywhere, so anything
    thinner than the kernel loses strength. Taking max(hard mask, blurred mask) keeps every
    selected pixel at full strength and blurs only the falloff outside.
    """
    from scipy.ndimage import gaussian_filter
    hard = mask.astype(float)
    if sigma <= 0:
        return hard
    return np.clip(np.maximum(hard, gaussian_filter(hard, sigma=sigma)), 0, 1)


def sample_colour(img, box):
    """Return the median RGB colour in `box` -- for resolving "the orange of her shirt"
    style references, where the target colour comes from a region, not a name."""
    a = _arr(img)
    # Clamped: a negative coordinate is a slice from the other end, so (-5, 0, W, H) would
    # silently read a strip at the right edge instead.
    x0, y0, x1, y1 = _clamp(box, a.shape[1], a.shape[0])
    region = a[y0:y1, x0:x1].reshape(-1, 3)
    if region.size == 0:
        raise ValueError(f'box {box} contains no pixels')
    return tuple(int(v) for v in np.median(region, axis=0))


def _hsv_to_rgb(h, s, v):
    """Vectorised HSV -> RGB (numpy arrays in, numpy array out), for recolouring every
    pixel of an image at once instead of per-pixel colorsys calls."""
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = (i % 6).astype(int)
    conds = [i == k for k in range(6)]
    r = np.select(conds, [v, q, p, p, t, v])
    g = np.select(conds, [t, v, v, q, p, p])
    b = np.select(conds, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1) * 255.0


def _resolve_colour(colour, where):
    """Resolve a colour name to (r, g, b) via `photo_describe.COLOUR_WORDS` (lazy import:
    photo_describe imports this module)."""
    if not isinstance(colour, str):
        return colour
    from .photo_describe import COLOUR_WORDS
    try:
        return COLOUR_WORDS[colour.strip().lower()]
    except KeyError:
        raise ValueError(
            f'{where}: unknown colour name {colour!r}. Known names are '
            f"{', '.join(sorted(COLOUR_WORDS))}. Or pass an (r, g, b) triple -- "
            f'sample_colour(img, box) reads one off the image itself.') from None


def _oklab_selection(img, colour, oklab_tol, where):
    """Perceptual selection mask for `space='oklab'`; see perceptual.py.

    No feathering: `perceptual.colour_mask` already fades out with perceptual distance.
    """
    from . import perceptual
    if isinstance(colour, float):
        raise ValueError(
            f"{where}: space='oklab' selects by perceptual distance to a COLOUR, so it "
            f"needs an (r, g, b) target, not the bare hue {colour}. Either pass an RGB "
            f"triple -- sample_colour() on a region of the image gives one -- or keep the "
            f"default space='hsv', which is what a bare hue means.")
    return perceptual.colour_mask(img, colour, tolerance=oklab_tol)


def replace_colour(img, from_colour, to_colour, hue_tol=0.07, sat_floor='auto',
                    feather=4, hue_blend=1.0, space='hsv', oklab_tol=0.06):
    """Recolour pixels near `from_colour`'s hue to `to_colour`'s hue. Each pixel keeps its
    own saturation and value (shading, highlights); only hue moves.
    `to_colour` is usually sampled from a real region via `sample_colour()`, not a fixed
    name, so its natural variation carries over.
    `hue_blend` (0..1): 1.0 fully replaces the hue, lower blends toward the original --
    useful when `to_colour` came from a small, unrepresentative patch.
    `space` picks how pixels are selected: 'hsv' (default) is a hue window at `hue_tol`;
    'oklab' is perceptual distance at `oklab_tol` and follows an object through its shading,
    but is not intuitive on a neutral target since hue is meaningless without chroma.
    """
    from_colour = _resolve_colour(from_colour, 'replace_colour')
    to_colour = _resolve_colour(to_colour, 'replace_colour')
    a = _arr(img)
    h, s, v = _to_hsv(a)
    if space == 'oklab':
        soft = _oklab_selection(img, from_colour, oklab_tol, 'replace_colour') * hue_blend
        to_h = to_colour if isinstance(to_colour, float) else \
            colorsys.rgb_to_hsv(*(np.array(to_colour) / 255.0))[0]
        recoloured = _hsv_to_rgb(np.full_like(h, to_h), s, v)
        out = _blend(a, recoloured, soft)
        return _keep_alpha(img, Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))), soft
    if space != 'hsv':
        raise ValueError(f"replace_colour: space must be 'hsv' or 'oklab', not {space!r}")
    from_h = from_colour if isinstance(from_colour, float) else \
        colorsys.rgb_to_hsv(*(np.array(from_colour) / 255.0))[0]
    to_h = to_colour if isinstance(to_colour, float) else \
        colorsys.rgb_to_hsv(*(np.array(to_colour) / 255.0))[0]

    dh = np.minimum(np.abs(h - from_h), 1 - np.abs(h - from_h))
    if sat_floor == 'auto':
        nontrivial = s[s > 0.02]
        sat_floor = float(np.clip(np.percentile(nontrivial, 15), 0.03, 0.18)) \
            if nontrivial.size else 0.05
    match = (dh < hue_tol) & (s > sat_floor)

    soft = _feather(match, feather) * hue_blend

    new_h = np.full_like(h, to_h)
    recoloured = _hsv_to_rgb(new_h, s, v)          # same S, V -- only hue moves
    out = _blend(a, recoloured, soft)
    return _keep_alpha(img, Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))), soft


def _grey_outside(a, soft, v, grey_strength, flatten):
    """Keep `a` where the mask is, grey it elsewhere -- the tail of every colour-pop."""
    lum = v * 255
    flat_grey = np.full_like(lum, 190.0)
    grey_level = lum * (1 - flatten) + flat_grey * flatten
    grey = np.repeat(grey_level[..., None], 3, axis=2)
    out = a * soft[..., None] + grey * (1 - soft[..., None]) * grey_strength \
        + a * (1 - soft[..., None]) * (1 - grey_strength)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def isolate_colour(img, colour_name_or_rgb, hue_tol=0.07, sat_floor='auto',
                    grey_strength=1.0, feather=6, flatten=0.5, space='hsv', oklab_tol=0.06):
    """"Colour pop": keep pixels near a target hue in colour, desaturate the rest.
    `colour_name_or_rgb` is (r, g, b), a bare hue, or a name from `photo_describe.COLOUR_WORDS`
    -- a coarse guess; use `sample_colour(img, box)` for precision, especially with 'oklab'.
    `sat_floor='auto'` sets the near-grey cutoff from the image's own saturation, since no
    fixed constant works across subjects; pass a float to override.
    `flatten` blends per-pixel luminance (0.0) toward flat mid-grey (1.0, the stylised
    "colour-splash" look); default 0.5.
    `space`: 'hsv' (default) is a hue window (`hue_tol`); 'oklab' is perceptual distance
    (`oklab_tol`) and tracks a photo subject through its shading better. `sat_floor` and
    `feather` do not apply in 'oklab'.
    """
    colour_name_or_rgb = _resolve_colour(colour_name_or_rgb, 'isolate_colour')
    a = _arr(img)
    h, s, v = _to_hsv(a)
    if space == 'oklab':
        soft = _oklab_selection(img, colour_name_or_rgb, oklab_tol, 'isolate_colour')
        return _keep_alpha(img, _grey_outside(a, soft, v, grey_strength, flatten)), soft
    if space != 'hsv':
        raise ValueError(f"isolate_colour: space must be 'hsv' or 'oklab', not {space!r}")
    target_h = colour_name_or_rgb if isinstance(colour_name_or_rgb, float) else \
        colorsys.rgb_to_hsv(*(np.array(colour_name_or_rgb) / 255.0))[0]
    dh = np.minimum(np.abs(h - target_h), 1 - np.abs(h - target_h))

    if sat_floor == 'auto':
        # A low percentile of the non-trivial saturations rejects near-grey noise while
        # tracking "faint but real colour" across both vivid and pale subjects.
        nontrivial = s[s > 0.02]
        sat_floor = float(np.percentile(nontrivial, 15)) if nontrivial.size else 0.05
        sat_floor = float(np.clip(sat_floor, 0.03, 0.18))
    keep = (dh < hue_tol) & (s > sat_floor)

    soft = _feather(keep, feather)
    return _keep_alpha(img, _grey_outside(a, soft, v, grey_strength, flatten)), soft


def blur_background(img, focus='subject', radius=14, feather=8, session=None):
    """Simulated shallow depth of field.

    `focus` selects what stays sharp: 'subject' (default, neural mask, no caller-measured
    box needed), a (x0,y0,x1,y1) rectangle, or an RGB colour (hue-proximity mask).
    """
    from scipy.ndimage import gaussian_filter
    img = img.convert('RGB')          # a palette or 1-bit input cannot be Gaussian-blurred
    a = np.array(img)

    if isinstance(focus, np.ndarray):
        # A precomputed mask; re-segmenting here would pay for the neural pass twice.
        mask = gaussian_filter(np.clip(focus.astype(float), 0, 1),
                               sigma=max(1, feather // 3))
    elif isinstance(focus, str) and focus == 'subject':
        from . import advanced
        mask = advanced.subject_mask(img, session=session)
        mask = gaussian_filter(mask, sigma=max(1, feather // 3))
    elif (isinstance(focus, tuple) and len(focus) == 4
          and all(isinstance(v, numbers.Integral) for v in focus)):
        # Integral, not int: a box from numpy is np.int64, which would otherwise fall
        # through to the colour branch. Clamped: an overshooting box would blur the whole
        # picture, subject included.
        x0, y0, x1, y1 = _clamp(focus, a.shape[1], a.shape[0])
        if (x1 - x0) <= 0 or (y1 - y0) <= 0:
            raise ValueError(
                f'focus box {focus} is empty, which would blur the whole image including the '
                f"subject. Pass a real box, or use focus='subject' for the neural mask.")
        mask = np.zeros(a.shape[:2], float)
        mask[y0:y1, x0:x1] = 1.0
        mask = gaussian_filter(mask, sigma=feather)
    else:
        _, mask = isolate_colour(img, focus, feather=feather)

    mask = np.clip(mask, 0, 1)
    blurred = np.array(img.filter(ImageFilter.GaussianBlur(radius)).convert('RGB'))
    out = _blend(blurred, a, mask)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _keep_alpha(original, result):
    """Re-attach the input's transparency to a same-size RGB result.

    Every op here converts to RGB first, which drops alpha; without this, colour hidden
    under transparent pixels would come back opaque instead of staying transparent.
    """
    if result.mode == 'RGBA' or result.size != original.size:
        return result
    if original.mode in ('RGBA', 'LA'):
        alpha = original.split()[-1]
    elif original.mode == 'P' and 'transparency' in original.info:
        alpha = original.convert('RGBA').split()[-1]
    else:
        return result
    out = result.convert('RGBA')
    out.putalpha(alpha)
    return out


def vignette(img, strength=0.55, feather=0.8):
    a = _arr(img)
    H, W = a.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H / 2, W / 2
    r = np.sqrt(((xx - cx) / (W / 2)) ** 2 + ((yy - cy) / (H / 2)) ** 2)
    fall = 1 - strength * np.clip((r - (1 - feather)) / feather, 0, 1)
    out = a * fall[..., None]
    return _keep_alpha(img, Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)))


def sepia(img):
    a = _arr(img)
    m = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
    out = a @ m.T
    return _keep_alpha(img, Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)))


def sketch(img, strength=1.0, blur_radius=6, contrast=1.25):
    """Pencil-sketch effect: dodge-blend of the grayscale image against its blurred, inverted self.

    `blur_radius` trades off stroke looseness against blowout: the common default of 15
    washes out flat, high-contrast subjects (brick, tile) to near-white. 6 keeps midtones
    there; raise it for soft, low-contrast photos (faces, skies).
    """
    g = img.convert('L')
    inv = ImageOps.invert(g)
    blur = inv.filter(ImageFilter.GaussianBlur(blur_radius))
    g_arr = np.array(g).astype(float)
    blur_arr = np.array(blur).astype(float)
    dodge = np.clip(g_arr / np.clip(255 - blur_arr, 8, 255) * 255, 0, 255)
    out = g_arr * (1 - strength) + dodge * strength
    result = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert('RGB')
    if contrast != 1.0:
        result = ImageEnhance.Contrast(result).enhance(contrast)
    return _keep_alpha(img, result)


def adjust(img, brightness=1.0, contrast=1.0, saturation=1.0, sharpness=1.0):
    out = img.convert('RGB')
    for factor, Enh in [(brightness, ImageEnhance.Brightness), (contrast, ImageEnhance.Contrast),
                        (saturation, ImageEnhance.Color), (sharpness, ImageEnhance.Sharpness)]:
        if factor != 1.0:
            out = Enh(out).enhance(factor)
    return _keep_alpha(img, out)


def grayscale(img):
    return _keep_alpha(img, ImageOps.grayscale(img).convert('RGB'))


def crop_and_rotate(img, box=None, angle=0, flip=None):
    out = img
    if box:
        out = out.crop(box)
    if angle:
        # Transparent input must get transparent corners, not opaque white ones: a 3-tuple
        # fillcolor on an RGBA image fills alpha 255, so rotating a logo boxed it in white.
        fill = (255, 255, 255, 0) if out.mode in ('RGBA', 'LA') else (255, 255, 255)
        out = out.rotate(-angle, expand=True, fillcolor=fill)
    if flip == 'horizontal':
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip == 'vertical':
        out = out.transpose(Image.FLIP_TOP_BOTTOM)
    return out
