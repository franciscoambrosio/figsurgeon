"""Operations that need more than HSV masking to do well.

  Real neural network:
    remove_background / replace_background -- U2Net (via `rembg`), ~176 MB, downloaded on
    first use. Needed because subject/background separation isn't a colour-space question
    (a brown cat against a brown table defeats `isolate_colour` outright).

  Classical CV, no model download:
    remove_object -- OpenCV inpainting; good on smooth/repetitive backgrounds, cannot
    reconstruct unique structure (a face) it has no evidence for.
    denoise -- OpenCV non-local means; softens real detail at high strength.
    auto_white_balance -- grey-world assumption; fails when a scene is genuinely
    dominated by one colour.

  Out of reach here: object-aware edits ("make the car red") need instance segmentation,
  which this module does not add.
"""
import numpy as np
from PIL import Image

from .composite import _arr
# cv2 and rembg are deliberately not imported at module level: they're large optional
# dependencies, and importing figsurgeon for chart work alone shouldn't require them. Each
# function imports what it needs and fails with a clear pip-install message instead.


PYPI_NAME = {'cv2': 'opencv-python-headless'}


def _require(module, extra):
    try:
        return __import__(module)
    except ImportError:
        pip_name = PYPI_NAME.get(module, module)
        raise ImportError(
            f'{module!r} is required for this function. Install it with: '
            f'pip install "figsurgeon[{extra}]"  (or: pip install {pip_name})')


_DEFAULT_SESSION = None


def default_session():
    """The process-wide U2Net session, created on first use.

    `rembg.remove(img)` with `session=None` builds a fresh session per call, which re-reads
    and re-initialises the 176 MB model every time. In an interactive editing loop that is
    the dominant cost -- adjusting a background blur pays it once per look, and
    `blur_background` + `check_background_blur` alone segment the same image twice. Callers
    who want an isolated session still pass one explicitly.
    """
    global _DEFAULT_SESSION
    if _DEFAULT_SESSION is None:
        _DEFAULT_SESSION = new_session()
    return _DEFAULT_SESSION


def remove_background(img, session=None):
    """Return an RGBA cutout with the background made transparent.

    `session` lets you reuse a rembg session across calls (loading the ~176 MB model once)
    instead of reloading it per image -- pass `figsurgeon.advanced.new_session()`. When
    omitted, a process-wide cached session is used rather than building a new one per call.
    """
    from rembg import remove
    return remove(img.convert('RGB'), session=session or default_session())


def subject_mask(img, session=None):
    """The U2Net subject mask as a float 0..1 array, without the RGBA round-trip.

    Exists so a caller can check how much subject was found before acting on the answer:
    real subjects typically cover about half the frame, while images with no subject at
    all cover under 1%, which lets a caller catch a false "background" before acting on it.
    """
    cutout = remove_background(img, session=session)
    return np.array(cutout.split()[-1]).astype(float) / 255.0


def _glyph_shaped(mask, box_area):
    """`mask` restricted to components shaped like ink.

    Neutral-and-dark also describes grey/black content bleeding through a legend. What
    separates a letter from a bleeding shape is stroke width, not size: a letter is thin,
    so eroding it once removes most of it, while a solid shape barely changes.
    """
    from scipy import ndimage
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if not n:
        return mask
    thin = np.zeros(n + 1, bool)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
    whole = np.bincount(labels.ravel(), minlength=n + 1)
    left = np.bincount(labels[eroded].ravel(), minlength=n + 1) if eroded.any() else np.zeros(n + 1, int)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(whole > 0, left / np.maximum(whole, 1), 0.0)
    thin[1:] = (ratio[1:] < 0.45) | (whole[1:] < 0.0005 * box_area)
    return thin[labels]


def saturation_mask(mx, mn, floor):
    """Pixels saturated enough to be a drawn glyph rather than bleed-through."""
    return (mx - mn) > floor


def clean_transparent_box(img, box, glyph_saturation_floor=70, tint_threshold=20,
                          mirror_axis='auto', text_search_width=50):
    """Clean bleed-through inside a semi-transparent legend/annotation box.

    Handles a matplotlib legend with alpha < 1 over a dense plot, where data bleeds through
    the box in the same colours as the box's own glyph swatches, so no colour mask alone
    can tell "real swatch" from "ghost". Separates them by deviation from the box's fill
    colour: real swatches are opaque and deviate strongly, bleed-through lands close to the
    fill. `box` is an approximate (x0,y0,x1,y1); like `remove_object`, this does not locate
    the legend for you. Returns (cleaned_full_image, tight_crop_of_the_box).

    Known limits: `tint_threshold` and `text_search_width` are tuned defaults, not
    calibrated values -- widen `text_search_width` if real swatches vanish. On a densely
    overlapping legend (a semi-transparent box over opaque-ish markers), individual
    bleed-through pixels can be as saturated as a real swatch; inspect results there.
    """
    from scipy import ndimage
    a = np.array(img.convert('RGB')).astype(int)
    x0, y0, x1, y1 = box
    pad = 6
    wx0, wx1 = max(0, x0 - pad), min(a.shape[1], x1 + pad)
    wy0, wy1 = max(0, y0 - pad), min(a.shape[0], y1 + pad)
    region = a[wy0:wy1, wx0:wx1].copy()

    # Fill colour: the modal colour among the least-tinted light pixels -- not simply the
    # least-tinted ones, which matters a great deal on a legend that sits over saturated
    # content.
    #
    # The calmest pixels in a bleeding legend are its own black text: the fill is white
    # blended with whatever shows through, so it is tinted by definition (that is the thing
    # being cleaned), while the glyphs are drawn opaque and are perfectly neutral. The
    # heavier the bleed, the more certainly the mode lands on the text, which is exactly
    # backwards for a function meant to clean the bleed. On ground truth
    # (`evals/legend_clean.py`: the same legend rendered at framealpha 0.75 and at 1.0 over
    # three real figures), over a viridis contour plot the naive fill comes back (0, 0, 0)
    # and flat-fills the legend black, mean error against the opaque render 43 before
    # cleaning and 158 after.
    #
    # `mx >= 210` is the same light-vs-glyph line the text mask below already draws, made
    # explicit here: this function assumes a light box with dark text, which every legend it
    # was built for is, and a dark-themed legend is refused rather than silently inverted.
    mx, mn = region.max(axis=2), region.min(axis=2)
    # The pool the fill is sampled from: light, and not a drawn glyph. Dropping "light"
    # would fill a legend over a viridis contour plot with the colour of its own black
    # text -- in a bleeding legend the untinted pixels are the glyphs, because the fill is
    # tinted by definition. Dropping "not a glyph" would pick the red swatch instead,
    # because over a gradient every fill pixel is a different colour while a swatch is
    # hundreds of identical ones, so the swatch wins a vote on modal colour.
    light = (mx >= 210) & ((mx - mn) <= glyph_saturation_floor)
    # And the fill must be COMMON: white text on a dark legend is light and un-glyphlike
    # too, and sampling it would flat-fill the box white -- the same error as filling it
    # black, in the other direction. A legend's fill covers most of its box; its glyphs do
    # not.
    if light.mean() < 0.15:
        raise ValueError('box interior has no light region big enough to be a fill colour. '
                         'This cleaner assumes a light box with dark text; pass a tighter '
                         'or more accurate box, and check the legend is not a dark one '
                         '(which it cannot clean)')
    from collections import Counter
    # "Calm" assumes a near-neutral fill; a legend with chroma of its own (cream, pale
    # yellow) has no pixel that satisfies a fixed absolute threshold even unbled. Relaxing
    # this to a relative threshold was tried and fixed those legends by regressing the
    # common case, so it's left as a known limit.
    calm = light & ((mx - mn) < 6)
    if calm.sum() < 20:
        # Every pixel in the box is tinted, so the fill colour can't be recovered from an
        # unknown blend; guessing would produce a confidently wrong flat colour.
        raise ValueError(
            'every pixel of this box is tinted by what is behind it, so the legend\'s own '
            'fill colour cannot be recovered -- there is nothing untinted to measure it '
            'from. This happens when uniformly saturated content sits behind the whole box; '
            'try a box that includes part of the legend over lighter content.')
    # The lightest common calm colour, not simply the commonest: over a light background
    # the plain mode follows the background's blend rather than the true, brighter fill.
    calm_px = region[calm]
    top = np.percentile(calm_px.max(axis=1), 95)
    brightest = calm_px[calm_px.max(axis=1) >= top - 3]
    pool = brightest if len(brightest) >= 20 else calm_px
    fill = np.array(Counter(map(tuple, pool)).most_common(1)[0][0])

    # The threshold is lower away from the border: one value doing both jobs either lets a
    # faint ghost survive in the interior or starts flagging the border stroke's own
    # anti-aliasing. The border is a ring of known width, so the two can use different
    # floors: the caller's threshold there, half of it in the interior.
    ring = np.zeros(region.shape[:2], bool)
    ring[:pad + 4, :] = ring[-(pad + 4):, :] = True
    ring[:, :pad + 4] = ring[:, -(pad + 4):] = True
    floor = np.where(ring, tint_threshold, max(8, tint_threshold // 2))
    # Contamination is a deviation from the fill in either direction: chroma (tinted bleed)
    # or lightness (grey/black bleed, which no chroma measure alone can see).
    tinted = ((mx - mn) > floor) | (np.abs(region - fill).max(axis=2) > floor)
    # Components are labelled at the glyph-saturation level, not the tint level: labelling
    # everything tinted would merge a swatch with adjoining bleed-through into one blob
    # whose mean saturation falls below the glyph floor, erasing the swatch too.
    labels, n = ndimage.label(saturation_mask(mx, mn, glyph_saturation_floor),
                              structure=np.ones((3, 3)))

    # Text is glyph-sized: neutral-and-dark also describes bleed-through of dark content
    # (pavement, clothing), which colour can't tell apart from ink, so the cut here is on
    # component area relative to the box instead.
    neutral_dark = (mx - mn < 25) & (mx < 210)
    is_text = _glyph_shaped(neutral_dark, region.shape[0] * region.shape[1])

    # Uses saturation (max-min channel spread) rather than distance from the fill's minimum
    # channel: that alternative is hue-biased, since a pale colour (pink) has a naturally
    # high minimum channel even at full opacity and can score below a saturated colour
    # blended halfway to white, dropping a real swatch below threshold.
    saturation = mx - mn
    candidates = []
    for i in range(1, n + 1):
        blob = labels == i
        if blob.sum() < 4:
            continue
        if saturation[blob].mean() > glyph_saturation_floor:
            candidates.append((i, blob))

    # A real swatch sits immediately left of its own text label. Two signals are combined
    # with OR since each catches cases the other misses: (a) text-ink in a thin strip right
    # of the candidate, (b) x-column alignment, for swatches in a single-column legend that
    # share an x-coordinate. Known limit: on a densely overlapping legend, individual
    # bleed-through pixels can be as saturated as a real swatch, and no local pixel rule
    # separates them.
    protect = np.zeros(region.shape[:2], bool)
    Hr, Wr = region.shape[:2]
    text_adjacent = set()
    for i, blob in candidates:
        yy, xx = np.where(blob)
        y0b, y1b, x_right = yy.min(), yy.max(), xx.max()
        sy0, sy1 = max(0, y0b - 4), min(Hr, y1b + 5)
        sx0, sx1 = x_right + 1, min(Wr, x_right + text_search_width)
        if sx0 < sx1 and is_text[sy0:sy1, sx0:sx1].any():
            protect |= blob
            text_adjacent.add(i)

    remaining = [(i, blob) for i, blob in candidates if i not in text_adjacent]
    if remaining:
        xs = np.array([float(np.where(b)[1].mean()) for _, b in remaining])
        order = np.argsort(xs)
        xs_sorted = xs[order]
        best_run, cur_run = [0], [0]
        for k in range(1, len(xs_sorted)):
            if xs_sorted[k] - xs_sorted[cur_run[0]] < 20:
                cur_run.append(k)
            else:
                cur_run = [k]
            if len(cur_run) > len(best_run):
                best_run = cur_run
        if len(best_run) > 1:                    # a real cluster, not a lone coincidence
            for k in best_run:
                protect |= remaining[order[k]][1]

    # text: near-neutral AND dark -- real ink, not tinted bleed (already computed above)
    contamination = tinted & ~protect & ~is_text
    region[contamination] = fill
    grey_val = region[is_text].mean(axis=1, keepdims=True)
    region[is_text] = grey_val

    # border ring reconstruction via mirroring, restricted to a thin band just inside `pad`
    # so it can't touch interior content
    if mirror_axis in ('auto', 'both'):
        vflip, hflip = region[::-1, :, :], region[:, ::-1, :]

        def tint(px):
            return px.max(axis=2) - px.min(axis=2)

        ring = ring & ~protect
        orig_t, v_t, h_t = tint(region), tint(vflip), tint(hflip)
        ring_contam = ring & (orig_t > tint_threshold)
        use_v = ring_contam & (v_t <= h_t)
        use_h = ring_contam & (v_t > h_t)
        region[use_v] = vflip[use_v]
        region[use_h] = hflip[use_h]

    out = a.copy()
    out[wy0:wy1, wx0:wx1] = region
    full = Image.fromarray(out.astype('uint8'))
    crop = full.crop((wx0, wy0, wx1, wy1))
    return full, crop


def new_session(model='u2net'):
    """Load the background-removal model once, to reuse across multiple images.

    'u2net' (176 MB) is the default, general-purpose model. 'u2netp' (4.5 MB) is a much
    smaller, faster, lower-quality alternative -- worth it for quick previews, not final output.
    """
    from rembg import new_session as _new_session
    return _new_session(model)


def replace_background(img, new_background, session=None, cutout=None):
    """Cut the subject out and composite it onto a new background.

    `new_background` is a solid RGB colour or another PIL Image (resized to fit) -- this is
    what makes `remove_background` useful for "put me on a white background" rather than
    just a transparent-PNG intermediate.

    `cutout`, if given, is used instead of running `remove_background` again, for a caller
    (`tools.dispatch`) that already ran it once to check how much subject was found.
    """
    cutout = cutout if cutout is not None else remove_background(img, session=session)
    if isinstance(new_background, tuple):
        bg = Image.new('RGB', cutout.size, new_background)
    else:
        bg = new_background.convert('RGB').resize(cutout.size)
    bg.paste(cutout, (0, 0), cutout)
    return bg


def remove_object(img, region, radius=6, method='telea'):
    """Paint out `region` using its surroundings.

    `region` is one of:
      - (x0, y0, x1, y1): a box
      - [(x0,y0), (x1,y1), ...]: a polyline, `radius`-px wide -- for masts, wires, cables
      - a single-channel PIL Image / array the same size as `img`: a hand-drawn mask

    This is texture interpolation, not scene understanding: it works well removing a thin
    object from a smooth or repetitive background, and works badly trying to reconstruct
    something structurally unique (a face, a landmark) that was fully covered.
    """
    a = np.array(img.convert('RGB'))
    if isinstance(region, Image.Image):
        mask = np.array(region.convert('L'))
    elif isinstance(region, np.ndarray):
        mask = region
    elif len(region) == 4 and all(isinstance(v, (int, float)) for v in region):
        x0, y0, x1, y1 = map(int, region)
        mask = np.zeros(a.shape[:2], np.uint8)
        mask[y0:y1, x0:x1] = 255
    else:
        cv2 = _require('cv2', 'advanced')
        mask = np.zeros(a.shape[:2], np.uint8)
        pts = np.array(region, dtype=np.int32)
        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=int(radius * 2))
    cv2 = _require('cv2', 'advanced')
    flag = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
    out = cv2.inpaint(a, mask.astype(np.uint8), inpaintRadius=radius, flags=flag)
    return Image.fromarray(out)


def denoise(img, strength=8):
    """Non-local means denoising. `strength` ~3 (light) to ~15 (heavy, softens real detail)."""
    cv2 = _require('cv2', 'advanced')
    a = np.array(img.convert('RGB'))
    out = cv2.fastNlMeansDenoisingColored(a, None, h=strength, hColor=strength,
                                          templateWindowSize=7, searchWindowSize=21)
    return Image.fromarray(out)


def auto_white_balance(img, warn=True):
    """Grey-world colour correction: scale each channel so the image's average is neutral grey.

    Returns (image, warning_or_None). Grey-world assumes the scene's average reflectance IS
    neutral; when a photo is genuinely dominated by one colour (an orange spacesuit, a
    sunset), correcting toward grey makes it worse. The warning reports that so a caller
    can check before accepting the result.
    """
    a = _arr(img)
    means = a.reshape(-1, 3).mean(axis=0)
    grey = means.mean()
    scale = grey / np.maximum(means, 1e-6)
    out = np.clip(a * scale, 0, 255)

    warning = None
    if warn:
        # Reports the size of the correction without judging whether it's appropriate:
        # distinguishing "orange cast" from "orange spacesuit" is a question about what the
        # subject IS, which no pixel statistic answers, so this is a prompt to look rather
        # than a verdict.
        deviation = float(np.abs(scale - 1).max())
        if deviation > 0.15:
            direction = ['red', 'green', 'blue'][int(np.argmax(means))]
            warning = (
                f'this shifts colours substantially (up to {deviation:.0%} on one channel; '
                f'the image is {direction}-dominant). Grey-world assumes the scene average '
                f'should be neutral -- true for a colour cast, false when a strong colour '
                f'genuinely belongs to the subject. Nothing measurable in the pixels tells '
                f'those apart, so inspect the result rather than trusting it.')
    return Image.fromarray(out.astype(np.uint8)), warning
