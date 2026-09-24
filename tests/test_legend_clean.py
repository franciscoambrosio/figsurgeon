"""Cleaning a legend that bleeds, and the two failure modes that only show over dark content.

Ground truth (`evals/legend_clean.py`) is the same legend at framealpha 1.0. Both failure
modes are invisible over light content: a naive fill takes the colour of its own text (the
untinted pixels are the glyphs), and a swatch touching bleed-through can merge with it and
get erased as a ghost.

The legend fixture is built the way matplotlib composites one -- fill blended with what is
behind it, glyphs opaque on top -- so the opaque render is available offline as ground truth.
"""
import numpy as np
import pytest
from PIL import Image

from figsurgeon import advanced, verify_photo

BOX = (100, 80, 300, 220)
SWATCHES = [((115, 95, 155, 103), (31, 119, 180)),        # blue
            ((115, 140, 155, 148), (214, 39, 40)),        # red
            ((115, 185, 155, 193), (44, 160, 44))]        # green
TEXT = [(165, 92, 265, 106), (165, 137, 265, 151), (165, 182, 265, 196)]


def background(dark=True, contours=True):
    """What sits behind the legend: a saturated ramp, dark or light.

    `contours` draws white lines (as a contour plot or heatmap would): where one passes
    behind the legend the blend is white-on-white, the only untinted pixels the cleaner can
    measure the fill colour from.
    """
    x = np.linspace(0, 1, 400)[None, :, None]
    y = np.linspace(0, 1, 300)[:, None, None]
    if dark:
        base = np.concatenate([40 + 60 * x + np.zeros_like(y),
                               20 + 120 * y + np.zeros_like(x),
                               120 + 90 * x + np.zeros_like(y)], axis=2)
    else:
        base = np.concatenate([230 - 30 * x + np.zeros_like(y),
                               225 + np.zeros_like(y) + np.zeros_like(x),
                               235 - 20 * y + np.zeros_like(x)], axis=2)
    img = np.broadcast_to(base, (300, 400, 3)).astype(float).copy()
    if contours:
        for y in (60, 130, 200, 260):
            img[y:y + 2, :] = 255
        for x in (70, 190, 310):
            img[:, x:x + 2] = 255
    return img


def legend(alpha=0.75, dark=True, contours=True):
    """The figure with a legend over it at `alpha`, and the same at 1.0 as ground truth."""
    out = []
    for a in (alpha, 1.0):
        img = background(dark, contours)
        x0, y0, x1, y1 = BOX
        img[y0:y1, x0:x1] = a * 255 + (1 - a) * img[y0:y1, x0:x1]
        for (sx0, sy0, sx1, sy1), colour in SWATCHES:
            img[sy0:sy1, sx0:sx1] = colour               # drawn opaque, on top
        img = _draw_labels(img)                        # real glyphs, not solid blocks
        out.append(Image.fromarray(img.astype('uint8')))
    return out


def _draw_labels(img):
    """Real text, drawn as letters: the cleaner tells ink from bleed-through by stroke
    width, so solid rectangles would not exercise that."""
    from PIL import ImageDraw
    im = Image.fromarray(img.astype('uint8'))
    draw = ImageDraw.Draw(im)
    for (tx0, ty0, _, _), label in zip(TEXT, ('Observed', 'Model', 'Residual')):
        draw.text((tx0, ty0), label, fill=(60, 60, 60))
    return np.asarray(im).astype(float)


def inside(img, box=BOX):
    x0, y0, x1, y1 = box
    return np.asarray(img).astype(int)[y0:y1, x0:x1]


def error(got, truth):
    return np.abs(inside(got) - inside(truth)).max(axis=2).mean()


def test_cleaning_a_legend_over_dark_content_gets_closer_to_the_opaque_render():
    """Cleaning must move the box towards the same legend drawn opaque, not away from it (a
    flat fill takes the colour of its own text)."""
    dirty, truth = legend(dark=True)
    cleaned, _ = advanced.clean_transparent_box(dirty, BOX)
    before, after = error(dirty, truth), error(cleaned, truth)
    assert after < before / 2, f'error {before:.1f} -> {after:.1f}'
    assert inside(cleaned).max(axis=2).mean() > 200, 'the box came out dark'


def test_every_swatch_survives_the_clean():
    """Every swatch pixel must come through unchanged; bleed-through touching a swatch must
    not merge with it and take it down as a ghost."""
    dirty, truth = legend(dark=True)
    cleaned, _ = advanced.clean_transparent_box(dirty, BOX)
    got, want = np.asarray(cleaned).astype(int), np.asarray(truth).astype(int)
    for (sx0, sy0, sx1, sy1), colour in SWATCHES:
        d = np.abs(got[sy0:sy1, sx0:sx1] - want[sy0:sy1, sx0:sx1]).max()
        assert d < 40, f'swatch {colour} was damaged (max deviation {d})'


def test_the_bleed_through_actually_goes():
    """The faint tint that is bleed-through must actually drop, not just avoid getting worse."""
    dirty, truth = legend(dark=True)
    cleaned, _ = advanced.clean_transparent_box(dirty, BOX)

    def ghosts(img):
        r = inside(img)
        sat = r.max(axis=2) - r.min(axis=2)
        return float(((sat > 8) & (sat <= 70)).mean())
    assert ghosts(cleaned) < ghosts(dirty) / 4, f'{ghosts(dirty):.1%} -> {ghosts(cleaned):.1%}'


def test_a_light_legend_is_not_made_worse():
    """A fix for dark backgrounds must not regress the light case the code always handled.
    The bound is "not materially worse", not "never worse by any amount": over a light
    background the fill can only be measured from what is visible, so flattening to it costs
    a small, expected amount on an already-near-correct box."""
    dirty, truth = legend(dark=False)
    cleaned, _ = advanced.clean_transparent_box(dirty, BOX)
    before, after = error(dirty, truth), error(cleaned, truth)
    assert after <= before + 1.0, f'{before:.1f} -> {after:.1f}'
    assert after < 12, f'a nearly-clean light legend came out at {after:.1f}'


def test_a_box_with_no_untinted_pixel_is_refused_rather_than_guessed():
    """With no untinted pixel behind the legend, the fill colour is not recoverable and
    must be refused, not guessed; the same legend with contours behind it must still clean
    normally."""
    dirty, _ = legend(dark=True, contours=False)
    with pytest.raises(ValueError) as e:
        advanced.clean_transparent_box(dirty, BOX)
    assert 'cannot be recovered' in str(e.value)
    with_lines, _ = legend(dark=True, contours=True)
    advanced.clean_transparent_box(with_lines, BOX)          # must not raise


def test_a_dark_legend_is_refused_rather_than_inverted():
    """The cleaner assumes a light box with dark text; a dark box must raise with an
    explanation, not silently sample a fill from the white text."""
    img = background(dark=True)
    x0, y0, x1, y1 = BOX
    img[y0:y1, x0:x1] = (25, 25, 28)                     # a dark-themed legend
    for tx0, ty0, tx1, ty1 in TEXT:
        img[ty0:ty1, tx0:tx1] = (245, 245, 245)
    with pytest.raises(ValueError) as e:
        advanced.clean_transparent_box(Image.fromarray(img.astype('uint8')), BOX)
    assert 'light box with dark text' in str(e.value)


# ----------------------------------------------------------------- the verdict

def test_the_check_passes_a_good_clean_and_fails_a_lost_swatch():
    """Both directions of `check_transparent_box_cleaned`, from the same pair of images."""
    dirty, _ = legend(dark=True)
    cleaned, _ = advanced.clean_transparent_box(dirty, BOX)
    ok, detail = verify_photo.check_transparent_box_cleaned(dirty, cleaned, BOX)
    assert ok is True, detail

    damaged = np.asarray(cleaned).astype(int).copy()
    (sx0, sy0, sx1, sy1), _ = SWATCHES[1]
    damaged[sy0:sy1, sx0:sx1] = 255                      # one swatch erased
    ok, detail = verify_photo.check_transparent_box_cleaned(
        dirty, Image.fromarray(damaged.astype('uint8')), BOX)
    assert ok is False and 'LOST a swatch' in detail, detail


def test_the_check_fails_a_box_that_went_black():
    """The box flat-filled with the colour of its own text: the failure mode described in
    the module docstring."""
    dirty, _ = legend(dark=True)
    blacked = np.asarray(dirty).astype(int).copy()
    x0, y0, x1, y1 = BOX
    blacked[y0:y1, x0:x1] = 0
    ok, detail = verify_photo.check_transparent_box_cleaned(
        dirty, Image.fromarray(blacked.astype('uint8')), BOX)
    assert ok is False and 'DARKER' in detail, detail


def test_the_check_says_nothing_happened_rather_than_passing_it():
    """A no-op is not a clean. It gets no verdict, not a pass."""
    dirty, _ = legend(dark=True)
    ok, detail = verify_photo.check_transparent_box_cleaned(dirty, dirty, BOX)
    assert ok is None and 'nothing inside the box changed' in detail
