"""Undo a generative fill's colour shift, using the context ring the model was shown.

A hosted editor given a padded crop often returns a fill whose background sits slightly off
the surroundings it continues -- a touch brighter, a touch cooler -- so the box reads as a
patch even when its content is right. Matching the boundary pixels is the obvious fix; this
is a stronger version of it.

`generative.fill` pads the box by `pad` times its own size, sends the crop, and keeps only
the box -- the model's rendering of the surrounding ring is normally discarded to protect the
outside-is-untouched promise. But that ring is the same content rendered twice, once by the
camera and once by the model, already registered because the crop's bounds are ours. It is a
paired sample of exactly the transform to undo, with thousands of pixels of it where the box
boundary offers only a rim a few pixels wide.

The fit is per-channel gain and offset, in linear light: the shift being corrected is an
exposure/white-balance difference, and a gain in gamma-encoded sRGB is not one. Gain plus
offset rather than a full 3x3 -- the extra six parameters would buy a colour rotation there
is no evidence for here.

The ring is not a clean pair: models redraw a little outside the box they were asked about,
and such a pixel is an outlier with a large residual that plain least squares would chase.
Two trimming passes on the residual (drop anything past ~2.5 median-absolute-deviations)
leave the genuinely corresponding pixels to set the fit.

The correction is not applied on faith. `generative.fill` composes the fill both ways and
keeps the correction only when the measured seam (`verify_photo.fill_seam`) improves,
recording both numbers.

The seam gate alone is not sufficient, which is why `agreement` exists. The method assumes
the model's ring and the original ring are the same content; when that assumption fails the
fit is unidentifiable -- a model that ignored its input and painted the crop flat gives a
ring with no variation, and any (gain, offset) through that one point fits it equally well.
Worse, the seam gate prefers that degenerate solution, since mapping the fill onto the
background colour drives the seam to zero: composed both ways and scored on the seam,
"delete the fill" wins. So identifiability is checked first: per-channel correlation between
the model's ring and the real one, and a ring with too little variation to fit through. Below
that, no correction is attempted and the reason is recorded.

That check is also useful on its own: how faithfully a model reproduced content it was shown
is evidence about how much to trust the region it invented. `info['colour_fit']['r']` is that
evidence.

This makes a fill agree with its surroundings, which is not the same as making it right. A
wrong fill, colour-matched, is a wrong fill that looks more convincing, which is why `info`
records the fit rather than applying it invisibly.
"""
import numpy as np

# Below this many usable ring pixels the fit is noise -- a tight box against a frame edge
# can leave almost no ring at all. Identity is returned instead.
MIN_RING_PX = 500

# Guards on identifiability, not calibrated cuts -- there is no corpus of good and bad
# generative fills to calibrate against. Each is set where the quantity stops carrying
# information at all; refusing just means no correction is made, the same as skipping this
# check entirely.
MIN_RING_R = 0.5          # below this the model's ring is more unlike the real one than like it
MIN_RING_SPREAD = 0.01    # linear-light std; under it a channel is flat and its gain is a free variable
MIN_GAIN, MAX_GAIN = 0.5, 2.0


def _to_linear(a):
    """sRGB 0-255 -> linear 0-1. The standard piecewise transfer function, not gamma 2.2."""
    c = np.asarray(a, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c):
    """linear 0-1 -> sRGB 0-255, the inverse of `_to_linear`."""
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    s = np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)
    return s * 255.0


def agreement(model_ring, original_ring):
    """Per-channel Pearson r between the model's ring and the real one, in linear light.

    High means the model reproduced the content it was shown, so a colour transform between
    them is a meaningful thing to fit. Low means it painted something else, and any fit is
    a fit between unrelated pixels. A channel with no variation on either side scores 0.0
    rather than nan -- there is nothing to correlate, which is exactly the unidentifiable
    case, and it should read as no evidence rather than as an error.
    """
    m = _to_linear(np.asarray(model_ring).reshape(-1, 3))
    o = _to_linear(np.asarray(original_ring).reshape(-1, 3))
    out = []
    for ch in range(3):
        x, y = m[:, ch], o[:, ch]
        if x.size < 2 or x.std() < 1e-6 or y.std() < 1e-6:
            out.append(0.0)
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        out.append(0.0 if not np.isfinite(r) else r)
    return out


def fit(model_ring, original_ring, trims=2):
    """Per-channel (gain, offset) in LINEAR light mapping `model_ring` onto `original_ring`.

    Both arguments are (N, 3) uint8-ish arrays of corresponding pixels. Returns
    (gain, offset, why): each of the first two shape (3,), and `why` a short string that is
    empty when the fit was made and says what stopped it otherwise. Identity is returned
    whenever it refuses, so a caller that ignores `why` still gets a harmless answer.
    """
    m = _to_linear(np.asarray(model_ring).reshape(-1, 3))
    o = _to_linear(np.asarray(original_ring).reshape(-1, 3))
    gain, offset = np.ones(3), np.zeros(3)
    if m.shape[0] < MIN_RING_PX:
        return gain, offset, f'ring too small ({m.shape[0]} px < {MIN_RING_PX})'
    r = agreement(model_ring, original_ring)
    if min(r) < MIN_RING_R:
        return gain, offset, (f'the model did not reproduce the ring it was shown '
                              f'(r={[round(v, 3) for v in r]}), so there is no correspondence '
                              f'to fit a colour transform through')
    for ch in range(3):
        x, y = m[:, ch], o[:, ch]
        if x.std() < MIN_RING_SPREAD:
            continue                      # flat channel: gain is unidentifiable, leave it
        keep = np.ones(x.shape, bool)
        g = b = None
        for _ in range(trims + 1):
            if keep.sum() < MIN_RING_PX // 4:
                break
            g, b = np.polyfit(x[keep], y[keep], 1)
            resid = np.abs(y - (g * x + b))
            mad = np.median(np.abs(resid - np.median(resid)))
            if mad <= 0:
                break
            keep = resid <= np.median(resid) + 2.5 * mad
        # An exposure/white-balance difference, not an arbitrary remap. A fit outside these
        # bounds is evidence the pairing is wrong, not evidence of a large shift.
        if g is not None and np.isfinite(g) and np.isfinite(b) and MIN_GAIN <= g <= MAX_GAIN:
            gain[ch], offset[ch] = g, b
    return gain, offset, ''


def apply(img_rgb, gain, offset):
    """Apply a `fit` result to a uint8 RGB array, in linear light. Returns uint8."""
    lin = _to_linear(img_rgb) * np.asarray(gain) + np.asarray(offset)
    return np.clip(_to_srgb(lin), 0, 255).astype(np.uint8)


def ring_mask(shape_hw, box_in_crop):
    """True everywhere in the crop EXCEPT the box -- the context the model got for free."""
    h, w = shape_hw
    m = np.ones((h, w), bool)
    x0, y0, x1, y1 = (int(v) for v in box_in_crop)
    m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = False
    return m



def holdout_gain(model_ring, original_ring, gain, offset, seed=0):
    """How much closer to the real ring the correction gets, on ring pixels it was NOT fit on.

    The seam cannot referee this correction. It is an edge measure -- a thin band just
    inside the box against a thin band just outside -- and what this fixes is an offset
    across the WHOLE patch. Measured on a real return: a fill whose sky was visibly off
    scored seam 0.004 before the correction and 0.003 after, both far below the 0.02 that
    counts as invisible, so a change a person could see moved the refereeing number by
    0.001. Gated on that alone the correction survives by luck.

    So the referee is the quantity actually being corrected, scored honestly: fit on half
    the ring, evaluate on the other half. Held out, a fit that captured a real colour shift
    still helps, and one that chased noise does not. Returns (before, after) as mean OKLab
    distance from the true ring -- lower is closer.
    """
    from . import perceptual
    m = np.asarray(model_ring).reshape(-1, 3)
    o = np.asarray(original_ring).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    held = rng.random(m.shape[0]) < 0.5
    if held.sum() < MIN_RING_PX // 2 or (~held).sum() < MIN_RING_PX // 2:
        return None, None
    g2, o2, why = fit(m[~held], o[~held])
    if why:
        return None, None
    before = float(perceptual.delta_e(m[held], o[held]).mean())
    after = float(perceptual.delta_e(apply(m[held][None], g2, o2)[0], o[held]).mean())
    return round(before, 5), round(after, 5)
