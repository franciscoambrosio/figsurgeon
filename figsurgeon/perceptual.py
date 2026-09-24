"""Colour selection in a perceptual space (OKLab), instead of an HSV hue window.

A hue window (`dh < hue_tol` in HSV) has two defects on real photographs that a synthetic
test image does not show:

  * It cuts continuous objects at a hard edge the object does not have. A real flower is one
    object with a gradient running red -> orange -> yellow across the same petals; selecting
    "the red" at the default tolerance shredded the bloom into 262 disconnected mask
    components (measured on a quarter-size copy of an 8 MP macro), leaving it blotched with
    grey holes.
  * Hue is not perceptual distance: it ignores lightness and saturation, so a near-black
    pixel and a vivid one are "the same colour" if their hues agree, while two visibly
    identical colours land on opposite sides of the window if they straddle it. Measured on
    the astronaut photo: selecting the orange suit at hue_tol=0.07 matched 59 % of the
    frame, including her face, her hair and the flag -- everything warm.

OKLab (Ottosson, 2020) is a perceptually uniform space: Euclidean distance in it corresponds
much more closely to "how different do these look". Selecting by distance to a target colour
therefore follows the object rather than a slice of the colour wheel.

At each space's best swept tolerance, OKLab is better even on a chart with a few flat,
saturated series colours, the case a hue window looks suited to. Measured against
constructed ground truth in `evals/colour_spaces.py` it scores IoU 0.77 against 0.68, because the HSV path's saturation floor drops
the antialiased edge of a line where it blends into white: those pixels have median
saturation 0.056 against an auto floor of 0.099. On neutrals it is not close -- selecting a
white patch scores 1.00 against 0.00, since hue carries no information without chroma.

HSV is the default nonetheless. Comparing the two spaces at their actual default tolerances
(`evals/colour_space_defaults.py`: 0.07 of a hue wheel against an OKLab distance of 0.06,
rather than each space's best swept setting) the answer splits by domain. The blocking case
is a chart: a series drawn as a line plus a translucent confidence band is one colour to a
reader and two to a distance metric, so OKLab at 0.06 takes the line and leaves the band
(IoU 0.07 against HSV's 0.84); the same mechanism costs it the plain chart case too (0.56
against 0.68), dropping the part of the line running under a semi-transparent legend frame.
On photographs it is the other way round and not close: the astronaut's suit is 25.5 % of
the frame in 154 pieces against HSV's 62.7 % in 2403. No single OKLab tolerance serves both:
the band needs 0.08-0.10, the neutral patch needs 0.02-0.04, and at 0.10 the shaded object
falls back to 0.68.

The plain-language front-end settles it: `photo_describe` can only resolve a colour word to
one canonical RGB, and a hue window is coarse in the same way a word is coarse. With OKLab
as the default, "colour pop the red" would select under 0.1 % of the frame on 7 of the 12
corpus photographs, and `evals/text_path.py` would lose 21 of its 141 correct edits.

`space='oklab'` is opt-in per call; the tool descriptions say to pass it on a photograph.
"""
import numpy as np

# sRGB -> LMS, then LMS' -> OKLab. Constants from Bjorn Ottosson's derivation.
_SRGB_TO_LMS = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
_LMS_TO_OKLAB = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])


def _srgb_to_linear(a):
    """Undo the sRGB transfer function. Skipping this is the usual way OKLab code goes
    subtly wrong: the matrices below are defined on LINEAR light, and feeding them gamma-
    encoded values produces a space that is neither sRGB nor OKLab."""
    a = np.clip(a.astype(np.float64) / 255.0, 0.0, 1.0)
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def to_oklab(rgb):
    """RGB (0-255, any shape ending in 3) -> OKLab (L, a, b), same leading shape."""
    linear = _srgb_to_linear(np.asarray(rgb))
    lms = linear @ _SRGB_TO_LMS.T
    # Cube root, sign-preserving: tiny negative values appear from out-of-gamut input and
    # numerical error, and np.cbrt handles them where a fractional power would give nan.
    return np.cbrt(lms) @ _LMS_TO_OKLAB.T


def delta_e(rgb_a, rgb_b):
    """Perceptual distance between two colours (or an array against one colour).

    Euclidean distance in OKLab. Typical magnitudes: ~0.02 is a just-noticeable difference,
    0.1 is clearly different, 0.3+ is a different colour entirely.
    """
    lab_a, lab_b = to_oklab(rgb_a), to_oklab(rgb_b)
    return np.sqrt(((lab_a - lab_b) ** 2).sum(axis=-1))


def paintable(rgb, chroma_floor=0.020, chroma_full=0.055,
              dark_floor=0.12, dark_full=0.26):
    """Per-pixel weight in [0, 1] for "this is painted surface, not material".

    Painting every pixel of a mask with the target hue scaled by luminance is right for the
    paint and wrong for everything else the mask contains. On a car, that means the windows,
    the tyres and the chrome all come back the target colour too, and the result reads as a
    cut-out laid over the photo, even though the shading itself is preserved faithfully
    (luminance correlation inside the mask, original vs recoloured: +0.973). The defect is
    not the shading -- it is that one hue gets applied to several materials.

    Glass, rubber, chrome and shadow are either near-neutral (little chroma of their own) or
    very dark, and paint is neither. Both tests are ramps rather than cutoffs, because a
    hard threshold leaves a speckled edge where a panel falls into shadow.

    The weight only ever shrinks what gets painted, so anything relying on "nothing outside
    the mask changes" still holds.
    """
    lab = to_oklab(np.asarray(rgb, dtype=np.uint8))
    L = lab[..., 0]
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    w_chroma = np.clip((chroma - chroma_floor) / max(chroma_full - chroma_floor, 1e-6), 0, 1)
    w_light = np.clip((L - dark_floor) / max(dark_full - dark_floor, 1e-6), 0, 1)
    return w_chroma * w_light


def colour_mask(img, target_rgb, tolerance=0.06, chroma_weight=1.0, lightness_weight=0.25,
                smooth=0):
    """A soft 0..1 mask of pixels perceptually close to `target_rgb`.

    `tolerance` is an OKLab distance, not a hue fraction: 0.04 is tight, 0.06 the default,
    0.12 loose. Unlike a hue window it degrades gracefully -- the mask fades out with
    perceptual distance instead of ending at a hard boundary partway through an object.

    The mask ramp reaches zero at twice the tolerance, so a loose-sounding 0.12 already
    selects out to a perceptual distance of 0.24, a different colour entirely: measured in
    evals/colour_spaces.py, at 0.12 the astronaut's orange suit takes 83.9 % of the frame,
    worse than the HSV hue window this exists to improve on; at 0.06 it takes 25.5 %. Across
    the four photographs there, 0.04-0.06 lands near the size of the object meant; the
    flat-colour cases prefer 0.10, which is what `space='hsv'` is for anyway.

    `lightness_weight` below 1.0 makes the selection tolerant of shading while staying
    strict about hue and chroma, which is what "select this colour" almost always means on
    a photograph: the shadowed side of a red car is still the red car. At 1.0 the selection
    is strictly perceptual and a shadow is a different colour, which is correct but rarely
    what was wanted. Swept in evals/colour_spaces.py, 0.25 has the best mean IoU across the
    constructed cases (0.92 against 0.87 at 0.5), mostly by being far better on flat chart
    colours (0.77 against 0.61). Lower is not better: at 0.0 white, grey and black are the
    same colour, and selecting a white product takes the entire frame (IoU 0.16 against
    1.00) -- that case is in the eval precisely because every other case there prefers 0.0.
    """
    a = np.array(img.convert('RGB'))
    lab = to_oklab(a)
    target = to_oklab(np.array(target_rgb, dtype=float))

    dl = (lab[..., 0] - target[0]) * lightness_weight
    da = (lab[..., 1] - target[1]) * chroma_weight
    db = (lab[..., 2] - target[2]) * chroma_weight
    distance = np.sqrt(dl ** 2 + da ** 2 + db ** 2)

    # A ramp in the distance itself, rather than a step plus a Gaussian blur. This is the
    # substantive difference from the HSV path: the mask edge follows how far each pixel
    # actually is from the target colour, so a gradient fades out instead of being cut at a
    # boundary the object does not have. Full weight within `tolerance`, reaching zero at
    # twice it. No spatial feathering is needed or wanted -- blurring a mask is a way to
    # fake a soft edge when the selection itself is hard.
    soft = np.clip((2.0 * tolerance - distance) / max(tolerance, 1e-6), 0.0, 1.0)
    if smooth:
        # Only for suppressing single-pixel speckle on noisy JPEGs; off by default.
        from scipy.ndimage import gaussian_filter
        soft = np.clip(gaussian_filter(soft, sigma=smooth), 0.0, 1.0)
    return soft


def describe_selection(img, target_rgb, tolerance=0.06, **kwargs):
    """Measured facts about what a perceptual selection would take, for a caller to judge."""
    from scipy import ndimage
    mask = colour_mask(img, target_rgb, tolerance=tolerance, **kwargs)
    solid = mask > 0.5
    _, components = ndimage.label(solid)
    return {
        'matched_fraction': float(solid.mean()),
        'components': int(components),
        'mean_weight': float(mask[solid].mean()) if solid.any() else 0.0,
    }
