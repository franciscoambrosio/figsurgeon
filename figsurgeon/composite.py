"""Keep an edit inside the region it was asked for -- provably.

This is a public primitive because "only what is inside this box changes" is a promise the
editors in this package already try to keep (`objects.segment_object` clips its feathered
mask back to the box for exactly this reason) and that `verify_photo.check_unchanged_outside`
already tests for. What was missing is the guarantee itself, separated from any particular
editor, so it can wrap one it knows nothing about.

That separation is the point. A generative editor cannot offer this guarantee by
construction: encoding an image to latents and decoding it back leaves measurable error
everywhere in the frame, not only where the edit was asked for -- the untouched half of the
picture comes back slightly different. `compose_within` makes that irrelevant, because it
takes the edited pixels only from inside `region` and copies the original bytes everywhere
else. The result is byte-identical outside `region` no matter what the editor did.

It also reports the leak instead of silently swallowing it. An editor that changed 300 000
pixels outside the box it was given is misbehaving, and the caller should be told so even
though the composite has already made it harmless.
"""
import numpy as np

# Spatial tools whose whole promise is "this changes ONLY the region you named", and the
# argument each one names it with. Kept here rather than in workspace.py because both the
# verification (workspace.verify) and the enforcement (tools.dispatch) need the same answer
# to "what was this operation allowed to touch"; two copies of that table would drift.
#
# `isolate_object` is deliberately not here despite taking a box: it desaturates the whole
# frame except the object, so changing pixels outside the box is its entire purpose. Listing
# it produced a confident "FAILED: 131100 px changed outside the target region" on a correct
# edit. Taking a box does not make an operation localised.
LOCALISED = {'recolour_object': 'box', 'erase_object': 'box',
             'remove_object': 'box', 'clean_transparent_box': 'box',
             'remap_colormap': 'box', 'add_text': None,
             'generative_fill': 'box', 'scale_object': 'box'}

# Two of those legitimately reach a few pixels beyond their box, and the amount is knowable
# rather than fuzzy: `erase_object` dilates the segmented mask by `grow` before inpainting
# (so the object's own anti-aliased edge doesn't survive as a faint outline), and OpenCV's
# inpainting then samples a further `radius` around it. Clipping or failing those at the
# bare box would break a real, intended behaviour. Everything else must honour its box
# exactly.
EDGE_SLACK = {
    'erase_object': lambda a: int(a.get('grow', 3)) + int(a.get('radius', 6)) + 2,
    'remove_object': lambda a: int(a.get('radius', 6)) + 2,
    # clean_transparent_box works on the box plus a 6 px margin, because it reconstructs the
    # legend's own border by mirroring -- which needs the border, and the border is at the
    # edge of the box the caller drew.
    'clean_transparent_box': lambda a: 8,
    # `scale_object` normally reports its own region (objects.Located.region, unioning where
    # the object was with where it ended up) and `enforce_region`/`verify` are handed that
    # directly -- this lambda is the fallback for when only `args` is available, and is not
    # reached through `tools.dispatch`'s own call graph today (that region is always on
    # hand by the time either consumer runs); `tests/test_composite.py` exercises it
    # directly rather than leaving it unverified. A fixed margin is right for a shrink (its
    # only reach past the mask is the inpainted ring, +3 px dilation and +6 px inpaint
    # radius, same numbers `erase_object` budgets for the same call) but is wrong for a
    # grow: scaling the box up by `scale` can push the object's far edge out by roughly
    # (scale-1) times the box's own half-extent, and a caller drawing a tight box around a
    # large object could grow it well past a small constant. Estimating that reach from the
    # box and scale, rather than a constant, is what keeps a large `scale` from being
    # silently clipped whenever this fallback is the only region on hand.
    'scale_object': lambda a: (
        11 if float(a.get('scale', 1.0)) <= 1.0 or not a.get('box') else
        int(max(int(a['box'][2]) - int(a['box'][0]),
               int(a['box'][3]) - int(a['box'][1])) * (float(a['scale']) - 1.0) / 2) + 11),
}


def allowed_region(name, args):
    """The region a localised operation may touch, box plus its own known reach, or None."""
    key = LOCALISED.get(name, False)
    if key is False:
        return None
    box = args.get(key) if key else None
    if not box or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = (int(v) for v in box)
        slack = EDGE_SLACK[name](args) if name in EDGE_SLACK else 0
    except (TypeError, ValueError):
        return None
    return (x0 - slack, y0 - slack, x1 + slack, y1 + slack)


def enforce_region(name, args, before, after, region=None):
    """Clip a localised tool's result back to what that tool was allowed to touch.

    Returns (image, note_or_None). This is the difference between VERIFYING the promise --
    which `workspace.verify` already did, after the damage -- and KEEPING it: a leak is
    removed here and reported in the same breath, rather than described to a model that
    then has to choose to undo.

    `region` overrides the box from `args`, and exists for ONE case: a segmenter that found
    the object continuing past the caller's box (`objects.Located.region`). Deriving the
    region from the arguments alone can cut an object off at the line the caller happened to
    draw -- e.g. a box around a suit that clips a sleeve, reported as verified because the
    cut sat inside the box. Stating the region the operation actually reached, and holding it
    to that, is still a region fixed before the pixels are compared, not one fitted to them:
    the part of the mask connected to the box decides it, and the note says what it was.

    Skips silently when the operation is not localised, when no box was given, or when the
    result changed size or mode (a crop or a cutout is a different picture, not a leak).
    """
    if LOCALISED.get(name, False) is False:
        # Not a localised operation, and a stated region must not make it one.
        # `isolate_object` desaturates the whole frame by design and is deliberately absent
        # from LOCALISED -- honouring its segmentation's region here reverted 59,900 px of
        # the grey it is supposed to produce, i.e. turned a working tool off.
        return after, None
    allowed = allowed_region(name, args)
    if region is None and allowed is None:
        return after, None                    # localised, but nothing said where
    region = region if region is not None else allowed
    if region is None or before.size != after.size or before.mode != after.mode:
        return after, None
    try:
        out, report = compose_within(before, region, after)
    except ValueError:
        return after, None            # an empty or off-image box: verify still reports it
    if not report['leaked_px']:
        return after, None
    return out, (f'{report["leaked_px"]} px outside the region this tool is allowed to '
                 f'touch were reverted (up to {report["leaked_max"]}/255)')


def _clamp(region, W, H):
    x0, y0, x1, y1 = (int(v) for v in region)
    # Clamping is not defensive boilerplate here -- see verify_photo.check_unchanged_outside:
    # a negative coordinate turns a slice into an empty one, which would silently make this
    # function a no-op that still claims to have composited.
    x0, x1 = max(0, min(W, x0)), max(0, min(W, x1))
    y0, y1 = max(0, min(H, y0)), max(0, min(H, y1))
    return x0, y0, x1, y1


def _arr(img):
    """A PIL image as an RGB float array. Nearly every measurement in this package starts
    here; named once so it is written once."""
    return np.array(img.convert('RGB')).astype(float)


def _padded(box, size, pad):
    """`box` expanded by `pad` times its own width/height on each side, clamped to `size`.
    The context margin a generative fill and a phrase-grounding crop both widen a tight box
    by, so the model or the classifier sees the background around it, not just the box."""
    W, H = size
    x0, y0, x1, y1 = (int(v) for v in box)
    mx, my = int((x1 - x0) * pad), int((y1 - y0) * pad)
    return _clamp((x0 - mx, y0 - my, x1 + mx, y1 + my), W, H)


def _blend(a, b, mask):
    """`a` where `mask` is 0, `b` where it is 1, linearly in between -- the feathered
    composite most masked edits in this package build by hand. `mask` broadcasts over a
    trailing channel axis if it has one fewer dimension than `a`/`b`."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.asarray(mask, dtype=float)
    if m.ndim == a.ndim - 1:
        m = m[..., None]
    return a * (1 - m) + b * m


def compose_within(image, region, edited):
    """Return `image` with `region` taken from `edited`, byte-identical everywhere else.

    `region` is [x0, y0, x1, y1] with x1/y1 exclusive, the convention used throughout this
    package. `edited` is either a full-frame result from some editor, or just the edited
    crop at the size of the (clamped) region -- a crop-in, crop-out editor needs no
    full-frame round trip.

    Returns (out, report). `report` carries what the editor did outside its region before
    this function discarded it:

        leaked_px    pixels outside `region` that the editor changed (0 for a crop)
        leaked_max   the largest channel difference among them, 0-255
        note         a sentence saying so

    The composite is a hard paste: `region` is the caller's contract, and softening it would
    mean changing pixels outside the region, which is the one thing this function promises
    not to do. Feather inside the region, before calling this.
    """
    W, H = image.size
    x0, y0, x1, y1 = _clamp(region, W, H)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f'region {tuple(region)} is empty or outside the {W}x{H} image')

    if edited.mode != image.mode:
        edited = edited.convert(image.mode)
    a = np.array(image)
    e = np.array(edited)

    if e.shape[:2] == (y1 - y0, x1 - x0):
        patch, leaked_px, leaked_max = edited, 0, 0
        outside = 'the editor returned only the crop, so nothing outside it could change'
    elif e.shape == a.shape:
        d = np.abs(e.astype(int) - a.astype(int))
        if d.ndim == 2:
            d = d[:, :, None]
        d = d.max(axis=2)
        d[y0:y1, x0:x1] = 0
        leaked_px, leaked_max = int((d > 0).sum()), int(d.max())
        patch = edited.crop((x0, y0, x1, y1))
        outside = (f'discarded {leaked_px} px the editor changed outside it '
                   f'(up to {leaked_max}/255)' if leaked_px else
                   'the editor changed nothing outside it')
    else:
        raise ValueError(
            f'`edited` is {edited.size[0]}x{edited.size[1]}, which is neither the whole '
            f'{W}x{H} image nor the {x1 - x0}x{y1 - y0} region')

    out = image.copy()          # a PIL paste keeps the image's own mode (P, CMYK, ...)
    out.paste(patch, (x0, y0))
    note = (f'composited into [{x0}, {y0}, {x1}, {y1}]; {outside}')
    return out, {'leaked_px': leaked_px, 'leaked_max': leaked_max,
                 'region': (x0, y0, x1, y1), 'note': note}
