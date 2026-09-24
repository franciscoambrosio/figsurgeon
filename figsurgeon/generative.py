"""Redraw ONE region with a hosted image model, and keep the rest byte-identical.

    from figsurgeon import generative
    out, info = generative.fill(img, (120, 80, 300, 240),
                                'fill this area with the wall behind it')

A generative editor given a whole picture regenerates the whole picture: the object comes
back convincing and everything around it has moved a few pixels, changed size, or been
quietly restyled. `evals/hybrid.py` exists to undo that damage after the fact, by searching
for the scale and shift that re-registers the return against the original. Sending a crop
removes the problem instead of correcting it -- the bounds are ours, so the return goes back
exactly where it came from and no search is needed. It is also cheaper, because image output
is priced per token and a region is smaller than a frame.

The crop is padded because a tight crop of a hole is a picture of a hole: the model cannot
match a background it cannot see, and a bare box, sent and pasted back, produces fills that
are internally plausible and wrong at every edge. `pad` widens the crop so the surrounding
background travels with the request as context. The model's rendering of that ring is then
discarded: only the box itself is pasted back, which is what makes the outside-is-untouched
promise hold by construction rather than by inspection afterwards.

The call is non-deterministic, costs money, and needs a network and an OPENROUTER_API_KEY.
This module does not tell you the fill looks right -- see `seam` in the returned info, and
the note on thresholds in `verify_photo.check_generative_fill`.
"""
import base64
import io
import time

import numpy as np
from PIL import Image

from .composite import compose_within, _blend, _padded
from .openrouter import _post
from . import ringfit
from .verify_photo import fill_seam, fill_seam_mask

# Matches `evals/openrouter_edit.py`'s default, deliberately: what the evals grade and what
# the tool calls should not be two different models.
DEFAULT_MODEL = 'google/gemini-3.1-flash-lite-image'

# Appended to every instruction. The models do the asked-for edit and then keep going --
# recolour the taxi and a second car appears in front of it -- and anything they invent
# outside the requested change is damage the caller has to notice and undo.
#
# Worded to avoid priming what it forbids: naming the thing not to do ("do not add cars")
# hands an image model a noun it tends to draw, since negation is the weakest part of a
# sentence. So this states the boundary in terms of the region and what must survive, not in
# terms of objects to avoid, and says it once rather than three times over.
#
# Not measured: whether it lowers the rate of invented content needs a corpus of repeated
# calls with and without it, and each trial costs money, so it ships on judgement -- as a
# constant here an eval can switch off, rather than a string buried in the prompt.
# `mask_to_object=True` remains the real guarantee: it keeps invented content from reaching
# the output at all, rather than asking the model to refrain.
ONLY_WHAT_WAS_ASKED = (
    'Apply only the change described above. Everything else already in this region -- every '
    'other object, its position, its shape and its colour -- must come back exactly as it is '
    'now. Introduce nothing that is not there.')


def fill(image, box, instruction, model=DEFAULT_MODEL, pad=0.6, timeout=300, seed=None,
        mask_to_object=False, match_colour=True):
    """Redraw `box` per `instruction`. Returns (out, info); everything outside `box` is kept.

    `pad` is how much surrounding background to send as context, as a fraction of the box's
    own size on each side. 0 sends the bare box and the model has nothing to match.

    `mask_to_object=False` (default) keeps the box's own promise: every pixel inside it may
    change, which is right when there is nothing to preserve there -- a hole, a gap, a
    smeared erase. `mask_to_object=True` is for redrawing ONE object rather than a region:
    `objects.segment_object` finds it inside `box` (the same call `recolour_object` and
    `erase_object` make, `follow_object=False` so the mask cannot reach past the box the
    model was shown), and only pixels inside that mask are taken from the model's answer --
    everywhere else in the box keeps the ORIGINAL pixels, not the model's version of them.
    Without this, "repaint the car" sends back a whole rectangle that includes the
    background around the car, and the model is free to shift or restyle it slightly even
    when asked only about the car -- the exact failure a hard mask-composite guarantee
    exists to rule out everywhere else in this package. Raises if the box contains nothing
    the segmenter can find, rather than silently falling back to filling the whole box.

    `match_colour=True` (default) corrects the return's exposure/white balance against the
    context ring -- the padded margin the model was shown, whose true pixels we already have
    -- so a fill whose content is right does not read as a patch because its background sits
    a shade off. See `ringfit`. It is not applied on faith: the fill is composed both ways
    and the correction is kept only if the measured seam (`info['seam']`) improves.
    `info['colour_fit']` records the fit, both seam numbers and whether it was used, because
    a colour-matched wrong fill is a wrong fill that looks more convincing -- better picture,
    worse evidence.

    """
    W, H = image.size
    x0, y0, x1, y1 = (int(v) for v in box)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f'box {tuple(box)} is empty')
    rgb = image.convert('RGB')          # converted once; every crop below reads from it
    ctx = _padded(box, image.size, pad)
    crop = rgb.crop(ctx)

    buf = io.BytesIO()
    crop.save(buf, 'JPEG', quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode()
    # The model is told what the crop is, because it is not being shown the whole picture
    # and the instruction was written about the whole picture.
    prompt = (f'{instruction}\n\n{ONLY_WHAT_WAS_ASKED}\n\nThis is a {crop.size[0]}x'
              f'{crop.size[1]} region cut out of a larger {W}x{H} image. Edit it and return '
              f'an image of the SAME size and framing. Match the existing lighting, texture '
              f'and colour so the result can be pasted back into the larger image without a '
              f'visible seam.')
    body = {'model': model, 'modalities': ['image', 'text'],
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url',
                 'image_url': {'url': 'data:image/jpeg;base64,' + b64}}]}]}
    if seed is not None:
        body['seed'] = seed

    t0 = time.time()
    r = _post(body, timeout)
    info = {'model': model, 'seconds': round(time.time() - t0, 1),
            'context': list(ctx), 'sent': list(crop.size),
            'cost': (r.get('usage') or {}).get('cost')}
    if 'choices' not in r:
        raise RuntimeError(f'the image model returned no choices: {str(r.get("error", r))[:300]}')
    msg = r['choices'][0]['message']
    images = msg.get('images') or []
    if not images:
        # A refusal and an empty return are different failures and the caller can act on
        # neither by retrying blindly, so both say which one happened.
        why = msg.get('refusal') or (msg.get('content') or '')[:200] or 'no reason given'
        raise RuntimeError(f'the image model returned no image ({why})')

    raw = base64.b64decode(images[0]['image_url']['url'].split(',', 1)[1])
    got = Image.open(io.BytesIO(raw)).convert('RGB')
    info['returned'] = list(got.size)
    if got.size != crop.size:
        # Expected, not exceptional: several of these models answer at their own fixed
        # sizes. The crop's bounds are known, so this is one honest resize back onto them,
        # not the registration search a full-frame return would need.
        got = got.resize(crop.size, Image.LANCZOS)
        info['rescaled'] = True

    # Paste the BOX only. The padded ring was context; the model's version of it is dropped,
    # which is what keeps the promise to everything outside the box.
    box_in_crop = (x0 - ctx[0], y0 - ctx[1], x1 - ctx[0], y1 - ctx[1])
    full_mask = mask = None
    if mask_to_object:
        from . import objects
        full_mask, coverage = objects.segment_object(image, (x0, y0, x1, y1),
                                                      follow_object=False)
        if coverage <= 0.0001:
            raise RuntimeError(
                f'no object found inside {(x0, y0, x1, y1)} to mask the fill to -- check '
                f'the box with preview_object_mask, or drop mask_to_object to fill the '
                f'whole box')
        mask = full_mask[y0:y1, x0:x1]
        info['masked_fraction'] = round(float(mask.mean()), 4)
    orig_box = np.asarray(rgb.crop((x0, y0, x1, y1)))

    def _compose(answer):
        """The whole paste for ONE version of the model's return -- cheap enough to run twice,
        which is what lets the colour correction be kept on evidence rather than on faith."""
        piece = answer.crop(box_in_crop)
        if full_mask is not None:
            piece = Image.fromarray(np.clip(_blend(orig_box, piece, mask), 0, 255)
                                    .astype(np.uint8))
        composed, rep = compose_within(image, (x0, y0, x1, y1), piece)
        seam = (fill_seam_mask(image, composed, full_mask) if full_mask is not None
                else fill_seam(image, composed, (x0, y0, x1, y1)))
        return composed, rep, seam

    out, report, seam = _compose(got)
    fitted = {'applied': False, 'seam_before': seam, 'seam_after': None,
              'gain': None, 'offset': None}
    if match_colour:
        ring = ringfit.ring_mask((got.size[1], got.size[0]), box_in_crop)
        fitted['ring_px'] = int(ring.sum())
        if fitted['ring_px'] >= ringfit.MIN_RING_PX:
            model_ring, real_ring = np.asarray(got)[ring], np.asarray(crop)[ring]
            fitted['r'] = [round(v, 3) for v in ringfit.agreement(model_ring, real_ring)]
            gain, offset, why = ringfit.fit(model_ring, real_ring)
            fitted['gain'] = [round(float(g), 4) for g in gain]
            fitted['offset'] = [round(float(o), 5) for o in offset]
            fitted['refused'] = why
            if not why:
                fixed = Image.fromarray(ringfit.apply(np.asarray(got), gain, offset))
                out2, report2, seam2 = _compose(fixed)
                fitted['seam_after'] = seam2
                # Judged on the held-out half of the ring, not on the seam. The seam is an
                # edge measure and this correction is a whole-patch one: on a real return
                # whose sky was visibly off it read 0.004 before and 0.003 after, so a
                # difference a person could see moved it by 0.001 and the correction
                # survived by luck. The held-out half of the ring measures the quantity
                # actually being corrected, on pixels the fit never saw -- and it rejects a
                # fit with nothing to correct, since applying one costs a round-trip
                # through 8-bit.
                before, after = ringfit.holdout_gain(model_ring, real_ring, gain, offset)
                fitted['ring_err'] = [before, after]
                if before is not None and after < before:
                    out, report, seam = out2, report2, seam2
                    fitted['applied'] = True
    info['colour_fit'] = fitted
    info['leaked_px'] = report['leaked_px']
    info['seam'] = seam
    a0 = np.asarray(rgb.crop((x0, y0, x1, y1)))
    a1 = np.asarray(out.convert('RGB').crop((x0, y0, x1, y1)))
    info['changed_fraction'] = round(float((np.abs(a0.astype(int) - a1.astype(int)).max(2)
                                            > 2).mean()), 4)
    if full_mask is not None:
        info['mask'] = full_mask
    return out, info
