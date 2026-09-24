"""Turn a PHRASE into a mask, so an edit can name what it means.

Everything else in this package needs a rectangle; this resolves natural-language subjects
("the sky", "the flowers") to pixels, so requests like "colour the sky" don't need a box.

Uses CLIPSeg rather than a detector+SAM pipeline, because "stuff" like sky has no object
outline for a detector to draw a box around, and CLIPSeg goes from words straight to a
region. The mask is produced at 352x352 and scaled up, so its edge is softer than SAM's;
pass `sharpen=True` to re-cut the mask's bounding box with SAM for a crisper edge, at the
cost of ~3 s extra.

See `evals/grounding.py` for measured phrases and timings.
"""
import numpy as np
from PIL import Image

MODEL_ID = 'CIDAS/clipseg-rd64-refined'          # Apache-2.0, ~150 MB
SHARPEN_MODEL_ID = 'nielsr/slimsam-77-uniform'   # Apache-2.0, ~40 MB

# Below this share of the frame there is nothing to act on: a mask that small is more
# likely to be nothing found than something tiny. Above the upper bound the phrase matches
# almost the whole picture (e.g. flowers filling the frame), so there is nothing to
# separate it from; refusing beats punching a grey blob through the centre.
MIN_COVERAGE = 0.002
MAX_COVERAGE = 0.90

_MODELS = {}


def _require():
    try:
        import torch                                                     # noqa: F401
        import transformers                                              # noqa: F401
    except ImportError:
        raise ImportError(
            'Naming a subject ("the sky", "the flowers") needs a grounding model. '
            'Install it with: pip install "figsurgeon[grounding]"  (torch + transformers, '
            'CPU is fine; the model itself is ~150 MB and downloads on first use).')


def available():
    """True when a phrase can be resolved. `photo_describe` asks before it refuses."""
    try:
        _require()
        return True
    except ImportError:
        return False


def _model():
    if MODEL_ID not in _MODELS:
        _require()
        from transformers import AutoProcessor, CLIPSegForImageSegmentation
        _MODELS[MODEL_ID] = (CLIPSegForImageSegmentation.from_pretrained(MODEL_ID).eval(),
                             AutoProcessor.from_pretrained(MODEL_ID))
    return _MODELS[MODEL_ID]


def mask_for(img, phrase, threshold=0.5, sharpen=False, feather=2):
    """Return (mask, note) for `phrase` in `img`, or (None, reason) if it found nothing.

    `mask` is float in [0, 1] at the image's size, ready for the same compositing every
    other operation here uses. `note` says what fraction of the frame it took, which is the
    number a caller should look at before trusting it -- see MIN_COVERAGE / MAX_COVERAGE.
    """
    _require()                      # a clear install message, before any torch import
    from .photo import _feather

    rgb = img.convert('RGB')
    up = _probability(rgb, phrase)
    hard = up > threshold
    coverage = float(hard.mean())

    if coverage < MIN_COVERAGE:
        return None, (f'could not find {phrase!r} in this image (the best match covers '
                      f'{coverage:.2%} of the frame, which is not a region). Try naming it '
                      f'differently, or pass a box explicitly.')
    if coverage > MAX_COVERAGE:
        return None, (f'{phrase!r} matched {coverage:.0%} of the frame, so it selects the '
                      f'picture rather than a part of it. Nothing here would be left '
                      f'unedited; name a smaller part, or edit the whole image directly.')

    if sharpen:
        hard = _sharpen(rgb, hard)
        coverage = float(hard.mean())

    hard, recovered = _recover_at_the_border(rgb, phrase, hard, threshold)
    if recovered:
        coverage = float(hard.mean())
    return _feather(hard, feather), (f'{phrase!r} selected {coverage:.1%} of the frame'
                                     + (' (edge re-cut with SAM)' if sharpen else '')
                                     + (f' (+{recovered:.1%} recovered at the frame edge)'
                                        if recovered else ''))


BORDER_BAND = 0.12      # of the shorter side: where CLIPSeg's own falloff lives
PAD = 0.25              # reflect-pad fraction for the second pass
MIN_RECOVERY = 0.005    # below this share of the frame, say nothing


def _recover_at_the_border(rgb, phrase, hard, threshold):
    """Add back the part of the region CLIPSeg loses at the edge of the frame.

    The model's probability fades towards the frame border, so a region running off the
    edge comes back cut off inside it. Re-runs on a reflect-padded copy and merges in only
    the border band that connects to what the first pass already found, so the whole-frame
    refusal and small-object detection both still see the first pass's own coverage. Only
    runs when the first pass's mask actually touches the border.
    """
    H, W = hard.shape
    band = max(4, int(BORDER_BAND * min(H, W)))
    border = np.zeros((H, W), bool)
    border[:band, :] = border[-band:, :] = True
    border[:, :band] = border[:, -band:] = True
    if not hard.any() or not (hard & border).any():
        return hard, 0.0

    import cv2
    from scipy import ndimage
    a = np.asarray(rgb)
    py, px = int(H * PAD), int(W * PAD)
    big = Image.fromarray(cv2.copyMakeBorder(a, py, py, px, px, cv2.BORDER_REFLECT))
    padded = _probability(big, phrase)[py:py + H, px:px + W] > threshold

    grown = hard | (padded & border)
    labels, n = ndimage.label(grown)
    keep = np.unique(labels[hard])
    grown = np.isin(labels, keep[keep > 0])
    gained = float(grown.mean() - hard.mean())
    if gained < MIN_RECOVERY:
        # Nothing meaningful was cut off: keep the first pass exactly, and say nothing.
        return hard, 0.0
    return grown, gained


def _probability(rgb, phrase):
    """CLIPSeg's per-pixel probability for `phrase`, at the image's own size."""
    import torch
    model, proc = _model()
    inputs = proc(text=[phrase], images=[rgb], padding=True, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    if logits.ndim == 2:                      # a single prompt comes back without a batch
        logits = logits[None]
    small = torch.sigmoid(logits)[0].numpy()
    return np.asarray(Image.fromarray((small * 255).astype(np.uint8))
                      .resize(rgb.size, Image.BILINEAR)) / 255.0


def _sharpen(rgb, hard):
    """Re-cut this mask's bounding box with SAM, for a crisper edge."""
    import torch
    from transformers import AutoProcessor, SamModel
    if SHARPEN_MODEL_ID not in _MODELS:
        _MODELS[SHARPEN_MODEL_ID] = (SamModel.from_pretrained(SHARPEN_MODEL_ID).eval(),
                                     AutoProcessor.from_pretrained(SHARPEN_MODEL_ID))
    model, proc = _MODELS[SHARPEN_MODEL_ID]
    ys, xs = np.where(hard)
    h, w = hard.shape
    box = [max(0, int(xs.min()) - 2), max(0, int(ys.min()) - 2),
           min(w, int(xs.max()) + 2), min(h, int(ys.max()) + 2)]
    inputs = proc(images=rgb, input_boxes=[[[float(v) for v in box]]], return_tensors='pt')
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)
    masks = proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs['original_sizes'].cpu(),
        inputs['reshaped_input_sizes'].cpu())[0][0]
    scores = np.ravel(out.iou_scores.cpu().numpy())
    return np.asarray(masks[int(np.argmax(scores))]).astype(bool)


def isolate_subject(img, phrase, flatten=0.5, threshold=0.5, sharpen=False):
    """Keep the named thing in colour, desaturate the rest. Returns (image, note).

    The spatial counterpart to `photo.isolate_colour`, addressed by NAME. This is the call
    behind "make it black and white except the flowers".
    """
    from .photo import _keep_alpha
    mask, note = mask_for(img, phrase, threshold=threshold, sharpen=sharpen)
    if mask is None:
        raise ValueError(note)
    a = np.array(img.convert('RGB')).astype(float)
    lum = a.mean(axis=2, keepdims=True)
    grey = np.repeat(lum, 3, axis=2) * (1 - flatten) + np.full_like(a, 128.0) * flatten
    m = mask[:, :, None]
    out = Image.fromarray(np.clip(a * m + grey * (1 - m), 0, 255).astype(np.uint8))
    return _keep_alpha(img, out), note


def recolour_subject(img, phrase, to_rgb, preserve_shading=True, threshold=0.5,
                     sharpen=False, keep_materials=True):
    """Recolour the named thing, keeping its own shading. Returns (image, note).

    Like `photo.replace_colour` but addressed by what the thing IS rather than its current
    colour, so it won't also recolour other pixels that happen to share that hue.

    `keep_materials=True` (default) leaves near-neutral and very dark pixels in the mask
    unpainted (a car's glass, tyres, chrome), since painting those the same hue reads as a
    cut-out. Pass False when the named thing really is one material throughout.
    """
    from .photo import _keep_alpha
    from .perceptual import paintable
    mask, note = mask_for(img, phrase, threshold=threshold, sharpen=sharpen)
    if mask is None:
        raise ValueError(note)
    a = np.array(img.convert('RGB')).astype(float)
    target = np.array(to_rgb, dtype=float)
    if preserve_shading:
        lum = a.mean(axis=2) / 255.0
        scale = np.clip(lum / max(target.mean() / 255.0, 1e-6), 0.35, 1.9)
        painted = target[None, None, :] * scale[:, :, None]
    else:
        painted = np.broadcast_to(target, a.shape)
    if keep_materials:
        w = paintable(a.astype(np.uint8))
        kept = float((mask > 0.05).sum() and (mask * (1 - w))[mask > 0.05].mean())
        mask = mask * w
        note += f'; {kept:.0%} of it left as its own material (glass, rubber, shadow)'
    m = mask[:, :, None]
    out = Image.fromarray(np.clip(a * (1 - m) + np.clip(painted, 0, 255) * m, 0, 255)
                          .astype(np.uint8))
    return _keep_alpha(img, out), note


def region_for(img, phrase, threshold=0.5, pad=0.02):
    """The bounding box of what `phrase` points at, or None.

    Bridges phrase-based requests (e.g. "erase the bird") to the box-based object tools,
    padded slightly so the segmenter inside those tools has edge context. Returns a box
    rather than the mask itself so that segmenter's own estimate stays independent evidence,
    used by `verify_photo.check_subject_recoloured`.
    """
    import numpy as np
    mask, note = mask_for(img, phrase, threshold=threshold)
    if mask is None:
        return None
    hit = np.asarray(mask) > threshold
    if not hit.any():
        return None
    ys, xs = np.where(hit)
    H, W = hit.shape
    dx, dy = int(W * pad), int(H * pad)
    return (max(0, int(xs.min()) - dx), max(0, int(ys.min()) - dy),
            min(W, int(xs.max()) + 1 + dx), min(H, int(ys.max()) + 1 + dy))
