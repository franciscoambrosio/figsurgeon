"""Take the object a generative model drew, and put it back in the original picture.

A generative editor draws a convincing object but redraws the whole scene; a mask editor
leaves the scene alone but flattens the object to one hue. `composite.compose_within` takes
each one's good half: the model's object, the original's everything-else.

Alignment is the hard part: a regenerated frame sits a few pixels off, so the object is
re-registered locally before compositing, to avoid a seam of old colour. Where the model
drew a different object, the paste takes its pixels inside the original's mask and nothing
else; `silhouette_gap` reports how much of the mask it failed to fill.
"""
import numpy as np
from PIL import Image, ImageFilter

from figsurgeon import perceptual


def _replicate(img, size, at):
    """Place `img` at `at` in a `size` frame, extending edge pixels to fill gaps left by a
    shifted paste. Not real content -- `register` reports the true coverage separately.
    """
    W, H = size
    x, y = at
    a = np.asarray(img.convert('RGB'))
    pad_l, pad_t = max(0, x), max(0, y)
    pad_r, pad_b = max(0, W - (x + a.shape[1])), max(0, H - (y + a.shape[0]))
    padded = np.pad(a, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode='edge')
    ox, oy = pad_l - x, pad_t - y
    return Image.fromarray(padded[oy:oy + H, ox:ox + W])


def poisson(original, edited, mask):
    """Gradient-domain clone of `edited` into `original` over `mask` (OpenCV seamlessClone).

    A hard paste shows a seam wherever illumination disagrees; Poisson cloning matches the
    boundary and takes only gradients from the source. It can bleed past the mask, so the
    result is re-composited through it afterwards: outside stays byte-identical.
    """
    import cv2
    m = (np.asarray(mask) > 0.05).astype(np.uint8)
    if not m.any():
        return edited
    ys, xs = np.where(m)
    centre = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
    dst = np.asarray(original.convert('RGB'))[:, :, ::-1].copy()
    src = np.asarray(edited.convert('RGB'))[:, :, ::-1].copy()
    out = cv2.seamlessClone(src, dst, m * 255, centre, cv2.NORMAL_CLONE)
    blended = Image.fromarray(out[:, :, ::-1])
    keep = m.astype(bool)
    a0 = np.asarray(original.convert('RGB')).copy()
    a0[keep] = np.asarray(blended)[keep]
    return Image.fromarray(a0)


def _bbox(mask, pad, size):
    ys, xs = np.where(mask > 0.05)
    if not len(ys):
        raise ValueError('empty mask')
    W, H = size
    return (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(W, int(xs.max()) + pad), min(H, int(ys.max()) + pad))


def register(original, edited, mask, pad=24, scales=(0.94, 0.97, 1.0, 1.03, 1.06), reach=14):
    """Shift/scale `edited` so its object lands where the original's object is.

    Scored on the ring around the mask, not the object -- the object changed colour by
    design, so matching it would score the thing deliberately altered.
    """
    x0, y0, x1, y1 = _bbox(mask, pad, original.size)
    ring = (mask[y0:y1, x0:x1] <= 0.05)
    o = np.asarray(original.convert('L')).astype(float)[y0:y1, x0:x1]
    best = None
    for s in scales:
        W, H = original.size
        scaled = edited.resize((max(1, int(W * s)), max(1, int(H * s))), Image.LANCZOS)
        for dy in range(-reach, reach + 1, 2):
            for dx in range(-reach, reach + 1, 2):
                frame = _replicate(scaled, (W, H),
                                   ((W - scaled.size[0]) // 2 + dx,
                                    (H - scaled.size[1]) // 2 + dy))
                e = np.asarray(frame.convert('L')).astype(float)[y0:y1, x0:x1]
                err = float(np.abs(e - o)[ring].mean()) if ring.any() else float('inf')
                if best is None or err < best[0]:
                    best = (err, s, dx, dy, frame)
    err, s, dx, dy, frame = best
    # The shift pastes onto an empty canvas, so it creates uncovered pixels of its own --
    # coverage has to travel with the transform.
    W, H = original.size
    scaled = edited.resize((max(1, int(W * s)), max(1, int(H * s))), Image.LANCZOS)
    cov = Image.new('L', (W, H), 0)
    cov.paste(Image.new('L', scaled.size, 255),
              ((W - scaled.size[0]) // 2 + dx, (H - scaled.size[1]) // 2 + dy))
    return frame, {'scale': s, 'dx': dx, 'dy': dy, 'ring_error': round(err, 2)}, \
        np.asarray(cov) > 0


def paste(original, edited, mask, feather=1.5, register_first=True, covered=None,
          blend='hard', grow=0, fallback=None):
    """Original everywhere, `edited` inside `mask`. Byte-identical outside it, by construction.

    Feathering happens strictly inside the mask (the mask is eroded before it is blurred),
    because softening outward would change pixels the caller was promised would not move.
    """
    info = {}
    if register_first:
        edited, info, reg_covered = register(original, edited, mask)
        covered = reg_covered if covered is None else (covered & reg_covered)

    # Only take pixels the model's frame actually reaches: a reframing model (GPT-5.4 returns
    # square regardless of input) leaves part of the frame uncovered; the original stands there.
    if covered is not None:
        info['uncovered_in_mask'] = round(float(((mask > 0.5) & ~covered).mean() /
                                                max((mask > 0.5).mean(), 1e-9)), 3)

    m = np.clip(np.asarray(mask, dtype=float), 0, 1)
    if grow:
        # The segmenter's edge sits inside the object's own edge, leaving a 1-3 px rim of the
        # old colour on a hard paste. That rim is part of the object, so the paste region is
        # widened (and reported as `grown_px`) rather than blending the seam away.
        from PIL import ImageFilter as _IF
        widened = np.asarray(Image.fromarray(((m > 0.05) * 255).astype(np.uint8))
                             .filter(_IF.MaxFilter(2 * int(grow) + 1))).astype(float) / 255
        info['grown_px'] = int((widened > 0.5).sum() - (m > 0.05).sum())
        m = np.maximum(m, widened)
    hard = (m > 0.05).astype(np.uint8) * 255
    inner = Image.fromarray(hard).filter(ImageFilter.MinFilter(3))
    soft = np.asarray(inner.filter(ImageFilter.GaussianBlur(feather))).astype(float) / 255.0
    soft = np.minimum(soft, (m > 0.05).astype(float))          # never reach outside the mask
    if covered is not None:
        if fallback is None:
            soft = soft * covered
        else:
            # Where the model's object doesn't reach, use the deterministic editor's output
            # instead of the untouched original -- leaving the gap unedited is wrong, since
            # it's inside the object. `fallback` is normally `recolour_subject`'s output.
            edited = Image.fromarray(np.where(covered[:, :, None],
                                              np.asarray(edited.convert('RGB')),
                                              np.asarray(fallback.convert('RGB'))))
            info['filled_from_fallback'] = round(float(((mask > 0.5) & ~covered).sum() /
                                                       max((mask > 0.5).sum(), 1)), 3)

    if blend == 'poisson':
        edited = poisson(original, edited, soft > 0.5)
    a0 = np.asarray(original.convert('RGB')).astype(float)
    a1 = np.asarray(edited.convert('RGB')).astype(float)
    out = a0 * (1 - soft[:, :, None]) + a1 * soft[:, :, None]
    out = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    # How much of the mask did the model leave looking like the original? Some of this is
    # correct -- the glass and the tyres of a recoloured car should not move -- so it is
    # reported as what it is, an unchanged share, and not as an error.
    inside = m > 0.5
    if inside.any():
        same = perceptual.delta_e(np.asarray(original.convert('RGB'))[inside],
                                  np.asarray(edited.convert('RGB'))[inside])
        info['unchanged_in_mask'] = round(float((same < 0.04).mean()), 3)
    return out, info
