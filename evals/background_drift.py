"""What did the edit change outside the thing it was asked to change?

`composite.py` claims a generative editor can't promise byte-identical pixels outside the
given region. This grades a real editor against figsurgeon on the same picture, request and
mask: DRIFT outside the mask (what changed that shouldn't) and INTENT inside it (whether the
requested change happened -- doing nothing scores zero drift). Drift is measured above a
resample floor and reported at three visibility thresholds plus a connected-blob count.

    python evals/background_drift.py --image horse.jpg --phrase "the horse" --to 128,0,128 \
        --model gpt-image-1=~/Downloads/horse_from_chatgpt.png

Writes a comparison sheet per editor and prints one JSON row per editor to stdout.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figsurgeon import grounding, perceptual

VISIBLE = 8        # /255 on some channel: visible against a smooth gradient
QUIET = 2          # /255: about what a JPEG round trip costs, reported not judged
# OKLab distance of roughly one just-noticeable difference at mid-tones. Not uniform in
# the blacks (OKLab's L is a cube root), so `share_jnd` over-reports on very dark frames;
# the verdict uses `share_visible` instead, with both printed.
JND = 0.02
MIN_BLOB = 64      # px: smaller connected regions are noise, not a changed thing


def align(before, after):
    """`after` at `before`'s size, plus what had to be done to get there.

    A returned frame of a different size is recorded in notes rather than silently resized
    away. A plain resize would misalign the whole frame and light up every edge as drift, so
    this searches a small scale/shift neighbourhood for the best fit -- giving the editor the
    benefit of the doubt on framing. Drift measured this way is a LOWER BOUND.
    """
    notes = []
    if after.size == before.size:
        return after.convert('RGB'), np.ones(before.size[::-1], bool), notes

    notes.append(f'returned {after.size[0]}x{after.size[1]} for a '
                 f'{before.size[0]}x{before.size[1]} input')
    ar0, ar1 = before.size[0] / before.size[1], after.size[0] / after.size[1]
    if abs(ar0 - ar1) > 0.01:
        notes.append(f'aspect ratio changed {ar0:.3f} -> {ar1:.3f}: content is stretched, '
                     'cropped or outpainted, so per-pixel drift is a lower bound')
    best, covered, (scale, dx, dy) = _best_fit(before, after)
    notes.append(f'best fit at scale {scale:.3f}, offset ({dx}, {dy})')
    missing = 1 - covered.mean()
    if missing > 0.001:
        notes.append(f'{missing:.1%} of the original frame is not covered by the return at '
                     'that fit -- those pixels are unknown and excluded, not counted as drift')
    return best, covered, notes


def _phase_shift(g0, g1):
    """Integer (dy, dx) that lines `g1` up with `g0`, by phase correlation."""
    F = np.fft.fft2(g0) * np.conj(np.fft.fft2(g1))
    mag = np.abs(F)
    mag[mag == 0] = 1
    r = np.real(np.fft.ifft2(F / mag))
    dy, dx = np.unravel_index(int(np.argmax(r)), r.shape)
    if dy > g0.shape[0] // 2:
        dy -= g0.shape[0]
    if dx > g0.shape[1] // 2:
        dx -= g0.shape[1]
    return int(dy), int(dx)


def _place(img, size, offset=(0, 0)):
    """Paste `img` into a `size` frame at centre + `offset`. `covered` is False where the
    return doesn't reach (unknown, not unchanged, and excluded from drift).
    """
    w, h = size
    canvas = Image.new('RGB', size)
    cov = Image.new('L', size, 0)
    x = (w - img.size[0]) // 2 + int(offset[0])
    y = (h - img.size[1]) // 2 + int(offset[1])
    canvas.paste(img, (x, y))
    cov.paste(Image.new('L', img.size, 255), (x, y))
    return canvas, np.asarray(cov) > 0


def _best_fit(before, after, scales=(0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.33, 1.50)):
    """Scale and shift that line `after` up with `before`, keeping AFTER's own aspect ratio
    (a square return of a 4:3 scene has invented or cropped content, not squashed it).
    `scale` is relative to the original's width. Returns (frame, covered, (scale, dx, dy)).
    """
    W, H = before.size
    sw = 256
    sh = max(1, int(sw * H / W))
    g0 = np.asarray(before.convert('L').resize((sw, sh), Image.LANCZOS)).astype(float)
    g0c = g0 - g0.mean()          # centred: for the correlation only, NOT for the error

    aw, ah = after.size
    rgb = after.convert('RGB')
    blind = float(np.abs(g0c).mean())    # the cost charged for a pixel the fit does not reach
    best = None
    for k in scales:
        cw = max(8, int(sw * k))
        ch = max(8, int(cw * ah / aw))
        cand = rgb.resize((cw, ch), Image.LANCZOS)
        centred, _ = _place(cand, (sw, sh))
        g1 = np.asarray(centred.convert('L')).astype(float)
        g1 -= g1.mean()
        dy, dx = _phase_shift(g0c, g1)
        # Score both the correlation's answer and no shift, keep the better: phase
        # correlation is confident even when wrong on periodic content (railings, tiling),
        # so this catches a confident misfire.
        for cand_dx, cand_dy in ((dx, dy), (0, 0)):
            placed, cov = _place(cand, (sw, sh), (cand_dx, cand_dy))
            if cov.sum() < 0.5 * cov.size:
                continue          # a fit covering less than half the frame is not a fit
            g2 = np.asarray(placed.convert('L')).astype(float)
            # Scored over the whole frame, charging each uncovered pixel the image's own
            # average deviation -- scoring only covered pixels would reward a smaller,
            # shrunken claim over the correct full-frame fit.
            err = float((np.abs(g2 - g0)[cov].sum() + (~cov).sum() * blind) / cov.size)
            if best is None or err < best[0]:
                best = (err, k, cand_dx, cand_dy)

    if best is None:
        return rgb.resize((W, H), Image.LANCZOS), np.ones((H, W), bool), (1.0, 0, 0)
    _, k, dx, dy = best
    cw = max(8, int(W * k))
    ch = max(8, int(cw * ah / aw))
    full = rgb.resize((cw, ch), Image.LANCZOS)
    frame, covered = _place(full, (W, H), (dx * W // sw, dy * H // sh))
    return frame, covered, (round(k, 3), dx * W // sw, dy * H // sh)


def _blobs(flag):
    """Connected regions of `flag` of at least MIN_BLOB px: (count, largest)."""
    try:
        from scipy import ndimage
    except ImportError:
        return None, None
    lab, n = ndimage.label(flag)
    if not n:
        return 0, 0
    sizes = np.bincount(lab.ravel())[1:]
    big = sizes[sizes >= MIN_BLOB]
    return int(big.size), int(sizes.max())


def drift(before, after, keep, covered=None):
    """Per-pixel change OUTSIDE `keep` (the mask the edit was allowed to touch).

    `covered` marks pixels the edited image actually reaches; uncovered pixels are unknown,
    not unchanged, and are excluded rather than counted either way.
    """
    a0 = np.asarray(before.convert('RGB')).astype(np.int16)
    a1 = np.asarray(after.convert('RGB')).astype(np.int16)
    outside = ~(np.asarray(keep) > 0.05)
    if covered is not None:
        outside &= covered
    n = int(outside.sum())
    if not n:
        return {'error': 'the mask covers the whole frame; nothing outside it to measure'}

    d = np.abs(a1 - a0).max(axis=2)
    de = perceptual.delta_e(a0.astype(np.uint8), a1.astype(np.uint8))
    vis = (d > VISIBLE) & outside
    count, largest = _blobs(vis)
    return {
        'background_px': n,
        'share_any': round(float(((d > 0) & outside).sum()) / n, 5),
        'share_quiet': round(float(((d > QUIET) & outside).sum()) / n, 5),
        'share_visible': round(float(vis.sum()) / n, 5),
        'share_jnd': round(float(((de > JND) & outside).sum()) / n, 5),
        'max_channel': int(d[outside].max()),
        'mean_channel': round(float(d[outside].mean()), 3),
        'mean_delta_e': round(float(de[outside].mean()), 4),
        'blobs': count, 'largest_blob_px': largest,
    }


def intent(before, after, keep, to_rgb):
    """Did the requested change happen INSIDE `keep`? The other half of the ledger.

    `moved_share` is the fraction of masked pixels that closed at least half the perceptual
    distance to the target, the same test `check_replace_colour` uses.
    """
    a0 = np.asarray(before.convert('RGB'))
    a1 = np.asarray(after.convert('RGB'))
    m = np.asarray(keep) > 0.05
    if not m.any():
        return {'error': 'empty mask'}
    target = np.zeros_like(a0)
    target[:, :] = np.array(to_rgb, dtype=np.uint8)
    d0 = perceptual.delta_e(a0, target)[m]
    d1 = perceptual.delta_e(a1, target)[m]
    return {
        'target_px': int(m.sum()),
        'delta_e_to_target_before': round(float(d0.mean()), 4),
        'delta_e_to_target_after': round(float(d1.mean()), 4),
        'moved_share': round(float((d1 <= d0 / 2).mean()), 4),
    }


def resample_floor(before, size, keep, to_rgb=None):
    """What the alignment round trip alone costs, so drift can be judged above it."""
    there_and_back = before.resize(size, Image.LANCZOS).resize(before.size, Image.LANCZOS)
    return drift(before, there_and_back, keep)


def sheet(path, before, after, keep, label, covered=None):
    """before | after | where it changed outside the mask (amplified, mask outlined)."""
    a0 = np.asarray(before.convert('RGB')).astype(np.int16)
    a1 = np.asarray(after.convert('RGB')).astype(np.int16)
    outside = ~(np.asarray(keep) > 0.05)
    if covered is not None:
        outside &= covered
    d = np.abs(a1 - a0).max(axis=2)
    heat = np.zeros(a0.shape, dtype=np.uint8)
    heat[..., 0] = np.clip(d * 8, 0, 255) * outside          # drift in red
    heat[..., 1] = np.clip(d * 8, 0, 255) * (np.asarray(keep) > 0.05)   # asked-for: green
    if covered is not None:                                  # unknown pixels: blue
        heat[..., 2] = np.where(covered, 0, 90).astype(np.uint8)
    panel = Image.fromarray(heat)

    w, h = before.size
    scale = min(1.0, 900 / (3 * w))
    tw, th = int(w * scale), int(h * scale)
    out = Image.new('RGB', (tw * 3, th), 'white')
    for i, im in enumerate((before.convert('RGB'), after.convert('RGB'), panel)):
        out.paste(im.resize((tw, th), Image.LANCZOS), (i * tw, 0))
    out.save(path)
    return path


def mask_from(img, phrase=None, box=None, mask_png=None):
    """One mask, used for every editor -- the comparison is void if they differ."""
    if mask_png:
        m = np.asarray(Image.open(mask_png).convert('L').resize(img.size)).astype(float) / 255
        return m, f'mask read from {mask_png}'
    if box:
        m = np.zeros(img.size[::-1], dtype=float)
        x0, y0, x1, y1 = box
        m[y0:y1, x0:x1] = 1.0
        return m, f'box {box}'
    if not grounding.available():
        raise SystemExit('a phrase needs figsurgeon[grounding]: '
                         'pip install "figsurgeon[grounding]" -- or pass --box/--mask')
    m, note = grounding.mask_for(img, phrase)
    if m is None:
        raise SystemExit(f'the phrase found nothing: {note}')
    return m, note


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--image', required=True, help='the original')
    p.add_argument('--phrase', help='what to edit, e.g. "the horse"')
    p.add_argument('--box', help='x0,y0,x1,y1 instead of a phrase')
    p.add_argument('--mask', help='a mask PNG instead of a phrase')
    p.add_argument('--to', default='128,0,128', help='target colour R,G,B')
    p.add_argument('--model', action='append', default=[], metavar='NAME=PATH',
                   help='an editor\'s result to grade; repeatable')
    p.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 'out_background_drift'))
    a = p.parse_args(argv)
    if not (a.phrase or a.box or a.mask):
        p.error('one of --phrase, --box or --mask is required')

    before = Image.open(os.path.expanduser(a.image)).convert('RGB')
    to_rgb = tuple(int(v) for v in a.to.split(','))
    box = tuple(int(v) for v in a.box.split(',')) if a.box else None
    keep, mask_note = mask_from(before, a.phrase, box, a.mask)
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.image))[0]

    editors = []
    if a.phrase and grounding.available():
        ours, note = grounding.recolour_subject(before, a.phrase, to_rgb)
        editors.append(('figsurgeon', ours, [note], before.size,
                        np.ones(before.size[::-1], bool)))
    for spec in a.model:
        name, _, path = spec.partition('=')
        img = Image.open(os.path.expanduser(path))
        aligned, covered, notes = align(before, img)
        editors.append((name, aligned, notes, img.size, covered))

    rows = []
    for name, edited, notes, returned_size, covered in editors:
        row = {'case': stem, 'editor': name, 'mask': mask_note, 'notes': notes,
               'drift': drift(before, edited, keep, covered),
               'covered': round(float(covered.mean()), 4),
               'intent': intent(before, edited, keep, to_rgb)}
        if returned_size != before.size:
            row['resample_floor'] = resample_floor(before, returned_size, keep)
        row['sheet'] = sheet(os.path.join(a.out, f'{stem}_{name}.png'),
                             before, edited, keep, name, covered)
        rows.append(row)
        print(json.dumps(row, indent=2))
    return rows


if __name__ == '__main__':
    main()
