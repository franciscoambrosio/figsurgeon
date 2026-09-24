"""Seeing where to edit: coordinate overlays, box previews, and box refinement.

Almost every spatial operation in this package takes a box (`recolour_object`,
`erase_object`, `crop`, ...) with no detection of its own -- the box comes from a vision
model reading pixel coordinates off an image, which it does unreliably, and a wrong one
commits an irreversible edit and reports success.

The fix is a loop, not a better guess:

    1. `grid_overlay`   -- read coordinates off the image instead of estimating them
    2. `draw_boxes`     -- see the proposed box on the image before committing to it
    3. `refine_box`     -- snap a roughly-right box onto the object actually inside it
    4. `mask_overlay`   -- see exactly which pixels an edit would touch
    5. `zoom`           -- inspect a small region (or a finished edit) at a readable size

None of these modify the image; they aim the destructive call that follows.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .composite import _arr, _blend, _padded

BOX_COLOURS = [(255, 0, 0), (0, 120, 255), (0, 200, 0), (255, 140, 0),
               (200, 0, 255), (0, 200, 200)]


def _font(size):
    """A truetype font at `size`, falling back to PIL's bitmap font at the closest size."""
    from . import rebrand
    try:
        return ImageFont.truetype(rebrand.LIBERATION_SANS, size)
    except OSError:
        pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _nice_step(extent, target_lines=10):
    """A round grid step (…25, 50, 100, 250…) giving roughly `target_lines` divisions."""
    raw = max(1.0, extent / max(1, target_lines))
    magnitude = 10 ** int(np.floor(np.log10(raw)))
    for mult in (1, 2, 2.5, 5, 10):
        if magnitude * mult >= raw:
            return int(magnitude * mult)
    return int(magnitude * 10)


def grid_overlay(img, step='auto', colour=(255, 0, 255), opacity=0.55, label_size=None):
    """Return a copy of `img` with a labelled pixel-coordinate grid drawn over it, so a
    model can read coordinates instead of estimating them. `step` defaults to a round
    number giving ~10 divisions on the longer side."""
    base = img.convert('RGB')
    W, H = base.size
    if step == 'auto':
        step = _nice_step(max(W, H))
    step = max(1, int(step))

    layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a = int(255 * float(np.clip(opacity, 0, 1)))
    bright = colour + (a,)
    shadow = (0, 0, 0, a)

    for x in range(0, W, step):
        d.line([(x + 1, 0), (x + 1, H)], fill=shadow, width=1)
        d.line([(x, 0), (x, H)], fill=bright, width=1)
    for y in range(0, H, step):
        d.line([(0, y + 1), (W, y + 1)], fill=shadow, width=1)
        d.line([(0, y), (W, y)], fill=bright, width=1)

    out = Image.alpha_composite(base.convert('RGBA'), layer).convert('RGB')
    d = ImageDraw.Draw(out)
    size = label_size or max(11, min(20, int(step / 3)))
    fnt = _font(size)

    def label(text, xy, anchor):
        # A filled plate behind the text keeps it legible over busy content.
        x, y = xy
        box = d.textbbox((x, y), text, font=fnt, anchor=anchor)
        if box[2] > W:
            # Avoid clipping the label into a shorter, wrong-looking number.
            x, anchor = x - 4, 'r' + anchor[1]
            box = d.textbbox((x, y), text, font=fnt, anchor=anchor)
        d.rectangle([box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1], fill=(0, 0, 0))
        d.text((x, y), text, font=fnt, fill=(255, 255, 255), anchor=anchor)

    # Skip the last line near the edge: its pushed-back label would overlap the previous
    # one into a single wrong-looking number. The size is printed in the corner anyway.
    edge = step * 0.6
    for x in range(step, W, step):
        if W - x < edge:
            continue
        label(str(x), (x + 2, 2), 'la')
        label(str(x), (x + 2, H - 2), 'ld')
    for y in range(step, H, step):
        if H - y < edge:
            continue
        label(str(y), (2, y + 2), 'la')
        label(str(y), (W - 2, y + 2), 'ra')
    label(f'{W}x{H}', (W - 2, H - 2), 'rd')
    return out


def draw_boxes(img, boxes, labels=None, colours=None, width=None, dim_outside=0.0):
    """Draw one or more [x0,y0,x1,y1] boxes on a copy of `img`, labelled. `dim_outside`
    (0..1) darkens everything outside the boxes, to check nothing else is caught inside."""
    out = img.convert('RGB').copy()
    W, H = out.size
    boxes = [tuple(int(v) for v in b) for b in boxes]

    if dim_outside > 0:
        a = np.array(out).astype(float)
        keep = np.zeros((H, W), bool)
        for x0, y0, x1, y1 in boxes:
            keep[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
        a[~keep] *= (1.0 - float(np.clip(dim_outside, 0, 1)))
        out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

    d = ImageDraw.Draw(out)
    width = width or max(2, int(round(max(W, H) / 400)))
    fnt = _font(max(12, int(round(max(W, H) / 45))))
    for i, box in enumerate(boxes):
        c = (colours or BOX_COLOURS)[i % len(colours or BOX_COLOURS)]
        x0, y0, x1, y1 = box
        d.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=width + 2)
        d.rectangle([x0, y0, x1, y1], outline=c, width=width)
        text = (labels[i] if labels and i < len(labels) else None) or f'{box}'
        # Label above the box, or inside it when the box sits against the top edge.
        ty = y0 - 4 if y0 > 24 else y1 + 4
        anchor = 'ld' if y0 > 24 else 'la'
        tb = d.textbbox((x0, ty), text, font=fnt, anchor=anchor)
        d.rectangle([tb[0] - 3, tb[1] - 2, tb[2] + 3, tb[3] + 2], fill=c)
        d.text((x0, ty), text, font=fnt, fill=(255, 255, 255), anchor=anchor)
    return out


def mask_overlay(img, mask, outline_colour=(0, 255, 0), dim=0.75, outline=True):
    """Show which pixels a mask selects: selected stay in full colour, the rest is dimmed.
    `mask` is a float 0..1 array the size of the image, as returned by
    `photo.replace_colour`, `photo.isolate_colour`, and `objects.segment_object`."""
    a = _arr(img)
    m = np.clip(np.asarray(mask, dtype=float), 0, 1)
    if m.shape != a.shape[:2]:
        raise ValueError(f'mask shape {m.shape} does not match image {a.shape[:2]}')

    grey = np.repeat(a.mean(axis=2, keepdims=True), 3, axis=2) * (1.0 - float(dim))
    out = _blend(grey, a, m)

    if outline:
        from scipy.ndimage import binary_dilation
        solid = m > 0.5
        edge = binary_dilation(solid, iterations=2) & ~solid
        out[edge] = np.array(outline_colour, dtype=float)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _largest_component(binary):
    """The biggest connected blob in a boolean mask, or the input if it is empty.

    Segmentation backends routinely return the object plus scattered specks elsewhere in
    the box (a shadow, a same-coloured fragment); a bbox over all foreground pixels would
    inflate to cover those specks instead of the object.
    """
    from scipy import ndimage
    labels, n = ndimage.label(binary)
    if n <= 1:
        return binary
    sizes = ndimage.sum(binary, labels, range(1, n + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def refine_box(img, box, expand=0.15, min_coverage=0.02, max_coverage=0.92):
    """Snap a roughly-right box onto the object inside it. Returns (box, info).

    Pads the caller's box, segments inside it, and returns the tight bounding box of the
    largest connected component. When segmentation finds almost everything
    (`max_coverage`) or almost nothing (`min_coverage`), that's not a found object: the
    original box is returned unchanged with `accepted=False` and a reason.
    """
    from .objects import segment_object

    W, H = img.size
    x0, y0, x1, y1 = (int(v) for v in box)
    x0, x1 = sorted((max(0, x0), min(W, x1)))
    y0, y1 = sorted((max(0, y0), min(H, y1)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return tuple(box), {'accepted': False, 'reason': f'box {tuple(box)} is empty'}

    padded = _padded((x0, y0, x1, y1), (W, H), expand)

    try:
        mask, coverage = segment_object(img, padded, feather=0)
    except (ValueError, ImportError) as e:
        return tuple(box), {'accepted': False, 'reason': f'segmentation failed: {e}'}

    if coverage >= max_coverage:
        return tuple(box), {
            'accepted': False, 'coverage': coverage,
            'reason': (f'segmentation kept {coverage:.0%} of the box -- it did not find a '
                       f'distinct object, so there is nothing to snap to. The object may '
                       f'not contrast with its surroundings, or the box may already be '
                       f'tight around it.')}
    if coverage <= min_coverage:
        return tuple(box), {
            'accepted': False, 'coverage': coverage,
            'reason': (f'segmentation found only {coverage:.1%} of the box -- too little to '
                       f'be the object. Try a box centred more tightly on it.')}

    solid = _largest_component(mask > 0.5)
    ys, xs = np.nonzero(solid)
    if xs.size == 0:
        return tuple(box), {'accepted': False, 'reason': 'no foreground pixels found'}
    new = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    shift = max(abs(new[i] - (x0, y0, x1, y1)[i]) for i in range(4))
    return new, {
        'accepted': True, 'coverage': coverage, 'original': (x0, y0, x1, y1),
        'max_edge_shift': shift,
        'reason': (f'snapped to the segmented object: {(x0, y0, x1, y1)} -> {new} '
                   f'(largest edge moved {shift} px; segmentation covered {coverage:.0%} '
                   f'of the padded search area)')}


def zoom(img, box, max_side=768, pad=0.1):
    """A padded crop of `box`, scaled up to a size a vision model can actually read.

    Used before an edit (confirm the box contains what was meant) and after (inspect the
    boundary for artifacts) -- neither is visible in a downscaled full frame.
    """
    W, H = img.size
    x0, y0, x1, y1 = (int(v) for v in box)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
    x0, y0 = max(0, x0 - px), max(0, y0 - py)
    x1, y1 = min(W, x1 + px), min(H, y1 + py)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f'box {tuple(box)} is empty or outside the {W}x{H} image')

    crop = img.convert('RGB').crop((x0, y0, x1, y1))
    scale = min(max_side / max(crop.size), 8.0)
    if scale > 1.0:
        crop = crop.resize((max(1, int(crop.size[0] * scale)),
                           max(1, int(crop.size[1] * scale))), Image.LANCZOS)
    return crop, (x0, y0, x1, y1)


def describe_box(img, box):
    """Measured facts about a region: size, share of the frame, and its colour statistics.

    Catches two mistakes a box preview does not: a box in the wrong coordinate convention
    (shows up as an absurd size or share) and a near-uniform region (likely background,
    not the object).
    """
    W, H = img.size
    x0, y0, x1, y1 = (int(v) for v in box)
    x0c, x1c = max(0, min(W, x0)), max(0, min(W, x1))
    y0c, y1c = max(0, min(H, y0)), max(0, min(H, y1))
    info = {'box': (x0, y0, x1, y1), 'image_size': (W, H),
            'clipped': (x0c, y0c, x1c, y1c) != (x0, y0, x1, y1)}
    if x1c <= x0c or y1c <= y0c:
        info['error'] = f'box {(x0, y0, x1, y1)} is empty or outside the {W}x{H} image'
        return info

    a = _arr(img)[y0c:y1c, x0c:x1c]
    info.update({
        'width': x1c - x0c, 'height': y1c - y0c,
        'frame_fraction': ((x1c - x0c) * (y1c - y0c)) / float(W * H),
        'median_rgb': tuple(int(v) for v in np.median(a.reshape(-1, 3), axis=0)),
        'mean_rgb': tuple(int(v) for v in a.reshape(-1, 3).mean(axis=0)),
        'detail_std': float(a.mean(axis=2).std()),
    })
    return info
