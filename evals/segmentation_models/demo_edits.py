"""The comparison as edits, not as masks -- what a user would actually get.

    python evals/segmentation_models/demo_edits.py [sam2tiny]

A mask is an intermediate: what matters is what it does to "make the car blue" or "keep
the pizza in colour". Runs the package's own compositing maths (copied from
`objects.recolour_object` / `objects.isolate_object`, since those segment internally and
cannot take an outside mask) on each mask in turn, feathered the same way, and puts the
results side by side.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cases as C                                                        # noqa: E402

from figsurgeon import objects as O                                     # noqa: E402
from figsurgeon.locate import _font                                     # noqa: E402
from figsurgeon.photo import _feather                                   # noqa: E402

# (case id, what to do, argument) -- one real request per case.
EDITS = [
    ('car_red', 'recolour', (40, 80, 200)),
    ('astronaut_suit', 'recolour', (40, 80, 200)),
    ('food_pizza', 'isolate', None),
    ('portrait_studio', 'isolate', None),
    ('flower_macro', 'isolate', None),
]


def recolour(img, mask, to_rgb):
    """objects.recolour_object's maths, driven by a mask given from outside."""
    a = np.array(img.convert('RGB')).astype(float)
    target = np.array(to_rgb, dtype=float)
    lum = a.mean(axis=2) / 255.0
    scale = np.clip(lum / max(target.mean() / 255.0, 1e-6), 0.35, 1.9)
    recoloured = np.clip(target[None, None, :] * scale[:, :, None], 0, 255)
    m = mask[:, :, None]
    return Image.fromarray((a * (1 - m) + recoloured * m).astype(np.uint8))


def isolate(img, mask, flatten=0.5):
    """objects.isolate_object's maths, driven by a mask given from outside."""
    a = np.array(img.convert('RGB')).astype(float)
    lum = a.mean(axis=2, keepdims=True)
    grey = np.repeat(lum, 3, axis=2) * (1 - flatten) + np.full_like(a, 128.0) * flatten
    m = mask[:, :, None]
    return Image.fromarray(np.clip(a * m + grey * (1 - m), 0, 255).astype(np.uint8))


def apply(img, mask, kind, arg):
    mask = _feather(np.asarray(mask) > 0.5, 2) if mask.dtype != float else mask
    return recolour(img, mask, arg) if kind == 'recolour' else isolate(img, mask)


def sheet(panels, title, width=1500):
    cell = width // len(panels) - 10
    thumbs = []
    for cap, p in panels:
        t = p.copy()
        t.thumbnail((cell, cell * 2), Image.LANCZOS)
        thumbs.append((cap, t))
    out = Image.new('RGB', (width, max(t.size[1] for _, t in thumbs) + 56), (246, 246, 248))
    d = ImageDraw.Draw(out)
    d.text((10, 6), title, font=_font(20), fill=(10, 10, 10))
    for i, (cap, t) in enumerate(thumbs):
        x = i * (width // len(panels)) + 6
        out.paste(t, (x, 44))
        d.text((x, 28), cap, font=_font(14), fill=(60, 60, 60))
    return out


def run(tag='slimsam'):
    out_dir = os.path.join(C.HERE, 'edits')
    os.makedirs(out_dir, exist_ok=True)
    by_id = {c['id']: c for c in C.load()}
    for case_id, kind, arg in EDITS:
        case = by_id[case_id]
        img = C.image(case)
        gc = O.segment_object(img, case['box'], backend='grabcut')
        gc = np.asarray(gc[0] if isinstance(gc, tuple) else gc).astype(float)
        gc = gc / 255 if gc.max() > 1.5 else gc
        sam = np.asarray(Image.open(
            os.path.join(C.HERE, f'masks_{tag}', case_id + '.png'))) > 127
        sam = _feather(sam, 2)
        request = ('"make it blue"' if kind == 'recolour'
                   else f'"keep {case["what"]} in colour"')
        W, H = img.size
        bx = case['box']
        px, py = int((bx[2] - bx[0]) * 0.12), int((bx[3] - bx[1]) * 0.12)
        crop = (max(0, bx[0] - px), max(0, bx[1] - py),
                min(W, bx[2] + px), min(H, bx[3] + py))
        panels = [('original', img.crop(crop)),
                  ('GrabCut (ships today)', apply(img, gc, kind, arg).crop(crop)),
                  ('SlimSAM-77', apply(img, sam, kind, arg).crop(crop))]
        sheet(panels, f'{case_id} -- {request}').save(
            os.path.join(out_dir, f'{case_id}_{kind}.png'))
        print(f'{case_id:18} {kind}')
    print('edits in', out_dir)


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else 'slimsam')
