"""Run GrabCut on the same eight boxes and build the comparison sheets.

    python evals/segmentation_models/compare.py [sam2tiny slimsam samvitb ...]

Reads whatever `masks_<tag>/` directories `run_models.py` has produced and puts GrabCut
beside them, one sheet per case. Uses the project environment; no torch needed.

The overlay drains everything outside the mask to grey rather than tinting the mask,
since a colour tint is unreadable when the subject shares its hue. Every panel is
cropped to the box: at full-frame scale a thumbnail hides real misses.
"""
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cases as C                                                        # noqa: E402

from figsurgeon import objects as O                                     # noqa: E402
from figsurgeon.locate import _font                                     # noqa: E402


def overlay(img, mask, box):
    a = np.asarray(img.convert('RGB')).astype(float)
    m = np.asarray(mask).astype(float)
    if m.max() > 1.5:
        m = m / 255.0
    m = m[..., None]
    drained = a.mean(axis=2, keepdims=True) * 0.35 + 40
    out = Image.fromarray((a * m + drained * (1 - m)).astype(np.uint8))
    ImageDraw.Draw(out).rectangle(list(box), outline=(0, 220, 255), width=3)
    return out


def sheet(img, box, panels, title, width=1900):
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
        d.text((x, 28), cap, font=_font(13), fill=(60, 60, 60))
    return out


def run(tags):
    out_dir = os.path.join(C.HERE, 'sheets')
    os.makedirs(out_dir, exist_ok=True)
    timings = {t: json.load(open(os.path.join(C.HERE, f'masks_{t}', 'timings.json')))
               for t in tags}
    for case in C.load():
        img = C.image(case)
        box = case['box']
        W, H = img.size
        px, py = int((box[2] - box[0]) * 0.12), int((box[3] - box[1]) * 0.12)
        crop = (max(0, box[0] - px), max(0, box[1] - py),
                min(W, box[2] + px), min(H, box[3] + py))
        t0 = time.time()
        gc = O.segment_object(img, box, backend='grabcut')
        gc_seconds = time.time() - t0
        gc = gc[0] if isinstance(gc, tuple) else gc
        panels = [(f'GrabCut {gc_seconds:.1f}s', overlay(img, gc, box).crop(crop))]
        line = f"{case['id']:18} GrabCut {gc_seconds:6.1f}s"
        for tag in tags:
            mask = np.asarray(Image.open(
                os.path.join(C.HERE, f'masks_{tag}', case['id'] + '.png'))) > 127
            seconds = timings[tag][case['id']]['seconds']
            panels.append((f'{tag} {seconds:.1f}s', overlay(img, mask, box).crop(crop)))
            line += f" | {tag} {seconds:5.1f}s"
        print(line)
        sheet(img, box, panels, f"{case['id']} -- {case['what']}").save(
            os.path.join(out_dir, case['id'] + '.png'))
    print('sheets in', out_dir)


if __name__ == '__main__':
    run(sys.argv[1:] or ['slimsam', 'sam2tiny', 'samvitb'])
