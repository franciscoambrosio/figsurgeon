"""What clipping a SAM mask to the prompt box actually costs, in pixels and in pictures.

    python evals/segmentation_models/clipping.py        # needs figsurgeon[grounding]

`objects.segment_object` clips its mask to the caller's box, since `composite.enforce_region`
promises a localised tool changes nothing outside it -- but a SAM mask finds the object and
can stick out of the box. This measures the cost of clipping it anyway. Panels are cropped
to the box union the mask's own bounding box, plus a margin, so what lies outside the box is
visible.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cases as C                                                        # noqa: E402
import compare                                                           # noqa: E402

from figsurgeon import objects as O                                     # noqa: E402


def run():
    out_dir = os.path.join(C.HERE, 'sheets_clipping')
    os.makedirs(out_dir, exist_ok=True)
    rows = {}
    for case in C.load():
        img = C.image(case)
        box = tuple(int(v) for v in case['box'])
        raw = O._slimsam_binary(img, box).astype(bool)
        clipped = np.zeros_like(raw)
        clipped[box[1]:box[3], box[0]:box[2]] = raw[box[1]:box[3], box[0]:box[2]]
        lost = int(raw.sum() - clipped.sum())
        frac = lost / max(int(raw.sum()), 1)
        rows[case['id']] = {'mask_px': int(raw.sum()), 'clipped_away_px': lost,
                            'fraction_of_mask': frac}
        print(f"{case['id']:18} mask {int(raw.sum()):8d} px   clipped away {lost:7d} px "
              f"({frac:5.1%})")

        ys, xs = np.nonzero(raw)
        W, H = img.size
        x0 = min(box[0], int(xs.min())) if xs.size else box[0]
        y0 = min(box[1], int(ys.min())) if ys.size else box[1]
        x1 = max(box[2], int(xs.max()) + 1) if xs.size else box[2]
        y1 = max(box[3], int(ys.max()) + 1) if ys.size else box[3]
        px, py = int((x1 - x0) * 0.08) + 8, int((y1 - y0) * 0.08) + 8
        crop = (max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py))

        gc = O.segment_object(img, box, backend='grabcut')[0] > 0.5
        panels = [('the image', img.crop(crop)),
                  (f'SlimSAM, unclipped (+{frac:.1%} outside the box)',
                   compare.overlay(img, raw, box).crop(crop)),
                  ('SlimSAM clipped to the box -- what ships',
                   compare.overlay(img, clipped, box).crop(crop)),
                  ('GrabCut -- what shipped before',
                   compare.overlay(img, gc, box).crop(crop))]
        compare.sheet(img, box, panels,
                      f"{case['id']} -- {case['what']}: the cost of clipping").save(
            os.path.join(out_dir, case['id'] + '.png'))

    json.dump(rows, open(os.path.join(out_dir, 'clipped_away.json'), 'w'), indent=1)
    edit_sheet(out_dir)
    print('sheets in', out_dir)


def edit_sheet(out_dir):
    """The two worst-clipped cases (astronaut, car) recoloured with and without clipping,
    since a percentage of a mask is not a visible defect but a straight edge across a
    shoulder is."""
    for case in C.load():
        if case['id'] not in ('astronaut_suit', 'car_red'):
            continue
        img = C.image(case)
        box = tuple(int(v) for v in case['box'])
        raw = O._slimsam_binary(img, box).astype(bool)
        W, H = img.size
        ys, xs = np.nonzero(raw)
        crop = (max(0, min(box[0], int(xs.min())) - 20),
                max(0, min(box[1], int(ys.min())) - 20),
                min(W, max(box[2], int(xs.max()) + 1) + 20),
                min(H, max(box[3], int(ys.max()) + 1) + 20))
        panels = [('the image', img.crop(crop))]
        for label, m in (('unclipped: the whole object recoloured', raw),
                         ('clipped to the box -- what ships', None)):
            if m is None:
                mask = O.segment_object(img, box, backend='slimsam')[0]
            else:
                from figsurgeon.photo import _feather
                mask = _feather(m.astype(np.uint8), 2)
            a = np.asarray(img.convert('RGB')).astype(float)
            lum = a.mean(axis=2) / 255.0
            target = np.array([30.0, 60.0, 200.0])
            painted = target[None, None, :] * np.clip(
                lum / (target.mean() / 255.0), 0.35, 1.9)[:, :, None]
            m3 = mask[:, :, None]
            out = Image.fromarray(np.clip(a * (1 - m3) + np.clip(painted, 0, 255) * m3,
                                          0, 255).astype(np.uint8))
            panels.append((label, out.crop(crop)))
        compare.sheet(img, box, panels,
                      f"{case['id']} -- recoloured with the clipped and unclipped mask").save(
            os.path.join(out_dir, case['id'] + '_edit.png'))


if __name__ == '__main__':
    run()
