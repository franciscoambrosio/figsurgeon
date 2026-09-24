"""Two ways to recolour one object, on the same box, judged side by side.

    .venv/bin/python evals/recolour_vs_generative.py [out_dir]

`recolour_object` multiplies a target colour by each pixel's own luminance in the segmented
mask; `generative_fill(mask_to_object=True)` sends the masked box to a hosted image model.
Both promise containment, but fail oppositely: the deterministic path leaves segmenter
misses untouched and flattens material outside its luminance clip; the generative path
redraws instead, so it has no leftovers but nothing ties it to the original object.

Four arms per case (shaded, flattened control, generative repaint under investigation,
generative replace control) are scored on colour accuracy, leftovers, structural similarity,
containment, and `check_object_recoloured`'s verdict. Sheets go to out_dir at 1:1 -- look at
them; no number alone says the picture is right.
"""
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from figsurgeon import objects, verify_photo                            # noqa: E402
from figsurgeon.verify_photo import texture_kept                        # noqa: E402
from figsurgeon.locate import _font                                     # noqa: E402
from figsurgeon.perceptual import delta_e, to_oklab                     # noqa: E402
from figsurgeon.tools import dispatch                                   # noqa: E402

sys.path.insert(0, os.path.join(HERE, 'segmentation_models'))
import cases as SHARED                                                   # noqa: E402

CORPUS = os.path.join(HERE, '_corpus')

# Real photographs, real boxes: every one is a box a model actually drew in
# `evals/agent_loop/` (mined from the transcripts), so this compares the two paths on the
# segmentation a driver really produced rather than on a box drawn to make a point.
CASES = [
    {'id': 'door', 'image': 'agent_loop/door.jpg', 'box': [635, 0, 1805, 1278],
     'subject': 'the front door', 'to_rgb': [30, 90, 40], 'words': 'dark green',
     'other': 'a plain panelled wooden door',
     'why': 'the deterministic path already gets this one right -- a control that must '
            'not regress'},
    {'id': 'two_dogs', 'image': 'agent_loop/two_dogs.jpg', 'box': [200, 90, 1290, 1014],
     'subject': 'the left rust-brown dog', 'to_rgb': [200, 148, 88], 'words': 'golden',
     'other': 'a black labrador',
     'why': 'dark fur asked for a light colour: the luminance clip flattens it'},
    {'id': 'two_cars', 'image': 'agent_loop/two_cars.jpg', 'box': [20, 540, 1000, 970],
     'subject': 'the red estate car in the foreground', 'to_rgb': [30, 80, 190],
     'words': 'blue', 'other': 'a white van',
     'why': 'the mask misses trim and bumper, and those pixels stay red'},
    {'id': 'chart_bar', 'image': 'agent_loop/chart_bar.png', 'box': [99, 243, 175, 601],
     'subject': 'the blue bar', 'to_rgb': [0, 64, 122], 'words': 'dark navy #00407A',
     'other': 'a green bar', 'why': 'a flat fill next to text and an axis, where the '
            'EXACT colour is the requirement -- the case the generative path should lose'},

    # `evals/segmentation_models/cases.json`: the eight hand-drawn boxes the segmenter
    # comparison and the completeness measures are all built on. Reused deliberately -- the
    # masks behind them have been looked at, so a surprise here is about the ARM, not the box.
    {'id': 'astronaut_suit', 'shared': 'astronaut_suit', 'to_rgb': [40, 80, 200],
     'words': 'blue', 'other': 'a black wetsuit',
     'why': 'white fabric with shadow and printed patches -- material the flat maths has '
            'no notion of'},
    {'id': 'coffee_cup', 'shared': 'coffee_cup', 'to_rgb': [30, 150, 60], 'words': 'green',
     'other': 'a glass tumbler',
     'why': 'a glazed cup: specular highlights are the whole look of it'},
    {'id': 'car_red', 'shared': 'car_red', 'to_rgb': [30, 80, 190], 'words': 'blue',
     'other': 'a yellow taxi',
     'why': 'a small, far object -- the same request as two_cars at a tenth the pixels'},
    {'id': 'flower_macro', 'shared': 'flower_macro', 'to_rgb': [140, 60, 180],
     'words': 'purple', 'other': 'a daisy centre',
     'why': 'dense fine structure: stamens are texture at the pixel level'},
    {'id': 'portrait_studio', 'shared': 'portrait_studio', 'to_rgb': [30, 150, 60],
     'words': 'green', 'other': 'a man in a suit',
     'why': 'a PERSON -- the case where a redrawn object is not a cosmetic difference'},
]

# Told to change only the colour. Deliberately explicit: if the model redraws the object
# anyway, that is the finding, not a prompt that invited it.
INSTRUCTION = ('repaint {subject} {words}. Keep its exact shape, position, shading, '
               'texture and every detail identical -- change only its colour. Do not '
               'change anything else in the region.')
# The bought must-FAIL: a genuine regeneration, same box, same mask, same money.
REPLACE = ('replace {subject} with {other}, in the same position and roughly the same '
           'size. It should look natural where it is.')

# Filled in by looking at every sheet at 1:1, not from the numbers:
#   same     the object that was there, recoloured
#   redrawn  recognisably a different object, pose or shape
#   damaged  the region is mangled, smeared or half-finished
JUDGED = {
    # All eight read as 'same' on inspection: told to change only the colour, the model
    # does, and the mask holds it in place -- including the two cases where the
    # deterministic path breaks.
    ('door', 'generative_repaint'): 'same',
    ('two_dogs', 'generative_repaint'): 'same',
    ('two_cars', 'generative_repaint'): 'same',
    ('astronaut_suit', 'generative_repaint'): 'same',
    ('coffee_cup', 'generative_repaint'): 'same',
    ('car_red', 'generative_repaint'): 'same',
    ('flower_macro', 'generative_repaint'): 'same',
    ('portrait_studio', 'generative_repaint'): 'same',
}


def load_case(case):
    """The image and the box, from whichever corpus the case names."""
    if 'shared' in case:
        shared = {c['id']: c for c in SHARED.load()}[case['shared']]
        img = SHARED.image(shared).convert('RGB')
        return img, [int(v) for v in shared['box']], shared['what'], shared['name']
    img = Image.open(os.path.join(CORPUS, case['image'])).convert('RGB')
    return img, [int(v) for v in case['box']], case['subject'], case['image']


def _mask(img, box):
    """The segmenter both paths use, run ONCE so neither is graded on its own mask."""
    m, cov = objects.segment_objects(img, [box])
    m = np.asarray(m, dtype=float)
    if m.max() > 1.5:
        m = m / 255.0
    return m, float(np.mean(cov))


def _lum(img):
    return np.asarray(img.convert('RGB'), dtype=float).mean(axis=2)


def structure_only(a, b, win=7):
    """SSIM's STRUCTURE term alone: local correlation, with luminance and contrast divided
    out.

    Full SSIM conflates l*c*s, but a recolour is entitled to change l and c -- that's what
    recolouring is. The structure term alone asks the right question: are the same edges
    and texture still in the same places?
    """
    C3 = (0.03 * 255) ** 2 / 2
    mu_a, mu_b = uniform_filter(a, win), uniform_filter(b, win)
    va = uniform_filter(a * a, win) - mu_a * mu_a
    vb = uniform_filter(b * b, win) - mu_b * mu_b
    vab = uniform_filter(a * b, win) - mu_a * mu_b
    sa, sb = np.sqrt(np.maximum(va, 0)), np.sqrt(np.maximum(vb, 0))
    return (vab + C3) / (sa * sb + C3)


def measure(before, after, m, target):
    """Colour, leftovers, structure and containment, all over the same mask."""
    sel = m > 0.5
    a = np.asarray(before.convert('RGB'), dtype=float)
    b = np.asarray(after.convert('RGB'), dtype=float)
    out = {}

    mean_after = b[sel].mean(axis=0)
    out['dE'] = float(delta_e(mean_after, np.array(target, dtype=float)))

    # "Still the old colour" is what a leftover IS: the mask's own original mean, and how
    # much of the mask still sits within 0.06 OKLab of it after the edit.
    orig_mean = a[sel].mean(axis=0)
    lab_after = to_oklab(b[sel])
    lab_orig = to_oklab(orig_mean[None, :])
    out['left'] = float((np.linalg.norm(lab_after - lab_orig, axis=-1) < 0.06).mean())

    ys, xs = np.nonzero(sel)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    la, lb = _lum(before)[y0:y1, x0:x1], _lum(after)[y0:y1, x0:x1]
    crop = sel[y0:y1, x0:x1]
    _, ssim_map = structural_similarity(la, lb, data_range=255.0, full=True)
    out['ssim'] = float(ssim_map[crop].mean())
    out['struct'] = float(structure_only(la, lb)[crop].mean())
    # `verify_photo.texture_kept` itself, not a copy: what the package now FAILS on.
    kept, _n = texture_kept(before, after, sel)
    out['kept'] = kept if kept is not None else float('nan')

    outside = (np.abs(a - b).max(axis=2) > 0) & ~(m > 0)
    out['outside'] = int(outside.sum())
    return out


def sheet(panels, title, width=1900):
    cell = width // len(panels) - 12
    thumbs = []
    for cap, p in panels:
        t = p.copy()
        t.thumbnail((cell, cell * 2), Image.LANCZOS)
        thumbs.append((cap, t))
    height = max(t.size[1] for _, t in thumbs) + 118
    out = Image.new('RGB', (width, height), (246, 246, 248))
    d = ImageDraw.Draw(out)
    d.text((10, 6), title, font=_font(20), fill=(10, 10, 10))
    for i, (cap, t) in enumerate(thumbs):
        x = i * (width // len(panels)) + 6
        out.paste(t, (x, 106))
        for j, line in enumerate(cap.split('\n')):
            d.text((x, 32 + 14 * j), line, font=_font(12), fill=(60, 60, 60))
    return out


def _arms(img, box, subject, case, paid):
    """The four arms, in the order they are cheapest: two free, then the paid pair."""
    out = []

    t0 = time.time()
    shaded = objects.recolour_object(img, box=box, to_rgb=case['to_rgb'])
    out.append(('recolour_shaded', shaded[0], round(time.time() - t0, 1), '', 'same'))

    t0 = time.time()
    flat = objects.recolour_object(img, box=box, to_rgb=case['to_rgb'],
                                   preserve_shading=False)
    out.append(('recolour_flat', flat[0], round(time.time() - t0, 1), '', 'flattened'))

    if not paid:
        return out
    for arm, text, truth in [
            ('generative_repaint',
             INSTRUCTION.format(subject=subject, words=case['words']), None),
            ('generative_replace',
             REPLACE.format(subject=subject, other=case['other']), 'replaced')]:
        t0 = time.time()
        got, note = dispatch('generative_fill',
                             {'box': box, 'mask_to_object': True, 'instruction': text}, img)
        secs = round(time.time() - t0, 1)
        if note.startswith(('generative_fill did not run', 'WARNING')):
            print(f'    {arm}: no picture -- {note[:160]}')
            continue
        out.append((arm, got, secs, note, truth))
    return out


def run(out_dir=None, paid=True, only=None):
    out_dir = out_dir or os.path.join(HERE, 'out_recolour_vs_generative')
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for case in CASES:
        if only and case['id'] not in only:
            continue
        img, box, subject, source = load_case(case)
        m, cov = _mask(img, box)
        print(f'\n=== {case["id"]} ({source} {img.size[0]}x{img.size[1]}): {case["why"]}')
        print(f'    mask covers {cov:.0%} of the box, {(m > 0.5).mean():.2%} of the frame')
        if not (m > 0.5).any():
            # Not a bug: the segmenter isn't trained to find a bar in a chart, so neither
            # arm has a mask to work on. The chart route is remap_colormap/replace_colour.
            print('    SKIPPED: the segmenter found no object in this box, so neither arm '
                  'has a mask to work on')
            continue

        panels = [(f'original\n{source}\nbox {tuple(box)}', img)]
        for arm, out, secs, note, truth in _arms(img, box, subject, case, paid):
            stat = measure(img, out, m, case['to_rgb'])
            ok, detail = verify_photo.check_object_recoloured(img, out, box,
                                                              subject=subject)
            label = truth or JUDGED.get((case['id'], arm), 'unjudged')
            rows.append((case['id'], arm, stat, ok, secs, label))
            print(f'    {arm:19s} dE {stat["dE"]:.3f}  leftover {stat["left"]:.0%}  '
                  f'texture kept {stat["kept"]:.2f}  ssim {stat["ssim"]:.2f}  '
                  f'struct {stat["struct"]:.2f}  outside {stat["outside"]}  {secs}s  '
                  f'verdict {ok}  [{label}]')
            print(f'        {detail[:220]}')
            if note:
                print(f'        note: {note[:220]}')
            panels.append((f'{arm}  {secs}s  [{label}]\ndE {stat["dE"]:.3f} to the asked '
                           f'colour, leftover {stat["left"]:.0%} of the mask\n'
                           f'texture kept {stat["kept"]:.2f} (0 = flattened), '
                           f'ssim {stat["ssim"]:.2f}, structure {stat["struct"]:.2f}\n'
                           f'outside {stat["outside"]} px  '
                           f'check_object_recoloured: {ok}', out))
            out.save(os.path.join(out_dir, f'{case["id"]}_{arm}.png'))

        sheet(panels, f'{case["id"]}: "{subject}" -> {case["words"]} -- {case["why"]}') \
            .save(os.path.join(out_dir, f'{case["id"]}_sheet.png'))

    print(f'\n{"case":16s} {"arm":19s} {"dE":>6s} {"left":>6s} {"kept":>6s} '
          f'{"ssim":>6s} {"struct":>7s} {"outside":>8s} {"secs":>6s} {"truth":9s} verdict')
    for cid, arm, st, ok, secs, label in rows:
        print(f'{cid:16s} {arm:19s} {st["dE"]:6.3f} {st["left"]:6.0%} {st["kept"]:6.2f} '
              f'{st["ssim"]:6.2f} {st["struct"]:7.2f} {st["outside"]:8d} {secs:6.1f} '
              f'{label:9s} {ok}')
    for key in ('kept', 'ssim', 'struct'):
        separation(rows, key)
    print(f'\nsheets: {out_dir} -- LOOK at them; none of the numbers above says the '
          f'picture is right')
    return rows


def separation(rows, key='kept'):
    """Each class's range, and which of them the measure can actually tell apart.

    Three classes, not two, because the two ways of not-being-the-object-any-more behave
    completely differently: `flattened` has no texture at all, `replaced` has plenty of its
    own. Pooling them into one "bad" class hides the only separation there is.
    """
    classes = {}
    for r in rows:
        classes.setdefault(r[5], []).append(r[2][key])
    same = classes.get('same', []) + classes.get('same*', [])
    if not same:
        return
    print(f'\n{key}, by class:')
    for name, vals in sorted(classes.items()):
        print(f'  {name:10s} {min(vals):.2f} - {max(vals):.2f}  (n={len(vals)})')
    for name, vals in sorted(classes.items()):
        if name == 'same' or not vals:
            continue
        gap = min(same) - max(vals)
        print(f'  vs same: {"SEPARATED" if gap > 0 else "OVERLAP":9s} {name} '
              + (f'tops out at {max(vals):.2f}, nothing correct reads below {min(same):.2f}'
                 if gap > 0 else
                 f'reaches {max(vals):.2f}, into the correct range ({min(same):.2f} up)'))


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    run(args[0] if args else None, paid='--free' not in sys.argv)
