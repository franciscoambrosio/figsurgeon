"""The check must select pixels the way the operation did, or it grades a different edit.

    .venv/bin/python evals/space_aware_checks.py [out_dir]

`check_isolate_colour`/`check_replace_colour` defined "the target colour" as a hue window
regardless of which space the operation used. An OKLab selection excludes pixels that share
the hue but are perceptually far (grey pavement under a yellow cast); a hue-window check
then measures exactly those pixels and finds them greyed, so a worse HSV isolation that
leaks into the pavement and trees can pass while a cleaner OKLab one fails.

Measures both checks against both spaces on real photographs; results go to `out_dir` and
are labelled by eye.
"""
import os
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from figsurgeon import verify_photo as V                                # noqa: E402
from figsurgeon.workspace import ImageWorkspace                         # noqa: E402

SHEETS = os.path.join(HERE, 'out_space_aware')

# (case, image, target rgb sampled off the thing, what it is, a colour to replace it with)
CASES = [
    ('taxi', 'evals/_corpus/agent_loop/taxi.jpg', (251, 181, 4), 'the taxi yellow', (30, 90, 200)),
    ('cones', 'evals/_corpus/agent_loop/roadworks.jpg', (142, 68, 38), 'the cone orange', (30, 90, 200)),
    ('car', 'evals/_corpus/car_red.jpg', (139, 24, 24), 'the red car', (20, 90, 190)),
    ('shoe', 'evals/_corpus/product_shoe.jpg', (139, 10, 5), 'the red shoes', (20, 90, 190)),
    ('flower', 'evals/_corpus/flower_macro.jpg', (214, 108, 24), 'the orange petals', (30, 90, 200)),
]
WORKING = 1200

# Judged by looking at the sheets (2026-09-15). The key is (case, operation) and the value
# names the space whose picture a person would keep, with what the other one does wrong.
LOOKED_AT = {
    ('taxi', 'isolate_colour'): ('oklab', 'hsv also keeps the pavement, the street trees and a facade'),
    ('cones', 'isolate_colour'): ('oklab', 'much the same; hsv leaves a brown cast on the forecourt'),
    ('car', 'isolate_colour'): ('oklab', 'hsv keeps the whole wooden shopfront as well as the car'),
    ('shoe', 'isolate_colour'): ('oklab', 'hsv keeps the warm tone of the paper the shoes stand on'),
    ('flower', 'isolate_colour'): ('hsv', 'the one the other way round: oklab washes the petals out'),
    ('taxi', 'replace_colour'): ('hsv', 'oklab leaves half the taxis orange'),
    ('cones', 'replace_colour'): ('hsv', 'oklab turns some cones blue and leaves others orange'),
    ('car', 'replace_colour'): ('hsv', 'oklab barely touches the car'),
    ('shoe', 'replace_colour'): ('oklab', 'both turn the shoes blue, but hsv blues the paper '
                                          'they stand on as well'),
    ('flower', 'replace_colour'): ('either', 'both replace the petals; neither is cleaner'),
}


def image(rel):
    img = Image.open(os.path.join(ROOT, rel)).convert('RGB')
    img.thumbnail((WORKING, WORKING), Image.LANCZOS)
    return img


def run(out_dir=SHEETS):
    os.makedirs(out_dir, exist_ok=True)
    print(f'{"case":8s} {"op":15s} {"space":6s} {"selected":>8s}  verdict')
    rows = []
    for case, rel, rgb, what, to_rgb in CASES:
        img = image(rel)
        for op, args_for in (('isolate_colour', lambda sp: (
                {'target_rgb': list(rgb), 'space': sp}
                | ({'oklab_tolerance': 0.06} if sp == 'oklab' else {'hue_tolerance': 0.07}))),
                ('replace_colour', lambda sp: (
                    {'from_rgb': list(rgb), 'to_rgb': list(to_rgb), 'space': sp}
                    | ({'oklab_tolerance': 0.06} if sp == 'oklab' else {'hue_tolerance': 0.07})))):
            for space in ('hsv', 'oklab'):
                ws = ImageWorkspace(img)
                out = ws.apply(op, args_for(space))
                note = out[1]
                ok = (ws.history[-1].ok if ws.history else None)
                selected = note.split('isolated ')[1].split(' ')[0] if 'isolated ' in note else (
                    note.split('recoloured ')[1].split(' ')[0] if 'recoloured ' in note else '?')
                ws.image.save(os.path.join(out_dir, f'{case}_{op}_{space}.jpg'), quality=85)
                verdict = {True: 'PASS', False: 'FAIL', None: 'none'}[ok]
                detail = note.split('verified: ')[-1].split('CHECK FAILED: ')[-1][:90]
                print(f'{case:8s} {op:15s} {space:6s} {selected:>8s}  {verdict:4s} {detail}')
                rows.append({'case': case, 'op': op, 'space': space, 'verdict': verdict,
                             'selected': selected, 'note': note})
        rows += constructed(img, rgb)
        rows += constructed_replace(img, rgb, to_rgb)
    summarise(rows)
    print(f'\nsheets in {out_dir} -- the picture is the ground truth here, not the number.')
    return rows


def constructed_replace(img, from_rgb, to_rgb):
    """Does the replace check still say anything once its region is the operation's own?

    If the checked region is exactly what the operation recoloured, moving those pixels is
    not evidence. `noop` (operation didn't run) and `half` (recolours half the selection)
    must both FAIL in both spaces.
    """
    import numpy as np
    from figsurgeon.perceptual import colour_mask
    out = []
    a = np.asarray(img.convert('RGB')).astype(float)
    m = colour_mask(img.convert('RGB'), tuple(int(c) for c in from_rgb), tolerance=0.06)
    half = m.copy()
    ys, xs = np.nonzero(m > 0.5)
    if len(xs):
        half[:, (xs.min() + xs.max()) // 2:] = 0.0
    hit = np.clip(np.array(to_rgb, float)[None, None, :] * (a.mean(axis=2, keepdims=True) / 128.0), 0, 255)
    edits = {'replace_noop': img,
             'replace_half': Image.fromarray(
                 (a * (1 - half[:, :, None]) + hit * half[:, :, None]).astype('uint8'))}
    for label, after in edits.items():
        for space in ('hsv', 'oklab'):
            kw = ({'oklab_tol': 0.06, 'space': 'oklab'} if space == 'oklab'
                  else {'hue_tol': 0.07, 'space': 'hsv'})
            ok, note = V.check_replace_colour(img, after, from_rgb, to_rgb, **kw)
            out.append({'case': label, 'op': 'must FAIL', 'space': space,
                        'verdict': {True: 'PASS', False: 'FAIL', None: 'none'}[ok],
                        'selected': '-', 'note': note})
    return out


def constructed(img, rgb):
    """Two edits that must FAIL in either space: `nothing_greyed` (operation didn't run) and
    `all_greyed` (greys the target too). Both must stay caught after the region is chosen
    perceptually.
    """
    out = []
    grey = img.convert('L').convert('RGB')
    for label, after in (('nothing_greyed', img), ('all_greyed', grey)):
        for space in ('hsv', 'oklab'):
            kw = ({'oklab_tol': 0.06, 'space': 'oklab'} if space == 'oklab'
                  else {'hue_tol': 0.07, 'space': 'hsv'})
            ok, note = V.check_isolate_colour(img, after, rgb, **kw)
            out.append({'case': label, 'op': 'must FAIL', 'space': space,
                        'verdict': {True: 'PASS', False: 'FAIL', None: 'none'}[ok],
                        'selected': '-', 'note': note})
    return out


def summarise(rows):
    if not LOOKED_AT:
        print('\nno labels yet: look at the sheets and fill LOOKED_AT before reading these')
        return
    print(f'\n{"case":8s} {"op":15s} {"better by eye":14s} {"hsv":6s} {"oklab":6s}  agreement')
    wrong = 0
    for case, _rel, _rgb, _what, _to in CASES:
        for op in ('isolate_colour', 'replace_colour'):
            got = {r['space']: r['verdict'] for r in rows if r['case'] == case and r['op'] == op}
            better = LOOKED_AT.get((case, op), ('?', ''))[0]
            agree = 'ok'
            if better in got and got[better] == 'FAIL':
                agree = 'FAILS THE BETTER PICTURE'
                wrong += 1
            print(f'{case:8s} {op:15s} {better:14s} {got.get("hsv", "-"):6s} '
                  f'{got.get("oklab", "-"):6s}  {agree}')
    print(f'\n{wrong} verdicts fail the picture a person would keep')
    must = [r for r in rows if r['op'] == 'must FAIL']
    for kind in ('nothing_greyed', 'all_greyed', 'replace_noop', 'replace_half'):
        rs = [r for r in must if r['case'] == kind]
        for space in ('hsv', 'oklab'):
            v = [r['verdict'] for r in rs if r['space'] == space]
            print(f'  {kind:15s} {space:6s} {v.count("FAIL")}/{len(v)} failed, '
                  f'{v.count("PASS")} wrongly passed, {v.count("none")} abstained')
    passed = [r for r in must if r['verdict'] == 'PASS']
    print(f'constructed failures: {len(must)} run, {len(passed)} wrongly passed, '
          f'{sum(1 for r in must if r["verdict"] == "FAIL")} failed, '
          f'{sum(1 for r in must if r["verdict"] == "none")} abstained')


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else SHEETS)
