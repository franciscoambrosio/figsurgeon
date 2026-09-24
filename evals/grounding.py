"""Naming a thing in the picture: does the phrase select what a person meant?

    /path/to/env/bin/python evals/grounding.py [outdir]

Needs the optional extra (`pip install "figsurgeon[grounding]"`); skips with a message
otherwise. Writes a before/after sheet per request, since coverage percentages don't say
whether the right region was chosen -- a grey blob through the middle of a flower macro
measured a perfectly ordinary 87%.

Two cases must refuse: a phrase covering the whole frame, and one for something absent.
"""
import os
import sys
import time

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals import real_corpus                                            # noqa: E402
from figsurgeon import grounding                                        # noqa: E402
from figsurgeon.locate import _font                                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKING = 1000

# (id, corpus image, request, what should happen)
CASES = [
    ('sky_rocket',   'rocket_sample',  'make the sky pink',        'edit'),
    ('sky_city',     'city_wide',      'make the sky pink',        'edit'),
    ('car',          'car_red',        'paint the car blue',       'edit'),
    ('woman',        'portrait_studio', 'keep the woman in colour', 'edit'),
    ('pizza',        'food_pizza',     'make the pizza green',     'edit'),
    ('shoes',        'product_white',  'keep the shoes in colour', 'edit'),
    # Must refuse: the blooms are the whole photograph, nothing to separate them from.
    ('whole_frame',  'flower_macro',   'make it black and white except the flowers',
     'refuse'),
    # Must refuse: there is no cat in a picture of a pizza.
    ('absent',       'food_pizza',     'keep the cat in colour',   'refuse'),
]


def _image(name):
    if name == 'rocket_sample':
        from skimage import data
        img = Image.fromarray(data.rocket())
    else:
        img = real_corpus.load(name)
    img.thumbnail((WORKING, WORKING), Image.LANCZOS)
    return img.convert('RGB')


def sheet(before, after, title, verdict, width=1400):
    panel = width // 2
    cell = (panel - 14, int(panel * 1.4))
    tb, ta = before.copy(), after.copy()
    for t in (tb, ta):
        t.thumbnail(cell, Image.LANCZOS)
    out = Image.new('RGB', (width, max(tb.size[1], ta.size[1]) + 66), (246, 246, 248))
    out.paste(tb, ((panel - tb.size[0]) // 2, 58))
    out.paste(ta, (panel + (panel - ta.size[0]) // 2, 58))
    d = ImageDraw.Draw(out)
    d.text((10, 8), title[:120], font=_font(19), fill=(10, 10, 10))
    d.text((10, 34), verdict[:150], font=_font(13),
           fill=(0, 120, 0) if verdict.startswith('OK') else (190, 0, 0))
    return out


def run(outdir):
    if not grounding.available():
        print('figsurgeon[grounding] is not installed; nothing to measure.')
        print('  pip install "figsurgeon[grounding]"')
        return False
    os.makedirs(outdir, exist_ok=True)
    from figsurgeon.photo_describe import apply_text
    problems = []
    for case_id, name, request, expect in CASES:
        img = _image(name)
        t0 = time.time()
        try:
            out = apply_text(img, request)
            refused = None
        except ValueError as e:
            out, refused = img, str(e)
        secs = time.time() - t0
        happened = 'refuse' if refused else 'edit'
        ok = happened == expect
        if not ok:
            problems.append(f'{case_id}: expected to {expect}, did {happened} '
                            f'{refused or ""}'[:150])
        verdict = ('OK: ' if ok else 'PROBLEM: ') + (refused or f'edited in {secs:.1f}s')
        sheet(img, out, f'{case_id} -- "{request}"', verdict).save(
            os.path.join(outdir, f'{case_id}.png'))
        print(f'{"OK " if ok else "!! "}{case_id:14} {secs:5.1f}s  {verdict[:88]}')
    print(f'\n{len(CASES) - len(problems)}/{len(CASES)} behaved; sheets in {outdir}')
    print('LOOK AT THEM. Coverage cannot tell you whether it chose the right region.')
    for p in problems:
        print('   ', p)
    return not problems


if __name__ == '__main__':
    ok = run(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'eval_out_grounding'))
    sys.exit(0 if ok else 1)
