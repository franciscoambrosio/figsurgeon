"""Does the plain-language path's verdict tell the truth, on real photographs?

Both front-ends a person types at (`photo_describe.apply_text`, `ImageSession.do`) are wired
to the same checks. This eval runs correct edits, which must not be reported as failures, and
broken edits (no-op, wrong direction, half-finished), which must not be reported as verified.

Real photographs surface failure modes teaching samples don't: whole-frame high-frequency
energy fails a visibly correct denoise on a detailed facade, and a pure-black corner can
verify a vignette that never ran (`0 / 1e-6` reads as "the corner darkened").

Sheets are written for every case the checks could not decide.

Run:  python evals/text_path.py [outdir]      (downloads the corpus on first use)
"""
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals import real_corpus                                            # noqa: E402
from evals.real_world import sheet                                       # noqa: E402
from figsurgeon import photo as P                                       # noqa: E402
from figsurgeon.photo_describe import apply_text, parse                 # noqa: E402
from figsurgeon.workspace import verify_call                            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Everything a person plausibly types that reaches an operation with a check behind it.
CORRECT = [
    'brighten it', 'make it darker', 'increase the contrast', 'reduce the contrast',
    'make the colours more vivid', 'desaturate it a bit', 'make it pop',
    'make it black and white', 'add a vignette', 'colour pop the red',
    'change red to blue', 'remove the noise',
]

# The same instructions, with a result that did not do what was asked. Each is a real
# failure shape: the tool silently no-opped, it went the wrong way, or it half-finished.
BROKEN = [
    ('brighten it', lambda i: i, 'nothing happened'),
    ('brighten it', lambda i: P.adjust(i, brightness=0.8), 'went the wrong way'),
    ('make it black and white', lambda i: i, 'nothing happened'),
    ('make it black and white', lambda i: P.adjust(i, saturation=0.6), 'only half done'),
    ('increase the contrast', lambda i: P.adjust(i, contrast=0.7), 'went the wrong way'),
    ('add a vignette', lambda i: i, 'nothing happened'),
    ('remove the noise', lambda i: i, 'nothing happened'),
]

# Working size. The verdicts are the subject here, not the timings (real_world.py covers
# cost at native resolution), and 1400 px keeps a twelve-image sweep to a couple of minutes.
WORKING = 1400


def run(outdir):
    os.makedirs(outdir, exist_ok=True)
    names = sorted(real_corpus.CORPUS)
    verified = false_failures = undecided = 0
    caught = missed = not_decided = 0
    problems = []

    for name in names:
        img = real_corpus.load(name)
        img.thumbnail((WORKING, WORKING), Image.LANCZOS)

        for text in CORRECT:
            out = apply_text(img, text)
            ok, detail = apply_text.last_check
            if ok is True:
                verified += 1
            elif ok is False:
                false_failures += 1
                problems.append(f'FALSE FAILURE {name}/{text}: {detail}')
                sheet(img, out, f'{name} -- {text!r}', f'FALSE FAILURE: {detail}',
                      'a correct edit reported as a failure').save(
                          os.path.join(outdir, f'{name}__{text.replace(" ", "_")}__fail.png'))
            else:
                undecided += 1
                sheet(img, out, f'{name} -- {text!r}', f'OK (no verdict): {detail}',
                      'nothing measurable here -- look at the picture').save(
                          os.path.join(outdir, f'{name}__{text.replace(" ", "_")}__open.png'))

        for text, break_it, why in BROKEN:
            fn, kwargs = parse(text)
            broken = break_it(img)
            ok, detail = verify_call(fn, kwargs, img, broken)
            if ok is False:
                caught += 1
            elif ok is True:
                missed += 1
                problems.append(f'MISSED {name}/{text} ({why}): {detail}')
                sheet(img, broken, f'{name} -- {text!r} ({why})',
                      f'MISSED: verified an edit that {why}', detail).save(
                          os.path.join(outdir, f'{name}__{text.replace(" ", "_")}__missed.png'))
            else:
                not_decided += 1

        print(f'{name:22s} {len(CORRECT)} correct, {len(BROKEN)} broken -- '
              f'{false_failures} false failures, {missed} missed so far')

    print(f'\ncorrect edits : {verified} verified, {false_failures} FALSE FAILURES, '
          f'{undecided} no verdict')
    print(f'broken edits  : {caught} caught, {missed} MISSED, {not_decided} no verdict')
    for p in problems:
        print('   ', p)
    print(f'\nsheets for every undecided or wrong verdict in {outdir}')
    return not problems


if __name__ == '__main__':
    ok = run(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'eval_out_text'))
    sys.exit(0 if ok else 1)
