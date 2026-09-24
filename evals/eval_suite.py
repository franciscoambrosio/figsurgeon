"""Automated edit-quality eval: run prompts across images, verify each output measurably.

Run: python evals/eval_suite.py    (requires: pip install "figsurgeon[demo,advanced]")
"""
import os
import sys

import numpy as np
from PIL import Image
from skimage import data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figsurgeon.photo_describe import apply_text
from figsurgeon import verify_photo as V


def get(name):
    a = getattr(data, name)()
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[2] == 4:
        a = a[:, :, :3]
    return Image.fromarray(a.astype('uint8'))


# (image, prompt, checker) -- checker receives (before, after)
CASES = [
    ('astronaut', 'make everything black and white except the red',
     lambda b, a: V.check_isolate_colour(b, a, (220, 30, 30))),
    ('coffee', 'colour pop the red',
     lambda b, a: V.check_isolate_colour(b, a, (220, 30, 30))),
    ('immunohistochemistry', 'colour pop the blue',
     lambda b, a: V.check_isolate_colour(b, a, (40, 70, 200))),
    ('astronaut', 'blur the background', V.check_background_blur),
    ('chelsea', 'blur the background', V.check_background_blur),
    ('astronaut', 'remove the background', V.check_background_removed),
    ('chelsea', 'remove the background', V.check_background_removed),
    ('astronaut', 'make it black and white', V.check_grayscale),
    ('coins', 'make it black and white', V.check_grayscale),
    ('hubble_deep_field', 'brighten it', lambda b, a: V.check_brightness(b, a, 'up')),
    ('astronaut', 'make it darker', lambda b, a: V.check_brightness(b, a, 'down')),
    ('retina', 'increase the contrast', lambda b, a: V.check_contrast(b, a, 'up')),
    ('coffee', 'reduce the contrast', lambda b, a: V.check_contrast(b, a, 'down')),
    ('chelsea', 'make the colours more vivid', lambda b, a: V.check_saturation(b, a, 'up')),
    ('coffee', 'desaturate it a bit', lambda b, a: V.check_saturation(b, a, 'down')),
    ('clock', 'add a strong vignette', V.check_vignette),
    ('rocket', 'add a subtle vignette', V.check_vignette),
    ('astronaut', 'replace the red with orange',
     lambda b, a: V.check_replace_colour(b, a, (220, 30, 30), (255, 140, 0))),
]


def run(verbose=True):
    passed = failed = skipped = 0
    problems = []
    for name, prompt, checker in CASES:
        before = get(name)
        try:
            after = apply_text(before, prompt)
        except Exception as e:
            problems.append(f'ERROR  {name}/{prompt!r}: {e}')
            failed += 1
            continue
        ok, detail = checker(before, after)
        if ok is None:
            skipped += 1
            tag = 'SKIP'
        elif ok:
            passed += 1
            tag = 'PASS'
        else:
            failed += 1
            tag = 'FAIL'
            problems.append(f'{name}/{prompt!r}: {detail}')
        if verbose:
            print(f'  {tag}  {name:22s} {prompt:48s} {detail}')
    print(f'\n{passed} passed, {failed} failed, {skipped} skipped, of {len(CASES)}')
    if problems:
        print('PROBLEMS:')
        for p in problems:
            print('  ', p)
    return failed == 0


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
