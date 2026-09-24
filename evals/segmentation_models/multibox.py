"""Encode once, prompt many: what a second box on the same photograph actually costs.

    python evals/segmentation_models/multibox.py        # needs figsurgeon[grounding]

A SAM call's cost is the image encoder, which does not depend on the prompt, so extra
boxes (`recolour_object(boxes=[...])`) should be nearly free. Measures the whole
`segment_objects` call one box at a time vs. all together, the model's own share alone,
and how far the batched mask moves from the one-at-a-time mask per box -- the saving is
only worth having if the mask does not change.
"""
import os
import sys
import time

import numpy as np
from PIL import Image
from skimage import data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evals import real_corpus                                            # noqa: E402
from figsurgeon import objects as O                                     # noqa: E402

STREET = [(455, 415, 600, 840), (590, 460, 725, 850), (775, 410, 925, 920), (520, 60, 720, 350)]
ASTRONAUT = [(60, 180, 420, 512), (300, 380, 420, 512)]


def street():
    img = real_corpus.load('street_people')
    img.thumbnail((1400, 1400), Image.LANCZOS)
    return img


def run():
    img = street()
    O.segment_object(img, (0, 0, 8, 8), backend='slimsam')     # warm the load, don't time it

    print('segment_objects on the street photograph, 1400 px:')
    for n in (1, 2, 4):
        one = time_per_box(img, STREET[:n])
        t = time.time()
        O.segment_objects(img, STREET[:n], backend='slimsam')
        together = time.time() - t
        print(f'  {n} box(es):  one at a time {one:5.2f} s   together {together:5.2f} s')

    print('\nthe model alone (_slimsam_binaries), same photograph:')
    for n in (1, 2, 4):
        t = time.time()
        O._slimsam_binaries(img, STREET[:n])
        print(f'  {n} box(es):  {time.time() - t:5.2f} s')

    print('\nhow far the batched mask moves from the one-at-a-time mask:')
    for name, im, boxes in (('street_people', img, STREET),
                            ('astronaut', Image.fromarray(data.astronaut()), ASTRONAUT)):
        singles = [O._slimsam_binary(im, b) for b in boxes]
        batched = O._slimsam_binaries(im, boxes)
        for i, (a, b) in enumerate(zip(singles, batched)):
            d = int((a != b).sum())
            print(f'  {name} box {i}: {d} px of {a.size} differ ({d / a.size:.6%})')


def time_per_box(img, boxes):
    """The cost of a full call per box, then the union -- the baseline this compares against."""
    t = time.time()
    masks = [O.segment_object(img, b, backend='slimsam')[0] for b in boxes]
    np.maximum.reduce(masks)
    return time.time() - t


if __name__ == '__main__':
    run()
