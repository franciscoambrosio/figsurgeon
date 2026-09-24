"""The eight boxes, and how to load the image each one belongs to.

Shared by both halves of the comparison (the SAM run, in a separate torch environment,
and the GrabCut run) so they cannot disagree about which pixels a box refers to.
"""
import json
import os
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

# Only bounds GrabCut's cost and the sheets' size; SAM masks are resolution-independent.
WORKING = 1400


def load():
    return json.load(open(os.path.join(HERE, 'cases.json')))


def image(case):
    if case['source'] == 'skimage':
        from skimage import data
        return Image.fromarray(getattr(data, case['name'])())
    from evals import real_corpus
    img = real_corpus.load(case['name'])
    img.thumbnail((WORKING, WORKING), Image.LANCZOS)
    return img
