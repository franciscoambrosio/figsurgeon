"""The drift measurement, against editors whose answer is known in advance.

Every case here exists because the measurement could get it wrong in a specific way: pass an
editor that repainted the wall, fail one that only added imperceptible noise, credit a
resize for drift that resampling caused, or -- the one that makes the whole comparison
worthless -- award a perfect drift score to an editor that did nothing at all.
"""
import os

import numpy as np
import pytest
from PIL import Image

from evals import background_drift as bd

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'evals', '_corpus', 'car_red.jpg')


def _photo():
    """A real photograph: synthetic content is too self-similar or too structureless for
    phase correlation, which then returns a confident wrong shift. Skips if the gitignored
    corpus (fetched by `evals/real_corpus.py`) is absent.
    """
    if not os.path.exists(CORPUS):
        pytest.skip('run `python evals/real_corpus.py` to fetch the corpus')
    img = Image.open(CORPUS).convert('RGB')
    img.thumbnail((256, 256), Image.LANCZOS)
    return img

BOX = (40, 40, 90, 90)


def _base():
    """A picture with structure and no periodicity, since phase correlation returns a
    confident wrong answer on periodic content."""
    y, x = np.mgrid[0:128, 0:128].astype(float)
    a = np.dstack([160 - x * 0.9 + y * 0.3, 40 + y * 1.2, 90 + (x + y) * 0.4])
    rng = np.random.RandomState(4)
    for cx, cy, r in rng.randint(10, 118, (7, 3)):
        blob = ((x - cx) ** 2 + (y - cy) ** 2) < max(6, r // 3) ** 2
        a[blob] = rng.randint(0, 255, 3)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _mask():
    m = np.zeros((128, 128), dtype=float)
    m[BOX[1]:BOX[3], BOX[0]:BOX[2]] = 1.0
    return m


def _painted(base, rgb=(128, 0, 128)):
    a = np.asarray(base).copy()
    a[BOX[1]:BOX[3], BOX[0]:BOX[2]] = rgb
    return Image.fromarray(a)


def test_a_localised_edit_drifts_nowhere():
    base = _base()
    d = bd.drift(base, _painted(base), _mask())
    assert d['share_any'] == 0 and d['max_channel'] == 0 and d['blobs'] == 0


def test_invisible_noise_is_reported_but_not_judged_visible():
    """An editor that shifts every pixel by 1 differs everywhere and looks identical."""
    base = _base()
    a = np.asarray(_painted(base)).astype(np.int16)
    a += np.random.RandomState(0).randint(-1, 2, a.shape)
    d = bd.drift(base, Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), _mask())
    assert d['share_any'] > 0.5, 'it did change almost every pixel'
    assert d['share_visible'] == 0, 'and none of it is visible'
    assert d['max_channel'] <= 1


def test_a_repainted_background_is_caught_as_one_blob():
    base = _base()
    a = np.asarray(_painted(base)).astype(np.int16)
    a[:30] = np.clip(a[:30] * 1.15, 0, 255)          # the sky, which nobody asked about
    d = bd.drift(base, Image.fromarray(a.astype(np.uint8)), _mask())
    assert d['share_visible'] > 0.1
    assert d['blobs'] >= 1 and d['largest_blob_px'] > 1000


def test_doing_nothing_scores_zero_drift_and_zero_intent():
    """The check that stops the comparison being rigged in figsurgeon's favour."""
    base = _base()
    d = bd.drift(base, base, _mask())
    i = bd.intent(base, base, _mask(), (128, 0, 128))
    assert d['share_any'] == 0, 'a no-op drifts nowhere'
    assert i['moved_share'] == 0, 'and achieves nothing, which must be visible too'


def test_intent_separates_a_full_edit_from_a_half_one():
    base = _base()
    full = bd.intent(base, _painted(base), _mask(), (128, 0, 128))
    a = np.asarray(_painted(base)).copy()
    a[BOX[1]:(BOX[1] + BOX[3]) // 2, BOX[0]:BOX[2]] = np.asarray(base)[
        BOX[1]:(BOX[1] + BOX[3]) // 2, BOX[0]:BOX[2]]          # put half of it back
    half = bd.intent(base, Image.fromarray(a), _mask(), (128, 0, 128))
    assert full['moved_share'] > 0.95
    assert half['moved_share'] < 0.75, 'a half-done edit must not read as done'


def test_a_resize_is_measured_against_what_resampling_alone_costs():
    """A model that returns a fixed size resamples the frame; that is not the editor's drift."""
    photo = _photo()
    W, H = photo.size
    mask = np.zeros((H, W), dtype=float)
    mask[H // 3:2 * H // 3, W // 3:2 * W // 3] = 1.0

    smaller = photo.resize((W // 2, H // 2), Image.LANCZOS)
    aligned, covered, notes = bd.align(photo, smaller)
    assert any(f'returned {W // 2}x{H // 2}' in n for n in notes)
    assert covered.mean() > 0.99, 'a same-shape return covers the whole frame'

    d = bd.drift(photo, aligned, mask, covered)
    floor = bd.resample_floor(photo, (W // 2, H // 2), mask)
    assert d['share_visible'] == pytest.approx(floor['share_visible'], abs=0.05), \
        'essentially all of it is the round trip, not an edit'


def test_a_return_that_does_not_cover_the_frame_leaves_those_pixels_unknown():
    """A model that hands back a slice of the scene has not changed what it omitted, so
    uncovered pixels must be dropped rather than counted as drift."""
    photo = _photo()
    W, H = photo.size
    mask = np.zeros((H, W), dtype=float)
    mask[H // 3:2 * H // 3, W // 3:2 * W // 3] = 1.0

    band = photo.crop((0, H // 6, W, H - H // 6))          # the middle of the scene only
    aligned, covered, notes = bd.align(photo, band)
    assert not covered.all(), 'the top and bottom of the frame are not covered'
    assert any('not covered by the return' in n for n in notes)

    d = bd.drift(photo, aligned, mask, covered)
    assert d['background_px'] < W * H, 'uncovered pixels are out of the denominator'
    assert d['share_visible'] < 0.05, 'and what IS covered is the same picture'
