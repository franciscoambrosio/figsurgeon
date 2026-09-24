"""Regression tests for `scale_object`: resizing a single object in place, not the whole frame.

Uses synthetic figures (flat background + flat fill), which `choose_backend` sends to
GrabCut deterministically, so shrink/grow/region behaviour does not depend on segmentation
quality. The empty-mask case reuses the known-empty rocket setup from
`test_input_robustness.py`.
"""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from figsurgeon import ImageWorkspace, objects
from figsurgeon.tools import dispatch


@pytest.fixture
def disc():
    """A 100x100 flat-colour circle on a flat background, comfortably inside `BOX`."""
    img = Image.new('RGB', (320, 240), (40, 40, 40))
    ImageDraw.Draw(img).ellipse([110, 70, 210, 170], fill=(200, 200, 60))
    return img


BOX = (90, 50, 230, 190)   # margin around the circle, for GrabCut's background model
FILL = (200, 200, 60)


def _fill_area(img, tol=10):
    """Pixels close to the circle's own flat fill colour: a trustworthy ground truth that
    avoids re-running GrabCut (unstable on the near-degenerate shrunk box)."""
    a = np.array(img.convert('RGB')).astype(int)
    d = np.abs(a - np.array(FILL)).max(axis=2)
    return float((d < tol).sum())


def test_shrink_reduces_the_objects_own_area(disc):
    out, frame_frac, warning = objects.scale_object(disc, BOX, scale=0.5, backend='grabcut')
    assert frame_frac > 0.05, f'segmentation found nothing to scale ({frame_frac:.1%})'
    before_area = _fill_area(disc)
    after_area = _fill_area(out)
    ratio = after_area / before_area
    # scale**2 = 0.25 by area.
    assert ratio < 0.4, f'shrunk area is {ratio:.2f}x the original, expected roughly 0.25x'
    assert not np.array_equal(np.array(disc), np.array(out))


def test_shrink_fills_the_vacated_ring_rather_than_leaving_a_hole(disc):
    """A pixel now outside the shrunk circle should read close to the background colour
    (inpainted), not stay the circle's own colour as a hole."""
    out, _, _ = objects.scale_object(disc, BOX, scale=0.5, backend='grabcut')
    a = np.array(out)
    # (200, 120) is on the circle's rim in the original (edge of [110,70,210,170]) and well
    # outside a circle scaled to half its radius about the same centre.
    edge_px = a[120, 195].astype(int)
    bg = np.array([40, 40, 40])
    assert np.abs(edge_px - bg).max() < 40, f'vacated ring pixel is {tuple(edge_px)}, expected near background {tuple(bg)}'


def test_grow_increases_the_objects_own_area_and_its_region(disc):
    out, frame_frac, warning = objects.scale_object(disc, BOX, scale=1.5, backend='grabcut')
    assert frame_frac > 0.05
    before_area = _fill_area(disc)
    after_area = _fill_area(out)
    ratio = after_area / before_area
    assert ratio > 1.3, f'grown area is only {ratio:.2f}x the original, expected roughly 2.25x'

    res = objects.scale_object(disc, BOX, scale=1.5, backend='grabcut')
    rx0, ry0, rx1, ry1 = res.region
    x0, y0, x1, y1 = BOX
    # Growing must reach past the caller's box on every side, or downstream region
    # enforcement would clip the grown object.
    assert rx0 <= x0 and ry0 <= y0 and rx1 >= x1 and ry1 >= y1, (res.region, BOX)
    assert (rx1 - rx0) > (x1 - x0), 'grown region is no bigger than the caller box'


def test_empty_mask_warns_and_does_not_enter_history(monkeypatch):
    """Same trap `erase_object` guards against: an empty mask must not report success."""
    monkeypatch.setattr(objects, 'available', lambda: False)
    from skimage import data
    rocket = Image.fromarray(data.rocket())
    empty_box = [415, 0, 470, 200]
    w = ImageWorkspace(rocket)
    result = w.apply('scale_object', {'box': empty_box, 'scale': 0.5})
    assert 'WARNING: nothing was scaled' in result.note, result.note[:120]
    assert np.array_equal(np.array(w.image), np.array(rocket))
    assert w.history == []


def test_a_working_scale_is_recorded_with_an_inconclusive_verdict(disc):
    """`check_object_scaled` never returns True/False for a real scale, so the note must
    carry the measured ratio instead."""
    w = ImageWorkspace(disc)
    result = w.apply('scale_object', {'box': list(BOX), 'scale': 0.5})
    assert 'WARNING: nothing was scaled' not in result.note
    assert len(w.history) == 1
    assert 'object area' in result.note and 'by area' in result.note


def test_region_containment_nothing_leaks_outside_the_reported_region(disc):
    out, frame_frac, warning = objects.scale_object(disc, BOX, scale=1.5, backend='grabcut')
    rx0, ry0, rx1, ry1 = objects.scale_object(disc, BOX, scale=1.5, backend='grabcut').region
    before, after = np.array(disc), np.array(out)
    d = np.abs(before.astype(int) - after.astype(int)).max(axis=2)
    d[ry0:ry1, rx0:rx1] = 0
    assert not (d > 8).any(), f'{int((d > 8).sum())} px changed outside the reported region'


def test_scale_object_via_dispatch_matches_the_tool_schema(disc):
    result = dispatch('scale_object', {'box': list(BOX), 'scale': 0.5}, disc)
    assert 'object scaled by 0.5x' in result.note
    assert not np.array_equal(np.array(disc), np.array(result.image))
