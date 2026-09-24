"""Regression tests for the looking tools, locked to the failures that motivated them.

The value of this layer is not that it draws pictures; it is that a wrong box gets caught
before an irreversible edit. So these test the properties that make that true: previews
never touch the working image, refinement reports honestly when it cannot help, and the
mask overlay stays legible on a subject whose own colour matches the overlay.
"""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from figsurgeon import locate
from figsurgeon.tools import dispatch


@pytest.fixture(scope='module')
def cat():
    from skimage import data
    return Image.fromarray(data.chelsea())


@pytest.fixture(scope='module')
def disc():
    """A bright disc on a dark field: an object segmentation should find easily."""
    img = Image.new('RGB', (300, 220), (30, 40, 60))
    ImageDraw.Draw(img).ellipse([100, 60, 200, 160], fill=(220, 200, 90))
    return img


def test_grid_labels_stay_inside_the_canvas(cat):
    """A label clipped at the edge would render half a number -- a wrong coordinate that
    reads as real."""
    out = locate.grid_overlay(cat, step=100)
    assert out.size == cat.size
    assert np.array(out).shape == np.array(cat).shape


def test_grid_step_is_a_round_number(cat):
    for extent, expected_set in [(451, {50, 100}), (1024, {100, 200, 250}),
                                 (120, {10, 20, 25})]:
        step = locate._nice_step(extent)
        assert step in expected_set, f'{extent} -> {step}, not a round step'


def test_looking_tools_never_change_the_working_image(cat):
    """Looking tools must never return their annotation as the working image, or asking to
    see a grid would permanently replace the picture with one."""
    calls = [
        ('show_grid', {}),
        ('preview_region', {'boxes': [[10, 10, 100, 100]]}),
        ('refine_box', {'box': [141, 85, 211, 145]}),
        ('zoom', {'box': [141, 85, 211, 145]}),
        ('preview_colour_mask', {'target_rgb': [190, 180, 90]}),
        ('preview_object_mask', {'box': [141, 85, 211, 145]}),
    ]
    before = np.array(cat)
    for name, args in calls:
        result = dispatch(name, args, cat)
        assert result.mutates is False, f'{name} reported itself as an edit'
        assert result.image is cat, f'{name} returned a different working image'
        assert result.preview is not None, f'{name} produced nothing to look at'
        assert np.array_equal(np.array(cat), before), f'{name} mutated the source image'


def test_refine_box_snaps_onto_a_clear_object(disc):
    """A loose box around an obvious object must tighten onto it."""
    loose = (70, 30, 240, 200)
    refined, info = locate.refine_box(disc, loose)
    assert info['accepted'], info['reason']
    truth = (100, 60, 201, 161)
    for got, want in zip(refined, truth):
        assert abs(got - want) <= 6, f'refined {refined}, expected about {truth}'


def test_refine_box_refuses_rather_than_guessing(disc):
    """A box on featureless background has no object to snap to; refinement must decline
    rather than return a confidently wrong tighter box."""
    flat = (10, 10, 80, 80)
    refined, info = locate.refine_box(disc, flat)
    assert info['accepted'] is False
    assert tuple(refined) == flat, 'refused refinement must return the box unchanged'
    assert 'reason' in info and info['reason']


def test_mask_overlay_is_legible_on_a_same_coloured_subject():
    """A tint matching the subject's own colour would hide exactly the mistake the preview
    exists to reveal, so selected pixels must stay distinguishable regardless."""
    img = Image.new('RGB', (100, 100), (220, 30, 30))
    mask = np.zeros((100, 100))
    mask[:, :50] = 1.0
    out = np.array(locate.mask_overlay(img, mask)).astype(int)
    selected, unselected = out[:, :20].mean(), out[:, 70:].mean()
    assert selected - unselected > 40, (
        f'selected ({selected:.0f}) and unselected ({unselected:.0f}) regions of a red '
        f'subject are not visually distinguishable')


def test_mask_overlay_rejects_a_mismatched_mask(cat):
    with pytest.raises(ValueError):
        locate.mask_overlay(cat, np.zeros((10, 10)))


def test_describe_box_flags_a_clipped_box(cat):
    info = locate.describe_box(cat, (400, 200, 900, 700))
    assert info['clipped'] is True
    assert info['width'] == cat.size[0] - 400

    outside = locate.describe_box(cat, (900, 900, 1000, 1000))
    assert 'error' in outside


def test_zoom_enlarges_a_small_region(cat):
    crop, used = locate.zoom(cat, (200, 100, 240, 130))
    assert max(crop.size) > 300, 'a 40x30 region must come back readable, not 40x30'
    assert used[0] <= 200 and used[2] >= 240, 'the requested region must be inside the crop'
