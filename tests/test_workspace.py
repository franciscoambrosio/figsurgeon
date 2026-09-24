"""Regression tests for the verified, undoable editing loop.

Locks in the behaviours that make the verdict worth reading. A check that fails correct work
is worse than no check at all -- it teaches the caller to ignore the field -- so several of
these test that specific correct edits come back GREEN, not just that broken ones come back
red. Each corresponds to a false failure measured during the quality eval.
"""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from figsurgeon import ImageWorkspace
from figsurgeon.workspace import verify


@pytest.fixture(scope='module')
def cat():
    from skimage import data
    return Image.fromarray(data.chelsea())


@pytest.fixture(scope='module')
def disc():
    img = Image.new('RGB', (320, 240), (60, 130, 60))
    ImageDraw.Draw(img).ellipse([110, 70, 210, 170], fill=(95, 175, 95))
    return img


def test_undo_restores_the_exact_pixels(cat):
    w = ImageWorkspace(cat)
    w.apply('adjust', {'brightness': 3.0})
    assert not np.array_equal(np.array(w.image), np.array(cat))
    w.apply('undo', {})
    assert np.array_equal(np.array(w.image), np.array(cat)), 'undo did not restore exactly'
    assert w.history == []


def test_looking_tools_do_not_enter_the_undo_stack(cat):
    """An undo stack full of no-ops makes "undo" mean something different every time."""
    w = ImageWorkspace(cat)
    w.apply('show_grid', {})
    w.apply('preview_region', {'boxes': [[0, 0, 50, 50]]})
    assert w.history == []
    w.apply('adjust', {'brightness': 1.2})
    w.apply('zoom', {'box': [0, 0, 50, 50]})
    assert len(w.history) == 1
    w.apply('undo', {})
    assert np.array_equal(np.array(w.image), np.array(cat))


def test_a_failed_edit_is_reported_not_silently_applied():
    """Saturating a greyscale image cannot work. It must come back FAILED with the reason,
    which is the signal that tells a caller to try something else instead of a bigger number."""
    from skimage import data
    grey = Image.fromarray(np.stack([data.camera()] * 3, axis=-1))
    w = ImageWorkspace(grey)
    result = w.apply('adjust', {'saturation': 1.8})
    assert 'CHECK FAILED' in result.note
    assert len(w.failures()) == 1


def test_reset_returns_to_the_original(cat):
    w = ImageWorkspace(cat)
    w.apply('stylise', {'effect': 'grayscale'})
    w.apply('stylise', {'effect': 'vignette', 'strength': 0.5})
    w.apply('reset', {})
    assert np.array_equal(np.array(w.image), np.array(cat))
    assert w.history == []


def test_object_recolour_stays_inside_the_region_it_reports(cat):
    """The spatial tools' promise, in the form it takes now: an edit changes nothing outside
    the region the tool NAMED -- which is the caller's box, grown to wherever the object it
    found continues. Growing it is the point: clipping the mask at the box recoloured the
    astronaut's suit and left her sleeve orange, with a seam down the line the caller drew.

    PASS: nothing changes outside the reported region, and the note says where it reached.
    FAIL, and this is what makes the test worth having: the same edit is NOT contained by
    the box alone, so a test written against the box would pass vacuously if the object
    never left it."""
    box = [141, 85, 211, 145]
    w = ImageWorkspace(cat)
    result = w.apply('recolour_object', {'box': box, 'to_rgb': [30, 60, 200]})
    assert 'CHECK FAILED' not in result.note, result.note
    region = [int(v) for v in result.data['region']]
    assert (region[0] <= box[0] and region[1] <= box[1]
            and region[2] >= box[2] and region[3] >= box[3]), region

    a0, a1 = np.array(cat), np.array(w.image)
    d = (a0 != a1).any(axis=2)
    outside_region = d.copy()
    outside_region[region[1]:region[3], region[0]:region[2]] = False
    assert outside_region.sum() == 0, f'{outside_region.sum()} px changed outside {region}'

    if region != box:                      # the object did continue past the box
        assert 'continued past the box' in result.note, result.note


def test_follow_object_false_restores_the_hard_box(cat):
    """The old contract is still available and still exact: with the mask clipped, not one
    pixel outside the box may move -- the feathered mask used to spread outside it anyway
    (measured: 1567 px, up to 0.40 strength)."""
    from figsurgeon import objects
    box = (141, 85, 211, 145)
    out, _ = objects.recolour_object(cat, box, to_rgb=(30, 60, 200), follow_object=False)
    d = (np.array(cat) != np.array(out)).any(axis=2)
    d[box[1]:box[3], box[0]:box[2]] = False
    assert d.sum() == 0, f'{d.sum()} px changed outside the box'


def test_isolate_object_is_not_treated_as_a_localised_edit(disc):
    """Desaturating everything outside the box is `isolate_object`'s purpose. Checking it
    with the unchanged-outside rule reported a correct edit as a 131,100-pixel violation."""
    w = ImageWorkspace(disc)
    result = w.apply('isolate_object', {'box': [95, 55, 225, 185]})
    assert 'CHECK FAILED' not in result.note, result.note
    assert 'verified' in result.note


def test_isolate_colour_is_checked_at_the_tolerance_it_was_called_with():
    """A check run at a wider tolerance than the operation counts deliberately-excluded
    pixels as failures -- measured on the rocket photo as a false FAILED verdict on a
    visibly good edit."""
    from skimage import data
    rocket = Image.fromarray(data.rocket())
    w = ImageWorkspace(rocket)
    result = w.apply('isolate_colour', {'target_rgb': [230, 150, 60],
                                        'hue_tolerance': 0.06})
    assert 'CHECK FAILED' not in result.note, result.note


def test_colour_isolation_on_a_same_hue_image_is_inconclusive_not_green(disc):
    """When subject and background share a hue there is nothing to grey out, so the check
    has no evidence. It used to substitute zero and return a confident PASS."""
    ok, detail = verify('isolate_colour', {'target_rgb': [95, 175, 95]},
                        disc, disc)
    assert ok is None, f'expected inconclusive, got {ok}: {detail}'
    assert 'isolate_object' in detail, 'the verdict should name the tool that can do this'


def test_undo_on_an_empty_history_is_harmless(cat):
    w = ImageWorkspace(cat)
    result = w.apply('undo', {})
    assert 'nothing to undo' in result.note
    assert np.array_equal(np.array(w.image), np.array(cat))


def test_a_bad_argument_comes_back_as_a_note_not_an_exception(cat):
    w = ImageWorkspace(cat)
    result = w.apply('crop', {'box': [900, 900, 1000, 1000]})
    assert result.note.startswith('error')
    assert w.history == [], 'a failed call must not enter the undo stack'
    assert np.array_equal(np.array(w.image), np.array(cat))


def test_multiple_regions_in_one_isolate_call(cat):
    """"Colour pop the eyes" is two regions. Running the tool twice is not equivalent: the
    second pass desaturates everything outside ITS box, undoing the first."""
    boxes = [[141, 85, 211, 145], [288, 100, 352, 158]]
    w = ImageWorkspace(cat)
    result = w.apply('isolate_object', {'boxes': boxes, 'flatten': 0.6})
    assert 'CHECK FAILED' not in result.note, result.note

    def colour_left_in(img, box):
        a = np.array(img.convert('RGB')).astype(float)[box[1]:box[3], box[0]:box[2]]
        mx, mn = a.max(axis=2), a.min(axis=2)
        return float(((mx - mn) / np.maximum(mx, 1e-6)).mean())

    for box in boxes:
        assert colour_left_in(w.image, box) > 0.10, f'region {box} lost its colour'


def test_replace_background_is_checked_both_ways(cat):
    """Checking only the background colour would pass a flood-filled frame; checking only
    the subject would pass an unedited image. Both halves, or the check is decorative."""
    from figsurgeon.verify_photo import check_background_replaced
    w = ImageWorkspace(cat)
    result = w.apply('replace_background', {'colour_rgb': [255, 255, 255]})
    assert 'CHECK FAILED' not in result.note, result.note

    flooded = Image.new('RGB', cat.size, (255, 255, 255))
    ok, detail = check_background_replaced(cat, flooded, (255, 255, 255))
    assert ok is False, f'a frame flood-filled white must not pass: {detail}'

    ok, detail = check_background_replaced(cat, cat, (255, 255, 255))
    assert ok is False, f'an unedited image must not pass: {detail}'


def test_crop_catches_the_wrong_region(cat):
    """A box in the wrong coordinate convention produces a plausible picture of the wrong
    part of the image, which nothing else here would notice."""
    from figsurgeon.verify_photo import check_crop
    w = ImageWorkspace(cat)
    assert 'CHECK FAILED' not in w.apply('crop', {'box': [10, 20, 110, 140]}).note

    wrong = cat.crop((20, 10, 140, 110))          # x and y swapped
    ok, detail = check_crop(cat, wrong, (10, 20, 110, 140))
    assert ok is False, f'a swapped-axis crop must not pass: {detail}'


def test_a_colour_given_by_name_still_gets_a_verdict():
    """dispatch accepts "red" where the schema asks for [r, g, b] (tools._rgb), but verify
    ran tuple('red') -> ('r', 'e', 'd') and every such edit came back "check could not run
    (ufunc 'divide' not supported ...)" -- unverified, for a reason the caller never caused."""
    img = Image.new('RGB', (200, 120), (40, 90, 200))
    ImageDraw.Draw(img).rectangle([20, 20, 90, 100], fill=(210, 30, 30))
    ImageDraw.Draw(img).rectangle([120, 20, 180, 100], fill=(30, 160, 60))
    w = ImageWorkspace(img)
    note = w.apply('isolate_colour', {'target_rgb': 'red'}).note
    assert 'could not run' not in note, note
    assert w.history[-1].ok is not None, note
    note = w.apply('replace_colour', {'from_rgb': 'green', 'to_rgb': 'blue'}).note
    assert 'could not run' not in note, note


def test_a_point_prompt_that_finds_nothing_names_the_points():
    """The "nothing was found" note printed `inside ()` for a point prompt, because it only
    ever looked at the (empty) list of boxes."""
    from figsurgeon.tools import _empty_segmentation_note
    note = _empty_segmentation_note([0.0], [], 'recoloured', points=True, where=[(12, 30)])
    assert '()' not in note and '(12, 30)' in note, note
