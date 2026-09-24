"""Tests for `check_object_erased`, which grades an erase by asking whether the caller's
phrase still matches anything where the box was, resolved against that phrase's extent
before the edit -- unlike the other object checks, which grade against what changed.
"""
import numpy as np
import pytest
from PIL import Image
from skimage import data

from figsurgeon import grounding, objects, workspace
from figsurgeon.verify_photo import check_object_erased

SUIT_BOX = [60, 180, 420, 512]           # the astronaut's suit
SHUTTLE_BOX = [368, 0, 500, 195]         # the small shuttle model, top right of the frame


def astronaut():
    return Image.fromarray(data.astronaut())


def test_a_no_op_erase_fails():
    img = astronaut()
    ok, detail = check_object_erased(img, img.copy(), SUIT_BOX, subject='the suit')
    assert ok is False, detail
    assert 'nothing inside the box changed' in detail


needs_model = pytest.mark.skipif(not grounding.available(),
                                 reason='needs figsurgeon[grounding] (torch + transformers)')


def test_erase_without_subject_gets_no_verdict():
    """No words means no verdict on whether the named thing is actually gone."""
    img = astronaut()
    out, _frame_frac, _warning = objects.erase_object(img, box=SHUTTLE_BOX)
    ok, detail = check_object_erased(img, out, SHUTTLE_BOX)
    assert ok is None, detail
    assert 'WHAT was erased is not checked' in detail


@needs_model
@pytest.mark.parametrize('fill', ['telea', 'lama'])
def test_a_clean_erase_abstains_rather_than_passing(fill):
    """An erase of a small isolated object gets an abstention, not a PASS, since a smear can
    read the same as a real removal; correct work must never be called a failure."""
    img = astronaut()
    out, frame_frac, _warning = objects.erase_object(img, box=SHUTTLE_BOX, fill=fill)
    assert frame_frac < 0.08                # small object -- not the smeared-large-object case
    ok, detail = check_object_erased(img, out, SHUTTLE_BOX, subject='the rocket')
    assert ok is None, detail
    assert ok is not False, detail
    assert "still reads as 'the rocket'" in detail
    assert ('NOT a verdict that it is gone' in detail          # low band
            or 'too much left to call it gone' in detail), detail


@needs_model
def test_a_partial_erase_is_not_verified():
    """A box covering only part of an object (the spoon's handle, not its bowl) must fail
    even though containment alone would call it clean."""
    img = Image.fromarray(data.coffee())
    box = [330, 130, 460, 260]
    out, _frame_frac, _warning = objects.erase_object(img, box=box)
    ok, detail = check_object_erased(img, out, box, subject='the spoon')
    assert ok is False, detail
    assert 'the erase did not remove it' in detail


@needs_model
def test_a_wrong_box_with_the_object_untouched_fails():
    """A box drawn beside the object, half over changed background and half over the
    untouched object itself, must still fail."""
    img = astronaut()
    mask, _cov = objects.segment_object(img, SUIT_BOX)
    mask = np.asarray(mask, dtype=float)
    ys, xs = np.nonzero(mask > 0.5)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    crop = img.crop((x0, y0, x1 + 1, y1 + 1))
    crop_mask_full = mask[y0:y1 + 1, x0:x1 + 1]

    # Shrunk to fit a clear patch of sky in the rocket photograph.
    size = (140, 138)
    crop_r = np.array(crop.resize(size, Image.LANCZOS), dtype=float)
    crop_mask = np.asarray(Image.fromarray((crop_mask_full * 255).astype(np.uint8))
                           .resize(size, Image.LANCZOS), dtype=float) / 255.0

    # Paste the suit onto a plain patch of sky, then simulate an erase aimed at a box shifted
    # half off it: the half beyond the suit's true edge changes, the suit's own pixels do not.
    bg = Image.fromarray(data.rocket()).convert('RGB')
    a = np.array(bg).astype(float)
    px, py = 385, 10
    h, w = crop_mask.shape
    region_bg = a[py:py + h, px:px + w].copy()
    m3 = crop_mask[:, :, None]
    a[py:py + h, px:px + w] = region_bg * (1 - m3) + crop_r * m3
    before = Image.fromarray(a.astype(np.uint8))

    box_wrong = [px + 40, py, px + 40 + w, py + h]
    changed = np.array(before).astype(float)
    edge = px + w                                          # the suit's own right edge
    changed[py:py + h, edge:box_wrong[2]] = np.clip(
        changed[py:py + h, edge:box_wrong[2]] + 40, 0, 255)
    after = Image.fromarray(changed.astype(np.uint8))

    ok, detail = check_object_erased(before, after, box_wrong, subject='the suit')
    assert ok is False, detail
    assert 'the erase did not remove it' in detail


@needs_model
def test_words_that_match_nothing_before_the_edit_get_no_verdict():
    """The words must point at something in the box before the edit."""
    img = astronaut()
    out, _frame_frac, _warning = objects.erase_object(img, box=SHUTTLE_BOX)
    ok, detail = check_object_erased(img, out, SHUTTLE_BOX, subject='the giraffe')
    assert ok is None, detail
    assert 'matches nothing in this box before the edit' in detail


def test_without_the_grounding_extra_it_reports_containment_only(monkeypatch):
    """Without torch/transformers it still catches a no-op, but says nothing about whether
    the named thing is gone once something has changed."""
    monkeypatch.setattr(grounding, 'available', lambda: False)
    img = astronaut()
    a = np.array(img).astype(np.uint8)
    a[SUIT_BOX[1]:SUIT_BOX[3], SUIT_BOX[0]:SUIT_BOX[2]] = 128
    out = Image.fromarray(a)
    ok, detail = check_object_erased(img, out, SUIT_BOX, subject='the suit')
    assert ok is None, detail
    assert 'cannot check whether' in detail and 'grounding' in detail


@needs_model
def test_the_workspace_verifies_an_erase_call():
    """`subject` rides along as a tool argument, as for `recolour_object`."""
    img = astronaut()
    res = objects.erase_object(img, box=SHUTTLE_BOX)
    out = res[0]
    args = {'box': SHUTTLE_BOX, 'subject': 'the rocket'}
    ok, detail = workspace.verify('erase_object', args, img, out, region=res.region)
    assert ok is None, detail            # no PASS verdict exists -- see the check's docstring
    assert "still reads as 'the rocket'" in detail


def test_the_workspace_still_gives_a_real_erase_no_verdict_without_subject():
    img = astronaut()
    res = objects.erase_object(img, box=SHUTTLE_BOX)
    out = res[0]
    ok, detail = workspace.verify('erase_object', {'box': SHUTTLE_BOX}, img, out,
                                  region=res.region)
    assert ok is None, detail
    assert 'WHAT was erased is not checked' in detail


# `edge_ratio` (a candidate PASS signal from `evals/erase_structure.py`) was rejected: it
# fails in both directions. These two tests pin that behaviour synthetically.

from evals.erase_structure import edge_ratio                             # noqa: E402


def _checkerboard(size, square=8):
    a = np.indices(size)
    return (((a[0] // square) + (a[1] // square)) % 2 * 255).astype(np.uint8)


def test_edge_ratio_reads_high_when_the_box_has_more_structure_than_its_surround():
    """The direction the candidate is meant to catch: a flat surround, a structured box."""
    H, W = 200, 200
    a = np.full((H, W), 180, np.uint8)
    box = (60, 60, 140, 140)
    x0, y0, x1, y1 = box
    a[y0:y1, x0:x1] = _checkerboard((y1 - y0, x1 - x0))
    after = Image.fromarray(np.stack([a] * 3, axis=-1))
    assert edge_ratio(after, box) > 2.0


def test_edge_ratio_cannot_tell_a_smooth_ghost_from_a_clean_fill():
    """A smooth leftover object against a texture-rich surround reads like a clean fill,
    since edge density is a property of the scene as much as the box's own content."""
    H, W = 200, 200
    a = _checkerboard((H, W))
    box = (60, 60, 140, 140)
    x0, y0, x1, y1 = box
    a[y0:y1, x0:x1] = 180                     # a flat, "ghost" patch left in a busy scene
    after = Image.fromarray(np.stack([a] * 3, axis=-1))
    assert edge_ratio(after, box) < 0.5       # reads "clean" despite being a leftover
