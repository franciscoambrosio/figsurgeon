"""A recolour that recoloured nothing must not come back verified.

`check_unchanged_outside` reports containment, which a no-op recolour satisfies trivially
(zero changed outside a box where zero changed inside it too). Every check here is paired:
a real edit must not get a wrong verdict either, and a half-done one must get no verdict
rather than a false pass or fail, per `evals/object_completeness.py`.
"""
import numpy as np
import pytest
from PIL import Image
from skimage import data

from figsurgeon import grounding, objects, workspace
from figsurgeon.verify_photo import check_object_recoloured

BOX = [60, 180, 420, 512]        # the astronaut's suit


def astronaut():
    return Image.fromarray(data.astronaut())


def test_a_recolour_that_changed_nothing_fails():
    img = astronaut()
    ok, detail = check_object_recoloured(img, img.copy(), BOX)
    assert ok is False, detail
    assert 'nothing inside the box changed' in detail


def test_a_real_recolour_is_not_failed():
    img = astronaut()
    out, _ = objects.recolour_object(img, box=BOX, to_rgb=(40, 80, 200))
    ok, detail = check_object_recoloured(img, out, BOX)
    assert ok is not False, detail
    assert 'px recoloured' in detail


def test_a_half_recolour_gets_no_verdict_rather_than_a_wrong_one():
    """Completeness is not measurable from the output alone, so a half recolour gets no
    verdict -- not verified, but not failed either."""
    img = astronaut()
    mask = np.zeros(np.array(img).shape[:2], float)
    mask[BOX[1]:(BOX[1] + BOX[3]) // 2, BOX[0]:BOX[2]] = 1.0
    # Shaded, not flat: a flat fill is its own failure (see `recoloured`), not this one.
    ok, detail = check_object_recoloured(img, recoloured(img, mask), BOX)
    assert ok is None, detail


def test_the_workspace_fails_a_recolour_that_did_nothing():
    """Through `workspace.verify`, not just the check function directly."""
    img = astronaut()
    ok, detail = workspace.verify('recolour_object',
                                  {'box': BOX, 'to_rgb': [40, 80, 200]}, img, img.copy())
    assert ok is False, detail
    assert 'no object to recolour' in detail


def test_the_workspace_still_gives_a_real_object_edit_no_verdict():
    img = astronaut()
    res = objects.recolour_object(img, box=BOX, to_rgb=(40, 80, 200))
    out, _ = res
    ok, detail = workspace.verify('recolour_object',
                                  {'box': BOX, 'to_rgb': [40, 80, 200]}, img, out,
                                  region=res.region)
    assert ok is None, detail
    assert 'px recoloured' in detail and 'preview_object_mask' in detail


def test_verifying_a_followed_edit_against_the_bare_box_says_what_is_missing():
    """An edit that followed the object past its box correctly fails containment against the
    bare box, but the message must name the cause (no region was stated), or a true 'leak'
    reads as a broken tool. Verified again WITH the region, the same edit is fine."""
    img = astronaut()
    res = objects.recolour_object(img, box=BOX, to_rgb=(40, 80, 200))
    out, _ = res
    if res.region == tuple(BOX):
        return                                  # the object stayed inside; nothing to show
    ok, detail = workspace.verify('recolour_object',
                                  {'box': BOX, 'to_rgb': [40, 80, 200]}, img, out)
    assert ok is False and 'followed the object past its box' in detail, detail
    ok2, _ = workspace.verify('recolour_object', {'box': BOX, 'to_rgb': [40, 80, 200]},
                              img, out, region=res.region)
    assert ok2 is None


# --------------------------------------------------------------------------------------
# `subject` is the caller's own words, the only evidence not derived from the box, so it is
# what can tell a complete recolour from a half-done one or catch a box on the wrong object.
# Every abstention below is pinned too (`evals/object_completeness.py`).

needs_model = pytest.mark.skipif(not grounding.available(),
                                 reason='needs figsurgeon[grounding] (torch + transformers)')


def suit_mask():
    return np.asarray(objects.segment_object(astronaut(), BOX, follow_object=False)[0])


def recoloured(img, mask, rgb=(40, 80, 200)):
    """`objects.recolour_object`'s own maths, driven by a mask given from outside. Shaded,
    not flat, since a flat fill is graded as a flattened object (see `texture_kept`), and
    these tests are about completeness, not flattening."""
    a = np.array(img.convert('RGB')).astype(float)
    target = np.array(rgb, float)
    scale = np.clip(a.mean(axis=2) / 255.0 / max(target.mean() / 255.0, 1e-6), 0.35, 1.9)
    painted = np.clip(target[None, None, :] * scale[:, :, None], 0, 255)
    m = np.asarray(mask, float)[:, :, None]
    return Image.fromarray((a * (1 - m) + painted * m).astype(np.uint8))


@needs_model
def test_a_complete_recolour_is_verified_when_the_caller_says_what_it_was():
    """Without `subject` this same edit gets no verdict at all."""
    img = astronaut()
    out = recoloured(img, suit_mask())
    ok, detail = check_object_recoloured(img, out, BOX, subject='the suit')
    assert ok is True, detail
    assert "of 'the suit' was recoloured" in detail


@needs_model
def test_half_the_object_recoloured_fails_and_says_how_much_was_left():
    """The segmenter found part of the object; the rest kept its original colour and
    containment passed because nothing leaked, so this must still fail."""
    img = astronaut()
    mask = suit_mask().copy()
    ys = np.nonzero(mask.any(axis=1))[0]
    mask[(ys.min() + ys.max()) // 2:, :] = 0.0             # the leg-left-behind case
    ok, detail = check_object_recoloured(img, recoloured(img, mask), BOX,
                                         subject='the suit')
    assert ok is False, detail
    assert 'kept its original colour' in detail


@needs_model
def test_a_box_that_found_the_wrong_thing_is_reported_rather_than_verified():
    """A box mostly containing sky gets the sky segmented, not the rocket; that disagreement
    must be named rather than silently verified."""
    img = Image.fromarray(data.rocket())
    box = [30, 30, 560, 300]
    out, _ = objects.recolour_object(img, box=box, to_rgb=(40, 80, 200))
    ok, detail = check_object_recoloured(img, out, box, subject='the rocket')
    assert ok is None, detail
    assert 'disagree about which thing this is' in detail


@needs_model
def test_words_that_match_nothing_get_no_verdict_instead_of_a_failure():
    """Words matching nothing in the image would make every edit look 0 % complete; must
    abstain, not fail correct work."""
    img = astronaut()
    ok, detail = check_object_recoloured(img, recoloured(img, suit_mask()), BOX,
                                         subject='the giraffe')
    assert ok is None, detail
    assert 'matches nothing in this picture' in detail


@needs_model
def test_words_that_cover_the_whole_region_get_no_verdict():
    """When the box lies wholly inside the named object, "how much of it changed" is just
    the box's own coverage number again, so this must abstain rather than double-report."""
    img = Image.fromarray(data.chelsea())
    box = [70, 50, 290, 250]
    out, _ = objects.recolour_object(img, box=box, to_rgb=(40, 80, 200))
    ok, detail = check_object_recoloured(img, out, box, subject='the cat')
    assert ok is None, detail
    assert 'covers the whole region' in detail


@needs_model
def test_the_workspace_verifies_the_edit_when_the_tool_call_carries_the_words():
    """`subject` rides along as a tool argument, so a model gets the verdict just by saying
    what it edited."""
    img = astronaut()
    args = {'box': BOX, 'to_rgb': [40, 80, 200], 'subject': 'the suit'}
    res = objects.recolour_object(img, box=BOX, to_rgb=(40, 80, 200))
    ok, detail = workspace.verify('recolour_object', args, img, res[0], region=res.region)
    assert ok is True, detail
    assert "of 'the suit' was recoloured" in detail


@needs_model
def test_a_box_in_the_wrong_place_is_told_apart_from_words_that_match_nothing():
    """A box moved off the object leaves the named thing present in the picture but absent
    from the edited region -- a different silence than "matches nothing", and must be
    reported as such, or the caller looks at their words instead of their box."""
    img = astronaut()
    elsewhere = [0, 0, 120, 120]                  # a corner of the background
    out, _ = objects.recolour_object(img, box=elsewhere, to_rgb=(40, 80, 200))
    ok, detail = check_object_recoloured(img, out, elsewhere, subject='the suit')
    assert ok is None, detail
    assert 'IS in this picture' in detail, detail
    assert 'none of it is in the region that changed' in detail, detail


# ------------------------------------------------- where the phrase is resolved

def test_the_phrase_is_resolved_on_the_box_not_the_whole_frame():
    """CLIPSeg sees a fixed 352x352 input, so a small or thin object (a boat's rigging, say)
    reads back as a filled silhouette if asked about the whole frame -- grading a correct
    recolour against enclosed sky it never touched. The phrase must be asked about the box
    plus context only, with nothing outside it claimed."""
    from figsurgeon.verify_photo import _phrase_probability

    img = astronaut()
    W, H = img.size
    box = (60, 180, 220, 320)
    calls = []

    class FakeProb:
        """Records what the model was shown; answers 'everything' so only geometry is tested."""
        def __call__(self, rgb, phrase):
            calls.append(rgb.size)
            return np.ones((rgb.size[1], rgb.size[0]), dtype=float)

    import figsurgeon.grounding as G
    real = G._probability
    G._probability = FakeProb()
    try:
        p = _phrase_probability(img.convert('RGB'), 'the suit', box)
    finally:
        G._probability = real

    assert p.shape == (H, W)
    # asked about the box plus 30 % of context on each side, not the frame
    assert calls == [(int(160 * 1.6), int(140 * 1.6))], calls
    # and nothing outside that crop is claimed, in either direction
    assert p[:int(180 - 140 * 0.3) - 1, :].max() == 0.0
    assert p[box[1]:box[3], box[0]:box[2]].min() == 1.0


def test_a_box_covering_the_frame_still_asks_about_the_frame():
    """A box that IS the whole frame has nothing to zoom into: the caller gets exactly what
    they got before."""
    from figsurgeon.verify_photo import _phrase_probability

    img = astronaut()
    W, H = img.size
    seen = []

    import figsurgeon.grounding as G
    real = G._probability
    G._probability = lambda rgb, phrase: seen.append(rgb.size) or np.zeros(
        (rgb.size[1], rgb.size[0]), dtype=float)
    try:
        _phrase_probability(img.convert('RGB'), 'the suit', (0, 0, W, H))
    finally:
        G._probability = real
    assert seen == [(W, H)]


# ------------------------------------------- the grounded path, which had no check at all

@needs_model
def test_a_phrase_edit_is_corroborated_by_a_model_that_cannot_read():
    """`recolour_subject` paints what a phrase resolves to, with no box, so it needs its own
    check: an object-shaped edit is corroborated by a segmenter given only the region, none
    of the words. A rectangle painted over the same photo must not pass."""
    from figsurgeon.verify_photo import check_subject_recoloured

    img = astronaut()
    good = recoloured(img, suit_mask())
    ok, detail = check_subject_recoloured(img, good, 'the suit')
    assert ok is True, detail
    assert 'IoU' in detail and 'not the same as the RIGHT object' in detail

    a = np.array(img.convert('RGB'))
    a[150:400, 100:260] = (40, 80, 200)                # a slab, not an object
    ok, detail = check_subject_recoloured(img, Image.fromarray(a), 'the suit')
    assert ok is False, detail
    assert 'disagree about what' in detail


@needs_model
def test_a_phrase_edit_that_did_nothing_fails_rather_than_abstaining():
    """The words matched nothing or the recolour was a no-op; either way it must fail, not
    abstain."""
    from figsurgeon.verify_photo import check_subject_recoloured
    img = astronaut()
    ok, detail = check_subject_recoloured(img, img, 'the wombat')
    assert ok is False and 'nothing changed' in detail, detail


@needs_model
def test_several_boxes_are_graded_one_by_one():
    """Several boxes ("recolour all three people") must be graded per box, not pooled as one
    union -- pooling grades mostly the gaps between them and can abstain on work that was
    right. A box where nothing changed must be named, not averaged away by the others."""
    from figsurgeon.verify_photo import check_object_recoloured

    img = astronaut()
    mask = suit_mask()
    out = recoloured(img, mask)
    boxes = [(60, 180, 240, 512), (240, 180, 420, 512)]       # the suit, split in two
    region = (60, 180, 420, 512)
    ok, detail = check_object_recoloured(img, out, region, subject='the suit', boxes=boxes)
    assert 'box 1' in detail and 'box 2' in detail, detail
    assert ok is not False, detail

    untouched = [(60, 180, 240, 512), (0, 0, 80, 80)]          # the second box: sky
    ok, detail = check_object_recoloured(img, out, region, subject='the suit',
                                         boxes=untouched)
    assert ok is False, detail
    assert 'box 2: nothing changed inside it' in detail, detail


@needs_model
def test_the_shape_check_declines_the_three_requests_it_cannot_judge():
    """A segmenter prompted with one box cannot reproduce a mask of several separate things,
    and it segments whole objects, so a thin part scores low regardless of truth: several
    disjoint regions, a part-of-a-thing phrase, and a skeletal/thin shape must each get no
    verdict rather than a wrong one, while a single compact object still gets one."""
    from figsurgeon.verify_photo import check_subject_recoloured
    img = astronaut()
    a = np.array(img.convert('RGB'))

    two = a.copy()
    two[100:180, 60:140] = (40, 80, 200)
    two[300:380, 300:380] = (40, 80, 200)          # two separate regions
    ok, detail = check_subject_recoloured(img, Image.fromarray(two), 'the suits')
    assert ok is None and 'separate regions' in detail, detail

    whole = recoloured(img, suit_mask())
    ok, detail = check_subject_recoloured(img, whole, "the astronaut's sleeve")
    assert ok is None and 'names PART of a thing' in detail, detail

    skeletal = a.copy()                            # a lattice: mostly hull, little area
    for x in range(120, 400, 40):
        skeletal[120:400, x:x + 4] = (40, 80, 200)
    for y in range(120, 400, 40):
        skeletal[y:y + 4, 120:400] = (40, 80, 200)
    ok, detail = check_subject_recoloured(img, Image.fromarray(skeletal), 'the railing')
    assert ok is None and 'skeletal' in detail, detail

    ok, detail = check_subject_recoloured(img, whole, 'the suit')
    assert ok is True, detail


@needs_model
def test_two_models_can_still_agree_on_the_wrong_thing():
    """Two independent models can still agree on the wrong object (e.g. "the fence"
    resolving to grass), so a passing verdict must keep saying what it does not guarantee."""
    from figsurgeon.verify_photo import check_subject_recoloured
    img = astronaut()
    ok, detail = check_subject_recoloured(img, recoloured(img, suit_mask()), 'the suit')
    assert ok is True
    assert 'not the same as the RIGHT object' in detail, detail


# ------------------------------- which one, versus what it is (evals/instance_phrases.py)

def test_position_words_are_dropped_but_attributes_are_not():
    """A locator says WHICH one, and the box already says that; an attribute says WHAT the
    thing is and must survive, or "the red car" would be graded as "the car"."""
    from figsurgeon.verify_photo import _drop_position

    for phrase, bare in [('the sheep on the left', 'the sheep'),
                         ('the middle sheep', 'the sheep'),
                         ('the second saddle from the front', 'the saddle'),
                         ('the man in the middle', 'the man'),
                         ('the bicycle in the foreground', 'the bicycle')]:
        assert _drop_position(phrase) == (bare, phrase), phrase

    # nothing to drop, or nothing left if it were dropped
    for phrase in ['the red car', 'the suit', 'the wooden chair',
                   'the one on the left', 'the dog behind the fence',
                   'the car in front of the house']:
        assert _drop_position(phrase) == (phrase, ''), phrase


@needs_model
def test_the_phrase_is_asked_without_its_position_and_the_note_says_so():
    """The model is asked about a crop of the box, where a position word like "on the left"
    can only mean the one object in it, so it must be shown the bare phrase -- and the
    caller told which half of their phrase was actually checked."""
    from figsurgeon import verify_photo as V

    img = astronaut()
    W, H = img.size
    box = (60, 180, 420, 512)
    asked = []

    # The rectangle answered is exactly what changed: a complete recolour, so this is a PASS.
    crop = (0, 81, W, H)                      # the box padded by _EXTENT_PAD, clipped
    said = (128, 189, 384, 404)

    def fake(rgb, phrase):
        asked.append(phrase)
        p = np.zeros((rgb.size[1], rgb.size[0]), dtype=float)
        p[said[1] - crop[1]:said[3] - crop[1], said[0] - crop[0]:said[2] - crop[0]] = 0.9
        return p

    changed = np.zeros((H, W), bool)
    changed[said[1]:said[3], said[0]:said[2]] = True
    real = grounding._probability
    grounding._probability = fake
    try:
        ok, note = V._completeness(img.convert('RGB'), changed, box,
                                   'the suit on the left', 'took')
    finally:
        grounding._probability = real

    assert asked == ['the suit'], asked
    assert "'the suit on the left'" in note and "asked as 'the suit'" in note, note
    assert 'not checked' in note, note
    assert ok is True, note


# ------------------- the colour-pop, which could not say WHICH thing kept its colour

def isolated(img, mask, flatten=0.5):
    """`objects.isolate_object`'s maths, driven by a mask given from outside."""
    a = np.array(img.convert('RGB')).astype(float)
    lum = a.mean(axis=2, keepdims=True)
    grey = np.repeat(lum, 3, axis=2) * (1 - flatten) + 128.0 * flatten
    m = np.asarray(mask, float)[:, :, None]
    return Image.fromarray(np.clip(a * m + grey * (1 - m), 0, 255).astype(np.uint8))


@needs_model
def test_a_colour_pop_of_the_background_passes_the_check_that_has_no_subject():
    """Without a `subject`, keeping the BACKGROUND in colour and greying the object still
    passes the check -- indistinguishable from a segmenter taking the sky instead of the
    rocket. This is why `subject` was wired through."""
    from figsurgeon.verify_photo import check_isolate_object

    img = astronaut()
    x0, y0, x1, y1 = BOX
    inside = np.zeros(np.asarray(img).shape[:2], bool)
    inside[y0:y1, x0:x1] = True
    background = inside & (suit_mask() < 0.5)

    ok, detail = check_isolate_object(img, isolated(img, background), [list(BOX)])
    assert ok is True, detail          # blind, and wrong -- hence `subject`


@needs_model
def test_the_colour_pop_is_graded_against_the_words_in_both_directions():
    """With `subject`, the two edits above must separate: the object kept its colour, or
    something else did. The wrong one must not come back verified (fail or abstain, either
    is honest)."""
    from figsurgeon.verify_photo import check_isolate_object

    img = astronaut()
    x0, y0, x1, y1 = BOX
    inside = np.zeros(np.asarray(img).shape[:2], bool)
    inside[y0:y1, x0:x1] = True
    mask = suit_mask()

    ok, detail = check_isolate_object(img, isolated(img, mask), [list(BOX)],
                                      subject='the suit')
    assert ok is True, detail
    assert "of 'the suit' kept its colour" in detail, detail

    ok, detail = check_isolate_object(img, isolated(img, inside & (mask < 0.5)),
                                      [list(BOX)], subject='the suit')
    assert ok is not True, detail


def test_a_cutout_says_nothing_about_itself_without_words():
    """Without words, an RGBA cutout can only report that it is not empty and not the whole
    frame -- a cutout of the BACKGROUND satisfies both, so the verdict must stay open."""
    from figsurgeon.verify_photo import check_object_extracted

    img = astronaut()
    x0, y0, x1, y1 = BOX
    inside = np.zeros(np.asarray(img).shape[:2], bool)
    inside[y0:y1, x0:x1] = True
    background = inside & (suit_mask() < 0.5)

    ok, detail = check_object_extracted(img, cut_out(img, background), BOX)
    assert ok is None, detail
    assert 'not checked' in detail, detail

    empty = cut_out(img, np.zeros_like(inside, float))
    assert check_object_extracted(img, empty, BOX)[0] is False
    whole = cut_out(img, np.ones_like(inside, float))
    assert check_object_extracted(img, whole, BOX)[0] is False


def cut_out(img, mask):
    """`objects.extract_object`'s maths, driven by a mask given from outside."""
    a = np.array(img.convert('RGB')).astype(np.uint8)
    alpha = np.clip(np.asarray(mask, float) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([a, alpha]))


@needs_model
def test_the_cutout_is_graded_against_the_words_in_both_directions():
    """With `subject`, a correct cutout must verify and a half cutout must not, on the same
    image."""
    from figsurgeon.verify_photo import check_object_extracted

    img = astronaut()
    x0, y0, x1, y1 = BOX
    inside = np.zeros(np.asarray(img).shape[:2], bool)
    inside[y0:y1, x0:x1] = True
    mask = suit_mask()

    ok, detail = check_object_extracted(img, cut_out(img, mask), BOX, subject='the suit')
    assert ok is True, detail
    assert "of 'the suit' survived the cutout" in detail, detail

    ok, detail = check_object_extracted(img, cut_out(img, inside & (mask < 0.5)), BOX,
                                        subject='the suit')
    assert ok is not True, detail


# ---------------------------------------------------------------------------
# Flattening: the failure every other check here passes.

def test_a_flattened_recolour_fails_where_a_shaded_one_passes():
    """A flat fill and a shaded recolour of the same object and box: the flat one must fail
    for lost texture, the shaded one must not."""
    img = astronaut()
    mask = suit_mask()

    shaded = recoloured(img, mask)
    ok, detail = check_object_recoloured(img, shaded, BOX)
    assert ok is not False, detail
    assert 'texture kept' in detail, detail

    flat = Image.fromarray(
        (np.array(img).astype(float) * (1 - np.asarray(mask, float)[:, :, None])
         + np.array([40., 80, 200]) * np.asarray(mask, float)[:, :, None]).astype(np.uint8))
    ok, detail = check_object_recoloured(img, flat, BOX)
    assert ok is False, detail
    assert 'FLATTENED' in detail and 'texture kept 0.00' in detail, detail


def test_the_flattening_check_abstains_on_an_object_that_never_had_texture():
    """A flat painted surface (a door, a chart bar) has no texture to lose, so recolouring
    it flat must not fail -- a check that cries wolf is worse than no check."""
    from figsurgeon.verify_photo import texture_kept

    a = np.full((300, 300, 3), 240, np.uint8)
    a[60:240, 60:240] = (200, 40, 40)                      # a flat red square, no texture
    before = Image.fromarray(a)
    b = a.copy()
    b[60:240, 60:240] = (30, 90, 180)                      # recoloured flat, correctly
    after = Image.fromarray(b)

    changed = np.zeros(a.shape[:2], bool)
    changed[60:240, 60:240] = True
    kept, textured = texture_kept(before, after, changed)
    assert kept is None, f'fired on an object with no texture: {kept}'

    ok, detail = check_object_recoloured(before, after, [50, 50, 250, 250])
    assert ok is not False, detail
    assert 'FLATTENED' not in detail, detail


def test_the_flattening_measure_reads_how_much_was_lost():
    """Calibrated, not just ordered: blending a correct recolour toward flat by t reads
    (1 - t). The midpoint must fail, a quarter must not -- the cut is a stated threshold
    (over half the texture gone), not a fitted gap."""
    from figsurgeon.verify_photo import texture_kept

    img = astronaut()
    mask = np.asarray(suit_mask(), float)[:, :, None]
    shaded = np.array(recoloured(img, suit_mask())).astype(float)
    flat = (np.array(img).astype(float) * (1 - mask)
            + np.array([40., 80, 200]) * mask)
    changed = np.asarray(suit_mask(), float) > 0.5

    readings = {}
    for t in (0.25, 0.5):
        blended = Image.fromarray((shaded * (1 - t) + flat * t).astype(np.uint8))
        readings[t], _ = texture_kept(img, blended, changed)
    assert 0.6 < readings[0.25] < 0.85, readings
    assert 0.35 < readings[0.5] < 0.6, readings
