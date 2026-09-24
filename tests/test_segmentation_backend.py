"""The SlimSAM backend, and the two decisions that made it a decision rather than a port.

Every behaviour here is paired with a case it must PASS and one it must FAIL, so a backend
switch that silently fell back to GrabCut, or a clip test that never crossed the box, cannot
go green vacuously.
"""
import numpy as np
import pytest
from PIL import Image
from skimage import data

from figsurgeon import composite, objects

SUIT_BOX = (60, 180, 420, 512)
HELMET = (300, 380, 420, 512)          # the OTHER helmet, inside the box, not the suit

needs_slimsam = pytest.mark.skipif(not objects.available(),
                                   reason='needs figsurgeon[grounding] (torch)')


@pytest.fixture(scope='module')
def astronaut():
    return Image.fromarray(data.astronaut())


# ---------------------------------------------------------------- backend selection

@needs_slimsam
def test_auto_uses_slimsam_when_torch_is_present(astronaut):
    """'auto' must agree with 'slimsam' and disagree with 'grabcut', or the switch is not
    real."""
    auto, _ = objects.segment_object(astronaut, SUIT_BOX)
    sam, _ = objects.segment_object(astronaut, SUIT_BOX, backend='slimsam')
    grab, _ = objects.segment_object(astronaut, SUIT_BOX, backend='grabcut')
    assert np.array_equal(auto, sam)
    assert not np.array_equal(auto, grab)


def test_auto_falls_back_to_grabcut_when_torch_is_missing(astronaut, monkeypatch):
    """With torch unimportable, 'auto' still returns GrabCut's mask, but asking for SlimSAM
    by name must not fall back silently -- it raises, naming the install line."""
    monkeypatch.setitem(__import__('sys').modules, 'torch', None)
    monkeypatch.setitem(__import__('sys').modules, 'transformers', None)
    monkeypatch.setattr(objects, '_SAM', {})           # an already-loaded model would hide it
    assert objects.available() is False

    auto, coverage = objects.segment_object(astronaut, SUIT_BOX)
    grab, cov_grab = objects.segment_object(astronaut, SUIT_BOX, backend='grabcut')
    assert np.array_equal(auto, grab) and coverage == cov_grab

    with pytest.raises(ImportError) as e:
        objects.segment_object(astronaut, SUIT_BOX, backend='slimsam')
    assert 'figsurgeon[grounding]' in str(e.value)


def test_an_unknown_backend_is_rejected_rather_than_guessed(astronaut):
    with pytest.raises(ValueError) as e:
        objects.segment_object(astronaut, SUIT_BOX, backend='sam2')
    assert 'slimsam' in str(e.value)


@needs_slimsam
def test_auto_sends_a_flat_colour_figure_to_grabcut(astronaut):
    """On a flat-fill figure, SlimSAM returns the background and misses the element; the
    routing must send it to GrabCut instead, while SlimSAM-by-name still shows the failure."""
    from PIL import ImageDraw
    fig = Image.new('RGB', (300, 220), (30, 40, 60))
    ImageDraw.Draw(fig).ellipse([100, 60, 200, 160], fill=(220, 200, 90))
    loose = (44, 4, 266, 219)
    disc = np.zeros((220, 300), bool)
    disc[60:161, 100:201] = np.asarray(fig)[60:161, 100:201, 0] > 120

    assert objects.choose_backend(fig) == 'grabcut'
    assert objects.choose_backend(astronaut) == 'slimsam'

    auto = objects.segment_object(fig, loose, feather=0)[0] > 0.5
    sam = objects.segment_object(fig, loose, feather=0, backend='slimsam')[0] > 0.5
    iou = (auto & disc).sum() / max((auto | disc).sum(), 1)
    iou_sam = (sam & disc).sum() / max((sam | disc).sum(), 1)
    assert iou > 0.9, f'auto found IoU {iou:.2f} of the disc'
    assert iou_sam < 0.1, f'SlimSAM unexpectedly found the disc (IoU {iou_sam:.2f})'


def test_the_flatness_statistic_separates_figures_from_photographs(astronaut):
    """The flatness threshold only works if a figure and a photograph land far apart on
    either side of it."""
    from PIL import ImageDraw
    fig = Image.new('RGB', (300, 220), (30, 40, 60))
    ImageDraw.Draw(fig).ellipse([100, 60, 200, 160], fill=(220, 200, 90))
    assert objects._flat_fraction(fig) > objects.FLAT_FIGURE + 0.2
    assert objects._flat_fraction(astronaut) < objects.FLAT_FIGURE - 0.2


@needs_slimsam
def test_the_model_is_loaded_once_not_once_per_call(astronaut):
    """The second call must reuse the cached model, not call from_pretrained again."""
    objects.segment_object(astronaut, (60, 180, 200, 300), backend='slimsam')
    loaded = dict(objects._SAM)
    objects.segment_object(astronaut, (60, 180, 200, 300), backend='slimsam')
    assert len(objects._SAM) == 1
    assert objects._SAM[objects.MODEL_ID] is loaded[objects.MODEL_ID]


# ---------------------------------------------------------------- the mask is better

@needs_slimsam
def test_slimsam_leaves_the_other_helmet_alone_and_grabcut_does_not(astronaut):
    """A second, visually distinct object inside the same box: SlimSAM must leave it alone,
    GrabCut takes nearly all of it."""
    x0, y0, x1, y1 = HELMET
    sam, _ = objects.segment_object(astronaut, SUIT_BOX, backend='slimsam')
    grab, _ = objects.segment_object(astronaut, SUIT_BOX, backend='grabcut')
    assert float((sam[y0:y1, x0:x1] > 0.5).mean()) < 0.10
    assert float((grab[y0:y1, x0:x1] > 0.5).mean()) > 0.50


# ---------------------------------------------------------------- the box still holds

@needs_slimsam
def test_the_mask_follows_the_object_out_of_the_box_and_says_where_it_went(astronaut):
    """A SAM mask can legitimately leave the prompt box (the suit's far sleeve here);
    clipping it produced a seam down the caller's box edge, so the object is kept whole and
    the reached region reported for `enforce_region`. follow_object=False keeps the old,
    exact clip-to-box contract."""
    x0, y0, x1, y1 = SUIT_BOX
    seg = objects.segment_object(astronaut, SUIT_BOX, backend='slimsam')
    mask = seg[0]
    outside = np.ones(mask.shape, bool)
    outside[y0:y1, x0:x1] = False
    assert float(mask[outside].sum()) > 0, 'the suit does continue past this box'
    assert seg.outside_frac > 0.05
    rx0, ry0, rx1, ry1 = seg.region
    assert rx0 <= x0 and ry0 <= y0 and rx1 >= x1 and ry1 >= y1

    beyond = np.ones(mask.shape, bool)
    beyond[ry0:ry1, rx0:rx1] = False
    assert float(mask[beyond].sum()) == 0.0, 'the reported region must contain the mask'

    clipped = objects.segment_object(astronaut, SUIT_BOX, backend='slimsam',
                                     follow_object=False)
    assert float(clipped[0][outside].sum()) == 0.0
    assert clipped.region == SUIT_BOX and clipped.outside_frac == 0.0


@needs_slimsam
def test_a_blob_that_never_touches_the_box_is_not_adopted(astronaut):
    """Only the part of the mask CONNECTED to something inside the box is the object: a
    disconnected blob must be dropped, while one that reaches into the box (the sleeve case)
    is kept whole."""
    binary = np.zeros((200, 200), np.uint8)
    binary[90:110, 90:150] = 1          # crosses the box edge: the object, kept whole
    binary[10:20, 10:20] = 1            # unconnected, elsewhere: not the object
    kept = objects._connected_to_box(binary, (80, 80, 120, 120))
    assert kept[90:110, 120:150].all(), 'the part outside the box must survive'
    assert not kept[10:20, 10:20].any(), 'the disconnected blob must be dropped'


@needs_slimsam
def test_a_recolour_is_byte_identical_outside_the_region_it_reported(astronaut):
    """The region an operation states is what it is held to: a recolour that respects it
    must find nothing to revert, one that ignores it must be reverted and reported."""
    args = {'box': list(SUIT_BOX), 'to_rgb': [30, 60, 200]}
    res = objects.recolour_object(astronaut, SUIT_BOX, to_rgb=(30, 60, 200))
    out, _ = res
    rx0, ry0, rx1, ry1 = res.region

    before, after = np.array(astronaut), np.array(out)
    diff = np.abs(after.astype(int) - before.astype(int)).max(axis=2)
    diff[ry0:ry1, rx0:rx1] = 0
    assert int((diff > 0).sum()) == 0
    kept, note = composite.enforce_region('recolour_object', args, astronaut, out,
                                          region=res.region)
    assert note is None and np.array_equal(np.array(kept), after)

    everywhere = np.ones(before.shape[:2], float)
    leaky = _recolour_with(astronaut, everywhere)
    fixed, note = composite.enforce_region('recolour_object', args, astronaut, leaky,
                                           region=res.region)
    assert note is not None and 'reverted' in note
    fixed_diff = np.abs(np.array(fixed).astype(int) - before.astype(int)).max(axis=2)
    fixed_diff[ry0:ry1, rx0:rx1] = 0
    assert int((fixed_diff > 0).sum()) == 0


def _recolour_with(img, mask):
    """recolour_object's own arithmetic, on a mask supplied instead of segmented."""
    a = np.array(img.convert('RGB')).astype(float)
    target = np.array([30.0, 60.0, 200.0])
    scale = np.clip(a.mean(axis=2) / 255.0 / (target.mean() / 255.0), 0.35, 1.9)
    painted = np.clip(target[None, None, :] * scale[:, :, None], 0, 255)
    m = mask[:, :, None]
    return Image.fromarray((a * (1 - m) + painted * m).astype(np.uint8))


# ---------------------------------------------------------------- the lever reaches the edits

@needs_slimsam
def test_the_backend_argument_reaches_the_editing_wrappers(astronaut):
    """`backend=` must reach `recolour_object` and the other wrappers, not just
    `segment_object`: the two backends must disagree in the OUTPUT (a dropped argument
    would still pass a signature check but produce identical results)."""
    x0, y0, x1, y1 = HELMET
    before = np.array(astronaut).astype(int)
    frac = []
    for backend in ('grabcut', 'slimsam'):
        out, _ = objects.recolour_object(astronaut, SUIT_BOX, to_rgb=(30, 60, 200),
                                         backend=backend)
        d = np.abs(np.array(out).astype(int) - before)[y0:y1, x0:x1].max(axis=2)
        frac.append(float((d > 8).mean()))
    assert frac[0] > 0.5, f'GrabCut left the helmet alone ({frac[0]:.1%})'
    assert frac[1] < 0.1, f'SlimSAM recoloured the helmet ({frac[1]:.1%})'


# ---------------------------------------------------------------- points, where a box fails

ROCKET_BOX = (30, 30, 560, 300)        # drawn around the rocket, and mostly sky
ROCKET_BODY = (295, 120, 325, 280)     # the rocket itself
ROCKET_SKY = (60, 60, 200, 120)        # unmistakably sky, inside that box


@pytest.fixture(scope='module')
def rocket():
    return Image.fromarray(data.rocket())


@needs_slimsam
def test_a_point_gets_the_object_a_box_cannot(rocket):
    """A box around the rocket is mostly sky and every backend segments the sky from it --
    unfixable from a box. Two points on the rocket body must return the rocket instead."""
    sx0, sy0, sx1, sy1 = ROCKET_SKY
    bx0, by0, bx1, by1 = ROCKET_BODY

    by_point, _ = objects.segment_object(rocket, points=[(310, 160), (310, 260)])
    assert float((by_point[sy0:sy1, sx0:sx1] > 0.5).mean()) < 0.02
    assert float((by_point[by0:by1, bx0:bx1] > 0.5).mean()) > 0.35

    by_box, _ = objects.segment_object(rocket, ROCKET_BOX, backend='slimsam')
    assert float((by_box[sy0:sy1, sx0:sx1] > 0.5).mean()) > 0.5, 'the box really does take sky'


@needs_slimsam
def test_a_box_and_a_point_together_are_refused_rather_than_silently_ignored(rocket):
    """This checkpoint silently ignores points when a box is present, so passing both must
    raise rather than drop the thing the caller relied on; either prompt alone must work."""
    with pytest.raises(ValueError) as e:
        objects.segment_object(rocket, ROCKET_BOX, points=(310, 200))
    assert 'box OR points' in str(e.value)
    assert objects.segment_object(rocket, ROCKET_BOX, backend='slimsam')[1] > 0
    assert objects.segment_object(rocket, points=(310, 200))[1] > 0


def test_points_without_the_extra_say_what_to_install(rocket, monkeypatch):
    """Without torch, points must name the install line rather than fall back to GrabCut
    (which has no point prompt); a box must still work on the same machine."""
    monkeypatch.setitem(__import__('sys').modules, 'torch', None)
    monkeypatch.setitem(__import__('sys').modules, 'transformers', None)
    monkeypatch.setattr(objects, '_SAM', {})
    with pytest.raises(ImportError) as e:
        objects.segment_object(rocket, points=(310, 200))
    assert 'figsurgeon[grounding]' in str(e.value)
    assert objects.segment_object(rocket, ROCKET_BOX)[1] > 0


@needs_slimsam
def test_a_point_prompt_is_verified_against_the_region_it_reports(rocket):
    """A point prompt has no caller-drawn box, so the reported region is the whole contract
    and must still hold: nothing changes outside it, and the workspace gets an open verdict
    rather than 'no region given to check against'."""
    from figsurgeon import ImageWorkspace
    w = ImageWorkspace(rocket)
    result = w.apply('recolour_object', {'point': [[310, 160], [310, 260]],
                                         'to_rgb': [220, 60, 60]})
    region = result.data['region']
    rx0, ry0, rx1, ry1 = region
    d = np.abs(np.array(w.image).astype(int) - np.array(rocket).astype(int)).max(axis=2) > 8
    d[ry0:ry1, rx0:rx1] = False
    assert int(d.sum()) == 0, f'{d.sum()} px changed outside {region}'
    assert 'no region given' not in result.note
    assert 'px recoloured' in result.note


@needs_slimsam
def test_several_boxes_are_one_encoder_pass_and_the_same_masks(astronaut, monkeypatch):
    """Extra boxes on one photograph share one encoder pass, so segmenting them together must
    cost nothing extra AND produce the same mask as segmenting them one at a time (with the
    per-box path disabled, so the saving is real). Near-bit-exact, not exact: a boundary
    logit can land on either side of zero, but no pixel may cross from inside the mask to
    outside it -- far below the scale of a real backend disagreement."""
    one_at_a_time = np.maximum(
        objects.segment_object(astronaut, SUIT_BOX, backend='slimsam')[0],
        objects.segment_object(astronaut, HELMET, backend='slimsam')[0])

    def refuse(*a, **k):
        raise AssertionError('the encoder was paid per box')
    monkeypatch.setattr(objects, '_slimsam_binary', refuse)
    together, coverages = objects.segment_objects(astronaut, [SUIT_BOX, HELMET],
                                                  backend='slimsam')

    assert len(coverages) == 2
    worst = float(np.abs(together - one_at_a_time).max())
    assert worst <= 0.5, f'a pixel changed side of the mask by {worst:.2f}'
