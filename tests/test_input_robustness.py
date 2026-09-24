"""Regression tests for input the package was never built against, and for silent no-ops.

From sweeping input shapes and multi-step workflows in `evals/robustness.py` and
`evals/use_cases.py`, beyond what a well-formed RGB photo exercises.
"""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from figsurgeon import ImageWorkspace, rebrand
from figsurgeon.tools import dispatch


@pytest.fixture(scope='module')
def cat():
    from skimage import data
    return Image.fromarray(data.chelsea())


@pytest.fixture(scope='module')
def cutout():
    from skimage import data
    img = Image.fromarray(data.chelsea()).convert('RGBA')
    alpha = Image.new('L', img.size, 0)
    ImageDraw.Draw(alpha).ellipse([80, 40, 380, 280], fill=255)
    img.putalpha(alpha)
    return img


# ---------------------------------------------------------------- input modes
@pytest.mark.parametrize('mode', ['L', 'P', 'CMYK', 'LA', '1'])
@pytest.mark.parametrize('call', [('rotate', {'angle': 12}),
                                  ('blur_background', {'radius': 6}),
                                  ('adjust', {'brightness': 1.2}),
                                  ('stylise', {'effect': 'vignette'})])
def test_unusual_input_modes_are_handled(cat, mode, call):
    """Palette PNGs, scanned 'L' pages and CMYK assets must not raise -- some calls built
    PIL operations against the input's own mode instead of converting first."""
    name, args = call
    result = dispatch(name, args, cat.convert(mode))
    assert not result.note.startswith('error'), f'{mode} + {name}: {result.note}'


def test_sixteen_bit_input_is_handled(cat):
    a = (np.array(cat.convert('L')).astype(np.uint16) * 257).astype(np.int32)
    sixteen = Image.fromarray(a)
    result = dispatch('adjust', {'brightness': 1.2}, sixteen)
    assert not result.note.startswith('error'), result.note


# ---------------------------------------------------------------- transparency
@pytest.mark.parametrize('name,args', [
    ('adjust', {'brightness': 1.3}),
    ('stylise', {'effect': 'sepia'}),
    ('stylise', {'effect': 'vignette'}),
    ('isolate_colour', {'target_rgb': [190, 150, 90]}),
    ('denoise', {'strength': 4}),
])
def test_transparency_survives_operations_that_do_not_understand_it(cutout, name, args):
    """Operations that convert to RGB internally must not flatten a cutout's alpha onto
    black; a brightness check alone would still verify that."""
    result = dispatch(name, args, cutout)
    assert result.image.mode == 'RGBA', f'{name} dropped the alpha channel'
    alpha = np.array(result.image)[..., 3]
    assert (alpha < 250).mean() > 0.05, f'{name} made the image fully opaque'


def test_rotating_a_cutout_does_not_paint_the_corners_white(cutout):
    """A rotated cutout's transparent surround must stay transparent, not fill white."""
    result = dispatch('rotate', {'angle': 20}, cutout)
    assert result.image.mode == 'RGBA'
    corner = np.array(result.image)[2, 2]
    assert corner[3] == 0, f'corner is opaque {tuple(corner)} after rotating a cutout'


def test_a_plain_rgb_image_is_not_given_an_alpha_channel(cat):
    result = dispatch('adjust', {'brightness': 1.2}, cat)
    assert result.image.mode == 'RGB'


# ---------------------------------------------------------------- silent no-ops
def test_segmentation_finding_nothing_warns_instead_of_claiming_success(monkeypatch):
    """A box GrabCut resolves nothing in must warn rather than report success: a change of
    zero pixels trivially passes the localisation check too."""
    from figsurgeon import objects
    from skimage import data
    # Forces the GrabCut path: SlimSAM finds something on this box, so only GrabCut
    # exercises the "found nothing" state this warning guards.
    monkeypatch.setattr(objects, 'available', lambda: False)
    rocket = Image.fromarray(data.rocket())
    empty_box = [415, 0, 470, 200]
    for name, args in [('erase_object', {'box': empty_box}),
                       ('recolour_object', {'box': empty_box, 'to_rgb': [255, 0, 0]}),
                       ('isolate_object', {'box': empty_box}),
                       ('extract_object', {'box': empty_box})]:
        w = ImageWorkspace(rocket)
        result = w.apply(name, args)
        assert 'WARNING: nothing was' in result.note, f'{name}: {result.note[:100]}'
        assert np.array_equal(np.array(w.image), np.array(rocket))
        assert w.history == [], f'{name} recorded a no-op as an undoable edit'


def test_a_box_that_does_segment_still_works():
    """The guard must not fire on a working call."""
    from skimage import data
    w = ImageWorkspace(Image.fromarray(data.rocket()))
    result = w.apply('erase_object', {'box': [150, 80, 230, 340]})
    assert 'WARNING: nothing was' not in result.note
    assert len(w.history) == 1


# ---------------------------------------------------------------- missing dependencies
def test_a_missing_system_binary_does_not_end_the_session(cat):
    """A missing tesseract binary must reach the caller as an error note, not kill the
    session with a raw TesseractNotFoundError."""
    import pytesseract
    try:
        pytesseract.get_tesseract_version()
        pytest.skip('tesseract IS installed here, so this path cannot be exercised')
    except Exception:
        pass
    result = dispatch('replace_font', {}, cat)
    assert result.note.startswith('error')
    assert 'environment' in result.note or 'install' in result.note


def test_the_missing_binary_is_named_along_with_the_command_that_installs_it(cat):
    """The library call a notebook makes directly, bypassing dispatch: the error it raises
    must still be an OSError (so dispatch's missing-dependency branch catches it) and must
    name the install command (apt-get/brew), not just pytesseract's own unhelpful readme
    pointer. Simulated so it runs on machines that do have tesseract too."""
    import pytesseract
    real = pytesseract.image_to_data

    def missing(*a, **k):
        raise pytesseract.TesseractNotFoundError()

    pytesseract.image_to_data = missing
    try:
        with pytest.raises(OSError) as e:
            rebrand.find_text_regions(cat)
        assert 'tesseract-ocr' in str(e.value) and 'brew' in str(e.value)
        assert dispatch('replace_font', {}, cat).note.startswith('error')
    finally:
        pytesseract.image_to_data = real


# ---------------------------------------------------------------- multi-step hazards
def test_changing_the_geometry_warns_that_boxes_are_now_stale(cat):
    """A box still in range after a crop is the dangerous stale one -- it edits silently in
    the wrong place instead of erroring."""
    w = ImageWorkspace(cat)
    result = w.apply('crop', {'box': [0, 0, 200, 200]})
    assert 'no longer valid' in result.note
    assert '200x200' in result.note

    plain = w.apply('adjust', {'brightness': 1.1})
    assert 'no longer valid' not in plain.note, 'an edit that keeps the size must not warn'


def test_combined_adjustments_are_inconclusive_rather_than_failed(cat):
    """A large contrast change can push mean brightness down even with a positive brightness
    factor in the same call; verify() must call that inconclusive, not failed."""
    from figsurgeon.workspace import verify
    before = cat
    after = dispatch('adjust', {'brightness': 1.05, 'contrast': 1.9}, cat).image
    ok, detail = verify('adjust', {'brightness': 1.05, 'contrast': 1.9}, before, after)
    assert ok is not False, f'combined adjustments must not be failed outright: {detail}'

    # A single adjustment keeps its strict verdict.
    from skimage import data
    grey = Image.fromarray(np.stack([data.camera()] * 3, axis=-1))
    after_grey = dispatch('adjust', {'saturation': 1.8}, grey).image
    ok, detail = verify('adjust', {'saturation': 1.8}, grey, after_grey)
    assert ok is False, f'a lone impossible adjustment must still fail: {detail}'


def test_undo_reaches_back_past_a_rejected_edit(cat):
    """Try a colour, reject it, try another: the retry must build on the state before the
    reject, not on top of it."""
    from skimage import data
    astro = Image.fromarray(data.astronaut())
    box = [60, 180, 420, 512]
    w = ImageWorkspace(astro)
    w.apply('recolour_object', {'box': box, 'to_rgb': [20, 70, 30]})
    w.apply('undo', {})
    w.apply('recolour_object', {'box': box, 'to_rgb': [40, 90, 220]})
    px = np.array(w.image)[400:460, 150:250].reshape(-1, 3).mean(axis=0)
    assert px[2] > px[1] + 25 and px[2] > px[0] + 25, (
        f'suit is {tuple(int(v) for v in px)}, not blue -- the retry stacked on the reject')


# ---------------------------------------------------------------- degenerate images
def test_degenerate_images_do_not_produce_verdicts_from_nan(cat):
    """A check that divides by zero returns nan, and `nan > 0.9` is False -- a FAILED verdict
    manufactured from no data. Covers a flat image (undefined correlation) and a tiny one
    (empty corner slice)."""
    from figsurgeon.verify_photo import check_denoise, check_vignette
    flat = Image.new('RGB', (200, 150), (90, 90, 90))
    ok, detail = check_denoise(flat, flat)
    assert ok is None, f'flat image denoise check should be inconclusive: {detail}'

    tiny = cat.resize((12, 10))
    ok, detail = check_vignette(tiny, tiny)
    assert ok is None, f'a 12x10 image has no corner to measure: {detail}'


def test_an_already_black_corner_does_not_verify_a_vignette_that_never_ran():
    """A `max(denominator, 1e-6)` floor can manufacture a verdict rather than prevent one:
    on an already-black corner, a no-op vignette measures 0/1e-6 and must not verify."""
    from figsurgeon.verify_photo import check_vignette
    import numpy as np
    a = np.zeros((240, 320, 3), np.uint8)
    a[60:180, 80:240] = 190
    black_corners = Image.fromarray(a)
    ok, detail = check_vignette(black_corners, black_corners)
    assert ok is None, f'a no-op vignette must not be verified: {detail}'
    assert 'already black' in detail, detail


def test_no_subject_warns_instead_of_blurring_the_whole_photo():
    """A subjectless image (a lawn) makes U2Net return an arbitrary speck, so a background
    op would blur the whole photo while verification still passes. Must warn instead."""
    from skimage import data
    grass = Image.fromarray(np.stack([data.grass()] * 3, axis=-1))
    for name in ('blur_background', 'remove_background'):
        w = ImageWorkspace(grass)
        args = {'colour_rgb': [255, 255, 255]} if name == 'replace_background' else {}
        result = w.apply(name, args)
        assert 'no clear subject' in result.note, f'{name}: {result.note[:90]}'
        assert np.array_equal(np.array(w.image), np.array(grass)), \
            f'{name} changed the image despite finding no subject'
        assert w.history == []


def test_a_real_subject_still_passes(cat):
    """The no-subject guard must not fire on a photo that does have one."""
    w = ImageWorkspace(cat)
    result = w.apply('blur_background', {'radius': 12})
    assert 'no clear subject' not in result.note
    assert 'CHECK FAILED' not in result.note, result.note


# ---------------------------------------------------------------- large images
def test_segmentation_returns_a_full_resolution_mask_after_downscaling():
    """Above the pixel cap, segmentation runs on a downscaled copy, but the returned mask
    must still match the original image size or every composite downstream misaligns."""
    from figsurgeon import objects
    from skimage import data
    big = Image.fromarray(data.astronaut()).resize((3600, 3600))
    # max_pixels is GrabCut's lever; SlimSAM always encodes at 1024x1024 regardless.
    mask, coverage = objects.segment_object(big, (400, 1200, 2900, 3550), backend='grabcut')
    assert mask.shape == (3600, 3600)
    assert 0.05 < coverage < 0.98, f'coverage {coverage} on a large image'


def test_small_images_are_not_downscaled_at_all():
    """The downscale cap must not perturb results below the threshold."""
    from figsurgeon import objects
    from skimage import data
    img = Image.fromarray(data.astronaut())
    a, cov_a = objects.segment_object(img, (60, 180, 420, 512), backend='grabcut')
    b, cov_b = objects.segment_object(img, (60, 180, 420, 512), max_pixels=10 ** 9,
                                      backend='grabcut')
    assert cov_a == cov_b
    assert np.array_equal(a, b)


def test_a_slow_segmentation_call_gives_the_remedy_for_the_BACKEND_that_ran(monkeypatch):
    """The two backends want opposite advice on a slow call: GrabCut's cost scales with
    pixel count, so it should say to resize; SlimSAM always encodes at a fixed 1024x1024,
    so the same advice there would be wrong."""
    from figsurgeon import objects, workspace
    from skimage import data
    img = Image.fromarray(data.chelsea())
    box = {'box': [141, 85, 211, 145], 'to_rgb': [30, 60, 200]}
    saved = workspace.SLOW_CALL_SECONDS
    try:
        workspace.SLOW_CALL_SECONDS = 0.0        # force the slow path on a fast call
        monkeypatch.setattr(objects, 'choose_backend', lambda im: 'grabcut')
        grab = ImageWorkspace(img).apply('recolour_object', box)
        assert 'resize first' in grab.note, grab.note
        assert 'MP image' in grab.note

        monkeypatch.setattr(objects, 'choose_backend', lambda im: 'slimsam')
        sam = ImageWorkspace(img).apply('recolour_object', box)
        assert 'resize' not in sam.note.split('NOTE:')[1], sam.note
        assert 'nearly flat in image size' in sam.note, sam.note
    finally:
        workspace.SLOW_CALL_SECONDS = saved

    quick = ImageWorkspace(img).apply('adjust', {'brightness': 1.1})
    assert 'resize first' not in quick.note, 'a non-segmentation call must not advise this'


# ---------------------------------------------------------------- colour names
def test_a_colour_name_selects_the_same_pixels_as_its_rgb(cat):
    """A colour name and its RGB equivalent from the same lookup table must select the same
    pixels, whichever `isolate_colour`/`replace_colour` is called with."""
    from figsurgeon import photo
    from figsurgeon.photo_describe import COLOUR_WORDS
    _, by_name = photo.isolate_colour(cat, 'blue')
    _, by_rgb = photo.isolate_colour(cat, COLOUR_WORDS['blue'])
    assert np.array_equal(by_name, by_rgb)
    _, from_name = photo.replace_colour(cat, 'brown', 'green')
    _, from_rgb = photo.replace_colour(cat, COLOUR_WORDS['brown'], COLOUR_WORDS['green'])
    assert np.array_equal(from_name, from_rgb)


@pytest.mark.parametrize('call', ['isolate', 'replace'])
def test_an_unknown_colour_name_names_the_ones_that_work(cat, call):
    from figsurgeon import photo
    with pytest.raises(ValueError) as excinfo:
        if call == 'isolate':
            photo.isolate_colour(cat, 'chartreuse')
        else:
            photo.replace_colour(cat, 'chartreuse', 'blue')
    message = str(excinfo.value)
    assert 'chartreuse' in message
    assert 'blue' in message and 'teal' in message, message
    assert 'r, g, b' in message, message


def test_dispatch_accepts_a_colour_name_where_the_schema_asks_for_numbers(cat):
    """A colour name where the schema asks for [r, g, b] must not reach int('r')."""
    result = dispatch('isolate_colour', {'target_rgb': 'red'}, cat)
    assert not result.note.startswith('error'), result.note
    bad = dispatch('isolate_colour', {'target_rgb': 'chartreuse'}, cat)
    assert 'teal' in bad.note, bad.note


def test_a_box_past_the_image_edge_is_clamped_not_wrapped(cat):
    """A box that overshoots the frame with a negative coordinate must clamp to the edge,
    not numpy-slice from the other end of the image."""
    from figsurgeon import photo
    W, H = cat.size
    assert photo.sample_colour(cat, (-5, 0, W, H)) == photo.sample_colour(cat, (0, 0, W, H))

    out = photo.blur_background(cat, focus=(-10, 40, 200, 200), radius=10, feather=0)
    a, b = np.array(cat), np.array(out)
    assert np.array_equal(a[60:180, 0:180], b[60:180, 0:180]), \
        'the focus box must stay sharp when it overshoots the left edge'


def test_a_focus_box_of_numpy_integers_is_a_box(cat):
    """A box of np.int64 (as refine_box or any numpy computation produces) must be accepted
    as a box, not fail an `isinstance(int)` check."""
    from figsurgeon import photo
    box = tuple(np.int64(v) for v in (10, 10, 120, 120))
    photo.blur_background(cat, focus=box, radius=6)


def test_a_colour_that_is_not_rgb_is_refused_not_used(cat):
    """An out-of-range or wrong-length RGB triple must be refused with an error, not used to
    compute a hue no pixel can have."""
    for bad in ([300, -20, 0], [255, 0], [1, 2, 3, 4]):
        r = dispatch('replace_colour', {'from_rgb': [200, 30, 30], 'to_rgb': bad}, cat)
        assert r.note.startswith('error') and 'RGB' in r.note, r.note
        assert r.mutates is False, r.note


def test_extract_object_segments_once(cat, monkeypatch):
    """extract_object must segment once, not once in dispatch to read coverage and again to
    build the cutout."""
    from figsurgeon import objects
    calls = []
    real = objects.segment_object

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(objects, 'segment_object', counting)
    r = dispatch('extract_object', {'box': [100, 40, 360, 290]}, cat)
    assert r.image.mode == 'RGBA', r.note
    assert len(calls) == 1, f'segmented {len(calls)} times'
