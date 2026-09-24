"""`generative.fill`, driven without a key or a network call.

`_post` is stubbed to answer with a flat colour, standing in for a model that repaints the
whole region including the background. Tests check the compositing, not the model.
"""
import base64
import io

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from figsurgeon import generative
from figsurgeon.tools import dispatch

ANSWER_COLOUR = (10, 200, 10)


def _stub(body, timeout):
    content = body['messages'][0]['content']
    data_uri = content[1]['image_url']['url']
    raw = base64.b64decode(data_uri.split(',', 1)[1])
    size = Image.open(io.BytesIO(raw)).size
    answer = Image.new('RGB', size, ANSWER_COLOUR)
    buf = io.BytesIO()
    answer.save(buf, 'PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {'choices': [{'message': {'images': [
        {'image_url': {'url': 'data:image/png;base64,' + b64}}]}}],
            'usage': {}}


@pytest.fixture(autouse=True)
def stub_post(monkeypatch):
    monkeypatch.setattr(generative, '_post', _stub)   # intercepts every call, no key needed


@pytest.fixture
def disc():
    """A flat-colour circle on a flat background: GrabCut is deterministic on it, so the
    masked path is tested without depending on segmentation quality."""
    img = Image.new('RGB', (320, 240), (40, 40, 40))
    ImageDraw.Draw(img).ellipse([110, 70, 210, 170], fill=(200, 200, 60))
    return img


BOX = (90, 50, 230, 190)


def test_unmasked_fill_changes_the_whole_box(disc):
    out, info = generative.fill(disc, BOX, 'irrelevant, the stub ignores it')
    a, c = np.array(disc), np.array(out)
    inside = c[BOX[1]:BOX[3], BOX[0]:BOX[2]]
    assert (np.abs(inside.astype(int) - np.array(ANSWER_COLOUR)).max(axis=2) < 3).all()
    outside_changed = np.abs(a.astype(int) - c.astype(int)).max(axis=2) > 2
    outside_changed[BOX[1]:BOX[3], BOX[0]:BOX[2]] = False
    assert not outside_changed.any()
    assert 'mask' not in info


def test_masked_fill_leaves_the_background_inside_the_box_untouched(disc):
    out, info = generative.fill(disc, BOX, 'irrelevant, the stub ignores it',
                                mask_to_object=True)
    a, c = np.array(disc), np.array(out)
    d = np.abs(a.astype(int) - c.astype(int)).max(axis=2)
    mask = info['mask']
    assert (mask > 0.5).any(), 'segmentation found nothing to mask the fill to'

    # Zero-weight pixels (inside the box, not just outside it) must be byte-identical.
    zero = mask <= 0.0
    assert not (d[zero] > 2).any(), 'a pixel with zero mask weight changed'
    solid = mask > 0.95
    assert solid.any() and (d[solid] > 2).any(), 'nothing inside the mask changed'
    changed_colour = c[solid]
    assert (np.abs(changed_colour.astype(int) - np.array(ANSWER_COLOUR)).max(axis=1) < 3).all()
    assert 0 < info['masked_fraction'] < 1
    assert info['seam'] is not None


def test_masked_fill_raises_when_the_box_has_nothing_to_segment(monkeypatch):
    """An empty mask must not silently fall back to filling the whole box, which would
    quietly do the thing `mask_to_object` was passed to avoid."""
    from figsurgeon import objects
    monkeypatch.setattr(objects, 'available', lambda: False)   # force the GrabCut path
    from skimage import data
    rocket = Image.fromarray(data.rocket())
    empty_box = (415, 0, 470, 200)
    with pytest.raises(RuntimeError, match='no object found'):
        generative.fill(rocket, empty_box, 'irrelevant', mask_to_object=True)


def test_dispatch_wires_mask_to_object_through(disc):
    result = dispatch('generative_fill',
                      {'box': list(BOX), 'instruction': 'x', 'mask_to_object': True},
                      disc)
    assert 'masked to the segmented object' in result.note
    c = np.array(result.image)
    box_bg = (40, 40, 40)
    still_background = np.abs(c.astype(int) - np.array(box_bg)).max(axis=2) < 3
    # some pixels inside the box are still exactly the background colour
    assert still_background[BOX[1]:BOX[3], BOX[0]:BOX[2]].any()


# The context-ring colour fit (`figsurgeon/ringfit.py`): the flat stub carries no
# correspondence for it, so these use a second stub returning the crop under a known shift.

SHIFT_GAIN, SHIFT_OFFSET = (1.22, 1.0, 0.82), (0.015, 0.0, -0.006)


def _shifting_stub(body, timeout):
    """Answers with the crop it was sent, under a known linear-light colour shift."""
    from figsurgeon import ringfit
    content = body['messages'][0]['content']
    raw = base64.b64decode(content[1]['image_url']['url'].split(',', 1)[1])
    sent = Image.open(io.BytesIO(raw)).convert('RGB')
    answer = Image.fromarray(ringfit.apply(np.array(sent), SHIFT_GAIN, SHIFT_OFFSET))
    buf = io.BytesIO()
    answer.save(buf, 'PNG')
    return {'choices': [{'message': {'images': [
        {'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()}}]}}],
            'usage': {}}


@pytest.fixture
def textured():
    """A photograph-like frame: the ring must vary smoothly, or the gain is either
    unidentifiable (`ringfit` refuses) or destroyed by JPEG chroma subsampling."""
    rng = np.random.default_rng(7)
    img = Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(6))
    d = ImageDraw.Draw(img)
    d.ellipse([30, 30, 150, 140], fill=(190, 120, 60))
    d.rectangle([180, 120, 300, 220], fill=(60, 90, 160))
    return img


def test_ring_fit_recovers_a_known_colour_shift(textured, monkeypatch):
    monkeypatch.setattr(generative, '_post', _shifting_stub)
    out, info = generative.fill(textured, BOX, 'irrelevant')
    fit = info['colour_fit']
    assert fit['refused'] == '', fit
    assert fit['applied'] is True, fit
    # The referee is the held-out ring half, not the seam, which barely moves for this shift.
    before, after = fit['ring_err']
    assert after < before, fit['ring_err']
    assert fit['gain'][0] < 0.9 and fit['gain'][2] > 1.1, fit['gain']   # undoes the shift

    # Compared against the encoder's own JPEG q92 floor, not a constant, since that round
    # trip rings around hard edges regardless of colour shift.
    a, c = np.array(textured), np.array(out)
    box = (slice(BOX[1], BOX[3]), slice(BOX[0], BOX[2]))
    resid = np.abs(c[box].astype(int) - a[box].astype(int))
    buf = io.BytesIO()
    textured.save(buf, 'JPEG', quality=92)
    floor = np.abs(np.array(Image.open(io.BytesIO(buf.getvalue())).convert('RGB'))[box]
                   .astype(int) - a[box].astype(int))
    assert resid.mean() <= floor.mean() + 0.5, (resid.mean(), floor.mean())
    assert np.percentile(resid, 95) <= np.percentile(floor, 95) + 1
    assert np.percentile(resid, 99) <= np.percentile(floor, 99) + 1


def test_a_model_that_ignored_the_crop_gets_no_correction(disc):
    """The flat stub's ring fits every (gain, offset) equally, so a seam-only gate would
    prefer deleting the fill onto the background; the fit must refuse first."""
    out, info = generative.fill(disc, BOX, 'irrelevant, the stub ignores it')
    fit = info['colour_fit']
    assert fit['refused'], 'an unidentifiable ring was fitted anyway'
    assert fit['applied'] is False
    assert fit['r'] == [0.0, 0.0, 0.0], fit
    inside = np.array(out)[BOX[1]:BOX[3], BOX[0]:BOX[2]]
    assert (np.abs(inside.astype(int) - np.array(ANSWER_COLOUR)).max(axis=2) < 3).all(), \
        'the fill was collapsed toward the background instead of being left alone'


def test_match_colour_false_does_not_fit_at_all(textured, monkeypatch):
    monkeypatch.setattr(generative, '_post', _shifting_stub)
    _out, info = generative.fill(textured, BOX, 'irrelevant', match_colour=False)
    assert info['colour_fit'] == {'applied': False, 'seam_before': info['seam'],
                                  'seam_after': None, 'gain': None, 'offset': None}


def test_a_fit_with_nothing_to_correct_is_discarded(textured, monkeypatch):
    """A model that returned the crop unchanged gives a fit with no shift in it, and
    applying one is not free (a round trip through 8 bit), so the correction is dropped."""
    def _faithful_stub(body, timeout):
        content = body['messages'][0]['content']
        raw = base64.b64decode(content[1]['image_url']['url'].split(',', 1)[1])
        buf = io.BytesIO()
        Image.open(io.BytesIO(raw)).convert('RGB').save(buf, 'PNG')
        return {'choices': [{'message': {'images': [{'image_url': {
            'url': 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()}}]}}],
                'usage': {}}

    monkeypatch.setattr(generative, '_post', _faithful_stub)
    _out, info = generative.fill(textured, BOX, 'irrelevant')
    fit = info['colour_fit']
    assert fit['refused'] == '', fit            # the ring IS identifiable here
    assert fit['applied'] is False, fit         # ... there is just nothing to correct
    before, after = fit['ring_err']
    assert after >= before, fit['ring_err']


def test_every_instruction_carries_the_only_what_was_asked_clause(textured, monkeypatch):
    """Pins that the clause is sent, and that it names no object itself, since an image
    model handed a noun tends to draw it -- the failure this clause exists to prevent."""
    seen = {}

    def _capture(body, timeout):
        seen['prompt'] = body['messages'][0]['content'][0]['text']
        return _shifting_stub(body, timeout)

    monkeypatch.setattr(generative, '_post', _capture)
    generative.fill(textured, BOX, 'make the taxi green')
    assert generative.ONLY_WHAT_WAS_ASKED in seen['prompt']
    assert seen['prompt'].startswith('make the taxi green')
    for noun in ('car', 'person', 'tree', 'building', 'animal'):
        assert noun not in generative.ONLY_WHAT_WAS_ASKED.lower(), \
            f'the clause names {noun!r}, which is an invitation to draw one'
