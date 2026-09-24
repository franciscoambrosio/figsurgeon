"""Naming a subject: the parser routes it, and the refusal stays honest without the extra.

The model itself is an optional dependency, so the tests that need it skip; the routing and
the refusal must hold either way, since those are what a caller without torch meets.
"""
import numpy as np
import pytest
from PIL import Image

from figsurgeon import grounding
from figsurgeon.photo_describe import parse

has_model = grounding.available()
needs_model = pytest.mark.skipif(not has_model,
                                 reason='needs figsurgeon[grounding] (torch + transformers)')


def test_the_refusal_names_the_way_out_when_the_extra_is_absent():
    if has_model:
        pytest.skip('the extra is installed, so these are served rather than refused')
    for text in ('make it black and white except the flowers', 'make the sky pink'):
        with pytest.raises(ValueError) as e:
            parse(text)
        assert 'figsurgeon[grounding]' in str(e.value), str(e.value)


def test_a_missing_model_explains_itself_rather_than_raising_importerror_from_a_dependency():
    if has_model:
        pytest.skip('the extra is installed')
    with pytest.raises(ImportError) as e:
        grounding.mask_for(Image.new('RGB', (16, 16)), 'the sky')
    assert 'figsurgeon[grounding]' in str(e.value)


@needs_model
def test_subject_phrases_route_to_the_grounding_operations():
    fn, kwargs = parse('make it black and white except the flowers')
    assert fn is grounding.isolate_subject
    # The article is passed through as typed: CLIPSeg reads it differently from the bare noun.
    assert kwargs['phrase'] == 'the flowers'

    fn, kwargs = parse('make the sky pink')
    assert fn is grounding.recolour_subject
    assert kwargs['phrase'] == 'the sky'
    assert kwargs['to_rgb'] == (230, 130, 180)


def test_whole_picture_words_are_not_treated_as_subjects():
    """"make it blue" is not a request to find an object called "it"."""
    for text in ('make it blue', 'make the colours more vivid', 'make it black and white'):
        try:
            fn, _ = parse(text)
        except ValueError:
            continue
        assert fn not in (grounding.isolate_subject, grounding.recolour_subject), text


@needs_model
def test_a_phrase_that_takes_the_whole_frame_is_refused_rather_than_applied():
    """On a macro where blooms fill the frame, "the flowers" matches almost the whole
    picture, and applying the edit anyway would punch a grey blob through the photograph."""
    from evals import real_corpus
    img = real_corpus.load('flower_macro')
    img.thumbnail((1000, 1000), Image.LANCZOS)
    mask, note = grounding.mask_for(img, 'the flowers')
    assert mask is None
    assert 'selects the picture rather than a part of it' in note


@needs_model
def test_a_real_phrase_selects_a_real_region_and_the_edit_lands_there():
    from evals import real_corpus
    img = real_corpus.load('car_red')
    img.thumbnail((1000, 1000), Image.LANCZOS)
    mask, note = grounding.mask_for(img, 'the red car')
    assert mask is not None, note
    assert 0.002 < float((mask > 0.5).mean()) < 0.1, note      # a car in a street scene

    out, note = grounding.recolour_subject(img, 'the red car', to_rgb=(40, 70, 200))
    a, b = np.asarray(img).astype(int), np.asarray(out).astype(int)
    changed = np.abs(a - b).max(axis=2) > 8
    assert changed.mean() < 0.1, 'the edit escaped the car'
    inside = changed[mask > 0.5].mean()
    assert inside > 0.5, f'only {inside:.0%} of the selected region actually changed'


@needs_model
def test_a_region_running_off_the_frame_is_not_left_cut_off_inside_it():
    """CLIPSeg's probability fades toward the frame border, so a region that continues off
    the edge of the picture comes back cut off inside it; a reflect-padded second pass puts
    that falloff in the padding instead. PASS: the sky reaches the border, recovery noted.
    FAIL: the same recovery must not run on a region that does not touch the border, or a
    small car elsewhere in the frame gets lost by the padded pass."""
    from evals import real_corpus
    from skimage import data
    rocket = Image.fromarray(data.rocket())
    rocket.thumbnail((1000, 1000), Image.LANCZOS)
    mask, note = grounding.mask_for(rocket, 'the sky')
    assert mask is not None, note
    H, W = np.asarray(rocket).shape[:2]
    band = int(0.12 * min(H, W))
    upper_sides = np.zeros((H, W), bool)
    upper_sides[:H // 3, :band] = True
    upper_sides[:H // 3, -band:] = True
    assert float((mask > 0.5)[upper_sides].mean()) > 0.75, 'the border sky is still missing'
    assert 'recovered at the frame edge' in note

    car = real_corpus.load('car_red')
    car.thumbnail((1000, 1000), Image.LANCZOS)
    car_mask, car_note = grounding.mask_for(car, 'the car')
    assert car_mask is not None, car_note
    assert 0.002 < float((car_mask > 0.5).mean()) < 0.1, car_note
    assert 'recovered at the frame edge' not in car_note


@needs_model
def test_the_border_recovery_does_not_disarm_the_whole_frame_refusal():
    """Run over the whole frame, the padded pass reads coverage low enough to let through
    the exact whole-frame edit the refusal exists to stop, so coverage must stay the first
    pass's and only the border band is taken from the second. PASS: the flower macro is
    still refused for the coverage reason, not because the model found nothing."""
    from evals import real_corpus
    img = real_corpus.load('flower_macro')
    img.thumbnail((1000, 1000), Image.LANCZOS)
    mask, note = grounding.mask_for(img, 'the flowers')
    assert mask is None and 'selects the picture rather than a part of it' in note
