"""The plain-language path (`apply_text`, `ImageSession.do`) must return the same verdict
as the tool path (`ImageWorkspace`), not just an image with no correctness signal.

Every case below is paired: a correct edit that must be VERIFIED, and a broken one on the
same image that must FAIL.
"""
import numpy as np
from PIL import Image
from skimage import data

from figsurgeon import photo
from figsurgeon.photo_describe import apply_text, parse
from figsurgeon.session import ImageSession
from figsurgeon.workspace import as_tool_call, verify, verify_call


def astro():
    return Image.fromarray(data.astronaut())


def test_correct_edits_come_back_verified():
    img = astro()
    for text in ('make it black and white', 'brighten it', 'increase the contrast',
                 'add a vignette', 'colour pop the red'):
        apply_text(img, text)
        ok, detail = apply_text.last_check
        assert ok is True, f'{text!r} was correct but reported {ok}: {detail}'


def test_an_edit_that_missed_comes_back_failed():
    """The same instruction, forced to do nothing, must not report success."""
    img = astro()
    fn, kwargs = parse('make it black and white')
    ok, detail = verify_call(fn, kwargs, img, img)          # the tool silently no-opped
    assert ok is False, detail

    fn, kwargs = parse('brighten it')
    ok, detail = verify_call(fn, kwargs, img, photo.adjust(img, brightness=0.8))
    assert ok is False, detail                              # darker, not brighter


def test_operations_with_no_check_say_so_rather_than_claiming_success():
    img = astro()
    for text in ('sepia tone', 'turn it into a pencil sketch', 'mirror it'):
        apply_text(img, text)
        ok, detail = apply_text.last_check
        assert ok is None, f'{text!r} has no check but reported {ok}'
        assert 'no automatic check' in detail, detail


def test_the_two_front_ends_cannot_disagree():
    """`as_tool_call` is the only translation, so the text path reaches the SAME check."""
    img = astro()
    fn, kwargs = parse('increase the contrast')
    out = fn(img, **kwargs)
    name, args = as_tool_call(fn, kwargs)
    assert (name, args) == ('adjust', {'contrast': 1.4})
    assert verify_call(fn, kwargs, img, out) == verify(name, args, img, out)


def test_a_session_records_the_verdict_with_the_step():
    s = ImageSession(astro())
    s.do('make it black and white')
    assert s.last_check[0] is True
    assert 'verified' in s.transcript()[-1][1]

    s.do('add a vignette')
    assert s.last_check[0] is True
    assert len(s.transcript()) == 2


def test_a_session_verifies_the_result_before_it_is_flattened():
    """The session flattens RGBA to RGB to keep chaining; verification must happen before
    that, or `check_background_removed` measures an image its own premise already destroyed."""
    from figsurgeon import advanced
    cutout = Image.fromarray(
        np.dstack([np.asarray(astro()), np.full((512, 512), 255, np.uint8)]))
    cutout.putalpha(Image.fromarray(
        np.where(np.indices((512, 512))[1] < 256, 255, 0).astype(np.uint8)))
    ok, detail = verify_call(advanced.remove_background, {}, astro(), cutout)
    assert ok is True, detail
    flattened = Image.new('RGB', cutout.size, (255, 255, 255))
    flattened.paste(cutout, (0, 0), cutout)
    assert verify_call(advanced.remove_background, {}, astro(), flattened)[0] is False


def test_verification_can_be_turned_off():
    img = astro()
    apply_text(img, 'make it black and white', verify=False)
    assert apply_text.last_check[0] is None
    s = ImageSession(img, verify=False)
    s.do('make it black and white')
    assert s.last_check[0] is None


def test_a_second_colour_pop_is_not_a_failure_just_because_the_first_worked():
    """Grading whatever off-hue pixels remain after a second colour-pop, however few, must
    not report CHECK FAILED on work that did exactly what was asked -- below an evidence
    floor there is no verdict to give."""
    s = ImageSession(astro())
    s.do('colour pop the red')
    assert s.last_check[0] is True, s.last_check
    fn, kwargs = parse('colour pop the red')
    once = fn(astro(), **kwargs)[0]
    twice = fn(fn(once, **dict(kwargs, flatten=1.0))[0], **dict(kwargs, flatten=1.0))[0]
    ok, detail = verify_call(fn, kwargs, once, twice)
    assert ok is not False, detail
    assert 'not enough to tell' in detail, detail


def test_the_evidence_floor_does_not_silence_a_real_failure():
    """The other half: an image with plenty of off-hue colour still gets a verdict, and a
    colour pop that greyed nothing must still fail."""
    img = astro()
    fn, kwargs = parse('colour pop the red')
    assert verify_call(fn, kwargs, img, img)[0] is False
    ok, detail = verify_call(fn, kwargs, img, photo.isolate_colour(img, (220, 30, 30))[0])
    assert ok is True, detail
