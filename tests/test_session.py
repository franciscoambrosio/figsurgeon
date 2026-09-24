"""Conversation-level tests: state, refinement direction, undo, chaining.

These check properties the stateless tests structurally cannot:
  * "increase the contrast" then "less" must lower it, not raise it further -- re-applying a
    gentler edit on top of the boosted image would still boost. "less" replaces the last edit.
  * "make it black and white and add a vignette" must keep both operations: splitting on
    "and" without protecting operation names would cut inside "black and white".
"""
import os

import numpy as np
from PIL import Image
from skimage import data

from figsurgeon.session import ImageSession, split_instructions

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

contrast = lambda i: float(np.array(i).astype(float).std())
bright = lambda i: float(np.array(i).astype(float).mean())
satur = lambda i: float((lambda a: ((a.max(axis=2) - a.min(axis=2)) /
                                    np.maximum(a.max(axis=2), 1e-6)).mean())
                        (np.array(i).astype(float)))
corner = lambda i: float(np.array(i).astype(float).mean(axis=2)[:40, :40].mean())

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@check('"more" escalates in the user\'s original direction (down)')
def t1():
    s = ImageSession(Image.fromarray(data.coffee()))
    s.do('reduce the contrast'); a = contrast(s.image)
    s.do('more'); b = contrast(s.image)
    return b < a, f'{a:.1f} -> {b:.1f}'


@check('"more" escalates upward when the user went up')
def t2():
    s = ImageSession(Image.fromarray(data.coffee()))
    s.do('increase the contrast'); a = contrast(s.image)
    s.do('more'); b = contrast(s.image)
    return b > a, f'{a:.1f} -> {b:.1f}'


@check('"less" walks BACK (regression: it used to push further)')
def t3():
    s = ImageSession(Image.fromarray(data.coffee()))
    s.do('increase the contrast'); a = contrast(s.image)
    s.do('less'); b = contrast(s.image)
    return b < a, f'{a:.1f} -> {b:.1f}'


@check('undo restores the exact previous pixels')
def t4():
    s = ImageSession(Image.fromarray(data.astronaut()))
    before = np.array(s.image)
    s.do('add a vignette')
    s.undo()
    return np.array_equal(np.array(s.image), before), 'byte-identical' \
        if np.array_equal(np.array(s.image), before) else 'differs'


@check('reset returns to the original after several edits')
def t5():
    s = ImageSession(Image.fromarray(data.astronaut()))
    before = np.array(s.image)
    s.do('add a vignette'); s.do('increase the contrast'); s.do('make it black and white')
    s.do('reset')
    return np.array_equal(np.array(s.image), before), 'restored'


@check('compound instruction applies BOTH operations')
def t6():
    s = ImageSession(Image.fromarray(data.astronaut()))
    c0 = corner(s.image)
    s.do('make it black and white and add a vignette')
    got_bw = satur(s.image) < 0.02
    got_vig = corner(s.image) < c0 * 0.8
    return got_bw and got_vig, f'grayscale={got_bw}, vignette={got_vig}'


@check('"black and white" is not split on its internal "and"')
def t7():
    parts = split_instructions('make it black and white and add a vignette')
    return parts == ['make it black and white', 'add a vignette'], str(parts)


@check('refinement before any edit raises a clear error')
def t8():
    s = ImageSession(Image.fromarray(data.coffee()))
    try:
        s.do('more')
        return False, 'no error raised'
    except ValueError as e:
        return 'no edit has been made' in str(e), str(e)[:60]


@check('"more" escalates isolate_colour (flatten), not a silent no-op')
def t10():
    s = ImageSession(Image.fromarray(data.astronaut()))
    s.do('colour pop the red')
    sat0 = satur(s.image)
    s.do('more')
    sat1 = satur(s.image)
    return sat1 < sat0, f'saturation {sat0:.4f} -> {sat1:.4f} (flatten should increase, sat should drop)'


@check('"less" after "more" on isolate_colour walks flatten back down')
def t11():
    s = ImageSession(Image.fromarray(data.astronaut()))
    s.do('colour pop the red')
    s.do('more')
    _, kw_more = s._last_call
    s.do('less')
    _, kw_less = s._last_call
    return kw_less['flatten'] < kw_more['flatten'], f"{kw_more['flatten']:.3f} -> {kw_less['flatten']:.3f}"


@check('"more" escalates replace_colour (hue_tol widens)')
def t12():
    s = ImageSession(Image.fromarray(data.astronaut()))
    s.do('replace the red with orange')
    _, kw0 = s._last_call
    s.do('more')
    _, kw1 = s._last_call
    return kw1['hue_tol'] > kw0.get('hue_tol', 0.07), f"{kw0.get('hue_tol',0.07):.3f} -> {kw1['hue_tol']:.3f}"


@check('edits chain: state carries across turns')
def t9():
    s = ImageSession(Image.fromarray(data.astronaut()))
    s.do('make it black and white')
    s.do('add a strong vignette')
    return satur(s.image) < 0.02 and corner(s.image) < 60, \
        f'sat={satur(s.image):.3f}, corner={corner(s.image):.0f}'


def run():
    ok = 0
    for name, fn in CHECKS:
        try:
            passed, detail = fn()
        except Exception as e:
            passed, detail = False, f'{type(e).__name__}: {e}'
        print(f'  {"PASS" if passed else "FAIL"}  {name}  [{detail}]')
        ok += bool(passed)
    print(f'\n{ok}/{len(CHECKS)} conversation checks passed')
    return ok == len(CHECKS)


if __name__ == '__main__':
    import sys
    sys.exit(0 if run() else 1)


def test_conversation_checks():
    assert run(), 'see printed output for which conversation check failed'


def test_more_strengthens_the_last_edit_rather_than_stacking_a_second_one():
    """"more" must re-apply the last edit at a stronger setting from the image before that
    edit, not stack the stronger edit on top of the already-edited one."""
    from figsurgeon import photo
    img = Image.open(os.path.join(ROOT, 'evals', '_corpus', 'car_red.jpg')).convert('RGB')
    img = img.resize((400, 300))
    s = ImageSession(img)
    s.do('add a vignette')
    s.do('more')
    strength = s._last_call[1]['strength']
    direct = photo.vignette(img, strength=strength)
    assert np.array_equal(np.array(s.image), np.array(direct)), \
        f'"more" should equal one vignette at strength {strength}'
    s.do('undo')
    assert np.array_equal(np.array(s.image), np.array(img)), \
        'undo after "more" should return to the unedited picture, not a weaker vignette'


def test_the_word_original_in_an_edit_does_not_wipe_the_session():
    """_RESET must not match "original" as a substring of an ordinary instruction, or "boost
    the contrast but keep the original colours" would be treated as a reset and silently
    throw away every earlier edit."""
    s = ImageSession(Image.fromarray(data.astronaut()))
    s.do('make it brighter')
    s.do('boost the contrast but keep the original colours')
    assert len(s.history) == 2, s.log
    s.do('go back to the original')
    assert len(s.history) == 0, s.log
