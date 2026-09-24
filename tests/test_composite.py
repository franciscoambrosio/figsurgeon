"""`compose_within` keeps a leaky editor's edit inside its region.

Editors here are deliberately leaky, as a generative editor is by construction. The tests
check the composite is byte-identical outside the region, that a leak is reported rather
than swallowed, and that the edit inside the region survives.

The last three checks cover `tools.dispatch`, which enforces the same guarantee on every
localised tool, without altering an already-correct tool or clipping to the bare box when
an operation legitimately reaches further.
"""
import numpy as np
from PIL import Image, ImageFilter
from skimage import data

from figsurgeon import compose_within, objects
from figsurgeon.composite import allowed_region
from figsurgeon.tools import dispatch
from figsurgeon.verify_photo import check_unchanged_outside

CHECKS = []
BOX = (60, 180, 420, 512)          # the astronaut's suit, as in test_objects.py


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def base():
    return Image.fromarray(data.astronaut())


def leaky_editor(img):
    """An 'editor' that blurs the ENTIRE frame -- every pixel changes."""
    return img.filter(ImageFilter.GaussianBlur(4))


def generative_editor(img, box):
    """Stand-in for a latent round trip: a real edit in the box, +-1 noise everywhere."""
    a = np.array(img).astype(int)
    rng = np.random.default_rng(0)
    a += rng.integers(-1, 2, size=a.shape)
    x0, y0, x1, y1 = box
    a[y0:y1, x0:x1] = np.clip(a[y0:y1, x0:x1] * 0.4 + 120, 0, 255)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


@check('a whole-frame editor is caught: byte-identical outside the region')
def t1():
    img = base()
    edited = leaky_editor(img)
    raw_ok, _ = check_unchanged_outside(img, edited, BOX)
    out, rep = compose_within(img, BOX, edited)
    ok, detail = check_unchanged_outside(img, out, BOX)
    return (ok and raw_ok is False), f'raw editor ok={raw_ok}, composited: {detail}'


@check('the leak is reported, not silently swallowed')
def t2():
    img = base()
    _, rep = compose_within(img, BOX, leaky_editor(img))
    W, H = img.size
    outside_px = W * H - (BOX[2] - BOX[0]) * (BOX[3] - BOX[1])
    return (rep['leaked_px'] > 0.5 * outside_px and rep['leaked_max'] > 0,
            f"leaked_px={rep['leaked_px']}/{outside_px}, max={rep['leaked_max']}")


@check('not vacuous: the edit inside the region survives intact')
def t3():
    img = base()
    edited = leaky_editor(img)
    out, _ = compose_within(img, BOX, edited)
    x0, y0, x1, y1 = BOX
    inside_kept = np.array_equal(np.array(out)[y0:y1, x0:x1],
                                 np.array(edited)[y0:y1, x0:x1])
    changed = int((np.array(out) != np.array(img)).any(axis=2).sum())
    return inside_kept and changed > 0, f'inside identical to editor, {changed} px changed'


@check('a generative-style +-1 leak is removed exactly')
def t4():
    img = base()
    edited = generative_editor(img, BOX)
    raw_ok, raw_detail = check_unchanged_outside(img, edited, BOX)
    out, rep = compose_within(img, BOX, edited)
    ok, _ = check_unchanged_outside(img, out, BOX)
    return (ok and raw_ok is False and rep['leaked_max'] == 1,
            f'raw: {raw_detail}; composited leak removed, max was {rep["leaked_max"]}')


@check('a crop-in/crop-out editor needs no full-frame round trip')
def t5():
    img = base()
    x0, y0, x1, y1 = BOX
    crop = leaky_editor(img.crop(BOX))
    out, rep = compose_within(img, BOX, crop)
    ok, _ = check_unchanged_outside(img, out, BOX)
    inside = np.array_equal(np.array(out)[y0:y1, x0:x1], np.array(crop))
    return (ok and inside and rep['leaked_px'] == 0,
            f'{crop.size[0]}x{crop.size[1]} crop pasted, leaked_px=0')


@check('a region running past the image edge still composites (no silent no-op)')
def t6():
    img = base()
    region = (-8, -8, 200, 200)
    out, rep = compose_within(img, region, leaky_editor(img))
    changed = int((np.array(out) != np.array(img)).any(axis=2).sum())
    ok, _ = check_unchanged_outside(img, out, rep['region'])
    return (ok and rep['region'] == (0, 0, 200, 200) and changed > 1000,
            f"clamped to {rep['region']}, {changed} px changed")


@check('an empty region raises rather than returning the original unchanged')
def t7():
    img = base()
    for region in ((100, 100, 100, 200), (600, 10, 700, 90)):
        try:
            compose_within(img, region, leaky_editor(img))
            return False, f'no exception for region {region}'
        except ValueError:
            pass
    return True, 'ValueError for both an empty and an off-image region'


@check('an `edited` of the wrong size is rejected, not pasted at the wrong offset')
def t8():
    img = base()
    try:
        compose_within(img, BOX, leaky_editor(img).resize((320, 320)))
        return False, 'no exception'
    except ValueError as e:
        return True, str(e)[:60]


@check('mode is preserved: an RGBA image stays RGBA, alpha included')
def t9():
    img = base().convert('RGBA')
    a = np.array(img)
    a[:, :, 3] = 200
    img = Image.fromarray(a)
    edited = img.copy()
    e = np.array(edited)
    e[:, :, 3] = 40                       # the editor also wrecks alpha everywhere
    out, rep = compose_within(img, BOX, Image.fromarray(e))
    o = np.array(out)
    x0, y0, x1, y1 = BOX
    outside_alpha = o[:y0, :, 3]
    return (out.mode == 'RGBA' and (outside_alpha == 200).all()
            and (o[y0:y1, x0:x1, 3] == 40).all() and rep['leaked_px'] > 0,
            'alpha kept outside, taken from the editor inside')


@check('dispatch REVERTS a localised tool that leaks, and says so in the note')
def t10():
    img = base()
    box = [200, 300, 320, 420]
    real = objects.recolour_object

    def leaky(*a, **k):
        out, coverage = real(*a, **k)
        return leaky_editor(out), coverage        # a tool that blurs the whole frame too

    objects.recolour_object = leaky
    try:
        out, note = dispatch('recolour_object', {'box': box, 'to_rgb': [30, 60, 180]}, img)
    finally:
        objects.recolour_object = real
    ok, detail = check_unchanged_outside(img, out, box)
    return (ok and 'reverted' in note,
            f'{detail}; note says: {note.split("|")[-1].strip()[:60]}')


@check('enforcement is a no-op on a tool that already honours its box')
def t11():
    img = base()
    box = [200, 300, 320, 420]
    out, note = dispatch('recolour_object', {'box': box, 'to_rgb': [30, 60, 180]}, img)
    direct, _ = objects.recolour_object(img, box=tuple(box), to_rgb=(30, 60, 180))
    same = np.array_equal(np.array(out), np.array(direct))
    return same and 'reverted' not in note, f'identical to the unenforced result: {same}'


@check("the enforced region is the operation's own reach, not the bare box")
def t12():
    # erase_object's mask dilation and inpaint radius extend past the bare box on purpose.
    bare = allowed_region('recolour_object', {'box': [10, 10, 50, 50]})
    erase = allowed_region('erase_object', {'box': [10, 10, 50, 50]})
    none_for_global = allowed_region('isolate_object', {'box': [10, 10, 50, 50]})
    return (bare == (10, 10, 50, 50) and erase == (-1, -1, 61, 61)
            and none_for_global is None,
            f'recolour {bare}, erase {erase}, isolate_object not localised')


@check("scale_object's EDGE_SLACK estimates a GROW's reach from box and scale, not a constant")
def t12b():
    # Covers the fallback path for a caller with no region of its own for scale_object.
    box = [100, 100, 200, 200]     # 100x100 box
    shrink = allowed_region('scale_object', {'box': box, 'scale': 0.5})
    grow = allowed_region('scale_object', {'box': box, 'scale': 2.0})
    return (shrink == (89, 89, 211, 211) and grow == (39, 39, 261, 261),
            f'shrink {shrink}, grow {grow}')


@check('a segmentation-backed edit reports containment without claiming the mask was right')
def t_segmentation_verdict_is_open():
    """Containment is not identity, so a segmentation-backed edit reports containment
    without a PASS verdict; a non-segmentation tool (`remove_object`) still gets one."""
    from figsurgeon.workspace import verify
    img = base()
    # A no-leak, SHADED stand-in, since a flat colour fails check_object_recoloured's own
    # flattening clause.
    a = np.array(img).astype(float)
    block = a[200:300, 100:200]
    scale = np.clip(block.mean(axis=2) / 255.0 / (140 / 255.0), 0.35, 1.9)
    a[200:300, 100:200] = np.clip(np.array([40., 80, 200]) * scale[:, :, None], 0, 255)
    edited = Image.fromarray(a.astype(np.uint8))
    seg_ok, seg_detail = verify('recolour_object', {'box': [90, 190, 210, 310]}, img, edited)
    plain_ok, _ = verify('remove_object', {'box': [90, 190, 210, 310]}, img, edited)
    return (seg_ok is None and plain_ok is True and 'is not checked' in seg_detail,
            f'recolour_object -> {seg_ok}, remove_object -> {plain_ok}')


@check('a segmentation-backed edit that LEAKS still fails outright')
def t_segmentation_leak_still_fails():
    """Leaving the verdict open must not swallow the containment guarantee itself."""
    from figsurgeon.workspace import verify
    img = base()
    leaked = leaky_editor(img)
    ok, detail = verify('recolour_object', {'box': [90, 190, 210, 310]}, img, leaked)
    return ok is False, detail


def run():
    ok = 0
    for name, fn in CHECKS:
        try:
            passed, detail = fn()
        except Exception as e:
            passed, detail = False, f'{type(e).__name__}: {e}'
        print(f'  {"PASS" if passed else "FAIL"}  {name}  [{detail}]')
        ok += bool(passed)
    print(f'\n{ok}/{len(CHECKS)} compose_within checks passed')
    return ok == len(CHECKS)


if __name__ == '__main__':
    import sys
    sys.exit(0 if run() else 1)


def test_compose_within_checks():
    assert run(), 'see printed output for which compose_within check failed'
