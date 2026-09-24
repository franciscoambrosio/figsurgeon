"""Real requests, on real photographs, at the size they actually arrive.

The other evals run on small, uncompressed scikit-image samples. This one runs the same
requests on 0.2-22 MP JPEGs/PNGs from Wikimedia Commons, so it can catch what those miss:
timing at native resolution (a 40s call is unusable interactively), real subjects (hair,
umbrella spokes, shadows, same-hue backgrounds), and scale-dependent pixel defaults that
behave differently at 451px vs. 3840px.

Run:  python evals/real_world.py [outdir]      (downloads the corpus on first use)
"""
import os
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals import real_corpus                                            # noqa: E402
from figsurgeon import ImageWorkspace                                   # noqa: E402
from figsurgeon.locate import _font                                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# An interactive editing loop means a person or a model waiting on each call. Anything past
# this is worth knowing about even when the result is correct.
SLOW_SECONDS = 20.0


# --------------------------------------------------------------------------- assertions
def _sat(a):
    mx, mn = a.max(axis=2), a.min(axis=2)
    return (mx - mn) / np.maximum(mx, 1e-6)


def _region(img, frac_box):
    """Sample a region given as fractions of the image, so it survives any resolution."""
    a = np.array(img.convert('RGB')).astype(float)
    h, w = a.shape[:2]
    x0, y0, x1, y1 = frac_box
    return a[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def car_turned_blue(before, after):
    """The car body must go blue while the rest of the street does not.

    Sampled as fractions of the frame so the regions survive any resolution. The sampled
    region must not overlap anything a wrong box could plausibly recolour instead (e.g. the
    shopfront), or a pass would not mean the car was ever touched.
    """
    car = (0.48, 0.90, 0.62, 0.96)          # inside the car body, read off show_grid
    bld = (0.10, 0.30, 0.40, 0.50)          # upper facade: the control
    car_b, car_a = _region(before, car), _region(after, car)
    bld_b, bld_a = _region(before, bld), _region(after, bld)
    car_shift = float(car_a.mean(axis=(0, 1))[2] - car_b.mean(axis=(0, 1))[2])
    bld_shift = float(np.abs(bld_a - bld_b).mean())
    ok = car_shift > 25 and bld_shift < 12
    return ok, (f'car blue channel +{car_shift:.0f}, building moved {bld_shift:.1f} '
                f'(the building must stay put)')


def subject_kept_and_background_gone(before, after):
    """A cutout of a person: the face must survive, the backdrop must not."""
    if after.mode != 'RGBA':
        return False, f'expected RGBA, got {after.mode}'
    alpha = np.array(after)[..., 3].astype(float) / 255
    h, w = alpha.shape
    face = alpha[int(0.16 * h):int(0.30 * h), int(0.44 * w):int(0.62 * w)].mean()
    # Top-left, not bottom-left: her knee reaches into that corner. Confirmed by looking
    # that top-left is backdrop.
    backdrop = alpha[:int(0.12 * h), :int(0.18 * w)].mean()
    ok = face > 0.85 and backdrop < 0.25
    return ok, f'face kept {face:.0%} opaque, backdrop corner {backdrop:.0%} opaque'


def night_is_cleaner(before, after):
    """A night photo rescue: brighter, and the sky grain must not grow with it."""
    from scipy.ndimage import gaussian_filter
    sky_b = _region(before, (0.0, 0.0, 1.0, 0.18)).mean(axis=2)
    sky_a = _region(after, (0.0, 0.0, 1.0, 0.18)).mean(axis=2)
    n_b = float(np.abs(sky_b - gaussian_filter(sky_b, 2)).mean())
    n_a = float(np.abs(sky_a - gaussian_filter(sky_a, 2)).mean())
    ok = sky_a.mean() > sky_b.mean() and n_a < n_b
    return ok, (f'sky brightness {sky_b.mean():.0f} -> {sky_a.mean():.0f}, '
                f'sky grain {n_b:.2f} -> {n_a:.2f}')


def mask_is_fragmented(before, after):
    """Confirms the hue window shredded one continuous bloom into disconnected pieces.
    Measured on the output (greyed holes), not the mask, since that's what the user gets.
    """
    from scipy import ndimage
    a = np.array(after.convert('RGB')).astype(float)
    greyed = _sat(a) < 0.15
    labels, n = ndimage.label(greyed)
    frac = float(greyed.mean())
    ok = n > 100 and 0.02 < frac < 0.6
    return ok, (f'{frac:.0%} of the bloom greyed out in {n} disconnected patches '
                f'(a continuous object cut by a hue window)')


def is_size(w, h):
    def check(before, after):
        return after.size == (w, h), f'{after.size[0]}x{after.size[1]} (wanted {w}x{h})'
    return check


def aspect_is(ratio, tol=0.02):
    def check(before, after):
        got = after.size[0] / after.size[1]
        return abs(got - ratio) < tol, f'aspect {got:.3f} (wanted {ratio:.3f})'
    return check


def text_was_drawn(before, after):
    """A caption must be legible in the finished image. Measured on the output alone: this
    workflow crops/resizes before drawing, so before/after have different shapes.
    """
    a = np.array(after.convert('RGB')).astype(float)
    h, w = a.shape[:2]
    band = a[int(0.02 * h):int(0.20 * h), int(0.02 * w):int(0.75 * w)]
    ink = float((band.min(axis=2) > 235).mean())
    return ink > 0.005, f'{ink:.1%} of the caption band is near-white text'


def unchanged(before, after):
    same = np.array_equal(np.array(before), np.array(after))
    return same, 'byte-identical' if same else 'differs'


# --------------------------------------------------------------------------- scenarios
# (id, image, request, steps, assertion[, expected_warning])
#
# Written against what is actually in each photograph, checked by looking at it.
#
# `expected_warning` marks a scenario where a warning is the correct outcome, so a real
# limitation being reported does not read as a failure.
SCENARIOS = [
    # ---- things people ask for constantly -----------------------------------------
    ('menu_photo', 'food_pizza',
     'brighten this pizza photo and crop it square for the menu',
     [('adjust', {'brightness': 1.15, 'saturation': 1.15}),
      ('crop', {'box': [480, 0, 3360, 2880]}),
      ('resize', {'width': 1200})],
     is_size(1200, 1200)),

    ('product_cutout', 'product_white',
     'cut these shoes out for the shop listing',
     [('remove_background', {})],
     lambda b, a: (a.mode == 'RGBA' and 0.1 < (np.array(a)[..., 3] < 40).mean() < 0.9,
                   f'{(np.array(a)[..., 3] < 40).mean():.0%} of the frame is transparent'
                   if a.mode == 'RGBA' else f'mode {a.mode}')),

    ('shoe_recolour', 'product_shoe',
     'show these red shoes in blue',
     [('replace_colour', {'from_rgb': [175, 30, 40], 'to_rgb': [30, 70, 190]})],
     None),

    # The box must cover the car, not a nearby shopfront window, or a wrongly-recoloured
    # neighbour could pass the assertion. Read off show_grid, confirmed with
    # preview_object_mask.
    ('car_to_blue', 'car_red',
     'make the red car blue',
     [('recolour_object', {'box': [1680, 2520, 2600, 2830], 'to_rgb': [30, 70, 190]})],
     car_turned_blue),

    ('portrait_cutout', 'portrait_studio',
     'cut her out of the studio backdrop',
     [('remove_background', {})],
     subject_kept_and_background_gone),

    ('portrait_bokeh', 'portrait_studio',
     'blur the backdrop behind her',
     [('blur_background', {'focus': 'subject', 'radius': 18})],
     None),

    ('night_rescue', 'night_city',
     'this night shot is noisy -- clean it up and lift the shadows',
     [('denoise', {'strength': 12}),
      ('adjust', {'brightness': 1.3})],
     night_is_cleaner),

    # The bloom's hue gradient (red->orange->yellow across the same petals) means no
    # hue_tolerance isolates it cleanly -- 0.07 fragments the flower, 0.20 matches
    # everything. Mask statistics don't catch this; preview_colour_mask does.
    ('flower_hue_gradient', 'flower_macro',
     'colour pop the red petals -- a hue window fragments a continuous bloom',
     [('isolate_colour', {'target_rgb': [200, 40, 30], 'hue_tolerance': 0.07})],
     lambda b, a: mask_is_fragmented(b, a)),

    # Already near-maximum saturation (0.92), so boosting it clips rather than saturates;
    # the tool should say so instead of claiming a change it didn't make.
    ('flower_for_print', 'flower_macro',
     'make this flower photo pop for a print (it is already near maximum saturation)',
     [('adjust', {'saturation': 1.25, 'contrast': 1.1})],
     None, 'saturation barely moved'),

    ('banner_crop', 'city_wide',
     'crop this skyline to a 16:9 banner and boost the contrast',
     [('crop', {'box': [0, 1600, 3840, 3760]}),
      ('adjust', {'contrast': 1.2})],
     aspect_is(16 / 9)),

    ('social_caption', 'city_wide',
     'crop it square and put a caption on it',
     [('crop', {'box': [0, 1200, 3840, 5040]}),
      ('resize', {'width': 1080}),
      ('add_text', {'text': 'NEW YORK', 'position': [60, 60], 'size': 90,
                    'colour_rgb': [255, 255, 255]})],
     text_was_drawn),

    ('street_cleanup', 'street_people',
     'remove the air-conditioning unit on the right',
     [('erase_object', {'box': [3300, 1980, 3800, 2450]})],
     None),

    ('screenshot_for_slide', 'ui_screenshot',
     'crop the window out of the wallpaper for a slide',
     [('crop', {'box': [250, 190, 1690, 1130]})],
     is_size(1440, 940)),

    # ---- real charts --------------------------------------------------------------
    ('chart_rebrand', 'chart_bar',
     're-theme this published chart to our brand blues',
     [('replace_colour', {'from_rgb': [61, 102, 156], 'to_rgb': [0, 64, 122]})],
     None),

    ('chart_crop_plot', 'chart_health',
     'crop out just the plot area of this chart',
     [('crop', {'box': [40, 30, 590, 330]})],
     is_size(550, 300)),

    # ---- the ones that should NOT work, on real input ------------------------------
    # A street scene has no single subject -- on the sample images this was a synthetic
    # texture; here it is a real photograph.
    ('street_no_subject', 'street_people',
     'blur the background of a street scene (there is no one subject)',
     [('blur_background', {'focus': 'subject', 'radius': 20})],
     None),

    # Her umbrella is blue and the backdrop behind it is blue. No hue tolerance separates
    # them -- this is the real-photograph version of the synthetic same-hue case.
    ('umbrella_hue_trap', 'portrait_studio',
     'recolour the blue umbrella (it shares its hue with the backdrop)',
     [('replace_colour', {'from_rgb': [40, 90, 160], 'to_rgb': [180, 60, 30]})],
     None),
]


# --------------------------------------------------------------------------- running
def sheet(before, after, title, verdict, timing, width=1400):
    panel = width // 2
    h = max(before.size[1] / before.size[0], after.size[1] / after.size[0])
    cell = (panel - 16, int(panel * min(h, 1.4)))

    def thumb(img):
        if img.mode == 'RGBA':
            board = Image.new('RGB', img.size, (255, 255, 255))
            d0 = ImageDraw.Draw(board)
            step = max(8, img.size[0] // 60)
            for yy in range(0, img.size[1], step):
                for xx in range(0, img.size[0], step):
                    if (xx // step + yy // step) % 2:
                        d0.rectangle([xx, yy, xx + step - 1, yy + step - 1],
                                     fill=(214, 216, 220))
            board.paste(img, (0, 0), img)
            img = board
        t = img.convert('RGB').copy()
        t.thumbnail(cell, Image.LANCZOS)
        return t

    tb, ta = thumb(before), thumb(after)
    H = max(tb.size[1], ta.size[1]) + 84
    out = Image.new('RGB', (width, H), (246, 246, 248))
    out.paste(tb, ((panel - tb.size[0]) // 2, 74))
    out.paste(ta, (panel + (panel - ta.size[0]) // 2, 74))
    d = ImageDraw.Draw(out)
    d.text((10, 8), title[:130], font=_font(19), fill=(10, 10, 10))
    colour = (0, 120, 0) if verdict.startswith('OK') else (190, 0, 0)
    d.text((10, 32), verdict[:150], font=_font(13), fill=colour)
    d.text((10, 52), timing[:150], font=_font(12), fill=(90, 90, 90))
    return out


def run(outdir):
    os.makedirs(outdir, exist_ok=True)
    cache, rows = {}, []

    for scenario in SCENARIOS:
        sid, image_name, intent, steps, assertion = scenario[:5]
        expected_warning = scenario[5] if len(scenario) > 5 else None
        if image_name not in cache:
            cache[image_name] = real_corpus.load(image_name)
        img = cache[image_name]
        mp = img.size[0] * img.size[1] / 1e6

        ws = ImageWorkspace(img)
        problems, timings, warnings_seen = [], [], []
        try:
            for name, args in steps:
                t0 = time.time()
                result = ws.apply(name, args)
                dt = time.time() - t0
                timings.append((name, dt))
                if dt > SLOW_SECONDS:
                    problems.append(f'{name} took {dt:.0f}s on {mp:.1f} MP')
                if 'CHECK FAILED' in result.note:
                    problems.append(f'{name} check failed: '
                                    f'{result.note.split("CHECK FAILED:")[1][:70]}')
                elif result.note.startswith('error'):
                    problems.append(f'{name}: {result.note[:90]}')
                elif 'WARNING' in result.note:
                    warning = result.note[result.note.index('WARNING'):]
                    if expected_warning and expected_warning in warning:
                        warnings_seen.append(warning[:80])
                    else:
                        problems.append(f'{name}: {warning[:90]}')
        except Exception as e:
            problems.append(f'RAISED {type(e).__name__}: {e}')
            traceback.print_exc()

        detail = ''
        if assertion is not None and not any(p.startswith('RAISED') for p in problems):
            ok, detail = assertion(img, ws.image)
            if not ok:
                problems.append(f'outcome: {detail}')
        if expected_warning and not warnings_seen:
            problems.append(f'expected a warning mentioning {expected_warning!r}, got none')
        elif warnings_seen:
            detail = (detail + ' | ' if detail else '') + 'warned as expected: ' + warnings_seen[0]

        total = sum(t for _, t in timings)
        timing = (f'{mp:.1f} MP, {total:.1f}s total: '
                  + ', '.join(f'{n} {t:.1f}s' for n, t in timings))
        status = 'OK' if not problems else 'PROBLEM'
        verdict = f'{status}: {detail}' if not problems else f'PROBLEM: {"; ".join(problems)}'
        sheet(img, ws.image, f'{sid} -- {intent}', verdict, timing).save(
            os.path.join(outdir, f'{sid}.png'))
        rows.append((sid, status, verdict, total, mp))

        print(f'{status:8s} {sid:22s} {mp:5.1f}MP {total:6.1f}s  {detail[:56]}')
        for p in problems:
            print(f'{"":8s} {"":22s}                -> {p[:100]}')

    bad = [r for r in rows if r[1] != 'OK']
    slow = sorted(rows, key=lambda r: -r[3])[:3]
    print(f'\n{len(rows) - len(bad)}/{len(rows)} real-world scenarios behaved; '
          f'sheets in {outdir}')
    print('slowest: ' + ', '.join(f'{r[0]} {r[3]:.1f}s ({r[4]:.1f} MP)' for r in slow))
    return rows


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'eval_out_real'))
