"""Quality eval: realistic edits across a deliberately diverse set of images.

Not a test suite. Tests answer "did the code do what it did last time"; this answers "is the
result any good", which needs looking at pictures. Every scenario writes a before/after pair
and a contact sheet, with measured verdicts printed alongside them.

Groups are chosen against the package's actual failure modes, each with at least one hard
case: portrait (shared-hue subject/background), animal (fur matching the wall), still life
(large-object erase), outdoor (thin object against sky), scientific (faint colour),
greyscale, chart (flat colours/text/legend), and synthetic (same-hue by construction).

Run:  python evals/edit_quality.py [outdir]
"""
import os
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figsurgeon import ImageWorkspace                                   # noqa: E402
from figsurgeon.locate import _font                                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- images
def _skimage(name):
    from skimage import data
    a = getattr(data, name)()
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[2] == 4:
        a = a[:, :, :3]
    return Image.fromarray(a.astype('uint8'))


def _repo(filename):
    path = os.path.join(ROOT, filename)
    if not os.path.exists(path):
        return None
    return Image.open(path).convert('RGB')


def _same_hue_subject():
    """A subject that hue matching provably cannot separate from its background: same hue,
    differing only in lightness, so `isolate_colour` has nothing to key on -- the correct
    tool here is the spatial one.
    """
    img = Image.new('RGB', (320, 240), (60, 130, 60))
    d = ImageDraw.Draw(img)
    d.ellipse([110, 70, 210, 170], fill=(95, 175, 95))
    d.rectangle([0, 200, 320, 240], fill=(40, 95, 40))
    return img


IMAGES = {
    'astronaut': lambda: _skimage('astronaut'),
    'chelsea': lambda: _skimage('chelsea'),
    'coffee': lambda: _skimage('coffee'),
    'rocket': lambda: _skimage('rocket'),
    'retina': lambda: _skimage('retina'),
    'ihc': lambda: _skimage('immunohistochemistry'),
    'camera': lambda: _skimage('camera'),
    'moon': lambda: _skimage('moon'),
    'chart': lambda: _repo('tests/data/timeseries_demo.png'),
    'heatmap': lambda: _repo('tests/data/rebrand_source.png'),
    'legend': lambda: _repo('tests/data/legend_source.png'),
    'same_hue': _same_hue_subject,
    # Added after the workflow sweep, to cover shapes of image the earlier roster had none
    # of: a picture with no subject to segment, one that is almost entirely black, flat
    # vector-style graphics, a document, and pure high-frequency texture.
    'grass': lambda: _skimage('grass'),
    'deep_field': lambda: _skimage('hubble_deep_field'),
    'logo': lambda: _skimage('logo'),
    'page': lambda: _skimage('page'),
    'checkerboard': lambda: _skimage('checkerboard'),
    'cells': lambda: _skimage('human_mitosis'),
}


def _damaged_the_plot(before, after):
    """Confirms the wrong-box legend clean wrecked plot data, with every check still green.
    Asserts the damage, not its absence, so this stops meaning anything if it stops
    happening.
    """
    b = np.array(before.convert('RGB')).astype(int)[60:160, 300:520]
    a = np.array(after.convert('RGB')).astype(int)[60:160, 300:520]
    changed = float((np.abs(b - a).max(axis=2) > 8).mean())
    return changed > 0.05, (
        f'{changed:.0%} of the pixels in a region of real plot data were overwritten, and '
        f'no automatic check flagged it')


def _flag_survived(before, after, should_survive=True):
    """Did the American flag's red stripes survive an edit aimed at the orange suit?

    The stripes share the suit's hue but sit far from it, so the spatial tool can tell them
    apart and a hue-based one can't. `should_survive` lets the same assertion pass for both
    `recolour_object` (stripes survive) and `replace_colour` (stripes destroyed).
    """
    b = np.array(before.convert('RGB'))[178:186, 64:74].reshape(-1, 3).mean(axis=0)
    a = np.array(after.convert('RGB'))[178:186, 64:74].reshape(-1, 3).mean(axis=0)
    moved = float(np.abs(a - b).max())
    survived = moved < 12
    return survived == should_survive, (
        f'flag stripe RGB {tuple(int(v) for v in b)} -> {tuple(int(v) for v in a)} '
        f'(max channel move {moved:.0f}; '
        f'{"survived" if survived else "destroyed"}, '
        f'{"as required" if survived == should_survive else "NOT as required"})')


# --------------------------------------------------------------------------- scenarios
# (id, image, plain-language intent, [tool calls], extras). `extras` documents scenarios
# expected to FAIL or be inconclusive (the eval fails if they quietly start passing instead):
#   'expect'  -- 'pass' (default), 'fail', 'inconclusive', or 'any' (exploratory)
#   'warn'    -- a substring that must appear in some result note
#   'assert'  -- fn(before, after) -> (ok, detail), for the claim the verdict cannot express
SCENARIOS = [
    ('portrait_suit_blue', 'astronaut', 'make the orange suit blue, leave the flag alone',
     [('recolour_object', {'box': [60, 180, 420, 512], 'to_rgb': [30, 60, 180]})],
     {'assert': lambda b, a: _flag_survived(b, a, should_survive=True)}),
    # The paired counter-example: the hue-based tool passes its own verification (it did
    # shift the hue given) while wrecking the picture -- no per-operation check can catch a
    # wrong tool choice. The eval fails if the flag ever survives here.
    ('portrait_hue_trap', 'astronaut', 'the WRONG tool for the same job, on purpose',
     [('replace_colour', {'from_rgb': [201, 87, 59], 'to_rgb': [30, 60, 180]})],
     {'assert': lambda b, a: _flag_survived(b, a, should_survive=False)}),
    ('portrait_bg_white', 'astronaut', 'put her on a white background',
     [('replace_background', {'colour_rgb': [255, 255, 255]})]),
    ('portrait_crop', 'astronaut', 'crop to a head-and-shoulders portrait',
     [('crop', {'box': [130, 60, 400, 400]})]),

    ('cat_eyes_pop', 'chelsea', 'colour pop BOTH eyes (two regions, one call)',
     [('isolate_object', {'boxes': [[141, 85, 211, 145], [288, 100, 352, 158]],
                          'flatten': 0.6})]),
    ('cat_bokeh', 'chelsea', 'blur the background behind the cat',
     [('blur_background', {'focus': 'subject', 'radius': 14})]),
    ('cat_sketch', 'chelsea', 'turn it into a pencil sketch',
     [('stylise', {'effect': 'sketch'})]),

    ('coffee_dof', 'coffee', 'shallow depth of field on the cup',
     [('blur_background', {'focus': 'subject', 'radius': 16})]),
    ('coffee_erase_big', 'coffee', 'remove the cup (a large object -- must warn)',
     [('erase_object', {'box': [140, 60, 460, 380]})],
     {'warn': 'of the frame'}),
    ('coffee_wb', 'coffee', 'auto white balance (known bad on a warm-dominated scene)',
     [('auto_white_balance', {})]),

    # The box must hold the mast itself: on a box where GrabCut finds nothing the edit is a
    # no-op, and a no-op satisfies "0 px changed outside the box" trivially.
    ('rocket_erase_mast', 'rocket', 'remove the mast beside the rocket',
     [('erase_object', {'box': [150, 80, 230, 340]})]),
    ('rocket_warm_pop', 'rocket', 'colour pop the warm lights against the blue sky',
     [('isolate_colour', {'target_rgb': [230, 150, 60], 'hue_tolerance': 0.06})]),

    ('retina_vessels', 'retina', 'emphasise the red vessels (faint, low saturation)',
     [('isolate_colour', {'target_rgb': [150, 60, 50], 'hue_tolerance': 0.06})]),
    ('ihc_stain', 'ihc', 'isolate the brown stain',
     [('isolate_colour', {'target_rgb': [140, 90, 60], 'hue_tolerance': 0.05})]),

    ('grey_saturate', 'camera', 'make the colours pop (image has no colour)',
     [('adjust', {'saturation': 1.8})],
     {'expect': 'fail', 'warn': 'essentially greyscale'}),
    ('grey_contrast', 'moon', 'increase the contrast',
     [('adjust', {'contrast': 1.5})]),

    ('chart_recolour', 'chart', 'recolour a chart region to brand navy',
     [('remap_colormap', {'box': [102, 3, 735, 252], 'new_colours': ['#00407A', '#52BDEC']})]),
    ('heatmap_rebrand', 'heatmap', 'rebrand the heatmap to a navy gradient',
     [('remap_colormap', {'box': [102, 3, 735, 252], 'new_colours': ['#00407A', '#52BDEC']})]),
    # Box read off `show_grid`, not estimated. The first version of this scenario used a
    # box guessed from the plain image and landed on empty plot area 200 px above the
    # legend, punching a rectangular hole in the data -- see legend_wrong_box below.
    ('legend_clean', 'legend', 'clean the data bleeding through the legend box',
     [('clean_transparent_box', {'box': [258, 292, 692, 538]})]),
    # THE FAILURE NO CHECK CATCHES, kept on purpose: the box covers plot area, not the
    # legend, yet every automatic check passes (0 px changed outside the region asked for).
    # Only looking catches this; the assertion here knows where the legend actually is.
    ('legend_wrong_box', 'legend', 'the same call with a box guessed instead of read',
     [('clean_transparent_box', {'box': [300, 60, 520, 160]})],
     {'assert': lambda b, a: _damaged_the_plot(b, a)}),

    ('samehue_wrong_tool', 'same_hue', 'isolate the disc by colour (cannot work)',
     [('isolate_colour', {'target_rgb': [95, 175, 95], 'hue_tolerance': 0.07})],
     {'expect': 'inconclusive'}),
    ('samehue_right_tool', 'same_hue', 'isolate the disc spatially (should work)',
     [('isolate_object', {'box': [95, 55, 225, 185]})]),

    # --- images with nothing for the subject/colour machinery to grab on to --------
    # rembg on a lawn: there is no subject, so whatever mask comes back is arbitrary. The
    # requirement is that this is REPORTED, not that it succeeds.
    ('no_subject_blur', 'grass', 'blur the background of a picture with no subject',
     [('blur_background', {'focus': 'subject', 'radius': 14})],
     {'expect': 'any'}),
    ('mostly_black_pop', 'deep_field', 'colour pop the reddish galaxies in a near-black frame',
     [('isolate_colour', {'target_rgb': [150, 90, 70], 'hue_tolerance': 0.08})],
     {'expect': 'any'}),
    ('flat_graphic_recolour', 'logo', 'recolour a flat vector-style graphic',
     [('replace_colour', {'from_rgb': [220, 30, 30], 'to_rgb': [30, 80, 200]})],
     {'expect': 'any'}),
    ('document_contrast', 'page', 'make the scanned text crisper',
     [('adjust', {'contrast': 1.6})]),
    # High-frequency texture IS the signal here, not noise. Denoising it should either
    # report a small effect or warn -- what it must not do is claim a clean success.
    ('denoise_pure_texture', 'checkerboard', 'denoise an image whose detail is the point',
     [('denoise', {'strength': 10})],
     {'expect': 'any'}),
    ('microscopy_contrast', 'cells', 'boost contrast on low-contrast microscopy',
     [('adjust', {'contrast': 1.5})]),

    ('multi_step', 'chelsea', 'black and white, then a vignette, then sharpen',
     [('stylise', {'effect': 'grayscale'}),
      ('stylise', {'effect': 'vignette', 'strength': 0.5}),
      ('adjust', {'sharpness': 1.6})]),
    ('undo_recovers', 'chelsea', 'a bad edit followed by undo must restore exactly',
     [('adjust', {'brightness': 3.0}), ('undo', {})],
     # 'noop' is the correct outcome, not a shortfall: after the undo there is no edit left
     # in the history to verify. The assertion carries the real claim.
     {'expect': 'noop', 'assert': lambda b, a: (np.array_equal(np.array(b), np.array(a)),
                              'byte-identical to the original' if np.array_equal(
                                  np.array(b), np.array(a)) else 'undo did NOT restore')}),
]


# --------------------------------------------------------------------------- running
def contact_sheet(before, after, title, verdict, width=1100):
    """A before/after pair, captioned -- the artifact a human (or a model) actually judges."""
    h = max(before.size[1] / before.size[0], after.size[1] / after.size[0])
    panel = width // 2
    cell = (panel, int(panel * h))
    sheet = Image.new('RGB', (width, cell[1] + 62), (245, 245, 245))
    for i, im in enumerate((before, after)):
        thumb = im.convert('RGB').copy()
        thumb.thumbnail(cell, Image.LANCZOS)
        sheet.paste(thumb, (i * panel + (panel - thumb.size[0]) // 2, 56))
    d = ImageDraw.Draw(sheet)
    d.text((10, 8), title, font=_font(19), fill=(0, 0, 0))
    colour = {'ok': (0, 120, 0), 'FAILED': (190, 0, 0)}.get(verdict.split()[0], (90, 90, 90))
    d.text((10, 32), verdict[:150], font=_font(14), fill=colour)
    d.text((panel // 2 - 20, cell[1] + 40), 'before', font=_font(14), fill=(60, 60, 60))
    d.text((panel + panel // 2 - 15, cell[1] + 40), 'after', font=_font(14), fill=(60, 60, 60))
    return sheet


def run(outdir):
    os.makedirs(outdir, exist_ok=True)
    cache, rows = {}, []

    for scenario in SCENARIOS:
        sid, image_name, intent, calls = scenario[:4]
        extras = scenario[4] if len(scenario) > 4 else {}
        if image_name not in cache:
            try:
                cache[image_name] = IMAGES[image_name]()
            except Exception as e:
                cache[image_name] = None
                print(f'  ! could not load {image_name}: {e}')
        img = cache[image_name]
        if img is None:
            rows.append((sid, image_name, 'PROBLEM', 'skip', 'image unavailable', 0.0))
            print(f'{"PROBLEM":8s} {"skip":13s} {sid:22s}        image unavailable')
            continue

        ws = ImageWorkspace(img)
        notes, t0 = [], time.time()
        try:
            for name, args in calls:
                notes.append(f'{name}: {ws.apply(name, args).note}')
            error = None
        except Exception as e:
            error = f'{type(e).__name__}: {e}'
            traceback.print_exc()
        elapsed = time.time() - t0

        if error:
            outcome, detail = 'error', error
        else:
            failed = ws.failures()
            unchecked = [s for s in ws.history if s.ok is None]
            if failed:
                outcome, detail = 'fail', f'{failed[0].name}: {failed[0].detail}'
            elif not ws.history:
                outcome, detail = 'noop', 'nothing was recorded as an edit'
            elif unchecked and len(unchecked) == len(ws.history):
                outcome = 'inconclusive'
                detail = ' | '.join(s.detail for s in ws.history if s.detail)
            else:
                outcome = 'pass'
                detail = ' | '.join(s.detail for s in ws.history if s.detail)

        # Compare against what the scenario says SHOULD happen, not against "green is good".
        problems = []
        expected = extras.get('expect', 'pass')
        if outcome == 'error':
            problems.append(f'raised: {error}')
        elif expected == 'any':
            # Exploratory: the image type is one nothing here was designed for, so no
            # particular verdict is required -- only that it does not crash and that the
            # outcome gets recorded for a human to look at.
            pass
        elif outcome != expected and not (expected == 'pass' and outcome == 'inconclusive'):
            problems.append(f'expected {expected}, got {outcome}')
        if extras.get('warn') and not any(extras['warn'] in n for n in notes):
            problems.append(f'expected a note mentioning {extras["warn"]!r}; none did')
        if extras.get('assert') and not error:
            ok, why = extras['assert'](img, ws.image)
            detail = f'{detail} | assertion: {why}' if detail else f'assertion: {why}'
            if not ok:
                problems.append(f'assertion failed: {why}')

        status = 'OK' if not problems else 'PROBLEM'
        sheet = contact_sheet(img, ws.image, f'{sid}  --  {intent}',
                              ('ok' if status == 'OK' else 'FAILED ') +
                              f'[{outcome}] {detail}')
        sheet.save(os.path.join(outdir, f'{sid}.png'))
        rows.append((sid, image_name, status, outcome, '; '.join(problems) or detail, elapsed))

        print(f'{status:8s} {outcome:13s} {sid:22s} {elapsed:5.1f}s  {detail[:80]}')
        for p in problems:
            print(f'{"":8s} {"":13s} {"":22s}        -> {p[:110]}')
        for w in [n for n in notes if 'WARNING' in n]:
            print(f'{"":8s} {"":13s} {"":22s}        warned: {w[:100]}')

    bad = [r for r in rows if r[2] != 'OK']
    print(f'\n{len(rows) - len(bad)}/{len(rows)} scenarios behaved as expected; '
          f'sheets written to {outdir}')
    for r in bad:
        print(f'  PROBLEM {r[0]}: {r[4][:120]}')
    return rows


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'eval_out'))
