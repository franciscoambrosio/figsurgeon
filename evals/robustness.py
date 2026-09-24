"""Robustness sweep: hostile and unusual INPUT, rather than realistic edits.

`edit_quality.py` asks whether a result is good on sensible input; this asks what happens
when the input isn't what the package was built against (a palette PNG, an inverted box, a
0..1 parameter given as 5, a 40x1200 panorama).

Asserted for every combination: nothing raises, an error names the problem (a silent no-op
is worse than a failure), and transparency survives (RGBA isn't silently flattened).

Run:  python evals/robustness.py
"""
import os
import sys
import traceback

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figsurgeon import ImageWorkspace                                   # noqa: E402
from figsurgeon.tools import TOOL_SCHEMAS                               # noqa: E402


# --------------------------------------------------------------------------- inputs
def _base():
    from skimage import data
    return Image.fromarray(data.chelsea())


def _cutout():
    """A real RGBA cutout -- the output of one tool becoming the input of the next."""
    img = _base().convert('RGBA')
    alpha = Image.new('L', img.size, 0)
    ImageDraw.Draw(alpha).ellipse([80, 40, 380, 280], fill=255)
    img.putalpha(alpha)
    return img


def _sixteen_bit():
    a = (np.array(_base().convert('L')).astype(np.uint16)) * 257
    return Image.fromarray(a.astype(np.int32))


INPUTS = {
    # Modes a real file arrives in: blur_background and rotate build PIL calls straight off
    # the input's mode (P/1 breaks blur_background, L/LA/1 breaks rotate).
    'RGB': _base,
    'RGBA_cutout': _cutout,
    'L_grayscale': lambda: _base().convert('L'),
    'P_palette': lambda: _base().convert('P'),
    'CMYK_print': lambda: _base().convert('CMYK'),
    'LA': lambda: _base().convert('LA'),
    '1_bilevel': lambda: _base().convert('1'),
    'I_16bit': _sixteen_bit,
    # Shapes. A 16x12 thumbnail and a 40x1200 strip break assumptions that a box has room
    # for margin, that a grid step is meaningful, and that a subject exists to segment.
    'tiny_16x12': lambda: _base().resize((16, 12)),
    'panorama_1200x40': lambda: _base().resize((1200, 40)),
    'column_40x1200': lambda: _base().resize((40, 1200)),
    'single_pixel': lambda: Image.new('RGB', (1, 1), (128, 64, 32)),
    'flat_colour': lambda: Image.new('RGB', (200, 150), (90, 90, 90)),
}


# --------------------------------------------------------------------------- calls
# Arguments a model actually gets wrong, alongside the ordinary ones.
CALLS = [
    ('adjust', {'brightness': 1.2}),
    ('adjust', {'contrast': 0.6, 'saturation': 1.4}),
    ('stylise', {'effect': 'grayscale'}),
    ('stylise', {'effect': 'vignette', 'strength': 0.5}),
    ('stylise', {'effect': 'sketch'}),
    ('stylise', {'effect': 'oil_painting'}),                 # not a real effect
    ('show_grid', {}),
    ('show_grid', {'step': 0}),                              # degenerate step
    ('preview_region', {'boxes': [[5, 5, 30, 30]]}),
    ('preview_region', {'boxes': [[30, 30, 5, 5]]}),         # inverted box
    ('preview_region', {'boxes': [[-50, -50, 999, 999]]}),   # out of bounds both ways
    ('zoom', {'box': [0, 0, 4, 4]}),
    ('crop', {'box': [2, 2, 10, 10]}),
    ('crop', {'box': [10, 10, 2, 2]}),                       # inverted
    ('crop', {'box': [0.5, 0.5, 9.5, 9.5]}),                 # floats, not ints
    ('resize', {'scale': 0.5}),
    ('resize', {'scale': 0}),                                # degenerate
    ('resize', {}),                                          # nothing to resize by
    ('rotate', {'angle': 13}),
    ('rotate', {'angle': 400}),                              # beyond one turn
    ('flip', {'axis': 'diagonal'}),                          # not an axis
    ('sample_colour', {'box': [2, 2, 10, 10]}),
    ('isolate_colour', {'target_rgb': [200, 150, 100]}),
    ('isolate_colour', {'target_rgb': [200, 150, 100], 'hue_tolerance': 5.0}),  # 0..1 param
    ('isolate_colour', {'target_rgb': [300, -20, 'red']}),   # nonsense channel values
    ('isolate_colour', {}),                                  # required arg missing
    ('replace_colour', {'from_rgb': [200, 150, 100], 'to_rgb': [40, 90, 200]}),
    ('preview_colour_mask', {'target_rgb': [200, 150, 100]}),
    ('blur_background', {'radius': 6}),
    ('remove_background', {}),
    ('recolour_object', {'box': [5, 4, 40, 30], 'to_rgb': [0, 0, 255]}),
    ('recolour_object', {'to_rgb': [0, 0, 255]}),            # no box at all
    ('isolate_object', {'boxes': [[5, 4, 40, 30], [50, 20, 90, 60]]}),
    ('erase_object', {'box': [5, 4, 40, 30]}),
    ('preview_object_mask', {'box': [5, 4, 40, 30]}),
    ('refine_box', {'box': [5, 4, 40, 30]}),
    ('add_text', {'text': 'hello', 'position': [2, 2], 'size': 14}),
    ('add_text', {'text': ''}),                              # empty string
    ('denoise', {'strength': 4}),
    ('auto_white_balance', {}),
    ('replace_background', {'colour_rgb': [255, 255, 255]}),
    ('extract_object', {'box': [5, 4, 40, 30]}),
    ('remove_object', {'box': [5, 4, 40, 30]}),
    ('remove_object', {'polyline': [[2, 2], [20, 20]]}),
    ('clean_transparent_box', {'box': [5, 4, 40, 30]}),
    ('remap_colormap', {'box': [5, 4, 40, 30], 'new_colours': ['#00407A', '#52BDEC']}),
    ('remap_colormap', {'box': [5, 4, 40, 30], 'source_cmap': 'not_a_colormap',
                        'new_colours': ['#00407A', '#52BDEC']}),
    ('remap_colormap', {'box': [5, 4, 40, 30], 'source_cmap': 'RdBu',
                        'new_colours': ['#00407A', '#52BDEC']}),   # diverging: must refuse
    # Depends on the tesseract binary, which pip cannot install. A missing system
    # dependency must come back as a note naming it, not end the session.
    ('replace_font', {}),
    ('undo', {}),
    ('reset', {}),
    ('nonexistent_tool', {'whatever': 1}),
]


def run(verbose=False):
    problems, total = [], 0

    for input_name, make in INPUTS.items():
        try:
            img = make()
        except Exception as e:
            problems.append((input_name, '(construction)', f'could not build: {e}'))
            continue

        for name, args in CALLS:
            total += 1
            ws = ImageWorkspace(img)
            try:
                result = ws.apply(name, args)
            except Exception as e:
                problems.append((input_name, name,
                                 f'RAISED {type(e).__name__}: {e}'))
                if verbose:
                    traceback.print_exc()
                continue

            note = result.note or ''
            if not note.strip():
                problems.append((input_name, name, 'returned an empty note'))
            if note.startswith('error') and len(note) < 25:
                problems.append((input_name, name, f'unhelpful error note: {note!r}'))

            # Transparency must survive an operation that does not understand it.
            if (img.mode == 'RGBA' and result.mutates
                    and result.image.size == img.size
                    and name not in ('remove_background', 'replace_background')):
                if result.image.mode != 'RGBA':
                    problems.append((input_name, name,
                                     f'dropped the alpha channel (mode {result.image.mode})'))

            # A looking tool must leave the image alone whatever the input mode.
            if not result.mutates and result.image is not ws.original:
                if np.array(result.image).shape != np.array(img).shape:
                    problems.append((input_name, name,
                                     'a non-mutating call changed the image'))

    print(f'{total - len(problems)}/{total} input x call combinations behaved')
    for input_name, call, why in problems:
        print(f'  PROBLEM  {input_name:18s} {call:22s} {why[:90]}')
    return problems


def check_schema_coverage():
    """Every declared tool must be exercised here, or the sweep has a blind spot."""
    declared = {t['name'] for t in TOOL_SCHEMAS}
    exercised = {name for name, _ in CALLS}
    missing = sorted(declared - exercised)
    if missing:
        print(f'\n  NOT EXERCISED: {", ".join(missing)}')
    return missing


if __name__ == '__main__':
    problems = run(verbose='-v' in sys.argv)
    missing = check_schema_coverage()
    sys.exit(1 if (problems or missing) else 0)
