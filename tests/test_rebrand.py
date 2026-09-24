"""Regression tests for rebrand.py, from a real chart with a viridis heatmap+colorbar and
DejaVu-font labels. Locks in five real bugs found during development, each with its own
check so a future change can't silently reintroduce any of them.
"""
import os

import numpy as np
from PIL import Image

from figsurgeon import rebrand as R

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
SRC = os.path.join(DATA, 'rebrand_source.png')


def run():
    checks = []
    im = Image.open(SRC).convert('RGB')

    # --- colormap remap ---
    # The chart is rendered at alpha~0.76 over white; matching the pure LUT without
    # accounting for that fails almost completely.
    alpha, err = R.detect_colormap_alpha(im, box=(776, 5, 788, 252), source_cmap='viridis')
    checks.append(('detected alpha is close to the known true value (0.76)',
                   abs(alpha - 0.76) < 0.05))
    # generous: the test box includes border pixels anti-aliased against the black frame
    checks.append(('alpha detection residual is small once corrected',
                   err < 25))

    out, frac, used_alpha = R.remap_colormap(im, box=(776, 5, 788, 252),
                                             new_colours=('#00407A', '#52BDEC'),
                                             match_tolerance=30)
    checks.append(('colorbar: most pixels matched and remapped (was 4.3%)', frac > 0.6))
    cb = np.array(out.crop((776, 5, 788, 252)).convert('RGB')).astype(int)
    checks.append(('colorbar top is light brand blue, not yellow',
                   cb[5:15].mean(axis=(0, 1))[2] > cb[5:15].mean(axis=(0, 1))[0]))
    # NOT pure navy: remap_colormap re-blends the new colour at the chart's own alpha (0.76)
    # to preserve visual style, not just hue.
    expected_bottom = 0.76 * np.array([0, 64, 122]) + 0.24 * np.array([255, 255, 255])
    actual_bottom = cb[-15:-5].mean(axis=(0, 1))
    checks.append(('colorbar bottom matches the alpha-reblended brand navy, not raw navy',
                   np.abs(actual_bottom - expected_bottom).max() < 15))

    # --- OCR / font replacement ---
    # Rotated titles read as high-confidence garbage under plain OCR; explicit
    # rotated_regions are required. Runs the full pipeline (remap then font), since the
    # y-tick detection fix depends on the remapped image.
    _s1, _, _ = R.remap_colormap(im, box=(103, 4, 735, 252),
                                 new_colours=('#00407A', '#52BDEC'),
                                 match_tolerance=30, alpha=0.76)
    _s2, _, _ = R.remap_colormap(_s1, box=(776, 5, 788, 252),
                                 new_colours=('#00407A', '#52BDEC'),
                                 match_tolerance=30, alpha=0.76)
    fout, log = R.replace_font(_s2, min_confidence=55,
                               rotated_regions=[((15, 0, 56, 260), 90),
                                                ((826, 15, 896, 240), 90)])
    texts = [t for t, b, s in log]
    checks.append(('rotated y-axis label read correctly as one coherent line',
                   any('Batch time' in t for t in texts)))
    checks.append(('rotated colorbar label read correctly as one coherent line',
                   any('Predicted score' in t for t in texts)))

    # An already-fixed rotated label must not be picked up again by the horizontal pass.
    checks.append(('no oversized duplicate-processing fragments (was size 53-62)',
                   all(s < 40 for t, b, s in log)))

    # Words on one line must be grouped, not redrawn independently with unnatural gaps.
    checks.append(('"Material use [kg]" reflowed as one connected line, not 3 words',
                   any(t == 'Material use [kg]' for t in texts)))
    # ...but words genuinely far apart must NOT be merged into one nonsense string.
    checks.append(('tick numbers stayed separate, not merged into "10 12 14 16 18 20"',
                   not any('10' in t and '20' in t for t in texts)))
    checks.append(('two-column legend stayed separate, not cross-merged',
                   not any('Feasible region' in t and 'Sample' in t for t in texts)))

    # A stray gridline pulled into tesseract's box must not make one tick label render
    # visibly larger than its neighbours.
    sizes = {t: s for t, b, s in log if t in ('10', '12', '14', '16', '18', '20', '22')}
    checks.append(('x-axis tick labels render at a consistent size (was 14 vs 24)',
                   len(sizes) >= 5 and max(sizes.values()) - min(sizes.values()) <= 2))

    # A nearby rotated title's fragments merging into the y-tick column (via line-grouping)
    # must not drop tick confidence below threshold and leave them in the original font.
    yticks = {t: s for t, b, s in log
              if t in ('7.00', '6.75', '6.50', '6.25', '6.00', '5.75', '5.50')}
    checks.append(('all 7 y-axis tick labels detected and restyled (was 2 of 7)',
                   len(yticks) == 7))
    checks.append(('y-axis tick labels render at a consistent size',
                   len(yticks) >= 6 and max(yticks.values()) - min(yticks.values()) <= 2))

    # Legend marker glyphs must not be OCR'd as characters and redrawn as text.
    legend = [t for t, b, s in log if b[1] > 305]
    checks.append(('legend marker glyphs not misread into label text (was "A Sample ...")',
                   not any(t.startswith('A ') or t.startswith('@ ') for t in legend)))

    # Legend entries must render at a consistent size across the same group.
    legend_sizes = [s for t, b, s in log if b[1] > 305]
    checks.append(('legend entries render at a consistent size (was 10 vs 14)',
                   len(legend_sizes) >= 4
                   and max(legend_sizes) - min(legend_sizes) <= 2))

    # --- four defects reported on the shipped output, each locked in here ---
    s1, _, _ = R.remap_colormap(im, box=(103, 4, 735, 252),
                                new_colours=('#00407A', '#52BDEC'),
                                match_tolerance=30, alpha=0.76)
    s2, _, _ = R.remap_colormap(s1, box=(776, 5, 788, 252),
                                new_colours=('#00407A', '#52BDEC'),
                                match_tolerance=30, alpha=0.76)
    _, flog = R.replace_font(s2, min_confidence=55,
                             rotated_regions=[((15, 0, 56, 260), 90),
                                              ((826, 15, 896, 240), 90)])
    yt = {t: sz for t, b, sz in flog if '.' in t and t.replace('.', '').isdigit()}
    xt = {t: sz for t, b, sz in flog if t in ('10', '12', '14', '16', '18', '20', '22')}
    leg = {t: sz for t, b, sz in flog
           if 'Sample' in t or 'Frontier' in t or 'Feasible' in t}

    checks.append(('all 7 y-tick labels redrawn (was 2 of 7)', len(yt) == 7))
    checks.append(('y-tick sizes uniform', len(set(yt.values())) == 1))
    checks.append(('all 7 x-tick labels redrawn (10 and 22 were missing)', len(xt) == 7))
    checks.append(('x-tick sizes uniform', len(set(xt.values())) == 1))
    # 1px tolerance: legend labels vary slightly in height (ascenders, descenders, digits)
    # at one true font size; exact equality would over-fit the heuristic to this chart.
    checks.append(('legend entry sizes uniform within 1px',
                   max(leg.values()) - min(leg.values()) <= 1))
    checks.append(('no legend entry begins with a stray marker-as-letter',
                   not any(t.startswith('A ') or t.startswith('@ ') for t in leg)))

    ok = True
    for label, passed in checks:
        print(f'  {"PASS" if passed else "FAIL"}  {label}')
        ok &= passed
    return ok





def run_diverging():
    """Diverging colormaps must be refused by default: a diverging map encodes sign (red
    positive, blue negative), and flattening it onto a sequential gradient would destroy
    that meaning while looking like a clean, successful remap."""
    from PIL import Image
    checks = []
    im = Image.open(os.path.join(DATA, 'diverging_source.png')).convert('RGB')

    try:
        R.remap_colormap(im, (148, 35, 470, 370), source_cmap='RdBu_r',
                         new_colours=('#00407A', '#52BDEC'))
        checks.append(('diverging colormap refused by default', False))
    except ValueError as e:
        checks.append(('diverging colormap refused by default', 'DIVERGING' in str(e)))

    out, frac, _ = R.remap_colormap(im, (148, 35, 470, 370), source_cmap='RdBu_r',
                                    new_colours=('#00407A', '#F7F7F7', '#B33A3A'),
                                    allow_diverging=True)
    checks.append(('3-stop diverging replacement allowed and works', frac > 0.9))

    # a sequential map must be unaffected by the guard
    seq = Image.open(SRC).convert('RGB')
    _, f2, _ = R.remap_colormap(seq, (776, 5, 788, 252), new_colours=('#00407A', '#52BDEC'),
                                match_tolerance=30, alpha=0.76)
    checks.append(('sequential colormap still remaps (no false refusal)', f2 > 0.6))

    ok = True
    for label, passed in checks:
        print(f'  {"PASS" if passed else "FAIL"}  {label}')
        ok &= bool(passed)
    return ok


if __name__ == '__main__':
    import sys
    a = run()
    b = run_diverging()
    sys.exit(0 if (a and b) else 1)


def _needs_tesseract():
    """Skip rather than fail when the tesseract binary is absent (pip cannot install it)."""
    import pytest
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception as e:
        pytest.skip(f'tesseract binary not available ({type(e).__name__})')


def test_rebrand_checks():
    _needs_tesseract()
    assert run(), 'see printed output for which rebrand check failed'


def test_diverging_colormap_checks():
    assert run_diverging(), 'see printed output for which diverging-cmap check failed'
